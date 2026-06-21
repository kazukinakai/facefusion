import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy
from tqdm import tqdm

from facefusion import ffmpeg
from facefusion import logger, process_manager, state_manager, translator, video_manager
from facefusion.audio import create_empty_audio_frame, get_audio_frame, get_voice_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_video
from facefusion.filesystem import filter_audio_paths, is_file, is_video, remove_file
from facefusion.processors.core import get_processors_modules
from facefusion.temp_helper import clear_temp_directory, create_processed_directory, create_temp_directory, force_clear_temp_directory, get_processed_frame_path, move_temp_file, resolve_processed_frame_set, resolve_temp_frame_set
from facefusion.time_helper import calculate_end_time
from facefusion.types import ErrorCode
from facefusion.vision import conditional_merge_vision_mask, detect_video_resolution, extract_vision_mask, pack_resolution, predict_video_frame_total, read_static_image, read_static_images, read_static_video_frame, restrict_trim_frame, restrict_video_fps, restrict_video_resolution, scale_resolution, select_video_frames, write_image
from facefusion.workflows.core import is_process_stopping


def process(start_time : float) -> ErrorCode:
	tasks =\
	[
		setup,
		extract_frames,
		process_video,
		merge_frames,
		restore_audio,
		partial(finalize_video, start_time)
	]
	process_manager.start()

	for task in tasks:
		error_code = task() #type:ignore[operator]

		if error_code > 0:
			process_manager.end()
			return error_code

	process_manager.end()
	return 0


def setup() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

	if analyse_video(state_manager.get_item('target_path'), trim_frame_start, trim_frame_end):
		return 3

	if clear_temp_directory(state_manager.get_item('target_path')):
		logger.debug(translator.get('clearing_temp'), __name__)

	if create_temp_directory(state_manager.get_item('target_path')):
		logger.debug(translator.get('creating_temp'), __name__)

	# Resume mode (keep_temp): processed frames accumulate in a `processed/`
	# subdir that survives across runs, so an interrupted job continues instead
	# of restarting. clear_temp_directory above is a no-op under keep_temp.
	if state_manager.get_item('keep_temp'):
		create_processed_directory(state_manager.get_item('target_path'))

	return 0


def extract_frames() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	output_video_resolution = scale_resolution(detect_video_resolution(state_manager.get_item('target_path')), state_manager.get_item('output_video_scale'))
	temp_video_resolution = restrict_video_resolution(state_manager.get_item('target_path'), output_video_resolution)
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))

	# Resume: a previous run already extracted every frame (each frame is now
	# either still raw or already processed). Re-extracting would re-decode the
	# whole video for nothing, so skip it.
	if state_manager.get_item('keep_temp'):
		frame_total = predict_video_frame_total(state_manager.get_item('target_path'), temp_video_fps, trim_frame_start, trim_frame_end)
		extracted_total = len(resolve_temp_frame_set(state_manager.get_item('target_path'))) + len(resolve_processed_frame_set(state_manager.get_item('target_path')))
		if frame_total and extracted_total >= frame_total:
			logger.info(translator.get('extracting_frames_succeeded'), __name__)
			return 0

	logger.info(translator.get('extracting_frames').format(resolution=pack_resolution(temp_video_resolution), fps=temp_video_fps), __name__)

	if ffmpeg.extract_frames(state_manager.get_item('target_path'), temp_video_resolution, temp_video_fps, trim_frame_start, trim_frame_end):
		logger.debug(translator.get('extracting_frames_succeeded'), __name__)
	else:
		if is_process_stopping():
			return 4
		logger.error(translator.get('extracting_frames_failed'), __name__)
		return 1
	return 0


def process_video() -> ErrorCode:
	temp_frame_set = resolve_temp_frame_set(state_manager.get_item('target_path'))

	# Resume: raw frames are removed as they complete, so an empty raw set with a
	# non-empty processed store means every frame is already done — fall through
	# to merge instead of erroring on "temp_frames_not_found".
	if not temp_frame_set and state_manager.get_item('keep_temp') and resolve_processed_frame_set(state_manager.get_item('target_path')):
		logger.info('all frames already processed, resuming at merge', __name__)
		return 0

	if temp_frame_set:
		if state_manager.get_item('keep_temp'):
			processed_total = len(resolve_processed_frame_set(state_manager.get_item('target_path')))
			if processed_total:
				logger.info('resume: ' + str(processed_total) + ' frame(s) already processed, ' + str(len(temp_frame_set)) + ' remaining', __name__)

		with tqdm(total = len(temp_frame_set), desc = translator.get('processing'), unit = 'frame', ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
			progress.set_postfix(execution_providers = state_manager.get_item('execution_providers'))

			with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
				futures = []

				for frame_number, temp_frame_path in temp_frame_set.items():
					future = executor.submit(process_temp_frame, temp_frame_path, frame_number)
					futures.append(future)

				for future in as_completed(futures):
					if is_process_stopping():
						for __future__ in futures:
							__future__.cancel()

					if not future.cancelled():
						future.result()
						progress.update()

		for processor_module in get_processors_modules(state_manager.get_item('processors')):
			processor_module.post_process()

		if is_process_stopping():
			return 4
	else:
		logger.error(translator.get('temp_frames_not_found'), __name__)
		return 1
	return 0


def merge_frames() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	output_video_resolution = scale_resolution(detect_video_resolution(state_manager.get_item('target_path')), state_manager.get_item('output_video_scale'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))

	logger.info(translator.get('merging_video').format(resolution = pack_resolution(output_video_resolution), fps = state_manager.get_item('output_video_fps')), __name__)
	if ffmpeg.merge_video(state_manager.get_item('target_path'), temp_video_fps, output_video_resolution, state_manager.get_item('output_video_fps'), trim_frame_start, trim_frame_end):
		logger.debug(translator.get('merging_video_succeeded'), __name__)
	else:
		if is_process_stopping():
			return 4
		logger.error(translator.get('merging_video_failed'), __name__)
		return 1
	return 0


def restore_audio() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

	if state_manager.get_item('output_audio_volume') == 0:
		logger.info(translator.get('skipping_audio'), __name__)
		move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	else:
		source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
		if source_audio_path:
			if ffmpeg.replace_audio(state_manager.get_item('target_path'), source_audio_path, state_manager.get_item('output_path')):
				video_manager.clear_video_pool()
				logger.debug(translator.get('replacing_audio_succeeded'), __name__)
			else:
				video_manager.clear_video_pool()
				if is_process_stopping():
					return 4
				logger.warn(translator.get('replacing_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
		else:
			if ffmpeg.restore_audio(state_manager.get_item('target_path'), state_manager.get_item('output_path'), trim_frame_start, trim_frame_end):
				video_manager.clear_video_pool()
				logger.debug(translator.get('restoring_audio_succeeded'), __name__)
			else:
				video_manager.clear_video_pool()
				if is_process_stopping():
					return 4
				logger.warn(translator.get('restoring_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	return 0


def process_temp_frame(temp_frame_path : str, frame_number : int) -> bool:
	# Resume: if this frame was already committed to the processed store in a
	# prior run, skip it (and drop the lingering raw frame from a crash between
	# the processed write and the raw delete). The processed file's existence is
	# the durable per-frame "done" marker — robust to the parallel, out-of-order
	# nature of this loop.
	if state_manager.get_item('keep_temp') and is_file(get_processed_frame_path(state_manager.get_item('target_path'), frame_number)):
		remove_file(temp_frame_path)
		return True

	reference_vision_frame = read_static_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	target_vision_frames = select_video_frames(state_manager.get_item('target_path'), frame_number, state_manager.get_item('target_frame_amount'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	temp_vision_frame = read_static_image(temp_frame_path, 'rgba')
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	source_audio_frame = get_audio_frame(source_audio_path, temp_video_fps, frame_number)
	source_voice_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number)

	if not numpy.any(source_audio_frame):
		source_audio_frame = create_empty_audio_frame()
	if not numpy.any(source_voice_frame):
		source_voice_frame = create_empty_audio_frame()

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		temp_vision_frame, temp_vision_mask = processor_module.process_frame(
		{
			'reference_vision_frame': reference_vision_frame,
			'source_vision_frames': source_vision_frames,
			'source_audio_frame': source_audio_frame,
			'source_voice_frame': source_voice_frame,
			'target_vision_frames': target_vision_frames,
			'temp_vision_frame': temp_vision_frame[:, :, :3],
			'temp_vision_mask': temp_vision_mask
		})

	temp_vision_frame = conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)

	# Resume mode: write to the processed store atomically (tmp + replace) so a
	# crash never leaves a half-written frame that looks "done", then drop the
	# raw frame to keep peak disk at ~1x. Merge reads from the processed store.
	if state_manager.get_item('keep_temp'):
		processed_frame_path = get_processed_frame_path(state_manager.get_item('target_path'), frame_number)
		processed_frame_temp_path = processed_frame_path + '.tmp'
		if not write_image(processed_frame_temp_path, temp_vision_frame):
			return False
		os.replace(processed_frame_temp_path, processed_frame_path)
		remove_file(temp_frame_path)
		return True

	return write_image(temp_frame_path, temp_vision_frame)


def finalize_video(start_time : float) -> ErrorCode:
	if not is_video(state_manager.get_item('output_path')):
		# Output is missing/invalid — keep the temp store (under keep_temp) so the
		# job can resume instead of restarting from scratch.
		logger.error(translator.get('processing_video_failed'), __name__)
		return 1

	logger.debug(translator.get('clearing_temp'), __name__)
	# Success: drop the temp store. Force-clear so the keep_temp resume store
	# (raw + processed frames) is removed too, not just left behind.
	if state_manager.get_item('keep_temp'):
		force_clear_temp_directory(state_manager.get_item('target_path'))
	else:
		clear_temp_directory(state_manager.get_item('target_path'))

	logger.info(translator.get('processing_video_succeeded').format(seconds = calculate_end_time(start_time)), __name__)
	return 0
