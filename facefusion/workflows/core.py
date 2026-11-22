import os
import time
import uuid
from collections import deque
from concurrent.futures import Future
from typing import Deque, List, Optional

import cv2
import numpy

from facefusion import audio, content_analyser, logger, process_manager, state_manager, thread_helper, translator
from facefusion.audio import create_empty_audio_frame, get_audio_frame, get_voice_frame
from facefusion.clis import cli_progress
from facefusion.common_helper import get_first
from facefusion.face_selector import decode_face_selector_maps
from facefusion.ffprobe import extract_static_video_metadata
from facefusion.filesystem import copy_file, create_directory, filter_audio_paths, get_file_extension, get_file_size, has_audio, has_image, has_video, remove_file, resolve_file_pattern
from facefusion.json import read_json, write_json
from facefusion.processors.core import get_processors_modules
from facefusion.temp_helper import clear_temp_directory, create_temp_directory, get_temp_frames_pattern, resolve_temp_frame_set
from facefusion.types import AudioFrame, ErrorCode, FrameSet, VisionFrame, WorkflowMode
from facefusion.vision import conditional_merge_vision_mask, detect_video_fps, extract_vision_mask, read_image, read_static_image, read_static_images, read_static_video_frame, resolve_extract_frame_index, resolve_target_frame_index, restrict_video_fps, restrict_video_range, select_video_frames, write_image


FRAME_SEQUENCE_FILE = '.sequence.json'


def clear_input_caches(input_paths : List[str]) -> None:
	if has_video(input_paths):
		extract_static_video_metadata.cache_clear()
		content_analyser.analyse_video.cache_clear()
		read_static_video_frame.cache_clear()
	if has_image(input_paths):
		content_analyser.analyse_image.cache_clear()
		read_static_image.cache_clear()
	if has_audio(input_paths):
		audio.read_static_audio.cache_clear()
		audio.read_static_voice.cache_clear()


def detect_workflow_mode() -> WorkflowMode:
	target_path = state_manager.get_item('target_path')
	output_path = state_manager.get_item('output_path')

	if has_video([ target_path ]):
		if get_file_extension(output_path):
			return 'image-to-video'
		return 'image-to-video:frames'

	if has_audio(state_manager.get_item('source_paths')) and has_image([ target_path ]):
		if get_file_extension(output_path):
			return 'audio-to-image:video'
		return 'audio-to-image:frames'

	return 'image-to-image'


def setup() -> ErrorCode:
	if create_temp_directory(state_manager.get_temp_path(), state_manager.get_item('output_path')):
		logger.debug(translator.get('creating_temp'), __name__)
	return 0


def clear() -> ErrorCode:
	if clear_temp_directory(state_manager.get_temp_path(), state_manager.get_item('output_path')):
		logger.debug(translator.get('clearing_temp'), __name__)
	return 0


def conditional_get_source_audio_frame(frame_index : int) -> AudioFrame:
	if state_manager.get_item('workflow_mode') in [ 'audio-to-image:frames', 'audio-to-image:video', 'image-to-video' ]:
		source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
		fps = state_manager.get_item('output_audio_fps')

		if state_manager.get_item('workflow_mode') == 'image-to-video':
			trim_frame_start, _ = restrict_video_range(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
			fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
			frame_index = frame_index - trim_frame_start

		source_audio_frame = get_audio_frame(source_audio_path, fps, frame_index)

		if numpy.any(source_audio_frame):
			return source_audio_frame

	return create_empty_audio_frame()


def conditional_get_source_voice_frame(frame_index : int) -> AudioFrame:
	if state_manager.get_item('workflow_mode') in [ 'audio-to-image:frames', 'audio-to-image:video', 'image-to-video' ]:
		source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
		temp_fps = state_manager.get_item('output_audio_fps')

		if state_manager.get_item('workflow_mode') == 'image-to-video':
			trim_frame_start, _ = restrict_video_range(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
			temp_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
			frame_index = frame_index - trim_frame_start

		source_voice_frame = get_voice_frame(source_audio_path, temp_fps, frame_index)

		if numpy.any(source_voice_frame):
			return source_voice_frame

	return create_empty_audio_frame()


def conditional_get_reference_vision_frames() -> List[VisionFrame]:
	face_selector_maps = decode_face_selector_maps(state_manager.get_item('face_selector_maps'))
	reference_vision_frames = []

	for face_selector_map in face_selector_maps:
		for reference_face in face_selector_map.get('reference_faces'):
			reference_vision_frames.append(conditional_get_reference_vision_frame(reference_face.get('frame_index')))

	return reference_vision_frames


def conditional_get_reference_vision_frame(reference_frame_index : int) -> VisionFrame:
	if state_manager.get_item('workflow_mode') in [ 'image-to-video', 'image-to-video:frames' ]:
		return read_static_video_frame(state_manager.get_item('target_path'), reference_frame_index)
	return read_static_image(state_manager.get_item('target_path'))


def conditional_get_target_vision_frames(frame_index : int) -> List[VisionFrame]:
	if state_manager.get_item('workflow_mode') in [ 'image-to-video', 'image-to-video:frames' ]:
		trim_frame_start, _ = restrict_video_range(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
		temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
		video_fps = detect_video_fps(state_manager.get_item('target_path'))
		temp_frame_index = resolve_extract_frame_index(video_fps, temp_video_fps, trim_frame_start) + frame_index - trim_frame_start
		video_frame_index = resolve_target_frame_index(video_fps, temp_video_fps, temp_frame_index)

		return select_video_frames(state_manager.get_item('target_path'), video_frame_index, state_manager.get_item('target_frame_amount'))
	return [ read_static_image(state_manager.get_item('target_path')) ]


def process_temp_frame(temp_frame_path : str, frame_index : int, output_frame_path : Optional[str] = None) -> bool:
	reference_vision_frames = conditional_get_reference_vision_frames()
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_frame = conditional_get_source_audio_frame(frame_index)
	source_voice_frame = conditional_get_source_voice_frame(frame_index)
	target_vision_frames = conditional_get_target_vision_frames(frame_index)
	temp_vision_frame = read_image(temp_frame_path, 'rgba') if output_frame_path else read_static_image(temp_frame_path, 'rgba')
	if temp_vision_frame is None:
		return False
	temp_vision_frame = temp_vision_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		temp_vision_frame, temp_vision_mask = processor_module.process_frame(
		{
			'reference_vision_frames': reference_vision_frames,
			'source_vision_frames': source_vision_frames,
			'source_audio_frame': source_audio_frame,
			'source_voice_frame': source_voice_frame,
			'target_vision_frames': target_vision_frames,
			'temp_vision_frame': temp_vision_frame[:, :, :3],
			'temp_vision_mask': temp_vision_mask
		})

	temp_vision_frame = conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)
	return write_image(output_frame_path or temp_frame_path, temp_vision_frame)


def process_temp_vision_frame(target_vision_frames : List[VisionFrame], temp_vision_frame : VisionFrame, frame_index : int) -> VisionFrame:
	reference_vision_frames = conditional_get_reference_vision_frames()
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_frame = conditional_get_source_audio_frame(frame_index)
	source_voice_frame = conditional_get_source_voice_frame(frame_index)
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		temp_vision_frame, temp_vision_mask = processor_module.process_frame(
		{
			'reference_vision_frames': reference_vision_frames,
			'source_vision_frames': source_vision_frames,
			'source_audio_frame': source_audio_frame,
			'source_voice_frame': source_voice_frame,
			'target_vision_frames': target_vision_frames,
			'temp_vision_frame': temp_vision_frame[:, :, :3],
			'temp_vision_mask': temp_vision_mask
		})

	return conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)


def is_readable_frame(frame_path : str) -> bool:
	try:
		return get_file_size(frame_path) > 0 and read_image(frame_path, 'rgba') is not None
	except (OSError, cv2.error):
		return False


def save_completed_frame_sequence(frame_directory : str, temp_frame_set : FrameSet) -> bool:
	frame_start = min(temp_frame_set)
	frame_total = len(temp_frame_set)

	if max(temp_frame_set) - frame_start + 1 != frame_total:
		return False

	sequence_path = os.path.join(frame_directory, FRAME_SEQUENCE_FILE)
	partial_sequence_path = sequence_path + '.partial'
	if write_json(partial_sequence_path, { 'frame_start': frame_start, 'frame_total': frame_total }):
		os.replace(partial_sequence_path, sequence_path)
		return True
	return False


def restore_temp_frame(retained_frame_path : str, temp_frame_path : str) -> bool:
	staged_frame_path = os.path.join(os.path.dirname(temp_frame_path), '.' + uuid.uuid4().hex)
	try:
		try:
			os.link(retained_frame_path, staged_frame_path)
		except OSError:
			if not copy_file(retained_frame_path, staged_frame_path):
				return False
		os.replace(staged_frame_path, temp_frame_path)
		return True
	finally:
		remove_file(staged_frame_path)


def restore_completed_frames(frame_directory : str, verify_images : bool = True) -> bool:
	try:
		sequence = read_json(os.path.join(frame_directory, FRAME_SEQUENCE_FILE))
		if not isinstance(sequence, dict):
			return False

		frame_start = sequence.get('frame_start')
		frame_total = sequence.get('frame_total')
		if type(frame_start) is not int or type(frame_total) is not int or frame_start < 0 or frame_total < 1:
			return False

		temp_frame_format = state_manager.get_item('temp_frame_format')
		temp_path = state_manager.get_temp_path()
		output_path = state_manager.get_item('output_path')
		retained_frames = resolve_file_pattern(os.path.join(frame_directory, '*.' + temp_frame_format))
		if len(retained_frames) != frame_total:
			return False

		temp_frames = [ get_temp_frames_pattern(temp_path, output_path, temp_frame_format, format(frame_index, '08d')) for frame_index in range(frame_start, frame_start + frame_total) ]
		if [ os.path.basename(frame_path) for frame_path in retained_frames ] != [ os.path.basename(frame_path) for frame_path in temp_frames ]:
			return False
		if verify_images and not all(map(is_readable_frame, retained_frames)):
			return False
		setup()
		return all(restore_temp_frame(retained_frame, temp_frame) for retained_frame, temp_frame in zip(retained_frames, temp_frames))
	except (OSError, ValueError):
		return False


def process_resumable_frame(temp_frame_path : str, frame_index : int, frame_directory : str) -> bool:
	frame_name, frame_extension = os.path.splitext(os.path.basename(temp_frame_path))
	retained_frame_path = os.path.join(frame_directory, frame_name + frame_extension)
	partial_frame_path = os.path.join(frame_directory, '.' + frame_name + '.partial' + frame_extension)

	try:
		if not is_readable_frame(retained_frame_path):
			if not process_temp_frame(temp_frame_path, frame_index, partial_frame_path):
				remove_file(partial_frame_path)
				return False
			os.replace(partial_frame_path, retained_frame_path)
		return restore_temp_frame(retained_frame_path, temp_frame_path)
	except Exception as exception:
		logger.error(str(exception), __name__)
		return False


def process_frames(frame_directory : Optional[str] = None) -> ErrorCode:
	if frame_directory:
		try:
			if not create_directory(frame_directory):
				return 1
		except OSError:
			return 1
	failed_write = False
	temp_frame_set = resolve_temp_frame_set(state_manager.get_temp_path(), state_manager.get_item('output_path'), state_manager.get_item('temp_frame_format'))

	if temp_frame_set:

		time_current = time.monotonic()
		with process_manager.stage('infer', translator.get('processing'),
		{
			'unit': 'frame',
			'current': 0,
			'started_at': time_current,
			'updated_at': time_current
		}):
			process_manager.count(temp_frame_set)

			with cli_progress.create() as progress:
				with thread_helper.create_executor(state_manager.get_item('execution_thread_count')) as executor:
					futures : Deque[Future[bool]] = deque()

					for frame_index, temp_frame_path in temp_frame_set.items():
						if frame_directory:
							future = executor.submit(process_resumable_frame, temp_frame_path, frame_index, frame_directory)
						else:
							future = executor.submit(process_temp_frame, temp_frame_path, frame_index)
						futures.append(future)

					while futures and not process_manager.is_stopping():
						future = futures.popleft()
						failed_write = not future.result() and bool(frame_directory)
						if failed_write:
							break
						process_manager.update()
						progress.update()

					for pending_future in futures:
						pending_future.cancel()

		for processor_module in get_processors_modules(state_manager.get_item('processors')):
			processor_module.post_process()

		if process_manager.is_stopping():
			return 4
		if failed_write or frame_directory and not save_completed_frame_sequence(frame_directory, temp_frame_set):
			logger.error(translator.get('processing_frames_failed'), __name__)
			return 1
	else:
		logger.error(translator.get('temp_frames_not_found'), __name__)
		return 1
	return 0
