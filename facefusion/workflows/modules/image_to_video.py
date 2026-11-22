import os
from functools import partial
from typing import Callable, List, Optional

from facefusion import logger, state_manager
from facefusion.filesystem import is_directory, remove_directory
from facefusion.temp_helper import get_temp_directory_path, resolve_temp_frame_set
from facefusion.types import ErrorCode
from facefusion.workflows.core import clear, process_frames, restore_completed_frames, save_completed_frame_sequence, setup
from facefusion.workflows.to_video import analyse_video, create_temp_frames, finalize_video, merge_frames, process_memory_frames, restore_audio


def prepare_resumable_frames(frame_directory : str) -> ErrorCode:
	clear()
	if restore_completed_frames(frame_directory):
		return 0

	raw_directory = os.path.join(frame_directory, 'raw')
	if not restore_completed_frames(raw_directory):
		clear()
		setup()
		error_code = create_temp_frames()
		if error_code:
			return error_code

		temp_path = state_manager.get_temp_path()
		output_path = state_manager.get_item('output_path')
		temp_directory = get_temp_directory_path(temp_path, output_path)
		frame_set = resolve_temp_frame_set(temp_path, output_path, state_manager.get_item('temp_frame_format'))
		if not frame_set or not save_completed_frame_sequence(temp_directory, frame_set):
			return 1
		if is_directory(raw_directory) and not remove_directory(raw_directory):
			return 1
		os.replace(temp_directory, raw_directory)
		if not restore_completed_frames(raw_directory, verify_images = False):
			return 1

	return process_frames(frame_directory)


def process(start_time : float, frame_directory : Optional[str] = None) -> ErrorCode:
	if not frame_directory:
		return process_video(start_time)

	temp_path = state_manager.get_item('temp_path')
	state_manager.set_item('temp_path', os.path.join(frame_directory, 'work'))

	try:
		return process_video(start_time, frame_directory)
	except Exception as exception:
		logger.error(str(exception), __name__)
		return 1
	finally:
		try:
			clear()
		finally:
			state_manager.set_item('temp_path', temp_path)


def process_video(start_time : float, frame_directory : Optional[str] = None) -> ErrorCode:
	tasks : List[Callable[[], ErrorCode]] = [ analyse_video ]

	if frame_directory:
		tasks.extend([ partial(prepare_resumable_frames, frame_directory), merge_frames ])
	else:
		tasks.extend([ clear, setup ])
		workflow_strategy = state_manager.get_item('workflow_strategy')

		if workflow_strategy == 'disk':
			tasks.extend([ create_temp_frames, process_frames, merge_frames ])
		elif workflow_strategy == 'memory':
			tasks.append(process_memory_frames)

	tasks.extend([ restore_audio, partial(finalize_video, start_time) ])
	if not frame_directory:
		tasks.append(clear)

	for task in tasks:
		error_code = task()

		if error_code > 0:
			return error_code

	return 0
