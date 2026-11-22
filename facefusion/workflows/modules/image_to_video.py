import os
from functools import partial
from typing import Callable, List, Optional

from facefusion import logger, state_manager
from facefusion.types import ErrorCode
from facefusion.workflows.core import clear, process_frames, restore_completed_frames, setup
from facefusion.workflows.to_video import analyse_video, create_temp_frames, finalize_video, merge_frames, process_memory_frames, restore_audio


def prepare_resumable_frames(frame_directory : str) -> ErrorCode:
	clear()
	if restore_completed_frames(frame_directory):
		return 0

	clear()
	setup()
	error_code = create_temp_frames()
	if error_code:
		return error_code
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
