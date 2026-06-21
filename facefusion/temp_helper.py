import os

from facefusion import state_manager
from facefusion.filesystem import create_directory, get_file_extension, get_file_name, move_file, remove_directory, resolve_file_pattern
from facefusion.types import FrameSet


def get_temp_file_path(file_path : str) -> str:
	temp_directory_path = get_temp_directory_path(file_path)
	temp_file_extension = get_file_extension(file_path)
	return os.path.join(temp_directory_path, 'temp' + temp_file_extension)


def move_temp_file(file_path : str, move_path : str) -> bool:
	temp_file_path = get_temp_file_path(file_path)
	return move_file(temp_file_path, move_path)


def resolve_temp_frame_set(target_path : str) -> FrameSet:
	temp_frame_pattern = get_temp_frame_pattern(target_path, '*')
	temp_frame_set = {}

	for temp_frame_path in resolve_file_pattern(temp_frame_pattern):
		frame_number = int(get_file_name(temp_frame_path))
		temp_frame_set[frame_number] = temp_frame_path

	return temp_frame_set


def get_temp_frame_pattern(target_path : str, temp_frame_prefix : str) -> str:
	temp_directory_path = get_temp_directory_path(target_path)
	return os.path.join(temp_directory_path, temp_frame_prefix + '.' + state_manager.get_item('temp_frame_format'))


# Processed-frame store (resume): processed frames live in a `processed/` subdir
# so their existence is an unambiguous per-frame "done" marker, separate from the
# raw extracted frames. Only used in keep-temp (resume) mode; merge reads from
# here and process writes here, leaving raw frames untouched until committed.
def get_processed_directory_path(target_path : str) -> str:
	return os.path.join(get_temp_directory_path(target_path), 'processed')


def get_processed_frame_pattern(target_path : str, temp_frame_prefix : str) -> str:
	return os.path.join(get_processed_directory_path(target_path), temp_frame_prefix + '.' + state_manager.get_item('temp_frame_format'))


def get_processed_frame_path(target_path : str, frame_number : int) -> str:
	return get_processed_frame_pattern(target_path, format(frame_number, '08d'))


def resolve_processed_frame_set(target_path : str) -> FrameSet:
	processed_frame_set = {}

	for processed_frame_path in resolve_file_pattern(get_processed_frame_pattern(target_path, '*')):
		frame_number = int(get_file_name(processed_frame_path))
		processed_frame_set[frame_number] = processed_frame_path

	return processed_frame_set


def create_processed_directory(target_path : str) -> bool:
	return create_directory(get_processed_directory_path(target_path))


def get_temp_directory_path(file_path : str) -> str:
	temp_file_name = get_file_name(file_path)
	return os.path.join(state_manager.get_item('temp_path'), 'facefusion', temp_file_name)


def create_temp_directory(file_path : str) -> bool:
	temp_directory_path = get_temp_directory_path(file_path)
	return create_directory(temp_directory_path)


def clear_temp_directory(file_path : str) -> bool:
	if not state_manager.get_item('keep_temp'):
		temp_directory_path = get_temp_directory_path(file_path)
		return remove_directory(temp_directory_path)
	return True


def force_clear_temp_directory(file_path : str) -> bool:
	# Remove the temp directory regardless of keep_temp — used on a successful
	# video run so the resume store (raw + processed frames) is cleaned up even
	# though keep_temp kept it alive across retries.
	return remove_directory(get_temp_directory_path(file_path))
