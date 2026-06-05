import hashlib
import os
from typing import Optional

from facefusion import state_manager
from facefusion.filesystem import get_file_extension, get_file_name, is_image, is_video


def convert_int_none(value : int) -> Optional[int]:
	if value == 'none':
		return None
	return value


def convert_str_none(value : str) -> Optional[str]:
	if value == 'none':
		return None
	return value


def suggest_output_path(output_directory_path : str, target_path : str) -> Optional[str]:
	if is_image(target_path) or is_video(target_path):
		target_file_extension = get_file_extension(target_path)
		output_file_naming = state_manager.get_item('output_file_naming')
		if output_file_naming == 'hash':
			output_file_name = hashlib.sha1(str(state_manager.get_state()).encode()).hexdigest()[:8]
			return os.path.join(output_directory_path, output_file_name + target_file_extension)
		if output_file_naming == 'target':
			return os.path.join(output_directory_path, get_file_name(target_path) + target_file_extension)
		# 'target_index' (default): keep the target name but never overwrite an existing output
		output_file_name = get_file_name(target_path)
		output_path = os.path.join(output_directory_path, output_file_name + target_file_extension)
		output_index = 1
		while os.path.exists(output_path):
			output_path = os.path.join(output_directory_path, output_file_name + '_' + str(output_index) + target_file_extension)
			output_index += 1
		return output_path
	return None
