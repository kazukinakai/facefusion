import os
import secrets
from datetime import datetime
from typing import Dict, List, Optional

import facefusion.choices
from facefusion import state_manager
from facefusion.filesystem import get_file_extension, get_file_format, get_file_name
from facefusion.sanitizer import sanitize_job_id
from facefusion.types import Args, JobId, StepIndex


def get_step_input_paths(step_args : Args) -> List[str]:
	input_paths = list(step_args.get('source_paths') or [])
	target_path = step_args.get('target_path')

	if target_path:
		input_paths.append(target_path)
	return [ os.path.normcase(os.path.abspath(input_path)) for input_path in input_paths ]


def resolve_step_input_paths(step_args : Args, relocated_step_outputs : Dict[str, str]) -> Args:
	def resolve_input_path(input_path : str) -> str:
		return relocated_step_outputs.get(os.path.normcase(os.path.realpath(input_path)), input_path)

	resolved_args = step_args.copy()
	if step_args.get('target_path'):
		resolved_args['target_path'] = resolve_input_path(step_args['target_path'])
	if step_args.get('source_paths'):
		resolved_args['source_paths'] = [ resolve_input_path(source_path) for source_path in step_args['source_paths'] ]
	return resolved_args


def get_step_video_directory(job_id : JobId) -> str:
	return os.path.join(state_manager.get_jobs_path(), 'step-videos', sanitize_job_id(job_id))


def get_step_output_path(job_id : JobId, step_index : StepIndex, output_path : str, resume : bool = False) -> Optional[str]:
	if output_path:
		output_directory_path, output_file_path = os.path.split(output_path)
		output_file_name = get_file_name(output_file_path)
		output_file_extension = get_file_extension(output_file_path)

		if resume is True and get_file_format(output_path) in facefusion.choices.video_formats:
			output_directory_path = get_step_video_directory(job_id)
			job_id = sanitize_job_id(job_id)
		if output_file_name and output_file_extension:
			return os.path.join(output_directory_path, output_file_name + '-' + job_id + '-' + str(step_index) + output_file_extension)

		if output_file_path and output_directory_path:
			return os.path.join(output_directory_path, output_file_path + '-' + job_id + '-' + str(step_index))

	return None


def suggest_job_id(job_prefix : str = 'job') -> JobId:
	return job_prefix + '-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-' + secrets.token_hex(4)
