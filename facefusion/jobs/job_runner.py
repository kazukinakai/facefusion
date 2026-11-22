import os
from typing import Dict, Optional

from facefusion import process_manager, session_manager, state_manager, store_manager
from facefusion.ffmpeg import concat_video
from facefusion.ffprobe import probe_video_entries
from facefusion.filesystem import are_images, are_videos, copy_file, create_directory, is_directory, is_file, move_directory, move_file, remove_directory, remove_file, resolve_file_paths
from facefusion.jobs import job_helper, job_manager
from facefusion.types import Args, JobId, JobOutputSet, ProcessStep, Step, StepIndex, StepInputSet
from facefusion.workflows.core import clear_input_caches


def run_job(job_id : JobId, process_step : ProcessStep) -> bool:
	queued_job_ids = job_manager.find_job_ids('queued')

	if job_id in queued_job_ids:
		fork_id = session_manager.fork_session()
		session_manager.observe_cli_session(fork_id)
		store_manager.observe_cli_session(fork_id)
		state_manager.clone_state()

		process_manager.start()
		job_manager.clear_expired_step_videos(state_manager.get_jobs_path())

		if run_steps(job_id, process_step) and job_id in job_manager.find_job_ids('queued') and finalize_steps(job_id):
			process_manager.end()
			session_manager.join_session()
			try:
				completed = job_manager.move_job_file(job_id, 'completed')
			except OSError:
				completed = False
			clean_steps(job_id, keep_completed_videos = not completed)
			return completed

		clean_steps(job_id, keep_completed_videos = True)
		process_manager.end()
		session_manager.join_session()

		if job_id in job_manager.find_job_ids('queued'):
			job_manager.move_job_file(job_id, 'failed')
	return False


def run_jobs(process_step : ProcessStep, halt_on_error : bool) -> bool:
	queued_job_ids = job_manager.find_job_ids('queued')
	has_error = False

	if queued_job_ids:
		for job_id in queued_job_ids:
			if not run_job(job_id, process_step):
				has_error = True
				if halt_on_error:
					return False
		return not has_error
	return False


def retry_job(job_id : JobId, process_step : ProcessStep) -> bool:
	failed_job_ids = job_manager.find_job_ids('failed')

	if job_id in failed_job_ids:
		return job_manager.set_steps_status(job_id, 'queued', retry = True) and job_manager.move_job_file(job_id, 'queued') and run_job(job_id, process_step)
	return False


def retry_jobs(process_step : ProcessStep, halt_on_error : bool) -> bool:
	failed_job_ids = job_manager.find_job_ids('failed')
	has_error = False

	if failed_job_ids:
		for job_id in failed_job_ids:
			if not retry_job(job_id, process_step):
				has_error = True
				if halt_on_error:
					return False
		return not has_error
	return False


def get_resume_inputs(step_args : Args) -> Optional[StepInputSet]:
	resume_inputs : StepInputSet = {}

	for input_path in job_helper.get_step_input_paths(step_args):
		try:
			input_stat = os.stat(input_path)
			resume_inputs[input_path] = [ input_stat.st_size, input_stat.st_mtime_ns ]
		except (FileNotFoundError, NotADirectoryError):
			resume_inputs[input_path] = None
		except OSError:
			return None
	return resume_inputs


def has_matching_available_inputs(saved_inputs : Optional[StepInputSet], current_inputs : Optional[StepInputSet]) -> bool:
	if saved_inputs is None or current_inputs is None or saved_inputs.keys() != current_inputs.keys():
		return False

	for input_path, input_stat in current_inputs.items():
		if input_stat is not None and input_stat != saved_inputs[input_path]:
			return False
	return True


def run_step(job_id : JobId, step_index : StepIndex, step : Step, process_step : ProcessStep) -> bool:
	step_args = step.get('args')
	output_path = step_args.get('output_path')
	step_output_path = job_helper.get_step_output_path(job_id, step_index, output_path, step_args.get('resume'))
	resume_inputs = get_resume_inputs(step_args) if step_args.get('resume') is True else None

	if job_manager.can_resume_step(step):
		inputs_match = has_matching_available_inputs(step.get('resume_inputs'), resume_inputs)
		if inputs_match and job_manager.has_completed_video(step, step_output_path) and probe_video_entries(step_output_path, [ 'codec_name' ]):
			return True
		if not inputs_match:
			clear_input_caches(job_helper.get_step_input_paths(step_args))

	if step_args.get('resume') is True and is_directory(job_helper.get_step_frame_directory(job_id, step_index)):
		if resume_inputs is None or step.get('resume_inputs') != resume_inputs:
			if not job_manager.clear_step_frames(job_id, step_index):
				return False
			clear_input_caches(job_helper.get_step_input_paths(step_args))

	if job_manager.set_step_status(job_id, step_index, 'started', resume_inputs, step_output_path) and process_step(job_id, step_index, step_args) == 0:
		if step_args.get('resume') is True and step_output_path and os.path.dirname(step_output_path) == job_helper.get_step_video_directory(job_id):
			if not create_directory(os.path.dirname(step_output_path)):
				return False
		if is_directory(output_path):
			return move_directory(output_path, step_output_path) and job_manager.set_step_status(job_id, step_index, 'completed')
		if move_file(output_path, step_output_path) and job_manager.set_step_status(job_id, step_index, 'completed', resume_inputs):
			if step_args.get('resume') is True:
				job_manager.clear_step_frames(job_id, step_index)
			return True
		return False
	job_manager.set_step_status(job_id, step_index, 'failed', resume_inputs)
	return False


def run_steps(job_id : JobId, process_step : ProcessStep) -> bool:
	steps = job_manager.get_steps(job_id)

	if steps:
		relocated_step_outputs : Dict[str, str] = {}
		for index, step in enumerate(steps):
			if step.get('args').get('resume') is True:
				current_steps = job_manager.get_steps(job_id)
				if index >= len(current_steps):
					return False
				step = current_steps[index]
			step_args = step.get('args')
			if relocated_step_outputs:
				step = step.copy()
				step['args'] = job_helper.resolve_step_input_paths(step_args, relocated_step_outputs)
			if not run_step(job_id, index, step, process_step):
				return False
			if step_args.get('resume') is True and step_args.get('output_path'):
				original_step_output_path = job_helper.get_step_output_path(job_id, index, step_args['output_path'])
				retained_step_output_path = job_helper.get_step_output_path(job_id, index, step_args['output_path'], True)
				if original_step_output_path != retained_step_output_path:
					relocated_step_outputs[os.path.normcase(os.path.realpath(original_step_output_path))] = retained_step_output_path
		return True
	return False


def finalize_steps(job_id : JobId) -> bool:
	output_set = collect_output_set(job_id)

	for output_path, temp_output_paths in output_set.items():
		has_videos = are_videos(temp_output_paths)
		has_images = are_images(temp_output_paths)

		if has_videos:
			if not concat_video(output_path, temp_output_paths):
				return False
		if not has_videos and has_images:
			for temp_output_path in temp_output_paths:
				if not move_file(temp_output_path, output_path):
					return False
		if not has_videos and not has_images:
			if not create_directory(output_path):
				return False

			for temp_output_path in temp_output_paths:
				if is_directory(temp_output_path):
					temp_frame_paths = resolve_file_paths(temp_output_path)
					for temp_frame_path in temp_frame_paths:
						if not copy_file(temp_frame_path, os.path.join(output_path, os.path.basename(temp_frame_path))):
							return False
	return True


def clean_steps(job_id : JobId, keep_completed_videos : bool = False) -> bool:
	output_set = collect_output_set(job_id, keep_completed_videos)

	for temp_output_paths in output_set.values():
		for temp_output_path in temp_output_paths:
			if is_file(temp_output_path):
				if not remove_file(temp_output_path):
					return False
			if is_directory(temp_output_path):
				if not remove_directory(temp_output_path):
					return False
	directory_path = job_helper.get_step_video_directory(job_id)
	if is_directory(directory_path) and (not keep_completed_videos or not os.listdir(directory_path)):
		return remove_directory(directory_path)
	return True


def collect_output_set(job_id : JobId, keep_completed_videos : bool = False) -> JobOutputSet:
	steps = job_manager.get_steps(job_id)
	job_output_set : JobOutputSet = {}

	for index, step in enumerate(steps):
		output_path = step.get('args').get('output_path')

		if output_path:
			step_output_path = job_helper.get_step_output_path(job_id, index, output_path, step.get('args').get('resume'))
			if keep_completed_videos and job_manager.has_completed_video(step, step_output_path):
				continue
			job_output_set.setdefault(output_path, []).append(step_output_path)
	return job_output_set
