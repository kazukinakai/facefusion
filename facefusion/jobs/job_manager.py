import os
from copy import copy
from datetime import timedelta
from glob import iglob
from time import time
from typing import List, Optional, cast

import facefusion.choices
from facefusion import logger, state_manager, translator
from facefusion.filesystem import create_directory, get_file_name, get_file_size, is_directory, is_file, is_video, move_file, remove_directory, remove_file, resolve_file_paths, resolve_file_pattern
from facefusion.jobs.job_helper import get_step_frame_directory, get_step_output_path, get_step_video_directory
from facefusion.json import read_json, write_json
from facefusion.sanitizer import sanitize_job_id
from facefusion.time_helper import get_current_date_time
from facefusion.types import Args, Content, Job, JobId, JobSet, JobStatus, Step, StepIndex, StepInputSet, StepStatus


def clear_expired_step_videos(jobs_path : str) -> None:
	cache_path = os.path.join(jobs_path, 'step-videos')
	if os.path.islink(jobs_path) or os.path.islink(cache_path):
		return
	expiration = time() - timedelta(days = 7).total_seconds()
	try:
		directory_paths = resolve_file_paths(cache_path)
	except OSError:
		return
	for directory_path in directory_paths:
		try:
			if not os.path.islink(directory_path) and is_directory(directory_path) and os.path.getmtime(directory_path) < expiration:
				frame_directories = iglob(os.path.join(directory_path, 'frames-*', '*'))
				if any(os.path.getmtime(frame_directory) >= expiration for frame_directory in frame_directories):
					continue
				if remove_directory(directory_path):
					logger.info(translator.get('step_videos_expired').format(job_id = os.path.basename(directory_path)), __name__)
		except OSError:
			continue


def init_jobs(jobs_path : str) -> bool:
	job_status_paths = [ os.path.join(jobs_path, job_status) for job_status in facefusion.choices.job_statuses ]

	for job_status_path in job_status_paths:
		create_directory(job_status_path)
	clear_expired_step_videos(jobs_path)
	return all(is_directory(status_path) for status_path in job_status_paths)


def clear_jobs(jobs_path : str) -> bool:
	return remove_directory(jobs_path)


def create_job(job_id : JobId) -> bool:
	job : Job =\
	{
		'version': '1',
		'date_created': get_current_date_time().isoformat(),
		'date_updated': None,
		'steps': []
	}

	return create_job_file(job_id, job)


def submit_job(job_id : JobId) -> bool:
	drafted_job_ids = find_job_ids('drafted')
	steps = get_steps(job_id)

	if job_id in drafted_job_ids and steps:
		return set_steps_status(job_id, 'queued') and move_job_file(job_id, 'queued')
	return False


def submit_jobs(halt_on_error : bool) -> bool:
	drafted_job_ids = find_job_ids('drafted')
	has_error = False

	if drafted_job_ids:
		for job_id in drafted_job_ids:
			if not submit_job(job_id):
				has_error = True
				if halt_on_error:
					return False
		return not has_error
	return False


def cancel_job(job_id : JobId) -> bool:
	queued_job_ids = find_job_ids('queued')

	if job_id in queued_job_ids and set_steps_status(job_id, 'drafted') and move_job_file(job_id, 'drafted'):
		if not clear_job_videos(job_id):
			logger.warn(translator.get('step_videos_not_removed').format(job_id = job_id), __name__)
		return True
	return False


def cancel_jobs(halt_on_error : bool) -> bool:
	queued_job_ids = find_job_ids('queued')
	has_error = False

	if queued_job_ids:
		for job_id in queued_job_ids:
			if not cancel_job(job_id):
				has_error = True
				if halt_on_error:
					return False
		return not has_error
	return False


def can_resume_step(step : Step) -> bool:
	return step.get('args').get('resume') is True and step.get('status') == 'completed'


def has_completed_video(step : Step, step_output_path : str) -> bool:
	return can_resume_step(step) and is_video(step_output_path) and get_file_size(step_output_path) > 0


def clear_step_frames(job_id : JobId, step_index : StepIndex) -> bool:
	directory_path = get_step_frame_directory(job_id, step_index)
	return not is_directory(directory_path) or remove_directory(directory_path)


def discard_step_videos(job_id : JobId, steps : List[Step], start_index : StepIndex = 0) -> bool:
	for index, step in enumerate(steps[start_index:], start_index):
		step_args = step.get('args')
		if step_args.get('resume') is True:
			if not clear_step_frames(job_id, index):
				return False
			video_path = get_step_output_path(job_id, index, step_args.get('output_path'), True)
			if is_video(video_path) and not remove_file(video_path):
				return False
	return True


def clear_job_videos(job_id : JobId) -> bool:
	directory_path = get_step_video_directory(job_id)
	return not is_directory(directory_path) or remove_directory(directory_path)


def delete_job(job_id : JobId) -> bool:
	return bool(job_id) and clear_job_videos(job_id) and delete_job_file(job_id)


def delete_jobs(halt_on_error : bool) -> bool:
	job_ids = find_job_ids('drafted') + find_job_ids('queued') + find_job_ids('failed') + find_job_ids('completed')
	has_error = False

	if job_ids:
		for job_id in job_ids:
			if not delete_job(job_id):
				has_error = True
				if halt_on_error:
					return False
		return not has_error
	return False


def find_jobs(job_status : JobStatus) -> JobSet:
	job_ids = find_job_ids(job_status)
	job_set : JobSet = {}

	for job_id in job_ids:
		job_set[job_id] = read_job_file(job_id)
	return job_set


def find_job_ids(job_status : JobStatus) -> List[JobId]:
	jobs_path = state_manager.get_jobs_path()
	job_file_pattern = os.path.join(jobs_path, job_status, '*.json')
	job_file_paths = resolve_file_pattern(job_file_pattern)
	job_file_paths.sort(key = os.path.getmtime)
	job_ids = []

	for job_file_path in job_file_paths:
		job_id = get_file_name(job_file_path)
		job_ids.append(job_id)
	return job_ids


def validate_job(job_id : JobId) -> bool:
	job = read_job_file(job_id)
	return bool(job and 'version' in job and 'date_created' in job and 'date_updated' in job and 'steps' in job)


def has_step(job_id : JobId, step_index : StepIndex) -> bool:
	step_total = count_step_total(job_id)
	return step_index in range(step_total)


def add_step(job_id : JobId, step_args : Args) -> bool:
	job = read_job_file(job_id)

	if job:
		job.get('steps').append(
		{
			'args': step_args,
			'status': 'drafted'
		})
		return update_job_file(job_id, job)
	return False


def remix_step(job_id : JobId, step_index : StepIndex, step_args : Args) -> bool:
	steps = get_steps(job_id)
	step_args = copy(step_args)

	if step_index and step_index < 0:
		step_index = count_step_total(job_id) - 1

	if has_step(job_id, step_index):
		output_path = steps[step_index].get('args').get('output_path')
		step_args['target_path'] = get_step_output_path(job_id, step_index, output_path, steps[step_index].get('args').get('resume'))
		return add_step(job_id, step_args)
	return False


def invalidate_step_indices(steps : List[Step]) -> None:
	for step in steps:
		if can_resume_step(step):
			step['status'] = 'drafted'
			step.pop('resume_inputs', None)


def insert_step(job_id : JobId, step_index : StepIndex, step_args : Args) -> bool:
	job = read_job_file(job_id)
	step_args = copy(step_args)

	if step_index and step_index < 0:
		step_index = count_step_total(job_id) - 1

	if job and has_step(job_id, step_index) and discard_step_videos(job_id, job.get('steps'), step_index):
		invalidate_step_indices(job.get('steps')[step_index:])
		job.get('steps').insert(step_index,
		{
			'args': step_args,
			'status': 'drafted'
		})
		return update_job_file(job_id, job)
	return False


def remove_step(job_id : JobId, step_index : StepIndex) -> bool:
	job = read_job_file(job_id)

	if step_index and step_index < 0:
		step_index = count_step_total(job_id) - 1

	if job and has_step(job_id, step_index) and discard_step_videos(job_id, job.get('steps'), step_index):
		invalidate_step_indices(job.get('steps')[step_index:])
		job.get('steps').pop(step_index)
		return update_job_file(job_id, job)
	return False


def get_steps(job_id : JobId) -> List[Step]:
	job = read_job_file(job_id)

	if job:
		return job.get('steps')
	return []


def count_step_total(job_id : JobId) -> int:
	steps = get_steps(job_id)

	if steps:
		return len(steps)
	return 0


def has_step_status(job_id : JobId, step_status : StepStatus) -> bool:
	return any(step.get('status') == step_status for step in get_steps(job_id))


def set_step_status(job_id : JobId, step_index : StepIndex, step_status : StepStatus, resume_inputs : Optional[StepInputSet] = None, step_output_path : Optional[str] = None) -> bool:
	job = read_job_file(job_id)

	if job:
		steps = job.get('steps')
		if step_index in range(len(steps)):
			step = steps[step_index]
			step['status'] = step_status
			if resume_inputs is not None:
				step['resume_inputs'] = resume_inputs
			else:
				step.pop('resume_inputs', None)
			if step_status == 'started' and step_output_path:
				step_output_path = os.path.normcase(os.path.realpath(step_output_path))
				for later_step in steps[step_index + 1:]:
					if 'resume_inputs' in later_step and any(os.path.normcase(os.path.realpath(input_path)) == step_output_path for input_path in later_step['resume_inputs']):
						later_step.pop('resume_inputs', None)
			return update_job_file(job_id, job)
	return False


def set_steps_status(job_id : JobId, step_status : StepStatus, retry : bool = False) -> bool:
	job = read_job_file(job_id)

	if job:
		for step in job.get('steps'):
			if retry and can_resume_step(step):
				continue
			step['status'] = step_status
			if not retry:
				step.pop('resume_inputs', None)
		return update_job_file(job_id, job)
	return False


def read_job_file(job_id : JobId) -> Optional[Job]:
	job_path = find_job_path(job_id)
	return cast(Job, read_json(job_path))


def create_job_file(job_id : JobId, job : Job) -> bool:
	job_path = find_job_path(job_id)

	if not is_file(job_path):
		job_create_path = suggest_job_path(job_id, 'drafted')
		return write_json(job_create_path, cast(Content, job))
	return False


def update_job_file(job_id : JobId, job : Job) -> bool:
	job_path = find_job_path(job_id)

	if is_file(job_path):
		job['date_updated'] = get_current_date_time().isoformat()
		return write_json(job_path, cast(Content, job))
	return False


def move_job_file(job_id : JobId, job_status : JobStatus) -> bool:
	job_path = find_job_path(job_id)
	job_move_path = suggest_job_path(job_id, job_status)
	return move_file(job_path, job_move_path)


def delete_job_file(job_id : JobId) -> bool:
	job_path = find_job_path(job_id)
	return remove_file(job_path)


def suggest_job_path(job_id : JobId, job_status : JobStatus) -> Optional[str]:
	jobs_path = state_manager.get_jobs_path()
	job_file_name = get_job_file_name(job_id)

	if job_file_name:
		return os.path.join(jobs_path, job_status, job_file_name)
	return None


def find_job_path(job_id : JobId) -> Optional[str]:
	jobs_path = state_manager.get_jobs_path()
	job_file_name = get_job_file_name(job_id)

	if job_file_name:
		for job_status in facefusion.choices.job_statuses:
			job_file_pattern = os.path.join(jobs_path, job_status, job_file_name)
			job_file_paths = resolve_file_pattern(job_file_pattern)

			for job_file_path in job_file_paths:
				return job_file_path
	return None


def get_job_file_name(job_id : JobId) -> Optional[str]:
	if job_id:
		job_id = sanitize_job_id(job_id)
		return job_id + '.json'
	return None
