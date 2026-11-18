import os
from pathlib import Path
from unittest.mock import patch

import pytest

from facefusion import ffmpeg, ffmpeg_builder, process_manager, session_context, session_manager, state_manager
from facefusion.download import conditional_download
from facefusion.filesystem import copy_file, create_directory, get_file_extension
from facefusion.jobs import job_helper, job_manager
from facefusion.jobs.job_manager import add_step, clear_jobs, create_job, find_job_ids, init_jobs, move_job_file, submit_job, submit_jobs
from facefusion.jobs.job_runner import clean_steps, collect_output_set, finalize_steps, retry_job, retry_jobs, run_job, run_jobs, run_steps
from facefusion.types import Args, ErrorCode, JobId, StepIndex, StepStatus
from facefusion.vision import count_video_frame_total
from tests.assert_helper import get_test_example_file, get_test_examples_directory, get_test_jobs_directory, get_test_output_path, is_test_output_file, is_test_output_sequence, prepare_test_output_directory


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	state_manager.init()
	state_manager.init_item('jobs_path', get_test_jobs_directory())

	process_manager.start()
	conditional_download(get_test_examples_directory(),
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/source.jpg',
		'https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-240p.mp4'
	])

	ffmpeg.run_ffmpeg(
		ffmpeg_builder.chain(
			ffmpeg_builder.set_input(get_test_example_file('target-240p.mp4')),
			[
				'-vframes',
				'1'
			],
			ffmpeg_builder.set_output(get_test_example_file('target-240p.jpg'))
		)
	).wait()


@pytest.fixture(scope = 'function', autouse = True)
def before_each() -> None:
	clear_jobs(get_test_jobs_directory())
	init_jobs(state_manager.get_jobs_path())
	prepare_test_output_directory()


def process_step(job_id : JobId, step_index : StepIndex, step_args : Args) -> ErrorCode:
	output_path = step_args.get('output_path')
	target_path = step_args.get('target_path')

	if output_path and not get_file_extension(output_path):
		move_path = os.path.join(output_path, os.path.basename(target_path))

		if create_directory(output_path) and copy_file(target_path, move_path):
			return 0
		return 1

	if copy_file(target_path, output_path):
		return 0
	return 1


def test_run_job() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-2.mp4')
	}
	args_3 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-3.jpg')
	}
	args_4 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-4')
	}
	args_5 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-4')
	}
	args_6 =\
	{
		'source_path': get_test_example_file('invalid'),
		'target_path': get_test_example_file('invalid'),
		'output_path': get_test_output_path('invalid')
	}

	assert run_job('job-invalid', process_step) is False

	create_job('job-test-run-job')
	add_step('job-test-run-job', args_1)
	add_step('job-test-run-job', args_2)
	add_step('job-test-run-job', args_2)
	add_step('job-test-run-job', args_3)
	add_step('job-test-run-job', args_4)
	add_step('job-test-run-job', args_5)

	assert run_job('job-test-run-job', process_step) is False

	submit_job('job-test-run-job')

	with patch('facefusion.session_manager.fork_session', wraps = session_manager.fork_session) as fork_session_mock:
		with patch('facefusion.process_manager.start', wraps = process_manager.start) as start_mock:
			with patch('facefusion.process_manager.end', wraps = process_manager.end) as end_mock:
				assert run_job('job-test-run-job', process_step) is True

	session_id = session_context.get_session_id()

	assert fork_session_mock.call_count == 1
	assert start_mock.call_count == 1
	assert end_mock.call_count == 1
	assert session_manager.has_fork_session(session_id) is False

	create_job('job-test-run-job-cancel')
	add_step('job-test-run-job-cancel', args_1)
	add_step('job-test-run-job-cancel', args_3)
	submit_job('job-test-run-job-cancel')

	with patch('facefusion.jobs.job_manager.find_job_ids', side_effect = [ [ 'job-test-run-job-cancel' ], [], [] ]):
		assert run_job('job-test-run-job-cancel', process_step) is False

	assert find_job_ids('queued') == [ 'job-test-run-job-cancel' ]
	assert is_test_output_file('output-1.mp4') is False

	create_job('job-test-run-job-invalid')
	add_step('job-test-run-job-invalid', args_6)
	submit_job('job-test-run-job-invalid')

	assert run_job('job-test-run-job-invalid', process_step) is False
	assert find_job_ids('failed') == [ 'job-test-run-job-invalid' ]

	create_job('job-test-run-job-cancel-invalid')
	add_step('job-test-run-job-cancel-invalid', args_6)
	submit_job('job-test-run-job-cancel-invalid')

	with patch('facefusion.jobs.job_manager.find_job_ids', side_effect = [ [ 'job-test-run-job-cancel-invalid' ], [] ]):
		assert run_job('job-test-run-job-cancel-invalid', process_step) is False

	assert find_job_ids('failed') == [ 'job-test-run-job-invalid' ]


def test_run_jobs() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-2.mp4')
	}
	args_3 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-3.jpg')
	}
	halt_on_error = True

	assert run_jobs(process_step, halt_on_error) is False

	create_job('job-test-run-jobs-1')
	create_job('job-test-run-jobs-2')
	add_step('job-test-run-jobs-1', args_1)
	add_step('job-test-run-jobs-1', args_1)
	add_step('job-test-run-jobs-2', args_2)
	add_step('job-test-run-jobs-3', args_3)

	assert run_jobs(process_step, halt_on_error) is False

	submit_jobs(halt_on_error)

	assert run_jobs(process_step, halt_on_error) is True


def test_retry_job() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}

	assert retry_job('job-invalid', process_step) is False

	create_job('job-test-retry-job')
	add_step('job-test-retry-job', args_1)
	submit_job('job-test-retry-job')

	assert retry_job('job-test-retry-job', process_step) is False

	move_job_file('job-test-retry-job', 'failed')

	assert retry_job('job-test-retry-job', process_step) is True


def test_retry_jobs() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-2.mp4')
	}
	args_3 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-3.jpg')
	}
	halt_on_error = True

	assert retry_jobs(process_step, halt_on_error) is False

	create_job('job-test-retry-jobs-1')
	create_job('job-test-retry-jobs-2')
	add_step('job-test-retry-jobs-1', args_1)
	add_step('job-test-retry-jobs-1', args_1)
	add_step('job-test-retry-jobs-2', args_2)
	add_step('job-test-retry-jobs-3', args_3)

	assert retry_jobs(process_step, halt_on_error) is False

	move_job_file('job-test-retry-jobs-1', 'failed')
	move_job_file('job-test-retry-jobs-2', 'failed')

	assert retry_jobs(process_step, halt_on_error) is True


def test_run_steps() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-2.mp4')
	}
	args_3 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-3.jpg')
	}

	assert run_steps('job-invalid', process_step) is False

	create_job('job-test-run-steps')
	add_step('job-test-run-steps', args_1)
	add_step('job-test-run-steps', args_1)
	add_step('job-test-run-steps', args_2)
	add_step('job-test-run-steps', args_3)

	assert run_steps('job-test-run-steps', process_step) is True


def test_finalize_steps() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-2.mp4')
	}
	args_3 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-3.jpg')
	}
	args_4 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-4')
	}
	args_5 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-4')
	}

	create_job('job-test-finalize-steps')
	add_step('job-test-finalize-steps', args_1)
	add_step('job-test-finalize-steps', args_1)
	add_step('job-test-finalize-steps', args_2)
	add_step('job-test-finalize-steps', args_3)
	add_step('job-test-finalize-steps', args_4)
	add_step('job-test-finalize-steps', args_5)

	copy_file(args_1.get('target_path'), get_test_output_path('output-1-job-test-finalize-steps-0.mp4'))
	copy_file(args_1.get('target_path'), get_test_output_path('output-1-job-test-finalize-steps-1.mp4'))
	copy_file(args_2.get('target_path'), get_test_output_path('output-2-job-test-finalize-steps-2.mp4'))
	copy_file(args_3.get('target_path'), get_test_output_path('output-3-job-test-finalize-steps-3.jpg'))

	temp_directory_1 = get_test_output_path('output-4-job-test-finalize-steps-4')
	temp_directory_2 = get_test_output_path('output-4-job-test-finalize-steps-5')
	create_directory(temp_directory_1)
	create_directory(temp_directory_2)
	copy_file(args_4.get('target_path'), os.path.join(temp_directory_1, '00000001.jpg'))
	copy_file(args_5.get('target_path'), os.path.join(temp_directory_2, '00000002.jpg'))
	process_manager.start()

	assert finalize_steps('job-test-finalize-steps') is True
	assert is_test_output_file('output-1.mp4') is True
	assert is_test_output_file('output-2.mp4') is True
	assert is_test_output_file('output-3.jpg') is True
	assert is_test_output_sequence(get_test_output_path('output-4')) is True


def test_clean_steps() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-2.jpg')
	}

	create_job('job-test-clean-steps')
	add_step('job-test-clean-steps', args_1)
	add_step('job-test-clean-steps', args_2)

	copy_file(args_1.get('target_path'), get_test_output_path('output-1-job-test-clean-steps-0.mp4'))
	copy_file(args_2.get('target_path'), get_test_output_path('output-2-job-test-clean-steps-1.jpg'))

	assert is_test_output_file('output-1-job-test-clean-steps-0.mp4') is True
	assert is_test_output_file('output-2-job-test-clean-steps-1.jpg') is True

	assert clean_steps('job-test-clean-steps') is True
	assert is_test_output_file('output-1-job-test-clean-steps-0.mp4') is False
	assert is_test_output_file('output-2-job-test-clean-steps-1.jpg') is False


def test_collect_output_set() -> None:
	args_1 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-1.mp4')
	}
	args_2 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-2.mp4')
	}
	args_3 =\
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.jpg'),
		'output_path': get_test_output_path('output-3.jpg')
	}
	args_4 = \
	{
		'source_path': get_test_example_file('source.jpg'),
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output-4')
	}

	create_job('job-test-collect-output-set')
	add_step('job-test-collect-output-set', args_1)
	add_step('job-test-collect-output-set', args_1)
	add_step('job-test-collect-output-set', args_2)
	add_step('job-test-collect-output-set', args_3)
	add_step('job-test-collect-output-set', args_4)

	output_set =\
	{
		get_test_output_path('output-1.mp4'):
		[
			get_test_output_path('output-1-job-test-collect-output-set-0.mp4'),
			get_test_output_path('output-1-job-test-collect-output-set-1.mp4')
		],
		get_test_output_path('output-2.mp4'):
		[
			get_test_output_path('output-2-job-test-collect-output-set-2.mp4')
		],
		get_test_output_path('output-3.jpg'):
		[
			get_test_output_path('output-3-job-test-collect-output-set-3.jpg')
		],
		get_test_output_path('output-4'):
		[
			get_test_output_path('output-4-job-test-collect-output-set-4')
		]
	}

	assert collect_output_set('job-test-collect-output-set') == output_set


@pytest.fixture
def video_step_args() -> Args:
	return {
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': get_test_output_path('output.mp4'),
		'resume': True
	}


@pytest.mark.parametrize('resume', [ False, True ])
def test_retry_retains_and_concatenates_step_videos(resume : bool, video_step_args : Args) -> None:
	job_id = 'job-test-resume'
	args = video_step_args
	args['resume'] = resume
	calls = []

	def interrupted_step(job_id : JobId, step_index : StepIndex, step_args : Args) -> ErrorCode:
		calls.append(step_index)
		if calls == [ 0, 1 ]:
			return 1
		return process_step(job_id, step_index, step_args)

	create_job(job_id)
	add_step(job_id, args)
	add_step(job_id, args)
	submit_job(job_id)
	assert run_job(job_id, interrupted_step) is False
	step_output_path = job_helper.get_step_output_path(job_id, 0, args.get('output_path'))
	assert os.path.isfile(step_output_path) is resume
	assert retry_job(job_id, interrupted_step) is True
	assert calls == ([ 0, 1, 1 ] if resume else [ 0, 1, 0, 1 ])
	assert count_video_frame_total(args.get('output_path')) == count_video_frame_total(args.get('target_path')) * 2
	assert os.path.isfile(step_output_path) is False
	assert find_job_ids('completed') == [ job_id ]


@pytest.mark.parametrize(('status', 'output', 'reused'), [
	('started', 'video', False),
	('completed', 'missing', False),
	('completed', 'empty', False),
	('completed', 'image', False),
	('completed', 'video', True)
])
def test_run_job_reuses_completed_video(status : StepStatus, output : str, reused : bool, video_step_args : Args) -> None:
	job_id = 'job-test-resume'
	if output == 'image':
		video_step_args['target_path'] = get_test_example_file('target-240p.jpg')
		video_step_args['output_path'] = get_test_output_path('output.jpg')
	create_job(job_id)
	add_step(job_id, video_step_args)
	submit_job(job_id)
	job_manager.set_step_status(job_id, 0, status)
	step_output_path = job_helper.get_step_output_path(job_id, 0, video_step_args.get('output_path'))
	if output == 'empty':
		Path(step_output_path).touch()
	if output in [ 'image', 'video' ]:
		assert copy_file(video_step_args.get('target_path'), step_output_path)
	with patch('tests.jobs.test_job_runner.process_step', wraps = process_step) as process_step_mock:
		assert run_job(job_id, process_step_mock)
	assert process_step_mock.call_count == (0 if reused else 1)


@pytest.mark.parametrize('failure', [ 'concat', 'completion' ])
def test_resume_retains_videos_until_job_completion(failure : str, video_step_args : Args) -> None:
	job_id = 'job-test-resume-finalize'
	args = video_step_args
	create_job(job_id)
	add_step(job_id, args)
	submit_job(job_id)
	step_output_path = job_helper.get_step_output_path(job_id, 0, args.get('output_path'))
	if failure == 'concat':
		with patch('facefusion.jobs.job_runner.concat_video', return_value = False):
			assert run_job(job_id, process_step) is False
	else:
		with patch('facefusion.jobs.job_manager.move_job_file', return_value = False):
			assert run_job(job_id, process_step) is False
	assert os.path.isfile(step_output_path) is True
	with patch('tests.jobs.test_job_runner.process_step', wraps = process_step) as process_step_mock:
		if failure == 'concat':
			assert retry_job(job_id, process_step_mock) is True
		else:
			assert run_job(job_id, process_step_mock) is True
	assert process_step_mock.call_count == 0
	assert os.path.isfile(step_output_path) is False


@pytest.mark.parametrize('operation', [ 'delete', 'insert', 'remove', 'cancel' ])
def test_resume_cleans_videos_on_job_edit_or_deletion(operation : str, video_step_args : Args) -> None:
	job_id = 'job-test-resume-edit'
	args = video_step_args
	create_job(job_id)
	add_step(job_id, args)
	submit_job(job_id)
	step_output_path = job_helper.get_step_output_path(job_id, 0, args.get('output_path'))
	job_manager.set_step_status(job_id, 0, 'completed')
	other_step_path = job_helper.get_step_output_path('other-job', 0, args.get('output_path'))
	assert copy_file(args.get('target_path'), step_output_path) is True
	assert copy_file(args.get('target_path'), other_step_path) is True
	if operation == 'cancel':
		assert job_manager.cancel_job(job_id) is True
	if operation == 'delete':
		assert job_manager.delete_job(job_id) is True
	if operation == 'insert':
		assert job_manager.insert_step(job_id, 0, args) is True
	if operation == 'remove':
		assert job_manager.remove_step(job_id, 0) is True
	assert os.path.isfile(step_output_path) is False
	assert os.path.isfile(other_step_path) is True
