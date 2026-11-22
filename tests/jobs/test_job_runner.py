import os
from unittest.mock import Mock, patch

import pytest

from facefusion import ffmpeg, ffmpeg_builder, process_manager, session_context, session_manager, state_manager
from facefusion.download import conditional_download
from facefusion.filesystem import copy_file, create_directory, get_file_extension
from facefusion.jobs import job_manager, job_runner
from facefusion.jobs.job_manager import add_step, clean_steps, clear_jobs, create_job, find_job_ids, init_jobs, move_job_file, submit_job, submit_jobs
from facefusion.jobs.job_runner import collect_output_set, finalize_steps, retry_job, retry_jobs, run_job, run_jobs, run_steps
from facefusion.types import Args, ErrorCode, JobId, StepIndex
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


def test_optimize_job() -> None:
	job_id = 'job-test-optimize'
	args = { 'target_path': get_test_example_file('target-240p.mp4'), 'output_path': get_test_output_path('output.mp4'), 'trim_frame_start': 37 }
	assert create_job(job_id) is True
	assert add_step(job_id, args) is True
	with patch('facefusion.vision.count_video_frame_total', return_value = 2250):
		assert job_manager.optimize_job(job_id)
	assert job_manager.get_steps(job_id) == [
		{ 'args': dict(args, trim_frame_start = start, trim_frame_end = end), 'status': 'drafted' }
		for start, end in [ (37, 1037), (1037, 2037), (2037, 2250) ]
	]
	assert not job_manager.optimize_job(job_id)


@pytest.mark.parametrize('frame_total, overrides',
[
	(0, {}),
	(3, { 'trim_frame_end': 0 }),
	(3, { 'output_video_fps': 1 }),
	(3, { 'output_path': get_test_example_file('target-240p.mp4') })
])
def test_optimize_preserves_invalid_draft(frame_total : int, overrides : Args) -> None:
	job_id = 'job-test-invalid-optimize'
	args = { 'target_path': get_test_example_file('target-240p.mp4'), 'output_path': get_test_output_path('output.mp4'), **overrides }
	assert create_job(job_id) is True
	assert add_step(job_id, args) is True
	before = job_manager.read_job_file(job_id)
	assert not job_manager.optimize_job(job_id, frame_total)
	assert job_manager.read_job_file(job_id) == before


@pytest.fixture
def interrupted_job() -> JobId:
	job_id = 'job-test-interrupted'
	args = { 'target_path': get_test_example_file('source.jpg'), 'output_path': get_test_output_path('output.jpg') }
	assert create_job(job_id)
	for _ in range(3):
		assert add_step(job_id, args)
	assert submit_job(job_id)
	for index, step in enumerate(job_manager.get_steps(job_id)[:2]):
		assert job_runner.run_step(job_id, index, step, process_step)
	assert job_manager.set_step_status(job_id, 2, 'started')
	return job_id


@pytest.fixture
def failed_job(interrupted_job : JobId) -> JobId:
	assert not run_job(interrupted_job, Mock(return_value = 1))
	return interrupted_job


def test_resume_queued_job(interrupted_job : JobId) -> None:
	process = Mock(wraps = process_step)
	assert run_job(interrupted_job, process)
	assert [ call.args[1] for call in process.call_args_list ] == [ 2 ]


def test_retry_failed_step(failed_job : JobId) -> None:
	process = Mock(wraps = process_step)
	assert retry_job(failed_job, process)
	assert [ call.args[1] for call in process.call_args_list ] == [ 2 ]


def test_retry_missing_output(failed_job : JobId) -> None:
	os.remove(get_test_output_path('output-job-test-interrupted-0.jpg'))
	process = Mock(wraps = process_step)
	assert retry_job(failed_job, process)
	assert [ call.args[1] for call in process.call_args_list ] == [ 0, 2 ]


def test_retry_after_insert(failed_job : JobId) -> None:
	assert job_manager.insert_step(failed_job, 1, job_manager.get_steps(failed_job)[0]['args'])
	process = Mock(wraps = process_step)
	assert retry_job(failed_job, process)
	assert [ call.args[1] for call in process.call_args_list ] == [ 1, 2, 3 ]


def test_retry_after_remove(failed_job : JobId) -> None:
	assert job_manager.remove_step(failed_job, 0)
	process = Mock(wraps = process_step)
	assert retry_job(failed_job, process)
	assert [ call.args[1] for call in process.call_args_list ] == [ 0, 1 ]


def test_cancel_interrupted_job(interrupted_job : JobId) -> None:
	assert job_manager.cancel_job(interrupted_job)
	assert not is_test_output_file('output-job-test-interrupted-0.jpg')


def test_delete_failed_job(failed_job : JobId) -> None:
	assert job_manager.delete_job(failed_job)
	assert not is_test_output_file('output-job-test-interrupted-0.jpg')


def test_retry_finalization() -> None:
	job_id = 'job-test-retry-finalization'
	assert create_job(job_id)
	for target, output in [ ('source.jpg', 'output.jpg'), ('target-240p.mp4', 'output.mp4'), ('target-240p.mp4', 'output.mp4') ]:
		assert add_step(job_id, { 'target_path': get_test_example_file(target), 'output_path': get_test_output_path(output) })
	assert submit_job(job_id)
	with patch.object(job_runner, 'concat_video', return_value = False):
		assert not run_job(job_id, process_step)
	process = Mock(wraps = process_step)
	with patch.object(job_runner, 'concat_video', wraps = ffmpeg.concat_video) as concat:
		assert retry_job(job_id, process)
	concat.assert_called_once_with(get_test_output_path('output.mp4'), [ get_test_output_path('output-job-test-retry-finalization-1.mp4'), get_test_output_path('output-job-test-retry-finalization-2.mp4') ])
	process.assert_not_called()
	assert is_test_output_file('output.jpg') is True
	assert is_test_output_file('output.mp4') is True
	assert not is_test_output_file('output-job-test-retry-finalization-0.jpg')
