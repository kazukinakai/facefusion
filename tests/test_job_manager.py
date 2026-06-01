from math import ceil
from time import sleep

import pytest

from facefusion.download import conditional_download
from facefusion.jobs.job_helper import get_step_output_path
from facefusion.jobs.job_manager import add_step, clear_jobs, count_step_total, create_job, delete_job, delete_jobs, find_job_ids, find_jobs, get_steps, init_jobs, insert_step, move_job_file, optimize_job, remix_step, remove_step, set_failed_steps_status, set_step_status, set_steps_status, submit_job, submit_jobs
from facefusion.vision import count_video_frame_total
from .helper import get_test_example_file, get_test_examples_directory, get_test_jobs_directory


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download(get_test_examples_directory(),
	[
		'https://github.com/facefusion/facefusion-assets/releases/download/examples-3.0.0/target-240p.mp4'
	])


@pytest.fixture(scope = 'function', autouse = True)
def before_each() -> None:
	clear_jobs(get_test_jobs_directory())
	init_jobs(get_test_jobs_directory())


def test_create_job() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}

	assert create_job('job-test-create-job') is True
	assert create_job('job-test-create-job') is False

	add_step('job-test-submit-job', args_1)
	submit_job('job-test-create-job')

	assert create_job('job-test-create-job') is False


def test_submit_job() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}

	assert submit_job('job-invalid') is False

	create_job('job-test-submit-job')

	assert submit_job('job-test-submit-job') is False

	add_step('job-test-submit-job', args_1)

	assert submit_job('job-test-submit-job') is True
	assert submit_job('job-test-submit-job') is False


def test_submit_jobs() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}
	halt_on_error = True

	assert submit_jobs(halt_on_error) is False

	create_job('job-test-submit-jobs-1')
	create_job('job-test-submit-jobs-2')

	assert submit_jobs(halt_on_error) is False

	add_step('job-test-submit-jobs-1', args_1)
	add_step('job-test-submit-jobs-2', args_2)

	assert submit_jobs(halt_on_error) is True
	assert submit_jobs(halt_on_error) is False


def test_delete_job() -> None:
	assert delete_job('job-invalid') is False

	create_job('job-test-delete-job')

	assert delete_job('job-test-delete-job') is True
	assert delete_job('job-test-delete-job') is False


def test_delete_jobs() -> None:
	halt_on_error = True

	assert delete_jobs(halt_on_error) is False

	create_job('job-test-delete-jobs-1')
	create_job('job-test-delete-jobs-2')

	assert delete_jobs(halt_on_error) is True


def test_find_jobs() -> None:
	create_job('job-test-find-jobs-1')
	sleep(0.5)
	create_job('job-test-find-jobs-2')

	assert 'job-test-find-jobs-1' in find_jobs('drafted')
	assert 'job-test-find-jobs-2' in find_jobs('drafted')
	assert not find_jobs('queued')

	move_job_file('job-test-find-jobs-1', 'queued')

	assert 'job-test-find-jobs-2' in find_jobs('drafted')
	assert 'job-test-find-jobs-1' in find_jobs('queued')


def test_find_job_ids() -> None:
	create_job('job-test-find-job-ids-1')
	sleep(0.5)
	create_job('job-test-find-job-ids-2')
	sleep(0.5)
	create_job('job-test-find-job-ids-3')

	assert find_job_ids('drafted') == [ 'job-test-find-job-ids-1', 'job-test-find-job-ids-2', 'job-test-find-job-ids-3' ]
	assert find_job_ids('queued') == []
	assert find_job_ids('completed') == []
	assert find_job_ids('failed') == []

	move_job_file('job-test-find-job-ids-1', 'queued')
	move_job_file('job-test-find-job-ids-2', 'queued')
	move_job_file('job-test-find-job-ids-3', 'queued')

	assert find_job_ids('drafted') == []
	assert find_job_ids('queued') == [ 'job-test-find-job-ids-1', 'job-test-find-job-ids-2', 'job-test-find-job-ids-3' ]
	assert find_job_ids('completed') == []
	assert find_job_ids('failed') == []

	move_job_file('job-test-find-job-ids-1', 'completed')

	assert find_job_ids('drafted') == []
	assert find_job_ids('queued') == [ 'job-test-find-job-ids-2', 'job-test-find-job-ids-3' ]
	assert find_job_ids('completed') == [ 'job-test-find-job-ids-1' ]
	assert find_job_ids('failed') == []

	move_job_file('job-test-find-job-ids-2', 'failed')

	assert find_job_ids('drafted') == []
	assert find_job_ids('queued') == [ 'job-test-find-job-ids-3' ]
	assert find_job_ids('completed') == [ 'job-test-find-job-ids-1' ]
	assert find_job_ids('failed') == [ 'job-test-find-job-ids-2' ]

	move_job_file('job-test-find-job-ids-3', 'completed')

	assert find_job_ids('drafted') == []
	assert find_job_ids('queued') == []
	assert find_job_ids('completed') == [ 'job-test-find-job-ids-1', 'job-test-find-job-ids-3' ]
	assert find_job_ids('failed') == [ 'job-test-find-job-ids-2' ]


def test_add_step() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}

	assert add_step('job-invalid', args_1) is False

	create_job('job-test-add-step')

	assert add_step('job-test-add-step', args_1) is True
	assert add_step('job-test-add-step', args_2) is True

	steps = get_steps('job-test-add-step')

	assert steps[0].get('args') == args_1
	assert steps[1].get('args') == args_2
	assert count_step_total('job-test-add-step') == 2


def test_remix_step() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}

	assert remix_step('job-invalid', 0, args_1) is False

	create_job('job-test-remix-step')
	add_step('job-test-remix-step', args_1)
	add_step('job-test-remix-step', args_2)

	assert remix_step('job-test-remix-step', 99, args_1) is False
	assert remix_step('job-test-remix-step', 0, args_2) is True
	assert remix_step('job-test-remix-step', -1, args_2) is True

	steps = get_steps('job-test-remix-step')

	assert steps[0].get('args') == args_1
	assert steps[1].get('args') == args_2
	assert steps[2].get('args').get('source_path') == args_2.get('source_path')
	assert steps[2].get('args').get('target_path') == get_step_output_path('job-test-remix-step', 0, args_1.get('output_path'))
	assert steps[2].get('args').get('output_path') == args_2.get('output_path')
	assert steps[3].get('args').get('source_path') == args_2.get('source_path')
	assert steps[3].get('args').get('target_path') == get_step_output_path('job-test-remix-step', 2, args_2.get('output_path'))
	assert steps[3].get('args').get('output_path') == args_2.get('output_path')
	assert count_step_total('job-test-remix-step') == 4


def test_insert_step() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}
	args_3 =\
	{
		'source_path': 'source-3.jpg',
		'target_path': 'target-3.jpg',
		'output_path': 'output-3.jpg'
	}

	assert insert_step('job-invalid', 0, args_1) is False

	create_job('job-test-insert-step')
	add_step('job-test-insert-step', args_1)
	add_step('job-test-insert-step', args_1)

	assert insert_step('job-test-insert-step', 99, args_1) is False
	assert insert_step('job-test-insert-step', 0, args_2) is True
	assert insert_step('job-test-insert-step', -1, args_3) is True

	steps = get_steps('job-test-insert-step')

	assert steps[0].get('args') == args_2
	assert steps[1].get('args') == args_1
	assert steps[2].get('args') == args_3
	assert steps[3].get('args') == args_1
	assert count_step_total('job-test-insert-step') == 4


def test_remove_step() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}
	args_3 =\
	{
		'source_path': 'source-3.jpg',
		'target_path': 'target-3.jpg',
		'output_path': 'output-3.jpg'
	}

	assert remove_step('job-invalid', 0) is False

	create_job('job-test-remove-step')
	add_step('job-test-remove-step', args_1)
	add_step('job-test-remove-step', args_2)
	add_step('job-test-remove-step', args_1)
	add_step('job-test-remove-step', args_3)

	assert remove_step('job-test-remove-step', 99) is False
	assert remove_step('job-test-remove-step', 0) is True
	assert remove_step('job-test-remove-step', -1) is True

	steps = get_steps('job-test-remove-step')

	assert steps[0].get('args') == args_2
	assert steps[1].get('args') == args_1
	assert count_step_total('job-test-remove-step') == 2


def test_get_steps() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}

	assert get_steps('job-invalid') == []

	create_job('job-test-get-steps')
	add_step('job-test-get-steps', args_1)
	add_step('job-test-get-steps', args_2)
	steps = get_steps('job-test-get-steps')

	assert steps[0].get('args') == args_1
	assert steps[1].get('args') == args_2
	assert count_step_total('job-test-get-steps') == 2


def test_set_step_status() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}

	assert set_step_status('job-invalid', 0, 'completed') is False

	create_job('job-test-set-step-status')
	add_step('job-test-set-step-status', args_1)
	add_step('job-test-set-step-status', args_2)

	assert set_step_status('job-test-set-step-status', 99, 'completed') is False
	assert set_step_status('job-test-set-step-status', 0, 'completed') is True
	assert set_step_status('job-test-set-step-status', 1, 'failed') is True

	steps = get_steps('job-test-set-step-status')

	assert steps[0].get('status') == 'completed'
	assert steps[1].get('status') == 'failed'
	assert count_step_total('job-test-set-step-status') == 2


def test_set_steps_status() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}

	assert set_steps_status('job-invalid', 'queued') is False

	create_job('job-test-set-steps-status')
	add_step('job-test-set-steps-status', args_1)
	add_step('job-test-set-steps-status', args_2)

	assert set_steps_status('job-test-set-steps-status', 'queued') is True

	steps = get_steps('job-test-set-steps-status')

	assert steps[0].get('status') == 'queued'
	assert steps[1].get('status') == 'queued'
	assert count_step_total('job-test-set-steps-status') == 2


def test_set_failed_steps_status() -> None:
	args_1 =\
	{
		'source_path': 'source-1.jpg',
		'target_path': 'target-1.jpg',
		'output_path': 'output-1.jpg'
	}
	args_2 =\
	{
		'source_path': 'source-2.jpg',
		'target_path': 'target-2.jpg',
		'output_path': 'output-2.jpg'
	}

	assert set_failed_steps_status('job-invalid', 'queued') is False

	create_job('job-test-set-failed-steps-status')
	add_step('job-test-set-failed-steps-status', args_1)
	add_step('job-test-set-failed-steps-status', args_2)
	set_step_status('job-test-set-failed-steps-status', 0, 'completed')
	set_step_status('job-test-set-failed-steps-status', 1, 'failed')

	assert set_failed_steps_status('job-test-set-failed-steps-status', 'queued') is True

	steps = get_steps('job-test-set-failed-steps-status')

	assert steps[0].get('status') == 'completed'
	assert steps[1].get('status') == 'queued'


def test_optimize_job() -> None:
	frame_total = count_video_frame_total(get_test_example_file('target-240p.mp4'))
	step_frame_total = frame_total // 3
	args =\
	{
		'source_path': 'source-1.jpg',
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': 'output-1.mp4'
	}

	assert optimize_job('job-invalid', step_frame_total) is False

	create_job('job-test-optimize-job')
	add_step('job-test-optimize-job', args)

	assert optimize_job('job-test-optimize-job', 0) is False
	assert optimize_job('job-test-optimize-job', step_frame_total) is True

	steps = get_steps('job-test-optimize-job')

	assert len(steps) == ceil(frame_total / step_frame_total)
	assert steps[0].get('args').get('trim_frame_start') == 0
	assert steps[0].get('status') == 'drafted'
	assert steps[-1].get('args').get('trim_frame_end') == frame_total


def test_optimize_job_respects_trim() -> None:
	args =\
	{
		'source_path': 'source-1.jpg',
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': 'output-1.mp4',
		'trim_frame_start': 50,
		'trim_frame_end': 200
	}

	create_job('job-test-optimize-job-respects-trim')
	add_step('job-test-optimize-job-respects-trim', args)

	assert optimize_job('job-test-optimize-job-respects-trim', 60) is True

	steps = get_steps('job-test-optimize-job-respects-trim')

	assert len(steps) == 3
	assert steps[0].get('args').get('trim_frame_start') == 50
	assert steps[0].get('args').get('trim_frame_end') == 110
	assert steps[1].get('args').get('trim_frame_start') == 110
	assert steps[2].get('args').get('trim_frame_end') == 200


def test_optimize_job_queued_step_status() -> None:
	frame_total = count_video_frame_total(get_test_example_file('target-240p.mp4'))
	args =\
	{
		'source_path': 'source-1.jpg',
		'target_path': get_test_example_file('target-240p.mp4'),
		'output_path': 'output-1.mp4'
	}

	create_job('job-test-optimize-job-queued')
	add_step('job-test-optimize-job-queued', args)
	submit_job('job-test-optimize-job-queued')

	assert optimize_job('job-test-optimize-job-queued', frame_total // 3) is True

	steps = get_steps('job-test-optimize-job-queued')

	assert len(steps) > 1

	for step in steps:
		assert step.get('status') == 'queued'
