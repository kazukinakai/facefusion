import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy
import pytest
from pytest_mock import MockerFixture

from facefusion import frame_checkpoint, state_manager
from facefusion.types import Args, ErrorCode, JobId, StepIndex
from facefusion.workflows import to_video


@pytest.fixture(autouse = True)
def checkpoint_state(tmp_path : Path) -> Iterator[None]:
	state_manager.init()
	previous_state = state_manager.get_state().copy()
	target_path = tmp_path / 'target.mp4'
	target_path.write_bytes(b'target')
	state_manager.init_item('resume', True)
	state_manager.init_item('resume_path', str(tmp_path / 'checkpoints'))
	state_manager.init_item('workflow_mode', 'image-to-video')
	state_manager.init_item('workflow_strategy', 'memory')
	state_manager.init_item('target_path', str(target_path))
	state_manager.init_item('source_paths', [])
	state_manager.init_item('output_path', str(tmp_path / 'output.mp4'))
	state_manager.init_item('temp_path', str(tmp_path / 'temp'))
	state_manager.init_item('temp_frame_format', 'png')
	yield
	state_manager.set_state(previous_state)


@pytest.mark.parametrize('channels', [ 3, 4 ])
def test_round_trip_and_cleanup(channels : int) -> None:
	frame = numpy.arange(24 * 32 * channels, dtype = numpy.uint8).reshape(24, 32, channels)
	with frame_checkpoint.session():
		assert frame_checkpoint.write_frame(17, frame)
		assert numpy.array_equal(frame_checkpoint.read_frame(17), frame)
		assert frame_checkpoint.read_frame(18) is None
	with frame_checkpoint.session():
		assert numpy.array_equal(frame_checkpoint.read_frame(17), frame)
		assert frame_checkpoint.finish() == 0
	assert not frame_checkpoint.get_directory().exists()


def test_corrupt_or_uncommitted_frames_are_not_reused() -> None:
	frame = numpy.zeros((24, 32, 3), dtype = numpy.uint8)
	with frame_checkpoint.session():
		assert frame_checkpoint.write_frame(0, frame)
		path = frame_checkpoint.get_frame_path(0)
		path.write_bytes(path.read_bytes()[:-1] + b'!')
		assert frame_checkpoint.read_frame(0) is None
		assert frame_checkpoint.write_frame(0, frame)
		path.with_suffix('.sha256').unlink()
		assert frame_checkpoint.read_frame(0) is None


@pytest.mark.parametrize('change', [ 'target', 'source', 'setting' ])
def test_changes_invalidate_checkpoints(change : str, tmp_path : Path) -> None:
	source_path = tmp_path / 'source.png'
	source_path.write_bytes(b'source')
	state_manager.init_item('source_paths', [ str(source_path) ])
	with frame_checkpoint.session():
		assert frame_checkpoint.write_frame(0, numpy.zeros((24, 32, 3), dtype = numpy.uint8))
	if change == 'target':
		Path(state_manager.get_item('target_path')).write_bytes(b'change')
	if change == 'source':
		source_path.write_bytes(b'change')
	if change == 'setting':
		state_manager.init_item('output_video_scale', 0.5)
	with frame_checkpoint.session():
		assert frame_checkpoint.read_frame(0) is None


def test_disabled_mode_does_not_create_a_store() -> None:
	state_manager.init_item('resume', False)
	with frame_checkpoint.session():
		assert not frame_checkpoint.is_enabled()
	assert not frame_checkpoint.get_directory().exists()


def test_memory_processing_reuses_only_committed_results(mocker : MockerFixture) -> None:
	frame = numpy.arange(24 * 32 * 3, dtype = numpy.uint8).reshape(24, 32, 3)
	state_manager.init_item('temp_pixel_format', 'bgr24')
	mocker.patch.object(to_video, 'conditional_get_target_vision_frames', return_value = [ frame ])
	process = mocker.patch.object(to_video, 'process_temp_vision_frame', side_effect = lambda frames, current, index: 255 - current)
	with frame_checkpoint.session():
		first = to_video.process_memory_frame(0, (32, 24), (32, 24))
		second = to_video.process_memory_frame(1, (32, 24), (32, 24))
	assert process.call_count == 2
	with frame_checkpoint.session():
		assert numpy.array_equal(to_video.process_memory_frame(0, (32, 24), (32, 24)), first)
		assert numpy.array_equal(to_video.process_memory_frame(1, (32, 24), (32, 24)), second)
		to_video.process_memory_frame(2, (32, 24), (32, 24))
	assert process.call_count == 3


def test_failed_frame_commit_does_not_report_success(mocker : MockerFixture) -> None:
	frame = numpy.zeros((24, 32, 3), dtype = numpy.uint8)
	mocker.patch.object(to_video, 'conditional_get_target_vision_frames', return_value = [ frame ])
	mocker.patch.object(to_video, 'process_temp_vision_frame', return_value = frame)
	mocker.patch.object(frame_checkpoint, 'write_frame', return_value = False)
	with frame_checkpoint.session(), pytest.raises(OSError):
		to_video.process_memory_frame(0, (32, 24), (32, 24))


def test_checkpoint_survives_process_exit() -> None:
	configuration = repr(dict(state_manager.get_state()))
	code = '\n'.join(
	[
		'import os',
		'import numpy',
		'from facefusion import frame_checkpoint, state_manager',
		'state_manager.init()',
		'state_manager.set_state(' + configuration + ')',
		'with frame_checkpoint.session():',
		'\tassert frame_checkpoint.write_frame(9, numpy.full((24, 32, 3), 91, dtype = numpy.uint8))',
		'\tos._exit(7)'
	])
	assert subprocess.run([ sys.executable, '-c', code ], check = False).returncode == 7
	with frame_checkpoint.session():
		assert numpy.array_equal(frame_checkpoint.read_frame(9), numpy.full((24, 32, 3), 91, dtype = numpy.uint8))


def test_concurrent_process_cannot_share_the_same_checkpoint_store() -> None:
	configuration = repr(dict(state_manager.get_state()))
	code = '\n'.join(
	[
		'from facefusion import frame_checkpoint, state_manager',
		'state_manager.init()',
		'state_manager.set_state(' + configuration + ')',
		'try:',
		'\twith frame_checkpoint.session():',
		'\t\tpass',
		'except OSError:',
		'\traise SystemExit(7)'
	])
	with frame_checkpoint.session():
		assert subprocess.run([ sys.executable, '-c', code ], check = False).returncode == 7
	assert subprocess.run([ sys.executable, '-c', code ], check = False).returncode == 0


@pytest.mark.parametrize('strategy', [ 'memory', 'disk' ])
@pytest.mark.parametrize('frame_format', [ 'png', 'jpeg' ])
def test_resumed_video_matches_uninterrupted_video(strategy : str, frame_format : str, tmp_path : Path, mocker : MockerFixture) -> None:
	from facefusion import process_manager, video_manager
	from facefusion.workflows import core

	target_path = tmp_path / 'synthetic.mp4'
	subprocess.run(
	[
		'ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=10',
		'-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '1',
		'-c:v', 'libx264', '-c:a', 'aac', str(target_path)
	], check = True)
	settings =\
	{
		'workflow_strategy': strategy,
		'target_path': str(target_path),
		'trim_frame_start': 0,
		'trim_frame_end': 10,
		'target_frame_amount': 0,
		'output_video_scale': 1.0,
		'output_video_fps': 10.0,
		'output_video_encoder': 'libx264',
		'output_video_quality': 80,
		'output_video_preset': 'veryfast',
		'temp_pixel_format': 'bgr24',
		'temp_frame_format': frame_format,
		'processors': [],
		'output_audio_encoder': 'aac',
		'output_audio_quality': 80,
		'output_audio_volume': 100,
		'execution_thread_count': 1,
		'video_memory_strategy': 'tolerant'
	}
	for key, value in settings.items():
		state_manager.init_item(key, value)
	video_manager.init()
	process_manager.init()
	process_manager.start()
	baseline_path = tmp_path / 'baseline.mp4'
	resumed_path = tmp_path / 'resumed.mp4'
	state_manager.init_item('resume', True)
	state_manager.init_item('output_path', str(baseline_path))
	with frame_checkpoint.session():
		core.setup()
		if strategy == 'memory':
			assert to_video.process_memory_frames() == 0
		else:
			assert to_video.create_temp_frames() == 0
			assert core.process_frames() == 0
			assert to_video.merge_frames() == 0
		assert to_video.restore_audio() == 0
		assert frame_checkpoint.finish() == 0
	video_manager.clear()
	state_manager.init_item('resume', True)
	state_manager.init_item('output_path', str(resumed_path))
	with frame_checkpoint.session():
		core.setup()
		if strategy == 'memory':
			for index in (0, 3, 7):
				to_video.process_memory_frame(index, (64, 48), (64, 48))
		else:
			assert to_video.create_temp_frames() == 0
			for index in (0, 3, 7):
				assert core.process_temp_frame(str(Path(core.state_manager.get_temp_path()) / 'facefusion' / 'resumed' / (format(index, '08d') + '.' + frame_format)), index)
	with frame_checkpoint.session():
		core.setup()
		if strategy == 'memory':
			process = mocker.spy(to_video, 'process_temp_vision_frame')
			assert to_video.process_memory_frames() == 0
			assert process.call_count == 7
		else:
			assert to_video.create_temp_frames() == 0
			assert core.process_frames() == 0
			assert to_video.merge_frames() == 0
		assert to_video.restore_audio() == 0
		assert frame_checkpoint.finish() == 0
	video_manager.clear()
	for options in ([ '-map', '0:v:0', '-pix_fmt', 'bgr24', '-f', 'rawvideo' ], [ '-map', '0:a:0', '-f', 's16le' ]):
		baseline = subprocess.run([ 'ffmpeg', '-nostdin', '-v', 'error', '-i', str(baseline_path), *options, '-' ], capture_output = True, check = True).stdout
		resumed = subprocess.run([ 'ffmpeg', '-nostdin', '-v', 'error', '-i', str(resumed_path), *options, '-' ], capture_output = True, check = True).stdout
		assert baseline and baseline == resumed


@pytest.mark.parametrize('change', [ 'code', 'model' ])
def test_code_and_model_changes_invalidate_checkpoints(change : str, tmp_path : Path, mocker : MockerFixture) -> None:
	package_path = tmp_path / 'package' / 'facefusion'
	package_path.mkdir(parents = True)
	code_path = package_path / 'frame_checkpoint.py'
	code_path.write_bytes(b'version one')
	model_path = package_path.parent / '.assets' / 'models' / 'model.onnx'
	model_path.parent.mkdir(parents = True)
	model_path.write_bytes(b'weights one')
	mocker.patch.object(frame_checkpoint, '__file__', str(code_path))
	with frame_checkpoint.session():
		assert frame_checkpoint.write_frame(0, numpy.zeros((24, 32, 3), dtype = numpy.uint8))
	if change == 'code':
		code_path.write_bytes(b'version two')
	else:
		model_path.write_bytes(b'weights two')
	with frame_checkpoint.session():
		assert frame_checkpoint.read_frame(0) is None


def test_cli_flags_reach_processing_state(mocker : MockerFixture) -> None:
	from facefusion import args_helper
	from facefusion.program import create_program

	mocker.patch.object(args_helper, 'get_processors_modules', return_value = [])
	program = create_program()
	args = vars(program.parse_args([ 'run', '--resume', '--resume-path', '/checkpoint-root' ]))
	args_helper.apply_args(args, state_manager.set_item)
	assert state_manager.get_item('resume') is True
	assert state_manager.get_item('resume_path') == '/checkpoint-root'


def test_completed_job_cleanup_preserves_another_job_store(mocker : MockerFixture) -> None:
	from facefusion.jobs import job_manager

	mocker.patch.object(job_manager, 'get_steps', return_value = [ { 'args': dict(state_manager.get_state()) } ])
	with frame_checkpoint.job('current-job'), frame_checkpoint.session():
		assert frame_checkpoint.write_frame(0, numpy.zeros((24, 32, 3), dtype = numpy.uint8))
	frame_checkpoint.finish_job('previous-job')
	assert frame_checkpoint.get_directory().is_dir()
	frame_checkpoint.finish_job('current-job')
	assert not frame_checkpoint.get_directory().exists()


@pytest.mark.parametrize('frame_format', [ 'bmp', 'jpeg', 'png', 'tiff' ])
def test_checkpoint_uses_selected_frame_format(frame_format : str) -> None:
	state_manager.init_item('temp_frame_format', frame_format)
	with frame_checkpoint.session():
		assert frame_checkpoint.write_frame(5, numpy.full((24, 32, 3), 91, dtype = numpy.uint8))
		path = frame_checkpoint.get_frame_path(5)
		assert path.suffix == '.' + frame_format
		assert path.is_file()
		assert numpy.array_equal(frame_checkpoint.read_frame(5), numpy.full((24, 32, 3), 91, dtype = numpy.uint8))


def test_abrupt_exit_reuses_persistent_frames_with_new_ephemeral_root(tmp_path : Path) -> None:
	configuration = repr(dict(state_manager.get_state()))
	code = '\n'.join(
	[
		'import numpy',
		'from facefusion import frame_checkpoint, state_manager',
		'state_manager.init()',
		'state_manager.set_state(' + configuration + ')',
		'with frame_checkpoint.session():',
		'\tassert frame_checkpoint.write_frame(9, numpy.full((24, 32, 3), 91, dtype = numpy.uint8))',
		'\tpending = frame_checkpoint.get_directory().parent / "scratch" / "old-pod-partial.mp4"',
		'\tpending.parent.mkdir(parents = True, exist_ok = True)',
		'\tpending.write_bytes(b"incomplete")',
		'\tprint("committed", flush = True)',
		'\tinput()'
	])
	process = subprocess.Popen([ sys.executable, '-c', code ], stdin = subprocess.PIPE, stdout = subprocess.PIPE, text = True)
	try:
		assert process.stdout.readline().strip() == 'committed'
	finally:
		process.kill()
		process.communicate(timeout = 10)
	assert process.returncode != 0
	path = frame_checkpoint.get_frame_path(9)
	content = path.read_bytes()
	modified = path.stat().st_mtime_ns
	pending = path.parent.parent / 'scratch' / 'old-pod-partial.mp4'
	assert pending.is_file()
	state_manager.init_item('temp_path', str(tmp_path / 'new-pod-ephemeral'))
	with frame_checkpoint.session():
		assert numpy.array_equal(frame_checkpoint.read_frame(9), numpy.full((24, 32, 3), 91, dtype = numpy.uint8))
		assert path.read_bytes() == content
		assert path.stat().st_mtime_ns == modified
		assert not pending.exists()
		assert frame_checkpoint.finish() == 0
	assert not path.exists()


@pytest.mark.parametrize('invalidated', [ False, True ])
def test_completed_steps_survive_failure_and_are_cleaned_after_retry(tmp_path : Path, invalidated : bool) -> None:
	from facefusion.jobs import job_helper, job_manager, job_runner
	from facefusion.vision import write_image

	state_manager.init_item('jobs_path', str(tmp_path / 'jobs'))
	assert job_manager.init_jobs(state_manager.get_jobs_path())
	assert job_manager.create_job('resumable')
	for index in range(2):
		args = dict(state_manager.get_state())
		args['output_path'] = str(tmp_path / (str(index) + '.png'))
		assert job_manager.add_step('resumable', args)
	assert job_manager.submit_job('resumable')
	calls = []

	def process(job_id : JobId, step_index : StepIndex, args : Args) -> ErrorCode:
		calls.append(step_index)
		if calls == [ 0, 1 ]:
			return 1
		return 0 if write_image(args['output_path'], numpy.zeros((24, 32, 3), dtype = numpy.uint8)) else 1

	assert not job_runner.run_job('resumable', process)
	args = job_manager.get_steps('resumable')[0]['args']
	step_output = Path(job_helper.get_step_output_path('resumable', 0, str(args['output_path'])))
	assert step_output.is_file()
	assert Path(str(step_output) + '.resume.json').is_file()
	if invalidated:
		assert write_image(str(step_output), numpy.full((24, 32, 3), 255, dtype = numpy.uint8))
	assert job_runner.retry_job('resumable', process)
	assert calls == ([ 0, 1, 0, 1 ] if invalidated else [ 0, 1, 1 ])
	assert not step_output.exists()
	assert not Path(str(step_output) + '.resume.json').exists()
	assert Path(str(args['output_path'])).is_file()


def test_delete_job_cleans_checkpoints_without_deleting_inputs(tmp_path : Path) -> None:
	from facefusion.jobs import job_manager

	state_manager.init_item('jobs_path', str(tmp_path / 'jobs'))
	assert job_manager.init_jobs(state_manager.get_jobs_path())
	assert job_manager.create_job('discard')
	assert job_manager.add_step('discard', dict(state_manager.get_state()))
	with frame_checkpoint.job('discard'), frame_checkpoint.session():
		assert frame_checkpoint.write_frame(0, numpy.zeros((24, 32, 3), dtype = numpy.uint8))
		assert not job_manager.delete_job('discard')
		assert job_manager.read_job_file('discard') is not None
	assert job_manager.delete_job('discard')
	assert not frame_checkpoint.get_directory().exists()
	assert Path(state_manager.get_item('target_path')).is_file()


def test_fingerprint_detects_same_size_change_with_restored_modification_time(tmp_path : Path) -> None:
	import os

	path = tmp_path / 'model.onnx'
	path.write_bytes(b'first')
	status = path.stat()
	original = frame_checkpoint.hash_file(path)
	path.write_bytes(b'other')
	os.utime(path, ns = (status.st_atime_ns, status.st_mtime_ns))
	assert frame_checkpoint.hash_file(path) != original


@pytest.mark.parametrize('api_owner', [ False, True ])
def test_default_checkpoint_root_handles_pod_identity_without_merging_api_owners(api_owner : bool, tmp_path : Path, mocker : MockerFixture) -> None:
	state_manager.init_item('resume_path', None)
	local = mocker.patch.object(frame_checkpoint, 'resolve_local_id', return_value = 'pod-one')
	temp = mocker.patch.object(state_manager, 'get_temp_path', return_value = str(tmp_path / ('user-one' if api_owner else 'pod-one')))
	first = frame_checkpoint.get_directory()
	local.return_value = 'pod-two'
	temp.return_value = str(tmp_path / ('user-two' if api_owner else 'pod-two'))
	second = frame_checkpoint.get_directory()
	assert (first != second) if api_owner else (first == second)


def test_completion_marker_failure_keeps_encoded_steps_for_retry(tmp_path : Path, mocker : MockerFixture) -> None:
	import shutil

	from facefusion.jobs import job_helper, job_manager, job_runner

	target = tmp_path / 'video.mp4'
	subprocess.run([ 'ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=32x24:rate=10', '-t', '0.2', '-c:v', 'libx264', str(target) ], check = True)
	state_manager.init_item('target_path', str(target))
	state_manager.init_item('jobs_path', str(tmp_path / 'jobs'))
	assert job_manager.init_jobs(state_manager.get_jobs_path())
	assert job_manager.create_job('completion-marker')
	args = dict(state_manager.get_state())
	args['output_path'] = str(tmp_path / 'output.mp4')
	assert job_manager.add_step('completion-marker', args)
	assert job_manager.submit_job('completion-marker')
	calls = []

	def process(job_id : JobId, step_index : StepIndex, step_args : Args) -> ErrorCode:
		calls.append(step_index)
		shutil.copy2(target, step_args['output_path'])
		return 0

	move = job_manager.move_job_file
	marker = mocker.patch.object(job_manager, 'move_job_file', side_effect = lambda job_id, status: False if status == 'completed' else move(job_id, status))
	assert not job_runner.run_job('completion-marker', process)
	step_output = Path(job_helper.get_step_output_path('completion-marker', 0, str(args['output_path'])))
	assert step_output.is_file()
	assert Path(str(step_output) + '.resume.json').is_file()
	marker.side_effect = move
	assert job_runner.run_job('completion-marker', process)
	assert calls == [ 0 ]
	assert not step_output.exists()
	assert not Path(str(step_output) + '.resume.json').exists()
	assert Path(str(args['output_path'])).is_file()
