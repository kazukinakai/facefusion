import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy
import pytest
from pytest_mock import MockerFixture

from facefusion import frame_checkpoint, state_manager
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
		path = frame_checkpoint.get_directory() / '00000000.png'
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
def test_resumed_video_matches_uninterrupted_video(strategy : str, tmp_path : Path, mocker : MockerFixture) -> None:
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
		'temp_frame_format': 'png',
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
	state_manager.init_item('resume', False)
	state_manager.init_item('output_path', str(baseline_path))
	core.setup()
	if strategy == 'memory':
		assert to_video.process_memory_frames() == 0
	else:
		assert to_video.create_temp_frames() == 0
		assert core.process_frames() == 0
		assert to_video.merge_frames() == 0
	assert to_video.restore_audio() == 0
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
				assert core.process_temp_frame(str(Path(core.state_manager.get_temp_path()) / 'facefusion' / 'resumed' / (format(index, '08d') + '.png')), index)
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
