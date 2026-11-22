import subprocess
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Optional
from unittest.mock import DEFAULT, patch

import pytest

from facefusion import capability_store, config, content_analyser, core, ffmpeg, process_manager, state_manager, vision
from facefusion.jobs import job_helper, job_manager, job_runner
from facefusion.program import create_jobs_path_program, create_temp_path_program, create_workflow_program
from facefusion.types import Args
from facefusion.workflows import core as workflow_core


@pytest.mark.parametrize('failure', [ 'inference', 'merge', 'damaged-frame', 'damaged-raw', 'extraction', 'concat' ])
def test_video_job_resumes_processed_frames(failure : str, tmp_path : Path, monkeypatch : pytest.MonkeyPatch) -> None:
	state_manager.init()
	process_manager.init()
	state_manager.init_item('jobs_path', str(tmp_path / 'jobs'))
	state_manager.init_item('temp_path', str(tmp_path / 'temp'))
	monkeypatch.setattr(capability_store, 'CAPABILITY_STORE', { 'api': {}, 'cli': {}, 'sys': {} })
	monkeypatch.setattr(config, 'get_static_config_parser', ConfigParser)
	create_jobs_path_program()
	create_temp_path_program()
	workflow_program = create_workflow_program()

	frame_size = 16 * 16 * 3
	frame_levels = [ 32, 80, 128, 176 ]
	input_bytes = b''.join(bytes([ level ]) * frame_size for level in frame_levels)
	input_path = tmp_path / 'input.mkv'
	output_path = tmp_path / 'resumed.mp4'
	subprocess.run([ 'ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '16x16', '-framerate', '4', '-i', 'pipe:0', '-c:v', 'ffv1', str(input_path) ], input = input_bytes, check = True)

	step_args : Args = vars(workflow_program.parse_args([ '--resume', '--workflow-mode', 'image-to-video' ]))
	assert step_args.get('workflow_strategy') == 'memory'
	step_args.update(
	{
		'target_path': str(input_path),
		'output_path': str(output_path),
		'output_video_fps': 4.0,
		'output_video_scale': 1.0,
		'output_video_encoder': 'libx264',
		'output_video_preset': 'ultrafast',
		'output_video_quality': 100,
		'output_audio_volume': 0,
		'temp_frame_format': 'png',
		'temp_pixel_format': 'bgr24',
		'execution_thread_count': 1,
		'processors': []
	})
	assert job_manager.init_jobs(state_manager.get_jobs_path())
	job_id = 'video-resume'
	assert job_manager.create_job(job_id)
	assert job_manager.add_step(job_id, step_args)
	assert job_manager.submit_job(job_id)

	calls = []
	processed_frames = []
	fail_inference = failure in [ 'inference', 'damaged-raw' ]

	def process_temp_frame(frame_path : str, frame_index : int, output_frame_path : Optional[str] = None) -> bool:
		calls.append(frame_index)
		if fail_inference and frame_index > 0:
			raise RuntimeError('inference interrupted')
		frame = vision.read_image(frame_path)
		assert frame is not None
		assert abs(int(frame[0, 0, 0]) - frame_levels[frame_index]) <= 1
		assert vision.write_image(output_frame_path or frame_path, frame + 20)
		processed_frames.append(frame_index)
		return True

	native_extract = ffmpeg.extract_frames

	def interrupt_extraction(*args : Any) -> bool:
		assert native_extract(*args)
		return False

	monkeypatch.setattr(workflow_core, 'process_temp_frame', process_temp_frame)
	monkeypatch.setattr(content_analyser, 'analyse_video', lambda *_: False)
	monkeypatch.setattr(core, 'common_pre_check', lambda: True)
	with patch.object(ffmpeg, 'merge_video', wraps = ffmpeg.merge_video, return_value = DEFAULT if fail_inference or failure == 'concat' else False) as merge_video, patch.object(job_runner, 'concat_video', wraps = job_runner.concat_video, return_value = False if failure == 'concat' else DEFAULT) as concat_video:
		with patch.object(ffmpeg, 'extract_frames', wraps = interrupt_extraction if failure == 'extraction' else native_extract):
			assert not job_runner.run_job(job_id, core.process_step)
		assert job_manager.find_job_ids('failed') == [ job_id ]
		if failure == 'concat':
			assert len(list(Path(job_helper.get_step_frame_directory(job_id, 0)).glob('*/raw/*.png'))) == 4
		else:
			assert not output_path.exists()
		assert processed_frames == ([] if failure == 'extraction' else [ 0 ] if fail_inference else [ 0, 1, 2, 3 ])
		assert calls == list(range(len(calls)))
		assert len(calls) >= (0 if failure == 'extraction' else 2)
		assert merge_video.call_count == (0 if fail_inference or failure == 'extraction' else 1)

		fail_inference = False
		state_manager.set_item('temp_path', str(tmp_path / 'new-process-temp'))
		calls.clear()
		merge_video.return_value = DEFAULT
		concat_video.return_value = DEFAULT
		if failure == 'damaged-raw':
			frame_directory = Path(job_helper.get_step_frame_directory(job_id, 0))
			sorted(frame_directory.glob('*/raw/*.png'))[1].write_bytes(b'incomplete image')
		if failure == 'damaged-frame':
			frame_directory = Path(job_helper.get_step_frame_directory(job_id, 0))
			retained_images = sorted(frame_directory.glob('*/*.png'))
			assert len(retained_images) == 4
			retained_images[2].write_bytes(b'interrupted image')
		with patch.object(ffmpeg, 'extract_frames', wraps = ffmpeg.extract_frames) as extract_frames:
			assert job_runner.retry_job(job_id, core.process_step)
			assert extract_frames.call_count == int(failure in [ 'extraction', 'damaged-raw' ])
		assert job_manager.find_job_ids('completed') == [ job_id ]
		assert calls == { 'inference': [ 1, 2, 3 ], 'damaged-frame': [ 2 ], 'merge': [], 'concat': [], 'damaged-raw': [ 1, 2, 3 ], 'extraction': [ 0, 1, 2, 3 ] }[failure]
		assert processed_frames == ([ 0, 1, 2, 3, 2 ] if failure == 'damaged-frame' else [ 0, 1, 2, 3 ])
		assert merge_video.call_count == (2 if failure in [ 'merge', 'damaged-frame' ] else 1)

	assert not Path(job_helper.get_step_frame_directory(job_id, 0)).exists()
	decoded_bytes = subprocess.check_output([ 'ffmpeg', '-v', 'error', '-i', str(output_path), '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1' ])
	assert len(decoded_bytes) == frame_size * len(frame_levels)
	for frame_index, level in enumerate(frame_levels):
		pixels = decoded_bytes[frame_index * frame_size:(frame_index + 1) * frame_size]
		assert min(pixels) >= level + 18
		assert max(pixels) <= level + 22
