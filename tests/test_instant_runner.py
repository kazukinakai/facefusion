from configparser import ConfigParser

import pytest

from facefusion import config, state_manager
from facefusion.uis.components import instant_runner


@pytest.fixture(scope = 'function', autouse = True)
def before_each() -> None:
	config.CONFIG_PARSER = ConfigParser()
	config.CONFIG_PARSER.read_dict(
	{
		'frame_extraction':
		{
			'step_frame_total': ''
		}
	})
	state_manager.init_item('resume', True)
	state_manager.init_item('processors', [ 'face_swapper' ])
	state_manager.init_item('trim_frame_start', None)
	state_manager.init_item('trim_frame_end', None)


def test_resolve_step_frame_total_resume_off() -> None:
	state_manager.init_item('resume', False)

	assert instant_runner.resolve_step_frame_total('target.mp4') == 0


def test_resolve_step_frame_total_audio_processor() -> None:
	state_manager.init_item('processors', [ 'lip_syncer' ])

	assert instant_runner.resolve_step_frame_total('target.mp4') == 0


def test_resolve_step_frame_total_explicit() -> None:
	config.CONFIG_PARSER.read_dict(
	{
		'frame_extraction':
		{
			'step_frame_total': '100'
		}
	})

	assert instant_runner.resolve_step_frame_total('target.mp4') == 100


def test_resolve_step_frame_total_short(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(instant_runner, 'count_video_frame_total', lambda target_path : 270)

	assert instant_runner.resolve_step_frame_total('target.mp4') == 0


def test_resolve_step_frame_total_long(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(instant_runner, 'count_video_frame_total', lambda target_path : 1000)

	assert instant_runner.resolve_step_frame_total('target.mp4') == 250


def test_resolve_step_frame_total_respects_trim(monkeypatch : pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(instant_runner, 'count_video_frame_total', lambda target_path : 4000)
	state_manager.init_item('trim_frame_start', 1000)
	state_manager.init_item('trim_frame_end', 3000)

	assert instant_runner.resolve_step_frame_total('target.mp4') == 500
