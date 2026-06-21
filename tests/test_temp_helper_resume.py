import os.path
import tempfile

import pytest

from facefusion import state_manager
from facefusion.temp_helper import create_processed_directory, force_clear_temp_directory, get_processed_directory_path, get_processed_frame_path, get_processed_frame_pattern, get_temp_directory_path, resolve_processed_frame_set


@pytest.fixture(scope = 'function', autouse = True)
def before_each() -> None:
	# Isolated temp root per test so the resume store never collides with a real
	# run or another test. No network — these helpers are pure filesystem.
	state_manager.init_item('temp_path', tempfile.mkdtemp())
	state_manager.init_item('temp_frame_format', 'png')


def test_get_processed_directory_path() -> None:
	target_path = '/videos/clip.mp4'
	assert get_processed_directory_path(target_path) == os.path.join(get_temp_directory_path(target_path), 'processed')


def test_get_processed_frame_pattern() -> None:
	target_path = '/videos/clip.mp4'
	assert get_processed_frame_pattern(target_path, '%08d') == os.path.join(get_processed_directory_path(target_path), '%08d.png')


def test_get_processed_frame_path_is_zero_padded() -> None:
	target_path = '/videos/clip.mp4'
	assert get_processed_frame_path(target_path, 7) == os.path.join(get_processed_directory_path(target_path), '00000007.png')
	# absolute frame numbers (trim-based) round-trip through the 8-digit name
	assert get_processed_frame_path(target_path, 12345678) == os.path.join(get_processed_directory_path(target_path), '12345678.png')


def test_create_and_resolve_processed_frame_set() -> None:
	target_path = '/videos/clip.mp4'
	assert create_processed_directory(target_path) is True
	assert resolve_processed_frame_set(target_path) == {}

	# Commit a few processed frames (out of order, sparse) — existence is the marker.
	for frame_number in (1, 2, 5):
		open(get_processed_frame_path(target_path, frame_number), 'w').close()

	processed_frame_set = resolve_processed_frame_set(target_path)
	assert set(processed_frame_set) == { 1, 2, 5 }
	assert processed_frame_set[5] == get_processed_frame_path(target_path, 5)


def test_force_clear_temp_directory_ignores_keep_temp() -> None:
	target_path = '/videos/clip.mp4'
	state_manager.init_item('keep_temp', True)
	create_processed_directory(target_path)
	open(get_processed_frame_path(target_path, 1), 'w').close()

	assert os.path.isdir(get_temp_directory_path(target_path))
	assert force_clear_temp_directory(target_path) is True
	# Removed despite keep_temp = True (success-time cleanup of the resume store).
	assert not os.path.exists(get_temp_directory_path(target_path))
