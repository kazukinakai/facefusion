import os.path
import tempfile

from facefusion.temp_helper import create_processed_directory, get_processed_directory_path, get_processed_frames_pattern, get_temp_directory_path, resolve_processed_frame_paths


def test_get_processed_directory_path() -> None:
	temp_path = tempfile.mkdtemp()
	output_path = '/videos/clip__faceswap.mp4'
	assert get_processed_directory_path(temp_path, output_path) == os.path.join(get_temp_directory_path(temp_path, output_path), 'processed')


def test_get_processed_frames_pattern() -> None:
	temp_path = tempfile.mkdtemp()
	output_path = '/videos/clip__faceswap.mp4'
	assert get_processed_frames_pattern(temp_path, output_path, 'png', '%08d') == os.path.join(get_processed_directory_path(temp_path, output_path), '%08d.png')


def test_create_and_resolve_processed_frames() -> None:
	temp_path = tempfile.mkdtemp()
	output_path = '/videos/clip__faceswap.mp4'
	assert create_processed_directory(temp_path, output_path) is True
	assert resolve_processed_frame_paths(temp_path, output_path, 'png') == []

	# Commit a few processed frames (sparse, out of order) — existence is the marker.
	for frame_number in (1, 2, 5):
		open(os.path.join(get_processed_directory_path(temp_path, output_path), format(frame_number, '08d') + '.png'), 'w').close()

	processed_frame_paths = resolve_processed_frame_paths(temp_path, output_path, 'png')
	assert len(processed_frame_paths) == 3
	assert sorted(os.path.basename(path) for path in processed_frame_paths) == [ '00000001.png', '00000002.png', '00000005.png' ]
