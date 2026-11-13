import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy

from facefusion import logger, state_manager
from facefusion.temp_helper import get_temp_directory_path
from facefusion.types import ErrorCode, JobId, VisionFrame

JOB_ID : ContextVar[Optional[JobId]] = ContextVar('CHECKPOINT_JOB_ID', default = None)


@contextmanager
def job(job_id : JobId) -> Iterator[None]:
	token = JOB_ID.set(job_id)
	try:
		yield
	finally:
		JOB_ID.reset(token)


def is_enabled() -> bool:
	return bool(state_manager.get_item('resume')) and state_manager.get_item('workflow_mode') == 'image-to-video'


def get_directory() -> Path:
	temp_path = state_manager.get_item('resume_path') or state_manager.get_temp_path()
	return Path(get_temp_directory_path(temp_path, state_manager.get_item('output_path'))) / 'processed'


def hash_file(path : Path) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(block)
	return digest.hexdigest()


def create_signature() -> str:
	excluded = { 'command', 'config_path', 'jobs_path', 'temp_path', 'output_path', 'target_path', 'source_paths', 'resume', 'log_level', 'job_id', 'resume_path' }
	settings = { key: value for key, value in state_manager.get_state().items() if key not in excluded }
	digest = hashlib.sha256(json.dumps(settings, sort_keys = True, allow_nan = False).encode())
	packages = sorted((package.metadata['Name'], package.version) for package in importlib.metadata.distributions())
	digest.update(json.dumps(packages).encode())
	for path in [ state_manager.get_item('target_path'), *(state_manager.get_item('source_paths') or []) ]:
		digest.update(hash_file(Path(path)).encode())
	package_path = Path(__file__).parent
	for path in sorted(package_path.rglob('*.py')):
		digest.update(str(path.relative_to(package_path)).encode())
		digest.update(hash_file(path).encode())
	models_path = package_path.parent / '.assets' / 'models'
	for path in sorted(models_path.rglob('*')):
		if path.is_file():
			digest.update(str(path.relative_to(models_path)).encode())
			digest.update(hash_file(path).encode())
	return digest.hexdigest()


def write_atomic(path : Path, content : bytes) -> None:
	descriptor, temporary_path = tempfile.mkstemp(dir = path.parent, prefix = '.' + path.name + '.')
	try:
		with os.fdopen(descriptor, 'wb') as handle:
			handle.write(content)
			handle.flush()
			os.fsync(handle.fileno())
		os.replace(temporary_path, path)
	finally:
		if os.path.exists(temporary_path):
			os.remove(temporary_path)


@contextmanager
def lock(directory : Path) -> Iterator[None]:
	directory.parent.parent.mkdir(parents = True, exist_ok = True)
	lock_path = directory.parent.with_name(directory.parent.name + '.resume.lock')
	with lock_path.open('a+b') as handle:
		if sys.platform == 'win32':
			import msvcrt
			handle.seek(0)
			handle.write(b'\0')
			handle.flush()
			handle.seek(0)
			msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
		else:
			import fcntl
			fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
		try:
			yield
		finally:
			if sys.platform == 'win32':
				handle.seek(0)
				msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
			else:
				fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def session() -> Iterator[None]:
	if not is_enabled():
		yield
		return
	directory = get_directory()
	with lock(directory):
		signature = create_signature()
		manifest_path = directory / 'signature'
		previous_signature = manifest_path.read_text(errors = 'replace') if manifest_path.is_file() else None
		if previous_signature != signature and directory.parent.exists():
			shutil.rmtree(directory.parent)
		directory.mkdir(parents = True, exist_ok = True)
		write_atomic(manifest_path, signature.encode())
		if JOB_ID.get():
			write_atomic(directory / 'owner', str(JOB_ID.get()).encode())
		yield


def read_frame(frame_index : int) -> Optional[VisionFrame]:
	path = get_directory() / (format(frame_index, '08d') + '.png')
	try:
		content = path.read_bytes()
		if hashlib.sha256(content).hexdigest() != path.with_suffix('.sha256').read_text():
			return None
		frame = cv2.imdecode(numpy.frombuffer(content, dtype = numpy.uint8), cv2.IMREAD_UNCHANGED)
		if frame is not None and frame.dtype == numpy.uint8 and frame.ndim == 3 and frame.shape[2] in (3, 4):
			return frame
	except (OSError, ValueError, cv2.error):
		return None
	return None


def write_frame(frame_index : int, frame : VisionFrame) -> bool:
	if frame.dtype != numpy.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
		return False
	path = get_directory() / (format(frame_index, '08d') + '.png')
	try:
		encoded, buffer = cv2.imencode('.png', frame, [ cv2.IMWRITE_PNG_COMPRESSION, 1 ])
		if not encoded:
			return False
		content = buffer.tobytes()
		write_atomic(path, content)
		write_atomic(path.with_suffix('.sha256'), hashlib.sha256(content).hexdigest().encode())
		return True
	except (OSError, cv2.error):
		return False


def finish() -> ErrorCode:
	if is_enabled():
		directory = get_directory().parent
		shutil.rmtree(directory)
		temp_directory = Path(get_temp_directory_path(state_manager.get_temp_path(), state_manager.get_item('output_path')))
		if temp_directory != directory and temp_directory.exists():
			shutil.rmtree(temp_directory)
	return 0


def finish_job(job_id : JobId) -> None:
	from facefusion.jobs import job_manager

	for step in job_manager.get_steps(job_id):
		args = step.get('args')
		if args.get('resume'):
			directory = Path(get_temp_directory_path(args.get('resume_path') or state_manager.get_temp_path(), args.get('output_path'))) / 'processed'
			if (directory / 'signature').is_file():
				try:
					with lock(directory):
						owner_path = directory / 'owner'
						if owner_path.is_file() and owner_path.read_text() == job_id:
							shutil.rmtree(directory.parent)
							temp_directory = Path(get_temp_directory_path(state_manager.get_temp_path(), args.get('output_path')))
							if temp_directory != directory.parent and temp_directory.exists():
								shutil.rmtree(temp_directory)
				except (OSError, UnicodeError) as error:
					logger.warn(str(error), __name__)
