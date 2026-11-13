import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, cast

import cv2
import numpy

from facefusion import logger, state_manager
from facefusion.session_context import resolve_local_id
from facefusion.types import Args, ErrorCode, JobId, State, VisionFrame

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


def get_root() -> Path:
	resume_path = state_manager.get_item('resume_path')
	if resume_path:
		return Path(resume_path)
	temp_path = Path(state_manager.get_temp_path())
	if temp_path.name == resolve_local_id():
		return temp_path.parent
	return temp_path


def get_directory() -> Path:
	identity = str(Path(state_manager.get_item('output_path')).resolve())
	name = hashlib.sha256(identity.encode()).hexdigest()
	return get_root() / 'facefusion-resume' / name / 'processed'


def hash_file(path : Path) -> str:
	path = path.resolve()
	status = path.stat()
	identity = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
	return hash_file_version(path, identity)


@lru_cache(maxsize = 1024)
def hash_file_version(path : Path, identity : Tuple[int, ...]) -> str:
	digest = hashlib.sha256()
	with path.open('rb') as handle:
		for block in iter(lambda: handle.read(1024 * 1024), b''):
			digest.update(block)
	status = path.stat()
	if identity != (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns):
		raise OSError('Checkpoint input changed while fingerprinting')
	return digest.hexdigest()


def create_signature(args : Optional[Args] = None) -> str:
	state = args if args is not None else state_manager.get_state()
	excluded = { 'command', 'config_path', 'jobs_path', 'temp_path', 'output_path', 'target_path', 'source_paths', 'resume', 'log_level', 'job_id', 'resume_path' }
	settings = { key: value for key, value in state.items() if key not in excluded }
	digest = hashlib.sha256(json.dumps(settings, sort_keys = True, allow_nan = False).encode())
	packages = sorted((package.metadata['Name'], package.version) for package in importlib.metadata.distributions())
	digest.update(json.dumps(packages).encode())
	for path in [ state.get('target_path'), *cast(List[str], state.get('source_paths') or []) ]:
		digest.update(hash_file(Path(cast(str, path))).encode())
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
def job_session(job_id : JobId) -> Iterator[None]:
	from facefusion.jobs import job_manager

	directories = set()
	for step in job_manager.get_steps(job_id):
		args = step.get('args')
		if args.get('resume'):
			root = Path(args.get('resume_path')) if args.get('resume_path') else get_root()
			identity = hashlib.sha256(job_id.encode()).hexdigest()
			directories.add(root / 'facefusion-resume' / 'jobs' / identity / 'processed')
	with ExitStack() as stack:
		for directory in sorted(directories):
			stack.enter_context(lock(directory))
		yield


@contextmanager
def session() -> Iterator[None]:
	if not is_enabled():
		yield
		return
	directory = get_directory()
	root = get_root()
	with lock(directory):
		signature = create_signature()
		manifest_path = directory / 'signature'
		previous_signature = manifest_path.read_text(errors = 'replace') if manifest_path.is_file() else None
		if previous_signature != signature and directory.parent.exists():
			shutil.rmtree(directory.parent)
		directory.mkdir(parents = True, exist_ok = True)
		for path in directory.glob('.*'):
			if path.is_file():
				path.unlink()
		write_atomic(manifest_path, signature.encode())
		if JOB_ID.get():
			write_atomic(directory / 'owner', str(JOB_ID.get()).encode())
		scratch = directory.parent / 'scratch'
		if scratch.exists():
			shutil.rmtree(scratch)
		previous_temp_path = state_manager.get_item('temp_path')
		previous_resume_path = state_manager.get_item('resume_path')
		state_manager.set_item('resume_path', str(root))
		state_manager.set_item('temp_path', str(scratch))
		try:
			yield
		finally:
			state_manager.set_item('temp_path', previous_temp_path)
			state_manager.set_item('resume_path', previous_resume_path)


def get_frame_path(frame_index : int) -> Path:
	frame_format = state_manager.get_item('temp_frame_format')
	return get_directory() / (format(frame_index, '08d') + '.' + frame_format)


def read_frame(frame_index : int) -> Optional[VisionFrame]:
	path = get_frame_path(frame_index)
	try:
		content = path.read_bytes()
		if hashlib.sha256(content).hexdigest() != path.with_suffix('.sha256').read_text():
			return None
		frame = cv2.imdecode(numpy.frombuffer(content, dtype = numpy.uint8), cv2.IMREAD_UNCHANGED)
		if frame is not None and frame.dtype == numpy.uint8 and frame.ndim == 3 and frame.shape[2] in (3, 4):
			if state_manager.get_item('temp_pixel_format') == 'bgra' and frame.shape[2] == 3:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
			return frame
	except (OSError, ValueError, cv2.error):
		return None
	return None


def write_frame(frame_index : int, frame : VisionFrame) -> bool:
	if frame.dtype != numpy.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
		return False
	path = get_frame_path(frame_index)
	if path.suffix in ('.jpeg', '.bmp') and frame.shape[2] == 4 and numpy.any(frame[:, :, 3] != 255):
		logger.error('Selected temporary frame format cannot preserve transparency: ' + path.suffix, __name__)
		return False
	try:
		encoded, buffer = cv2.imencode(path.suffix, frame)
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
		shutil.rmtree(get_directory().parent)
	return 0


def finish_job(job_id : JobId) -> bool:
	from facefusion.jobs import job_manager

	for step in job_manager.get_steps(job_id):
		args = step.get('args')
		if args.get('resume'):
			previous_state = state_manager.get_state().copy()
			try:
				state_manager.set_state(cast(State, { **previous_state, **args }))
				directory = get_directory()
				if (directory / 'signature').is_file():
					with lock(directory):
						owner_path = directory / 'owner'
						if owner_path.is_file() and owner_path.read_text() == job_id:
							shutil.rmtree(directory.parent)
			except (OSError, UnicodeError) as error:
				logger.warn(str(error), __name__)
				return False
			finally:
				state_manager.set_state(previous_state)
	return True


def commit_step(args : Args, output_path : str) -> bool:
	try:
		path = Path(output_path)
		content = { 'signature': create_signature(args), 'output': hash_file(path) }
		write_atomic(Path(output_path + '.resume.json'), json.dumps(content).encode())
		return True
	except (OSError, ValueError):
		return False


def reuse_step(args : Args, output_path : str) -> bool:
	try:
		content = json.loads(Path(output_path + '.resume.json').read_text())
		return content.get('signature') == create_signature(args) and content.get('output') == hash_file(Path(output_path))
	except (OSError, ValueError, AttributeError):
		return False
