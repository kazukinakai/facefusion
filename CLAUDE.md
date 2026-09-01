# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FaceFusion is a face-manipulation platform (face swapping, enhancement, editing, lip sync, etc.). This is `kazukinakai/facefusion`, a fork of `facefusion/facefusion` (`upstream` remote). The fork carries local changes on top of upstream — chiefly job-splitting/resume for the instant runner, output-encoding tweaks, and TensorRT fp16. Keep fork changes minimal and upstream-shaped so they can be submitted as PRs.

## Commands

Python 3.10+ required (CI runs 3.12). System dependencies `curl` and `ffmpeg` must be on PATH or the program exits at `pre_check`.

```bash
# Install deps (picks the onnxruntime build). --skip-conda when not using the bundled conda env.
python install.py --onnxruntime default --skip-conda
# onnxruntime options: default, cuda, openvino, directml, qnn, migraphx, rocm

# Run (launches the Gradio UI by default)
python facefusion.py run

# Run a specific UI layout: default | webcam | benchmark | jobs
python facefusion.py run --ui-layouts benchmark

# Headless one-shot
python facefusion.py headless-run -s source.jpg -t target.mp4 -o out.mp4 --processors face_swapper

# Lint (must pass; CI gate)
flake8 facefusion.py install.py && flake8 facefusion tests
mypy facefusion.py install.py && mypy facefusion tests

# Tests
pytest                                  # full suite
pytest tests/test_cli_face_swapper.py   # single file
pytest tests/test_cli_face_swapper.py::test_swap_face_to_image   # single test
pytest tests --cov facefusion           # with coverage
```

Many tests download model/media fixtures on first run and shell out to real `ffmpeg`, so they are slow and network-dependent.

## Code style (enforced by CI — match exactly)

- **Tabs for indentation**, not spaces.
- List/dict literals use inner spacing: `[ x, y ]`, not `[x, y]`.
- Keyword arguments are spaced: `func(arg = value)`, not `func(arg=value)`.
- `dict`-of-dict literals open the brace on the next line (see `create_static_model_set`).
- Imports are ordered (`flake8-import-order`, pycharm style); `application_import_names = facefusion`.
- mypy is strict (`disallow_untyped_defs`, `disallow_any_generics`, `disallow_untyped_calls`). Annotate everything; types live in `types.py` files.

## Architecture

### Entry and routing

`facefusion.py` → `core.cli()` builds the argparse program (`program.create_program`), validates args, applies them into the state manager, then `core.route()` dispatches on the `command` item. Commands fall into groups: the live UI (`run`), one-shot processing (`headless-run`, `batch-run`), job management (`job-create`, `job-add-step`, `job-split`, …), and job execution (`job-run`, `job-retry`, …). `force-download` and `benchmark` are standalone.

### State manager — the global config bus

`state_manager.py` is the single source of truth for all config at runtime. Nothing threads config through call args; code reads `state_manager.get_item('...')` anywhere. Keys are typed in `types.py` (`StateKey`) and per-processor `types.py` (`ProcessorStateKey`).

Critically, state is **partitioned by app context** (`cli` vs `ui`), resolved at call time by `app_context.detect_app_context()`, which walks the stack looking for `facefusion/jobs` (→ `cli`) or `facefusion/uis` (→ `ui`) in the calling frame's filename. The same getter returns different values depending on who calls it. This lets the UI hold edits separately from CLI execution state; `sync_state`/`sync_item` copy `ui` → `cli`. Inference pools are partitioned the same way and shared across contexts when already loaded.

### Layered config

`config.py` reads `facefusion.ini` (path overridable via `--config-path`). The INI provides defaults; CLI args override them. Don't reintroduce env-var fallbacks for config.

### Processors — the plugin system

Each processor under `facefusion/processors/modules/<name>/` is a self-contained package (`core.py`, `choices.py`, `types.py`, `locales.py`). `processors/core.py` loads them dynamically by name and enforces a duck-typed contract: every module's `core.py` must implement `get_inference_pool`, `clear_inference_pool`, `register_args`, `apply_args`, `pre_check`, `pre_process`, `post_process`, `process_frame` (missing any → `NotImplementedError` → exit). The active processors come from `state_manager.get_item('processors')`.

`process_frame(inputs)` receives a dict with the reference/source/target/temp vision frames, source audio/voice frames, and the working mask; it returns `(temp_vision_frame, temp_vision_mask)`. Processors chain — the output of one feeds the next.

Adding a processor = create a new module directory implementing the eight methods and register its models via `create_static_model_set`. `live_portrait.py` and `pixel_boost.py` under `processors/` are shared helpers, not processors.

### Inference

`inference_manager.py` owns ONNX Runtime sessions. `get_inference_pool` caches sessions keyed by `module.model.device.providers`, reusing across cli/ui contexts. `execution.py` resolves execution providers (cuda, tensorrt, openvino, directml, coreml, …) and device ids from state. A module may export `resolve_execution_providers` to override the global provider list (e.g. forcing CPU for a specific model). Models, hashes, and sources are declared per-module and fetched on demand via `download.py` (`force-download` pre-fetches everything).

### Workflows — the processing pipeline

`workflows/image_to_image.py` and `workflows/image_to_video.py` are the actual frame-processing pipelines, selected in `core.conditional_process()` by whether the target is an image or video. The video workflow is a fixed task list: `setup → extract_frames → process_video → merge_frames → restore_audio → finalize_video`. `process_video` extracts frames with ffmpeg, fans them out across a `ThreadPoolExecutor` (`execution_thread_count`), runs each through the processor chain, then re-encodes. `ErrorCode` 4 means a user stop was requested (`process_manager`).

### Jobs

`jobs/job_manager.py` persists jobs as JSON in `jobs_path` with status dirs (drafted/queued/completed/failed). A job is a list of **steps**, each step a frozen `Args` snapshot. `headless-run`/`batch-run` are sugar: they create a temp job, add steps, submit, and run it. `job_runner.py` executes steps via the `process_step` callback passed down from `core`. Fork-specific: `job-split` splits one step into frame-range steps (`step_frame_total`), and retry resumes from where a failed job stopped rather than restarting.

### UI

`uis/core.py` loads a layout module from `uis/layouts/` (default, webcam, benchmark, jobs). Layouts compose `uis/components/*` (one component per option group, mirroring the processors). Gradio 5. UI edits write to the `ui` state partition; the instant runner / job runner syncs to `cli` before executing.

### Common-module integrity check

`core.common_pre_check()` hashes the source of `content_analyser.py` and asserts it equals a hardcoded value. The content analyser (NSFW gate) is intentionally tamper-checked — editing it breaks startup. Don't touch it unless that's the explicit task.

## Fork workflow

`origin` = this fork, `upstream` = facefusion/facefusion. Get explicit confirmation before push/deploy.
