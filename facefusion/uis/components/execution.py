from typing import List, Optional

import gradio

from facefusion import content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager, translator, voice_extractor
from facefusion.execution import get_available_execution_providers
from facefusion.filesystem import get_file_name, resolve_file_paths
from facefusion.processors.core import get_processors_modules
from facefusion.types import ExecutionProvider

EXECUTION_PROVIDERS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
EXECUTION_TENSORRT_FP16_CHECKBOX : Optional[gradio.Checkbox] = None


def render() -> None:
	global EXECUTION_PROVIDERS_CHECKBOX_GROUP
	global EXECUTION_TENSORRT_FP16_CHECKBOX

	EXECUTION_PROVIDERS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.execution_providers_checkbox_group'),
		choices = get_available_execution_providers(),
		value = state_manager.get_item('execution_providers')
	)
	EXECUTION_TENSORRT_FP16_CHECKBOX = gradio.Checkbox(
		label = translator.get('uis.execution_tensorrt_fp16_checkbox'),
		value = state_manager.get_item('tensorrt_fp16')
	)


def listen() -> None:
	EXECUTION_PROVIDERS_CHECKBOX_GROUP.change(update_execution_providers, inputs = EXECUTION_PROVIDERS_CHECKBOX_GROUP, outputs = EXECUTION_PROVIDERS_CHECKBOX_GROUP)
	EXECUTION_TENSORRT_FP16_CHECKBOX.change(update_tensorrt_fp16, inputs = EXECUTION_TENSORRT_FP16_CHECKBOX)


def update_execution_providers(execution_providers : List[ExecutionProvider]) -> gradio.CheckboxGroup:
	clear_inference_pools()
	execution_providers = execution_providers or get_available_execution_providers()
	state_manager.set_item('execution_providers', execution_providers)
	return gradio.CheckboxGroup(value = state_manager.get_item('execution_providers'))


def update_tensorrt_fp16(tensorrt_fp16 : bool) -> None:
	clear_inference_pools()
	state_manager.set_item('tensorrt_fp16', tensorrt_fp16)


def clear_inference_pools() -> None:
	common_modules =\
	[
		content_analyser,
		face_classifier,
		face_detector,
		face_landmarker,
		face_masker,
		face_recognizer,
		voice_extractor
	]
	available_processors = [ get_file_name(file_path) for file_path in resolve_file_paths('facefusion/processors/modules') ]
	processor_modules = get_processors_modules(available_processors)

	for module in common_modules + processor_modules:
		if hasattr(module, 'clear_inference_pool'):
			module.clear_inference_pool()
