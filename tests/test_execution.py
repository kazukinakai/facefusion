from facefusion import state_manager
from facefusion.execution import create_inference_providers, get_available_execution_providers, has_execution_provider


def test_has_execution_provider() -> None:
	assert has_execution_provider('cpu') is True
	assert has_execution_provider('openvino') is False


def test_get_available_execution_providers() -> None:
	assert 'cpu' in get_available_execution_providers()


def test_create_inference_providers() -> None:
	inference_providers =\
	[
		('CUDAExecutionProvider',
		{
			'device_id': 1,
			'cudnn_conv_algo_search': 'EXHAUSTIVE'
		}),
		'CPUExecutionProvider'
	]

	assert create_inference_providers(1, [ 'cpu', 'cuda' ]) == inference_providers


def test_create_inference_providers_tensorrt_fp16() -> None:
	state_manager.init_item('tensorrt_fp16', True)
	_, inference_option_set = create_inference_providers(0, [ 'tensorrt' ])[0]

	assert inference_option_set.get('trt_fp16_enable') is True

	state_manager.init_item('tensorrt_fp16', False)
	_, inference_option_set = create_inference_providers(0, [ 'tensorrt' ])[0]

	assert 'trt_fp16_enable' not in inference_option_set
