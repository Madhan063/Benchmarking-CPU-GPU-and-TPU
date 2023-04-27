import torch 

BATCH_SIZES = [16, 32, 64]

NUM_INFERENCE_STEPS = [512, 1024]

MODEL_NAMES = [
                'efficientnet_b0',
                'efficientnet_b1',
                'efficientnet_b2',
                'efficientnet_b3',
                'efficientnet_b4',
                'mobilenetv2_100',
                'nasnetalarge',
                'resnet18',
                'resnet26',
                'resnet34',
                'resnet50',
                'resnet101',
                'resnet152',
                'vgg11',
                'vgg13',
                'vgg16',
                'vgg19',
                'xception'
            ]

CPU_DATA_PRECISIONS = {
    "float32": torch.float32,
    "float64": torch.float64
}

GPU_DATA_PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64
}

TPU_DATA_PRECISIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64
}

MODEL_CATEGORY = [
                    'efficientnet',
                    'resnet',
                    'vgg',
                    'misc'
                ]

GPU_DF_COLS = [
                'device',
                'device_type',
                'model_name',
                'model_category',
                'batch_size',
                'num_inference_steps',
                'precision',
                'model_parameters',
                'inference_time',
                'device_to_host_time',
                'host_to_device_time',
                'avg_time',
                'gpu_utilization',
                'gpu_mem_used',
                'gpu_mem_total',
                'gpu_temp',
                'gpu_power',
                'gpu_count',
                'cuda_version',
                'cudnn_version',
                'cuda_device_name',
                'max_gpu_mem_allocated',
                'max_gpu_mem_reserved'
            ]

CPU_DF_COLS = [
                'device',
                'device_type',
                'model_name',
                'model_category',
                'batch_size',
                'num_inference_steps',
                'precision',
                'model_parameters',
                'inference_time',
                'device_to_host_time',
                'host_to_device_time',
                'avg_time',
                'cpu_count',
                'cpu_utilization',
                'cpu_memory_used',
                'cpu_memory_available'
            ]

TPU_DF_COLS = []

LOGNAMES_TO_DFNAMES = {
                        'Device' : 'device',
                        'Device Type' : 'device_type',
                        'Model Name' : 'model_name',
                        'Precision' : 'precision',
                        'Model Parameters' : 'model_parameters',
                        'Average Inference time' : 'inference_time',
                        'Average Device to Host time' : 'device_to_host_time',
                        'Average Host to Device time' : 'host_to_device_time',
                        'Average time' : 'avg_time',
                        'GPU Utilization' : 'gpu_utilization',
                        'GPU Memory Used' : 'gpu_mem_used',
                        'GPU Memory Total' : 'gpu_mem_total',
                        'GPU Temperature' : 'gpu_temp',
                        'GPU Power Consumption' : 'gpu_power',
                        'GPU Count' : 'gpu_count',
                        'CUDA Version' : 'cuda_version',
                        'Cudnn Version' : 'cudnn_version',
                        'CUDA Device Name' : 'cuda_device_name',
                        'Max GPU Memory Allocated' : 'max_gpu_mem_allocated',
                        'Max GPU Memory Reserved' : 'max_gpu_mem_reserved',
                        'Batch Size' : 'batch_size',
                        'CPU Count' : 'cpu_count',
                        'CPU Utilization' : 'cpu_utilization',
                        'CPU Memory Used' : 'cpu_memory_used',
                        'CPU Memory Available' : 'cpu_memory_available'
                    }

PRECISION_TO_NAMES = {
                        'torch.float16' : 'float16',
                        'torch.bfloat16' : 'bfloat16',
                        'torch.float32' : 'float32',
                        'torch.float64' : 'float64'
                    }

FLOAT_METRICS = [   
                    'batch_size',
                    'num_inference_steps',
                    'precision',
                    'model_parameters',
                    'inference_time',
                    'device_to_host_time',
                    'host_to_device_time',
                    'avg_time',
                    'gpu_utilization',
                    'gpu_mem_used',
                    'gpu_mem_total',
                    'gpu_temp',
                    'gpu_power',
                    'gpu_count',
                    'cuda_version',
                    'cudnn_version',
                    'cuda_device_name',
                    'max_gpu_mem_allocated',
                    'max_gpu_mem_reserved'
                ]