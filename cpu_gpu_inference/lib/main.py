import os
import torch
from inference import PyTorchInference

cur_path = os.getcwd()
path = os.path.join(cur_path, "outputs")

batch_sizes = [16, 32, 64]
num_inference_steps = [512, 1024]

model_names = [
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

cpu_data_precisions = {
    "float32": torch.float32,
    "float64": torch.float64
}

gpu_data_precisions = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64
}

data_precisions = gpu_data_precisions

dirs_list = os.listdir(os.path.join(cur_path, 'outputs_cpu'))

for model_name in model_names:
    for precision_name, precision in zip(data_precisions.keys(), data_precisions.values()):
        for batch_size in batch_sizes:
            for num_inference_step in num_inference_steps:
                output_dir = f"{model_name}_{precision_name}_{batch_size}_{num_inference_step}"
                if output_dir in dirs_list:
                    continue 
                output_path = os.path.join(path, output_dir)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                inference = PyTorchInference(output_path, model_name)
                inference.run_inference(model_name, batch_size, precision=precision, num_inference_steps=num_inference_step)