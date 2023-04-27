import os
import torch
import timm
import time
import json
import psutil
import subprocess
from mylogger import MyLogger
# from utils.config import MODEL_INPUT_SHAPES

class PyTorchInference:
    def __init__(self, output_dir, model_name):
        # create a logger object to record model inference events
        self.logger = MyLogger(output_dir, name = model_name)
        
        # # get the device (CPU/GPU/TPU) where the inference will be performed
        self.device = self._get_device()
        self.set_device_type()
        
        # initialize a metrics dictionary to keep track of model inference metrics
        self.metrics = {}

    # this method creates a PyTorch model given a model name
    def get_model(self, model_name):
        try:
            self.logger.info(f"Trying to load pre-trained {model_name} from the timm library")
            # create an instance of the PyTorch model using the timm library and set pretrained to True
            model = timm.create_model(model_name, pretrained=True)

            # Log model information
            self.logger.info(f"Loaded {model_name} model successfully")

            # move the model to the device
            model.to(self.device)

            # set the model to evaluation mode
            model.eval()

            # return the model
            return model

        except Exception as e:
            # Log any errors that occur during model loading
            self.logger.error(f"Error loading {model_name} model: {str(e)}")
            raise e

    # this method tells us if the device is CPU/GPU/TPU
    def set_device_type(self):
        # check if the device is a GPU
        if self.device.type == 'cuda':
            self.device_type = 'GPU'
        
        # check if the device is a CPU
        elif self.device.type == 'cpu':
            self.device_type = 'CPU'
        
        # check if the device is a TPU
        elif self.device.type == 'xla':
            self.device_type = 'TPU'
            
        
        # self.metrics['device'] = self.device.type
        # log the device type
        self.logger.metric(f"Device Type: {self.device.type}")
    
    # this method determines the device where inference will be performed
    def _get_device(self):
        # check if a CUDA-enabled GPU is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        
        # if running on Google Colab and a TPU is available
        elif 'COLAB_TPU_ADDR' in os.environ:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        
        # otherwise, use the CPU
        else:
            device = torch.device("cpu")
        
        # log the device type
        self.logger.metric(f"Device: {device}")

        # return the chosen device
        return device

    def get_model_params(self, model):
        num_params = None

        try:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except AttributeError as e:
            logger.error(f"Unable to measure model params due to error: {e}")

        return num_params
    
    # this method performs inference on the defined model_name
    def run_inference(self, model_name, batch_size, precision = torch.float16, inference_loader = None, num_inference_steps = None):
        
        # retrieve the model for inference
        model = self.get_model(model_name)

        # Try to set the precision of the model
        try:
            if precision == torch.float16:
                model = model.half()
            elif precision == torch.bfloat16:
                model = model.to(dtype = torch.bfloat16)
            elif precision == torch.float32:
                model = model
            elif precision == torch.float64:
                model = model.double()
        except:
            self.logger.info(f"Failed to set {precision} precision for the model.")
        
        # recording number of model parameters
        numparams = self.get_model_params(model)

        # turn off gradients during inference
        with torch.no_grad():
            
            # log that inference has started
            self.logger.info("Starting inference...")
            
            # create CUDA events to measure inference time
            # start_time = torch.cuda.Event(enable_timing=True)
            # end_time = torch.cuda.Event(enable_timing=True)
            
            avg_inference_time = None

            if inference_loader is not None:
                # perform inference on the given inference_loader

                self.logger.info("Running Inference on the provided Inference Loader")
                
                total_inference_time = 0
                total_dev_to_host_time = 0
                total_host_to_dev_time = 0
                total_time = 0

                num_samples = 0

                for inputs, targets in inference_loader:
                    # move the input tensor to the device
                    try:
                        inputs = inputs.to(dtype = precision)
                        start_time_devt = time.time()
                        inputs = torch.Tensor.to(inputs, self.device)
                        end_time_devt = time.time()

                    except (AttributeError, RuntimeError, TypeError):
                        self.logger.error(f"There was a problem with the input tensor or device while passing input tensor to the device: {self.device}.")
                        return
                    
                    # run the model on the input tensor to get the output
                    try:
                        # record the start time of the inference
                        start_time_inf = time.time()

                        output = model(inputs)
                        
                        # record the end time of the inference
                        end_time_inf = time.time()

                        output = torch.Tensor.to(output, "cpu")

                        end_time = time.time()

                    except (ValueError, IndexError):
                        self.logger.error("There was a problem with the input data while passing it to the model to compute the output.")
                        return
            
                    # calculate the inference time in seconds
                    dev_to_host_time = end_time_devt - start_time_devt
                    inference_time = end_time_inf - start_time_inf
                    host_to_dev_time = end_time - end_time_inf
                    
                    total_inference_time += inference_time
                    total_dev_to_host_time += dev_to_host_time
                    total_host_to_dev_time += host_to_dev_time
                    total_time += inference_time + dev_to_host_time + host_to_dev_time

                    num_samples += inputs.size(0)

                # calculate average inference time over all samples
                avg_inference_time = total_inference_time / num_samples
                avg_dev_to_host_time = total_dev_to_host_time / num_samples
                avg_host_to_dev_time = total_host_to_dev_time / num_samples
                avg_time = total_time / num_samples
            
            elif num_inference_steps is not None:
                # generate random images and perform inference for the given number of steps

                self.logger.info("Running Inference on randonly generated samples")

                total_inference_time = 0
                total_dev_to_host_time = 0
                total_host_to_dev_time = 0
                total_time = 0

                for step in range(num_inference_steps):
                    # generate random input tensor
                    # inputs_shape = MODEL_INPUT_SHAPES[model_name]
                    
                    # inputs = torch.randn(batch_size, inputs_shape[2], inputs_shape[1], inputs_shape[0]).to(self.device)
                    inputs = torch.randn(batch_size, 3, 224, 224, dtype = precision)
                    
                    # move the input tensor to the device
                    try:
                        start_time_devt = time.time()
                        inputs = torch.Tensor.to(inputs, self.device)
                        end_time_devt = time.time()

                    except (AttributeError, RuntimeError, TypeError):
                        self.logger.error(f"There was a problem with the input tensor or device while passing input tensor to the device: {self.device}.")
                        return

                    # run the model on the input tensor to get the output
                    try:
                        # record the start time of the inference
                        start_time_inf = time.time()

                        output = model(inputs)
                        
                        # record the end time of the inference
                        end_time_inf = time.time()

                        output = torch.Tensor.to(output, "cpu")

                        end_time = time.time()

                    except (ValueError, IndexError):
                        self.logger.error("There was a problem with the input data while passing it to the model to compute the output.")
                        return

                    # calculate the inference time in seconds
                    dev_to_host_time = end_time_devt - start_time_devt
                    inference_time = end_time_inf - start_time_inf
                    host_to_dev_time = end_time - end_time_inf
                    
                    total_inference_time += inference_time
                    total_dev_to_host_time += dev_to_host_time
                    total_host_to_dev_time += host_to_dev_time
                    total_time += inference_time + dev_to_host_time + host_to_dev_time

                # calculate average inference time over all steps
                avg_inference_time = total_inference_time / num_inference_steps
                avg_dev_to_host_time = total_dev_to_host_time / num_inference_steps
                avg_host_to_dev_time = total_host_to_dev_time / num_inference_steps
                avg_time = total_time / num_inference_steps

            # Store the metrics in the metrics dictionary
            self.metrics['Model Name'] = model_name
            self.metrics['Precision'] = precision
            if numparams is not None:
                self.metrics['Model Parameters'] = numparams
            self.metrics['Avegrage Inference time'] = avg_inference_time
            self.metrics['Average Device to Host time'] = avg_dev_to_host_time
            self.metrics['Average Host to Device time'] = avg_host_to_dev_time
            self.metrics['Average time'] = avg_time

            # log the model name and inference time
            self.logger.metric(f"Model Name: {model_name}")
            self.logger.metric(f"Batch Size: {batch_size}")
            self.logger.metric(f"Precision: {precision}")
            if numparams is not None:
                self.logger.metric(f"Model parameters: {numparams}")
            self.logger.metric(f"Average Inference time: {avg_inference_time:.4f} s")
            self.logger.metric(f"Average Device to Host time: {avg_dev_to_host_time:.4f} s")
            self.logger.metric(f"Average Host to Device time: {avg_host_to_dev_time:.4f} s")
            self.logger.metric(f"Average time: {avg_time:.4f} s")
            
            # log the performance metrics
            self.log_performance_metrics()
    
    # this method will log the performance metrics based on the device
    def log_performance_metrics(self):
        try:
            # check if the device is a GPU
            if self.device.type == 'cuda':
                self.metrics['device'] = self.device.type
                # log GPU metrics
                self._log_gpu_metrics()
            
            # check if the device is a CPU
            elif self.device.type == 'cpu':
                self.metrics['device'] = self.device.type
                # log CPU metrics
                self._log_cpu_metrics()
            
            # check if the device is a TPU
            elif self.device.type == 'xla':
                self.metrics['device'] = self.device.type
                # log TPU metrics
                self._log_tpu_metrics()
                
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {str(e)}")

    # this method logs the gpu metrics using nvprof and nvidia-smi
    def _log_gpu_metrics(self):
        
        try:
            # Log additional GPU metrics using Nvidia-smi
            # Run a command to collect more GPU metrics using nvidia-smi
            # The command includes the metrics to be collected and the `--format` flag to format the output as csv.
            gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader'], encoding='utf-8')
            # Parse the output to extract the metrics.
            gpu_info = gpu_info.strip().split('\n')
            gpu_utilization = []
            gpu_memory_used = []
            gpu_memory_total = []
            gpu_temperature = []
            gpu_power_draw = []
            for i in gpu_info:
                gpu_utilization.append(int(i.split(',')[1].split(' ')[1]))
                gpu_memory_used.append(int(i.split(',')[2].split(' ')[1]))
                gpu_memory_total.append(int(i.split(',')[3].split(' ')[1]))
                gpu_temperature.append(int(i.split(',')[4].split(' ')[1]))
                gpu_power_draw.append(float(i.split(',')[5].split(' ')[1]))
            # Calculate the average of each metric and store them in a dictionary.
            self.metrics['GPU Utilization'] = sum(gpu_utilization) / len(gpu_utilization)
            self.metrics['GPU Memory Used'] = sum(gpu_memory_used) / len(gpu_memory_used)
            self.metrics['GPU Memory Total'] = sum(gpu_memory_total) / len(gpu_memory_total)
            self.metrics['GPU Temperature'] = sum(gpu_temperature) / len(gpu_temperature)
            self.metrics['GPU Power Consumption'] = sum(gpu_power_draw) / len(gpu_power_draw)

            # Log the metrics.
            self.logger.metric(f"GPU Utilization: {self.metrics['GPU Utilization']:.4f}")
            self.logger.metric(f"GPU Memory Used: {self.metrics['GPU Memory Used']:.4f} MB")
            self.logger.metric(f"GPU Memory Total: {self.metrics['GPU Memory Total']:.4f} MB")
            self.logger.metric(f"GPU Temperature: {self.metrics['GPU Temperature']:.4f}")
            self.logger.metric(f"GPU Power Consumption: {self.metrics['GPU Power Consumption']:.4f} W")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error when running 'nvidia-smi': {e.output.strip()}")
        except Exception as e:
            self.logger.error(f"Error when running 'nvidia-smi' other exception occured: {e}")

        try:
            # Log PyTorch metrics
            self.metrics['GPU Count'] = torch.cuda.device_count()
            self.metrics['CUDA Version'] = torch.version.cuda
            self.metrics['Cudnn Version'] = torch.backends.cudnn.version()
            self.metrics['CUDA Device Name'] = torch.cuda.get_device_name(0) 
            self.metrics['Max GPU Memory Allocated'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            self.metrics['Max GPU Memory Reserved'] = torch.cuda.max_memory_reserved() / (1024 * 1024)

            self.logger.metric(f"GPU Count: {torch.cuda.device_count()}")
            self.logger.metric(f"CUDA Version: {torch.version.cuda}")
            self.logger.metric(f"Cudnn Version: {torch.backends.cudnn.version()}")
            self.logger.metric(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            self.logger.metric(f"Max GPU Memory Allocated: {self.metrics['Max GPU Memory Allocated']:.2f} MB")
            self.logger.metric(f"Max GPU Memory Reserved: {self.metrics['Max GPU Memory Reserved']:.2f} MB")
        except Exception as e:
            self.logger.error(f"Error when running pytorch metrics: {e}")
        
    # this method logs the tpu metrics
    def _log_tpu_metrics(self):
        pass

    # this method logs the cpu metrics
    def _log_cpu_metrics(self):
        try:
            # Retrieve the CPU count
            cpu_count = psutil.cpu_count()

            # Retrieve the CPU utilization percentage
            cpu_percent = psutil.cpu_percent()

            # Retrieve the virtual memory information
            mem_info = psutil.virtual_memory()

            # Calculate the used memory in MB
            mem_used = mem_info.used / 1024**2

            # Calculate the available memory in MB
            mem_avail = mem_info.available / 1024**2

            self.metrics['CPU Count'] = cpu_count
            self.metrics['CPU Utilization'] = cpu_percent
            self.metrics['CPU Memory Used'] = mem_used
            self.metrics['CPU Memory Available'] = mem_avail

            # Log the numver of CPUs available
            self.logger.metric(f"CPU Count: {cpu_count}")

            # Log the CPU utilization percentage
            self.logger.metric(f"CPU Utilization: {cpu_percent}")

            # Log the used memory in MB
            self.logger.metric(f"CPU Memory Used: {mem_used}")

            # Log the available memory in MB
            self.logger.metric(f"CPU Memory Available: {mem_avail}")

        except Exception as e:
            self.logger.error(f"Error while logging CPU metrics: {e}")

if __name__ == '__main__':
    model_name = "resnet18"
    batch_size = 16
    inference_loader = None
    num_inference_steps = 50
    inference = PyTorchInference()
    inference.run_inference(model_name, batch_size, inference_loader, num_inference_steps)
