# CPU and GPU Inference

## Requirements

This Python Inference library requires the following packages to be installed:

- `tensorflow==2.10.0`
- `keras`
- `h5py`
- `numpy`
- `nvsmi`
- `psutil`
- `pandas`
- `torch`
- `torchvision`
- `tqdm`
- `timm`
- `thop`
- `matplotlib`
- `pycuda`
- `py-cpuinfo`

## Usage

### For CPU inference
 Follow the following steps to run CPU inference:

 1. Make sure you are in the cpu_gpu_inference folder
 2. Create an outputs directory in this folder to save the results of various runs here. Use the folllowing command:
 ```
 mkdir outputs
 ```
 3. Run the sbatch script to schedule a job on the lonestar6 normal queue
 ```
 sbatch sample_gtx_cpu.slurm
 ```

### For GPU inference
 Follow the following steps to run GPU inference:

 1. Make sure you are in the cpu_gpu_inference folder
 2. Create an outputs directory in this folder to save the results of various runs here. Use the folllowing command:
 ```
 mkdir outputs
 ```
 3. Run the sbatch script to schedule a job on the lonestar6 normal queue
 ```
 sbatch sample_gtx.slurm
 ```