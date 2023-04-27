import subprocess

# Run nvidia-smi and capture the output
output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,pci.bus_id,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory', '--format=csv,noheader'])

# Parse the output to extract the desired information
lines = output.decode('utf-8').strip().split('\n')
header = lines[0].split(', ')
metrics = []

for line in lines[1:]:
    values = line.split(', ')
    gpu_metrics = {}
    for i, field in enumerate(header):
        gpu_metrics[field] = values[i]
    metrics.append(gpu_metrics)

# Print the extracted information
for i, gpu in enumerate(metrics):
    print(f"GPU {i}:")
    for metric, value in gpu.items():
        print(f"  {metric}: {value}")


import pycuda.driver as cuda
import pycuda.autoinit

# Get GPU device name
device_name = cuda.Device(0).name()
print("Device Name:", device_name)

# Get GPU design specs
compute_capability = cuda.Device(0).compute_capability()
clock_rate = cuda.Device(0).get_attribute(cuda.device_attribute.CLOCK_RATE) / 1000  # kHz to MHz
num_cores = cuda.Device(0).get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
print("Compute Capability:", compute_capability)
print("Clock Rate:", clock_rate, "GHz")
print("Number of Cores:", num_cores)


import cpuinfo

# Get CPU information
info = cpuinfo.get_cpu_info()

# Get CPU device name
device_name = info['brand_raw']
print("Device Name:", device_name)

print('CPU Specs')
for key, value in info.items():
    print(key, ' : ', value)