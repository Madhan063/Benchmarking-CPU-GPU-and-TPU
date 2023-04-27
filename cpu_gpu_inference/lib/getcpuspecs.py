import cpuinfo

# Get CPU information
info = cpuinfo.get_cpu_info()

# Get CPU device name
device_name = info['brand_raw']
print("Device Name:", device_name)

print('CPU Specs')
for key, value in info.items():
    print(key, ' : ', value)