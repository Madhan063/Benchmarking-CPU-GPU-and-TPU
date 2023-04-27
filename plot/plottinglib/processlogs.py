import pandas as pd
import numpy as np
import config
import os

def get_model_category(model_name: str):
    for cat in config.MODEL_CATEGORY:
        if cat in model_name:
            return cat
    return 'misc'

def is_float_metric(metric_name: str):
    if metric_name in config.FLOAT_METRICS:
        return True
    else:
        return False

def processlog_tpu(dirpath: str):
    columns = config.TPU_DF_COLS
    data = pd.DataFrame(columns= columns)

    files = os.listdir(dirpath)
    files.sort()

    for filename in files:
        if filename.startswith("."):
            continue
        
        fpath = os.path.join(dirpath, filename)
        fpath = os.path.join(fpath, 'mylog.txt')
        info = {}
    
        name = filename.rsplit('_', 3)
        model_name = name[0]
        precision = name[1]
        batch_size = int(name[2])
        num_inference_step = int(name[3])

        info['model_category'] = get_model_category(model_name)
        info['batch_size'] = batch_size
        info['num_inference_steps'] = num_inference_step

        with open(fpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Model' in line:
                    info['model_name'] = model_name
                elif 'Batch Size' in line:
                    continue
                elif 'Precision' in line:
                    info['precision'] = precision
                elif 'Inference' in line:
                    values = line.split(': ')
                    info['inference_time'] = float(values[1]) / batch_size
                elif 'Host-to-device' in line:
                    values = line.split(': ')
                    info['host_to_device_time'] = float(values[1]) / batch_size
                elif 'Device-to-host' in line:
                    values = line.split(': ')
                    info['device_to_host_time'] = float(values[1]) / batch_size

            info['model_parameters'] = config.MODEL_PARAMETERS[info['model_name']]
            info['avg_time'] = info['inference_time'] + info['host_to_device_time'] + info['device_to_host_time']
            info['communication_time'] = info['host_to_device_time'] + info['device_to_host_time']
        data.loc[len(data)] = info
        
    return data


def processlog_gpu(dirpath: str):

    columns = config.GPU_DF_COLS
    data = pd.DataFrame(columns = columns)

    files = os.listdir(dirpath)
    files.sort()

    for filename in files:
        
        if filename.startswith("."):
            continue

        path = os.path.join(dirpath, filename)
        fpath = os.path.join(path, 'mylog.log')

        info = {}
        model_name, precision, batch_size, num_inference_step = filename.rsplit('_', 3)

        info['model_category'] = get_model_category(model_name)
        info['batch_size'] = float(batch_size)
        info['num_inference_steps'] = float(num_inference_step)

        with open(fpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                values = line.split('[INFO] ')
                values = values[1]
                if ':' in values:
                    values = values.split(': ')
                    metric_name = config.LOGNAMES_TO_DFNAMES[values[0]]
                    
                    if metric_name != 'cuda_device_name':
                        metric_value = values[1].split()[0]
                    else:
                        metric_value = values[1]
                    if metric_name == 'precision':
                        metric_value = config.PRECISION_TO_NAMES[metric_value]
                    if is_float_metric(metric_name):
                        metric_value = float(metric_value)
                    
                    info[metric_name] = metric_value

            info['inference_time'] = info['inference_time'] / info['batch_size']
            info['host_to_device_time'] = info['host_to_device_time'] / info['batch_size']
            info['device_to_host_time'] = info['device_to_host_time'] / info['batch_size']
            info['communication_time'] = info['host_to_device_time'] + info['device_to_host_time']
        # print(info)
        data.loc[len(data)] = info
    
    return data

def processlog_cpu(dirpath: str):

    columns = config.CPU_DF_COLS
    data = pd.DataFrame(columns = columns)

    files = os.listdir(dirpath)
    files.sort()

    for filename in files:
        
        if filename.startswith("."):
            continue

        path = os.path.join(dirpath, filename)
        fpath = os.path.join(path, 'mylog.log')

        info = {}
        model_name, precision, batch_size, num_inference_step = filename.rsplit('_', 3)

        info['model_category'] = get_model_category(model_name)
        info['batch_size'] = float(batch_size)
        info['num_inference_steps'] = float(num_inference_step)

        with open(fpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '[METRIC]' in line:
                    values = line.split('[METRIC] ')
                    values = values[1]
                    if ':' in values:
                        values = values.split(': ')
                        metric_name = config.LOGNAMES_TO_DFNAMES[values[0]]
                        
                        metric_value = values[1].split()[0]
                        if metric_name == 'precision':
                            metric_value = config.PRECISION_TO_NAMES[metric_value]
                        if is_float_metric(metric_name):
                            metric_value = float(metric_value)
                        
                        info[metric_name] = metric_value
            # print(fpath)
            info['inference_time'] = info['inference_time'] / info['batch_size']
            info['host_to_device_time'] = info['host_to_device_time'] / info['batch_size']
            info['device_to_host_time'] = info['device_to_host_time'] / info['batch_size']

        data.loc[len(data)] = info
    # data = data.sort_values(by = ['model_name', 'precision', 'batch_size', 'num_inference_steps'])
    return data

if __name__ == '__main__':
    # This code should be run from the plot folder
    curr_path = os.getcwd()
    curr_path = os.path.join(curr_path, 'final data')

    gpu_dir_path = os.path.join(curr_path, 'gpuoutputs')
    gpu_df = processlog_gpu(gpu_dir_path)
    gpu_df.to_csv("gpu_data.csv")

    cpu_dir_path = os.path.join(curr_path, 'cpuoutputs_p1')
    cpu_df = processlog_cpu(cpu_dir_path)
    cpu_df.to_csv("cpu_data_p1.csv")

    cpu_dir_path = os.path.join(curr_path, 'cpuoutputs_p2')
    cpu_df = processlog_cpu(cpu_dir_path)
    cpu_df.to_csv("cpu_data_p2.csv")

    tpu_dir_path = os.path.join(curr_path, 'tpuoutputs')
    tpu_df = processlog_tpu(tpu_dir_path)
    tpu_df.to_csv('tpu_data.csv')


                