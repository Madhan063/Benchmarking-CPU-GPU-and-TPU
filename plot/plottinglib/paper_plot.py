import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcdefaults, rcParams

def plot(f1: str, f2:str, metric: str, metric_label:str, num_inf_steps1: int, num_inf_steps2: int, plotting_type: list, islog: bool):

    if plotting_type[0] == 0:
        df1 = pd.read_csv(f1)
        df1 = df1[df1['num_inference_steps'] == num_inf_steps1].dropna()
        df1 = df1.sort_values(['model_category', 'model_parameters', 'precision', 'batch_size','num_inference_steps'], ascending=[True, True, True, True, True])
        if plotting_type[1] == 0:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

            for ax, model_cat in zip(axs.flat, df1['model_category'].unique()):
                df1_info = df1[df1['model_category'] == model_cat].dropna()

                df1_values = {}

                if plotting_type[2] == 0:
                    shift_array = [-0.5 * 0.125, 0.5 * 0.125]
                    color_array = ['darkorange', 'blue']
                    precision_array = ['float32', 'float64']
                elif plotting_type[2] == 1 or plotting_type[2] == 2:
                    shift_array = [-1.5 * 0.125, -0.5 * 0.125, 0.5 * 0.125, 1.5 * 0.125]
                    color_array = ['darkorange', 'blue', 'red', 'green']
                    precision_array = config.PRECISION_TO_NAMES.values()
             
                for shift, precision, color in zip(shift_array, precision_array, color_array):
                    df1_values[precision] = df1_info[df1_info['precision'] == precision].dropna()

                    df1_data = df1_values[precision][['model_name', 'batch_size', metric]].dropna()

                    df1_data = np.array(df1_data[metric].dropna().tolist(), dtype = np.float64)
                    # tpu_data = np.array(merged_data.tpu_metric.dropna().tolist(), dtype = np.float64)
                    models = np.array(df1_values[precision]['model_name'].unique())

                    cur_range = np.arange(0, df1_data.shape[0])

                    if islog:
                        l1 = ax.bar(cur_range + shift, abs(np.log10(abs(df1_data))), color=color, width = .125, label = f'{precision}')
                    else:
                        l1 = ax.bar(cur_range + shift, abs(df1_data), color=color, width = .125, label = f'{precision}')
                
                axv_line_parameters = {'linewidth': 2, 'color': 'gray', 'linestyle': (0, (4, 1.2))}
                axv_line_x_coordinates = [i - 0.5 for i in range(cur_range[0], cur_range[-1] + 2, 3)]
                for x in axv_line_x_coordinates:
                    ax.axvline(x, **axv_line_parameters)
                
                # ax.axhline(y=1, linestyle='--', color='violet', label = 'speed up = 1')

                xticks_array = []
                for i, model in enumerate(models):
                    if i == 0:
                        xticks_array.append('16')
                        xticks_array.append(f'32\n{model}')
                        xticks_array.append('64')
                    else:
                        xticks_array.append('16')
                        xticks_array.append(f'32\n{model}')
                        xticks_array.append('64')

                ax.set_xticks(cur_range, xticks_array)
                ax.legend()
                ax.set_xlabel('Batch Size')
                if islog:
                    ax.set_ylabel(f"log({metric_label})")
                else:
                    ax.set_ylabel(metric_label)
                ax.set_title(f'{model_cat}')
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        elif plotting_type[1] == 1:

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

            for ax, model_cat in zip(axs.flat, df1['model_category'].unique()):
                df1_info = df1[df1['model_category'] == model_cat].dropna()
                
                df1_values = {}
                
                shift_array = [-0.25, 0, 0.25]
                batch_size_array = df1_info.batch_size.unique()
                color_array = ['darkorange', 'blue', 'red']

                for shift, batch_size, color in zip(shift_array, batch_size_array, color_array):
                    df1_values[batch_size] = df1_info[df1_info['batch_size'] == batch_size].dropna()

                    df1_data = df1_values[batch_size][['model_name', 'precision', metric]].dropna()

                    
                    df1_data = np.array(df1_data[metric].dropna().tolist(), dtype = np.float64)
                    # tpu_data = np.array(merged_data.tpu_metric.dropna().tolist(), dtype = np.float64)
                    models = np.array(df1_values[batch_size]['model_name'].unique())

                    cur_range = np.arange(0, df1_data.shape[0])

                    if islog:
                        l1 = ax.bar(cur_range + shift, abs(np.log10(abs(df1_data))), color=color, width = .125, label = f'{batch_size}')
                    else:
                        l1 = ax.bar(cur_range + shift, abs(df1_data), color=color, width = .125, label = f'BS: {int(batch_size)}')
                axv_line_parameters = {'linewidth': 2, 'color': 'gray', 'linestyle': (0, (4, 1.2))}

                if plotting_type[2] == 0 or plotting_type[2] == 2:
                    axv_line_x_coordinates = [i - 0.5 for i in range(cur_range[0], cur_range[-1] + 2, 2)]
                elif plotting_type[2] == 1:
                    axv_line_x_coordinates = [i - 0.5 for i in range(cur_range[0], cur_range[-1] + 2, 4)]

                for x in axv_line_x_coordinates:
                    ax.axvline(x, **axv_line_parameters)
                
                # ax.axhline(y=1, linestyle='--', color='violet', label = 'speed up = 1')

                xticks_array = []
                if plotting_type[2] == 0:
                    for i, model in enumerate(models):
                        xticks_array.append(f'fp32\n{model}')
                        xticks_array.append('fp64')
                elif plotting_type[2] == 1 or plotting_type[2] == 2:
                    for i, model in enumerate(models):
                        xticks_array.append('fp16')
                        xticks_array.append('bfp16')
                        xticks_array.append(f'fp32\n{model}')
                        xticks_array.append('fp64')
                    
                ax.set_xticks(cur_range, xticks_array)
                ax.legend()
                ax.set_xlabel('Batch Size')
                if islog:
                    ax.set_ylabel(f"log({metric_label})")
                else:
                    ax.set_ylabel(metric_label)
                ax.set_title(f'{model_cat}')
                ax.tick_params(axis='both', which='major', labelsize=6)
            
            fig.subplots_adjust(hspace=0.3, wspace=0.2)
        

    elif plotting_type[0] == 1 or plotting_type[0] == 2:
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)

        df1 = df1[df1['num_inference_steps'] == num_inf_steps1].dropna()
        df2 = df2[df2['num_inference_steps'] == num_inf_steps2].dropna()

        df1 = df1.sort_values(['model_category', 'model_parameters', 'precision', 'batch_size','num_inference_steps'], ascending=[True, True, True, True, True])
        df2 = df2.sort_values(['model_category', 'model_parameters', 'precision', 'batch_size','num_inference_steps'], ascending=[True, True, True, True, True])

        if plotting_type[1] == 0:
            
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

            for ax, model_cat in zip(axs.flat, df1['model_category'].unique()):
                df1_info = df1[df1['model_category'] == model_cat].dropna()
                df2_info = df2[df2['model_category'] == model_cat].dropna()

                df1_values = {}
                df2_values = {}

                if plotting_type[2] == 0:
                    shift_array = [-0.5 * 0.125, 0.5 * 0.125]
                    color_array = ['darkorange', 'blue']
                    precision_array = ['float32', 'float64']
                elif plotting_type[2] == 1:
                    shift_array = [-1.5 * 0.125, -0.5 * 0.125, 0.5 * 0.125, 1.5 * 0.125]
                    color_array = ['darkorange', 'blue', 'red', 'green']
                    precision_array = config.PRECISION_TO_NAMES.values()
                elif plotting_type[2] == 2:
                    shift_array = [-0.5 * 0.125, 0.5 * 0.125]
                    color_array = ['darkorange', 'blue']
                    precision_array = ['float32', 'float64']
                
                for shift, precision, color in zip(shift_array, precision_array, color_array):
                    df1_values[precision] = df1_info[df1_info['precision'] == precision].dropna()
                    df2_values[precision] = df2_info[df2_info['precision'] == precision].dropna()

                    df1_data = df1_values[precision][['model_name', 'batch_size', metric]].dropna()
                    df2_data = df2_values[precision][['model_name', 'batch_size', metric]].dropna()

                    merged_data = pd.merge(df1_data, df2_data, on=['model_name', 'batch_size'], how='outer')
                    merged_data = merged_data.rename(columns = {'model_name':'model_name', 'batch_size':'batch_size', f'{metric}_x':'df1_metric', f'{metric}_y':'df2_metric'})

                    merged_data.df1_metric = merged_data.df2_metric / merged_data.df1_metric
                    df1_data = np.array(merged_data.df1_metric.dropna().tolist(), dtype = np.float64)
                    # tpu_data = np.array(merged_data.tpu_metric.dropna().tolist(), dtype = np.float64)
                    models = np.array(df1_values[precision]['model_name'].unique())

                    cur_range = np.arange(0, df1_data.shape[0])

                    if islog:
                        l1 = ax.bar(cur_range + shift, abs(np.log10(abs(df1_data))), color=color, width = .125, label = f'{precision}')
                    else:
                        l1 = ax.bar(cur_range + shift, abs(df1_data), color=color, width = .125, label = f'{precision}')
                
                axv_line_parameters = {'linewidth': 2, 'color': 'gray', 'linestyle': (0, (4, 1.2))}
                axv_line_x_coordinates = [i - 0.5 for i in range(cur_range[0], cur_range[-1] + 2, 3)]
                for x in axv_line_x_coordinates:
                    ax.axvline(x, **axv_line_parameters)
                
                if metric_label == 'Speed Up':
                    ax.axhline(y=1, linestyle='--', color='violet', label = 'speed up = 1')

                xticks_array = []
                for i, model in enumerate(models):
                    if i == 0:
                        xticks_array.append('16')
                        xticks_array.append(f'32\n{model}')
                        xticks_array.append('64')
                    else:
                        xticks_array.append('16')
                        xticks_array.append(f'32\n{model}')
                        xticks_array.append('64')

                ax.set_xticks(cur_range, xticks_array)
                ax.legend()
                ax.set_xlabel('Batch Size')
                if islog:
                    ax.set_ylabel(f"log({metric_label})")
                else:
                    ax.set_ylabel(metric_label)
                ax.set_title(f'{model_cat}')
                ax.tick_params(axis='both', which='major', labelsize=8)
            
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            
        elif plotting_type[1] == 1:

            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

            for ax, model_cat in zip(axs.flat, df1['model_category'].unique()):
                df1_info = df1[df1['model_category'] == model_cat].dropna()
                df2_info = df2[df2['model_category'] == model_cat].dropna()
                
                df1_values = {}
                df2_values = {}
                
                shift_array = [-0.25, 0, 0.25]
                batch_size_array = df1_info.batch_size.unique()
                color_array = ['darkorange', 'blue', 'red']

                for shift, batch_size, color in zip(shift_array, batch_size_array, color_array):
                    df1_values[batch_size] = df1_info[df1_info['batch_size'] == batch_size].dropna()
                    df2_values[batch_size] = df2_info[df2_info['batch_size'] == batch_size].dropna()

                    df1_data = df1_values[batch_size][['model_name', 'precision', metric]].dropna()
                    df2_data = df2_values[batch_size][['model_name', 'precision', metric]].dropna()

                    merged_data = pd.merge(df1_data, df2_data, on=['model_name', 'precision'], how='outer')
                    merged_data = merged_data.rename(columns = {'model_name':'model_name', 'precision':'precision', f'{metric}_x':'df1_metric', f'{metric}_y':'df2_metric'})

                    merged_data.df1_metric = merged_data.df2_metric / merged_data.df1_metric
                    df1_data = np.array(merged_data.df1_metric.dropna().tolist(), dtype = np.float64)
                    # tpu_data = np.array(merged_data.tpu_metric.dropna().tolist(), dtype = np.float64)
                    models = np.array(df1_values[batch_size]['model_name'].unique())

                    cur_range = np.arange(0, df1_data.shape[0])

                    if islog:
                        l1 = ax.bar(cur_range + shift, abs(np.log10(abs(df1_data))), color=color, width = .125, label = f'{batch_size}')
                    else:
                        l1 = ax.bar(cur_range + shift, abs(df1_data), color=color, width = .125, label = f'BS: {int(batch_size)}')
                axv_line_parameters = {'linewidth': 2, 'color': 'gray', 'linestyle': (0, (4, 1.2))}

                if plotting_type[2] == 0 or plotting_type[2] == 2:
                    axv_line_x_coordinates = [i - 0.5 for i in range(cur_range[0], cur_range[-1] + 2, 2)]
                elif plotting_type[2] == 1:
                    axv_line_x_coordinates = [i - 0.5 for i in range(cur_range[0], cur_range[-1] + 2, 4)]

                for x in axv_line_x_coordinates:
                    ax.axvline(x, **axv_line_parameters)
                
                if metric_label == 'Speed Up':
                    ax.axhline(y=1, linestyle='--', color='violet', label = 'speed up = 1')

                xticks_array = []
                if plotting_type[2] == 0 or plotting_type[2] == 2:
                    for i, model in enumerate(models):
                        xticks_array.append(f'fp32\n{model}')
                        xticks_array.append('fp64')
                elif plotting_type[2] == 1:
                    for i, model in enumerate(models):
                        xticks_array.append('fp16')
                        xticks_array.append('bfp16')
                        xticks_array.append(f'fp32\n{model}')
                        xticks_array.append('fp64')
                    
                ax.set_xticks(cur_range, xticks_array)
                ax.legend()
                ax.set_xlabel('Batch Size')
                if islog:
                    ax.set_ylabel(f"log({metric_label})")
                else:
                    ax.set_ylabel(metric_label)
                ax.set_title(f'{model_cat}')
                ax.tick_params(axis='both', which='major', labelsize=6)

        fig.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()

# plotting_type0 : one data / two data
# plotting_type1 : batch size or floating point
# plotting_type2: GPU/CPU, GPU/TPU, TPU/CPU // CPU, GPU, TPU

# ############# Plotting Utilization CPUv1 ###############
# plot('cpu_data_p1.csv', None,
#      'cpu_utilization',
#      'CPUv1 Utilization',
#      512, None,
#      [0, 0, 0], False)

# plot('cpu_data_p1.csv', None,
#      'cpu_utilization',
#      'CPUv1 Utilization',
#      512, None,
#      [0, 1, 0], False)

# ############## Plotting CPUv1 Memory Usage ###############
# plot('cpu_data_p1.csv', None,
#      'cpu_memory_used',
#      'CPUv1 Memory Used (MB)',
#      512, None,
#      [0, 0, 0], False)

# plot('cpu_data_p1.csv', None,
#      'cpu_memory_used',
#      'CPUv1 Memory Used (MB)',
#      512, None,
#      [0, 1, 0], False)

# ############## Plotting CPUv2 Memory Usage ###############
# plot('cpu_data_p2.csv', None,
#      'cpu_memory_used',
#      'CPUv2 Memory Used (MB)',
#      512, None,
#      [0, 0, 0], False)

# plot('cpu_data_p2.csv', None,
#      'cpu_memory_used',
#      'CPUv2 Memory Used (MB)',
#      512, None,
#      [0, 1, 0], False)

# ############# Plotting Utilization GPU ###############
# plot('gpu_data.csv', None,
#      'gpu_utilization',
#      'GPU Utilization',
#      512, None,
#      [0, 0, 1], False)

# plot('gpu_data.csv', None,
#      'gpu_utilization',
#      'GPU Utilization',
#      512, None,
#      [0, 1, 1], False)

# ############## Plotting GPU Memory Usage ###############
# plot('gpu_data.csv', None,
#      'gpu_mem_used',
#      'GPU Memory Used (MB)',
#      512, None,
#      [0, 0, 1], False)

# plot('gpu_data.csv', None,
#      'gpu_mem_used',
#      'GPU Memory Used (MB)',
#      512, None,
#      [0, 1, 1], False)

# ############## Plotting GPU Power Consumption ###############
# plot('gpu_data.csv', None,
#      'gpu_power',
#      'GPU Power Consumption',
#      512, None,
#      [0, 0, 1], False)

# plot('gpu_data.csv', None,
#      'gpu_power',
#      'GPU Power Consumption',
#      512, None,
#      [0, 1, 1], False)

# ############## Plotting Infereince Times GPU vs TPU ###############
# plot('gpu_data.csv',
#      'tpu_data.csv',
#      'inference_time', 
#      'Speed Up (GPU/TPU)',
#      512, 10,
#      [1,0,1], False)

# plot('gpu_data.csv',
#      'tpu_data.csv',
#      'inference_time', 
#      'Speed Up (GPU/TPU)',
#      512, 10,
#      [1,1,1], False)

############## Plotting Infereince Times CPUv2 vs TPU ###############
plot('tpu_data.csv',
     'cpu_data_p2.csv',
     'inference_time', 
     'Speed Up (TPU/CPUv2)',
     10, 512,
     [1,0,2], False)

plot('tpu_data.csv',
     'cpu_data_p2.csv',
     'inference_time', 
     'Speed Up (TPU/CPUv2)',
     10, 512,
     [1,1,2], False)

############## Plotting Infereince Times CPUv2 vs GPU ###############
plot('gpu_data.csv',
     'cpu_data_p2.csv',
     'inference_time', 
     'Speed Up (GPU/CPUv2)',
     512, 512,
     [1,0,0], False)

plot('gpu_data.csv',
     'cpu_data_p2.csv',
     'inference_time', 
     'Speed Up (GPU/CPUv2)',
     512, 512,
     [1,1,0], False)

############## Plotting Infereince Times CPUv2 vs CPUv1 ###############
plot('cpu_data_p2.csv',
     'cpu_data_p1.csv',
     'inference_time', 
     'Speed Up (CPUv2/CPUv1)',
     512, 512,
     [1,0,0], False)

plot('cpu_data_p2.csv',
     'cpu_data_p1.csv',
     'inference_time', 
     'Speed Up (CPUv2/CPUv1)',
     512, 512,
     [1,1,0], False)

############## Plotting Communication Times GPU vs TPU ###############

plot('tpu_data.csv', 
     'gpu_data.csv',
     'communication_time', 
     'Communication Time Ratio (TPU/GPU)',
     10, 512,
     [1,0,1], True)

plot('tpu_data.csv', 
     'gpu_data.csv',
     'communication_time', 
     'Communication Time Ratio (TPU/GPU)',
     10, 512,
     [1,1, 1], True)