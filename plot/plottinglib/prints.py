
import config
import pandas as pd
import numpy as np

df2 = pd.read_csv('cpu_data_p2.csv')
df3 = pd.read_csv('gpu_data.csv')
df4 = pd.read_csv('tpu_data.csv')

df2 = df2[df2.num_inference_steps == 512].dropna()
df3 = df3[df3.num_inference_steps == 512].dropna()
df4 = df4[df4.num_inference_steps == 10].dropna()

###################### Average Inference Time and Overall Speedup ######################
df2_mean = np.mean(df2.inference_time.dropna())
df3_mean = np.mean(df3.inference_time.dropna())
df4_mean = np.mean(df4.inference_time.dropna())

print(f"GPU Vs CPUv2 Speed Up : {df2_mean/df3_mean:.2f}")
print(f"TPU Vs CPUv2 Speed Up : {df2_mean/df4_mean:.2f}")
print(f"GPU Vs TPU Speed Up : {df4_mean/df3_mean:.2f}")

###################### GPU Power Consumption with batch size ######################
df3_bs1 = df3[df3.batch_size == 16].dropna()
df3_bs2 = df3[df3.batch_size == 32].dropna()
df3_bs3 = df3[df3.batch_size == 64].dropna()

df3_bs1 = np.mean(df3_bs1.gpu_power.dropna())
df3_bs2 = np.mean(df3_bs2.gpu_power.dropna())
df3_bs3 = np.mean(df3_bs3.gpu_power.dropna())

print(f"From batch size 16 to batch size 32 power consumption becomes: {df3_bs2/df3_bs1:.2f}x times")
print(f"From batch size 32 to batch size 64 power consumption becomes: {df3_bs3/df3_bs2:.2f}x times")
