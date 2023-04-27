# Benchmarking CPU, GPU, and TPU

## Introduction
The increasing use of deep learning models for various applications necessitates efficient and fast computation. Therefore, the evaluation and comparison of different computing platforms' performance is essential to understand their strengths and limitations. This study focuses on benchmarking the performance of CPU, GPU, and TPU platforms using pre-trained ImageNet models. The results provide insights into each platform's strengths and weaknesses. They help in making informed decisions regarding the selection of the appropriate platform for specific applications. Benchmarking deep learning platforms has several challenges. For instance, the choice of software framework, optimization techniques, and hyperparameters can significantly affect the results. This study discusses the challenges associated with benchmarking deep learning platforms and makes recommendations for future research.

## Methodology
Our goal is to perform inference on pre-trained ImageNet models using CPU, GPU, and TPU devices. This is accomplished by developing a Python framework that generates random image data according to specified batch sizes and precisions. Using this framework, we feed pre-trained models with generated image data and record various device metrics, such as inference time, communication time, and device utilization. We investigate each device's performance under different conditions by varying several key parameters. 

- We evaluate the performance of CPU, GPU, and TPU devices for conducting inference on pre-trained ImageNet models.
- We vary key parameters, including batch size, data precision, and number of inference steps, to assess how each device performs under different conditions.
- We use the same pre-trained ImageNet models and data for each test to isolate the impact of hardware on model performance.

| Variations             | CPU                     | GPU                                         | TPU                                         |
|-----------------------|-------------------------|---------------------------------------------|---------------------------------------------|
| Data Precision         | float32, float64        | float16, bfloat16, float32, float64          | float16, bfloat16, float32, float64          |
| Number of Inference Steps | 512,1024             | 512,1024                                    | 32                                          |
| Batch Size             | 16,32,64                | 16,32,64                                    | 16,32,64                                    |

![Methodology Image](./plot/other_plots/diagram_metrics.png "Steps for measuring metrics for CPU, TPU, and GPU")

## Hardware Platforms
Our selection of hardware platforms varies widely on the devices that are available to us when the following experiments are done. 

**CPU Platform1 (CPUv1)** 

This CPU is a Intel(R) Core(TM) i7-8750H CPU. It has 6 cores and 12 threads. It has 16GB DDR4 memory and a peak FLOPS of 2.2 GHz x 6 cores x 2 FLOPS/cycle = 26.4 GFLOPS. 

**CPU Platform2 (CPUv2)** 

This CPU is an AMD EPYC 7763 64-Core Processor and is a server-grade CPU based on the Zen 3 architecture. It has 64 cores and 128 threads. It has 256GB DDR-4 and a peak FLOPS of 4.95 teraflops (single precision) and 2.475 teraflops (double precision). 

**GPU Platform**

This GPU is an NVIDIA A100-PCIE-40GB a high-performance GPU designed for use in data centers and scientific computing applications. It has 40GB HBM2 GPU memory, with high-bandwidth memory operating at up to 2TB/s. It has a peak performance of 9.7 TFlops in double precision and 312 TFlops in FP16 precision using the Tensor Cores. It has the AMD EPYC 7763 64-Core Processor as the host which communicates with the GPU. 

**TPU Platform**

This is the Google collab TPU platform that is accessible to the public. We use the TPU v2 of the google collab TPUs \cite{b3} which supports 45 TFLOPS (mixed precision) and 2 cores. Total ML acceleration for a Cloud TPU v2 platform is 180 TFLOPS. Memory size is 8 GB per core, or 64 GB per board, with 2400 GB/s overall memory bandwidth.

| Platform | Mem Type | Memory (GB) | Mem Bdw | Peak FLOPS |
| --- | --- | --- | --- | --- |
| CPUv1 | DDR4 | 16 | 41.8 GB/s | 26.4G SP* |
| CPUv2 | DDR4 | 256 | 204.8 GB/s | 4.95T SP* |
| GPU | HBM2 | 40 | 2 TB/s | 9.7T DP** |
| TPU v2 | HBM | 8 | 2400 GB/s | 180T |

\* SP - Single Precision, ** DP - Double Precision

## Results

### CPUv1 to CPUv2

#### Inference time and Speed up
![F1](./plot/Final_Plots/cpuv2_cpuv1_inf_bs.png "Inference time CPUv1 vs CPUv2")
#### CPU Utilization and Memory Usage
![F2](./plot/Final_Plots/cpuv1_util_bs.png "CPUv1 Utilization")
![F3](./plot/Final_Plots/cpuv1_mem_pres.png "CPUv1 Memory Usage")
![F4](./plot/Final_Plots/cpuv2_mem_pres.png "CPUv2 Memory Usage")
