# Plotting Library 

## Requirements

This Python plotting library requires the following packages to be installed:

- `numpy`
- `pandas`
- `matplotlib`

## Usage

To use the library, follow these steps:

1. Process the output log files:
   - Place the output log files from different devices into their respective folders in the `final_data` folder.
   - Run the `processlogs.py` file in the plotting library.
   - To run this file, make sure you are in the plot folder and then run the following command:
     ```
     python3 plottinglib\processlogs.py
     ```
   - This will generate the files `cpu_data_p1.csv`, `cpu_data_p2.csv`, `gpu_data.csv`, and `tpu_data.csv` that will be used for plotting later on.
2. Use the plotting functions to plot various plots:
   - Run the `paper_plots.py` file in the plotting library.
   - To run this file, make sure you are in the plot folder and then run the following command:
     ```
     python3 plottinglib\paper_plots.py
     ```
   - This will generate a series of plots comparing various metrics between CPU, GPU, and TPUs.

## License

This library is released under the MIT License. See [LICENSE](LICENSE) for details.
