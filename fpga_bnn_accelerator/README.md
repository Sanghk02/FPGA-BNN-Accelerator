# FPGA BNN Accelerator for MNIST

This project implements a high-performance **Binary Neural Network (BNN)** accelerator on the **PYNQ-Z2 (Zynq-7000)** FPGA. It features optimized hardware architectures for binary operations, achieving high throughput and low latency.

## Key Features
- **XNOR + Popcount Optimization**: Replaces expensive floating-point MAC operations with bitwise operations.
- **Hardware-Friendly Architecture**: Utilizes **Line Buffer** and **FIFO** for efficient data streaming and high-throughput convolution.
- **HLS-Based Design**: Written in C++ using Vivado HLS with aggressive pragmas (`DATAFLOW`, `PIPELINE`, `ARRAY_PARTITION`).
- **End-to-End Workflow**: Includes PyTorch training (using STE), HLS synthesis, and PYNQ deployment.

## Hardware Architecture
The accelerator is designed to minimize off-chip memory access by leveraging:
- **Bit-packing**: 9-bit weights and inputs are packed to maximize bandwidth.
- **Parallel Processing**: Parallel XNOR-Popcount engines for convolutional layers.
- **Stream-based Pipeline**: Optimized for the AXI-Stream interface.

## Performance
- **Target Board**: PYNQ-Z2 (Xilinx XC7Z020)
- **Clock Frequency**: 100 MHz
- **Accuracy**: ~94% on MNIST (Top-1)
- **Key Optimization**: Achieved significant speedup using manual line buffering compared to naive implementations.

## Repository Structure
.
├── hls/  \\
│   ├── bnn_stream_accel.cpp   \\
│   ├── bnn_stream_accel.h      \\
│   └── params.h   \\
├── training/  \\
│   └── learning.ipynb  \\   
├── pynq/ \\
│   ├── bitstream/  \\
│   └── BNN_LineBuffer_FIFO.ipynb \\
├── include/ \\
│   └── weights_94_packed_channel.h  # 최적화된 채널 패킹 가중치  \\
└── README.md

## Usage

### 1. Training & Export
Run `training/learning.ipynb` to train the BNN. The weights are quantized using Straight-Through Estimator (STE) and exported as `.h` files.

### 2. HLS Synthesis
Open Vivado HLS and import files in the `hls/` directory. 
- Run C-Simulation to verify logic.
- Run C-Synthesis to generate IP.

### 3. PYNQ Deployment
Copy the `.bit`, `.hwh` files and `pynq/inference.ipynb` to your PYNQ-Z2 board. Run the notebook to see the accelerator in action!

## References

- BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.


