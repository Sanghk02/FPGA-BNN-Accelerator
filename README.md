# FPGA-BNN-Accelerator

This repository implements a streaming Binary Neural Network (BNN) accelerator for MNIST on FPGA (PYNQ-Z2 / Zynq-7000).

The main goal is not only to run a BNN, but to make the design hardware-efficient: reduce memory traffic, simplify arithmetic, and maximize throughput with a stream-first architecture.


## What this project is trying to solve

Porting a standard CNN directly to FPGA often leads to:

- expensive floating-point MAC-heavy compute,
- high intermediate feature-map memory traffic,
- and bottlenecks in inter-layer data movement.

This project addresses those issues with four design choices:

1. 1-bit quantization for weights and activations (BNN),
2. XNOR + popcount instead of multiply-accumulate,
3. AXI-streamed layer pipeline for continuous dataflow,
4. line buffer and FIFO-based window processing to reduce off-chip accesses.

## Repository structure

```text
.
├── hls/
│   ├── bnn_stream_accel.h           # stream/data types, interfaces, popcount utilities
│   ├── bnn_stream_accel.cpp         # binarization, pooling, FC, top-level BNN
│   └── params.h                     # model and feature-map parameters
├── include/
│   └── weights_94_packed_channel.h  # channel-packed trained weights for hardware
├── training/
│   └── learning.ipynb               # QAT training and weight export
├── pynq/
│   ├── BNN_LineBuffer_FIFO.ipynb    # overlay load, DMA inference, measurements
│   └── bitstream/
│       ├── bnn_accel.bit
│       └── bnn_accel.hwh
└── README.md
```

## End-to-end workflow

The repository is intended as one connected flow:

1. `training/learning.ipynb`
   - Trains a QAT-style BNN using STE-based binary operators,
   - Exports trained parameters into HLS-friendly C header format.

2. `hls/`
   - Implements the hardware kernel in Vivado HLS C++,
   - Defines streamed conv/pool/FC stages,
   - Uses pragmas such as `DATAFLOW`, `PIPELINE`, and `ARRAY_PARTITION`.

3. `pynq/BNN_LineBuffer_FIFO.ipynb`
   - Loads the generated bitstream on PYNQ-Z2,
   - Runs DMA-based inference,
   - Measures accuracy and runtime.

In short, this repo ties together training code, hardware kernel code, and on-board execution code in a single pipeline.


## Model and hardware intent

### Network shape

Based on `hls/params.h`:

- Input: 28x28 (MNIST)
- Conv1: 16 output channels, 3x3
- Conv2: 32 output channels, 3x3
- Conv3: 32 output channels, 3x3
- 2x2 pooling
- FC: 10 classes

Dimensions are fixed with compile-time parameters to allow stronger static optimization during HLS synthesis.


### Core operator ideas

- Input binarization: float AXI input is thresholded at 0.5 into 1-bit activations.
- Convolution and FC: XNOR computes bit matches, popcount accumulates similarity.
- Pooling in binary domain: 2x2 majority-vote logic is used for binary activations.

This is a hardware-oriented implementation strategy, not a direct floating-point software model port.


## Novelty

1) **1-bit {-1, +1} quantization mapped to {0, 1} in hardware**  
   All layers operate in binary form using `ap_uint<1>` to minimize memory footprint.  
   Multiplication is replaced with **XNOR + popcount**, significantly improving computational efficiency and speed.

2) **Removal of redundant arithmetic via threshold-based comparison**  
   Expressions such as `2 * popcount - N` are eliminated.  
   Instead, a single threshold comparison is used, reducing arithmetic overhead and simplifying hardware logic.

3) **Input-channel pre-bitpacking**  
   Input channels are pre-packed before convolution.  
   - No inner unpacking loop is required during computation.  
   - Achieves an effect similar to full loop unrolling without increasing resource usage.  
   - Enables efficient FIFO transfer (e.g., 32-bit packed streaming per cycle).  
   - Eliminates runtime unpacking overhead.

4) **DATAFLOW and stream-based pipelining (FIFO architecture)**  
   Computation starts immediately when a patch is generated and is streamed forward.  
   - `#pragma HLS DATAFLOW` and streaming FIFOs enable deep pipelining.  
   - Achieves **II = 1** in steady state.  
   - After pipeline warm-up, one pixel (feature element) is produced per cycle.

5) **Bitwise majority-based pooling (3-out-of-4 logic)**  
   Average pooling is replaced with a 3-of-4 majority vote in the binary domain.  
   This reduces arithmetic complexity while preserving decision behavior in BNN.

6) **Bitwidth-optimized loop variables and indices**  
   Loop counters and control variables use the minimum required bitwidth (e.g., `ap_uint<4>` instead of `int`).  
   This significantly reduces MUX complexity and overall control logic resource usage.

---

## How to run


### 1) Train and export weights

- Run `training/learning.ipynb` to train the BNN.
- Use the notebook conversion/export code to generate HLS weight headers.
- Update `include/weights_94_packed_channel.h` with exported packed weights.


### 2) Synthesize with Vivado HLS

- Create an HLS project and add files in `hls/`.
- Set top function to `BNN`.
- Run C simulation and C synthesis to generate IP.


### 3) Deploy and validate on PYNQ

- Place `.bit` and `.hwh` on the board.
- Run `pynq/BNN_LineBuffer_FIFO.ipynb` for overlay loading and DMA inference.
- Check output accuracy and timing.




