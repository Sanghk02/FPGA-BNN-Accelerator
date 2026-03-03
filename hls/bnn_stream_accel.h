/**
 * @file bnn_stream_accel.h
 * @brief Header file for Streaming Binary Neural Network (BNN) Accelerator.
 * @details This file defines the hardware architecture for a BNN optimized 
 * for FPGA deployment. It includes streaming convolution layers with line buffers,
 * hierarchical popcount units, and AXI-Stream interface definitions.
 * * Target Hardware: PYNQ-Z2 (Zynq-7000)
 * Optimization: Dataflow, Pipelining, and Bit-level parallelism.
 */

#ifndef BNN_STREAM_ACCEL_H
#define BNN_STREAM_ACCEL_H

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

#include "params.h"
#include "weights_94_packed_channel.h"

/**
 * @brief AXI-Stream Sideband Signals Data Structure.
 * @details Standard structure for AXI4-Stream interface including LAST signal 
 * for TLAST generation and KEEP/STRB for byte-enable.
 */
typedef struct {
    ap_uint<32> data;  ///< 32-bit payload
    ap_uint<4>  keep;  ///< Byte qualification
    ap_uint<4>  strb;  ///< Data position qualification
    ap_uint<1>  last;  ///< End of stream indicator
} axis_t;

/* Packed Bitstream Types for Channel Parallelism */
typedef ap_uint<1>  axis_1bit_t;  ///< Single bit activation
typedef ap_uint<16> axis_16bit_t; ///< 16-channel packed activation
typedef ap_uint<32> axis_32bit_t; ///< 32-channel packed activation

/**
 * @name Pre-trained Quantized Weights
 * @brief Extern declarations for weights extracted from PyTorch training.
 * Weights are packed to match the hardware processing parallelism.
 */
///@{
extern const ap_uint<1>  conv1_weight[CONV1_OUT_CH][9];
extern const ap_uint<16> conv2_weight[CONV2_OUT_CH][9];
extern const ap_uint<32> conv3_weight[CONV3_OUT_CH][9];
extern const ap_uint<32> fc1_weight[FC_OUT][FLAT_SIZE / 32];
///@}

/**
 * @brief 4-bit Population Count using Look-Up Table (LUT).
 * @param x 4-bit input bit-vector.
 * @return Number of set bits (1s).
 */
inline int popcount4(ap_uint<4> x) {
    const int LUT[16] = {
        0, 1, 1, 2, 1, 2, 2, 3,
        1, 2, 2, 3, 2, 3, 3, 4
    };
    return LUT[x];
}

/**
 * @brief 16-bit Population Count with Tree Adder Structure.
 * @details Implements a hierarchical adder tree to minimize critical path 
 * and maximize clock frequency in FPGA hardware.
 * @param x 16-bit input bit-vector.
 * @return 5-bit sum (0 to 16).
 */
inline ap_uint<5> popcount16(ap_uint<16> x) {
    #pragma HLS INLINE
    ap_uint<3> acc[4];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    for (ap_uint<3> i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        acc[i] = popcount4(x.range(i * 4 + 3, i * 4));
    }

    ap_uint<4> sum0 = acc[0] + acc[1];
    ap_uint<4> sum1 = acc[2] + acc[3];

    return (ap_uint<5>)(sum0 + sum1);
}

/**
 * @brief 32-bit Population Count with Tree Adder Structure.
 * @details Highly parallel implementation for 32-channel bitwise XNOR-Popcount.
 * @param x 32-bit input bit-vector.
 * @return 6-bit sum (0 to 32).
 */
inline ap_uint<6> popcount32(ap_uint<32> x) {
    #pragma HLS INLINE
    ap_uint<3> acc[8];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    for (ap_uint<4> i = 0; i < 8; i++) {
        #pragma HLS UNROLL
        acc[i] = popcount4(x.range(i * 4 + 3, i * 4));
    }

    ap_uint<4> sum0 = acc[0] + acc[1];
    ap_uint<4> sum1 = acc[2] + acc[3];
    ap_uint<4> sum2 = acc[4] + acc[5];
    ap_uint<4> sum3 = acc[6] + acc[7];

    ap_uint<5> part0 = sum0 + sum1;
    ap_uint<5> part1 = sum2 + sum3;

    return (ap_uint<6>)(part0 + part1);
}

/**
 * @brief Streaming Convolution Layer Template with Line Buffer.
 * @tparam IN_CH   Input channel bit-packing factor.
 * @tparam OUT_CH  Output channel bit-packing factor.
 * @tparam INPUT_H Input feature map height.
 * @tparam INPUT_W Input feature map width.
 * @tparam OUT_H   Output feature map height.
 * @tparam OUT_W   Output feature map width.
 * * @param in_stream  Input activation stream.
 * @param out_stream Output activation stream.
 * @param weight     Constant weight array (quantized).
 * * @details This template implements a 3x3 convolution using a 3-line buffer
 * to enable sliding window processing without re-reading from off-chip memory.
 * Optimized with HLS PIPELINE and ARRAY_PARTITION for high throughput.
 */
template<int IN_CH, int OUT_CH, int INPUT_H, int INPUT_W, int OUT_H, int OUT_W>
void conv_layer1_stream(hls::stream<axis_1bit_t>& in_stream, 
                        hls::stream<axis_16bit_t>& out_stream, 
                        const ap_uint<IN_CH> weight[OUT_CH][9]);

template<int IN_CH, int OUT_CH, int INPUT_H, int INPUT_W, int OUT_H, int OUT_W>
void conv_layer2_stream(hls::stream<axis_16bit_t>& in_stream, 
                        hls::stream<axis_32bit_t>& out_stream, 
                        const ap_uint<IN_CH> weight[OUT_CH][9]);

template<int IN_CH, int OUT_CH, int INPUT_H, int INPUT_W, int OUT_H, int OUT_W>
void conv_layer3_stream(hls::stream<axis_32bit_t>& in_stream,
                        hls::stream<axis_32bit_t>& out_stream,
                        const ap_uint<IN_CH> weight[OUT_CH][9]);

/**
 * @brief Global Average Pooling for Binary Activations.
 * @details Uses majority voting logic to reduce spatial dimensions 
 * while maintaining binary precision.
 */
void avgpool2d_stream(hls::stream<axis_32bit_t>& in_stream, 
                      hls::stream<axis_32bit_t>& out_stream);

/**
 * @brief Flatten and Fully Connected Layer.
 * @details Computes final classification scores using XNOR-Popcount logic 
 * on flattened feature maps.
 */
void flatten_fc_stream(hls::stream<axis_32bit_t>& in_stream, 
                       hls::stream<axis_t>& out_stream);

/**
 * @brief Top-Level BNN Accelerator Entry Point.
 * @param input_stream  AXI-Stream input (typically from DMA).
 * @param output_stream AXI-Stream output (classification result).
 */
void BNN(hls::stream<axis_t>& input_stream, hls::stream<axis_t>& output_stream);

#endif // BNN_STREAM_ACCEL_H