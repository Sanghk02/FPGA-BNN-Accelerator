/**
 * @file bnn_stream_accel.cpp
 * @brief High-performance Streaming BNN Accelerator Implementation
 * @details This file implements a Binary Neural Network (BNN) accelerator 
 * optimized for FPGA using Xilinx Vivado HLS. It features AXI-Stream 
 * interfaces, Line Buffering, and XNOR-Popcount operations.
 */

#include "bnn_stream_accel.h"
#include <string.h>

/**
 * @brief 2x2 Average Pooling (Majority Voting for BNN)
 * @details Implements 2x2 pooling for binary activations using 3-out-of-4 
 * majority logic to preserve feature information in the binary domain.
 * @param in_stream  Input activation stream (32-bit packed)
 * @param out_stream Output pooled stream (32-bit packed)
 */
void avgpool2d_stream(hls::stream<axis_32bit_t>& in_stream, hls::stream<axis_32bit_t>& out_stream) {
    // Line buffer: Maintains two rows of feature maps to perform 2x2 pooling window
    ap_uint<32> linebuf[2][C3_W] = {0};
    #pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1

    for (ap_uint<16> img = 0; img < BATCH_SIZE; img++) {
        for (ap_uint<5> h = 0; h < C3_H; h++) {
            for (ap_uint<5> w = 0; w < C3_W; w++) {
                #pragma HLS PIPELINE II=1
                ap_uint<32> px = in_stream.read();
                
                // Line buffer shifting logic for sliding window
                linebuf[0][w] = linebuf[1][w];
                linebuf[1][w] = px;

                // Process only at the end of each 2x2 window stride
                if (h % 2 == 1 && w % 2 == 1) {
                    // Extract 2x2 patch: Top-Left, Top-Right, Bottom-Left, Bottom-Right
                    ap_uint<32> tl = linebuf[0][w-1];
                    ap_uint<32> tr = linebuf[0][w];
                    ap_uint<32> bl = linebuf[1][w-1];
                    ap_uint<32> br = linebuf[1][w];

                    // 3-out-of-4 Majority Voting: If 3 or more pixels are 1, output is 1
                    // Optimized using bitwise operations for all 32 channels simultaneously
                    ap_uint<32> pooled = (tl & tr & bl) | (tl & tr & br) | (tl & bl & br) | (tr & bl & br);
                    out_stream.write(pooled);
                }
            }
        }
    }
}

/**
 * @brief Flatten and Fully-Connected Layer (XNOR-Popcount)
 * @details Computes FC layer using XNOR and Popcount to replace standard MAC operations.
 * The input is flattened from the pooling output.
 * @param in_stream  Input pooled stream (32-bit packed)
 * @param out_stream Output classification result (AXI-Stream with LAST signal)
 */
void flatten_fc_stream(hls::stream<axis_32bit_t>& in_stream, hls::stream<axis_t>& out_stream) {
    const int CHUNKS = FLAT_SIZE / 32;

    for (ap_uint<16> img = 0; img < BATCH_SIZE; img++) {
        // Internal buffer for flattened features
        ap_uint<32> flat[CHUNKS];
        #pragma HLS ARRAY_PARTITION variable=flat complete

        // Load data from stream to on-chip memory for parallel access
        for (ap_uint<7> i = 0; i < CHUNKS; i++) {
            #pragma HLS PIPELINE II=1
            flat[i] = in_stream.read();
        }

        ap_uint<12> fc_out[FC_OUT];

        // FC layer computation using bitwise XNOR and Popcount (MAC equivalent)
        for (ap_uint<4> o = 0; o < FC_OUT; o++) {
            ap_uint<12> acc = 0;
            for (ap_uint<7> i = 0; i < CHUNKS; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS unroll factor=11

                ap_uint<32> input32 = flat[i];
                // Perform bitwise XNOR for 32 channels and count matches
                ap_uint<32> xnor32 = ~(input32 ^ fc1_weight[o][i]);
                acc += popcount32(xnor32);
            }
            fc_out[o] = acc; // Stored without thresholding to get raw confidence score

            // Package output as AXI-Stream packet
            axis_t pkt;
            pkt.data = (ap_uint<32>)fc_out[o];
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            // Set LAST signal at the final class of the final image
            pkt.last = (img == BATCH_SIZE-1 && o == FC_OUT-1) ? 1 : 0;
            out_stream.write(pkt);
        }
    }
}

/**
 * @brief Input Binarization Layer (32-bit Float to 1-bit Binary)
 * @details Converts floating-point pixel data to binary values (0 or 1) 
 * based on a fixed threshold (0.5f).
 */
void input_binarize_stream(hls::stream<axis_t>& input_stream, hls::stream<axis_1bit_t>& bin_stream) {
    for (ap_uint<16> img = 0; img < BATCH_SIZE; img++) {
        for (ap_uint<10> i = 0; i < IN_H * IN_W; i++) {
            #pragma HLS PIPELINE II=1

            axis_t pkt = input_stream.read();
            float f;
            ap_uint<32> raw = pkt.data;
            // Interpret bit-pattern as float using memory copy
            memcpy(&f, &raw, sizeof(float));
            // Thresholding to binarize input features
            bin_stream.write((f > 0.5f) ? 1 : 0);
        }
    }
}

/**
 * @brief Top-Level BNN Accelerator Engine
 * @details Top function of the design utilizing HLS DATAFLOW for pipeline parallelism.
 * Each layer runs concurrently, forming a deep hardware pipeline with AXI-Stream.
 * @param input_stream  Incoming AXI-Stream (floating point MNIST images)
 * @param output_stream Outgoing AXI-Stream (classification results)
 */
void BNN(hls::stream<axis_t>& input_stream, hls::stream<axis_t>& output_stream) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
    
    // Enable task-level pipelining to increase overall throughput
    #pragma HLS DATAFLOW

    // Internal streams between layers (FIFO buffers for dataflow)
    hls::stream<axis_1bit_t, 1024> bin_stream;
    hls::stream<axis_16bit_t, 1024> c1_stream;
    hls::stream<axis_32bit_t, 1024> c2_stream;
    hls::stream<axis_32bit_t, 1024> c3_stream;
    hls::stream<axis_32bit_t, 1024> pool_stream;

    // 1. Convert float input to binary stream
    input_binarize_stream(input_stream, bin_stream);
    
    // 2. Convolutional Neural Network (CNN) feature extraction stages
    conv_layer1_stream<1, CONV1_OUT_CH, IN_H, IN_W, C1_H, C1_W>(bin_stream, c1_stream, conv1_weight);
    conv_layer2_stream<CONV1_OUT_CH, CONV2_OUT_CH, C1_H, C1_W, C2_H, C2_W>(c1_stream, c2_stream, conv2_weight);
    conv_layer3_stream<CONV2_OUT_CH, CONV3_OUT_CH, C2_H, C2_W, C3_H, C3_W>(c2_stream, c3_stream, conv3_weight);

    // 3. Dimensionality reduction and classification
    avgpool2d_stream(c3_stream, pool_stream);
    flatten_fc_stream(pool_stream, output_stream);
}