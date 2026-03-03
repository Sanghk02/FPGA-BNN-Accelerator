#ifndef PARAMS_H
#define PARAMS_H
#include "ap_int.h"

// ==============================================
// Parameters for BNN Accelerator Design
// ==============================================
#define BATCH_SIZE 10000

#define IN_H 28
#define IN_W 28

#define CONV1_OUT_CH 16
#define CONV2_OUT_CH 32
#define CONV3_OUT_CH 32
#define POOL_OUT_CH 32

#define KERNEL_SIZE 3
#define FC_OUT 10

// Feature map 
#define C1_H (IN_H - KERNEL_SIZE + 1)       // 26
#define C1_W (IN_W - KERNEL_SIZE + 1)       // 26
#define C2_H (C1_H - KERNEL_SIZE + 1)       // 24
#define C2_W (C1_W - KERNEL_SIZE + 1)       // 24
#define C3_H (C2_H - KERNEL_SIZE + 1)       // 22
#define C3_W (C2_W - KERNEL_SIZE + 1)       // 22
#define POOL_OUT_H (C3_H / 2)               // 11
#define POOL_OUT_W (C3_W / 2)               // 11
#define FLAT_SIZE (CONV3_OUT_CH * POOL_OUT_H * POOL_OUT_W) // 3872

#endif // PARAMS_H
