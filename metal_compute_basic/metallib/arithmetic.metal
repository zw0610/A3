//
//  arithmetic.metal
//  arithmetic
//
//  Created by Wang Zhang on 3/12/20.
//  Copyright © 2020 Wang Zhang. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}
