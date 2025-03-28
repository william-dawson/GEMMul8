#pragma once
#include "common.hpp"
#include "table.hpp"

namespace oz2_util {

__global__ void conv_32i_2_8u_256_kernel(const size_t sizeC,                         // ((m * n + 15) >> 4) << 4; // multiple of 16
                                         const size_t m,                             //
                                         const int32_t *const __restrict__ C32i,     // input
                                         const int32_t *const __restrict__ correctA, //
                                         const int32_t *const __restrict__ correctB, //
                                         uint8_t *const __restrict__ C8u)            // output
{
    auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= sizeC) return;

    uchar4 *out = reinterpret_cast<uchar4 *>(C8u + idx);
    int4 in     = *reinterpret_cast<const int4 *>(C32i + idx);

    auto col     = idx / m;
    auto row     = idx - col * m;
    int32_t corA = correctA[row];
    int32_t corB = correctB[col];
    in.x += corA + corB;

    row++;
    if (row == m) {
        row = 0;
        col++;
        corA = correctA[row];
        corB = correctB[col];
    } else {
        corA = correctA[row];
    }
    in.y += corA + corB;

    row++;
    if (row == m) {
        row = 0;
        col++;
        corA = correctA[row];
        corB = correctB[col];
    } else {
        corA = correctA[row];
    }
    in.z += corA + corB;

    row++;
    if (row == m) {
        row = 0;
        col++;
        corA = correctA[row];
        corB = correctB[col];
    } else {
        corA = correctA[row];
    }
    in.w += corA + corB;

    uchar4 result;
    result.x = static_cast<uint8_t>(in.x);
    result.y = static_cast<uint8_t>(in.y);
    result.z = static_cast<uint8_t>(in.z);
    result.w = static_cast<uint8_t>(in.w);

    *out = result;
}

__global__ void conv_32i_2_8u_not256_kernel(const size_t sizeC,                         // m*n
                                            const size_t m,                             //
                                            const int32_t *const __restrict__ C32i,     // input
                                            const uint8_t modulus,                      //
                                            const int32_t invm,                         // 2^32 / modulus
                                            const int32_t k_mod_m,                      //
                                            const int32_t *const __restrict__ correctA, //
                                            const int32_t *const __restrict__ correctB, //
                                            uint8_t *const __restrict__ C8u)            // output
{
    auto idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= sizeC) return;

    uchar4 *out = reinterpret_cast<uchar4 *>(C8u + idx);
    int4 in     = *reinterpret_cast<const int4 *>(C32i + idx);

    auto col     = idx / m;
    auto row     = idx - col * m;
    int32_t corA = correctA[row];
    int32_t corB = correctB[col] + k_mod_m;
    in.x += corA + corB;
    in.x -= __mulhi(in.x, invm) * modulus;
    in.x -= (in.x >= modulus) * modulus;
    in.x += (in.x < 0) * modulus;

    row++;
    if (row == m) {
        row = 0;
        col++;
        corA = correctA[row];
        corB = correctB[col] + k_mod_m;
    } else {
        corA = correctA[row];
    }
    in.y += corA + corB;
    in.y -= __mulhi(in.y, invm) * modulus;
    in.y -= (in.y >= modulus) * modulus;
    in.y += (in.y < 0) * modulus;

    row++;
    if (row == m) {
        row = 0;
        col++;
        corA = correctA[row];
        corB = correctB[col] + k_mod_m;
    } else {
        corA = correctA[row];
    }
    in.z += corA + corB;
    in.z -= __mulhi(in.z, invm) * modulus;
    in.z -= (in.z >= modulus) * modulus;
    in.z += (in.z < 0) * modulus;

    row++;
    if (row == m) {
        row = 0;
        col++;
        corA = correctA[row];
        corB = correctB[col] + k_mod_m;
    } else {
        corA = correctA[row];
    }
    in.w += corA + corB;
    in.w -= __mulhi(in.w, invm) * modulus;
    in.w -= (in.w >= modulus) * modulus;
    in.w += (in.w < 0) * modulus;

    uchar4 result;
    result.x = static_cast<uint8_t>(in.x);
    result.y = static_cast<uint8_t>(in.y);
    result.z = static_cast<uint8_t>(in.z);
    result.w = static_cast<uint8_t>(in.w);

    *out = result;
}

// interface!!
__inline__ void conv_32i_2_8u(const unsigned i,              //
                              const size_t m,                //
                              const size_t sizeC,            //
                              const size_t k,                //
                              const int32_t *const C32i,     // input
                              const int32_t *const correctA, //
                              const int32_t *const correctB, //
                              uint8_t *const C8u)            // output
{
    if (i == 0) {
        conv_32i_2_8u_256_kernel<<<oz2_const::grids_conv32i8u, oz2_const::threads_conv32i8u>>>(sizeC, m, C32i, correctA, correctB, C8u);
    } else {
        const uint8_t modulus = static_cast<uint8_t>(-oz2_table::moduli[i].z);
        const int32_t invm    = oz2_table::invm_32i[i - 1];
        conv_32i_2_8u_not256_kernel<<<oz2_const::grids_conv32i8u, oz2_const::threads_conv32i8u>>>(sizeC, m, C32i, modulus, invm, (k << 14) % modulus, correctA, correctB, C8u);
    }
}

} // namespace oz2_util
