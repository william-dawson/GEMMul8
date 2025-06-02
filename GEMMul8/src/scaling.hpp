#pragma once
#include "common.hpp"
#include "table.hpp"

namespace oz2_util {

namespace {

template <typename T>
struct Vec4 {
    T x, y, z, w;
};

template <typename T> __forceinline__ __device__ T Tabs(T in);
template <> __forceinline__ __device__ double Tabs<double>(double in) { return fabs(in); };
template <> __forceinline__ __device__ float Tabs<float>(float in) { return fabsf(in); };
template <> __forceinline__ __device__ int32_t Tabs<int32_t>(int32_t in) { return abs(in); };

template <typename T> __forceinline__ __device__ int __T2int_ru(T in);
template <> __forceinline__ __device__ int __T2int_ru<double>(double in) { return __double2int_ru(in); };
template <> __forceinline__ __device__ int __T2int_ru<float>(float in) { return __float2int_ru(in); };

template <typename T> __forceinline__ __device__ T Tscalbn(T in, const int sft);
template <> __forceinline__ __device__ double Tscalbn<double>(double in, const int sft) { return scalbn(in, sft); };
template <> __forceinline__ __device__ float Tscalbn<float>(float in, const int sft) { return scalbnf(in, sft); };

template <typename T> __forceinline__ __device__ T Ttrunc(T in);
template <> __forceinline__ __device__ double Ttrunc<double>(double in) { return trunc(in); };
template <> __forceinline__ __device__ float Ttrunc<float>(float in) { return truncf(in); };

template <typename T> __forceinline__ __device__ int Tilogb(T in);
template <> __forceinline__ __device__ int Tilogb<double>(double in) { return ilogb(in); };
template <> __forceinline__ __device__ int Tilogb<float>(float in) { return ilogbf(in); };

template <typename T> __forceinline__ __device__ T Tzero() { return 0; };
template <> __forceinline__ __device__ double Tzero<double>() { return 0.0; };
template <> __forceinline__ __device__ float Tzero<float>() { return 0.0F; };
template <> __forceinline__ __device__ int32_t Tzero<int32_t>() { return 0; };

template <typename T> __forceinline__ __device__ T __Tfma_ru(T in1, T in2, T in3);
template <> __forceinline__ __device__ double __Tfma_ru<double>(double in1, double in2, double in3) { return __fma_ru(in1, in2, in3); };
template <> __forceinline__ __device__ float __Tfma_ru<float>(float in1, float in2, float in3) { return __fmaf_ru(in1, in2, in3); };

template <typename T> __forceinline__ __device__ void inner_warp_max(T &amax) {
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 16)); // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 8));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 4));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 2));  // warp-level reduction
    amax = max(amax, __shfl_down_sync(0xFFFFFFFFu, amax, 1));  // warp-level reduction
}

template <typename T> __forceinline__ __device__ void inner_warp_sum(T &sum);
template <> __forceinline__ __device__ void inner_warp_sum<double>(double &sum) {
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 16)); // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 8));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 4));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 2));  // warp-level reduction
    sum = __dadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 1));  // warp-level reduction
}
template <> __forceinline__ __device__ void inner_warp_sum<float>(float &sum) {
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 16)); // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 8));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 4));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 2));  // warp-level reduction
    sum = __fadd_ru(sum, __shfl_down_sync(0xFFFFFFFFu, sum, 1));  // warp-level reduction
}
template <> __forceinline__ __device__ void inner_warp_sum<int32_t>(int32_t &sum) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 16); // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 8);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 4);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 2);  // warp-level reduction
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, 1);  // warp-level reduction
}

template <typename T>
__device__ T find_amax(const T *const ptr,    //
                       const unsigned length, //
                       const unsigned inc,    // leading dimension
                       T *shm)                //
{
    // max in thread
    T amax = Tzero<T>();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i * inc]);
        amax  = max(amax, tmp);
    }

    // inner-warp reduction
    inner_warp_max<T>(amax);

    // inner-threadblock reduction
    if ((threadIdx.x & 0x1f) == 0) shm[threadIdx.x >> 5] = amax; // shm[warp-id] = max in warp

    __syncthreads();
    amax = Tzero<T>();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = shm[threadIdx.x];
        inner_warp_max<T>(amax);
        if (threadIdx.x == 0) shm[0] = amax;
    }

    __syncthreads();
    return shm[0];
}

template <typename T>
__device__ T find_amax_and_nrm(const T *const ptr,    //
                               const unsigned length, //
                               const unsigned inc,    // leading dimension
                               T *shm,                //
                               T &vecnrm)             //
{
    T *shm1 = shm;
    T *shm2 = shm + 32;

    // max in thread
    T amax = Tzero<T>();
    T sum  = Tzero<T>();
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
        T tmp = Tabs<T>(ptr[i * inc]);
        amax  = max(amax, tmp);
        sum   = __Tfma_ru<T>(tmp, tmp, sum); // round-up mode
    }

    // inner-warp reduction
    inner_warp_max<T>(amax);
    inner_warp_sum<T>(sum);

    // inner-threadblock reduction
    const auto id = (threadIdx.x & 0x1f);
    if (id == 0) {
        shm1[threadIdx.x >> 5] = amax; // shm[warp-id] = max in warp
    } else if (id == 1) {
        shm2[(threadIdx.x - 1) >> 5] = sum; // shm[warp-id] = sum in warp
    }

    __syncthreads();
    amax = Tzero<T>();
    sum  = Tzero<T>();
    if (threadIdx.x < 32) {
        if (threadIdx.x < (blockDim.x >> 5)) amax = shm1[threadIdx.x];
        inner_warp_max<T>(amax);
        if (threadIdx.x == 0) shm1[0] = amax;
    } else if (threadIdx.x < 64) {
        if ((threadIdx.x - 32) < (blockDim.x >> 5)) sum = shm[threadIdx.x];
        inner_warp_sum<T>(sum);
        if (threadIdx.x == 32) shm2[0] = sum;
    }

    __syncthreads();
    vecnrm = shm2[0];
    return shm[0];
}

template <typename T> __device__ int8_t mod_8i(T a, unsigned j);
template <> __device__ int8_t mod_8i<double>(double a, unsigned j) {
    const auto val = oz2_table::moduli_dev[j];
    float tmp      = __double2float_rn(fma(rint(a * val.y), val.x, a));
    tmp            = __fmaf_rn(rintf(tmp * val.w), val.z, tmp);
    tmp            = __fmaf_rn(rintf(tmp * val.w), val.z, tmp);
    return static_cast<int8_t>(tmp);
}
template <> __device__ int8_t mod_8i<float>(float a, unsigned j) {
    const auto val = oz2_table::modulif_dev[j];
    float tmp      = __fmaf_rn(rintf(a * val.y), val.x, a);
    tmp            = __fmaf_rn(rintf(tmp * val.y), val.x, tmp);
    tmp            = __fmaf_rn(rintf(tmp * val.y), val.x, tmp);
    return static_cast<int8_t>(tmp);
}

} // namespace

namespace int8tc {

__forceinline__ __device__ int compute_sft(int amax, int16_t sftA, const float log2M) {
    return sftA + __float2int_rd(__fmaf_rd(-0.51F, __log2f(__int2float_rn(amax)), log2M));
}

template <typename T>
__global__ void extract_A8i_kernel(const size_t k,                   // size(A,2)
                                   const T *const __restrict__ A,    // input (lda * k)
                                   const size_t lda,                 // leading dimension
                                   int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                   const size_t lda8i,               // leading dimension
                                   int16_t *const __restrict__ sftA) // exponent of shift values
{
    __shared__ T smem[32];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    const T amax                   = find_amax<T>(in, k, lda, smem);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftA[row_idx] = sft;
    }

    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx * lda]), sft));
        out4.y = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 1) * lda]), sft));
        out4.z = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 2) * lda]), sft));
        out4.w = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 3) * lda]), sft));

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = (idx < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx * lda]), sft)) : 0;
        out4.y = (idx + 1 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 1) * lda]), sft)) : 0;
        out4.z = (idx + 2 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 2) * lda]), sft)) : 0;
        out4.w = (idx + 3 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[(idx + 3) * lda]), sft)) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
}

template <typename T>
__global__ void extract_B8i_kernel(const size_t k,                   // size(B,1)
                                   const T *const __restrict__ B,    // input (ldb * n)
                                   const size_t ldb,                 // leading dimension
                                   int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                   const size_t ldb8i,               // leading dimension
                                   int16_t *const __restrict__ sftB) // exponent of shift values
{
    __shared__ T smem[32];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    const T amax                   = find_amax<T>(in, k, 1u, smem);
    const int sft                  = 5 - Tilogb<T>(amax); // 6-bit
    if (threadIdx.x == 0) {
        sftB[col_idx] = sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);
        out4.x      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.x), sft));
        out4.y      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.y), sft));
        out4.z      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.z), sft));
        out4.w      = __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in4.w), sft));

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        char4 out4;
        unsigned idx = i << 2;

        out4.x = (idx < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx]), sft)) : 0;
        out4.y = (idx + 1 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx + 1]), sft)) : 0;
        out4.z = (idx + 2 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx + 2]), sft)) : 0;
        out4.w = (idx + 3 < k) ? __T2int_ru<T>(Tscalbn<T>(Tabs<T>(in[idx + 3]), sft)) : 0;

        *reinterpret_cast<char4 *>(out + idx) = out4;
    }
}

template <typename T>
__global__ void scalingA_kernel(const size_t n,                         // size(C,2)
                                const size_t k,                         // size(A,2)
                                const size_t incA8i,                    // lda8i * m
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ A,          // input (lda * m)
                                const size_t lda,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ A8i,         // output (lda8i * m)
                                const size_t lda8i,                     // leading dimension
                                int16_t *const __restrict__ sftA,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftAi    = sftA[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx, n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    const T *const __restrict__ in = A + row_idx;
    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = Ttrunc<T>(Tscalbn<T>(in[idx * lda], sft));
        in4.y = Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * lda], sft));
        in4.z = Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * lda], sft));
        in4.w = Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * lda], sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx * lda], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * lda], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * lda], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * lda], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }

    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingB_kernel(const size_t m,                         // size(C,1)
                                const size_t k,                         // size(B,1)
                                const size_t incB8i,                    // ldb8i * n
                                const unsigned num_moduli,              // #moduli
                                const T *const __restrict__ B,          // input (ldb * n)
                                const size_t ldb,                       // leading dimension
                                const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                const size_t ldc32i,                    // leading dimension
                                int8_t *const __restrict__ B8i,         // output (ldb8i * n)
                                const size_t ldb8i,                     // leading dimension
                                int16_t *const __restrict__ sftB,       // exponent of shift values
                                const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftBi    = sftB[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx * ldc32i, m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    const T *const __restrict__ in = B + col_idx * ldb;
    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);
        in4.x       = Ttrunc<T>(Tscalbn<T>(in4.x, sft));
        in4.y       = Ttrunc<T>(Tscalbn<T>(in4.y, sft));
        in4.z       = Ttrunc<T>(Tscalbn<T>(in4.z, sft));
        in4.w       = Ttrunc<T>(Tscalbn<T>(in4.w, sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 1], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 2], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 3], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }

    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingAT_kernel(const size_t n,                         // size(C,2)
                                 const size_t k,                         // size(AT,1)
                                 const size_t incA8i,                    // lda8i * n
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ A,          // input (lda * n)
                                 const size_t lda,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ A8i,         // output (lda8i * n)
                                 const size_t lda8i,                     // leading dimension
                                 int16_t *const __restrict__ sftA,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto col_idx = blockIdx.x;
    const int sftAi    = sftA[col_idx];
    const int32_t amax = find_amax<int32_t>(C32i + col_idx, n, ldc32i, smem);
    const int sft      = compute_sft(amax, sftAi, log2M);

    const T *const __restrict__ in = A + col_idx * lda;
    int8_t *const __restrict__ out = A8i + col_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);
        in4.x       = Ttrunc<T>(Tscalbn<T>(in4.x, sft));
        in4.y       = Ttrunc<T>(Tscalbn<T>(in4.y, sft));
        in4.z       = Ttrunc<T>(Tscalbn<T>(in4.z, sft));
        in4.w       = Ttrunc<T>(Tscalbn<T>(in4.w, sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 1], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 2], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 3], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }

    if (threadIdx.x == 0) {
        sftA[col_idx] = -sft;
    }
}

template <typename T>
__global__ void scalingBT_kernel(const size_t m,                         // size(C,1)
                                 const size_t k,                         // size(B,2)
                                 const size_t incB8i,                    // ldb8i * m
                                 const unsigned num_moduli,              // #moduli
                                 const T *const __restrict__ B,          // input (ldb * m)
                                 const size_t ldb,                       // leading dimension
                                 const int32_t *const __restrict__ C32i, // input (ldc32i * n)
                                 const size_t ldc32i,                    // leading dimension
                                 int8_t *const __restrict__ B8i,         // output (ldb8i * m)
                                 const size_t ldb8i,                     // leading dimension
                                 int16_t *const __restrict__ sftB,       // exponent of shift values
                                 const float log2M)                      // log2(M-1)/2 - 0.5
{
    __shared__ int32_t smem[32];
    const auto row_idx = blockIdx.x;
    const int sftBi    = sftB[row_idx];
    const int32_t amax = find_amax<int32_t>(C32i + row_idx * ldc32i, m, 1u, smem);
    const int sft      = compute_sft(amax, sftBi, log2M);

    const T *const __restrict__ in = B + row_idx;
    int8_t *const __restrict__ out = B8i + row_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = Ttrunc<T>(Tscalbn<T>(in[idx * ldb], sft));
        in4.y = Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * ldb], sft));
        in4.z = Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * ldb], sft));
        in4.w = Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * ldb], sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx * ldb], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * ldb], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * ldb], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * ldb], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }

    if (threadIdx.x == 0) {
        sftB[row_idx] = -sft;
    }
}

template <typename T>
__inline__ void scaling(cublasHandle_t handle,        // handle
                        const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                        const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                        const size_t m,               // size(A,1) & size(C,1)
                        const size_t n,               // size(B,2) & size(C,2)
                        const size_t k,               // size(A,2) & size(B,1)
                        const unsigned num_moduli,    // #moduli
                        const T *const A,             // input
                        const size_t lda,             // leading dimension
                        const T *const B,             // input
                        const size_t ldb,             // leading dimension
                        int8_t *const A8i,            // output (k * m)
                        const size_t lda8i,           // leading dimension
                        int16_t *const sftA,          // exponent of shift values for rows of A
                        int8_t *const B8i,            // output (k * n)
                        const size_t ldb8i,           // leading dimension
                        int16_t *const sftB,          // exponent of shift values for cols of B
                        int32_t *const C32i,          // tmp (m * n)
                        const unsigned table_idx)     //
{
    // extract first 7-bit from A and B
    if (op_A == CUBLAS_OP_N) {
        extract_A8i_kernel<T><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    } else {
        extract_B8i_kernel<T><<<m, oz2_const::threads_scaling>>>(k, A, lda, A8i, lda8i, sftA);
    }
    if (op_B == CUBLAS_OP_N) {
        extract_B8i_kernel<T><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    } else {
        extract_A8i_kernel<T><<<n, oz2_const::threads_scaling>>>(k, B, ldb, B8i, ldb8i, sftB);
    }

    // C32i := A8i^T*B8i
    constexpr int32_t alpha = 1;
    constexpr int32_t beta  = 0;
    cudaDeviceSynchronize();
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, lda8i, &alpha, A8i, CUDA_R_8I, lda8i, B8i, CUDA_R_8I, ldb8i, &beta, C32i, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

    // extract high order bits from A and B
    cudaDeviceSynchronize();
    const float log2M = oz2_table::int8tc::log2M[table_idx]; // fld(log2(M-1)/2 - 0.5)
    if (op_A == CUBLAS_OP_N) {
        scalingA_kernel<T><<<m, oz2_const::threads_scaling>>>(n, k, lda8i * m, num_moduli, A, lda, C32i, m, A8i, lda8i, sftA, log2M);
    } else {
        scalingAT_kernel<T><<<m, oz2_const::threads_scaling>>>(n, k, lda8i * m, num_moduli, A, lda, C32i, m, A8i, lda8i, sftA, log2M);
    }
    if (op_B == CUBLAS_OP_N) {
        scalingB_kernel<T><<<n, oz2_const::threads_scaling>>>(m, k, ldb8i * n, num_moduli, B, ldb, C32i, m, B8i, ldb8i, sftB, log2M);
    } else {
        scalingBT_kernel<T><<<n, oz2_const::threads_scaling>>>(m, k, ldb8i * n, num_moduli, B, ldb, C32i, m, B8i, ldb8i, sftB, log2M);
    }
}

} // namespace int8tc

namespace vecnorm {

template <typename T> __forceinline__ __device__ int compute_sft(T amax, T vecnrm, const float log2M);
template <> __forceinline__ __device__ int compute_sft<double>(double amax, double vecnrm, const float log2M) {
    const int exponent  = ilogb(vecnrm);
    const float vecnrmf = __double2float_ru(scalbn(vecnrm, -exponent));
    const int k         = __float2int_rd(__fmaf_rd(-0.51F, __fadd_ru(__log2f(vecnrmf), exponent), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - ilogb(amax);
}
template <> __forceinline__ __device__ int compute_sft<float>(float amax, float vecnrm, const float log2M) {
    const int k = __float2int_rd(__fmaf_rd(-0.51F, __log2f(vecnrm), log2M));
    return min(__float2int_rd(log2M - 1.0f), k) - ilogbf(amax);
}

template <typename T>
__global__ void scalingA_kernel(const size_t k,                   // size(A,2)
                                const size_t incA8i,              // lda8i * m
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ A,    // input (lda * n)
                                const size_t lda,                 // leading dimension
                                int8_t *const __restrict__ A8i,   // output (lda8i * m)
                                const size_t lda8i,               // leading dimension
                                int16_t *const __restrict__ sftA, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    __shared__ T smem[64];
    const auto row_idx             = blockIdx.x;
    const T *const __restrict__ in = A + row_idx;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, lda, smem, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftA[row_idx] = -sft;
    }

    int8_t *const __restrict__ out = A8i + row_idx * lda8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = Ttrunc<T>(Tscalbn<T>(in[idx * lda], sft));
        in4.y = Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * lda], sft));
        in4.z = Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * lda], sft));
        in4.w = Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * lda], sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
    kmax = lda8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx * lda], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 1) * lda], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 2) * lda], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[(idx + 3) * lda], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incA8i + idx) = out4;
        }
    }
}

template <typename T>
__global__ void scalingB_kernel(const size_t k,                   // size(B,1)
                                const size_t incB8i,              // ldb8i * n
                                const unsigned num_moduli,        // #moduli
                                const T *const __restrict__ B,    // input (ldb * n)
                                const size_t ldb,                 // leading dimension
                                int8_t *const __restrict__ B8i,   // output (ldb8i * n)
                                const size_t ldb8i,               // leading dimension
                                int16_t *const __restrict__ sftB, // exponent of shift values
                                const float log2M)                // log2(M-1)/2 - 1.5
{
    __shared__ T smem[64];
    const auto col_idx             = blockIdx.x;
    const T *const __restrict__ in = B + col_idx * ldb;
    T vecnrm;
    const T amax  = find_amax_and_nrm<T>(in, k, 1u, smem, vecnrm);
    const int sft = compute_sft<T>(amax, vecnrm, log2M);
    if (threadIdx.x == 0) {
        sftB[col_idx] = -sft;
    }

    int8_t *const __restrict__ out = B8i + col_idx * ldb8i;

    unsigned kmax = k >> 2;
    unsigned i    = threadIdx.x;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4 = *reinterpret_cast<const Vec4<T> *>(in + idx);
        in4.x       = Ttrunc<T>(Tscalbn<T>(in4.x, sft));
        in4.y       = Ttrunc<T>(Tscalbn<T>(in4.y, sft));
        in4.z       = Ttrunc<T>(Tscalbn<T>(in4.z, sft));
        in4.w       = Ttrunc<T>(Tscalbn<T>(in4.w, sft));

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = mod_8i<T>(in4.x, j);
            out4.y = mod_8i<T>(in4.y, j);
            out4.z = mod_8i<T>(in4.z, j);
            out4.w = mod_8i<T>(in4.w, j);

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
    kmax = ldb8i >> 2;
    for (; i < kmax; i += blockDim.x) {
        unsigned idx = i << 2;

        Vec4<T> in4;
        in4.x = (idx < k) ? Ttrunc<T>(Tscalbn<T>(in[idx], sft)) : 0;
        in4.y = (idx + 1 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 1], sft)) : 0;
        in4.z = (idx + 2 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 2], sft)) : 0;
        in4.w = (idx + 3 < k) ? Ttrunc<T>(Tscalbn<T>(in[idx + 3], sft)) : 0;

        char4 out4;
        for (unsigned j = 0; j < num_moduli; ++j) {

            out4.x = (idx < k) ? mod_8i<T>(in4.x, j) : 0;
            out4.y = (idx + 1 < k) ? mod_8i<T>(in4.y, j) : 0;
            out4.z = (idx + 2 < k) ? mod_8i<T>(in4.z, j) : 0;
            out4.w = (idx + 3 < k) ? mod_8i<T>(in4.w, j) : 0;

            *reinterpret_cast<char4 *>(out + j * incB8i + idx) = out4;
        }
    }
}

template <typename T>
__inline__ void scaling(const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                        const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                        const size_t m,               // size(A,1) & size(C,1)
                        const size_t n,               // size(B,2) & size(C,2)
                        const size_t k,               // size(A,2) & size(B,1)
                        const unsigned num_moduli,    // #moduli
                        const T *const A,             // input
                        const size_t lda,             // leading dimension
                        const T *const B,             // input
                        const size_t ldb,             // leading dimension
                        int8_t *const A8i,            // output (k * m)
                        const size_t lda8i,           // leading dimension
                        int16_t *const sftA,          // exponent of shift values for rows of A
                        int8_t *const B8i,            // output (k * n)
                        const size_t ldb8i,           // leading dimension
                        int16_t *const sftB,          // exponent of shift values for cols of B
                        const unsigned table_idx)     //
{
    const float log2M = oz2_table::vecnorm::log2M[table_idx]; // fld(log2(M-1)/2 - 1.5)
    if (op_A == CUBLAS_OP_N) {
        scalingA_kernel<T><<<m, oz2_const::threads_scaling>>>(k, lda8i * m, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    } else {
        scalingB_kernel<T><<<m, oz2_const::threads_scaling>>>(k, lda8i * m, num_moduli, A, lda, A8i, lda8i, sftA, log2M);
    }
    if (op_B == CUBLAS_OP_N) {
        scalingB_kernel<T><<<n, oz2_const::threads_scaling>>>(k, ldb8i * n, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    } else {
        scalingA_kernel<T><<<n, oz2_const::threads_scaling>>>(k, ldb8i * n, num_moduli, B, ldb, B8i, ldb8i, sftB, log2M);
    }
}

} // namespace vecnorm

} // namespace oz2_util
