# GEMMul8

GEMMul8 (GEMMulate): GEMM emulation using int8 matrix engines based on the Ozaki Scheme2.

## Build

1. (Option) Build `cuMpSGEMM` and `ozIMMU_EF` according to [cuMpSGEMM](https://github.com/enp1s0/cuMpSGEMM) and [ozIMMU](https://github.com/enp1s0/ozIMMU) (see also [Accelerator for ozIMMU](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU)).

2. Navigate to the `GEMMul8` directory and modify the following values in the `Makefile` and `testing/Makefile` to the appropriate values:

   - `CUDA_PATH`: path to cuda (e.g., `CUDA_PATH := /usr/local/cuda-12.8`)
   - `ozIMMU_EF`: `yes` if the ozIMMU_EF is used in sample codes (e.g., `ozIMMU_EF := yes`)
   - `ozIMMU_EF_DIR`: path to ozIMMU_EF (e.g., `ozIMMU_EF_DIR := path/to/ozIMMU_EF`)
   - `cuMpSGEMM`: `yes` if the cuMpSGEMM is used in sample codes (e.g., `cuMpSGEMM := yes`)
   - `cuMpSGEMM_DIR`: path to cuMpSGEMM (e.g., `cuMpSGEMM_DIR := path/to/cuMpSGEMM`)
   - Note: Please specify `ozIMMU_EF_DIR` and `cuMpSGEMM_DIR` as the parent directories of `build` directories, i.e., `build` directories are located as `ozIMMU_EF_DIR/build` and `cuMpSGEMM_DIR/build`.

3. Run `make` in the `GEMMul8` directory to compile all files.

4. Navigate to the `testing` directory and then run following commands to run sample codes.
   - `make test_f MODE="mode1 mode2 mode3"` (testrun for SGEMM emulation)
   - `make test_d MODE="mode1 mode2 mode3"` (testrun for DGEMM emulation)
   - mode list:
     - `accuracy_check`
     - `flops_check`
     - `watt_check`
     - `all`
   - e.g.,
     - `make test_f MODE="watt_check"`
     - `make test_f MODE="accuracy_check flops_check"`
     - `make test_f test_d MODE="all"`

## Usage

```
// settings
const unsigned num_moduli = 14;   // #moduli (2 <= num_moduli <= 19 for SGEMM emu., 2 <= num_moduli <= 20 for DGEMM emu.)
const bool fastmode = true;       // true (fast-mode) or false (accurate-mode)

// allocate work space
const size_t worksize = workSize(m,n,k,num_moduli);
void *work;
cudaMalloc(&work, worksize);

// run emulation
std::vector<double> time_breakwown(4,0);
time_breakwown = gemmul8::gemm(cublas_handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               m, n, k,
                               &alpha, devA, lda,
                               devB, ldb,
                               &beta, devC, ldc,
                               num_moduli,
                               fastmode,
                               work);
``` 

## Attention

ozIMMU_EF is from [ozIMMU](https://github.com/enp1s0/ozIMMU) by Ootomo and [Accelerator for ozIMMU](https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU) by RIKEN R-CCS.

cuMpSGEMM is from [cuMpSGEMM](https://github.com/enp1s0/cuMpSGEMM) by Ootomo.

If you use these libraries, you must agree to the licenses terms of ozIMMU, Accelerator for ozIMMU, and cuMpSGEMM in addition to the license for this code (Ozaki Scheme2).

## References

- Hiroyuki Ootomo, Hidetaka Manabe, Kenji Harada, and Rio Yokota. 2023. Quantum Circuit Simulation by SGEMM Emulation on Tensor Cores and Automatic Precision Selection. In High Performance Computing. Springer Nature Switzerland, Cham, 259--276.
- Hiroyuki Ootomo, Katsuhisa Ozaki, and Rio Yokota. 2024. DGEMM on integer matrix multiplication unit. The International Journal of High Performance Computing Applications 38, 4 (2024), 297--313.
- Yuki Uchino, Katsuhisa Ozaki, and Toshiyuki Imamura. 2025. Performance enhancement of the Ozaki Scheme on integer matrix multiplication unit. The International Journal of High Performance Computing Applications 0, 0 (2025), 10943420241313064.

## Citation

TBD

## License

MIT
