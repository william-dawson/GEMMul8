#===============
# path to cuda
#===============
CUDA_PATH := /usr/local/cuda-12.8


export PATH := $(CUDA_PATH)/bin:$(PATH)
export PATH := $(CUDA_PATH)/bin/lib64:$(PATH)

GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
ARCHS := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCCFLAGS := -std=c++20 $(ARCHS) -lnvidia-ml -O3 -DGEMMul8_ARCH=$(GPU_ARCH) -Xcompiler -fPIC

LIBS := -lcublas -lcudart -lcuda

LIB_TARGET := lib/libgemmul8.a
DLIB_TARGET := lib/libgemmul8.so

HEADERS := include/gemmul8.hpp

CU_FILES := src/gemmul8.cu
CU_OBJS := $(CU_FILES:.cu=.o)
WRP_OBJS := src/wrap_dgemm.o src/offload_dgemm.o

all: $(LIB_TARGET) $(DLIB_TARGET)

$(LIB_TARGET): $(CU_OBJS) $(WRP_OBJS)
	mkdir -p lib
	ar rcs $@ $^
	cd testing ; make ;cd ../ 
$(DLIB_TARGET): $(CU_OBJS) $(WRP_OBJS)
	mkdir -p lib
	nvcc -shared $^ -o $@ $(LIBS)

%.o: %.cu $(HEAgfortranDERS)
	nvcc $(NVCCFLAGS) $(LIBS) -c $< -o $@
%.o: %.cpp
	nvcc $(NVCCFLAGS) $(LIBS) -c $< -o $@
%.o: %.F90 $(HEADERS)
	gfortran -fPIC -c $< -o $@

clean:
	cd testing ; make clean ;cd ../ 
	rm -f $(CU_OBJS) $(WRP_OBJS) $(LIB_TARGET)
