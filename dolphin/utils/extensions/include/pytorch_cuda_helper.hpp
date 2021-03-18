#ifndef PYTORCH_CUDA_HELPER
#define PYTORCH_CUDA_HELPER

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "common_cuda_helper.hpp"

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define __PHALF(x) (x)

#endif  // PYTORCH_CUDA_HELPER
