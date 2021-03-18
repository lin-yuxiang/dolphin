#ifndef NMS_CUDA
#define NMS_CUDA
#include "pytorch_cpp_helper.hpp"

#ifdef WITH_CUDA
Tensor NMSCUDAKernelLauncher(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor nms_cuda(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return NMSCUDAKernelLauncher(boxes, scores, iou_threshold, offset);
}
#endif

#endif // NMS_CUDA