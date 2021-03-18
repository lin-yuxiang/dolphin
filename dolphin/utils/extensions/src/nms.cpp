#include "pytorch_cpp_helper.hpp"
#include "nms_cpu.hpp"
#include "nms_cuda.hpp"

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  if (boxes.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA_INPUT(boxes);
    CHECK_CUDA_INPUT(scores);
    return nms_cuda(boxes, scores, iou_threshold, offset);
#else
    AT_ERROR("nms is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(boxes);
    CHECK_CPU_INPUT(scores);
    return nms_cpu(boxes, scores, iou_threshold, offset);
  }
}

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset) {
  if (boxes.device().is_cuda()) {
    AT_ERROR("softnms is not implemented on GPU");
  } else {
    return softnms_cpu(boxes, scores, dets, iou_threshold, sigma, min_score,
                       method, offset);
  }
}