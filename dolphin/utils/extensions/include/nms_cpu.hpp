#ifndef NMS_CPU
#define NMS_CPU
#include "pytorch_cpp_helper.hpp"

Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms_cpu(Tensor boxes, Tensor scores, Tensor dets,
                   float iou_threshold, float sigma, float min_score,
                   int method, int offset);

#endif // NMS_CPU