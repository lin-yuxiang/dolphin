#ifndef ROI_POOL_CUDA
#define ROI_POOL_CUDA
#include "pytorch_cpp_helper.hpp"

#ifdef WITH_CUDA
void ROIPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                      Tensor argmax, int pooled_height,
                                      int pooled_width, float spatial_scale);

void ROIPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                       Tensor argmax, Tensor grad_input,
                                       int pooled_height, int pooled_width,
                                       float spatial_scale);

void roi_pool_forward_cuda(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, int pooled_height, int pooled_width,
                           float spatial_scale) {
  ROIPoolForwardCUDAKernelLauncher(input, rois, output, argmax, pooled_height,
                                   pooled_width, spatial_scale);
}

void roi_pool_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax,
                            Tensor grad_input, int pooled_height,
                            int pooled_width, float spatial_scale) {
  ROIPoolBackwardCUDAKernelLauncher(grad_output, rois, argmax, grad_input,
                                    pooled_height, pooled_width, spatial_scale);
}
#endif

#endif // ROI_POOL_CUDA