#ifndef ROI_ALIGN_CPU
#define ROI_ALIGN_CPU
#include "pytorch_cpp_helper.hpp"

void ROIAlignForwardCPULauncher(Tensor input, Tensor rois, Tensor output,
                                Tensor argmax_y, Tensor argmax_x,
                                int aligned_height, int aligned_width,
                                float spatial_scale, int sampling_ratio,
                                int pool_mode, bool aligned);

void ROIAlignBackwardCPULauncher(Tensor grad_output, Tensor rois,
                                 Tensor argmax_y, Tensor argmax_x,
                                 Tensor grad_input, int aligned_height,
                                 int aligned_width, float spatial_scale,
                                 int sampling_ratio, int pool_mode,
                                 bool aligned);
                        
void roi_align_forward_cpu(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax_y, Tensor argmax_x, int aligned_height,
                           int aligned_width, float spatial_scale,
                           int sampling_ratio, int pool_mode, bool aligned) {
  ROIAlignForwardCPULauncher(input, rois, output, argmax_y, argmax_x,
                             aligned_height, aligned_width, spatial_scale,
                             sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_cpu(Tensor grad_output, Tensor rois, Tensor argmax_y,
                            Tensor argmax_x, Tensor grad_input,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  ROIAlignBackwardCPULauncher(grad_output, rois, argmax_y, argmax_x, grad_input,
                              aligned_height, aligned_width, spatial_scale,
                              sampling_ratio, pool_mode, aligned);
}


#endif // ROI_ALIGN_CPU