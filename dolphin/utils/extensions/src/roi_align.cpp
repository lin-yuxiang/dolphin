#include "pytorch_cpp_helper.hpp"
#include "roi_align_cpu.hpp"
#include "roi_align_cuda.hpp"

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(argmax_y);
    CHECK_CUDA_INPUT(argmax_x);

    roi_align_forward_cuda(input, rois, output, argmax_y, argmax_x,
                           aligned_height, aligned_width, spatial_scale,
                           sampling_ratio, pool_mode, aligned);
#else
    AT_ERROR("RoIAlign is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(input);
    CHECK_CPU_INPUT(rois);
    CHECK_CPU_INPUT(output);
    CHECK_CPU_INPUT(argmax_y);
    CHECK_CPU_INPUT(argmax_x);
    roi_align_forward_cpu(input, rois, output, argmax_y, argmax_x,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
  }
}

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned) {
  if (grad_output.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(argmax_y);
    CHECK_CUDA_INPUT(argmax_x);
    CHECK_CUDA_INPUT(grad_input);

    roi_align_backward_cuda(grad_output, rois, argmax_y, argmax_x, grad_input,
                            aligned_height, aligned_width, spatial_scale,
                            sampling_ratio, pool_mode, aligned);
#else
    AT_ERROR("RoIAlign is not compiled with GPU support");
#endif
  } else {
    CHECK_CPU_INPUT(grad_output);
    CHECK_CPU_INPUT(rois);
    CHECK_CPU_INPUT(argmax_y);
    CHECK_CPU_INPUT(argmax_x);
    CHECK_CPU_INPUT(grad_input);

    roi_align_backward_cpu(grad_output, rois, argmax_y, argmax_x, grad_input,
                           aligned_height, aligned_width, spatial_scale,
                           sampling_ratio, pool_mode, aligned);
  }
}