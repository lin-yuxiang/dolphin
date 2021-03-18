#include "pytorch_cpp_helper.hpp"

std::string get_compiler_version();
std::string get_compiling_cuda_version();


Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned);

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned);

void roi_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax,
                      int pooled_height, int pooled_width, float spatial_scale);

void roi_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax,
                       Tensor grad_input, int pooled_height, int pooled_width,
                       float spatial_scale);

Tensor dcn_v2_forward(const Tensor &input, const Tensor &weight, 
                      const Tensor &bias, const Tensor &offset, 
                      const Tensor &mask, const int kernel_h,
                      const int kernel_w, const int stride_h, 
                      const int stride_w, const int pad_h, const int pad_w, 
                      const int dilation_h, const int dilation_w, 
                      const int deformable_group);

std::vector<Tensor> dcn_v2_backward(const Tensor &input, const Tensor &weight, 
                                    const Tensor &bias, const Tensor &offset, 
                                    const Tensor &mask, 
                                    const Tensor &grad_output, int kernel_h, 
                                    int kernel_w, int stride_h, int stride_w,
                                    int pad_h, int pad_w, int dilation_h, 
                                    int dilation_w, int deformable_group);

std::tuple<Tensor, Tensor> dcn_v2_psroi_pooling_forward(
      const Tensor &input, const Tensor &bbox, const Tensor &trans,
      const int no_trans, const float spatial_scale, const int output_dim,
      const int group_size, const int pooled_size, const int part_size,
      const int sample_per_part, const float trans_std);

std::tuple<Tensor, Tensor> dcn_v2_psroi_pooling_backward(
      const Tensor &out_grad, const Tensor &input, const Tensor &bbox,
      const Tensor &trans, const Tensor &top_count, const int no_trans,
      const float spatial_scale, const int output_dim, const int group_size,
      const int pooled_size, const int part_size, const int sample_per_part,
      const float trans_std);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "nms (CPU/CUDA) ", py::arg("boxes"), py::arg("scores"),
        py::arg("iou_threshold"), py::arg("offset"));
  m.def("softnms", &softnms, "softnms (CPU) ", py::arg("boxes"),
        py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
        py::arg("sigma"), py::arg("min_score"), py::arg("method"),
        py::arg("offset"));
  m.def("roi_align_forward", &roi_align_forward, "roi_align forward",
        py::arg("input"), py::arg("rois"), py::arg("output"),
        py::arg("argmax_y"), py::arg("argmax_x"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_align_backward", &roi_align_backward, "roi_align backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax_y"),
        py::arg("argmax_x"), py::arg("grad_input"), py::arg("aligned_height"),
        py::arg("aligned_width"), py::arg("spatial_scale"),
        py::arg("sampling_ratio"), py::arg("pool_mode"), py::arg("aligned"));
  m.def("roi_pool_forward", &roi_pool_forward, "roi_pool forward",
        py::arg("input"), py::arg("rois"), py::arg("output"), py::arg("argmax"),
        py::arg("pooled_height"), py::arg("pooled_width"),
        py::arg("spatial_scale"));
  m.def("roi_pool_backward", &roi_pool_backward, "roi_pool backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax"),
        py::arg("grad_input"), py::arg("pooled_height"),
        py::arg("pooled_width"), py::arg("spatial_scale"));
  m.def("dcn_v2_forward", &dcn_v2_forward, "dcn_v2_forward");
  m.def("dcn_v2_backward", &dcn_v2_backward, "dcn_v2_backward");
  m.def("dcn_v2_psroi_pooling_forward", &dcn_v2_psroi_pooling_forward, 
        "dcn_v2_psroi_pooling_forward");
  m.def("dcn_v2_psroi_pooling_backward", &dcn_v2_psroi_pooling_backward, 
        "dcn_v2_psroi_pooling_backward");
}