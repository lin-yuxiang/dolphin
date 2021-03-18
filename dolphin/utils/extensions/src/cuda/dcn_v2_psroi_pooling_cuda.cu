#include "dcn_v2_psroi_pooling_cuda_kernel.cuh"

std::tuple<at::Tensor, at::Tensor>
dcn_v2_psroi_pooling_cuda_forward(const at::Tensor &input,
                                  const at::Tensor &bbox,
                                  const at::Tensor &trans,
                                  const int no_trans,
                                  const float spatial_scale,
                                  const int output_dim,
                                  const int group_size,
                                  const int pooled_size,
                                  const int part_size,
                                  const int sample_per_part,
                                  const float trans_std)
{
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(bbox.type().is_cuda(), "rois must be a CUDA tensor");
  AT_ASSERTM(trans.type().is_cuda(), "trans must be a CUDA tensor");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_trans = no_trans ? 2 : trans.size(1);
  const int num_bbox = bbox.size(0);

  AT_ASSERTM(channels == output_dim, "input channels and output channels must equal");
  auto pooled_height = pooled_size;
  auto pooled_width = pooled_size;

  auto out = at::empty({num_bbox, output_dim, pooled_height, pooled_width}, input.options());
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  auto top_count = at::zeros({num_bbox, output_dim, pooled_height, pooled_width}, input.options());

  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (out.numel() == 0)
  {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(out, top_count);
  }

  dim3 grid(std::min(THCCeilDiv(out_size, 512L), 4096L));
  dim3 block(512);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "dcn_v2_psroi_pooling_cuda_forward", [&] {
    DeformablePSROIPoolForwardKernelCuda<scalar_t><<<grid, block, 0, stream>>>(
        out_size,
        input.contiguous().data<scalar_t>(),
        spatial_scale,
        channels,
        height, width,
        pooled_height,
        pooled_width,
        bbox.contiguous().data<scalar_t>(),
        trans.contiguous().data<scalar_t>(),
        no_trans,
        trans_std,
        sample_per_part,
        output_dim,
        group_size,
        part_size,
        num_classes,
        channels_each_class,
        out.data<scalar_t>(),
        top_count.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(out, top_count);
}

std::tuple<at::Tensor, at::Tensor>
dcn_v2_psroi_pooling_cuda_backward(const at::Tensor &out_grad,
                                   const at::Tensor &input,
                                   const at::Tensor &bbox,
                                   const at::Tensor &trans,
                                   const at::Tensor &top_count,
                                   const int no_trans,
                                   const float spatial_scale,
                                   const int output_dim,
                                   const int group_size,
                                   const int pooled_size,
                                   const int part_size,
                                   const int sample_per_part,
                                   const float trans_std)
{
  AT_ASSERTM(out_grad.type().is_cuda(), "out_grad must be a CUDA tensor");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(bbox.type().is_cuda(), "bbox must be a CUDA tensor");
  AT_ASSERTM(trans.type().is_cuda(), "trans must be a CUDA tensor");
  AT_ASSERTM(top_count.type().is_cuda(), "top_count must be a CUDA tensor");

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  const int channels_trans = no_trans ? 2 : trans.size(1);
  const int num_bbox = bbox.size(0);

  AT_ASSERTM(channels == output_dim, "input channels and output channels must equal");
  auto pooled_height = pooled_size;
  auto pooled_width = pooled_size;
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;

  auto input_grad = at::zeros({batch, channels, height, width}, out_grad.options());
  auto trans_grad = at::zeros_like(trans);

  if (input_grad.numel() == 0)
  {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(input_grad, trans_grad);
  }

  dim3 grid(std::min(THCCeilDiv(out_size, 512L), 4096L));
  dim3 block(512);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(out_grad.type(), "dcn_v2_psroi_pooling_cuda_backward", [&] {
    DeformablePSROIPoolBackwardAccKernelCuda<scalar_t><<<grid, block, 0, stream>>>(
        out_size,
        out_grad.contiguous().data<scalar_t>(),
        top_count.contiguous().data<scalar_t>(),
        num_bbox,
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        output_dim,
        input_grad.contiguous().data<scalar_t>(),
        trans_grad.contiguous().data<scalar_t>(),
        input.contiguous().data<scalar_t>(),
        bbox.contiguous().data<scalar_t>(),
        trans.contiguous().data<scalar_t>(),
        no_trans,
        trans_std,
        sample_per_part,
        group_size,
        part_size,
        num_classes,
        channels_each_class);
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(input_grad, trans_grad);
}