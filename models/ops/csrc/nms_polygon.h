#pragma once
#include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>


#ifdef WITH_CUDA
#include "cuda/vision.h"
at::Tensor nms_polygon_cuda(
  const at::Tensor& dets,
  const at::Tensor& scores,
  float threshold);
#endif

inline at::Tensor nms_polygon(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float threshold) {
  if (dets.type().is_cuda()) {
#ifdef WITH_CUDA
    #include "cuda/vision.h"
    if (dets.numel() == 0) {
      // at::cuda::CUDAGuard device_guard(dets.device());
      return at::empty({0}, dets.options().dtype(at::kLong));
    }
    return nms_polygon_cuda(
        dets,
        scores,
        threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}
