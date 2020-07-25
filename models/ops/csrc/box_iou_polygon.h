#pragma once
#include <torch/extension.h>


#ifdef WITH_CUDA
#include "cuda/vision.h"
at::Tensor box_iou_polygon_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor box_iou_polygon(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#ifdef WITH_CUDA
    #include "cuda/vision.h"
    return box_iou_polygon_cuda(boxes1, boxes2);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not implemented");
}
