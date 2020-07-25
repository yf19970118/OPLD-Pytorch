//// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//#pragma once
//#include "cpu/vision.h"
//
//#ifdef WITH_CUDA
//#include "cuda/vision.h"
//#endif
//
//
//inline at::Tensor nms_rotated(const at::Tensor& dets,
//               const at::Tensor& scores,
//               const float threshold) {
//
//  if (dets.type().is_cuda()) {
//#ifdef WITH_CUDA
//    // TODO raise error if not compiled with CUDA
//    if (dets.numel() == 0)
//      return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
////    auto b = at::cat({dets, scores.unsqueeze(1)}, 1);
//    return nms_rotated_cuda(dets, scores, threshold);
//#else
//    AT_ERROR("Not compiled with GPU support");
//#endif
//  }
//
//  at::Tensor result = nms_rotated_cpu(dets, scores, threshold);
//  return result;
//}

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/extension.h>
#include "cpu/vision.h"

//namespace detectron2 {

at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);

#ifdef WITH_CUDA
#include "cuda/vision.h"
at::Tensor nms_rotated_cuda(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor nms_rotated(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#ifdef WITH_CUDA
    #include "cuda/vision.h"
    return nms_rotated_cuda(dets, scores, iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return nms_rotated_cpu(dets, scores, iou_threshold);
}

//} // namespace detectron2
