#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../box_iou_polygon_utils.h"
#include "../nms_polygon.h"


int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__global__ void nms_polyhon_kernel(
    const int n_polys,
    const float iou_threshold,
    const T* dev_polys,
    unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_polys - row_start * threadsPerBlock, threadsPerBlock);
  const int cols_size =
      min(n_polys - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ T block_polys[threadsPerBlock * 8];
  if (threadIdx.x < cols_size) {
    block_polys[threadIdx.x * 8 + 0] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 0];
    block_polys[threadIdx.x * 8 + 1] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 1];
    block_polys[threadIdx.x * 8 + 2] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 2];
    block_polys[threadIdx.x * 8 + 3] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 3];
    block_polys[threadIdx.x * 8 + 4] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 4];
    block_polys[threadIdx.x * 8 + 5] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 5];
    block_polys[threadIdx.x * 8 + 6] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 6];
    block_polys[threadIdx.x * 8 + 7] =
        dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 8 + 7];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_polys + cur_box_idx * 8;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
        start = threadIdx.x + 1;
    }
    for (i = start; i < cols_size; i++) {
      if (single_poly_iou<T>(cur_box, block_polys + i * 8) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = at::cuda::ATenCeilDiv(n_polys, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


at::Tensor nms_polygon_cuda(
    const at::Tensor& dets,
    const at::Tensor & scores,
    float iou_threshold) {
  // using scalar_t = float;
  AT_ASSERTM(dets.type().is_cuda(), "dets must be a CUDA tensor");
  AT_ASSERTM(scores.type().is_cuda(), "scores must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(dets.device());

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t);

  int dets_num = dets.size(0);

  const int col_blocks = at::cuda::ATenCeilDiv(dets_num, threadsPerBlock);

  at::Tensor mask =
      at::empty({dets_num * col_blocks}, dets.options().dtype(at::kLong));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dets_sorted.scalar_type(), "nms_polygon_kernel_cuda", [&] {
        nms_polyhon_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            dets_num,
            iou_threshold,
            dets_sorted.data<scalar_t>(),
            (unsigned long long*)mask.data<int64_t>());
      });

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data<int64_t>();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return order_t.index(
    {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
        .to(order_t.device(), keep.scalar_type())});
}