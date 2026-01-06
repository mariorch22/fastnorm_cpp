#pragma once
#include <torch/torch.h>

namespace fastnorm {

// CPU
torch::Tensor rmsnorm_cpu(torch::Tensor input);

// CUDA
#ifdef WITH_CUDA
torch::Tensor rmsnorm_cuda(torch::Tensor input);
#endif

}  // namespace fastnorm