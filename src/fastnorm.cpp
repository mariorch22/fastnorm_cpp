#include <torch/extension.h>
#include "fastnorm/rmsnorm.h"

torch::Tensor rmsnorm(torch::Tensor input) {
    if (input.is_cuda()) {
        #ifdef WITH_CUDA
        return fastnorm::rmsnorm_cuda(input);
        #else
        TORCH_CHECK(false, "fastnorm compiled without CUDA support");
        #endif
    }
    return fastnorm::rmsnorm_cpu(input);
}

PYBIND11_MODULE(fastnorm, m) {
    m.doc() = "Fast RMSNorm";
    m.def("rmsnorm", &rmsnorm, "RMSNorm (CPU/CUDA)");
}