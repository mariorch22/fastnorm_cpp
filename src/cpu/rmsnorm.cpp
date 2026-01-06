#include <cmath>
#include <torch/torch.h>
#include "fastnorm/rmsnorm.h"

namespace fastnorm {

// Normalize a single vector
// input: pointer to the input vector
// output: pointer to the output vector
// n: size of the vector
// eps: epsilon to avoid division by zero
void compute_rmsnorm_cpu(const float* input, float* output, size_t n, float eps = 1e-6f) {
    // define float to store the sum of the squares of the input vector
    float ss = 0.0f;

    // sum the squares of the input vector (SUM(x^2) over all elements)
    for (size_t i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }

    // calculate the inverse of the root mean square of the input vector
    float inv_rms = 1.0f / std::sqrt(ss / n + eps);

    // multiply the input vector by the inverse of the root mean square
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * inv_rms;
    }
}

// Normalize the last dimension of a PyTorch tensor
torch::Tensor rmsnorm_cpu(torch::Tensor input) {
    // Convert to input tensor to float32 (could be different in pytorch) and contiguous
    auto x = input.to(torch::kFloat32).contiguous();
    // Create empty output tensor with the same shape as the input
    auto output = torch::empty_like(x);
    
    // Get the size of the last dimension (norm_size)
    int64_t norm_size = x.size(-1);
    // Calculate the number of vectors to normalize
    int64_t num_vectors = x.numel() / norm_size;
    
    // Get the pointers to the input and output tensors
    float* in_ptr = x.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_vectors; i++) {
        compute_rmsnorm_cpu(in_ptr + i * norm_size, out_ptr + i * norm_size, norm_size);
    }
    
    return output;
}

}  // namespace fastnorm