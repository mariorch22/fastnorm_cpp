#include <cmath>
#include <vector>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void compute_rmsnorm(const float* input, float* output, size_t n) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }
    
    float inv_rms = 1.0f / std::sqrt(ss / n + 1e-6f);
    
    for (size_t i = 0; i < n; i++) {
        output[i] = input[i] * inv_rms;
    }
}

std::vector<float> rms_norm(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    compute_rmsnorm(input.data(), output.data(), input.size());
    return output;
}

py::array_t<float> rms_norm_numpy(py::array_t<float> input) {
    auto buf = input.request();
    auto result = py::array_t<float>(buf.size);
    compute_rmsnorm(static_cast<float*>(buf.ptr), 
                    static_cast<float*>(result.request().ptr), 
                    buf.size);
    return result;
}