#ifndef RMSNORM_H
#define RMSNORM_H

#include <vector>
#include <pybind11/numpy.h>

void compute_rmsnorm(const float* input, float* output, size_t n);

// Vector Binding
std::vector<float> rms_norm(const std::vector<float>& input);

// NumPy Binding
pybind11::array_t<float> rms_norm_numpy(pybind11::array_t<float> input);

#endif