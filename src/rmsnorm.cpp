#include "rmsnorm.h"
#include <cmath>

// rmsnorm.cpp
std::vector<float> rms_norm(std::vector<float> x) {
    float sum_sq = 0.0;
    for (float val : x) sum_sq += val * val;
    float rms = std::sqrt(sum_sq / x.size() + 1e-6);

    for (float& val : x) {
        val /= rms;
    }
    return x;
}