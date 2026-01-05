#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rmsnorm.h"

namespace py = pybind11;

PYBIND11_MODULE(fastnorm, m) {
    m.doc() = "Mein schnelles RMSNorm Modul"; 
    m.attr("__version__") = "0.1.3";
    m.def("rmsnorm", &rms_norm, "Berechnet RMSNorm für einen Vektor");
    m.def("rmsnorm_numpy", &rms_norm_numpy, "Berechnet RMSNorm direkt auf NumPy-Speicher (Zero-Copy)");
}