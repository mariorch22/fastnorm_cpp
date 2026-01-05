#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Wichtig für die automatische Umwandlung von std::vector
#include "rmsnorm.h"

namespace py = pybind11;

PYBIND11_MODULE(fastnorm, m) {
    m.doc() = "Mein schnelles RMSNorm Modul"; 
    // Wir registrieren die Funktion
    m.def("rmsnorm", &rms_norm, "Berechnet RMSNorm für einen Vektor");
}