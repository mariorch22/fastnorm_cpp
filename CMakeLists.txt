cmake_minimum_required(VERSION 3.10)
project(fastnorm_lib)

# 1. Suche nach pybind11 in der Python-Umgebung
find_package(pybind11 REQUIRED)

# 2. Das Python-Modul erstellen (statt add_executable)
# "fastnorm" ist der Name, den du später in Python importierst
pybind11_add_module(fastnorm src/bindings.cpp src/rmsnorm.cpp)

# 3. Header-Pfad angeben
target_include_directories(fastnorm PRIVATE include)

install(TARGETS fastnorm DESTINATION .)