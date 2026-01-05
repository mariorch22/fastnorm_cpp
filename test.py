import timeit
import numpy as np
import fastnorm
import math

# 1. Native Python Implementierung (Baseline)
def rmsnorm_python(x, eps=1e-6):
    n = len(x)
    ss = sum(v * v for v in x)
    inv_rms = 1.0 / math.sqrt(ss / n + eps)
    return [v * inv_rms for v in x]

# 2. Setup für den Test
size = 1_000_000  # 1 Million Elemente
data_list = [float(i) for i in range(size)]
data_np = np.array(data_list, dtype=np.float32)

print(f"Starte Benchmark mit {size} Elementen...\n")

# 3. Messungen
# Native Python
t_py = timeit.timeit(lambda: rmsnorm_python(data_list), number=10) / 10
print(f"Native Python:      {t_py*1000:.2f} ms")

# C++ mit Python-Listen
t_cpp_list = timeit.timeit(lambda: fastnorm.rmsnorm(data_list), number=10) / 10
print(f"C++ (Listen-Weg):   {t_cpp_list*1000:.2f} ms")

# C++ mit NumPy (Zero-Copy)
t_cpp_np = timeit.timeit(lambda: fastnorm.rmsnorm_numpy(data_np), number=10) / 10
print(f"C++ (NumPy-Weg):    {t_cpp_np*1000:.2f} ms")

# 4. Auswertung
speedup = t_py / t_cpp_np
print(f"\nErgebnis: Die C++ NumPy Version ist {speedup:.1f}x schneller als natives Python!")