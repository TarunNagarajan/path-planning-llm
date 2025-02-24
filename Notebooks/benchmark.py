import cupy as cp
import numpy as np
import time

# Create a large NumPy array (100 million elements)
cpu_array = np.random.rand(100_000_000)

# Move to GPU
start_gpu = time.time()
gpu_array = cp.array(cpu_array)  # Transfer CPU -> GPU
end_gpu = time.time()

# Move back to CPU
start_cpu = time.time()
cpu_result = cp.asnumpy(gpu_array)  # Transfer GPU -> CPU
end_cpu = time.time()

print(f"CPU -> GPU Transfer Time: {end_gpu - start_gpu:.6f} sec")
print(f"GPU -> CPU Transfer Time: {end_cpu - start_cpu:.6f} sec")

# Measure repeated transfers
repeats = 10
start_repeat = time.time()
for _ in range(repeats):
    gpu_array = cp.array(cpu_array)  # CPU -> GPU
    cpu_result = cp.asnumpy(gpu_array)  # GPU -> CPU
end_repeat = time.time()

print(f"Average Transfer Time (CPU ↔ GPU, {repeats} times): {(end_repeat - start_repeat) / repeats:.6f} sec")
