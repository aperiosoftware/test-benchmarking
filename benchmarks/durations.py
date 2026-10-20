# Benchmarks spanning roughly 300 ns to 100 ms, and a few different kinds of
# work (pure Python, compute-bound numpy, memory-bound numpy, allocation-heavy,
# file I/O), so that noise can be compared as a function of duration and type.

import os
import tempfile

import numpy as np


class PythonLoop:
    """Pure-Python interpreter loop, ~300 ns to ~3 ms."""

    params = [10, 1_000, 100_000]
    param_names = ["n"]

    def time_sum_range(self, n):
        total = 0
        for i in range(n):
            total += i


class NumpyReduce:
    """Sum of a float64 array: overhead-dominated for small n, memory-bound for large n."""

    params = [10, 10_000, 10_000_000]
    param_names = ["n"]

    def setup(self, n):
        self.array = np.random.default_rng(42).random(n)

    def time_sum(self, n):
        self.array.sum()


class NumpyCompute:
    """Compute-bound numpy work, ~1 ms to ~100 ms."""

    params = [100_000, 1_000_000, 10_000_000]
    param_names = ["n"]

    def setup(self, n):
        self.array = np.random.default_rng(42).random(n)

    def time_sort(self, n):
        np.sort(self.array)

    def time_sin(self, n):
        np.sin(self.array)


class MatMul:
    """Dense matrix multiply, ~5 ms (single-threaded via OPENBLAS_NUM_THREADS in CI)."""

    def setup(self):
        rng = np.random.default_rng(42)
        self.a = rng.random((500, 500))
        self.b = rng.random((500, 500))

    def time_matmul(self):
        self.a @ self.b


class Allocation:
    """Many small Python objects (~15 ms), sensitive to allocator and GC behaviour."""

    def time_list_of_dicts(self):
        [{"a": i, "b": i} for i in range(100_000)]


class FileIO:
    """Write and read back 10 MB (~20 ms), sensitive to filesystem and page cache."""

    def setup(self):
        self.data = os.urandom(10 * 1024 * 1024)
        fd, self.path = tempfile.mkstemp()
        os.close(fd)

    def teardown(self):
        os.remove(self.path)

    def time_write_read(self):
        with open(self.path, "wb") as f:
            f.write(self.data)
        with open(self.path, "rb") as f:
            f.read()
