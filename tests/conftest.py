import os

# Multiple OpenMP runtimes ship in the conda env (MKL libiomp5 + torch libomp);
# without these guards torch ops segfault on macOS and Windows. Set before torch import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
