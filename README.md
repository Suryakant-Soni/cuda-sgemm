!git clone https://github.com/Suryakant-Soni/cuda-sgemm.git

%cd cuda-sgemm

!bash scripts/setup_colab.sh        # verifies GPU, nvcc, and cmake

!cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75

!cmake --build build

!./build/device_query               # confirm you're on a T4

!./build/correctness                # verify kernel 1 is correct

!./build/benchmark                   # then measure it