#include "sgemm/utils.cuh"

#include <cstdio>

int main() {
    int device = 0;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("=== CUDA Device Query ===\n\n");

    // ------------------------------------------------------------
    // 1. Identity
    // ------------------------------------------------------------
    std::printf("[Identity]\n");
    std::printf("Device: %s\n", prop.name);
    std::printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);

    // ------------------------------------------------------------
    // 2. Memory bandwidth estimate
    //
    // prop.memoryClockRate is in kHz.
    // prop.memoryBusWidth is in bits.
    //
    // Bandwidth formula:
    //
    //   bandwidth = 2 * memory_clock_hz * bus_width_bytes
    //
    // The factor 2 is because GPU memory is DDR:
    // double data rate, two transfers per clock.
    // ------------------------------------------------------------
    double memory_clock_hz = static_cast<double>(prop.memoryClockRate) * 1000.0;
    double bus_width_bytes = static_cast<double>(prop.memoryBusWidth) / 8.0;

    double peak_bandwidth_gb_s =
        2.0 * memory_clock_hz * bus_width_bytes / 1.0e9;

    std::printf("[Memory Bandwidth]\n");
    std::printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
    std::printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    std::printf("Approx peak bandwidth: %.2f GB/s\n\n", peak_bandwidth_gb_s);

    // ------------------------------------------------------------
    // 3. Execution / occupancy limits
    // ------------------------------------------------------------
    std::printf("[Execution / Occupancy Limits]\n");
    std::printf("SM count: %d\n", prop.multiProcessorCount);
    std::printf("Warp size: %d\n", prop.warpSize);
    std::printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    std::printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("Registers per block: %d\n", prop.regsPerBlock);
    std::printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    std::printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    std::printf("Shared memory per SM: %zu bytes\n\n", prop.sharedMemPerMultiprocessor);

    // ------------------------------------------------------------
    // 4. Memory hierarchy / capacity
    // ------------------------------------------------------------
    double global_mem_gib =
        static_cast<double>(prop.totalGlobalMem) /
        (1024.0 * 1024.0 * 1024.0);

    double l2_cache_kib =
        static_cast<double>(prop.l2CacheSize) / 1024.0;

    std::printf("[Memory]\n");
    std::printf("Total global memory: %.2f GiB\n", global_mem_gib);
    std::printf("L2 cache size: %.2f KiB\n", l2_cache_kib);

    return 0;
}