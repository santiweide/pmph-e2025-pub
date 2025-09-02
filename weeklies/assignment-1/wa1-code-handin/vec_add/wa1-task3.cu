#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief Adds two vectors element-wise on the CPU.
 * @param a Input vector A.
 * @param b Input vector B.
 * @param c Output vector C (result of A + B).
 * @param n Number of elements in the vectors.
 */
void addVectorsCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief CUDA kernel to add two vectors element-wise on the GPU.
 * Each thread computes one element of the output vector.
 * @param a Input vector A (device pointer).
 * @param b Input vector B (device pointer).
 * @param c Output vector C (device pointer).
 * @param n Number of elements in the vectors.
 */
__global__ void addVectorsGPU(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief Validates if two float arrays are element-wise equal within a given epsilon.
 * @param cpu_res Result computed on the CPU.
 * @param gpu_res Result computed on the GPU.
 * @param n Number of elements.
 * @param epsilon Tolerance for floating point comparison.
 */
void validateResults(const float* cpu_res, const float* gpu_res, int n, float epsilon) {
    for (int i = 0; i < n; ++i) {
        if (fabs(cpu_res[i] - gpu_res[i]) > epsilon) {
            fprintf(stderr, "INVALID: Mismatch at index %d: CPU=%.6f, GPU=%.6f\n", i, cpu_res[i], gpu_res[i]);
            return;
        }
    }
    printf("VALID\n");
}

int main(int argc, char** argv) {

    int64_t n = 753411;

    if (argc > 1) {
        n = atoll(argv[1]);
        printf("Using custom vector length: %lld\n", n);
    } else {
        printf("Using default vector length: %lld\n", n);
    }
    
    const float epsilon = 1.0e-6f;
    const int gpu_runs = 300;
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c_cpu = (float*)malloc(bytes);
    h_c_gpu = (float*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    struct timeval start_cpu, end_cpu;
    gettimeofday(&start_cpu, NULL);
    
    addVectorsCPU(h_a, h_b, h_c_cpu, n);
    
    gettimeofday(&end_cpu, NULL);
    double cpu_time_ms = (end_cpu.tv_sec - start_cpu.tv_sec) * 1000.0 + (end_cpu.tv_usec - start_cpu.tv_usec) / 1000.0;
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    struct timeval start_gpu, end_gpu;
    gettimeofday(&start_gpu, NULL);

    for (int i = 0; i < gpu_runs; ++i) {
        addVectorsGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    gettimeofday(&end_gpu, NULL);
    double total_gpu_time_ms = (end_gpu.tv_sec - start_gpu.tv_sec) * 1000.0 + (end_gpu.tv_usec - start_gpu.tv_usec) / 1000.0;
    double avg_gpu_time_ms = total_gpu_time_ms / gpu_runs;

    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    
    printf("\n--- Validation ---\n");
    validateResults(h_c_cpu, h_c_gpu, n, epsilon);
    printf("Epsilon used for validation: %e\n", epsilon);

    // Calculate memory throughput
    // (Read A + Read B + Write C) = 3 * n * sizeof(float) bytes per call
    double total_bytes_accessed = 3.0 * n * sizeof(float);
    double gpu_throughput_gb_s = (total_bytes_accessed / (avg_gpu_time_ms / 1000.0)) / 1e9;
    
    double speedup = cpu_time_ms / avg_gpu_time_ms;
    
    printf("\n--- Performance ---\n");
    printf("CPU (1 run)           : %.3f ms\n", cpu_time_ms);
    printf("GPU (avg of %d runs)  : %.3f ms\n", gpu_runs, avg_gpu_time_ms);
    printf("Speedup (CPU/GPU)     : %.2fx\n", speedup);
    printf("GPU Memory Throughput   : %.3f GB/s\n", gpu_throughput_gb_s);
    
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}

