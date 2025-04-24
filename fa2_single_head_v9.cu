#include <iostream>
#include <fstream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cassert>

#include <cuda_runtime.h>

/*
Implementation of Flash Attention 2
https://arxiv.org/pdf/2307.08691

Q: Mxd
K: Nxd
V: Nxd
O: Mxd
l,m: Mx1
*/

#define FULL_MASK 0xffffffff

// A10G GPU has max 99KB of shared memory available per block
constexpr int A10G_SRAM_SIZE = 99 * 1024;
// We set SRAM SIZE for the algorithm to 25000 floats
// This is slightly smaller to fit the additional li/mi vectors too
constexpr int SRAM_SIZE = 25000;
constexpr int THREADS_PER_WARP = 32;
constexpr float NEGATIVE_INF = std::numeric_limits<float>::lowest();

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * Load a block of the matrix from src to dst.
 * src intended to be global, dst intended to be shared memory.
 * Matrix is size MxN
 */
__device__ void matrix_block_load(
    float* dst, 
    const float* src, 
    int M,
    int N,
    int block_size,
    int block_idx
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = M * N;
    int block_start = block_idx * block_size * N;
    int block_end = block_start + block_size * N;
    for (int i = block_start + tid; i < block_end; i += num_threads) {
        dst[i - block_start] = (i < num_elts) ? src[i] : 0;
    }
}

/**
 * Load a block of the matrix from src to dst, and transpose it.
 * src intended to be global, dst intended to be shared memory.
 * Matrix is size MxN
 * Loop block size is the size of the current block, either block size
 * or can be less for the very last block if block size not a divisor of M.
 * Matrix stored in dst will be size N x loop_block_size.
 */
 __device__ void matrix_block_load_transpose(
    float* dst,
    const float* src, 
    int M,
    int N,
    int block_size,
    int loop_block_size,
    int block_idx
) {

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = M * N;
    int block_start = block_idx * block_size * N;
    int block_end = block_start + block_size * N;
    for (int i = block_start + tid; i < min(block_end, num_elts); i += num_threads) {
        int r = (i - block_start) / N;
        int c = i % N;
        dst[c * loop_block_size + r] = src[i];
    }
}

/**
 * Store src into a block of dst.
 * src intended to be shared memory, dst intended to be global.
 * dst is size M x N, src is size block_size x N.
 */
__device__ void matrix_block_store(
    float* dst, 
    const float* src, 
    int M,
    int N,
    int block_size,
    int block_idx
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int block_start = block_idx * block_size * N;
    int block_end = min(M * N, block_start + block_size * N);
    for (int i = block_start + tid; i < block_end; i += num_threads) {
        dst[i] = src[i - block_start];
    }
}

/**
 * Fill array of size N with fill_value.
 */
__device__ void array_fill(
    float* array,
    float fill_value,
    int N
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < N; i += num_threads) {
        array[i] = fill_value;
    }
}

/**
 * Computes matrix multiplication A*B.
 * A is of size MxK, B is of size KxN.
 * Output C is of size MxN.
 * If add_to_output, A*B is added to C instead of overwriting it.
 * A, B, C are in shared memory. This assumes a single
 * block of 1024 threads, and 32 threads per warp.
 * Implements 2d blocktiling, inspired by 
 * https://siboehm.com/articles/22/CUDA-MMM.
 * This can still be optimized by using vectorized loads/stores
 * as well as avoiding bank conflicts.
 */
template <bool add_to_output = false>
__device__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C, 
    int M,
    int N,
    int K
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    constexpr int warp_size = 32;
    int num_warps = num_threads / warp_size;

    constexpr int T = 4;
    for (int sr = T * warp_id; sr < M; sr += num_warps * T) {
        int BA = min(T, M - sr);
        for (int sc = T * lane_id; sc < N; sc += warp_size * T) {
            int BB = min(T, N - sc);

            float Areg[T] = {0.0};
            float Breg[T] = {0.0};
            float Creg[T*T] = {0.0};
            for (int k = 0; k < K; k++) {
                for (int i = 0; i < T; i++) {
                    Areg[i] = A[(sr + i) * K + k];
                }
                for (int i = 0; i < T; i++) {
                    Breg[i] = B[k * N + sc + i];
                }
                for (int i = 0; i < T; i++) {
                    for (int j = 0; j < T; j++) {
                        Creg[i * T + j] += Areg[i] * Breg[j];
                    }
                }
            }
            for (int i = 0; i < BA; i++) {
                for (int j = 0; j < BB; j++) {
                    if constexpr (add_to_output) {
                        C[(sr + i) * N + (sc + j)] += Creg[i * T + j];
                    } else {
                        C[(sr + i) * N + (sc + j)] = Creg[i * T + j];
                    }
                }
            }
        }
    }
}

/**
 * Divides Si by d, and assigns mi_cur to max(mi_prev, rowmax(Si)).
 * mi_cur / mi_prev are vectors of size Br in smem.
 * Si is a matrix of size Br x Bc in smem.
 */
__device__ void divide_by_d_and_mi_update(
    float* mi_cur,
    const float* mi_prev,
    float* Si,
    int Br,
    int Bc,
    float sqrtd
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;
    for (int i = warp_id; i < Br; i += num_warps) {
        float max_val = mi_prev[i];
        for (int j = lane_id; j < Bc; j += THREADS_PER_WARP) {
            float Si_tmp = Si[i * Bc + j] / sqrtd;
            max_val = max(max_val, Si_tmp);
            Si[i * Bc + j] = Si_tmp;
        }
        // reduction across warp to get full mask
        for (int j = THREADS_PER_WARP / 2; j >= 1; j >>= 1) {
            max_val = max(max_val, __shfl_xor_sync(FULL_MASK, max_val, j));
        }
        if (lane_id == 0) {
            mi_cur[i] = max_val;
        }
    }
}

/**
 * Converts Si to Pi, where Pi = exp(Si - mi).
 * Si is a matrix of size Br x Bc in smem.
 * Pi is a matrix of size Br x Bc in smem.
 * mi is a vector of size Br in smem.
 */
__device__ void si_to_pi(
    float* SiPi,
    const float* mi,
    int Br,
    int Bc
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = tid; i < Br * Bc; i += num_threads) {
        int r = i / Bc;
        SiPi[i] = exp(SiPi[i] - mi[r]);
    }
}

/**
 * Update li to exp(mi_prev - mi_cur) * li + rowsum(Pi).
 * li is a vector of size Br in smem.
 * Pi is a matrix of size Br x Bc in smem.
 * mi_prev is a vector of size Br in smem.
 * mi_cur is a vector of size Br in smem.
 */
__device__ void li_update(
    float* li,
    const float* Pi,
    const float* mi_prev,
    const float* mi_cur,
    int Br,
    int Bc
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_threads = blockDim.x;
    int num_warps = num_threads / THREADS_PER_WARP;
    for (int i = warp_id; i < Br; i += num_warps) {
        float sum = 0;
        for (int j = lane_id; j < Bc; j += THREADS_PER_WARP) {
            sum += Pi[i * Bc + j];
        }
        // reduction across warp to get total sum
        for (int j = THREADS_PER_WARP / 2; j >= 1; j >>= 1) {
            sum += __shfl_xor_sync(FULL_MASK, sum, j);
        }
        if (lane_id == 0) {
            li[i] = exp(mi_prev[i] - mi_cur[i]) * li[i] + sum;
        }
    }
}

/**
 * Update Oi to diag(exp(mi_prev - mi_cur)) * Oi + Pi * V.
 * Oi is a matrix of size Br x d in smem.
 * Pi is a matrix of size Br x Bc in smem.
 * V is a matrix of size Bc x d in smem.
 * mi_prev, mi_cur are vectors of size Br in smem.
 */
__device__ void Oi_update(
    float* Oi,
    const float* Pi,
    const float* VT,
    const float* mi_prev,
    const float* mi_cur,
    int Br,
    int Bc,
    int d
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = Br * d;
    for (int i = tid; i < num_elts; i += num_threads) {
        int r = i / d;
        Oi[i] *= exp(mi_prev[r] - mi_cur[r]);
    }
    matrix_multiply<true>(Pi, VT, Oi, Br, d, Bc);
}

/**
 * Divide each row of Oi by that value of li.
 * Oi is a matrix of size Br x d in smem.
 * li is a vector of size Br in smem.
 */
__device__ void Oi_scale(
    float* Oi,
    const float* li,
    int Br,
    int d
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_elts = Br * d;
    for (int i = tid; i < num_elts; i += num_threads) {
        int r = i / d;
        Oi[i] /= li[r];
    }
}

__global__ void flash_attention_2_kernel(
    const float* Q, 
    const float* K, 
    const float* V, 
    float* O, 
    const int M, 
    const int N, 
    const int d, 
    const int Br, 
    const int Bc, 
    const int Tr, 
    const int Tc,
    const int alloc_size
) {
    extern __shared__ float s[];
    float *Oi = s;
    float *Qi = &s[alloc_size];
    // will first store Ki, then get overriden to ViT
    float *KiVi = &s[2 * alloc_size];
    // will first store Si, then get overriden to Pi
    float *SiPi = &s[3 * alloc_size];
    float *li = &s[4 * alloc_size];
    float *mi = &s[4 * alloc_size + Br];
    float *mi2 = &s[4 * alloc_size + 2 * Br];

    float* mi_prev = mi; // m(i,j-1)
    float* mi_cur = mi2; // m(i,j)

    int i = blockIdx.x;
    int loopBr = min(Br, M - i * Br);
    matrix_block_load(Qi, Q, M, d, Br, i);
    array_fill(Oi, 0, loopBr * d);
    array_fill(li, 0, loopBr);
    array_fill(mi_prev, NEGATIVE_INF, loopBr);
    __syncthreads();
    for (int j = 0; j < Tc; j++) {
        int loopBc = min(Bc, N - j * Bc);
        matrix_block_load_transpose(KiVi, K, N, d, Bc, loopBc, j);
        __syncthreads();
        matrix_multiply(Qi, KiVi, SiPi, loopBr, loopBc, d);
        __syncthreads();
        divide_by_d_and_mi_update(mi_cur, mi_prev, SiPi, loopBr, loopBc, sqrtf(d));
        __syncthreads();
        si_to_pi(SiPi, mi_cur, loopBr, loopBc);
        __syncthreads();
        li_update(li, SiPi, mi_prev, mi_cur, loopBr, loopBc);
        matrix_block_load(KiVi, V, N, d, Bc, j);
        __syncthreads();
        Oi_update(Oi, SiPi, KiVi, mi_prev, mi_cur, loopBr, loopBc, d);
        __syncthreads();

        // swap mi_prev / mi_cur
        auto tmp = mi_prev;
        mi_prev = mi_cur;
        mi_cur = tmp;
    }
    Oi_scale(Oi, li, loopBr, d);
    __syncthreads();
    matrix_block_store(O, Oi, M, d, Br, i);
}

// Q, K, V, O are device pointers
void flash_attention_2(const float* Q, const float* K, const float* V, float* O, int M, int N, int d) {
    int Bc = ceildiv(SRAM_SIZE, 4 * d);
    int Br = min(Bc, d);
    int Tr = ceildiv(M, Br);
    int Tc = ceildiv(N, Bc);

    int alloc_size = max(Br * Bc, Bc * d);
    int shmem_needed = (4 * alloc_size + 3 * Br) * sizeof(float);

    // call kernel
    const int threadsPerBlock = 512;
    const int blocksPerGrid = Tr;
    std::cout << "Shared memory needed: " << shmem_needed << " bytes" << std::endl;
    flash_attention_2_kernel<<<blocksPerGrid, threadsPerBlock, shmem_needed>>>(
        Q, K, V, O, M, N, d, Br, Bc, Tr, Tc, alloc_size
    );
}

int main(int argc, char* argv[]) {
    // Print device properties
    int device_id = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device ID: " << device_id << std::endl;
    std::cout << "Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
    std::cout << "Device name: " << device_prop.name << std::endl;
    std::cout << "Total global memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total number of multiprocessors: " << device_prop.multiProcessorCount << std::endl;
    std::cout << "Shared memory per block: " << device_prop.sharedMemPerBlock << " KB" << std::endl;
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dim: " << device_prop.maxThreadsDim[0] << ", " << device_prop.maxThreadsDim[1] << ", " << device_prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid size: " << device_prop.maxGridSize[0] << ", " << device_prop.maxGridSize[1] << ", " << device_prop.maxGridSize[2] << std::endl;
    std::cout << "Warp size: " << device_prop.warpSize << std::endl;
    std::cout << "Max threads per multiprocessor: " << device_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max shared memory per multiprocessor: " << device_prop.sharedMemPerMultiprocessor / (1024) << " KB" << std::endl;
    std::cout << "Max registers per multiprocessor: " << device_prop.regsPerMultiprocessor << std::endl;

    // We're using A10G GPU, which has 99KB available shared memory per block
    cudaFuncSetAttribute(flash_attention_2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, A10G_SRAM_SIZE);

    // Benchmark parameters
    constexpr int M = 10000;
    constexpr int N = 9000;
    constexpr int d = 32;

    std::cout << "M: " << M << ", N: " << N << ", d: " << d << std::endl;

    // Initialize a testcase
    float* Q = new float[M * d];
    float* K = new float[N * d];
    float* V = new float[N * d];
    float* O = new float[M * d];
    for (int i = 0; i < M * d; ++i) {
        Q[i] = static_cast<float>(i) / (M * d);
    }
    for (int i = 0; i < N * d; ++i) {
        K[i] = static_cast<float>(i) * 2 / (N * d);
        V[i] = static_cast<float>(i) * 3 / (N * d);
    }

    std::cout << "Matrices initialized on CPU." << std::endl;
    
    // Allocate memory on the device
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void**)&d_Q, M * d * sizeof(float));
    cudaMalloc((void**)&d_K, N * d * sizeof(float));
    cudaMalloc((void**)&d_V, N * d * sizeof(float));
    cudaMalloc((void**)&d_O, M * d * sizeof(float));

    // Copy input vectors from host to device
    cudaMemcpy(d_Q, Q, M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Matrices copied to GPU, running kernel..." << std::endl;

    // Call Flash Attention 2
    flash_attention_2(d_Q, d_K, d_V, d_O, M, N, d);

    // Copy result from device to host
    cudaMemcpy(O, d_O, M * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or write output
    if (argc > 1) {
        std::ofstream outfile(argv[1]);
        if (outfile.is_open()) {
            for (int i = 0; i < M * d; ++i) {
                outfile << std::setprecision(10) << O[i];
                if ((i + 1) % d == 0) {
                    outfile << "\n";
                } else {
                    outfile << " ";
                }
            }
            outfile.close();
            std::cout << "Output written to " << argv[1] << std::endl;
        } else {
            std::cerr << "Failed to open file: " << argv[1] << std::endl;
        }
    } else {
        std::cout << "Output:" << std::endl;
        for (int i = 0; i < 10 && i < M * d; ++i) {
            std::cout << std::setprecision(10) << O[i] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Free allocated memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    // Free dynamically allocated memory
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] O;

    return 0;
}
