## PMPH Assignment 2

by Wanjing Hu(fng685)

Device name: NVIDIA A100-PCIE-40GB
Number of hardware threads: 221184
Max block size: 1024
Shared memory size: 49152
RUNS_GPU: 500
ELEMS_PER_THREAD: 24(except task 4).

### Task1 Flat Implementation of Prime-Numbers Computation in Futhark

#### 1.1 Validation status

The implementation passed on both dataset with `futhark test --backend=cuda primes-flat.fut`:

![1.1 Validation Result](https://raw.githubusercontent.com/santiweide/pmph-e2025-pub/refs/heads/main/weeklies/assignment-2/image.png)

Large dataset (N = 1e7): After generating ref10000000.out with primes-seq.fut as instructed, futhark test will also validate the flat version against that reference (the file already contains the validation stanza).

#### 1.2 Code Explanation

In each iteration with bound len, for every prime `p` in `sqrn_primes` I can form an inner list: `L_p = [2p, 3p, ..., floor(len/p)p]`. Then I concatenate all `L_p` to a single vector not_primes, and scatter zeros at those indices into an all-ones flag vector of length `len+1`. Finally, I filter the indices that remain true to obtain the new `sqrn_primes` for the next (squared) len.

Here:

I have `mult_lens` are exactly the lengths of the irregular inner lists `arr = [2..m]` (multiplier space) per prime p.

`seg_starts` and the `heads+scan` realise the segment descriptor that replaces the irregular structure by indices into a flat array.

`seg_ends` is the inclusive scan of lengths.


```haskell
      -- 
      let seg_ends    = scan (+) 0 mult_lens
      let seg_starts  = map2 (-) seg_ends mult_lens

      -- Build "heads" with the same [S] dimension for indices and values to keep shape certain
      let ones        = map (const 1i64) seg_starts
      let heads       = scatter (replicate flat_size 0i64) seg_starts ones

    -- label each flat position with its segment id
      let seg_ids_inc = scan (+) 0 heads
      let seg_ids     = map (\x -> x - 1i64) seg_ids_inc
```

`j_vals` reconstruct the per-segment multiplier `(2..m)` for each flat element, and multiplying by `p_flat` yields the precise positions I would have produced in the nested `map/concat` comprehension.

```haskell
      let p_flat      = map (\sid -> sq_primes[sid]) seg_ids
      let start_flat  = map (\sid -> seg_starts[sid]) seg_ids
      let idx_flat    = iota flat_size
      let j_vals      = map (+2i64) (map2 (-) idx_flat start_flat)

      let not_primes  = map2 (*) p_flat j_vals
```

### 1.3 Performance on large dataset(1e7)
Here I use the work-depth model to analysis the performance expectation.
Work `W(n)` is the total number of primitive operations performed by the algorithm on an input of size `n`, summing across all parallel branches.

#### 1.3.1 Work analysis
The `primes-seq`, `primes-native` and `primes-flat` all have a work of $W(n)  =  \Theta \big(n \log\ log n\big).$ They directly construct the indices to be eliminated (the sequence of multiples for each base p), set these positions to zero at once; avoid performing division tests on each number individually.

Calculation: 

Work per iteration:
  
  $\sum_{p\in \text{primes} \le \texttt{len}}
  \left(\left\lfloor \frac{\texttt{len}}{p}\right\rfloor - 1\right)
   =  \Theta  \big(\texttt{len} \log\log \texttt{len}\big),$

The number of iterations: $\lceil \log_{2}\log_{2} n \rceil$

Total work: $W(n) = \Theta  \big(n \log\log n\big).$

The ad-hoc version does not meet my expectation. For each candidate i in the interval, this version tests divide it by all primes p in the "known prime table acc." Only retain i if none of these primes divide it evenly. So for the $k_th$ iteration, $W_k=|is_k|*|acc_k|$. And the total workload is larger than the direct filtering strategy.

#### 1.3.2 Depth analysis
Since the Depth treats map/scan/filter/scatter as constant-span primitives, I have depth of `prime-flat` and `prime-adhoc` as $\Theta  \big(\log\log n\big)$.

The depth of `prime-naive` is $\Theta  \big(\sqrt n\big)$, because its outer loop goes from 2 to $\sqrt n$, and inside the loop is `map/scatter` which contributes as constant.

The depth of `prime-seq` is fully sequencial so depth equals to the work: $D(n)  =  \Theta  \big(n \log\log n\big).$

#### 1.3.3 Experiment Result

Testing parallel basic blocks
N = 100003565
block size = 256
ELEMS_PER_THREAD = 24

| Implementation   | Backend | Avg time (ms) | Performance      |
| ---------------- | ------- | ------------- | -------------------------------------------------------------------- |
| primes-seq.fut   | C       | 183.570       | SloI st (no parallelism).        |
| primes-adhoc.fut | CUDA    | 183.319       | Optimal depth but more total work; can win on GPU due to regularity. |
| primes-naive.fut | CUDA    | 58.012        | Faster than seq            |
| primes-flat.fut  | CUDA    | 1.554.        | Fastest    |


## Task2 Copying from/to Global to/from Shared Memory in Coalesced Fashion
1) one-line replacement:
 I change from `uint32_t loc_ind = threadIdx.x * CHUNK + i;` to `uint32_t loc_ind = i * blockDim.x + threadIdx.x;`, so that i is accessed with Row-major and can coalesced across threads

2) In the new code, for a continuous section of threadIdx.x, the accessed `loc_ind` is also continuous, and so memory controller can coalesce the memory access into full-width transactions. The previous layout made neighboring threads stride by CHUNK, which is not able to be coalesced.

3) According to the experiment `Coalesced ON & Warp OFF ` vs `Coalesced OFF & Warp OFF`, the following tests have an improvement of bandwidth:

| Test Case                                | GMem-SMem Coalesced (GB/s) | Baseline (GB/s) | Relative Gain (%) |
|------------------------------------------|---------------------------|----------------------|-------------------|
| Optimized Reduce – MSSP                  | 373.85                   | 268.47              | +39.25%           |
| Scan Inclusive AddI32                    | 925.96                   | 329.68              | +180.86%            |
| Segmented Scan Inclusive AddI32          | 1038.61                  | 461.15             | +125.22%             |

The MSSP is not communitative, and AddInt32 is communicative. So when do the optimized map reduce they are of different implementations.

## Task3 Implement Inclusive Scan at WARP Level

1) Implementation: we do a warp level reduce add, because all the threads in a wrap are naturally synchronized. 

```c++
template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const uint32_t idx ) {
    const uint32_t lane = idx & (WARP-1);
    auto v = OP::remVolatile(ptr[idx]); 
    ptr[idx] = v;
    for (int offset = 1; offset < WARP; offset <<= 1) {
        if (lane >= offset) {
            v = OP::apply(ptr[idx - offset], ptr[idx]);
            ptr[idx] = v;
        }
    }
    return v;
}
```

2) Performance impact

The program is memory bound, so we keep task2's implementation as baseline. Here we can see `Optimized Reduce – MSSP ` is influenced most. Although `Optimized Reduce – Int32 Add` and `Scan Inclusive AddI32` also call `scanIncWarp`, they are still memory bound.

Testing parallel basic blocks
N = 100003565
block size = 256
ELEMS_PER_THREAD = 24

| Test Case                                | Warp-level reduce+GMem-SMem Coalesced (GB/s) | GMem-SMem Coalesced (GB/s) | Relative Gain (%) |
|------------------------------------------|-----------------------------------------------|-----------------------------|-------------------|
| Optimized Reduce – MSSP                  | 542.76                                        | 373.85                      | +45.18%            |


Also, we have a version with task2+task3 with original baseline. Here we can see `Optimized Reduce – MSSP ` and `Scan Inclusive AddI32` are influenced most. 

| Test Case                                | Warp-level reduce+GMem-SMem Coalesced (GB/s) | baseline (GB/s) | Relative Gain (%) |
|------------------------------------------|-----------------------------------------------|-----------------------------|-------------------|
| Optimized Reduce – MSSP                  | 542.76                                        | 268.47                      | +102.16%            |
| Scan Inclusive AddI32                    | 977.23                                        | 329.68                      | +196.41%             |


## Task4 Find the bug in `scanIncBlock` 
Error message: 
`INVALID, EXITING!!!`
Config:
```Shell
Testing parallel basic blocks
N = 100000
block size = 1024
ELEMS_PER_THREAD = 6
```

Analysis:
When block size is 1024, with a warp size of 32, a 1024-thread block has exactly 32 warps. Let's name the warp-ids 0...31.

The race condition happends in the following code:

```c++
if (lane == (WARP-1)) { ptr[warpid] = OP::remVolatile(ptr[idx]); }
```

Consider the (idx=31, lane=31, warp_id=0) as T1 and the (idx=1023, lane=31, warp_id=31) as T2. Let T1 reads and T2 writes, then T1 reads ptr[31] and T2 writes ptr[31] happends at the same time. 

This case only happends when block size is 1024, because only this size would have overlap on warpid and idx.


## Task5 Flat Sparse-Matrix Vector Multiplication in CUDA 

1) Validation:
![Validation result](https://raw.githubusercontent.com/santiweide/pmph-e2025-pub/refs/heads/main/weeklies/assignment-2/%E6%88%AA%E5%B1%8F2025-09-11%2022.41.14.png)

2) Implementations

spmv_mul_main.cu:
```c++
uint32_t num_blocks     = (tot_size  + block_size - 1) / block_size;
uint32_t num_blocks_shp = (mat_rows + block_size - 1) / block_size;
```

spmv_mul_kernels.cuh:

1) replicate0 zeroes the flags buffer.
```c++
__global__ void
replicate0(int tot_size, char* flags_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < tot_size; i += stride) {
        flags_d[i] = 0;
    }
}
```

2) `mkFlags` marks each beginning position of each scan result in each row.

```c++
__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = gid; r < mat_rows; r += stride) {
        int head_idx = (r == 0) ? 0 : mat_shp_sc_d[r - 1];
        flags_d[head_idx] = 1;
    }
}
```

3)  `mult_pairs` computes `mat_vals[i] * vct[mat_inds[i]]`.
```c++
__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < tot_size; i += stride) {
        int col = mat_inds[i];
        tmp_pairs[i] = mat_vals[i] * vct[col];
    }
}
```
4) `select_last_in_sgm` reads the last index per row: `last_idx = mat_shp_sc_d[r] - 1`.
```c++
__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = gid; r < mat_rows; r += stride) {
        int last_idx = mat_shp_sc_d[r] - 1;
        res_vct_d[r] = (last_idx >= 0) ? tmp_scan[last_idx] : 0.0f;
    }
}
```

3) Performance

| Metric                  |                CPU |                GPU |
| ----------------------- | -----------------: | -----------------: |
| Runtime                 |      **19,573 µs** |         **469 µs** |
| Rows per second         | **0.564 M rows/s** | **23.52 M rows/s** |

GPU is 41.73 times faster then CPU.