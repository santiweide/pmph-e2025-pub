#ifndef SPMV_MUL_KERNELS
#define SPMV_MUL_KERNELS

__global__ void
replicate0(int tot_size, char* flags_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < tot_size; i += stride) {
        flags_d[i] = 0;
    }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = gid; r < mat_rows; r += stride) {
        int head_idx = (r == 0) ? 0 : mat_shp_sc_d[r - 1];
        flags_d[head_idx] = 1;
    }
}

__global__ void
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < tot_size; i += stride) {
        int col = mat_inds[i];
        tmp_pairs[i] = mat_vals[i] * vct[col];
    }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int r = gid; r < mat_rows; r += stride) {
        int last_idx = mat_shp_sc_d[r] - 1;
        res_vct_d[r] = (last_idx >= 0) ? tmp_scan[last_idx] : 0.0f;
    }
}

#endif // SPMV_MUL_KERNELS
