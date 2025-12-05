// nbody_cuda_fast.cu
// Compatible with older NVCC (no <random>)

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define G_F 6.67430e-8f
#define EPS_F 1e-6f

struct BodyF {
    float x, y, z;
    float vx, vy, vz;
    float m;
};

inline void checkCuda(cudaError_t e, const char* msg=nullptr){
    if (e != cudaSuccess){
        fprintf(stderr, "CUDA error %s: %s\n", msg?msg:"", cudaGetErrorString(e));
        exit(1);
    }
}

// Simple linear congruential generator (CPU only)
__host__ unsigned lcg(unsigned& state){
    state = (1664525u * state + 1013904223u);
    return state;
}

__host__ float rand_float(unsigned& state, float lo, float hi){
    return lo + (float)(lcg(state) & 0xFFFFFF) / 16777215.0f * (hi - lo);
}

__host__ void init_host(BodyF* h, int N, unsigned seed=12345){
    unsigned rng = seed;
    for (int i=0;i<N;i++){
        h[i].x = rand_float(rng, 0.0f, 1000.0f);
        h[i].y = rand_float(rng, 0.0f, 1000.0f);
        h[i].z = rand_float(rng, 0.0f, 1000.0f);
        h[i].vx = rand_float(rng, -0.5f, 0.5f);
        h[i].vy = rand_float(rng, -0.5f, 0.5f);
        h[i].vz = rand_float(rng, -0.5f, 0.5f);
        h[i].m  = rand_float(rng, 1e3f, 1e6f);
    }
}

template<int UNROLL>
__global__ void nbody_tiled_unroll(BodyF* bodies, float* outFx, float* outFy, float* outFz, int N) {
    extern __shared__ BodyF tile[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid >= N) return;

    BodyF bi = bodies[gid];
    float ax = 0, ay = 0, az = 0;

    for (int t = 0; t < N; t += blockDim.x) {
        int idx = t + tid;
        if (idx < N) tile[tid] = bodies[idx];
        else tile[tid].m = 0.0f;
        __syncthreads();

        int limit = min(blockDim.x, N - t);
        int j = 0;

        for (; j + (UNROLL-1) < limit; j += UNROLL) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++){
                BodyF bj = tile[j+u];
                float dx = bj.x - bi.x;
                float dy = bj.y - bi.y;
                float dz = bj.z - bi.z;
                float r2 = dx*dx + dy*dy + dz*dz + EPS_F;
                float inv = rsqrtf(r2);
                float inv3 = inv * inv * inv;
                float f = G_F * bj.m * inv3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
        }

        for (; j < limit; j++){
            BodyF bj = tile[j];
            float dx = bj.x - bi.x;
            float dy = bj.y - bi.y;
            float dz = bj.z - bi.z;
            float r2 = dx*dx + dy*dy + dz*dz + EPS_F;
            float inv = rsqrtf(r2);
            float inv3 = inv * inv * inv;
            float f = G_F * bj.m * inv3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }

        __syncthreads();
    }

    outFx[gid] = ax * bi.m;
    outFy[gid] = ay * bi.m;
    outFz[gid] = az * bi.m;
}

__global__ void integrate_kernel(BodyF* b, float* Fx, float* Fy, float* Fz, int N, float dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    float ax = Fx[id] / b[id].m;
    float ay = Fy[id] / b[id].m;
    float az = Fz[id] / b[id].m;

    b[id].vx += ax * dt;
    b[id].vy += ay * dt;
    b[id].vz += az * dt;

    b[id].x += b[id].vx * dt;
    b[id].y += b[id].vy * dt;
    b[id].z += b[id].vz * dt;
}

int main(int argc, char** argv){
    if (argc < 4){
        printf("Usage: %s N STEPS DT\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int STEPS = atoi(argv[2]);
    float dt = atof(argv[3]);
    printf("N=%d STEPS=%d DT=%g\n", N, STEPS, dt);

    BodyF* h = (BodyF*) malloc(sizeof(BodyF)*N);
    init_host(h, N);

    BodyF *d_b;
    float *Fx, *Fy, *Fz;

    cudaMalloc(&d_b,  sizeof(BodyF)*N);
    cudaMalloc(&Fx, sizeof(float)*N);
    cudaMalloc(&Fy, sizeof(float)*N);
    cudaMalloc(&Fz, sizeof(float)*N);

    cudaMemcpy(d_b, h, sizeof(BodyF)*N, cudaMemcpyHostToDevice);

    int block = 1024;
    int grid = (N + block - 1) / block;
    size_t shared = sizeof(BodyF) * block;

    cudaEvent_t s,e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);

    for (int i=0;i<STEPS;i++){
        nbody_tiled_unroll<4><<<grid, block, shared>>>(d_b, Fx, Fy, Fz, N);
        integrate_kernel<<<grid, block>>>(d_b, Fx, Fy, Fz, N, dt);
    }

    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    printf("Elapsed: %.3f ms (%.6f s)\n", ms, ms/1000);

    cudaFree(d_b); cudaFree(Fx); cudaFree(Fy); cudaFree(Fz);
    free(h);
    return 0;
}
