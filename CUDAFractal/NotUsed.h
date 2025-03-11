__global__ void MandelIterAA(const int d_renderWidth, const int d_renderHeight, int* d_iterationBuffer, const int d_MaxIteration, uchar3* image_buffer, const uchar3* d_colorArray)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= d_renderWidth || row >= d_renderHeight)
        return;

    int idx = row * d_renderWidth + col;

    uchar3 color;

    if (idx < (d_renderWidth * d_renderHeight) - d_renderWidth && idx > d_renderWidth)
    {
        if (d_iterationBuffer[idx] < d_MaxIteration)
        {
            unsigned int sumIter = d_iterationBuffer[idx - 1];
            sumIter += d_iterationBuffer[idx + 1];
            sumIter += d_iterationBuffer[idx - d_renderWidth];
            sumIter += d_iterationBuffer[idx + d_renderWidth];

            int medIter = (int)(0.8f * d_iterationBuffer[idx]) + (0.2f * (sumIter / 4.0f));

            if (medIter < d_MaxIteration)
                medIter %= COLORMAP_LENGTH - 1;
            else
                medIter = COLORMAP_LENGTH - 1;

            color.x = d_colorArray[medIter].x;
            color.y = d_colorArray[medIter].y;
            color.z = d_colorArray[medIter].z;

            image_buffer[idx] = color;
        }
    }
}



#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

curandState* d_state;
cudaMalloc(&d_state, h_renderWidth* h_renderHeight * sizeof(curandState));
setup_CudaRand << <grid_size, block_size >> > (d_state, h_renderWidth, h_renderHeight, unsigned(time(NULL)));
Dithering << <grid_size, block_size >> > (d_state, h_renderWidth, h_renderHeight, h_image_buffer);
cudaFree(d_state);

__global__ void setup_CudaRand(curandState* state, const int d_renderWidth, const int d_renderHeight, unsigned long seed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= d_renderWidth || row >= d_renderHeight)
        return;

    int idx = row * d_renderWidth + col;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void Dithering(curandState* globalState, const int d_renderWidth, const int d_renderHeight, uchar3* image_buffer)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= d_renderWidth || row >= d_renderHeight)
        return;

    int idx = row * d_renderWidth + col;

    curandState localState = globalState[idx];
    float num = curand_uniform(&localState) * 255;
    globalState[idx] = localState;

    image_buffer[idx].x = (int)(image_buffer[idx].x * 0.975) + (num * 0.025);
    image_buffer[idx].y = (int)(image_buffer[idx].y * 0.975) + (num * 0.025);
    image_buffer[idx].z = (int)(image_buffer[idx].z * 0.975) + (num * 0.025);

}
