
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FractalKernel.cuh"

#include <stdio.h>

#pragma diag_suppress 29  // Disable E0029 error for Cuda kernel function syntax <<< x, y >>>
#pragma diag_suppress 20  // Disable E0020 error for Cuda math syntax

static int h_renderWidth = 0;
static int h_renderHeight = 0;
static int h_maxIteration;
static double h_minPosReal;
static double h_minPosImag;
static double h_radiusReal;
static double h_radiusImag;   
static int* h_iterationBuffer;

__global__ void FractalKernel(int d_renderWidth, int d_renderHeight, double d_minPosReal, double d_minPosImag,
                            double d_radiusReal, double d_radiusImag, int d_MaxIteration, uchar3* image_buffer,
                            const uchar3* d_colorArray, const int h_colorArrLength, const int d_fractalType, 
                            const bool juliaMode)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= d_renderWidth || row >= d_renderHeight)
        return;

    int idx = row * d_renderWidth + col;
    
    double ZnIm = 0.0;
    double ZnRe = 0.0;
    double ZnSQRe=0.0;
    double ZnSQIm=0.0;
    double tempZnIm = 0;
    double tempZnRe = 0;

    double s = d_radiusReal / (float)d_renderWidth;
    double CRe;
    double CIm;

    int iter = 0;
    uchar3 color;                    
    
    if (juliaMode)  // Julia Active
    {
        ZnRe = (col - (double)d_renderWidth * 0.5f) * s;
        ZnIm = (row - (double)d_renderHeight * 0.5f) * s;

        if (d_fractalType == 1) // Burning Ship
        {
            ZnRe = abs(ZnRe);
            ZnIm = abs(ZnIm);
        }

        ZnSQRe = ZnRe * ZnRe;
        ZnSQIm = ZnIm * ZnIm;

        CRe = d_minPosReal;
        CIm = d_minPosImag;
    }
    else
    {
        CRe = (col - (double)d_renderWidth * 0.5f) * s + d_minPosReal;
        CIm = (row - (double)d_renderHeight * 0.5f) * s + d_minPosImag;
    }

    for(; iter < d_MaxIteration && sqrt((ZnSQRe + ZnSQIm)) < 10; iter++)
    {
        if (d_fractalType == 2) // Tippetts Mandelbrot
        {
            tempZnRe = ZnSQRe - ZnSQIm + CRe;
            tempZnIm = (2 * (ZnRe * ZnRe) * ZnIm) - (2 * (ZnSQIm * ZnIm)) + (2 * CRe * ZnIm) + CIm;
            ZnIm = tempZnIm;
            ZnRe = tempZnRe;
        }
        else
        {
            ZnIm = ZnRe * ZnIm;
            ZnRe = ZnSQRe - ZnSQIm + CRe;
        }

        if (d_fractalType == 0) // Mandelbrot
        {
            ZnIm = 2 * ZnIm + CIm;
        }
        else if (d_fractalType == 1) // Burning Ship
        {
            ZnIm = 2 * abs(ZnIm) + CIm;
            ZnIm = abs(ZnIm);
            ZnRe = abs(ZnRe);
        }

        ZnSQRe = ZnRe * ZnRe;
        ZnSQIm = ZnIm * ZnIm;
    } 

    if (iter < d_MaxIteration)
    {
        double mu = iter - log(log(sqrt((ZnSQRe + ZnSQIm)))) / log(2.0);
        iter = (int)(mu / d_MaxIteration * h_colorArrLength);
        if (iter >= h_colorArrLength) iter = 0;
        if (iter < 0) iter = 0;
    }
    else
        iter = h_colorArrLength - 1;

    color.x = d_colorArray[iter].x;
    color.y = d_colorArray[iter].y;
    color.z = d_colorArray[iter].z;
    image_buffer[idx] = color;
}


__global__ void PseudoFXAA(const int d_renderWidth, const int d_renderHeight, uchar3* image_buffer)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= d_renderWidth || row >= d_renderHeight)
        return;

    int idx = row * d_renderWidth + col;

    float medF = 0.3f;
    float oriF = 1 - medF;

    //if (idx < (d_renderWidth * d_renderHeight) - d_renderWidth * 2 && idx > d_renderWidth * 2)
    //{
    //    unsigned int  sumx = image_buffer[idx - 1].x + image_buffer[idx + 1].x + image_buffer[idx - d_renderWidth].x + image_buffer[idx + d_renderWidth].x; 
    //    //sumx += image_buffer[idx - d_renderWidth - 1].x + image_buffer[idx - d_renderWidth + 1].x + image_buffer[idx + d_renderWidth - 1].x + image_buffer[idx + d_renderWidth + 1].x;

    //    unsigned int  sumy = image_buffer[idx - 1].y + image_buffer[idx + 1].y + image_buffer[idx - d_renderWidth].y + image_buffer[idx + d_renderWidth].y;
    //    //sumy += image_buffer[idx - d_renderWidth - 1].y + image_buffer[idx - d_renderWidth + 1].y + image_buffer[idx + d_renderWidth - 1].y + image_buffer[idx + d_renderWidth + 1].y;
    // 
    //    unsigned int  sumz = image_buffer[idx - 1].z + image_buffer[idx + 1].z + image_buffer[idx - d_renderWidth].z + image_buffer[idx + d_renderWidth].z;
    //    //sumz += image_buffer[idx - d_renderWidth - 1].z + image_buffer[idx - d_renderWidth + 1].z + image_buffer[idx + d_renderWidth - 1].z + image_buffer[idx + d_renderWidth + 1].z;

    //    uchar3 medColor;
    //    medColor.x = (unsigned char)(oriF * image_buffer[idx].x) + (medF * (sumx / 4.0f));
    //    medColor.y = (unsigned char)(oriF * image_buffer[idx].y) + (medF * (sumy / 4.0f));
    //    medColor.z = (unsigned char)(oriF * image_buffer[idx].z) + (medF * (sumz / 4.0f));

    //    image_buffer[idx] = medColor;
    //}


    if (idx < (d_renderWidth * d_renderHeight) - d_renderWidth * 2 && idx > d_renderWidth * 2)
    {
        unsigned int  sumOri = image_buffer[idx].x + image_buffer[idx].y + image_buffer[idx].z;

        unsigned int  sum1 = image_buffer[idx - 1].x + image_buffer[idx - 1].y + image_buffer[idx - 1].z;
        unsigned int  sum2 = image_buffer[idx + 1].x + image_buffer[idx + 1].y + image_buffer[idx + 1].z;
        unsigned int  sum3 = image_buffer[idx - d_renderWidth].x + image_buffer[idx - d_renderWidth].y + image_buffer[idx - d_renderWidth].z;
        unsigned int  sum4 = image_buffer[idx + d_renderWidth].x + image_buffer[idx + d_renderWidth].y + image_buffer[idx + d_renderWidth].z;
        
        uchar3 medColor;
        medColor.x = 0;
        medColor.y = 255;
        medColor.z = 0;
        if (sumOri < 730)
        {
            image_buffer[idx] = medColor;
        }

    }

}


extern "C" __declspec(dllexport) void __cdecl SetupRender(int width, int height, double minRe, double minIm, double radRe, double radIm)
{
    h_renderWidth = width;
    h_renderHeight = height;
    h_minPosReal = minRe;
    h_minPosImag = minIm;
    h_radiusReal = radRe;
    h_radiusImag = radIm;
}

extern "C" __declspec(dllexport) cudaError __cdecl StartRender(int fractalType, bool juliaMode, int Iteration, uchar3 * d_colorMap, int d_colorMapLength, uchar3 *result)
{
    if (h_renderHeight == 0 || h_renderWidth == 0)
        return (cudaError_t)-1; // return -1 as "renderNotDefined" in C# enum cudaError

    cudaError_t cudaStatus;

    int size = h_renderWidth * h_renderHeight;

    h_maxIteration = Iteration;

    uchar3* h_image_buffer;
    uchar3* h_colorMap;

    cudaStatus = cudaMalloc(&h_image_buffer, h_renderWidth * h_renderHeight * sizeof(uchar3));
    if (cudaStatus != cudaSuccess) goto Error;
    
    cudaStatus = cudaMalloc(&h_iterationBuffer, h_renderWidth * h_renderHeight * sizeof(int));
    if (cudaStatus != cudaSuccess) goto Error;
    
    cudaStatus = cudaMalloc(&h_colorMap, d_colorMapLength * sizeof(uchar3));
    if (cudaStatus != cudaSuccess) goto Error; 
    
    cudaStatus = cudaMemcpy(h_colorMap, d_colorMap, d_colorMapLength * sizeof(uchar3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) goto Error;

    dim3 block_size(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid_size(iDivUp(h_renderWidth, BLOCKDIM_X), iDivUp(h_renderHeight, BLOCKDIM_Y));

    FractalKernel<<<grid_size, block_size>>>(h_renderWidth, h_renderHeight, h_minPosReal, h_minPosImag, 
                                             h_radiusReal, h_radiusImag, h_maxIteration, h_image_buffer, 
                                             h_colorMap, d_colorMapLength, fractalType, juliaMode);

    //PseudoFXAA<<<grid_size, block_size>>>(h_renderWidth, h_renderHeight, h_image_buffer);

    cudaStatus = cudaMemcpy(result, h_image_buffer, h_renderHeight * h_renderWidth * sizeof(uchar3), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) goto Error;


Error:
    cudaFree(h_image_buffer);
    cudaFree(h_colorMap);
    cudaFree(h_iterationBuffer);

    fprintf(stderr, "CUDA Status : %s\n", cudaGetErrorString(cudaStatus));

    return cudaStatus;

}
