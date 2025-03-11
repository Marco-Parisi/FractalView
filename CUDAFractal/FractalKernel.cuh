#include <cstdio>

// Increase the grid size by 1 if the image width or height does 
// not divide evenly by the thread block dimensions
#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define ABS(x) ((x<0) ? -x : x)

/// <summary>
/// Divide a and b params in order to create a Grid for CUDA 
/// </summary>
inline int iDivUp(int a, int b) 
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
