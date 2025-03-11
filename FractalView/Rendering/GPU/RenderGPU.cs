using FractalView.GPUInterop;
using FractalView.Utility;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace FractalView.Rendering
{
    sealed class RenderGPU : Fractal
    {
        private uchar3[] cudaColorMap;
        private uchar3[] CUDAcolorMap
        {
            get
            {
                ColorMappings.ColorMap colorMap = ColorMappings.SelectedItem;
                cudaColorMap = new uchar3[colorMap.Buffer.Length];

                int i = 0;             
                foreach(int color in colorMap.Buffer)
                {
                    cudaColorMap[i] = IntTouchar3(color);
                    i++;
                }

                return cudaColorMap;
            }
        }

        protected override RenderStatus Calculate()
        {
            int ViewDataWidth = View.Width;
            int ViewDataHeight = View.Height;

            uchar3[] buffer = new uchar3[ViewDataWidth * ViewDataHeight];

            FractalTypes.Enumerator local_FType = FractalTypes.SelectedItem.EnumFType;

            try
            {
                //int bufferSize = buffer.Length * Marshal.SizeOf(typeof(int));

                CUDAInterop.SetupRender(View);
                CUDAInterop.StartRender((int)local_FType, JuliaMode, Iterations, CUDAcolorMap, buffer);
            }
            catch
            {
                return RenderStatus.InteropError;
            }

            long ptr = FImage.Lock();

            Parallel.For(0, ViewDataHeight, (y) =>
            {
                for (int x = 0; x < ViewDataWidth; x++)
                    FImage.SetPixel(ptr, x, y, uchar3ToInt(buffer[y * ViewDataWidth + x]));             
            });

            FImage.Unlock();

            return RenderStatus.Success;
        }

        private int uchar3ToInt(uchar3 u) => (int)(0xFF000000 | ((u.x << 16) | (u.y << 8) | u.z));
        private uchar3 IntTouchar3(int i)
        {
            uchar3 c; 

            c.x = (byte)(i >> 16 & 0x000000FF);
            c.y = (byte)(i >> 8 & 0x000000FF);
            c.z = (byte)(i & 0x000000FF);

            return c;
        }
    }
}
