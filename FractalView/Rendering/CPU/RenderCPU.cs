using ComplexNumbers;
using FractalView.Utility;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace FractalView.Rendering
{
    sealed class RenderCPU : Fractal
    {
        protected override RenderStatus Calculate()
        { 
            ColorMappings.ColorMap colorMap = ColorMappings.SelectedItem;

            var midToTop = RangeMinMaxInclusive(0, GetIndex(View.Height, 0.5, true)).Reverse();
            var midToBot = RangeMinMaxInclusive(GetIndex(View.Height, 0.5, false), View.Height - 1);
            var source = midToTop.Concat(midToBot).ToArray();
            OrderablePartitioner<int> partitioner = Partitioner.Create(source, true);

            ComplexNumber cMin = (View.Position.Real, View.Position.Imaginary);
            ComplexNumber JuliaC = (View.Position.Real, View.Position.Imaginary);

            FractalTypes.Enumerator local_FType = FractalTypes.SelectedItem.EnumFType;

            double local_Ratio = View.Ratio.X;
            double local_Width = View.Width;
            double local_Height = View.Height;
            int local_MaxIter = iterations;

            long ptr = FImage.Lock();

            Parallel.ForEach(partitioner, ParallelToken, () => 0, (y, state, threadLocal) =>
            {
                ComplexNumber C = new ComplexNumber(0, 0);

                if (JuliaMode)
                    C = JuliaC;

                ComplexNumber Zn = new ComplexNumber(0, 0);
                ComplexNumber ZnSQ = new ComplexNumber(0, 0);
                ComplexNumber tempZn = new ComplexNumber(0,0);

                for (int x = 0; x < View.Width; x++)
                {
                    int i = 0;
                    double ZnAbs = 0;

                    if (JuliaMode)
                    {
                        Zn.Real = (x - local_Width * 0.5f) * local_Ratio;
                        Zn.Imaginary = (y - local_Height * 0.5f) * local_Ratio;

                        if (local_FType == FractalTypes.Enumerator.BurningShip)
                        {
                            Zn.Real = Math.Abs(Zn.Real);
                            Zn.Imaginary = Math.Abs(Zn.Imaginary);
                        }

                        ZnSQ.Real = Zn.Real * Zn.Real;
                        ZnSQ.Imaginary = Zn.Imaginary * Zn.Imaginary;                 
                    }
                    else
                    {
                        Zn = ZnSQ = new ComplexNumber(0, 0);
                        C.Real = (x - local_Width * 0.5f) * local_Ratio + cMin.Real;
                        C.Imaginary = (y - local_Height * 0.5f) * local_Ratio + cMin.Imaginary;
                    }

                    for (; i < local_MaxIter && (ZnAbs < 6); i++)
                    {
                        if (local_FType == FractalTypes.Enumerator.TippettsMandel)
                        {
                            tempZn.Real = ZnSQ.Real - ZnSQ.Imaginary + C.Real;
                            tempZn.Imaginary = (2 * (Zn.Real * Zn.Real) * Zn.Imaginary) - (2 * (ZnSQ.Imaginary * Zn.Imaginary)) + (2 * C.Real * Zn.Imaginary) + C.Imaginary;
                            Zn = tempZn;
                        }
                        else
                        {
                            Zn.Imaginary = Zn.Real * Zn.Imaginary;
                            Zn.Real = ZnSQ.Real - ZnSQ.Imaginary + C.Real;
                        }

                        if (local_FType == FractalTypes.Enumerator.Mandelbrot)
                        {
                            Zn.Imaginary = 2 * Zn.Imaginary + C.Imaginary;                      
                        }
                        else if (local_FType == FractalTypes.Enumerator.BurningShip)
                        {
                            Zn.Imaginary = 2 * Math.Abs(Zn.Imaginary) + C.Imaginary;
                            Zn.Imaginary = Math.Abs(Zn.Imaginary);
                            Zn.Real = Math.Abs(Zn.Real);
                        }

                        ZnSQ.Real = Zn.Real * Zn.Real;
                        ZnSQ.Imaginary = Zn.Imaginary * Zn.Imaginary;
                        ZnAbs = Math.Sqrt(ZnSQ.Real + ZnSQ.Imaginary);
                    }

                    int colorIdx;
                    if (i < local_MaxIter)
                    {
                        double mu = i - (Math.Log(Math.Log(ZnAbs))) / Math.Log(2);
                        colorIdx = (int)(mu / local_MaxIter * colorMap.Buffer.Length);
                        if (colorIdx >= colorMap.Buffer.Length) colorIdx = 0;
                        if (colorIdx < 0) colorIdx = 0;
                    }
                    else
                        colorIdx = colorMap.Buffer.Length - 1;

                    FImage.SetPixel(ptr, x, y, colorMap.Buffer[colorIdx]);
                }

                return threadLocal;

            }, threadLocal => { });

            FImage.Unlock();

            return RenderStatus.Success;
        }

        private static int GetIndex(int value, double rate, bool lower)
        {
            double factor = 1.0 / rate;
            double part = value / factor;

            if (value % factor == 0)
            {
                return (int)(lower ? part : part + 1);
            }
            else
            {
                return (int)(lower ? Math.Floor(part) : Math.Ceiling(part));
            }
        }

        private static IEnumerable<int> RangeMinMaxInclusive(int min, int max)
        {
            return Enumerable.Range(min, max - min + 1);
        }

    }
}
