using ComplexNumbers;
using System;

namespace FractalView.Rendering
{
    public class ViewData
    {
        public int Width { get; set; }
        public int Height { get; set; }

        public bool IsMaxScale { get; private set; } = false;
        public bool IsMinScale { get; private set; } = false;

        private (int re, int im) MinScale = (5, 4);

        private (double X, double Y) ratio;
        public (double X, double Y) Ratio 
        {
            get => ratio;
            set
            {
                if (value.X < 1E-15 || value.Y < 1E-15)
                {
                    IsMaxScale = true;
                    ScaleLimitReach?.Invoke(1);
                }
                else
                {
                    if(IsMaxScale)
                    {
                        IsMaxScale = false;
                        ScaleLimitReach?.Invoke(0);
                    }

                    ratio.X = value.X;
                    ratio.Y = value.Y;
                }
            }
        }

        private readonly ComplexNumber JuliaScale = (3, 2.5);
        public ComplexNumber Position { get; set; }
        private ComplexNumber tempScale;
        private ComplexNumber scale = (3.4,2.8);
        public ComplexNumber Scale { get => scale; }

        public static event Action<int> ScaleLimitReach;

        public ViewData(int width, int height)
        {
            /*** DA RIVEDERE ***/
            double x = -0.5 + (800-width)*0.00045; //empirica 
            double y = 0;

            Position = new ComplexNumber(x, y);
            SetRatio(width, height);
        }

        public void SetRatio(int width, int height)
        {
            Width = width;
            Height = height;
            scale.Real = scale.Imaginary * (width / (double)height);

            ratio = (scale.Real / Width, scale.Imaginary / Height);
        }

        public void ChangePosition(int cx, int cy) => Position = ToComplex(cx, cy);

        public void JuliaScaleSwitch(bool IsEnabled)
        {
            if (IsEnabled)
            {
                tempScale = scale;
                scale = JuliaScale;
            }
            else
                scale = tempScale;

            ratio = (scale.Real / Width, scale.Imaginary / Height);
        }

        private void ChangeScale(double ReRad, double ImRad)
        {
            if (ReRad <= MinScale.re && ImRad <= MinScale.im)
            {
                scale.Real = ReRad;
                scale.Imaginary = ImRad;

                tempScale = scale;

                ratio = (scale.Real / Width, scale.Imaginary / Height);

                if (IsMinScale)
                {
                    IsMinScale = false;
                    ScaleLimitReach?.Invoke(0);
                }
            }
            else
            {
                IsMinScale = true;
                ScaleLimitReach?.Invoke(-1);
            }
        }

        public void StepIncScale()
        {
            double r = Scale.Real + Scale.Real / 4;
            double i = Scale.Imaginary + Scale.Imaginary / 4;
            ChangeScale(r, i);
        }

        public void StepDecScale()
        {
            double r = Scale.Real - Scale.Real / 4;
            double i = Scale.Imaginary - Scale.Imaginary / 4;
            ChangeScale(r, i);
        }

        public void ResetScale()
        {
            ChangeScale(MinScale.re, MinScale.im);
            Position = new ComplexNumber(-0.5, 0);
        }

        private ComplexNumber ToComplex(int viewx, int viewy)
        {
            double re = ((Ratio.X * (viewx)) + Position.Real);
            double im = ((Ratio.Y * (viewy)) + Position.Imaginary);

            return new ComplexNumber(re, im);
        }

        private (int ViewX, int ViewY) ToView(ComplexNumber z)
        {
            int viewx = (int)Math.Round(((z.Real - Position.Real) / Ratio.X), 1);
            int viewy = (int)Math.Round(((z.Imaginary - Position.Imaginary) / Ratio.Y), 1);

            return (viewx, viewy);
        }
    }
}
