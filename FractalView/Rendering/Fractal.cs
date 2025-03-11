using FractalView.Utility;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace FractalView.Rendering
{
    public abstract class Fractal
    {
        protected ViewData view;
        public ViewData View { get => view; set => view = value; }

        public FastBitmap FImage { get; set; }

        public int MaxIterations { get; set; }

        private const int MinIterations = 60;
        protected int iterations = MinIterations;
        public int Iterations { get => iterations; set => iterations=value; }

        private bool isBusy = false;
        public bool IsBusy { get => isBusy; }

        public bool JuliaMode = false;
  
        public static CancellationTokenSource CancTokenSource { get; set; } = new CancellationTokenSource();
        protected ParallelOptions ParallelToken = new ParallelOptions { CancellationToken = CancTokenSource.Token };

        protected abstract RenderStatus Calculate();

        public RenderStatus StartRender()
        {
            RenderStatus Status = RenderStatus.Busy;

            if (isBusy == false)
            {
                isBusy = true;
                Status = Calculate();
                isBusy = false;
            }

            return Status;
        }

        public void SetView(Image img)
        {
            FImage = new FastBitmap(img);

            if (view is null)
                view = new ViewData((int)img.Width, (int)img.Height);
            else
                view.SetRatio((int)img.Width, (int)img.Height);

            UpdateIterations();
        }


        public void UpdateIterations(bool dir = false)
        {
            /*** DA RIVEDERE ***/
            int temp_iter = 0;
            if (FractalTypes.SelectedItem.EnumFType == FractalTypes.Enumerator.Mandelbrot)
            {
                double min = Math.Min(Math.Abs(view.Scale.Real), Math.Abs(view.Scale.Imaginary));
                temp_iter = (int)(Math.Log(1 / Math.Abs(view.Scale.Real)) * 4) + 20;
            }
            else if (FractalTypes.SelectedItem.EnumFType == FractalTypes.Enumerator.BurningShip)
                temp_iter = (int)(Math.Log(1 / Math.Abs(view.Scale.Real))) + 0;
            /*******************/

            if (iterations < 0)
                temp_iter = 10;

            temp_iter = (int)(0.2 * temp_iter);
            if (dir)
                iterations += temp_iter;
            else
                iterations -= temp_iter;
            MaxIterations = iterations * 5;
        }

        public void ResetIterations()
        {
            iterations = 60;
            MaxIterations = iterations * 5;
        }

        public enum RenderStatus
        {
            Busy,
            Success,
            InteropError,
            NoFractalSelected,
        };
    }

}
