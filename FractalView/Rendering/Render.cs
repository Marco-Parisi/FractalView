using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Management;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;

namespace FractalView.Rendering
{
    public class Render : INotifyPropertyChanged
    {
        private static Fractal FractalHost;
        private DispatcherTimer IterChangedTimer;
        private Image updateImage;

        public event PropertyChangedEventHandler PropertyChanged;
        private readonly PropertyChangedEventArgs PCEarg = new PropertyChangedEventArgs("");

        public ViewData VD { get => FractalHost?.View; }

        public bool? IsBusy { get => FractalHost?.IsBusy; }

        public int MaxIterations { get => FractalHost.MaxIterations; }
        public int Iterations 
        {
            get => FractalHost.Iterations;
            set
            {
                if (value < 20)
                    FractalHost.Iterations = 20;
                else if (value > MaxIterations)
                    FractalHost.Iterations = MaxIterations;
                else
                    FractalHost.Iterations = value;

                PropertyChanged?.Invoke(this, PCEarg);
            }
        }

        public void ToggleJulia()
        {
            FractalHost.JuliaMode = !FractalHost.JuliaMode;
            FractalHost.View.JuliaScaleSwitch(FractalHost.JuliaMode);

            StartRender();
        }

        public void UpdateRequest(Image img)
        {
            updateImage = img;

            if (IterChangedTimer is null)
            {
                IterChangedTimer = new DispatcherTimer();
                IterChangedTimer.Interval = new TimeSpan(4000000);
                IterChangedTimer.Tick += IterChanged;
                IterChangedTimer.Start();
            }
            else if(!IterChangedTimer.IsEnabled)
                IterChangedTimer.Start();
        }

        private void IterChanged(object sender, EventArgs e)
        {
            FastBitmap bi = new FastBitmap(updateImage);
            if (FractalHost.FImage.Width.Equals(bi.Width) && FractalHost.FImage.Height.Equals(bi.Height))
            {
                UpdateRender(updateImage);

                IterChangedTimer.Stop();
                IterChangedTimer.IsEnabled = false;
            }
            else
                FractalHost.FImage = bi;           
        }

        private void UpdateRender(Image img)
        {
            FractalHost.SetView(img);
            StartRender();
        }

        public Render(Image dirtyImg)
        {
            dirtyImg.MouseUp += MoveEventUp;
            dirtyImg.MouseDown += MoveEventDown;
            dirtyImg.MouseMove += MoveEventMoving;
            dirtyImg.MouseWheel += MouseEventWheelZoom;

            ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");

            List<string> graphicCards = new List<string>();
            foreach (ManagementObject mo in searcher.Get())
                graphicCards.Add(mo.GetPropertyValue("Description").ToString().ToUpper());

            foreach(string gc in graphicCards) 
            {
                if (gc.Contains("NVIDIA"))
                {
                    FractalHost = new RenderGPU();
                    break;
                }
                else
                {
                    FractalHost = new RenderCPU();

                    if (gc.Contains("AMD"))
                        MessageBox.Show("AMD Graphics Card not supported (yet)\nRender with CPU");
                }
            }

            FractalHost.SetView(dirtyImg);

        }


        private Point P1;
        private bool isMoving = false;
        private void MoveEventUp(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Released)
            {
                isMoving = false;
                StartRender();
            }
        }

        private void MoveEventDown(object sender, MouseButtonEventArgs e)
        {
            if(e.LeftButton== MouseButtonState.Pressed)
                P1 = e.GetPosition((IInputElement)sender);
        }

        private void MoveEventMoving(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                Point P2 = e.GetPosition((IInputElement)sender);

                /*** DA RIVEDERE ***/

                if (isMoving == false)
                    isMoving = true;

                int x = 2 * (int)(P1.X - P2.X);
                int y = 2 * (int)(P1.Y - P2.Y);

                FractalHost.View.ChangePosition(x, y);

                P1 = P2;
                StartRender();               
                /******************/
            }
        }

        private void MouseEventWheelZoom(object sender, MouseEventArgs e)
        {
            var P = e.GetPosition((IInputElement)sender);

            if (!FractalHost.IsBusy && !FractalHost.JuliaMode)
            {
                bool dir = (e as MouseWheelEventArgs).Delta > 0;

                if (dir)
                    FractalHost.View.StepDecScale();
                else
                    FractalHost.View.StepIncScale();

                /*** DA RIVEDERE ***/
                if (!FractalHost.View.IsMaxScale && !FractalHost.View.IsMinScale)
                {
                    int x = (int)(0.35 * (P.X - (FractalHost.View.Width / 2)));
                    int y = (int)(0.35 * (P.Y - (FractalHost.View.Height / 2)));
                    FractalHost.View.ChangePosition(x, y);
                    FractalHost.UpdateIterations(dir);
                    PropertyChanged?.Invoke(this, PCEarg);

                    StartRender();
                }
                /******************/
            }
        }

        public void ResetScale()
        {
            VD.ResetScale();
            FractalHost.ResetIterations();
            PropertyChanged?.Invoke(this, PCEarg);

            StartRender();
        }

        public void StartRender()
        {
            Task.Run(() => FractalHost.StartRender());
        }
    }
}
