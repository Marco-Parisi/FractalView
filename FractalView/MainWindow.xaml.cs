using FractalView.Rendering;
using FractalView.Utility;
using Microsoft.Win32;
using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;

namespace FractalView
{
    public partial class MainWindow : Window
    {
        public Render MainRender;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            foreach(Button b in WinBarGrid.Children)
            {
                b.MouseEnter += MouseEnterWinBar;
                b.MouseLeave += MouseLeaveWinBar;
            }

            MainRender = new Render(DirtyImage);
            MainRender.StartRender();

            ColorMappings.ItemChanged += (cmap) => MainRender.StartRender();
            ViewData.ScaleLimitReach += MaxScale;

            DataContext = MainRender;

            SettingButtonClick(new Button() { Content = new TextBlock() }, null);
        }

        private void MaxScale(int ScaleLimit)
        {
            (RenderMessage.Parent as FrameworkElement).Visibility = ScaleLimit == 1 || ScaleLimit == -1 ? Visibility.Visible : Visibility.Collapsed;
            
            if (ScaleLimit == 1)
                RenderMessage.Text = "Max Scale";
            else if (ScaleLimit == -1)
                RenderMessage.Text = "Min Scale";
        }

        private void OnWindowStateChanged(object sender, EventArgs e)
        {
            if (WindowState == WindowState.Maximized)
            {
                ((MaximizeButton.Content as Grid).Children[0] as Grid).Visibility = Visibility.Collapsed;
                ((MaximizeButton.Content as Grid).Children[1] as Grid).Opacity = 1;

                MainGrid.Margin = new Thickness(10);
            }
            else
            {
                ((MaximizeButton.Content as Grid).Children[0] as Grid).Visibility = Visibility.Visible;
                ((MaximizeButton.Content as Grid).Children[1] as Grid).Opacity = 0;

                MainGrid.Margin = new Thickness(0);

            }
        }

        private void MaximizeClick(object sender, RoutedEventArgs e)
        {
            if (WindowState != WindowState.Maximized)
                WindowState = WindowState.Maximized;
            else
                WindowState = WindowState.Normal;
        }

        private void MinimizeClick(object sender, RoutedEventArgs e)
        {
            WindowState = WindowState.Minimized;
        }

        private void CloseClick(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void MouseEnterWinBar(object sender, MouseEventArgs e)
        {
            if (TranslateSetting.X > 0)
            {
                DoubleAnimation dao = new DoubleAnimation();
                dao.SpeedRatio = 3;
                dao.To = 1;
                WinBarGrid.BeginAnimation(OpacityProperty, dao);
            }
        }

        private void MouseLeaveWinBar(object sender, MouseEventArgs e)
        {
            if (TranslateSetting.X > 0)
            {
                DoubleAnimation dao = new DoubleAnimation();
                dao.SpeedRatio = 3;
                dao.To = 0;
                WinBarGrid.BeginAnimation(OpacityProperty, dao);
            }
        }

        private void SettingButtonClick(object sender, RoutedEventArgs e)
        {
            if (sender != null)
            {
                TextBlock tb = ((sender as Button).Content as TextBlock);

                DoubleAnimation dat = new DoubleAnimation();
                dat.SpeedRatio = 6;

                DoubleAnimation dao = new DoubleAnimation();
                dao.SpeedRatio = 3;

                if (WinBarGrid.Opacity > 0)
                {
                    dat.To = SettingPanel.Width+2;
                    dao.To = 0;
                    tb.Text = "\u2b9c";
                }
                else
                {
                    dat.To = 0;
                    dao.To = 1;
                    tb.Text = "\u2b9e";
                }

                WinBarGrid.BeginAnimation(OpacityProperty, dao);
                TranslateSetting.BeginAnimation(TranslateTransform.XProperty, dat);
            }
        }

        private void SaveImgClick(object sender, RoutedEventArgs e)
        {
            SaveFileDialog saveImageDialog = new SaveFileDialog
            {
                DefaultExt = ".bmp",
                Filter = "JPG Image (.jpg)|*.jpg|Bitmap Image (.bmp)|*.bmp"
            };

            if (saveImageDialog.ShowDialog() == true)
            {
                BitmapEncoder encoder;

                if (saveImageDialog.FilterIndex == 0)
                    encoder = new JpegBitmapEncoder();
                else
                    encoder = new BmpBitmapEncoder();

                RenderTargetBitmap bitmap = new RenderTargetBitmap((int)DirtyImage.ActualWidth, (int)DirtyImage.ActualHeight, 96, 96, PixelFormats.Pbgra32);
                bitmap.Render(DirtyImage);

                BitmapFrame frame = BitmapFrame.Create(bitmap);
                encoder.Frames.Add(frame);

                using (FileStream stream = File.Create(saveImageDialog.FileName))
                {
                    encoder.Save(stream);
                }
            }
        }

        private void WindowSizeChanged(object sender, SizeChangedEventArgs e)
        {
            MainRender?.UpdateRequest(DirtyImage);
        }

        private void IterSliderMove(object sender, System.Windows.Controls.Primitives.DragCompletedEventArgs e)
        {
            MainRender?.StartRender();
        }

        private void IterSliderKeyUp(object sender, KeyEventArgs e)
        {
            MainRender?.StartRender();
        }

        private void ResetCoordClick(object sender, RoutedEventArgs e)
        {
            MainRender?.ResetScale();
        }

        private void JuliaClick(object sender, RoutedEventArgs e)
        {
            MainRender?.ToggleJulia();
        }

        private void FractalTypeChanged(object sender, RoutedEventArgs e)
        {
            MainRender?.StartRender();
        }
    }
}
