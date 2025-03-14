﻿using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Image = System.Windows.Controls.Image;

namespace FractalView.Rendering
{
    /// <summary>
    /// Faster image implementation for low level set pixel performance
    /// </summary>
    public class FastBitmap : IDisposable
    {
        /// <summary>
        /// Tells if this object is still usable and not disposed yet
        /// </summary>
        public bool Disposed { get; private set; }

        // some image characteristics
        private readonly int _width;

        private readonly int _height;

        public int Width { get => _width; }

        public int Height { get => _height; }

        private readonly WriteableBitmap _writeableBitmap;

        private readonly Image imageRef;

        public FastBitmap(Image image)
        {
            imageRef = image;

            _width = (int)image.Width;
            _height = (int)image.Height;

            // bind the source of the image to this map
            _writeableBitmap = BitmapFactory.New(_width, _height);
            _writeableBitmap.Clear(Colors.Transparent);
            
            image.Source = _writeableBitmap;
            
        }

        /// <summary>
        /// Sets the pixel in the image directly and fast.
        /// </summary>
        /// <param name="ptr">The pointer to the back buffer</param>
        /// <param name="x">Horizontal coordinate</param>
        /// <param name="y">Vertical coordinate</param>
        /// <param name="color">32bit argb value of the pixel</param>
        public void SetPixel(long ptr, int x, int y, int color)
        {
            unsafe
            {
                ((int*) ptr)[y * _width + x] = color;
            }
        }

        public long GetPointer()
        {
            long ptr = 0;

            Application.Current.Dispatcher.Invoke(() => { ptr = (long) _writeableBitmap.BackBuffer; },
                DispatcherPriority.Render);

            return ptr;
        }

        public void Clear()
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                _writeableBitmap.Lock();
                _writeableBitmap.Clear(Colors.Transparent);
                _writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, _width, _height));
                _writeableBitmap.Unlock();
            }, DispatcherPriority.Render);
        }

        public void Dirty()
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                _writeableBitmap.Lock();
                _writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, _width, _height));
                _writeableBitmap.Unlock();
            }, DispatcherPriority.Render);
        }

        public void Dirty(int fromLine, int height)
        {
            // ignore invalid dirty requests
            if (fromLine + height <= _height)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    _writeableBitmap.Lock();
                    _writeableBitmap.AddDirtyRect(new Int32Rect(0, fromLine, _width, height));
                    _writeableBitmap.Unlock();
                }, DispatcherPriority.Render);
            }
        }

        public long Lock()
        {
            long ptr = 0;

            Application.Current.Dispatcher.Invoke(() =>
            {
                _writeableBitmap.Lock();
                ptr = (long) _writeableBitmap.BackBuffer;
            }, DispatcherPriority.Render);

            return ptr;
        }

        public void Unlock()
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                _writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, _width, _height));
                _writeableBitmap.Unlock();
            }, DispatcherPriority.Render);
        }

        /// <summary>
        /// Disposes all file and memory mappings.
        /// Makes the whole object unusable afterwards.
        /// </summary>
        public void Dispose()
        {
            Disposed = true;
        }
    }
}