using MathNet.Numerics;
using MathNet.Numerics.Interpolation;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.Linq;

namespace FractalView.Utility
{
    /// <summary>
    /// Supplies some color mapping arrays
    /// </summary>
    public class ColorMappings
    {

        public delegate void SelectedItemChangedEventHandler(ColorMap map);
        /// <summary>
        /// Represents a color mapping by a name and the corresponding color array
        /// </summary>
        public class ColorMap
        {
            private string Name { get; }
            public int[] Buffer { get; }
            public int MaxMapIndex { get; }

            public ColorMap(string name, IEnumerable<Color> colors, int maxIndex=0)
                : this(name, colors.Select(e => e.ToArgb()).ToArray(), maxIndex)
            {
            }

            public ColorMap(string name, int[] colors, int maxIndex)
            {
                Name = name;
                Buffer = colors;
                MaxMapIndex = maxIndex;
            }

            public override string ToString()
            {
                return Name;
            }
        }

        /// <summary>
        /// Holds the currently selected color mapping
        /// </summary>
        public static ColorMap SelectedItem
        {
            get => _selectedItem;
            set
            {
                _selectedItem = value;
                ItemChanged?.Invoke(value);
            }
        }

        private static ColorMap _selectedItem;

        public static ObservableCollection<ColorMap> Items { get; }

        public static event SelectedItemChangedEventHandler ItemChanged;

        public static readonly Color[] UltraFractal = new Color[4096];
        public static readonly Color[] GrayScale = new Color[512];
        public static readonly Color[] GrayScaleInv = new Color[1024];
        public static readonly Color[] DeepBlue = new Color[128];

        static ColorMappings()
        {

            var points = new[]
            {
                0.0,
                0.16 * UltraFractal.Length,
                0.3 * UltraFractal.Length,
                0.6425 * UltraFractal.Length,
                0.8575 * UltraFractal.Length
            };

            (double R, double G, double B)[] rgbComp = new (double, double, double)[]
            {
                (0, 7, 100),
                (32, 107, 203),
                (237, 255, 255),
                (255, 170, 0),
                (0, 2, 0),
            };

            IInterpolation iR = Interpolate.CubicSpline(points, rgbComp.Select(c => c.R));
            IInterpolation iG = Interpolate.CubicSpline(points, rgbComp.Select(c => c.G));
            IInterpolation iB = Interpolate.CubicSpline(points, rgbComp.Select(c => c.B));

            for (var i = 0; i < UltraFractal.Length; i++)
            {
                UltraFractal[i] = Color.FromArgb
                (
                    (int)iR.Interpolate(i).Clamp(0, 255),
                    (int)iG.Interpolate(i).Clamp(0, 255),
                    (int)iB.Interpolate(i).Clamp(0, 255)
                );
            }
            
            /* gray scale */

            for (var i = 0; i < GrayScale.Length; i++)
            {
                int comp = (int)((i * 255.0) / (float)GrayScale.Length);
                GrayScale[GrayScale.Length - 1 - i] = Color.FromArgb(comp, comp, comp);
            }

            for (int i = 0; i < GrayScaleInv.Length; i++)
            {
                int comp = (int)((i * 255.0) / (float)GrayScaleInv.Length);
                GrayScaleInv[i]= Color.FromArgb(comp, comp, comp);
            }

            DeepBlue[DeepBlue.Length - 1] = Color.FromArgb(255, 255, 0);
            for (int i = 0; i < DeepBlue.Length; i++)
                DeepBlue[i] = Color.FromArgb((int)(0.1 * i), (int)(0.6 * i), i);

            /* list which holds all the mappings */

            Items = new ObservableCollection<ColorMap>
            {
                new ColorMap("Ultra Fractal", UltraFractal, (int) (0.8575 * UltraFractal.Length)),
                new ColorMap("Grayscale", GrayScale),
                new ColorMap("Inverted Grayscale", GrayScaleInv),
                new ColorMap("Deep Blue", DeepBlue)
            };

            SelectedItem = Items[0];
        }
    }

    public static class ExtensionMethods
    {
        public static T Clamp<T>(this T value, T min, T max) where T : IComparable<T>
        {
            if (value.CompareTo(min) < 0) return min;
            return value.CompareTo(max) > 0 ? max : value;
        }
    }
}