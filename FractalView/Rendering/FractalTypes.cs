using System;
using System.Collections.ObjectModel;

namespace FractalView.Utility
{
    public class FractalTypes
    {
        public enum Enumerator
        {
            Mandelbrot,
            BurningShip,
            TippettsMandel
        }

        public class FractalType
        {
            public Enumerator EnumFType { get; set; }
            public string FractalName { get; set; }

            public override string ToString()
            {
                return FractalName;
            }
        }

        public static FractalType SelectedItem
        {
            get => _selectedItem;
            set
            {
                _selectedItem = value;
                ItemChanged?.Invoke(value);
            }
        }

        private static FractalType _selectedItem;
        public static event Action<FractalType> ItemChanged;
        public static ObservableCollection<FractalType> Items { get; private set; }

        static FractalTypes()
        {
            Items = new ObservableCollection<FractalType>
            {
                new FractalType() { EnumFType = Enumerator.Mandelbrot, FractalName = "Mandelbrot" },
                new FractalType() { EnumFType = Enumerator.BurningShip, FractalName = "Burning Ship" },
                new FractalType() { EnumFType = Enumerator.TippettsMandel, FractalName = "Tippetts Mandelbrot" }
            };

            SelectedItem = Items[0];
        }
    }
}
