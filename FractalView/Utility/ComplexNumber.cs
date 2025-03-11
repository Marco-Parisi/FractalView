using System;

namespace ComplexNumbers
{
    public static class Complex
    {
        public enum Operation { Sum, Sub, Mul, Div };

        public static double Absolute(ComplexNumber z) => Math.Sqrt(z.Real * z.Real + z.Imaginary * z.Imaginary);

        public static double Argument(ComplexNumber z)
        {
            if (z.Real == 0 && z.Imaginary > 0)
                return Math.PI / 2;
            else if (z.Real == 0 && z.Imaginary < 0)
                return -Math.PI / 2;
            else if (z.Real > 0)
                return Math.Atan(z.Imaginary / z.Real);
            else if (z.Real < 0 && z.Imaginary >= 0)
                return Math.Atan(z.Imaginary / z.Real) + Math.PI;
            else if (z.Real < 0 && z.Imaginary < 0)
                return Math.Atan(z.Imaginary / z.Real) - Math.PI;

            return 0;
        }

        public static ComplexNumber[] Root(ComplexNumber z, int n)
        {
            ComplexNumber[] roots = new ComplexNumber[n];

            for (int k = 0; k < n; k++)
            {
                roots[k].Real = Math.Pow(Absolute(z), 1.0 / n) * Math.Cos((Argument(z) + (2 * k * Math.PI)) / n);
                roots[k].Imaginary = Math.Pow(Absolute(z), 1.0 / n) * Math.Sin((Argument(z) + (2 * k * Math.PI)) / n);
            }
            return roots;
        }

        public static ComplexNumber Pow(ComplexNumber z, int n)
        {
            if ( n < 0)  // Se n è minore di 0 usa De Moivre
            {
                if (z.Equals(new ComplexNumber(0, 0)))
                    return new ComplexNumber
                    {
                        Real = Math.Pow(Absolute(z), n) * Math.Cos(Argument(z) * n),
                        Imaginary = Math.Pow(Absolute(z), n) * Math.Sin(Argument(z) * n)
                    };
                else
                    return new ComplexNumber(0, 0);

            }
            else if( n > 0 ) // se n è maggiore di 0, prodotto notevole, non usa Pow, Sin e Cos di Math
            {
                ComplexNumber r = z;

                for (int i = 1; i < n; i++)
                    r *= z;

                return r;
            }
            else // se n = 0 ritorna 1+0i
            {
                return new ComplexNumber(1,0);
            }
        }

        public static ComplexNumber Calculate(ComplexNumber a,  ComplexNumber b, Operation operation)
        {
            ComplexNumber result = new ComplexNumber();

            switch (operation)
            {
                case Operation.Sum:
                    result.Real = a.Real + b.Real;
                    result.Imaginary = a.Imaginary + b.Imaginary;
                    break;

                case Operation.Sub:
                    result.Real = a.Real - b.Real;
                    result.Imaginary = a.Imaginary - b.Imaginary;
                    break;

                case Operation.Mul:
                    result.Real = (a.Real * b.Real) - (a.Imaginary * b.Imaginary);
                    result.Imaginary = (a.Real * b.Imaginary) + (b.Real * a.Imaginary);
                    break;

                case Operation.Div:
                    result.Real = ((a.Real * b.Real) + (a.Imaginary * b.Imaginary)) / ((b.Real * b.Real) + (b.Imaginary * b.Imaginary));
                    result.Imaginary = ((b.Real * a.Imaginary) - (a.Real * b.Imaginary)) / ((b.Real * b.Real) + (b.Imaginary * b.Imaginary));
                    break;

            }

            return result;
        }

        public static ComplexNumber Exp(ComplexNumber a)
        {
            if( a.Imaginary >= 0)
                return new ComplexNumber(Absolute(a) * Math.Cos(Argument(a)), Absolute(a) * Math.Sin(Argument(a)));
            else 
                return new ComplexNumber(Absolute(a) * Math.Cos(Argument(a)), -Absolute(a) * Math.Sin(Argument(a)));
        }

        public static ComplexNumber Cos(ComplexNumber a)
        {
            return new ComplexNumber(Math.Cos(a.Real) * Math.Cosh(a.Imaginary), - Math.Sin(a.Real) * Math.Sinh(a.Imaginary));
        }

        public static ComplexNumber Sin(ComplexNumber a)
        {
            return new ComplexNumber(Math.Sin(a.Real) * Math.Cosh(a.Imaginary), Math.Cos(a.Real) * Math.Sinh(a.Imaginary));
        }

        public static string ToString(ComplexNumber z)
        {
            string realstr = z.Real.ToString();
            string imgstr = z.Imaginary.ToString();
            string str;

            if (realstr == "0")
                realstr = "";

            str = realstr;

            if (imgstr == "1")
                imgstr = "i";
            else if (imgstr == "0")
                imgstr = "";
            else if (!imgstr.Contains("-"))
                imgstr = "+" + imgstr + "i";
            else
                imgstr += "i";

            str += imgstr;

            return str;
        }
    }

    public struct ComplexNumber
    {
        public double Real;
        public double Imaginary;

        public ComplexNumber(double Real = 0, double Imaginary = 0)
        {
            this.Real = Real;
            this.Imaginary = Imaginary;
        }

        public static implicit operator ComplexNumber((double Real, double Imaginary) num) => new ComplexNumber(num.Real, num.Imaginary);
        public static ComplexNumber operator +(ComplexNumber a, ComplexNumber b) => Complex.Calculate(a, b, Complex.Operation.Sum);
        public static ComplexNumber operator -(ComplexNumber a, ComplexNumber b) => Complex.Calculate(a, b, Complex.Operation.Sub);
        public static ComplexNumber operator *(ComplexNumber a, ComplexNumber b) => Complex.Calculate(a, b, Complex.Operation.Mul);
        public static ComplexNumber operator /(ComplexNumber a, ComplexNumber b) => Complex.Calculate(a, b, Complex.Operation.Div);
        public static ComplexNumber operator +(double a, ComplexNumber b) => Complex.Calculate(new ComplexNumber(a, 0), b, Complex.Operation.Sum);
        public static ComplexNumber operator -(double a, ComplexNumber b) => Complex.Calculate(new ComplexNumber(a, 0), b, Complex.Operation.Sub);
        public static ComplexNumber operator *(double a, ComplexNumber b) => Complex.Calculate(new ComplexNumber(a, 0), b, Complex.Operation.Mul);
        public static ComplexNumber operator /(double a, ComplexNumber b) => Complex.Calculate(new ComplexNumber(a,0), b, Complex.Operation.Div);
        public static bool operator <(ComplexNumber a, ComplexNumber b) => Complex.Absolute(a) < Complex.Absolute(b);
        public static bool operator >(ComplexNumber a, ComplexNumber b) => Complex.Absolute(a) > Complex.Absolute(b);
        public static ComplexNumber operator ^(ComplexNumber a, int n) => Complex.Pow(a, n);

        public void Conjugate() => Imaginary = -Imaginary;
        
    }

}
