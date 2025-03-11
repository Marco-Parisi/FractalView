using System;

namespace FractalWPF.Utility
{
    public class ShiftBuffer
    {
        private int[] tempBuff;
        private int[] Buffer;

        private int Width;
        private int Height;

        public readonly int Length;

        public ShiftBuffer(int width, int height)
        {
            Length = width * height;
            tempBuff = Buffer = new int[Length];
            Width = width;
            Height = height;
        }

        public void SetBuffer(int value, int index)
        {
            if (index < 0 || index > Length)
                throw new IndexOutOfRangeException($"ShiftBuffer index \"{index}\" is out of range");
            
            Buffer[index] = value;
        }

        public int GetBuffer(int index)
        {
            if (index < 0 || index > Length)
                throw new IndexOutOfRangeException($"ShiftBuffer index \"{index}\" is out of range");

           return Buffer[index];
        }

        public void ShiftRight()
        {
            tempBuff[0] = 0;

            Array.Copy(Buffer, 0, tempBuff, 1, Length - 1);
            Array.Copy(tempBuff, 0, Buffer, 0, Length);
        }

        public void ShiftLeft()
        {
            Array.Copy(Buffer, 1, Buffer, 0, Length - 1);
        }

        public void ShiftDown()
        {
            tempBuff[0] = 0;

            Array.Copy(Buffer, 0, tempBuff, Width , Length - Width);
            Array.Copy(tempBuff, 0, Buffer, 0, Length);
        }

        public void ShiftUp()
        {
            Array.Copy(Buffer, Width, Buffer, 0, Length - Width);
        }
    }
}
