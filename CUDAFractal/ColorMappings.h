#include <cstdio>
#include <vector_types.h>
#include <vector>
#include "spline.h"

struct Colors
{
private:
	const int bLength;
public:
	const int Length;
	uchar3* CBuffer;

	Colors(int length) : Length(length), bLength(length-1)
	{
		CBuffer = new uchar3[length]{ 0 };
	}

	void SetBuffer(int index, unsigned char x, unsigned char y, unsigned char z)
	{
		if (index > bLength) index = bLength;
		if (index < 0) index = 0;

		CBuffer[index].x = x;
		CBuffer[index].y = y;
		CBuffer[index].z = z;
	}
};

class ColorMappings
{
public:
	Colors UltraFractal{ 4096 };
    Colors GrayScale{ 512 };
	Colors GrayScaleInv{ 1024 };
    Colors DeepBlue{ 128 };

public:
	ColorMappings()
	{
		GrayScale.SetBuffer(0, 255, 255, 255);
		for (int i = 0; i < GrayScale.Length; i++)
		{
			int comp = (int)((i * 255.0) / (float)GrayScale.Length);
			GrayScale.SetBuffer(GrayScale.Length - i, comp, comp, comp);
		}		

		for (int i = 0; i < GrayScaleInv.Length; i++)
		{
			int comp = (int)((i * 255.0) / (float)GrayScaleInv.Length);
			GrayScaleInv.SetBuffer(i, comp, comp, comp);
		}

		DeepBlue.CBuffer[DeepBlue.Length - 1].x = 255;
		DeepBlue.CBuffer[DeepBlue.Length - 1].y = 255;
		for (int i = 0; i < DeepBlue.Length; i++)
			DeepBlue.SetBuffer(i, 0.1 * i, 0.6 * i, i);
		
        std::vector<double> X =
		{
			0.0,
			0.16 * UltraFractal.Length,
			0.3 * UltraFractal.Length,
			0.6425 * UltraFractal.Length,
			0.8575 * UltraFractal.Length
        };

		std::vector<double> yR = { 0, 32, 237, 255, 0};
	    std::vector<double> yG = { 7, 107, 255, 170, 2};
		std::vector<double> yB = { 100, 203, 255, 0 , 0};

		tk::spline sR(X, yR, tk::spline::linear);
		tk::spline sG(X, yG, tk::spline::linear);
		tk::spline sB(X, yB, tk::spline::linear);

		double conv = 0.8575;

        for (int i = 0; i < UltraFractal.Length; i++)
            UltraFractal.SetBuffer(i, (int)sR(i * conv), (int)sG(i * conv), (int)sB(i * conv));      
	}

public:
	Colors GetColorMap(int idx)
	{
		if (idx < 0 || idx>3) return UltraFractal;

		switch (idx)
		{
			case 0: return UltraFractal;
			case 1: return GrayScale;
			case 2: return GrayScaleInv;
			case 3: return DeepBlue;
		}
	}
};