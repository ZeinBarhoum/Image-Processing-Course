#include "opencv2\opencv.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;

Mat Fimage(Mat image, string filter_file)
{
	Mat fimage = Mat::zeros(image.rows, image.cols, CV_8UC1);
	ifstream fin;
	fin.open(filter_file);
	int fcols, frows;
	fin >> fcols >> frows;
	vector<vector<double>> filter;
	vector <double> v(fcols);
	int sum = 0;
	for (int i = 0;i < frows;i++)
	{
		for (int j = 0;j < fcols;j++)
		{
			fin >> v[j];
			sum += v[j];
		}
		filter.push_back(v);
	}

	for (int i = 0;i < frows;i++)
	{
		for (int j = 0;j < fcols;j++)
		{
			filter[i][j] = filter[i][j] / sum;
		}
	}
	/*for (int i = 0;i < frows;i++)
	{
		for (int j = 0;j < fcols;j++)
		{
			cout << filter[i][j] << " ";
		}
		cout << endl;
	}*/
	for (int r = 0;r < image.rows;r++)
	{
		for (int c = 0;c < image.cols;c++)
		{
			double g = 0;
			for (int i = -(frows - 1) / 2;i <= (frows - 1) / 2;i++)
			{
				for (int j = -(fcols - 1) / 2;j <= (fcols - 1) / 2;j++)
				{
					int n = r + i;
					int m = c + j;
					int o = i + (frows - 1) / 2;
					int p = j + (fcols - 1) / 2;

					if (n < 0)n = 0;
					if (m < 0)m = 0;
					if (n >= image.rows)n = image.rows - 1;
					if (m >= image.cols)m = image.cols - 1;
					//	cout << n << " " << m << " " << o << " " << p << endl;
					g += filter[o][p] * image.at<uint8_t>(n, m);
				}
			}
			fimage.at<uint8_t>(r, c) = g;
		}

	}
	return fimage;
}
Mat KthFimage(Mat image, string filter_file,int k)
{
	Mat fimage = Mat::zeros(image.rows, image.cols, CV_8UC1);
	ifstream fin;
	fin.open(filter_file);
	int fcols, frows;
	fin >> fcols >> frows;
	vector<int>v(fcols*frows);

	for (int r = 0;r < image.rows;r++)
	{
		for (int c = 0;c < image.cols;c++)
		{
			int z = 0;
			for (int i = -(frows - 1) / 2;i <= (frows - 1) / 2;i++)
			{
				for (int j = -(fcols - 1) / 2;j <= (fcols - 1) / 2;j++)
				{
					int n = r + i;
					int m = c + j;
					if (n < 0)n = 0;
					if (m < 0)m = 0;
					if (n >= image.rows)n = image.rows - 1;
					if (m >= image.cols)m = image.cols - 1;
					//	cout << n << " " << m << " " << o << " " << p << endl;
					v[z] = image.at<uint8_t>(n, m);
					z++;
				}
			}
			sort(v.begin(), v.end());
			fimage.at<uint8_t>(r, c) = v[k];
		}

	}
	return fimage;
}
int main()
{
	string filter_file = "filter3.txt";
	Mat image = imread("blood.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat filterd = Fimage(image, filter_file);
	Mat filterd2 = KthFimage(image, filter_file,4);
	imshow("Original", image);
	imshow("Filterd", filterd);
	imshow("Filterd2", filterd2);
	waitKey();

}