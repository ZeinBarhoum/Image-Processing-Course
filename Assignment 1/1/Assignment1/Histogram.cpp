#include "opencv2\opencv.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//get the maximum value in a vector
long Get_Max(vector<long> v)
{
	long max = 0;
	for (int i = 0;i < v.size();i++)
	{
		if (v[i] > max)max = v[i];
	}
	return max;
}
//get the number of pixels in each level
vector<long> NoPix(Mat image)
{
	vector<long>noP(256);
	for (int x = 0;x < image.cols;x++) {
		for (int y = 0;y < image.rows; y++) {
			noP[image.at<uint8_t>(y, x) + 0]++;
		}
	}
	return noP;
}
Mat Get_Histogram(Mat image)
{
	// Create a black image with 256*256 size and gray_scale values
	Mat histogram = Mat::zeros(256, 256, CV_8UC1);
	vector<long>noP = NoPix(image); // Get the number of pixels of each Gray level
	long max = Get_Max(noP); // Get the maximum value in Nop vector
	for (int i = 0;i < 256;i++)
	{
		for (int j = 0;j < 256 - (noP[i] * 1.0 / max) * 256;j++)
		{
			histogram.at<uint8_t>(j, i) = 255;
		}
		for (int j = 256 - (noP[i] * 1.0 / max) * 256;j <256;j++)
		{
			histogram.at<uint8_t>(j, i) = 0;
		}
	}
	return histogram;
}


vector<int> TF(Mat image)
{
	vector<int> s(256);
	vector<long>noP = NoPix(image);
	long total = image.cols*image.rows;
	double accsum[256];
	double sum = 0;
	for (int i = 0;i < 256;i++)
	{
		sum += noP[i];
		accsum[i] = (sum*1.0 / total);
		s[i] = round(accsum[i] * 255);
	}
	return s;
}
Mat Apply_TF(Mat image, vector<int>TF)
{
	Mat modified = Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (int x = 0;x < image.cols;x++) {
		for (int y = 0;y < image.rows; y++) {
			modified.at<uint8_t>(y, x) = TF[image.at<uint8_t>(y, x) + 0];
		}
	}
	return modified;
}
vector<int> Inv_TF(vector<int> TF)
{
	
	vector<int> ITF(TF.size());
	for (int i = 0;i < TF.size();i++)ITF[i] = -1;
	for (int i = 0;i < TF.size();i++)
	{
		if(ITF[TF[i]]==-1)
			ITF[TF[i]] = i;
	}
	int now = 0;
	for (int i = 0;i < TF.size();i++)
	{
		if (ITF[i] == -1)ITF[i] = now;
		else now = ITF[i];

		cout << TF[i]<<" "<<ITF[i] << endl;
	}
	return ITF;
}
int main()
{
/*	string file = "tire.bmp";//="check.bmp"
	Mat test = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
	Mat his = Get_Histogram(test);
	imshow("histogram", his);
	imshow("image", test);
	waitKey();*/

	/*string file = "tire.bmp";//="check.bmp"
	Mat test = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
	Mat Eq = Apply_TF(test, TF(test));
	Mat newhis = Get_Histogram(Eq);
	imshow("new histogram", newhis);
	imshow("EQimage", Eq);
	waitKey();
	*/
	string file1 = "tire.bmp";
	string file2= "check.bmp";
	Mat test1 = imread(file1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat test2= imread(file2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat Eq = Apply_TF(test1, TF(test1));
	Mat Match = Apply_TF(Eq, Inv_TF(TF(test2)));
	Mat newhis = Get_Histogram(Match);
	imshow("new histogram", newhis);
	imshow("Matchimage", Match);
	waitKey();
}