#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sys/time.h>

using namespace cv;
using namespace std;


#define In_IMAGE_HEIGHT 1080
#define In_IMAGE_WIDTH 1920
#define Scale 4

int main(int argc, char* argv[])
{

	struct timeval t0,t1,t2,t3,t4;
	namedWindow("Original Image",WINDOW_AUTOSIZE); //create a window called "Original Image"
	namedWindow("DownScaled Image",WINDOW_AUTOSIZE); //create a window called "DownScaled Image"

	static Mat I_RGB(In_IMAGE_HEIGHT,In_IMAGE_WIDTH,CV_8UC3);
	static Mat O_RGB;
	
	Size size(In_IMAGE_WIDTH/4,In_IMAGE_HEIGHT/4);	

	I_RGB=imread("sample.png");
	imshow("Original Image" , I_RGB);
	waitKey(30);

	gettimeofday(&t0,0);
	
	resize(I_RGB,O_RGB,size);	

	gettimeofday(&t1,0);
	long elapsed = t1.tv_usec-t0.tv_usec;
	cout << "After Averaging:"<<  double(elapsed)/1000000 << endl;
	
	cout << "done " <<endl;

	while(true){
		imshow("DownScaled Image",O_RGB);
		waitKey(30);
	}

	return 0;

}
