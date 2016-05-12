#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main(){

	Mat src_img;
	
	src_img = imread("sample.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	
	imshow("Original Image",src_img);

	GaussianBlur(src_img,src_img,Size(3,3),1,1);

	

	imshow("Blurred Image",src_img);
	waitKey(30);
	while(1){
		if(waitKey(30) == 27){
			cout << "esc key pressed" << endl;
			break;

		}
	
	}
	return 0;
	
}
