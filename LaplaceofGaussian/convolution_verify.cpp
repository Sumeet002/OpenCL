#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main(){

	Mat src_img;
	
	src_img = imread("original.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	
	imshow("Original Image",src_img);

	/*Internally Laplacian uses Gaussian blur first to remove noise, 
	so this cannot be used as a test to verify the result of only the Laplacian operator*/

	Laplacian( src_img, src_img, CV_8U, 3, 1, 0, BORDER_DEFAULT);

	imshow("Laplacian filtered Image",src_img);
	waitKey(30);
	while(1){
		if(waitKey(30) == 27){
			cout << "esc key pressed" << endl;
			break;

		}
	
	}
	return 0;
	
}
