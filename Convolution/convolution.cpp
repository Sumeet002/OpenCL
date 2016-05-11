#include "ocl.h"
#define KERNEL_SOURCE_PATH "./convolution.cl"

void openCLConvolute(){


}

int main(){

	//Input image
	Mat src_img;
	src_img = imread("sample.jpg",-1);

	//Resize Image
	int scale = 2 ;
	Size size(src_img.cols/scale,src_img.rows/scale);
	resize(src_img,src_img,size);

	//Display input Image
	imshow("Input Image" , src_img);
	waitKey(30);

	Mat out_img;
	out_img.create(src_img.size(),src_img.type());

	uchar *src_img_ = src_img.ptr<uchar>(0);
	uchar *out_img_ = out_img.ptr<uchar>(0);




	openCLConvolute((cl_uchar *)src_img_,(cl_uchar *)out_img_,);
	



}
