#include "ocl.h"
#include <opencv2/opencv.hpp>
#define KERNEL_SOURCE_PATH "./naive_convolution2.cl"
#define PI_ 3.14 

using namespace cv;

float* createGaussianKernel(uint32_t size,float sigma)
{
    float* ret;
    uint32_t x,y;
    double center = size/2;
    float sum = 0;
    
    //allocate and create the gaussian kernel
    ret = (float *)malloc(sizeof(float) * size * size);
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            ret[ y*size+x] = exp( (((x-center)*(x-center)+(y-center)*(y-center))/(2.0f*sigma*sigma))*-1.0f ) / (2.0f*PI_*sigma*sigma);
            sum+=ret[ y*size+x];
        }
    }

    //normalize
    for(x = 0; x < size*size;x++)
    {
        ret[x] = ret[x]/sum;
    }

    //print the kernel so the user can see it
    printf("The generated Gaussian Kernel is:\n");
    for(x = 0; x < size; x++)
    {
        for(y=0; y < size; y++)
        {
            printf("%f ",ret[ y*size+x]);
        }
        printf("\n");
    }
    printf("\n\n");
    return ret;
}



void openCLConvolute(cl_uchar *src_img_ , cl_uchar *out_img_ , float *conv_mat, 
		     int img_sz , int W ,int conv_mat_sz, char *deviceName, int deviceNameLen){

	oclEnvParams params;
	cl_int status = CL_SUCCESS;
	
	params.maxNumOfPlatforms = params.maxNumOfDevices =  MAX_NUM_OF_PLATFORM_AND_DEVICE_ON_EACH_NODE;
	params.deviceType = CL_DEVICE_TYPE_ALL;
  	params.kernelBuildSpecs = BUILD_USER_DEF_KERNEL;
  	strcpy(params.pathtoKernelSrc,KERNEL_SOURCE_PATH);

	setOClExeEnv(&params);

	cl_device_id attachedDevWithQueue;
  	status = clGetCommandQueueInfo( params.queue, CL_QUEUE_DEVICE, sizeof(cl_device_id),&attachedDevWithQueue,NULL);

	// get device specific information
  	cl_device_type attachedDevTypeWithQueue;
  	status = clGetDeviceInfo(attachedDevWithQueue,CL_DEVICE_NAME,sizeof(char) * deviceNameLen,deviceName,NULL);
	cout << deviceName << endl;

	// create kernel handle
  	cl_kernel conv_kernel;
  	conv_kernel = clCreateKernel( params.hProgram, "conv_kernel", &status);

	STATUSCHKMSG("kernel handle");	
	
	//create input array
  	cl_mem dsrc_img_ = clCreateBuffer(params.context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,img_sz*sizeof(cl_uchar),(void*)src_img_, &status);
  	STATUSCHKMSG("i/p memory allocation ");

	
	//create convolution matrix array
	cl_mem dconv_mat = clCreateBuffer(params.context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						conv_mat_sz*conv_mat_sz*sizeof(cl_float),(void *)conv_mat, &status);
  	STATUSCHKMSG("i/p memory allocation ");



	//create output array
      cl_mem dout_img_ = clCreateBuffer(params.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,img_sz*sizeof(cl_uchar),(void*)out_img_, &status);
  	STATUSCHKMSG("o/p memory allocation");


	//create kernel arguments
  	
  	status = clSetKernelArg(conv_kernel,0,sizeof(cl_mem), (void*) &dsrc_img_);
  	STATUSCHKMSG("in arg 1 setting");
	
	status = clSetKernelArg(conv_kernel,1,sizeof(cl_mem), (void*) &dout_img_);
  	STATUSCHKMSG("in arg 2 setting");
	
	status = clSetKernelArg(conv_kernel,2,sizeof(cl_mem), (void*) &dconv_mat);
	STATUSCHKMSG("in arg 3 setting");

	status = clSetKernelArg(conv_kernel,3,sizeof(cl_uint), (void*)&W);
	STATUSCHKMSG("in arg 4 setting");
	
	int H= img_sz/W;
	status = clSetKernelArg(conv_kernel,4,sizeof(cl_uint), (void*)&H);
	STATUSCHKMSG("in arg 5 setting");
	
	status = clSetKernelArg(conv_kernel,5,sizeof(cl_uint), (void*)&conv_mat_sz);
	STATUSCHKMSG("in arg 6 setting");


	// enqueue a kernel run call
	size_t globalWorkItemSize = img_sz;
  	status = clEnqueueNDRangeKernel(params.queue,conv_kernel,1, NULL, &globalWorkItemSize ,NULL,0,NULL,NULL);	
	
  	STATUSCHKMSG("kernel enqueue");
	

	status = clFinish(params.queue);
  	STATUSCHKMSG("clFinish");

	cl_event events[1];
  
	// read output result
  	status = clEnqueueReadBuffer(params.queue, dout_img_, CL_TRUE, 0,img_sz*sizeof(cl_uchar),out_img_, 0, NULL, &events[0]);
  	STATUSCHKMSG("read output");

	// wait for read buffer to complete the read of output produce by kernel
  	status = clWaitForEvents(1, &events[0]);
  	STATUSCHKMSG("read event not completed");
		
	clReleaseEvent(events[0]);

	clReleaseKernel(conv_kernel);
  	clReleaseProgram(params.hProgram);
  	clReleaseCommandQueue(params.queue);
  	clReleaseContext(params.context);
	
	cout << "end" << endl;


}//end_of_opencl_routine

int main(){

	//Input image
	Mat src_img;
	src_img = imread("sample.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	

	//Resize Image
	if(src_img.cols > 1080){
		int scale = 2 ;
		Size size(src_img.cols/scale,src_img.rows/scale);
		resize(src_img,src_img,size);
	}
	
	int img_sz;
	img_sz = src_img.cols*src_img.rows;
	

	//Display input Image
	imshow("Input Image" , src_img);
	waitKey(30);

	Mat out_img(src_img.size(),CV_8U);
	cout << src_img.type() << src_img.size() << endl;

	unsigned char *src_img_ = src_img.ptr<unsigned char>(0);
	unsigned char *out_img_ = out_img.ptr<unsigned char>(0);
	

	int conv_mat_sz = 3 ;
	float conv_mat_sigma = 1;
	float *conv_mat=(float*)malloc(sizeof(float)*conv_mat_sz*conv_mat_sz);
	conv_mat=createGaussianKernel(conv_mat_sz,conv_mat_sigma);
	
	
	char deviceName[BUFFER_SIZE];
    	char deviceNameLen = BUFFER_SIZE;

	openCLConvolute((cl_uchar *)src_img_,(cl_uchar *)out_img_,conv_mat,img_sz,src_img.cols,conv_mat_sz,deviceName,deviceNameLen);

	//Display input Image
	imshow("Output Image" , out_img);
	waitKey(30);
	while(1);
	
	return 0;

}
