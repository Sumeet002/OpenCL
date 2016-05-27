#include "ocl.h"
#include <opencv2/opencv.hpp>
#define KERNEL_SOURCE_PATH "./borders.cl"

using namespace cv;

void oclBorders(cl_uchar *src_img_, cl_uchar *out_img_ , int W , int H , int B,char *deviceName, int deviceNameLen){

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
  	cl_kernel borders_kernel;
  	borders_kernel = clCreateKernel( params.hProgram, "borders_kernel", &status);

	STATUSCHKMSG("kernel handle");	

	//set image format	
	cl_image_format clImageFormat;
	 
	clImageFormat.image_channel_order = CL_R;
	clImageFormat.image_channel_data_type = CL_UNORM_INT8;

	// New in OpenCL 1.2, need to create image descriptor.
    	cl_image_desc src_clImageDesc;
    	src_clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    	src_clImageDesc.image_width = W;
    	src_clImageDesc.image_height = H;
    	src_clImageDesc.image_row_pitch = 0;
    	src_clImageDesc.image_slice_pitch = 0;
    	src_clImageDesc.num_mip_levels = 0;
    	src_clImageDesc.num_samples = 0;
    	src_clImageDesc.buffer = NULL;

	cl_image_desc out_clImageDesc;
    	out_clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    	out_clImageDesc.image_width = W+2*B;
    	out_clImageDesc.image_height = H;
    	out_clImageDesc.image_row_pitch = 0;
    	out_clImageDesc.image_slice_pitch = 0;
    	out_clImageDesc.num_mip_levels = 0;
    	out_clImageDesc.num_samples = 0;
    	out_clImageDesc.buffer = NULL;

	cl_mem dsrc_img_ = clCreateImage(params.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &clImageFormat,&src_clImageDesc,src_img_, &status);
	STATUSCHKMSG("Src clCreateImage");

	cl_mem dout_img_ = clCreateImage(params.context, CL_MEM_READ_WRITE, &clImageFormat,&out_clImageDesc,out_img_,&status);
	STATUSCHKMSG("Out clCreateImage");
	
	size_t origin[3] = {0,0,0};
	size_t region[3] = {(W+2*B),H,1};
	
	
	//create kernel arguments
  	
  	status = clSetKernelArg(borders_kernel,0,sizeof(cl_mem), (void*) &dsrc_img_);
  	STATUSCHKMSG("in arg 1 setting");
	
	status = clSetKernelArg(borders_kernel,1,sizeof(cl_mem), (void*) &dout_img_);
  	STATUSCHKMSG("in arg 2 setting");

	status = clSetKernelArg(borders_kernel,2,sizeof(cl_uint), (void*)&W);
	STATUSCHKMSG("in arg 3 setting");

	status = clSetKernelArg(borders_kernel,3,sizeof(cl_uint), (void*)&H);
	STATUSCHKMSG("in arg 4 setting");

	status = clSetKernelArg(borders_kernel,4,sizeof(cl_uint), (void*)&B);
	STATUSCHKMSG("in arg 5 setting");

	
	// enqueue a kernel run call
	size_t GWSize[]={(W+2*B),H,1};
  	status = clEnqueueNDRangeKernel(params.queue,borders_kernel,2, NULL, GWSize ,NULL,0,NULL,NULL);	
	STATUSCHKMSG("kernel enqueue");
	
	status = clFinish(params.queue);
  	STATUSCHKMSG("clFinish");

	cl_event events[1];
  
	// read output result
  	status = clEnqueueReadImage(params.queue, dout_img_, CL_TRUE, origin,region,0, 0, out_img_, 0,NULL,&events[0]);
  	STATUSCHKMSG("read output");

	// wait for read buffer to complete the read of output produce by kernel
  	status = clWaitForEvents(1, &events[0]);
  	STATUSCHKMSG("read event not completed");
		
	clReleaseEvent(events[0]);

	clReleaseKernel(borders_kernel);
  	clReleaseProgram(params.hProgram);
  	clReleaseCommandQueue(params.queue);
  	clReleaseContext(params.context);

}


int main(){

	Mat src_img;

	src_img = imread("test.png",CV_LOAD_IMAGE_GRAYSCALE);

	int img_size = src_img.cols * src_img.rows;

	//Display input Image
	imshow("Input Image" , src_img);
	waitKey(30);
	int B = 50; //border Columns selected for mirror symmetry
	Mat out_img(src_img.rows,src_img.cols + 2*B,CV_8U);

	unsigned char *src_img_ = src_img.ptr<unsigned char>(0);
	unsigned char *out_img_ = out_img.ptr<unsigned char>(0);

	char deviceName[BUFFER_SIZE];
    	char deviceNameLen = BUFFER_SIZE;

	oclBorders(src_img_,out_img_,src_img.cols,src_img.rows,B,deviceName,deviceNameLen);

	//Display input Image
	imshow("Output Image" , out_img);
	waitKey(30);
	while(1);	

	return 0;
}
