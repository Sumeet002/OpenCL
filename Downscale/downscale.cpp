#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fcntl.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>



using namespace cv;
using namespace std;

#define MAX_SRC_SIZE (0x100000)
#define In_IMAGE_HEIGHT 1080
#define In_IMAGE_WIDTH 1920
#define Scale 4
#define Out_IMAGE_HEIGHT In_IMAGE_HEIGHT/Scale
#define Out_IMAGE_WIDTH In_IMAGE_WIDTH/Scale


void err_check( int err, char err_code[30] ) {
        if ( err != CL_SUCCESS ) {
                printf("Error: %s %d", err_code,err);
                exit(-1);
        }
}

int main(int argc, char* argv[])
{

	struct timeval t0,t1,t2,t3,t4;
	namedWindow("Original Image",WINDOW_AUTOSIZE); //create a window called "Original Image"
	namedWindow("DownScaled Image",WINDOW_AUTOSIZE); //create a window called "DownScaled Image"

	static Mat I_RGB(In_IMAGE_HEIGHT,In_IMAGE_WIDTH,CV_8UC3);
	static Mat I_RGBA(I_RGB.rows,I_RGB.cols,CV_8UC4);
	static Mat O_RGB(Out_IMAGE_HEIGHT,Out_IMAGE_WIDTH,CV_8UC3,Scalar(0));
	static Mat O_RGBA(O_RGB.rows,O_RGB.cols,CV_8UC4,Scalar(0));
	

	I_RGB=imread("sample.png");
	cvtColor(I_RGB,I_RGBA,COLOR_RGB2RGBA);
	imshow("Original Image" , I_RGBA);
	waitKey(30);
	

	uchar *Indata = I_RGBA.ptr<uchar>(0);
	uchar *Outdata = O_RGBA.ptr<uchar>(0);
	
	int In_imgSize = In_IMAGE_HEIGHT*In_IMAGE_WIDTH;
	int Out_imgSize = Out_IMAGE_HEIGHT*Out_IMAGE_WIDTH;

	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*OpenCL initialisation*/

	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platform;
 
	cl_device_id device_id = NULL;
	cl_uint ret_num_device;
 
	cl_context context = NULL;
 
	cl_command_queue command_queue = NULL;
 
	cl_program program = NULL;
 
	cl_kernel kernel = NULL;
 
	cl_int err;
	cl_build_status status;
	size_t logSize;
 	char *programLog;

	// step 1 : getting platform ID
	err = clGetPlatformIDs(1, &platform_id, &ret_num_platform);
	err_check(err,"clGetPlatformIDs");

	

	// step 2 : Get Device ID
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_device );
	err_check(err,"clGetDeviceIDs");

	

	// step 3 : Create Context
	context = clCreateContext(NULL,1,&device_id,NULL,NULL,&err);
	err_check(err, "clCreateContext");
	
	

	cl_bool sup;
	size_t rsize;
	clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(sup), &sup, &rsize);
	if (sup != CL_TRUE){
		printf("Image not Supported\n");
	}

	

	// Step 4 : Create Command Queue
	command_queue =  clCreateCommandQueue(context, device_id, 0, &err);
	err_check(err, "clCreateCommandQueue");

	// Step 5 : Reading Kernel Program
 
	
	size_t kernel_src_size;
	char *kernel_src_std;
 
	FILE *fp;
	fp = fopen("downscale.cl","r");
 
	kernel_src_std = (char *)malloc(MAX_SRC_SIZE);
	kernel_src_size = fread(kernel_src_std, 1, MAX_SRC_SIZE,fp);
 
	fclose(fp);

	//Create Image data format
	cl_image_format img_fmt;
 
	img_fmt.image_channel_order = CL_RGBA;
	img_fmt.image_channel_data_type = CL_UNORM_INT8;
	//img_fmt.image_channel_data_type = CL_UNSIGNED_INT8;

	// Step 6 : Create Image Memory Object
	cl_mem Out_image,In_image;
        
	printf("2\n");

	In_image = clCreateImage2D(context, CL_MEM_READ_ONLY, &img_fmt, In_IMAGE_WIDTH , In_IMAGE_HEIGHT, 0, 0, &err);
	err_check(err, "In_image: clCreateImage2D");

	Out_image = clCreateImage2D(context, CL_MEM_READ_WRITE,&img_fmt, Out_IMAGE_WIDTH, Out_IMAGE_HEIGHT , 0,0,&err);
	err_check(err, "Out_image: clCreateImage2D");
	
	// Copy Data from Host to Device
	cl_event event[5];

	size_t origin[] = {0,0,0}; // Defines the offset in pixels in the image from where to write.
	size_t In_region[] = {In_IMAGE_WIDTH, In_IMAGE_HEIGHT, 1}; // Size of object to be transferred
	size_t Out_region[] = {Out_IMAGE_WIDTH, Out_IMAGE_HEIGHT, 1}; // Size of object to be transferred

	// Step 7 : Create and Build Program
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_std, (const size_t *)&kernel_src_size, &err);
	err_check(err, "clCreateProgramWithSource");

	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	printf("-----Checking for any build errors------\n");
	if(err != CL_SUCCESS){
		
		//check build error and build status first
		clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_STATUS,sizeof(cl_build_status),&status,NULL);

		//check build log
		clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG,0,NULL,&logSize);
		programLog=(char *)calloc(logSize+1,sizeof(char));
		
		clGetProgramBuildInfo(program,device_id,CL_PROGRAM_BUILD_LOG,logSize+1,programLog,NULL);
		printf("Build Failed; error=%d,status=%d,programLog:nn%s",err,status,programLog);
		free(programLog);
	
	}

	// Step 8 : Create Kernel
	kernel = clCreateKernel(program,"downscale",&err );

	// Step 9 : Set Kernel Arguments
 
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&In_image);
 	err_check(err, "Arg 1 : clSetKernelArg");
	
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Out_image);
 	err_check(err, "Arg 2 : clSetKernelArg");


	gettimeofday(&t0,0);
	
	err = clEnqueueWriteImage(command_queue, In_image, CL_TRUE, origin, In_region, 0, 0, Indata, 0, NULL,&event[0]);
	err_check(err,"clEnqueueWriteImage AImage");

	// Step 10 : Execute Kernel 
	size_t GWSize[]={Out_IMAGE_WIDTH,Out_IMAGE_HEIGHT,1};
       
        err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, GWSize,NULL,1, event,&event[1]);
	

	// Step 11 : Read output Data, from Device to Host
	err = clEnqueueReadImage(command_queue, Out_image, CL_TRUE, origin,Out_region, NULL , 0, Outdata, 2, event, &event[2] );	
	err_check(err, "EnqueueReadImage");
	
	
	gettimeofday(&t1,0);
	long elapsed = t1.tv_usec-t0.tv_usec;
	cout << "After Averaging:"<<  double(elapsed)/1000000 << endl;

	clReleaseMemObject(Out_image);
	clReleaseMemObject(In_image);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
 
	free(kernel_src_std);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//cout << L << endl;
	cvtColor(O_RGBA,O_RGB,COLOR_RGBA2RGB);
	while(true){
		imshow("DownScaled Image",O_RGB);
		waitKey(30);
	}
	
	
	
	cout << "done " <<endl;
	
	

	return 0;

}
