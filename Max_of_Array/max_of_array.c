#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <CL/cl.h>

#define KERNEL_MAX_SRC_SIZE 0x1000000

void err_check( int err, char err_code[30] ) {
        if ( err != CL_SUCCESS ) {
                printf("Error: %s %d", err_code,err);
                exit(-1);
        }
}


int main(){

	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platform;

	cl_device_id device_id = NULL;
	cl_uint ret_num_device;

	cl_context context = NULL;
	
	cl_command_queue command_queue = NULL;
	
	cl_program program =NULL;
	cl_kernel kernel=NULL;
	

	cl_int err;
	cl_build_status status;

	size_t logSize;
	char *programLog;

	//step0: Initialising Input
	int i;	
	int numElements = 409600;
	int WorkGroupSize = 16;
	size_t numWorkGroups = numElements / WorkGroupSize ; // Process in groups of WorkGroupSize
	size_t in_arr_size = numElements * sizeof(float);
	size_t out_arr_size = numWorkGroups*sizeof(float);
	float arrA[numElements];
	float arrB[numWorkGroups];

	for(i=0;i<numElements;i++){
		arrA[i]=i;	
	}
	

	//step1 : Getting Platform ID
	err= clGetPlatformIDs(1,&platform_id,&ret_num_platform);
	err_check(err,"clGetPlatformIDs");

	//step2 : Get Device ID
	err = clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_DEFAULT,1,&device_id,&ret_num_device);
	err_check(err,"clGetDeviceIDs");

	//step3 : Create Context
	context = clCreateContext(NULL,1,&device_id,NULL,NULL,&err);
	err_check(err,"clCreateContext");

	//step 4: Create Command Queue
	command_queue = clCreateCommandQueue(context,device_id,0,&err);
	err_check(err,"clCreateCommandQueue");


	//step 5 : Reading Kernel Program

	size_t kernel_src_size;
	char *kernel_src;
	
	FILE *fp;
	fp = fopen("max_with_reduction.cl","r");

	kernel_src = (char *)malloc(KERNEL_MAX_SRC_SIZE);
	kernel_src_size = fread(kernel_src,1,KERNEL_MAX_SRC_SIZE,fp);

	fclose(fp);

	//step 6: create buffer memory object

	cl_mem A = clCreateBuffer(context,CL_MEM_READ_ONLY,in_arr_size,NULL,&err);
	cl_mem B = clCreateBuffer(context,CL_MEM_WRITE_ONLY,out_arr_size,NULL,&err);
	
	//step 7: enqueue the buffers
	err = clEnqueueWriteBuffer(command_queue, A, CL_TRUE, 0, in_arr_size, arrA, 0, NULL, NULL);
  	

	//step 8: create and build program
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src, (const size_t *)&kernel_src_size, &err);
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
	printf("-----Build passed------\n");
	
	// Step 9 : Create Kernel
	kernel = clCreateKernel(program,"max_with_reduction",&err );
	err_check(err, "clCreateKernel");

	
	// Step 10 : Set Kernel Arguments
 	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&A);
 	err_check(err, "Arg 0 : clSetKernelArg");
	
	err = clSetKernelArg( kernel,1, sizeof(float), NULL);
	err_check(err, "Arg 1 : clSetKernelArg");
		
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B);
 	err_check(err, "Arg 2 : clSetKernelArg");
	

	//step 11: Execute the kernel
    
    	size_t global_item_size = numElements;  // Process the entire elements
    	size_t local_item_size = WorkGroupSize; // Process in groups of 64
    	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	err_check(err, "clEnqueueNDRangeKernel");
	
	//step 12: Read results back to host
	err = clEnqueueReadBuffer(command_queue, B , CL_TRUE, 0, out_arr_size , arrB, 0, NULL, NULL);
	err_check(err,"clEnqueueReadBuffer");

	float max = 0.;
	for(i = 0; i < numWorkGroups; i++ ){
		if(arrB[i] > arrB[i+1])
			max=arrB[i];
		else
			max=arrB[i+1];	
	}

	printf("Max of the elements of array is: %f\n",max);
	
		
	//step 13: Clean up
	err = clFlush(command_queue);
    	err = clFinish(command_queue);
    	err = clReleaseKernel(kernel);
    	err = clReleaseProgram(program);
    	err = clReleaseMemObject(A);
    	err = clReleaseMemObject(B);
    	
    	err = clReleaseCommandQueue(command_queue);
    	err = clReleaseContext(context);
	


	return 0;
}
