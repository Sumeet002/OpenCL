#include "ocl.h"
#define KERNEL_SOURCE_PATH      "./PrefixSum_kernel.cl"
#define ARRAY_LENGTH   1000

void fillInArray(cl_int *hInArray, size_t length){
	for(size_t count=0; count< length; count++)
		hInArray[count] = rand()%10;
}

void openCLPrefixSum(cl_int *hInArray, cl_int *hOutArray, size_t length, char *deviceName, int deviceNameLen){

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
  	cl_kernel prefixSum_kernel;
  	prefixSum_kernel = clCreateKernel( params.hProgram, "prefixSum_kernel", &status);
	
  	STATUSCHKMSG("kernel handle");	
	
	//create input array
  	cl_mem dInArray = clCreateBuffer(params.context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, length * sizeof(cl_int), (void*)hInArray, &status);
  	STATUSCHKMSG("memory allocation ");

	//create output buffer
      cl_mem dOutArray = clCreateBuffer(params.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, length * sizeof(cl_int),(void*)hOutArray, &status);
  	STATUSCHKMSG("o/p memory allocation");
	
	//create space for row and col argument 
  	cl_int hLength = length;
  	cl_mem dLength = clCreateBuffer(params.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_int),(void*)&hLength, &status);
  	STATUSCHKMSG("scaler memory setting");

	//create kernel argument
  	//the input array
  	status  = clSetKernelArg(prefixSum_kernel,0,sizeof(cl_mem), (void*) &dInArray);
  	STATUSCHKMSG("in arg setting");
  
	//output array
  	status = clSetKernelArg(prefixSum_kernel, 1, sizeof(cl_mem), (void*) &dOutArray);
  	STATUSCHKMSG("out arg set");
  
	// scalar value to be multiplied
  	status  = clSetKernelArg(prefixSum_kernel,2, sizeof(cl_mem), (void*) &dLength);
  	STATUSCHKMSG("scalar value argument");

	  // enqueue a kernel run call
  	size_t globalThreads[] = { NUMTHREAD };       // number of global items in work dimension 
  	size_t localThreads[] = { GROUP_SIZE }; 	// number of work item per group
  	status = clEnqueueNDRangeKernel( params.queue,prefixSum_kernel,1, NULL, globalThreads, localThreads,0,NULL,NULL);
  	STATUSCHKMSG("kernel enqueue");
	
	status = clFinish(params.queue);
  	STATUSCHKMSG("clFinish");

	cl_event events[1];
  
	// read output result
  	status = clEnqueueReadBuffer(params.queue, dOutArray, CL_TRUE, 0, length * sizeof(cl_int),hOutArray, 0, NULL, &events[0]);
  	STATUSCHKMSG("read output");
 
	// wait for read buffer to complete the read of output produce by kernel
  	status = clWaitForEvents(1, &events[0]);
  	STATUSCHKMSG("read event not completed");

  	clReleaseEvent(events[0]);

	clReleaseKernel(prefixSum_kernel);
  	clReleaseProgram(params.hProgram);
  	clReleaseCommandQueue(params.queue);
  	clReleaseContext(params.context);

}//end_of_opencl_routine

int main(int argc,char *argv[]){
	
	int hInArrayLen = ARRAY_LENGTH;
	int *hInArray = new int[hInArrayLen];
    	int *hOutArray = new int[hInArrayLen];

	char deviceName[BUFFER_SIZE];
    	char deviceNameLen = BUFFER_SIZE;

	fillInArray((cl_int*) hInArray, (size_t)hInArrayLen);
	cout<<"\n Input array to calculate prefix sum value:\n";
     	   
	for(size_t count=0; count < 10; count++){
              cout<<" "<<hInArray[ count ];
	}
       
	cout<<"\n";

	openCLPrefixSum((cl_int*)hInArray, (cl_int*) hOutArray,(size_t) hInArrayLen, deviceName, deviceNameLen );

     	for(size_t count=0; count < 10; count++){
              cout<<" "<<hOutArray[ count ];
	}
        cout<<"\n";

} 


