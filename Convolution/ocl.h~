#include<CL/cl.h>
#include "common.h"

/* common pragma defination that are goin to be used through out the OpenCL programs */
#define GROUP_SIZE 4
#define NUMTHREAD  4
#define STATUSCHKMSG(x) if(status != CL_SUCCESS) { cout<<"\n Operation is not successful : ";cout<<x<<"\n"; exit(1);}
#define STOP    cout<<"program terminated here";exit(-1);
#define BUFFER_SIZE 100                /* size to define temprary buffer size , which hold process, platform, device name */ 
#define MAX_NUM_OF_PLATFORM_AND_DEVICE_ON_EACH_NODE 4
#define BUILD_USER_DEF_KERNEL	1
#define BUILD_LIBR_DEF_KERNEL	0


//Structure contatining all parameters for opencl initialisation
typedef struct oclEnvParamL{

	//platform related parameters
	cl_uint maxNumOfPlatforms;
	cl_platform_id *platforms;
	cl_uint numOfPlatforms;

	//device related parameters
	cl_uint maxNumOfDevices;
	cl_device_id *devices;
	cl_uint numOfDevices;
	cl_device_type deviceType;

	//context and progrm compilation parameters
	cl_context context;
	cl_command_queue queue;
	
	//user defined kernel
	cl_uint kernelBuildSpecs;
	char pathtoKernelSrc[100];
	cl_program hProgram;

	
}oclEnvParams; 

//Read the kernel Source from .cl file
char *readKernelSrc(char* path){
	
	size_t kernel_src_size;
	char *kernel_src = NULL;
 
	FILE *fp;
	fp = fopen(path,"r");

	struct stat st;
	stat(path, &st);
	kernel_src_size = st.st_size;

	kernel_src = (char *)malloc(kernel_src_size + 1);
	kernel_src_size = fread(kernel_src, 1, kernel_src_size,fp);
	
	fclose(fp);
	kernel_src[kernel_src_size] = 0;
	
	return kernel_src;

}


//Setting environment for kernel compilation

void setOClExeEnv(oclEnvParams *params){

	cl_int status = CL_SUCCESS;

	cl_int err;
	cl_build_status build_status;
	size_t logSize;
 	char *programLog;
	
	//basic initialisation of ExeEnvParamList variables
	params->platforms = (cl_platform_id *)malloc(params->maxNumOfPlatforms*sizeof(cl_platform_id));
	params->devices = (cl_device_id *)malloc(params->maxNumOfDevices*sizeof(cl_device_id));

	//get all available platform ids
	status = clGetPlatformIDs(params->maxNumOfPlatforms,params->platforms,&(params->numOfPlatforms));
	STATUSCHKMSG("clGetPlatformIDs Failed ");

	// from a list of OpenCL capable platforms select one GPU device.
	//cout<<" Available OpenCL Platforms : \n";

	unsigned platformCount = 0;
	unsigned devCount = 0 ;
	cl_uint deviceFound  = CL_FALSE;
	cl_uint platItrFlag = CL_TRUE;

	
	for(platformCount = 0 ; platformCount < params->numOfPlatforms ; platformCount ++){

		status = clGetDeviceIDs(params->platforms[platformCount],params->deviceType,params->maxNumOfDevices,
					params->devices,&(params->numOfDevices));
		STATUSCHKMSG(status);

		// select every device available on system one by one 
		for(devCount = 0 ; devCount < params->numOfDevices ; devCount++){

			size_t paramValueSizeRet;
			cl_bool isDeviceAvail;
			status = clGetDeviceInfo((params->devices)[devCount],CL_DEVICE_AVAILABLE,sizeof(cl_bool), 
						  &isDeviceAvail,&paramValueSizeRet);

			STATUSCHKMSG(status);

			if( isDeviceAvail & CL_DEVICE_AVAILABLE){

				cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)params->platforms[platformCount],0};
				
				// create context type for given device type
                        	cl_context_properties* cprops = ( NULL == params->platforms[platformCount] ) ? NULL : cps;
                        	cl_uint numOfDevWithContext = 1;
				params->context = clCreateContext(cprops, numOfDevWithContext, ((params->devices)+devCount), NULL, NULL, &status);
				STATUSCHKMSG(" context ");
				

				// create command queue
                        	params->queue = clCreateCommandQueue( params->context, (params->devices)[devCount], 0, &status);
                        	STATUSCHKMSG("command queue");

				if( params->kernelBuildSpecs == BUILD_USER_DEF_KERNEL ) {
				
					// create a CL program using kernel source
                        		char* kProgramSrc = readKernelSrc(params->pathtoKernelSrc);
					
                        		size_t sourceSize = strlen(kProgramSrc);
					
					params->hProgram = clCreateProgramWithSource(params->context, 1, (const char **)&kProgramSrc, 
											(const size_t *)&sourceSize, &status);
                        		STATUSCHKMSG("create source handle");
					
                        		// build the program
                        		err = clBuildProgram(params->hProgram,numOfDevWithContext, ((params->devices)+devCount),NULL,NULL,NULL);
					//cout << status << endl;                        		
					//STATUSCHKMSG("build");
					if(err != CL_SUCCESS){
		
						//check build error and build status first
						clGetProgramBuildInfo(params->hProgram,params->devices[devCount],CL_PROGRAM_BUILD_STATUS,
								      sizeof(cl_build_status),&build_status,NULL);

		
						//check build log
						clGetProgramBuildInfo(params->hProgram,params->devices[devCount],
									CL_PROGRAM_BUILD_LOG,0,NULL,&logSize);
						programLog=(char *)calloc(logSize+1,sizeof(char));
		
						clGetProgramBuildInfo(params->hProgram,params->devices[devCount],CL_PROGRAM_BUILD_LOG,
									logSize+1,programLog,NULL);
						//printf("Build Failed; error=%d,status=%d,programLog:nn%s",err,build_status,programLog);
						cout << "Build Failed; error: "<<err<<" status: "<<build_status<<" programLog: "<<programLog<<endl;
						
						free(programLog);
	
					}	


				}//end of if
				
				deviceFound = CL_TRUE;
                        	break;

			}// end of if			
	
		}// end of for 
		if( deviceFound ){break;}
	}// end of for
}//end of setOClExeEnv
















