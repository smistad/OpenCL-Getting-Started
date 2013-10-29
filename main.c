#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> /* PRIuPTR and uintptr_t */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)


#define DEFINE_CHECK  \
    int CHECK_errors=0; \
    void *CHECK_malloc_tmp;
    

#define CHECKRET(msg,exitlabel)					\
    if (ret) {							\
	fprintf(stderr,"error: " msg " line %i: %i\n", __LINE__, ret);	\
	CHECK_errors++;						\
	goto exitlabel;						\
    }

#define CHECK_malloc(siz,lbl)					\
    CHECK_malloc_tmp=malloc(siz);				\
    if (!CHECK_malloc_tmp) {					\
	CHECK_errors++;						\
	fprintf (stderr,"out of memory, line %i\n", __LINE__);	\
	goto lbl;						\
    }



#define val_size 10000
char val[val_size];

int main(void) {
    DEFINE_CHECK;
    
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = CHECK_malloc(sizeof(int)*LIST_SIZE, err_malloc_A);
    int *B = CHECK_malloc(sizeof(int)*LIST_SIZE, err_malloc_B);

    for(i = 0; i < LIST_SIZE; i++) {
	A[i] = i;
	B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
	fprintf(stderr, "Failed to open kernel file.\n");
	CHECK_errors++;
	goto err_fopen;
    }

    char *source_str = CHECK_malloc(MAX_SOURCE_SIZE, err_malloc_source_str);
    size_t source_size= fread( source_str, 1, MAX_SOURCE_SIZE, fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret= clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CHECKRET ("clGetPlatformIDs",err_clGetPlatformIDs);

    printf("platform_id=%p, ret_num_platforms=%i\n",
	   platform_id, ret_num_platforms);

    ret = clGetDeviceIDs( platform_id,
			  CL_DEVICE_TYPE_GPU,
			  1,
			  &device_id,
			  &ret_num_devices);
    CHECKRET ("clGetDeviceIDs",err_clGetDeviceIDs);

    // Create an OpenCL context
    cl_context context =
	clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    CHECKRET ("clCreateContext",err_clCreateContext);

    // Create a command queue
    cl_command_queue command_queue =
	clCreateCommandQueue(context, device_id, 0, &ret);
    CHECKRET ("clCreateCommandQueue",err_clCreateCommandQueue);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj =
	clCreateBuffer(context, CL_MEM_READ_ONLY, 
		       LIST_SIZE * sizeof(int), NULL, &ret);
    CHECKRET ("clCreateBuffer",err_a_mem_obj);
    cl_mem b_mem_obj =
	clCreateBuffer(context, CL_MEM_READ_ONLY,
		       LIST_SIZE * sizeof(int), NULL, &ret);
    CHECKRET ("clCreateBuffer",err_b_mem_obj);
    cl_mem c_mem_obj =
	clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
		       LIST_SIZE * sizeof(int), NULL, &ret);
    CHECKRET ("clCreateBuffer",err_c_mem_obj);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			       LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    CHECKRET ("clEnqueueWriteBuffer",err_clEnqueueWriteBuffer_A);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			       LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
    CHECKRET ("clEnqueueWriteBuffer",err_clEnqueueWriteBuffer_B);

    // Create a program from the kernel source
    cl_program program =
	clCreateProgramWithSource(context,
				  1,
				  (const char**)&source_str,
				  &source_size,
				  &ret);
    CHECKRET("clCreateProgramWithSource",err_clCreateProgramWithSource);

    free(source_str); //XXX ok, can we do that while program is still alive ?

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret) {
	//cl_int ret0=ret; XX print it?
	size_t sizeused;
	ret = clGetProgramBuildInfo (program,
				     device_id,
				     CL_PROGRAM_BUILD_LOG,
				     val_size-1, //?
				     &val,
				     &sizeused);
	CHECKRET ("clGetProgramBuildInfo",err_clGetProgramBuildInfo);

	printf("clBuildProgram error: (sizeused %"PRIuPTR") '%s'\n",
	       (uintptr_t) sizeused, val);
    err_clGetProgramBuildInfo:
	goto err_clBuildProgram;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    CHECKRET ("clCreateKernel",err_clCreateKernel);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj);
    CHECKRET ("clSetKernelArg",err_clSetKernelArg);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj);
    CHECKRET ("clSetKernelArg",err_clSetKernelArg);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj);
    CHECKRET ("clSetKernelArg",err_clSetKernelArg);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Process in groups of 64
    ret = clEnqueueNDRangeKernel(command_queue,
				 kernel,
				 1,
				 NULL, 
				 &global_item_size,
				 &local_item_size,
				 0,
				 NULL,
				 NULL);
    CHECKRET ("clEnqueueNDRangeKernel",err_clEnqueueNDRangeKernel);

    // Read the memory buffer C on the device to the local variable C
    int *C = CHECK_malloc(sizeof(int)*LIST_SIZE, err_malloc_C);

    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
			      LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
    CHECKRET ("clEnqueueReadBuffer",err_clEnqueueReadBuffer);

    // Display the result to the screen
    for(i = 0; i < LIST_SIZE; i++)
	printf("%d + %d = %d\n", A[i], B[i], C[i]);

 err_clEnqueueReadBuffer:
    free(C);
 err_malloc_C:
    ret = clFlush(command_queue);
    CHECKRET ("clFlush",err_clFlush);
 err_clFlush:
    ret = clFinish(command_queue);
    CHECKRET ("clFinish",err_clEnqueueNDRangeKernel);
 err_clEnqueueNDRangeKernel:
 err_clSetKernelArg:
    ret = clReleaseKernel(kernel);
    CHECKRET ("clReleaseKernel",err_clCreateKernel);
 err_clCreateKernel:
 err_clBuildProgram:
    ret = clReleaseProgram(program);
    CHECKRET ("clReleaseProgram",err_clCreateProgramWithSource);
 err_clCreateProgramWithSource:
 err_clEnqueueWriteBuffer_B:
 err_clEnqueueWriteBuffer_A:
    ret = clReleaseMemObject(c_mem_obj);
    CHECKRET ("clReleaseMemObject",err_c_mem_obj);
 err_c_mem_obj:
    ret = clReleaseMemObject(b_mem_obj);
    CHECKRET ("clReleaseMemObject",err_b_mem_obj);
 err_b_mem_obj:
    ret = clReleaseMemObject(a_mem_obj);
    CHECKRET ("clReleaseMemObject",err_a_mem_obj);
 err_a_mem_obj:
    ret = clReleaseCommandQueue(command_queue);
    CHECKRET ("clReleaseCommandQueue",err_clCreateCommandQueue);
 err_clCreateCommandQueue:
    ret = clReleaseContext(context);
    CHECKRET ("clReleaseContext",err_clCreateContext);
 err_clCreateContext:
    // XXX deallocate device_id ?
 err_clGetDeviceIDs:
    // XXX deallocate platform_id ?
 err_clGetPlatformIDs:
 err_malloc_source_str:
    fclose( fp );
 err_fopen:
    free(B);
 err_malloc_B:
    free(A);
 err_malloc_A:
    return CHECK_errors; //ok?
}

