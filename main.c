#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h> /* PRIuPTR and uintptr_t */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)


#define DECLARE_CHECK				\
    int CHECK_errors=0;				\
    cl_int CHECK_ret;				\
    void *CHECK_malloc_tmp;

#define inc_CHECK_errors()			\
    if (CHECK_errors < 64) CHECK_errors++;

#define CHECK_malloc(siz,lbl)					\
    CHECK_malloc_tmp=malloc(siz);				\
    if (!CHECK_malloc_tmp) {					\
	inc_CHECK_errors();					\
	fprintf (stderr,"out of memory, line %i\n", __LINE__);	\
	goto lbl;						\
    }

/* and, CHECK_* for all OpenCL procedures?: */

#define CHECKRET_(msg,exitlabel)				\
    if (CHECK_ret) {						\
	fprintf(stderr,"error: " msg " line %i: %i\n",		\
		__LINE__, CHECK_ret);				\
	inc_CHECK_errors();					\
	goto exitlabel;						\
    }

#define CHECK_clGetPlatformIDs(a,b,c,lbl)	\
    CHECK_ret = clGetPlatformIDs(a,b,c);	\
    CHECKRET_("clGetPlatformIDs", lbl);
#define CHECK_clGetDeviceIDs(a,b,c,d,e,lbl) \
    CHECK_ret = clGetDeviceIDs(a,b,c,d,e);  \
    CHECKRET_("clGetDeviceIDs", lbl);
#define CHECK_clCreateContext(a,b,c,d,e,lbl) \
    clCreateContext(a,b,c,d,e,&CHECK_ret);   \
    CHECKRET_("clCreateContext", lbl);
#define CHECK_clCreateCommandQueue(a,b,c,lbl)	\
    clCreateCommandQueue(a,b,c,&CHECK_ret);	\
    CHECKRET_("clCreateCommandQueue", lbl);
#define CHECK_clCreateBuffer(a,b,c,d,lbl)	\
    clCreateBuffer(a,b,c,d,&CHECK_ret);		\
    CHECKRET_("clCreateBuffer", lbl);
#define CHECK_clEnqueueWriteBuffer(a,b,c,d,e,f,g,h,i,lbl) \
    CHECK_ret = clEnqueueWriteBuffer(a,b,c,d,e,f,g,h,i);  \
    CHECKRET_("clEnqueueWriteBuffer", lbl);
#define CHECK_clCreateProgramWithSource(a,b,c,d,lbl)	\
    clCreateProgramWithSource(a,b,c,d,&CHECK_ret);	\
    CHECKRET_("clCreateProgramWithSource", lbl);
#define CHECK_clGetProgramBuildInfo(a,b,c,d,e,f,lbl)	\
    CHECK_ret= clGetProgramBuildInfo(a,b,c,d,e,f);	\
    CHECKRET_("clGetProgramBuildInfo", lbl);

#define CHECK_clCreateKernel(a,b,lbl)  \
    clCreateKernel(a,b,&CHECK_ret);    \
    CHECKRET_("clCreateKernel", lbl);
#define CHECK_clSetKernelArg(a,b,c,d,lbl) \
    CHECK_ret= clSetKernelArg(a,b,c,d);	  \
    CHECKRET_("clSetKernelArg", lbl);
#define CHECK_clEnqueueNDRangeKernel(a,b,c,d,e,f,g,h,i,lbl)	\
    CHECK_ret= clEnqueueNDRangeKernel(a,b,c,d,e,f,g,h,i);	\
    CHECKRET_("clEnqueueNDRangeKernel", lbl);
#define CHECK_clEnqueueReadBuffer(a,b,c,d,e,f,g,h,i,lbl)	\
    CHECK_ret= clEnqueueReadBuffer(a,b,c,d,e,f,g,h,i);		\
    CHECKRET_("clEnqueueReadBuffer", lbl);
#define CHECK_clFlush(a,lbl)	\
    CHECK_ret= clFlush(a);	\
    CHECKRET_("clFlush", lbl);
#define CHECK_clFinish(a,lbl)	\
    CHECK_ret= clFinish(a);	\
    CHECKRET_("clFinish", lbl);
#define CHECK_clReleaseKernel(a,lbl)	\
    CHECK_ret= clReleaseKernel(a);	\
    CHECKRET_("clReleaseKernel", lbl);
#define CHECK_clReleaseProgram(a,lbl)	\
    CHECK_ret= clReleaseProgram(a);	\
    CHECKRET_("clReleaseProgram", lbl);
#define CHECK_clReleaseMemObject(a,lbl)		\
    CHECK_ret= clReleaseMemObject(a);		\
    CHECKRET_("clReleaseMemObject", lbl);
#define CHECK_clReleaseCommandQueue(a,lbl)		\
    CHECK_ret= clReleaseCommandQueue(a);		\
    CHECKRET_("clReleaseCommandQueue", lbl);
#define CHECK_clReleaseContext(a,lbl)		\
    CHECK_ret= clReleaseContext(a);		\
    CHECKRET_("clReleaseContext", lbl);

/* /CHECK_* */



#define val_size 10000
char val[val_size];

int main(void) {
    DECLARE_CHECK;
    
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
	inc_CHECK_errors();
	goto err_fopen;
    }

    char *source_str = CHECK_malloc(MAX_SOURCE_SIZE, err_malloc_source_str);
    size_t source_size= fread( source_str, 1, MAX_SOURCE_SIZE, fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    CHECK_clGetPlatformIDs(1, &platform_id, &ret_num_platforms,
			   err_clGetPlatformIDs);

    printf("platform_id=%p, ret_num_platforms=%i\n",
	   platform_id, ret_num_platforms);

    CHECK_clGetDeviceIDs( platform_id,
			  CL_DEVICE_TYPE_GPU,
			  1,
			  &device_id,
			  &ret_num_devices,
			  err_clGetDeviceIDs);

    // Create an OpenCL context
    cl_context context =
	CHECK_clCreateContext( NULL, 1, &device_id, NULL, NULL, err_clCreateContext);

    // Create a command queue
    cl_command_queue command_queue =
	CHECK_clCreateCommandQueue(context, device_id, 0, err_clCreateCommandQueue);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_READ_ONLY, 
			     LIST_SIZE * sizeof(int), NULL, err_a_mem_obj);
    cl_mem b_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_READ_ONLY,
			     LIST_SIZE * sizeof(int), NULL, err_b_mem_obj);
    cl_mem c_mem_obj =
	CHECK_clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
			     LIST_SIZE * sizeof(int), NULL, err_c_mem_obj);

    // Copy the lists A and B to their respective memory buffers
    CHECK_clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
			       LIST_SIZE * sizeof(int), A, 0, NULL, NULL,
			       err_clEnqueueWriteBuffer_A);
    CHECK_clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
			       LIST_SIZE * sizeof(int), B, 0, NULL, NULL,
			       err_clEnqueueWriteBuffer_B);

    // Create a program from the kernel source
    cl_program program =
	CHECK_clCreateProgramWithSource(context,
					1,
					(const char**)&source_str,
					&source_size,
					err_clCreateProgramWithSource);

    free(source_str); //XXX ok, can we do that while program is still alive ?

    // Build the program
    cl_int ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret) {
	//cl_int ret0=ret; XX print it?
	size_t sizeused;
	CHECK_clGetProgramBuildInfo (program,
				     device_id,
				     CL_PROGRAM_BUILD_LOG,
				     val_size-1, //?
				     &val,
				     &sizeused,
				     err_clGetProgramBuildInfo);

	printf("clBuildProgram error: (sizeused %"PRIuPTR") '%s'\n",
	       (uintptr_t) sizeused, val);
    err_clGetProgramBuildInfo:
	goto err_clBuildProgram;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = CHECK_clCreateKernel(program, "vector_add",
					    err_clCreateKernel);

    // Set the arguments of the kernel
    CHECK_clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj,
			 err_clSetKernelArg);
    CHECK_clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj,
			 err_clSetKernelArg);
    CHECK_clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj,
			 err_clSetKernelArg);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Process in groups of 64
    CHECK_clEnqueueNDRangeKernel(command_queue,
				 kernel,
				 1,
				 NULL, 
				 &global_item_size,
				 &local_item_size,
				 0,
				 NULL,
				 NULL,
				 err_clEnqueueNDRangeKernel);

    // Read the memory buffer C on the device to the local variable C
    int *C = CHECK_malloc(sizeof(int)*LIST_SIZE, err_malloc_C);

    CHECK_clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
			      LIST_SIZE * sizeof(int), C, 0, NULL, NULL,
			      err_clEnqueueReadBuffer);

    // Display the result to the screen
    for(i = 0; i < LIST_SIZE; i++)
	printf("%d + %d = %d\n", A[i], B[i], C[i]);

 err_clEnqueueReadBuffer:
    free(C);
 err_malloc_C:
    CHECK_clFlush(command_queue, err_clFlush);
 err_clFlush:
    CHECK_clFinish(command_queue, err_clEnqueueNDRangeKernel);
 err_clEnqueueNDRangeKernel:
 err_clSetKernelArg:
    CHECK_clReleaseKernel(kernel, err_clCreateKernel);
 err_clCreateKernel:
 err_clBuildProgram:
    CHECK_clReleaseProgram(program, err_clCreateProgramWithSource);
 err_clCreateProgramWithSource:
 err_clEnqueueWriteBuffer_B:
 err_clEnqueueWriteBuffer_A:
    CHECK_clReleaseMemObject(c_mem_obj, err_c_mem_obj);
 err_c_mem_obj:
    CHECK_clReleaseMemObject(b_mem_obj, err_b_mem_obj);
 err_b_mem_obj:
    CHECK_clReleaseMemObject(a_mem_obj, err_a_mem_obj);
 err_a_mem_obj:
    CHECK_clReleaseCommandQueue(command_queue, err_clCreateCommandQueue);
 err_clCreateCommandQueue:
    CHECK_clReleaseContext(context, err_clCreateContext);
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

