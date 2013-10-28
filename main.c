#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#define CHECKRET(msg,exitlabel)					\
    if (ret) {							\
	printf("error: " msg " line %i: %i\n", __LINE__, ret);	\
	exitcode = 1;						\
	goto exitlabel;						\
    }


int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int exitcode= 0;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CHECKRET ("clGetPlatformIDs",err10);
    printf("platform_id=%i, ret_num_platforms=%i\n",platform_id, ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
            &device_id, &ret_num_devices);
    CHECKRET ("clGetDeviceIDs",err10);

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    CHECKRET ("clCreateContext",err10);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    CHECKRET ("clCreateCommandQueue",err10);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &ret);
    CHECKRET ("clCreateBuffer",err10);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    CHECKRET ("clCreateBuffer",err10);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            LIST_SIZE * sizeof(int), NULL, &ret);
    CHECKRET ("clCreateBuffer",err10);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    CHECKRET ("clEnqueueWriteBuffer",err10);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
    CHECKRET ("clEnqueueWriteBuffer",err10);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    CHECKRET("clCreateProgramWithSource",err10);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    CHECKRET ("clBuildProgram",err10);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    CHECKRET ("clCreateKernel",err10);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    CHECKRET ("clSetKernelArg",err10);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    CHECKRET ("clSetKernelArg",err10);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    CHECKRET ("clSetKernelArg",err10);
    
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Process in groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
    CHECKRET ("clEnqueueNDRangeKernel",err10);

    // Read the memory buffer C on the device to the local variable C
    int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
    CHECKRET ("clEnqueueReadBuffer",err10);

    // Display the result to the screen
    for(i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
 err10:
    ret = clFlush(command_queue);
    CHECKRET ("clFlush",err9);
 err9:
    ret = clFinish(command_queue);
    CHECKRET ("clFinish",err8);
 err8:
    ret = clReleaseKernel(kernel);
    CHECKRET ("clReleaseKernel",err7);
 err7:
    ret = clReleaseProgram(program);
    CHECKRET ("clReleaseProgram",err6);
 err6:
    ret = clReleaseMemObject(a_mem_obj);
    CHECKRET ("clReleaseMemObject",err5);
 err5:
    ret = clReleaseMemObject(b_mem_obj);
    CHECKRET ("clReleaseMemObject",err4);
 err4:
    ret = clReleaseMemObject(c_mem_obj);
    CHECKRET ("clReleaseMemObject",err3);
 err3:
    ret = clReleaseCommandQueue(command_queue);
    CHECKRET ("clReleaseCommandQueue",err2);
 err2:
    ret = clReleaseContext(context);
    CHECKRET ("clReleaseContext",err1);
 err1:
    free(A);
    free(B);
    free(C);
    return exitcode;
}

