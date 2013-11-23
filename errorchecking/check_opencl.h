#ifndef _CHECK_OPENCL_H
#define _CHECK_OPENCL_H


/* Wrapper macros that check for errors on the OpenCL (and malloc)
   calls, and print error messages and increment a local error count
   variable called CHECK_errors if so. The macros all take as last
   argument the name of a label that the code will jump to if an error
   happens. The macros handle error code extraction themselves, you
   don't need to pass &ret. DECLARE_CHECK needs to be put at the start
   of the scope where CHECK_* macros are to be used (this defines
   a fresh CHECK_errors).
 */

#include "opencl_errors.h"

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

/* and, CHECK_* for all OpenCL procedures */

#define CHECKRET_(callstr,exitlabel)					\
    if (CHECK_ret) {							\
	fprintf(stderr,"error: " callstr ": %s at '%s' line %i\n",	\
		clGetErrorString(CHECK_ret), __FILE__, __LINE__);	\
	inc_CHECK_errors();						\
	goto exitlabel;							\
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


#endif /* _CHECK_OPENCL_H */
