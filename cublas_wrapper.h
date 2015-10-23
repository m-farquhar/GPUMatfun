#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cassert>
#include <cstdlib>

struct cublas_wrapper {
    cublasHandle_t handle;

    cublas_wrapper() : handle()
    {
        cublasStatus_t status  = cublasCreate(&handle);
        assert(status == CUBLAS_STATUS_SUCCESS);

        // Set to use device pointers    
        status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        assert(status == CUBLAS_STATUS_SUCCESS);

/*
        // Set to use host pointers    
        status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        assert(status == CUBLAS_STATUS_SUCCESS);
*/

    }
 
    ~cublas_wrapper()
    {
        cublasStatus_t status = cublasDestroy(handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }


};

#endif
