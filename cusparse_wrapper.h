#ifndef CUSPARSE_WRAPPER_H
#define CUSPARSE_WRAPPER_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>

#include <cassert>

struct sparse_matrix {
	cusparseMatDescr_t descrA;
	int m;
	int n;
	int nnz;
	double* csrValA;
	int* csrRowPtrA;
	int* csrColIndA;

	enum descriptor_t {
		non_symmetric   = 0,
		symmetric_lower = 1,
		symmetric_upper = 2
	};

    int rows() const {
        return n;   // it's transposed internally
    }
    
    int cols() const {
        return m;   // it's transposed internally
    }
    
    int nonzeros() const {
        return nnz;
    }

	sparse_matrix(descriptor_t descriptor,
	              int rows, int cols, int nonzeros,
	              const double* values, const int* col_ptr, const int* row_ind);

	~sparse_matrix();

};


struct cusparse_wrapper {
    cusparseHandle_t handle;

	enum operation_t {
		non_transpose = 0,
		transpose = 1
	};

    cusparse_wrapper() : handle()
    {
        cusparseStatus_t status  = cusparseCreate(&handle);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    
/*
        // Set to use device pointers    
        status = cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
        assert(status == CUSPARSE_STATUS_SUCCESS);
*/
        // Set to use host pointers    
        status = cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
        assert(status == CUSPARSE_STATUS_SUCCESS);

    }
 
    ~cusparse_wrapper()
    {
        cusparseStatus_t status = cusparseDestroy(handle);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
	
	void matvec(const sparse_matrix* A, const double* x, double* y,
	            bool transpose, double alpha, double beta);

	void matvec_nrhs(const sparse_matrix* A, const double* x, double* y,
	                 bool transpose, double alpha, double beta,
					 int nrhs, int ldx, int ldy);

};

#endif
