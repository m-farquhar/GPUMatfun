#include "cusparse_wrapper.h"

#include <cassert>

	sparse_matrix::sparse_matrix(sparse_matrix::descriptor_t descriptor,
	                             int rows, int cols, int nonzeros,
	                             const double* values, const int* col_ptr, const int* row_ind)
		: descrA(), m(), n(), nnz(), csrValA(), csrRowPtrA(), csrColIndA()
	{
		// Create descriptor
		assert(cusparseCreateMatDescr(&descrA) == cudaSuccess);
        assert(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO) == cudaSuccess);

		// Set descriptor fields
		switch (descriptor) {
		case non_symmetric:
			assert(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT)    == cudaSuccess);
			assert(cusparseSetMatType    (descrA, CUSPARSE_MATRIX_TYPE_GENERAL)   == cudaSuccess);
			assert(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER)       == cudaSuccess); // doesn't matter which, presumably
			break;
		case symmetric_lower:
			assert(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT)    == cudaSuccess);
			assert(cusparseSetMatType    (descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC) == cudaSuccess);
			assert(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER)       == cudaSuccess); // upper since we're coming with CSC and storing as CSR
			break;
		case symmetric_upper:
			assert(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT)    == cudaSuccess);
			assert(cusparseSetMatType    (descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC) == cudaSuccess);
			assert(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER)       == cudaSuccess); // lower since we're coming with CSC and storing as CSR
			break;
		}

		// Switch rows and cols becuase we're coming with CSC and storing as CSR
		n = rows;
		m = cols;
		nnz = nonzeros;

		// Allocate memory
		assert(cudaMalloc(reinterpret_cast<void**>(&csrValA),      nnz * sizeof(double)) == cudaSuccess);
		assert(cudaMalloc(reinterpret_cast<void**>(&csrRowPtrA), (m+1) * sizeof(int))    == cudaSuccess);
		assert(cudaMalloc(reinterpret_cast<void**>(&csrColIndA),   nnz * sizeof(int))    == cudaSuccess);

		// Copy values
		assert(cudaMemcpy(csrValA,    values,  nnz * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
		assert(cudaMemcpy(csrRowPtrA, col_ptr, (m+1) * sizeof(int),  cudaMemcpyHostToDevice) == cudaSuccess);
		assert(cudaMemcpy(csrColIndA, row_ind, nnz * sizeof(int),    cudaMemcpyHostToDevice) == cudaSuccess);
	}

	sparse_matrix::~sparse_matrix() {
		assert(cudaFree(csrValA)                == cudaSuccess);
		assert(cudaFree(csrRowPtrA)          == cudaSuccess);
		assert(cudaFree(csrColIndA)           == cudaSuccess);
		assert(cusparseDestroyMatDescr(descrA) == cudaSuccess);
	}

	void cusparse_wrapper::matvec(const sparse_matrix* A, const double* x, double* y,
                  	              bool transpose, double alpha, double beta)
	{

        // Flip the tranpose option, since we actually have the tranpose stored
        cusparseOperation_t transA = transpose ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                                 CUSPARSE_OPERATION_TRANSPOSE;

        cusparseStatus_t status =
            cusparseDcsrmv(handle,
                transA, A->m, A->n, A->nnz,
                &alpha,
                A->descrA, A->csrValA, A->csrRowPtrA, A->csrColIndA,
                x,
                &beta,
                y
            );
        assert(status == CUSPARSE_STATUS_SUCCESS);

	}

	void cusparse_wrapper::matvec_nrhs(const sparse_matrix* A, const double* x, double* y,
                  	                   bool transpose, double alpha, double beta,
									   int nrhs, int ldx, int ldy)
	{

        // Flip the tranpose option, since we actually have the tranpose stored
        cusparseOperation_t transA = transpose ? CUSPARSE_OPERATION_NON_TRANSPOSE :
                                                 CUSPARSE_OPERATION_TRANSPOSE;

        cusparseStatus_t status =
            cusparseDcsrmm(handle,
                transA, A->m, nrhs, A->n, A->nnz,
                &alpha,
                A->descrA, A->csrValA, A->csrRowPtrA, A->csrColIndA,
                x, ldx,
                &beta,
                y, ldy
            );
        assert(status == CUSPARSE_STATUS_SUCCESS);

	}
