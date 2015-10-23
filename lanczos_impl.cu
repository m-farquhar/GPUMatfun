#include "lanczos.h"

using namespace minlin;

    lanczos_solver::lanczos_solver(sparse_matrix::descriptor_t descriptor,
                          int dimension, int subspace_size,
                          int nonzeros, const double* values, const int* col_ptr, const int* row_ind,
                          int num_eigs, const double* Qvalues, const double* lambda_values,
						  const double* Uvalues,
                          int gamma_size, const double* gamma_values,
						  int contour_size, const double* gamma_s_real_values, const double* gamma_s_imag_values,
						  const double* exp_lambda_values, const double* phi_lambda_values, const double timestep,
						  const double* ode_values, const double aode_value, const double bode_value, const double code_value,
						  const double vrest_value, const double vamp_value, const double vth_value,  const double vpeak_value,
						  const double c1ion_value, const double c2ion_value, const double* mass_values, const double* wts1_values, 
						  const double*wts2_values, const double* shifts_values, const double* valvec_val)
    : 
        cublas(), cusparse(),

		valvec(HVector(minlin::Vector<Range>(Range(valvec_val, valvec_val + 3)))),
		
        A(descriptor, dimension, dimension, nonzeros, values, col_ptr, row_ind),
        Q(HMatrix(minlin::Matrix<Range>(Range(Qvalues, Qvalues + num_eigs*dimension), dimension, num_eigs))),
		lambda(HVector(minlin::Vector<Range>(Range(lambda_values, lambda_values + num_eigs)))),
        N(subspace_size),
		gamma_size(gamma_size),
		contour_size(contour_size),
		dt(timestep),
        gamma(HVector(minlin::Vector<Range>(Range(gamma_values, gamma_values + gamma_size)))),
		gamma_s_real(HMatrix(minlin::Matrix<Range>(Range(gamma_s_real_values, gamma_s_real_values + gamma_size*contour_size), gamma_size, contour_size))),
		gamma_s_imag(HMatrix(minlin::Matrix<Range>(Range(gamma_s_imag_values, gamma_s_imag_values + gamma_size*contour_size), gamma_size, contour_size))),
		exp_lambda(HVector(minlin::Vector<Range>(Range(exp_lambda_values, exp_lambda_values + num_eigs)))),
		phi_lambda(HVector(minlin::Vector<Range>(Range(phi_lambda_values, phi_lambda_values + num_eigs)))),
		ode(HVector(minlin::Vector<Range>(Range(ode_values, ode_values + dimension)))),
		aode(aode_value),
		bode(bode_value),
		code(code_value),
		vrest(vrest_value),
		vamp(vamp_value),
		vth(vth_value),
		vpeak(vpeak_value),
		c1ion(c1ion_value),
		c2ion(c2ion_value),
		mass(HVector(minlin::Vector<Range>(Range(mass_values, mass_values + dimension)))),
		wts1(HMatrix(minlin::Matrix<Range>(Range(wts1_values, wts1_values + 2*contour_size), contour_size, 2))),
		wts2(HMatrix(minlin::Matrix<Range>(Range(wts2_values, wts2_values + 2*contour_size), contour_size, 2))),
		shifts(HVector(minlin::Vector<Range>(Range(shifts_values, shifts_values + 4*contour_size*subspace_size)))),
		
        W(dimension, 2),
		B(dimension, 2),
		X(dimension, 2),
        Z(num_eigs, 2),
		
		odepart1(dimension),
		odepart2(dimension),
		source1(dimension),
		source2(dimension),
	
		long_L(4 * subspace_size * contour_size),
		long_D(4 * subspace_size * contour_size),
		long_U(4 * subspace_size * contour_size),
		long_X(4 * subspace_size * contour_size),
	
        V1(dimension, subspace_size),
        V2(dimension, subspace_size),
        aabb(subspace_size+1, 4),
		U(HMatrix(minlin::Matrix<Range>(Range(Uvalues, Uvalues + dimension*2), dimension, 2))),
		update(dimension, 2),
		Uold(dimension, 2),
		Usol(dimension,2),
		Yreal(subspace_size, 2*contour_size),
		Yimag(subspace_size, 2*contour_size),
		
		Y(subspace_size,2)

		
    {
		// Generate initial vectors to build subspace from and compute the update for the deflation preconditioner
			scale_in_place(U, 1, dt);

			copy(U, 0, Usol, 0);
			copy(U, 1, Usol, 1);
			
			matmatprod(Q, U, Z, true, false, valvec, 2, 1); 
			matmatprod(Q, Z, U, false, false, valvec, 0, 2); 
					
			diag_mult_on_left_column(exp_lambda, Z, 0);
			diag_mult_on_left_column(phi_lambda, Z, 1);
			matmatprod(Q, Z, update, false, false, valvec, 2,1); 

    }

	// cuSparse functions
    void lanczos_solver::sparse_mult(const sparse_matrix& A, const DVector& x, DVector& y, bool transpose, double alpha, double beta)
    {
        cusparse.matvec(&A, x.expression().data().get(), y.expression().data().get(), transpose, alpha, beta);
    }
    
    void lanczos_solver::sparse_mult(const sparse_matrix& A, const DMatrix& X, DMatrix& Y, bool transpose, double alpha, double beta)
    {
        cusparse.matvec_nrhs(&A, X.expression().data().get(), Y.expression().data().get(), transpose, alpha, beta, X.cols(), X.rows(), Y.rows());
    }

	// cuBlas functions
    void lanczos_solver::dot(const DMatrix& U, std::size_t Uidx, const DMatrix& V, std::size_t Vidx, double& w)
    {
		int n = U.rows();
		assert(n == V.rows());
        cublasStatus_t status = cublasDdot(cublas.handle, n, U.expression().data().get() + n*Uidx, 1, V.expression().data().get() + n*Vidx, 1, &w);
        assert(status == CUBLAS_STATUS_SUCCESS);
		//cublas.dot(U.expression().data().get(), Uidx, V.expression().data().get(), Vidx, &w, U.rows());
    }
	
	void lanczos_solver::dot(const DMatrix& U, std::size_t Uidx, const DMatrix& V, std::size_t Vidx, DMatrix& W, std::size_t row, std::size_t col)
	{
		int m = W.rows();
		int n = U.rows();
		assert(n == V.rows());
		cublasStatus_t status = cublasDdot(cublas.handle, n, U.expression().data().get() + n*Uidx, 1, V.expression().data().get() + n*Vidx, 1, W.expression().data().get() + m*col + row);
        assert(status == CUBLAS_STATUS_SUCCESS);
		//cublas.dot(U.expression().data().get(), Uidx, V.expression().data().get(), Vidx, W.expression().data().get() + m*col + row, U.rows());
	}

    void lanczos_solver::norm(const DMatrix& U, std::size_t Uidx, double& w)
    {

        cublasStatus_t status = cublasDnrm2(cublas.handle, U.rows(), U.expression().data().get() + U.rows()*Uidx, 1, &w);
        assert(status == CUBLAS_STATUS_SUCCESS);

        //cublas.norm(U.expression().data().get(), Uidx, &w, U.rows());
    }
	
	void lanczos_solver::norm(const DMatrix& U, std::size_t Uidx, DMatrix& w, std::size_t row, std::size_t col)
    {
		int m = w.rows();
		cublasStatus_t status = cublasDnrm2(cublas.handle, U.rows(), U.expression().data().get() + U.rows()*Uidx, 1, w.expression().data().get() + m*col + row);
        assert(status == CUBLAS_STATUS_SUCCESS);
		
        //cublas.norm(U.expression().data().get(), Uidx, w.expression().data().get() + m*col + row, U.rows());
    }

	void lanczos_solver::matvecprod(const DMatrix& A, const DVector& x, DVector& y, bool transA, const DVector& val, int alpha, int beta)
	{
		cublasOperation_t transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
		
	
		cublasStatus_t status =cublasDgemv(cublas.handle, transa, A.rows(), A.cols(),
			val.expression().data().get() + alpha,
			A.expression().data().get(), A.rows(),
			x.expression().data().get(), 1,
			val.expression().data().get() + beta,
			y.expression().data().get(), 1
		);

		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::matvecprod(const DMatrix& A, const DMatrix& x, std::size_t incx, DMatrix& y, std::size_t incy, bool transA, const DVector& val, int alpha, int beta)
	{
		cublasOperation_t transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
		int n = x.rows();
		int m = y.rows();
	
		cublasStatus_t status =cublasDgemv(cublas.handle, transa, A.rows(), A.cols(),
			val.expression().data().get() + alpha,
			A.expression().data().get(), A.rows(),
			x.expression().data().get() + incx*n, 1,
			val.expression().data().get() + beta,
			y.expression().data().get() + incy*m, 1
		);

		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::matmatprod(const DMatrix& A, const DMatrix& B, DMatrix& C, bool transA, bool transB, const DVector& val, int alpha, int beta)
	{

		cublasOperation_t transa = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t transb = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
		int m = transA ? A.cols() : A.rows();
		int n = transB ? B.rows() : B.cols();
		int k1 = transA ? A.rows() : A.cols();
		int k2 = transB ? B.cols() : B.rows();
		
		
		assert(C.rows() == m);
		assert(C.cols() == n);
		assert(k1 == k2);
	
		cublasStatus_t status = cublasDgemm(cublas.handle, transa, transb, m, n, k1,
			val.expression().data().get() + alpha,
			A.expression().data().get(), A.rows(),
			B.expression().data().get(), B.rows(),
			val.expression().data().get() + beta,
			C.expression().data().get(), C.rows()
		);
		assert(status == CUBLAS_STATUS_SUCCESS);

	}

	void lanczos_solver::diag_mult_on_left(const DVector& x, DMatrix& A)
	{
		int m = A.rows();
		int n = A.cols();
	
		cublasStatus_t status = cublasDdgmm(cublas.handle, CUBLAS_SIDE_LEFT, m, n,
			A.expression().data().get(), m,
			x.expression().data().get(), 1,
			A.expression().data().get(), m
		);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::diag_mult_on_left_column(const DVector& x, DMatrix& A, std::size_t j)
	{
		int m = A.rows();
	
		cublasStatus_t status = cublasDdgmm(cublas.handle, CUBLAS_SIDE_LEFT, m, 1,
			A.expression().data().get()+m*j, m,
			x.expression().data().get(), 1,
			A.expression().data().get()+m*j, m
		);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::scale(DMatrix& v, std::size_t vidx, double alpha)
	{
		int n = v.rows();
		cublasStatus_t status = cublasDscal(cublas.handle, n, &alpha, v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::scale(DVector& v, double alpha)
	{
		int n = v.rows();
		cublasStatus_t status = cublasDscal(cublas.handle, n, &alpha, v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::scale(DMatrix& v, std::size_t vidx, DMatrix& alpha, std::size_t row, std::size_t col)
	{
		int n = v.rows();
		int m = alpha.rows();
		cublasStatus_t status = cublasDscal(cublas.handle, n, alpha.expression().data().get() + m*col + row, v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::scale(DVector& v, DMatrix& alpha, std::size_t row, std::size_t col)
	{
		int n = v.rows();
		int m = alpha.rows();
		cublasStatus_t status = cublasDscal(cublas.handle, n, alpha.expression().data().get() + m*col + row, v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::scale(DMatrix& v, std::size_t vidx, DVector& alpha, std::size_t row)
	{
		int n = v.rows();
		cublasStatus_t status = cublasDscal(cublas.handle, n, alpha.expression().data().get() + row, v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::scale(DVector& v, DVector& alpha, std::size_t row)
	{
		int n = v.rows();
		cublasStatus_t status = cublasDscal(cublas.handle, n, alpha.expression().data().get() + row, v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	void lanczos_solver::copy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx)
	{
		int n = u.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDcopy(cublas.handle, n,
			u.expression().data().get() + n*uidx, 1,
			v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::copy(const DVector& u, DVector& v)
	{
		int n = u.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDcopy(cublas.handle, n,
			u.expression().data().get(), 1,
			v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::copyforsolve(const DMatrix& h, const DMatrix& scale1, const DMatrix& scale2, DVector& l, DVector& d, DVector& u, DVector& x)
		{
		int n = h.rows();
		int m = scale1.rows();
		assert(m==scale2.rows());
		assert(4*m*(n-1) == l.rows());
		assert(4*m*(n-1) == d.rows());
		assert(4*m*(n-1) == u.rows());
		assert(4*m*(n-1) == x.rows());
		
			cublasStatus_t status = cublasDscal(cublas.handle, 4*m*(n-1), l.expression().data().get(), d.expression().data().get(), 1);
			assert(status == CUBLAS_STATUS_SUCCESS);
			status = cublasDscal(cublas.handle, 4*m*(n-1), l.expression().data().get(), x.expression().data().get(), 1);
			assert(status == CUBLAS_STATUS_SUCCESS);
	
		
		for (int i=0; i<m; ++i){
			// Copy to D
		status = cublasDcopy(cublas.handle, n-1,
			h.expression().data().get(), 1,
			d.expression().data().get() + 2*i*(n-1), 2);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		status = cublasDcopy(cublas.handle, n-1,
			h.expression().data().get() + n, 1,
			d.expression().data().get() + 2*m*(n-1) + 2*i*(n-1), 2);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
			
			// Copy to L
		status = cublasDcopy(cublas.handle, n-2,
			h.expression().data().get() + 2*n+1, 1,
			l.expression().data().get() + 2*i*(n-1) + 2, 2);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		status = cublasDcopy(cublas.handle, n-2,
			h.expression().data().get() + 3*n+1, 1,
			l.expression().data().get() + 2*m*(n-1) + 2*i*(n-1) + 2, 2);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		
			// Copy to U
		status = cublasDcopy(cublas.handle, n-2,
			h.expression().data().get() + 2*n+1, 1,
			u.expression().data().get() + 2*i*(n-1), 2);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		status = cublasDcopy(cublas.handle, n-2,
			h.expression().data().get() + 3*n+1, 1,
			u.expression().data().get() + 2*m*(n-1) + 2*i*(n-1), 2);
		assert(status == CUBLAS_STATUS_SUCCESS);

	
		
			// Copy from h to first element of each x scale by function weightings
		status = cublasDcopy(cublas.handle, 2,
			h.expression().data().get() + 2*n, 0,
			x.expression().data().get() + 2*i*(n-1), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		status = cublasDcopy(cublas.handle, 2,
			h.expression().data().get() + 3*n, 0,
			x.expression().data().get() + 2*m*(n-1) + 2*i*(n-1), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);	

		
		status = cublasDscal(cublas.handle, 1, scale1.expression().data().get() + i, x.expression().data().get()+(n-1)*2*i, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
		status = cublasDscal(cublas.handle, 1, scale1.expression().data().get() + i + m, x.expression().data().get()+(n-1)*2*i + 1, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
		status = cublasDscal(cublas.handle, 1, scale2.expression().data().get() + i, x.expression().data().get()+(n-1)*2*i + 2*m*(n-1), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
		status = cublasDscal(cublas.handle, 1, scale2.expression().data().get() + i + m, x.expression().data().get()+(n-1)*2*i + 2*m*(n-1) + 1, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		}

	}
	
	
	void lanczos_solver::copyafter(const DVector& x, int m, DMatrix& yr, DMatrix& yi)
		{
		int n = yr.rows();
		assert(n == yi.rows());
		assert(4*n*m == x.rows());
		
		cublasStatus_t status = cublasDcopy(cublas.handle, 2*m*n,
			x.expression().data().get() , 2,
			yr.expression().data().get() , 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
		
		status = cublasDcopy(cublas.handle, 2*m*n,
			x.expression().data().get()+1 , 2,
			yi.expression().data().get() , 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::linsyssolve(const DVector& l, const DVector& d, const DVector& u, int sizem, int numm, DVector& x)
	{
		int n = l.rows();
		assert(n==d.rows());
		assert(n==u.rows());
		assert(n==x.rows());
		assert(n == numm*sizem);
		
		cusparseStatus_t status = cusparseZgtsvStridedBatch(cusparse.handle, sizem,         
                          reinterpret_cast<const cuDoubleComplex*>(l.expression().data().get()), 
                          reinterpret_cast<const cuDoubleComplex*>(d.expression().data().get()),  
                          reinterpret_cast<const cuDoubleComplex*>(u.expression().data().get()),
                          reinterpret_cast<cuDoubleComplex*>(x.expression().data().get()),     
                          numm, sizem);
		assert(status == CUSPARSE_STATUS_SUCCESS);
		
		
	}

	void lanczos_solver::axpy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx, double alpha)
	{
		int n = u.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDaxpy(cublas.handle, n, &alpha,
			u.expression().data().get() + n*uidx, 1,
			v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	void lanczos_solver::axpy(const DVector& u, DVector& v, double alpha)
	{
		int n = u.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDaxpy(cublas.handle, n, &alpha,
			u.expression().data().get(), 1,
			v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::axpy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx, DMatrix& alpha, std::size_t row, std::size_t col)
	{
		int n = u.rows();
		int m = alpha.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDaxpy(cublas.handle, n, alpha.expression().data().get() + m*col + row,
			u.expression().data().get() + n*uidx, 1,
			v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	void lanczos_solver::axpy(const DVector& u, DVector& v, DMatrix& alpha, std::size_t row, std::size_t col)
	{
		int n = u.rows();
		int m = alpha.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDaxpy(cublas.handle, n,  alpha.expression().data().get() + m*col + row,
			u.expression().data().get(), 1,
			v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	void lanczos_solver::axpy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx, DVector& alpha, std::size_t row)
	{
		int n = u.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDaxpy(cublas.handle, n,  alpha.expression().data().get() + row,
			u.expression().data().get() + n*uidx, 1,
			v.expression().data().get() + n*vidx, 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}

	void lanczos_solver::axpy(const DVector& u, DVector& v, DVector& alpha, std::size_t row)
	{
		int n = u.rows();
		assert(n == v.rows());
		
		cublasStatus_t status = cublasDaxpy(cublas.handle, n,  alpha.expression().data().get() + row,
			u.expression().data().get(), 1,
			v.expression().data().get(), 1);
		assert(status == CUBLAS_STATUS_SUCCESS);
	}
	
	// Scale/Invert in place functions
	void lanczos_solver::scale_in_place(DVector& v, double alpha)
	{
		thrust::for_each(v.expression().begin(), v.expression().end(), ScalarMultiply(alpha));
	}
	
	void lanczos_solver::scale_in_place(DMatrix& w, std::size_t col, double alpha)
	{
		int m = w.rows();
		thrust::for_each(w.expression().begin() + m*col, w.expression().begin() + m*(col+1), ScalarMultiply(alpha));
	}
	
	void lanczos_solver::scale_in_place(DMatrix& w, std::size_t row, std::size_t col, double alpha)
	{
		int m = w.rows();
		thrust::for_each(w.expression().begin() + m*col + row, w.expression().begin() + m*col +row +1, ScalarMultiply(alpha));
	}
	
	void lanczos_solver::invert(DMatrix& w, std::size_t row, std::size_t col)
	{
		int m = w.rows();
		thrust::for_each(w.expression().begin() + m*col + row, w.expression().begin() + m*col + row + 1, ScalarInvertMultiply(1.0));
	}
	
	
	// Functions for algorithm	
	void lanczos_solver::get_U(double* dst)
    {
        cudaError_t flag = cudaMemcpy(dst, Usol.expression().data().get(), Usol.rows() * Usol.cols() * sizeof(double), cudaMemcpyDeviceToHost);
        assert(flag == cudaSuccess);
    }
	
	void lanczos_solver::get_xsol(double* dst)
    {
        cudaError_t flag = cudaMemcpy(dst, X.expression().data().get(), X.rows() * X.cols() * sizeof(double), cudaMemcpyDeviceToHost);
        assert(flag == cudaSuccess);
    }
	
