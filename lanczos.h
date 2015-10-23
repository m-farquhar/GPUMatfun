#ifndef LANCZOS_H
#define LANCZOS_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>

#include <cassert>

#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>

#include "cublas_wrapper.h"
#include "cusparse_wrapper.h"

struct lanczos_solver {

    typedef minlin::threx::DeviceVector<double> DVector;
    typedef minlin::threx::DeviceMatrix<double> DMatrix;
    typedef minlin::threx::HostVector<double> HVector;
    typedef minlin::threx::HostMatrix<double> HMatrix;
    typedef minlin::threx::Range<const double*> Range;

    cublas_wrapper cublas;
    cusparse_wrapper cusparse;
    
	DVector valvec;
	
    sparse_matrix A;
    
	DMatrix Q;
	DVector lambda;
    int N;
	int gamma_size;
	int contour_size;
	double dt;
    DVector gamma;
	DMatrix gamma_s_real;
	DMatrix gamma_s_imag;
	DVector mass;
	DMatrix wts1;
	DMatrix wts2;
	
	DVector shifts;

    DMatrix W;
	DMatrix B;
	DMatrix Z;
    DMatrix V1;
    DMatrix V2;
    DMatrix aabb;

    DMatrix U;
	DMatrix Yreal;
	DMatrix Yimag;
	DMatrix Y;
		
	DVector exp_lambda;
	DVector phi_lambda;
	DMatrix update;
	DMatrix Usol;
	DMatrix Uold;
	DMatrix X;
		
	DVector ode;
	double aode;
	double bode;
	double code;
	double vamp;
	double vrest;
	double vth;
	double vpeak;
	double c1ion;
	double c2ion;
	
	DVector odepart1;
	DVector odepart2;
	DVector source1;
	DVector source2;

	DVector long_L;	
	DVector long_D;	
	DVector long_U;	
	DVector long_X;	
	
	int first;

    lanczos_solver(sparse_matrix::descriptor_t descriptor,
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
						  const double* wts2_values, const double* shifts_values, const double* valvec_val);
// Functions for Algorithm
    void build_subspace();
	void get_U(double*);
	void get_xsol(double*);
	void apply_V();
	void update_U();
	void lin_sys_solves();
	void update_U_monodomain();

// cuSparse functions
    void sparse_mult(const sparse_matrix& A, const DVector& x, DVector& y, bool transpose, double alpha, double beta);
    void sparse_mult(const sparse_matrix& A, const DMatrix& X, DMatrix& Y, bool transpose, double alpha, double beta);

// cuBlas functions
    void dot(const DMatrix& u, std::size_t uidx, const DMatrix& v, std::size_t vidx, double& w);	
    void dot(const DMatrix& u, std::size_t uidx, const DMatrix& v, std::size_t vidx, DMatrix& w, std::size_t row, std::size_t col);

    void norm(const DMatrix& U, std::size_t uidx, double& w);
	void norm(const DMatrix& U, std::size_t uidx, DMatrix& w, std::size_t row, std::size_t col);

	void scale(DMatrix& v, std::size_t vidx, double alpha);
	void scale(DVector& v, double alpha);
	void scale(DMatrix& v, std::size_t vidx, DMatrix& alpha, std::size_t row, std::size_t col);
	void scale(DVector& v, DMatrix& alpha, std::size_t row, std::size_t col);
	void scale(DMatrix& v, std::size_t vidx, DVector& alpha, std::size_t row);
	void scale(DVector& v, DVector& alpha, std::size_t row);
	
	void axpy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx, double alpha);
	void axpy(const DVector& u, DVector& v, double alpha);	
	void axpy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx, DMatrix& alpha, std::size_t row, std::size_t col);
	void axpy(const DVector& u, DVector& v, DMatrix& alpha, std::size_t row, std::size_t col);	
	void axpy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx, DVector& alpha, std::size_t row);
	void axpy(const DVector& u, DVector& v, DVector& alpha, std::size_t row);
	
	void copy(const DMatrix& u, std::size_t uidx, DMatrix& v, std::size_t vidx);
	void copy(const DVector& u, DVector& v);
		
	void copyforsolve(const DMatrix& h, const DMatrix& scale1, const DMatrix& scale2, DVector& l, DVector& d, DVector& u, DVector& x);
	void copyafter(const DVector& x, int m, DMatrix& yr, DMatrix& yi);
	
	void linsyssolve(const DVector& l, const DVector& d, const DVector& u, int sizem, int numm, DVector& x);

	void matvecprod(const DMatrix& A, const DVector& x, DVector& y, bool transA, const DVector& val, int alpha, int beta);
	void matvecprod(const DMatrix& A, const DMatrix& x, std::size_t incx, DMatrix& y, std::size_t incy, bool transA, const DVector& val, int alpha, int beta);

	void matmatprod(const DMatrix& A, const DMatrix& B, DMatrix& C, bool transA, bool transB, const DVector& val, int alpha, int beta);

	void diag_mult_on_left(const DVector& d, DMatrix& A);	
	void diag_mult_on_left_column(const DVector& d, DMatrix& A, std::size_t vidx);
	
	
	// Scale/Invert in place functions
struct ScalarMultiply {
  ScalarMultiply(double d) : d(d) {}
  double d;
  __host__ __device__ void operator()(double& x) {
    x *= d;
  }
};

struct ScalarInvertMultiply {
  ScalarInvertMultiply(double d) : d(d) {}
  double d;
  __host__ __device__ void operator()(double& x) {
    x = d / x;
  }
};

	void invert(DMatrix& w, std::size_t row, std::size_t col); // replace w(row,col) with 1.0/w(row, col)
	
	void scale_in_place(DVector& v, double alpha);
	void scale_in_place(DMatrix& w, std::size_t col, double alpha);
	void scale_in_place(DMatrix& w, std::size_t row, std::size_t col, double alpha);

};

#endif
