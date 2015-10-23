#include "lanczos.h"

using namespace minlin;

void lanczos_solver::apply_V() {

// Compute solution for exp function
for (int i=0; i<contour_size; ++i) {
	copy(Yreal, i, Y, 0);
	copy(Yimag, i, Y, 1);
// Scale the vectors up to full problem size
	matvecprod(V1, Y, 0, U, 0, false, valvec, 2, 1);
	matvecprod(V1, Y, 1, U, 1, false, valvec, 2, 1);
	
	// Apply polynomial preconditioner
		copy(U, 0, B, 0);
		copy(U, 1, B, 1);
		
		scale(U, 0, gamma_s_real,0,i);
		scale(U, 1, gamma_s_real,0,i);
		
			scale_in_place(gamma_s_imag,0,i,-1.0);
		axpy(B, 1, U, 0, gamma_s_imag,0,i);
		scale_in_place(gamma_s_imag,0,i,-1.0);
		axpy(B, 0, U, 1, gamma_s_imag,0,i);

		
		for (int j=1; j<gamma_size; ++j) {	
			sparse_mult(A, U, W, true, 1.0, 0.0);
			
			copy(B, 0, U, 0);
			copy(B, 1, U, 1);
			
			scale(U, 0, gamma_s_real,j,i);
			scale(U, 1, gamma_s_real,j,i);
			scale_in_place(gamma_s_imag,j,i,-1.0);
			axpy(B, 1, U, 0, gamma_s_imag,j,i); 
			scale_in_place(gamma_s_imag,j,i,-1.0);
			axpy(B, 0, U, 1, gamma_s_imag,j,i);
			
			axpy(W, 0, U, 0, valvec, 2);//1.0);
			axpy(W, 1, U, 1, valvec, 2);// 1.0);
		}
// Compute the sum of each of the vectors
		axpy(U, 1, X, 0, valvec, 2);//1.0);

}
// Compute solution for phi function
for (int i=contour_size; i< 2.0*contour_size; ++i) {
	copy(Yreal, i, Y, 0);
	copy(Yimag, i, Y, 1);
// Scale the vectors up to full problem size
	matvecprod(V2, Y, 0, U, 0, false, valvec, 2, 1);
	matvecprod(V2, Y, 1, U, 1, false, valvec, 2, 1);
	
// Apply polynomial preconditioner
		copy(U, 0, B, 0);
		copy(U, 1, B, 1);
		
		scale(U, 0, gamma_s_real,0,i-contour_size);
		scale(U, 1, gamma_s_real,0,i-contour_size);
		
		scale_in_place(gamma_s_imag,0,i-contour_size,-1.0);
		axpy(B, 1, U, 0, gamma_s_imag,0,i-contour_size);
		scale_in_place(gamma_s_imag,0,i-contour_size,-1.0);
		axpy(B, 0, U, 1, gamma_s_imag,0,i-contour_size);
		
		for (int j=1; j<gamma_size; ++j) {	
			sparse_mult(A, U, W, true, 1.0, 0.0);

			
			copy(B, 0, U, 0);
			copy(B, 1, U, 1);
			

			scale(U, 0, gamma_s_real,j,i-contour_size);
			scale(U, 1, gamma_s_real,j,i-contour_size);

			scale_in_place(gamma_s_imag,j,i-contour_size, -1.0);
			axpy(B, 1, U, 0, gamma_s_imag,j,i-contour_size); 
			scale_in_place(gamma_s_imag,j,i-contour_size, -1.0);
			axpy(B, 0, U, 1, gamma_s_imag,j,i-contour_size);

			

			axpy(W, 0, U, 0, valvec, 2);//1.0);
			axpy(W, 1, U, 1, valvec, 2);//1.0);

		}
		// Compute the sum of each of the vectors
		axpy(U, 1, X, 1, valvec, 2);//1.0);
}

cudaDeviceSynchronize();
}
