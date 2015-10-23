#include "lanczos.h"

using namespace minlin;

void lanczos_solver::build_subspace() {
// Compute initial vectors in the subspace and copy into V matrices
norm(U,0,aabb,0,2);
norm(U,1,aabb,0,3);
copy(U, 0, V1, 0);
copy(U, 1, V2, 0);	
invert(aabb, 0, 2);
invert(aabb, 0, 3);
scale(V1, 0, aabb,0,2);
scale(V2, 0, aabb,0,3);
invert(aabb, 0, 2);
invert(aabb, 0, 3);


	// Build subspace
	for ( int m = 0; m < N; ++m) {
		copy(V1, m, W, 0);
		copy(V2, m, W, 1);
		
		copy(V1, m, U, 0);
		copy(V2, m, U, 1);
		
		copy(V1, m, B, 0);
		copy(V2, m, B, 1);
		
		scale(U, 0, gamma, 0);
		scale(U, 1, gamma, 0);
		
		// Apply polynomial preconditioner
		for (int i=1; i<gamma_size; ++i) {
			sparse_mult(A, U, W, true, 1.0, 0.0);
			
			copy(B, 0, U, 0);
			copy(B, 1, U, 1);
			
			scale(U, 0, gamma,i);
			scale(U, 1, gamma,i);
			
			axpy(W, 0, U, 0, valvec, 2);
			axpy(W, 1, U, 1, valvec, 2);
			

		}
		
		// Matrix multiplication
		sparse_mult(A, U, W, true, 1.0, 0.0);

		// Orthogonalize against vector before last
		if ( m!= 0 ) {
			scale_in_place(aabb, m, 2, -1.0);
			scale_in_place(aabb, m, 3, -1.0);
			axpy(V1, m-1, W, 0, aabb,m,2);
			axpy(V2, m-1, W, 1, aabb,m,3);
			scale_in_place(aabb, m, 2, -1.0);
			scale_in_place(aabb, m, 3, -1.0);
		}
		
		// Orthogonalize against last vector
		dot(W, 0, V1, m, aabb, m, 0);
		dot(W, 1, V2, m, aabb,m, 1);
		
		
			scale_in_place(aabb, m, 0, -1.0);
			scale_in_place(aabb, m, 1, -1.0);
		axpy(V1, m, W, 0, aabb,m,0);
		axpy(V2, m, W, 1, aabb,m,1);	
			scale_in_place(aabb, m, 0, -1.0);
			scale_in_place(aabb, m, 1, -1.0);
	
	
		norm(W, 0, aabb,m+1, 2);
		norm(W, 1, aabb,m+1, 3);

		// Copy next vector into V matrices
		if(m != (N-1) ) {
			copy(W, 0, V1, m+1);
			copy(W, 1, V2, m+1);
			invert(aabb, m+1, 2);
			invert(aabb, m+1, 3);
			scale(V1, m+1, aabb,m+1,2); 
			scale(V2, m+1, aabb,m+1,3); 
			invert(aabb, m+1, 2);
			invert(aabb, m+1, 3);
		}
	}
	
}
