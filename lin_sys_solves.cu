#include "lanczos.h"

using namespace minlin;

void lanczos_solver::lin_sys_solves() {

// Copy data into the right format for the linear system solves
copyforsolve(aabb, wts1, wts2, long_L, long_D, long_U, long_X);

axpy(shifts, long_D, valvec, 0);

scale(long_X, valvec, 0);

// Perfform the batched linear system solves
linsyssolve(long_L, long_D, long_U, 2*N, 2*contour_size, long_X);

 // Get the solutions for the linear sustem solves
cudaDeviceSynchronize();
copyafter(long_X, contour_size, Yreal, Yimag);

}
