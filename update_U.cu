#include "lanczos.h"

using namespace minlin;

void lanczos_solver::update_U() {

// Update the solution in time
Usol(all,0) =  X(all, 0) + X(all, 1) + update(all,0) + update(all,1);
Usol(all,0) = mul(Usol(all,0), Usol(all,0) >= 0.0);

// Generate g(u) 
Uold(all, 0) = div(Usol(all,0), mass);
Usol(all,1) = mul(mass,Uold(all,0), (1 - Uold(all,0)));

// Scale by dt
scale_in_place(Usol, 1, dt);

// Generate vectors to build subspace from and compute the update for the deflation preconditioner
copy(Usol, 0, U,0);
copy(Usol, 1, U,1);

matmatprod(Q, U, Z, true, false, valvec, 2, 1);
matmatprod(Q, Z, U, false, false, valvec, 0,2); 

diag_mult_on_left_column(exp_lambda, Z, 0);
diag_mult_on_left_column(phi_lambda, Z, 1);
matmatprod(Q, Z, update, false, false, valvec, 2, 1); 

//reset X to zeros
scale(X, 0, valvec, 1); // scale by 0.0
scale(X, 1, valvec, 1);
}
