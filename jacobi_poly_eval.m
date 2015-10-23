function [xreal, ximag] = jacobi_poly_eval(gammareal, gammaimag, A, breal, bimag)
% [XREAL, XIMAG] = JACOBI_POLY_EVAL(GAMMAREAL, GAMMAIMAG, A, BREAL, BIMAG)
% Computes p(A)b using Horner's method where GAMMA is the coefficients of
% the polynomial split into real and imaginary parts and b is split into
% real and imaginary parts.


n = length(gammareal);
[~, z] = size(breal);


xreal = breal*diag(gammareal(1)) - bimag*diag(gammaimag(1));
ximag =  breal*diag(gammaimag(1)) +  bimag*diag(gammareal(1));

for i = 2:n
    Ax = A*[xreal, ximag];
    xreal = Ax(:,1:z) + breal*(gammareal(i))-bimag*(gammaimag(i));
    ximag = Ax(:,z+1:end) + breal*(gammaimag(i)) + bimag*(gammareal(i));
end