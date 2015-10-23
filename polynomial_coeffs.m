function gamma = polynomial_coeffs(degree, prob_dim)
% GAMMA = POLYNOMIAL_COEFFS(DEGREE, PROB_DIM) computes the coefficients of the least- 
% squares polynomial for Jacobi weights with mu = 1/2 and nu = -1/2.
% DEGREE is the degree of the polynomial and GAMMA is a vector of the
% coefficients of the polynomial going from the highest power to the lowest
% power. PROB_DIM represents the dimension in which the problem is being
% performed in such that the polynomial is an improvement on the range of
% eigenvalues (1D - [0,4], 2D - [0,8], 3D - [0,12])

%% Initialize the size of the least-squares residual polynomial
R = zeros(1, degree+2);


%% Loop over each term in the sum for the least-squares residual polynomial
for j = 1: degree+2
    % Generate the k-j^th row of pascals triangle
    bin_coeff = zeros(1, degree - j + 3);
    for i = 1: length(bin_coeff)
        bin_coeff(i) = nchoosek(degree - j + 2, i-1);
    end
    % Compute least-squares residual polynomial
    kappaj = nchoosek(degree+1, j-1) * prod( (degree+1 - 1/2 - (0:j-2))./((0:j-2) + 3/2)) ;
    R(j:end) = R(j:end) +  1/4^(j-1) * (-1)^(j-1) * kappaj * bin_coeff .* ((-4).^-(0:degree-j+2));
end


%% Compute least-squares polynomial sk = (1 - R(lambda))/lambda and scale to be on the domain [0,8] for 2D problem.
gamma = -R(end:-1:2)*(3 + 2*degree)./(prob_dim.^(degree:-1:0));
