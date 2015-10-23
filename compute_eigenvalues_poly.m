function [num_shifted_min, num_matvec, approx_limit] = compute_eigenvalues_poly(N,n, tol, type, plotyn)
%% COMPUTE_EIGENVALUES_POLY computes the number of eigenvalues needing to be 
% shifted to minimise storage for finite differences matrix 
% INPUTS:
% N:        Matrix size, matrix dimension of N^2, 2D finite differences
% n:        Degree of Jacobi least squares polynomial
% tol:      Vector of tolerances to test to find the minimum (range of
%           tolerances produces figure 4.1 and 4.2
% type:     boundary condition type 'dirichlet' or 'neumann'
% plotyn:   determines whether of not it generates the plot for figures 4.1
%           and 4.2
% OUTPUTS:
% num_shifted_min:      The number of eigenvalues to shift to minimise
%                       memory for each tolerance
% num_matvec:           The number of vectors required for this tolerance
%                       (the eigenvectors and the 2 subspaces
% approx_limit:         The maximum number of vectors of this size to fit
%                       on GPU
%

%% Creates initial eigenvalues dependent on boundary conditions
if strcmpi('dirichlet', type)
    lambdax = 4*sin((1:N)*pi/(2*(N+1))).^2;
elseif strcmpi('neumann', type)
    lambdax = 4*sin((0:N-1)*pi/(2*N)).^2;
end

%% Creates vector of all eigenvalues
lambda = lambdax'*ones (1,N) + ones (N,1)*lambdax;
lambda = lambda(:);

lambda = sort(lambda);


%% Determines the number of vectors needed for shifting up to each eigenvalue
kappa = max(lambda)./lambda;
kappa2 = kappa;
a = polynomial_coeffs(n,2);

filename2 = ['roots',num2str(n),'.mat'];

if exist(filename2, 'file') == 2
    load(filename2, 'y');
else
    y=roots(a.*(n+1:-1:1));
    save(filename2, 'y')
end

for i = 1:N
	d = [lambda(i), lambda(end)];
	x = polyval([a, 0],(y(y(y>d(1))<d(2))));
	z = polyval([a, 0], [d(1); d(2)]);
	l1 = min([x;z]);
	lN = max([x;z]);
    kappa2(i) = lN/l1;
end

k = (sqrt(kappa) - 1)./(sqrt(kappa)+1);

k2 = (sqrt(kappa2) - 1)./(sqrt(kappa2)+1);


Niters = zeros(length(k), length(tol));

for i = 1:length(tol)
    Niters(:,i) = ceil(log(tol(i)/2)./log(k2));
end

approx_limit = 3*10^9/(N^2*8);

defl = 0:N^2-1;
defl = defl';

%% Determines number of eigenvalues shifted that minimises memory
total_vec = 2*Niters+repmat(defl, 1,length(tol));
[num_matvec, num_shifted_min] = min(total_vec);

%% plots figures
if plotyn
    figure
    colmat = parula(length(tol));
    set(gcf,'DefaultAxesColorOrder',colmat)
    plot(0:N-1, total_vec(1:N,:), 'linewidth', 3)
    hold on
    plot(0:N-1, approx_limit*ones(1,N), 'k--', 'linewidth',2)
    % change these depending on figure to capture detail
    xlim([0,N])
    ylim([0, num_matvec(1) + num_matvec(end)])
    % change file name as well
    if n ~=0
        saveas(gcf, ['vectors2D', num2str(N), 'jacobi2.eps'], 'psc2')
        xlim([0,400])
        ylim([0,400])
    else
        saveas(gcf, ['vectors2D', num2str(N), '2.eps'], 'psc2')
        xlim([0,N])
        ylim([0, num_matvec(1) + num_matvec(end)])
    end
end
