function [t2, error] = startup_nicksmethod_GPU(N,Kdefl,tol, poly, problem)
% [T2, ERROR] = STARTUP_NICKSMETHOD_GPU(N, K, TOL, POLY) compute the solution to
% two matrix function vector products, computing the maximum ERROR and the
% time to compute the solution T2. N specifies the size of the problem, K is the
% number of eigenvalues to deflate with the deflation preconditioner, the tolerance
% of the methods is set to TOL, with a polynomial preconditioner of degree
% POLY. On the GPU.
monodomain =0;

if ~monodomain
    if strcmpi(problem, '2d dirichlet')||strcmpi(problem, '2d neumann')
        ode = zeros(N^2,1);
    else
        ode = zeros(N,1);
    end
    aode = 0;
    bode = 0;
    code = 0;
    vrest = 0;
    vamp = 0;
    vth = 0;
    vpeak = 0;
    c1ion = 0;
    c2ion = 0;
end

%% Set up parameters of function
eta = 1;
alpha = 1.5;
dt = 0.1;

%% Generate matrix and compute eigenvalues for deflation preconditioner
if strcmpi(problem, '2d dirichlet')
    [lambda, Q, A] = laplacian([N,N], {'DD', 'DD'}, Kdefl);
    l1 = max(lambda);
    lN = 8*sin(N*pi/(2*(N+1)))^2;
    lambdastar = (min(lambda) + lN)/2;
    mass = ones(N^2,1);
elseif strcmpi(problem, '2d neumann')
    [lambda, Q, A] = laplacian([N,N], {'NN', 'NN'}, Kdefl);
    l1 = max(lambda);
    lN = 8*sin((N-1)*pi/(2*N))^2;
    lambdastar = (min(lambda) + lN)/2;
    mass = ones(N^2,1);
elseif strcmpi(problem, '1d dirichlet')
    [lambda, Q, A] = laplacian(N, {'DD'}, Kdefl);
    l1 = max(lambda);
    lN = 4*sin(N*pi/(2*(N+1)))^2;
    lambdastar = (min(lambda) + lN)/2;
    mass = ones(N,1);
elseif strcmpi(problem, '1d neumann')
    [lambda, Q, A] = laplacian(N, {'NN'}, Kdefl);
    l1 = max(lambda);
    lN = 4*sin((N-1)*pi/(2*N))^2;
    lambdastar = (min(lambda) + lN)/2;
    mass = ones(N,1);
end

%% Generate Vectors for matrix function vector product
if strcmpi(problem, '2d dirichlet') || strcmpi(problem, '2d neumann')
    xh = rand(N^2,1);
    yh = rand(N^2,1);
else
    xh = rand(N,1);
    yh = rand(N,1);
end


%% Set up functions
phi = @(x)(exp(x) - 1)./x;
f1 = @(x) exp(-eta*x.^(alpha/2));
f2 = @(x) phi(-eta*x.^(alpha/2));
exp_lambda = f1(lambda);
phi_lambda = f2(lambda);
if strcmpi(problem, '1d neumann') || strcmpi(problem, '2d neumann')
    phi_lambda(1) = 1;
end

U = [xh,yh];

lambda = lambdastar - lambda;

%% Compute preconditioner updates
Qu = Q'*xh;
Qgu = Q'*yh;
update1 = Q*(diag(exp_lambda)*Qu);
update2 = Q*(diag(phi_lambda)*Qgu);


%% Compute exact solutions
if strcmpi(problem, '2D dirichlet')
    lambdax = 4*sin((1:N)*pi/(2*(N+1))).^2;
    mu1 = f1(lambdax'*ones (1,N) + ones (N,1)*lambdax);
    exact1 = reshape(xh, N, N);
    exact1 = dst(exact1);
    exact1 = dst(exact1');
    exact1 = exact1 .* mu1;
    exact1 = idst(exact1);
    exact1 = idst(exact1');
    exact1 = reshape(exact1, N^2, 1);
    
    mu2 = f2(lambdax'*ones (1,N) + ones (N,1)*lambdax);
    exact2 = reshape(yh, N, N);
    exact2 = dst(exact2);
    exact2 = dst(exact2');
    exact2 = exact2 .* mu2;
    exact2 = idst(exact2);
    exact2 = idst(exact2');
    exact2 = reshape(exact2, N^2, 1);
elseif strcmpi(problem, '2D neumann')
    lambdax = 4*sin((0:N-1)*pi/(2*(N))).^2;
    mu1 = f1(lambdax'*ones (1,N) + ones (N,1)*lambdax);
    exact1 = reshape(xh, N, N);
    exact1 = dct(exact1);
    exact1 = dct(exact1');
    exact1 = exact1 .* mu1;
    exact1 = idct(exact1);
    exact1 = idct(exact1');
    exact1 = reshape(exact1, N^2, 1);
    
    mu2 = f2(lambdax'*ones (1,N) + ones (N,1)*lambdax);
    mu2(1,1) = 1;
    exact2 = reshape(yh, N, N);
    exact2 = dct(exact2);
    exact2 = dct(exact2');
    exact2 = exact2 .* mu2;
    exact2 = idct(exact2);
    exact2 = idct(exact2');
    exact2 = reshape(exact2, N^2, 1);
elseif strcmpi(problem, '1D dirichlet')
    lambdax = 4*sin((1:N)*pi/(2*(N+1))).^2;
    mu1 = f1(lambdax(:));
    exact1 = dst(xh);
    exact1 = exact1 .* mu1;
    exact1 = idst(exact1);
    
    mu2 = f2(lambdax(:));
    exact2 = dst(yh);
    exact2 = exact2 .* mu2;
    exact2 = idst(exact2);
elseif strcmpi(problem, '1D neumann')
    lambdax = 4*sin((0:N-1)*pi/(2*(N))).^2;
    mu1 = f1(lambdax(:));
    exact1 = dct(xh);
    exact1 = exact1 .* mu1;
    exact1 = idct(exact1);
    
    mu2 = f2(lambdax(:));
    mu2(1,1) = 1;
    exact2 = dct(yh);
    exact2 = exact2 .* mu2;
    exact2 = idct(exact2);
end

normx = norm(xh);
normy = norm(yh);

%% Compute region of mapped contour
k = (sqrt(lN/l1) - 1)/ (sqrt(lN/l1) + 1);
[K, Kp] = ellipkkp(-log(k)/pi);                                  % elliptical integrals

% a = (Kp/4:Kp/40:Kp/2);
% ntol = zeros(size(a));
f = @(x) func_for_bound(x, f1, f2, k, l1, lN);


%% Compute the number of points needed in the contour using contour integral bound. Minimising over a
[~, n] = fminbnd(@(a)ceil(log((4*K*bound_M_val(K, Kp, a, f))/tol+1)*K/(pi*a)), 0, Kp/2);

%% Compute weightings for contour integral method
t = 0.5i*Kp - K + (n - 0.5:-1:0)*2*K/n;                         % Midpoint rule points
[sn, cn, dn] = ellipjc(t,-gather(log(k))/pi);                             % Jacobi elliptic functions
xi = sqrt(l1*lN)*(1/k+sn)./(1/k-sn);                            % Quadrature rule
dxidt = cn.*dn./(1/k-sn).^2;                                    % Derivative wrt t
s = -4*K*sqrt(l1*lN)/(k*pi*n);
wts1 = s*f1(xi).*dxidt;                                             % Quadrature weights
wts2 = s*f2(xi).*dxidt;
wts1mat = [real(wts1(:)), imag(wts1(:))];
wts2mat = [real(wts2(:)), imag(wts2(:))];


%% Compute polynomial preconditioner coefficents
if strcmpi(problem, '2d dirichlet') || strcmpi(problem, '2d neumann')
    gamma = polynomial_coeffs(poly, 2);
else
    gamma = polynomial_coeffs(poly, 1);
end

% shifted polynomial coefficients
gammas = zeros(size(gamma, 2),n);
gammas(1,:) = gamma(1);

for i = 2:size(gamma,2)
    gammas(i,:) = gamma(i) + xi.*gammas(i-1,:);
end

gamma_s_real = real(gammas);
gamma_s_imag = imag(gammas);

etaxi = xi.*gammas(end,:);

%% Compute max and min eigenvalue after the polynomial has been applied
y=roots(gamma.*(poly+1:-1:1));

x = polyval([gamma, 0],(y(y(y>l1)<lN)));
z = polyval([gamma, 0], [l1; lN]);
Lmin = min([x;z]);
Lmax = max([x;z]);

kpoly = (sqrt(Lmax/Lmin) - 1)/(sqrt(Lmax/Lmin) + 1);

%% Compute the size of the Krylov subspace
Niters = ceil(log(tol/2)/log(kpoly));
shiftsvec = zeros(Niters*n*4,1);
for i = 1:n
    shiftsvec([2*(i-1)*Niters+1:2:2*i*Niters, 2*(i-1)*Niters+1 + 2*n*Niters:2:2*i*Niters + 2*n*Niters]) = real(etaxi(i));
    shiftsvec([2*(i-1)*Niters+2:2:2*i*Niters, 2*(i-1)*Niters+2 + 2*n*Niters:2:2*i*Niters + 2*n*Niters]) = imag(etaxi(i));
end

%% Send date to GPU
L = lanczosEngine(U,A, Q, lambda, Niters, gamma, gamma_s_real, gamma_s_imag, exp_lambda, phi_lambda, dt, ode, aode, bode, code, vrest, vamp, vth, vpeak, c1ion, c2ion, mass, wts1mat, wts2mat, shiftsvec);


%% Generate Subspace and solve matrix function vector product
tstart = tic;
X = nicksmethodgpu(L);

t2=toc(tstart);

%% Update the solution for deflation preconditioner
%X = L.get_xsol;
expsol = X(:,1) + update1;
phisol = X(:,2) + dt*update2;
%% Compute the error of the system
err1 = norm(expsol - exact1)/normx;
err2 = norm(phisol - dt*exact2)/normy;

error = max(err1, err2);


