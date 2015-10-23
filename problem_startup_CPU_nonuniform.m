function t1 = problem_startup_CPU_nonuniform(poly,  time_steps, circlenum)
% [T2, ERROR] = STARTUP_NICKSMETHOD_GPU(N, K, TOL, POLY) compute the solution to
% two matrix function vector products, computing the maximum ERROR and the
% time to compute the solution T2. N specifies the size of the problem, K is the
% number of eigenvalues to deflate with the deflation preconditioner, the tolerance
% of the methods is set to TOL, with a polynomial preconditioner of degree
% POLY. On the GPU.


%% Set up parameters of function
Kalpha = 5e-4;
alpha = 1.5;
t_end = 24;
dt = t_end/time_steps;
num_plots = 8;
every = time_steps/num_plots;
eta = Kalpha*dt;

%% Generate matrix and compute eigenvalues for deflation preconditioner
filename = ['circle', num2str(circlenum)];
load( ['nonuniformmatrix', filename, num2str(poly),'.mat'])

%% Generate Vectors for matrix function vector product
tmp1 = sqrt((xvec - 0.1).^2 + (yvec + 0.1).^2);
xh = zeros(length(xvec), 1);
ode = zeros(length(xvec), 1);
xh(tmp1 < 1/sqrt(2900)) = 0.01;

if max(xh) == 0
    [~, loc] = min((xvec - 0.1).^2 + (yvec + 0.1).^2);
    xh(loc) = 0.01;
end

yh = xh.*(1-xh);

xh = mass.*xh;
yh = mass.*yh;

%% Set up functions
phi = @(x)(exp(x) - 1)./x;
f1 = @(x) exp(-eta*x.^(alpha/2));
f2 = @(x) phi(-eta*x.^(alpha/2));
exp_lambda = f1(lambda);
phi_lambda = f2(lambda);
phi_lambda(1) = 1;

U = [xh, yh];
lambda = lambdastar - lambda;


%% Compute region of mapped contour
k = (sqrt(lN/l1) - 1)/ (sqrt(lN/l1) + 1);
[K, Kp] = ellipkkp(-log(k)/pi);                                  % elliptical integrals

f = @(x) func_for_bound(x, f1, f2, k, l1, lN);


%% Compute the number of points needed in the contour using contour integral bound. Minimise over a.
[~, n] = fminbnd(@(a)ceil(log((4*K*bound_M_val(K, Kp, a, f))/tol+1)*K/(pi*a)), 0, Kp/2);


%% Compute weightings for contour integral method
t = 0.5i*Kp - K + (n - 0.5:-1:0)*2*K/n;                         % Midpoint rule points
[sn, cn, dn] = ellipjc(t,-gather(log(k))/pi);                             % Jacobi elliptic functions
xi = sqrt(l1*lN)*(1/k+sn)./(1/k-sn);                            % Quadrature rule
dxidt = cn.*dn./(1/k-sn).^2;                                    % Derivative wrt t
wts1 = f1(xi).*dxidt;                                             % Quadrature weights
wts2 = f2(xi).*dxidt;
s = -4*K*sqrt(l1*lN)/(k*pi*n);

%% Compute polynomial preconditioner coefficents

gamma = polynomial_coeffs(poly, lN/4);


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


u = xh;
gu = yh;
tic
%% Generate Subspace and solve matrix function vector product
solmat = zeros(size(A,1),num_plots+1);
solmat(:,1) = U(:,1)./mass;
t = 0;
for i = 1:time_steps
    Qu = Q'*u;
    Qgu = Q'*gu;
    xbar = u - Q*Qu;
    ybar = gu - Q*Qgu;
    update1 = Q*(diag(exp_lambda)*Qu);
    update2 = Q*(diag(phi_lambda)*Qgu);
    [xsol1exp, xsol1phi] = nicksmethodcpu(wts1, wts2, A, xbar, ybar, n, Niters, s, gamma, gamma_s_real, gamma_s_imag, etaxi);
    u = xsol1exp + dt*xsol1phi + update1 + dt*update2;
    ubar = u./mass;

    gu = mass.*(ubar).*(1-(ubar));
    t = t + dt;
    if mod(i,every)==0
        U = [u, gu];
        solmat(:, i/every +1) = U(:,1)./mass;
        disp(t)
    end
end
U = [u, gu];
output = U(:,1)./mass;
t1 = toc;
save(['problemsol',num2str(time_steps),'2nonuniformCPU', filename,'.mat'], 'output', 'solmat')


