%% Generates figure 3.2

%% Set up problem
N = 1024; %size of problem number of nodes in 1 dimension
poly = 8;
filename = ['kvec' num2str(N), 'dirichlet.mat'];
S = load(filename);
K = S.Kvec;
Kdefl = K(poly+1);
n = 1:60; % size of contour
eta = 1;
alpha = 1.5;
tol = 1/N^2;


%% Generate matrix and eigenpairs
[lambda, Q, A] = laplacian([N,N], {'DD', 'DD'}, Kdefl);
l1 = max(lambda);
lN = 8*sin(N*pi/(2*(N+1)))^2;
lambdastar = (min(lambda) + lN)/2;

%% Generate random vectors
xh = rand(N^2,1);
yh = rand(N^2,1);



phi = @(x)(exp(x) - 1)./x;

exp_lambda = exp(-eta*(lambda).^(alpha/2));
phi_lambda = phi(-eta*(lambda).^(alpha/2));
dt = 0.1;


phi = @(x) (exp(x) - 1)./(x);
f1 = @(x) exp(-eta*x.^(alpha/2));
f2 = @(x) phi(-eta*x.^(alpha/2));
phim = @(x) x\(expm(x) - eye(size(x)));

adjust1 = Q*(diag(exp_lambda) * (Q'*xh));
adjust2 = Q*(diag(phi_lambda) * (Q'*yh));

%% Get exact solutions
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


%% Set up error vectors
error1 = zeros(size(n));
error2 = zeros(size(n));

%% Generate bound vectors
k = (sqrt(lN/l1) - 1)/ (sqrt(lN/l1) + 1);
[K, Kp] = ellipkkp(-log(k)/pi);
a = (Kp/20:Kp/20:Kp/2);
f = @(x) func_for_bound(x, f1, f2, k, l1, lN);

normx = norm(xh);
normy = norm(yh);
bound = zeros(length(n), length(a));

for i = 1:length(a)
    M = bound_M_val(K, Kp, a(i), f);
    bound(:,i) = 8*M*K./(exp(pi*a(i)/K*n) - 1);
    bound(:,i) = bound(:,i) ;
    if M > 1e11
        disp(i)
        break
    end
end

bound = bound(:, 1:i-1);
a = a(1:i-1);

%% create Legend
clear legstr
legstr{length(a)+2}=0;
legstr{1} = 'Exact Error';
for i = 1:length(a)
    legstr{i+1} = ['a = ', num2str(a(i)/Kp), '*Kp'];
end

%% Test problem for all n
gamma = polynomial_coeffs(poly, 2);
norm1 = norm(exact1);
norm2 = norm(exact2);
U = [xh, yh];
lambda = lambda - lambdastar;

y=roots(gamma.*(poly+1:-1:1));

x = polyval([gamma, 0],(y(y(y>l1)<lN)));
z = polyval([gamma, 0], [l1; lN]);
Lmin = min([x;z]);
Lmax = max([x;z]);
kpoly = (sqrt(Lmax/Lmin) - 1)/(sqrt(Lmax/Lmin)+1);                          % constant used in the conformal mapping
Niters = ceil(log((tol)/2)/log(kpoly));

for i = n;
    %contour
    t = 0.5i*Kp - K + (i - 0.5:-1:0)*2*K/i;                         % Midpoint rule points
    [sn, cn, dn] = ellipjc(t,-gather(log(k))/pi);                             % Jacobi elliptic functions
    xi = sqrt(l1*lN)*(1/k+sn)./(1/k-sn);                            % Quadrature rule
    dxidt = cn.*dn./(1/k-sn).^2;                                    % Derivative wrt t
    s = -4*K*sqrt(l1*lN)/(k*pi*i);
    wts1 = s*f1(xi).*dxidt;                                             % Quadrature weights
    wts2 = s*f2(xi).*dxidt;
    wts1mat = [real(wts1(:)), imag(wts1(:))];
    wts2mat = [real(wts2(:)), imag(wts2(:))];
    
    
    gammas = zeros(size(gamma,2),i);
    gammas(1,:) = gamma(1);
    
    for j = 2:size(gamma,2)
        gammas(j,:) = gamma(j) + xi.*gammas(j-1,:);
    end
    
    gammasreal = real(gammas);
    gammasimag = imag(gammas);
    
    etaxi = xi.*gammas(end, :);
    
    ode = zeros(N^2,1);
    aode = 0;
    bode = 0;
    code = 0;
    vrest = 0;
    vamp = 0;
    vth = 0;
    vpeak = 0;
    c1ion = 0;
    c2ion = 0;
    clear L
    
    shiftsvec = zeros(Niters*i*4,1);
    for jj = 1:i
        shiftsvec([2*(jj-1)*Niters+1:2:2*jj*Niters, 2*(jj-1)*Niters+1 + 2*i*Niters:2:2*jj*Niters + 2*i*Niters]) = real(etaxi(jj));
        shiftsvec([2*(jj-1)*Niters+2:2:2*jj*Niters, 2*(jj-1)*Niters+2 + 2*i*Niters:2:2*jj*Niters + 2*i*Niters]) = imag(etaxi(jj));
    end
    mass = ones(N^2,1);
    % Shift everything to GPU and evaluate matrix function vector product
    L = lanczosEngine(U,A, Q, lambda, Niters, gamma, gammasreal, gammasimag, exp_lambda, phi_lambda, dt, ode, aode, bode, code, vrest, vamp, vth, vpeak, c1ion, c2ion, mass, wts1mat, wts2mat, shiftsvec);
    X = nicksmethodgpu(L);
    v1 = X(:,1);
    v2 = X(:,2);
    
    error1(i) = norm(v1 + adjust1 - exact1)/normx;
    error2(i) = norm(v2 + adjust2- exact2)/normy;
    disp(i)
    
end
save('bounderrs.mat', 'error1', 'error2', 'bound')


%% Plot figure
colmat = parula(length(a)); %change this if not 2014b or later
figure
set(gcf,'DefaultAxesColorOrder',colmat)
semilogy(n, error1, 'b.', 'linewidth',2)
hold all
semilogy(n, bound, 'linewidth',2)
legend(legstr)
ylabel('error', 'fontsize', 22)
xlabel('contour size', 'fontsize', 22)
set(gca, 'fontsize', 22)
saveas(gcf, 'bound.eps','psc2')



