function process_nonuniform(filename, poly)
% Processes the matrices for nonuniform meshes to get the eigenvalues and
% eigenvectors for the polynomial preconditioner
% Inputs:
% filename      Name of .mat file which has FVM matrices, nodes and
% elements, named K, M, nodes, elements
% poly          Degree of polynomial you want data for
% Outputs:
% Saves data for use in nonuniform solution

%% Load files
load([filename, '.mat'], 'K', 'M', 'nodes', 'elements')

%% Reorder matrix
p = symrcm(K);
[~, b] = sort(p);
K = K(p,p);
M = M(p,p);
nodes = nodes(p, :);
elements = b(elements);
N = size(K,1);
max_vec = min(floor(3e9/(N*8)), floor(sqrt(N)));
disp(max_vec)

%% Set up vector of x and y coordinates and triangles for plot
xvec = nodes(:,1);
yvec = nodes(:,2);
tri = elements;
tol = pi/ size(elements,1);

%% Compute smallest eigenvectors (either the maximum that we can fit on the
% GPU for that size matrix or the sqrt of the number of nodes, whichever is smaller) and largest
% uses data already computed if it exists
if exist(['nonuniformmatrix', filename,'0.mat'], 'file')==2
    load(['nonuniformmatrix', filename,'0.mat'],'lambda')
    max_vec = length(lambda);
else
    lambda = eigs(K, M, max_vec, 'sm');
end
lambda = [lambda(end:-1:1); eigs(K, M, 1, 'lm')];
lambda = sort(lambda);

%% Determine the total number of vectors that will be needed for shifting up to each of the smallest eigenvalues

kappa = max(lambda)./lambda;
kappa2 = kappa;
a = polynomial_coeffs(poly,max(lambda)/4);

filename2 = ['roots',num2str(poly),'.mat'];

if exist(filename2, 'file') == 2
    load(filename2, 'y');
else
    y=roots(a.*(n+1:-1:1));
    save(filename2, 'y')
end

for i = 1:size(lambda)
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

approx_limit = 3*10^9/(length(xvec)*8);

defl = 0:length(k)-1;
defl = defl';
defl(end) = length(xvec);

%% Determine the number shifted that minimises the storage
total_vec = 2*Niters+defl;
[num_matvec, num_shifted_min] = min(total_vec(1:end-1));
if num_shifted_min == max_vec + 1
    warning('Storage not minimised');
end
mass = full(sqrt(diag(M)));
A = spdiags((1./mass), 0,N, N)*K*spdiags((1./mass), 0,N, N);

%% Compute the Eigenvalues/Eigenvectors and saves them, if the files have already been computed previously, for a smaller polynomial preconditioner, these are used
if exist(['nonuniformmatrix', filename,'0.mat'], 'file')==2
    load(['nonuniformmatrix', filename,'0.mat'], 'A', 'l1', 'lN', 'lambdastar', 'lambda','Q')
    if poly ~=0
        l1 = lambda(num_shifted_min+1);
    end
    Q = Q(:, 1:num_shifted_min);
    lambda = lambda(1:num_shifted_min);
    save(['nonuniformmatrix', filename, num2str(poly),'.mat'], 'A', 'l1', 'lN', 'lambdastar', 'lambda', 'Q', 'num_shifted_min', 'xvec', 'yvec', 'tri', 'tol', 'mass')
else
    l1 = lambda(num_shifted_min+1);
    lN = lambda(end);
    lambdastar = (min(lambda) + max(lambda))/2;
    [Q, lambda ] = eigs(K,M, num_shifted_min, 'sm');
    Q = spdiags(mass, 0, N, N)*Q;
    lambda = diag(lambda);
    [lambda, i] = sort(lambda);
    Q = Q(:,i);
    save(['nonuniformmatrix', filename, num2str(poly),'.mat'], 'A', 'l1', 'lN', 'lambdastar', 'lambda', 'Q', 'num_shifted_min', 'xvec', 'yvec', 'tri', 'tol', 'mass')
end


disp('Processing done')
end