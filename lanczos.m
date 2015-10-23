function [V1, V2, H1, H2] = lanczos(A, a, b, N, gamma)
% [V1, V2, H1, H2] = LANCZOS(A, a, b, N, gamma) computes the matrices V and
% H of the subspace for the two right hand side vectors, a and b. Using the
% polynomial preconditioner with the coefficients gamma.

%% Compute initial vectors in the subspace and copy into V
beta1 = zeros(N+1,1);
beta2 = zeros(N+1,1);
beta1(1) = norm(a);
beta2(1) = norm(b);

V1 = zeros(length(a),N);
V2 = zeros(length(b),N);
alpha1 = zeros(N,1);
alpha2 = zeros(N,1);

V1(:,1) = a/beta1(1);
V2(:,1) = b/beta2(1);

%% Build subspace
for m = 1:N
    w = [V1(:,m) V2(:,m)];
    % Apply polynomial preconditioner
    w = jacobi_poly_eval(gamma, zeros(size(gamma)), A, w, zeros(size(w)));
    
    % Matrix multiplication
    w = A*w;
    
    % Orthogonalize against vector before last
    if m ~= 1
        w = w - [beta1(m)*V1(:, m-1) beta2(m)*V2(:, m-1)];
    end
    
    % Orthogonalize against last vector
    alpha1(m) = dot(w(:,1),V1(:,m));
    alpha2(m) = dot(w(:,2),V2(:,m));
    
    w = w- [alpha1(m)*V1(:,m) alpha2(m)*V2(:,m)];
    
    beta1(m+1) = norm(w(:,1));
    beta2(m+1) = norm(w(:,2));
    
    % Copy next vector into V matrices
    if m<N
        V1(:,m+1) = w(:,1)/beta1(m+1);
        V2(:,m+1) = w(:,2)/beta2(m+1);
    end
    
end

%% Form tridiagonal H matrix
H1 = spdiags([beta1(2:N+1) (alpha1(1:N)) beta1(1:N)], -1:1, N,N);
H2 = spdiags([beta2(2:N+1) (alpha2(1:N)) beta2(1:N)], -1:1, N,N);
end
