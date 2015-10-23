function [v1, v2] = nicksmethodcpu(wts1, wts2, A, a, b, n, N, scaling_factor, gamma, gammasreal, gammasimag, etaxi)
% X = NICKSMETHODCPU(WTS1, WTS2, A, Q, LAMBDASTAR, LAMBDA, a, b, n, N, SCALING_FACTOR, GAMMA, GAMMASREAL, GAMMASIMAG, ETAXI)
% performs the operations of the contour integral method of Hale et al.


%% Build subspace
[V1, V2, H1, H2] = lanczos(A, a, b, N, gamma);

%% Form beta*e1
beta1 = zeros(size(H1,1),1);
beta2 = zeros(size(H2,1),1);
beta1(1) = norm(a);
beta2(1) = norm(b);
y1 = zeros(size(H1,1), n);
y2 = zeros(size(H2,1), n);

%% Perform shifted linear system solves
for j = 1:n
    y1(:,j) = scaling_factor*wts1(j)*((etaxi(j)*speye(size(H1)) - H1)\beta1);
    y2(:,j) = scaling_factor*wts2(j)*((etaxi(j)*speye(size(H2)) - H2)\beta2);
end

y1real = real(y1);
y1imag = imag(y1);
y2real = real(y2);
y2imag = imag(y2);

v1 = zeros(size(A,1), 1);
v2 = zeros(size(A,1), 1);


%% Scale problem back up to full size and compute solution
for i = 1:n
    Y1real = V1*y1real(:,i);
    Y1imag = V1*y1imag(:,i);
    Y2real = V2*y2real(:,i);
    Y2imag = V2*y2imag(:,i);
    
    [~, Y1imag] = jacobi_poly_eval(gammasreal(:,i) ,gammasimag(:,i), A, Y1real ,Y1imag);
    [~, Y2imag] = jacobi_poly_eval(gammasreal(:,i) ,gammasimag(:,i), A, Y2real ,Y2imag);
    v1 = v1 + Y1imag;
    v2 = v2 + Y2imag;
end


end