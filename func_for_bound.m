function y = func_for_bound(x, f1, f2, k, l1, lN)
% Y = FUNC_FOR_BOUND(X, F1, F2, K, L1, LN) computes the max abs value of the
% functions from the contour integral method of Hale et al. for computing
% the matrix function vector products of F1 and F2.

%% Compute x plane contour
[sn, cn, dn] = ellipjc(x, -log(k)/pi);
z = sqrt(l1*lN)*(1/k + sn)/(1/k-sn);


%% Compute the contour integral function value
f = f1(z)*cn*dn/ (1/k-sn)^2;
g = f2(z)*cn*dn/(1/k - sn)^2;

F = max(abs(f), abs(g));

%% Compute the value that will maximise 1/(z - lambda)
c = real(z);

if c >= l1 && c <= lN
    d = 1/imag(z);
else
    d = max(abs(1/(z-l1)),abs(1/(z-lN)));
end

y = 2*sqrt(l1*lN)/(k*pi)*F*d;