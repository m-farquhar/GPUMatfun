function M = bound_M_val(K, Kp, a, f)
% M = BOUND_M_VAL(K, KP, A, F) computes the maximum value of the function
% on the domain -K < Re(x) < K, Kp/2 - a < Im(x) < Kp/2 + a. F is an
% anonymous function of x.

% Tests corners of rectangle
val1 = f(-K + 1i*(Kp/2-a));
val2 = f(K+ 1i*(Kp/2-a));

val3 = f(-K + 1i*(Kp/2+a));
val4 = f(K+ 1i*(Kp/2+a));

options = optimset('Display', 'off');

% Tests top and bottom boundary of domain for maximum value
[~, fval] = fminsearch(@(x)-abs(f(x + 1i*(Kp/2+a))), K, options);
[~, fval2] = fminsearch(@(x)-abs(f(x + 1i*(Kp/2-a))), K, options);

M = max(abs([val1, val2, val3, val4, fval, fval2]));