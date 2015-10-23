%close all, clear, clc
%% Forms vectors of deflation preconditioner sizes for finite difference matrices and also forms vector informing whether a problem will fit on GPU or not
script_shifts;

%% Produces Figure 3.2
script_test_bound_large;

%% Produces Figures 4.1 and 4.2
compute_eigenvalues_poly(1024,0, 2.^-(0:4:28), 'dirichlet', 1);
compute_eigenvalues_poly(1024,8, 2.^-(0:4:28), 'dirichlet', 1);

%% Generates table in results section (Example 1) Table 6.1
run_script;

%% Generates results for a nonuniform mesh (Example 2) Figure 6.1
script_test_problem_nonuniform;
plot_nonuniform_solution

%% Generates the results for the monodomain problem (Example 3) Figure 6.2
script_test_problem_monodomain;
plot_monodomain