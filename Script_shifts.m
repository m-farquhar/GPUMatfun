% Creates files with K sizes and whether problem fits on GPU or not for
% finite differences matrices

num_shifted_min2 = zeros(1,13);
num_matvecs2 = zeros(1,13);


disp('----------------------')
disp('2D Dirichlet')
for N = 2.^(7:10)
    for i = 0 : 12
        [num_shifted_min2(i+1), num_matvecs2(i+1), limit] = compute_eigenvalues_poly(N,i, 1/(N^2), 'dirichlet',0);
    end
    % Determines whether a polynomial size will fit on GPU
    num_matvecs2 = num_matvecs2 - limit;
    num_matvecs = num_matvecs2<-30;
    % saves data of number of eigenvalues needing to be shifted
    Kvec = num_shifted_min2;
    filename = ['kvec',num2str(N),'dirichlet.mat'];
    save(filename, 'Kvec', 'num_matvecs')
end


disp('----------------------')
disp('2D Neumann')
for N = 2.^(7:10)
    for i = 0 : 12
        [num_shifted_min2(i+1), num_matvecs2(i+1), limit] = compute_eigenvalues_poly(N,i, 1/(N^2), 'neumann',0);
    end
    % Determines whether a polynomial size will fit on GPU
    num_matvecs2 = num_matvecs2 - limit;
    num_matvecs = num_matvecs2<-30;
    % saves data of number of eigenvalues needing to be shifted
    Kvec = num_shifted_min2;
    filename = ['kvec',num2str(N),'neumann.mat'];
    save(filename, 'Kvec', 'num_matvecs')
end
disp('----------------------')
