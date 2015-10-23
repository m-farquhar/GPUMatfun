%% Fractional FitzHugh-Nagumo problem, results example 3

%% Set up matlab to be working with the correct files in the correct directories


diary('problemsolve.txt')

%% Set up the problems to run
% Choose problem size
N = 512;

% Get deflation preconditioner value
filename = ['kvec' num2str(N), 'neumann.mat'];
S = load(filename);
K = S.Kvec;
Kdefl = K(1);

tol = 1/N^2;
poly = 0;
problem = '2d';
time_steps = 2*10^4;
alpha = [2, 1.7, 1.5, 2, 2];
Kalpha = [1e-5, 1e-4, 1e-4, 3e-5, 1e-4];

%% Loop over each of the problems to generate figures
for i = 1:length(alpha)
    problem_startup_monodomain(N,Kdefl,tol, poly, problem, time_steps,alpha(i), Kalpha(i) )
end


%% Run one of the problems on both the CPU and GPU to get timings and speed up
gpustart = tic;
problem_startup_monodomain(N,Kdefl,tol, poly, problem, time_steps,alpha(1), Kalpha(1) )
gpu = toc(gpustart);
disp('GPU = ')
disp(gpu)

cpustart = tic;
problem_startup_CPU_monodomain(N,Kdefl,tol, poly, problem, time_steps,alpha(1), Kalpha(1) )
cpu = toc(cpustart);
disp('CPU = ')
disp(cpu)

disp('Speed up = ')
disp(cpu/gpu)


%% Save outputs to diary
diary('problemsolve.txt')