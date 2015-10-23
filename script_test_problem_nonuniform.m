%% Tests the problem on a nonuniform mesh circle1.mat, circle2.mat, etc. Results example 2

%% Set up matlab to work in the correct directory with path for files used and start diary to save results
diary('problemsolve.txt')

%% This is 1 if the matrices need to be processed to get the eigenvalues
process = 1;

%% Set up vectors to store the times for different polynomial preconditioners in
polydeg = 5;
circlenum = 5;
time_steps = 2^9;
t1 = zeros(polydeg+1,1); t2 = zeros(polydeg+1,1);

filename = ['circle', num2str(circlenum)];
%% Loop over polynomial preconditioners of degree 0:polydeg

for poly = 0:polydeg
    % Processes to save eigenvalue files
    if exist(['nonuniformmatrix', filename, num2str(poly),'.mat'], 'file')~=2
        process_nonuniform(filename, poly)
    end
    % Run computation on GPU
    t1(poly+1)=problem_startup_nonuniform(poly, time_steps, circlenum);
    
    % Run computation on CPU
    t2(poly+1)=problem_startup_CPU_nonuniform(poly, time_steps, circlenum);


    %Test to confirm the GPU and CPU results are the same
    load(['problemsol',num2str(time_steps),'nonuniform', filename,'2.mat'])
    first = output;
    load(['problemsol',num2str(time_steps), '2nonuniformCPU', filename,'.mat'])
    second = output;
    diffres = max(max(abs(first-second)));
    disp(diffres)

end

%% Find the polynomial that had the fastest simulation for both the GPU and CPU, display the times and degrees and the speed up
[a polyg] = min(t1);
[b polyc] = min(t2);
disp('GPU time')
disp(a)
disp('GPU poly Degree')
disp(polyg-1)
disp('CPU time')
disp(b)
disp('CPU poly Degree')
disp(polyc-1)
disp('Speed up')
disp(b/a)

%% Save the diary file
diary('problemsolve.txt')