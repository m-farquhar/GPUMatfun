%% Generates table in results example 1
%% Choose which problems to run
dirrun = 1;
neurun = 0;
dirrun1d = 0;
neurun1d = 0;
savedir = '';
polymax = 12; % max polynomial degree to test to

if dirrun
    diary([savedir,'DirichletRes.txt'])
    problem = '2D Dirichlet';
    disp(problem)
    N = 2.^(7:10);
    tol = 1./N.^2;
    
    
    K = zeros(length(N), polymax+1);
    tav = inf*ones(length(N), polymax+1);
    errorav = zeros(length(N), polymax+1);
    tbv = zeros(length(N), polymax+1);
    Nitersv = zeros(length(N), polymax+1);
    errorbv = zeros(length(N), polymax+1);
    ncontv = zeros(length(N), polymax+1);
    
    for i = 1:length(N)
        filename = [savedir,'kvec' num2str(N(i)), 'dirichlet.mat'];
        S = load(filename);
        K(i, :) = S.Kvec(1:polymax+1);
        yncheck = S.num_matvecs;
        for j = 0:polymax
            disp([N(i), K(i, j+1), j])
            if yncheck(j+1)
                [tav(i,j+1), errorav(i, j+1)] = startup_nicksmethod_GPU(N(i), K(i, j+1), tol(i), j, problem);
                disp(tav(i, j+1))
                disp(errorav(i, j+1))
            end
            [tbv(i,j+1), errorbv(i, j+1), Nitersv(i,j+1), ncontv(i,j+1)] = startup_nicksmethod_CPU(N(i), K(i, j+1), tol(i), j, problem);
            disp(tbv(i, j+1))
            diary([savedir,'DirichletRes.txt'])
        end
    end
    disp('$\mathbf{N}$ &  \textbf{tol} & & $\mathbf{q}$ & $\mathbf{\ell}$ & $\mathbf{P}$ &$\mathbf{m}$ & \textbf{error} & \textbf{time (s)} & \textbf{Speed up}\\')
    disp('\midrule')
    for i = 1:length(N)
        [ta, k] = min(tav(i,:));
        [tb, j] = min(tbv(i,:));
        disp(['{\multirow{2}{*}{$', num2str(N(i)),'$}} &  {\multirow{2}{*}{$2^{', num2str(-2*log2(N(i))),'}$}}& CPU & ',num2str(j-1),' &  ', num2str(K(i,j)),'  & ', num2str(ncontv(i,j)),' &  ', num2str(Nitersv(i,j)),' & $2^{', num2str(log2(errorbv(i,j))),'}$  &  $\mathbf{', num2str(tb),'}$& {\multirow{2}{*}{$', num2str(tb/ta),'$}}   \\'])
        disp(['\rule{0pt}{3ex} & &\shade GPU &   \shade ',num2str(k-1),'  &\shade ', num2str(K(i,k)),' &\shade  ', num2str(ncontv(i,k)),'  &\shade ', num2str(Nitersv(i,k)),' &\shade $2^{', num2str(log2(errorav(i,k))),'}$  & \shade $\mathbf{', num2str(ta),'}$&\\'])
        if i ~= length(N)
            disp('\rule{0pt}{3ex} ')
        end
    end
    disp('\bottomrule')
    diary([savedir,'DirichletRes.txt'])
    save([savedir,'dirichlet2dtimes.mat'], 'tav','tbv', 'errorav', 'errorbv', 'Nitersv', 'ncontv', 'K')
end

if neurun
    diary([savedir,'NeumannRes.txt'])
    problem = '2D Neumann';
    disp(problem)
    N = 2.^(7:10);
    tol = 1./N.^2;
    
    K = zeros(length(N), polymax+1);
    tav = inf*ones(length(N), polymax+1);
    errorav = zeros(length(N), polymax+1);
    tbv = zeros(length(N), polymax+1);
    Nitersv = zeros(length(N), polymax+1);
    errorbv = zeros(length(N), polymax+1);
    ncontv = zeros(length(N), polymax+1);
    
    for i = 1:length(N)
        filename = [savedir,'kvec' num2str(N(i)), 'neumann.mat'];
        S = load(filename);
        K(i, :) = S.Kvec(1:polymax+1);
        yncheck = S.num_matvecs;
        for j = 0:polymax
            disp([N(i), K(i, j+1), j])
            if yncheck(j+1)
                [tav(i,j+1), errorav(i, j+1)] = startup_nicksmethod_GPU(N(i), K(i, j+1), tol(i), j, problem);
                disp(tav(i, j+1))
            end
            [tbv(i,j+1), errorbv(i, j+1), Nitersv(i,j+1), ncontv(i,j+1)] = startup_nicksmethod_CPU(N(i), K(i, j+1), tol(i), j, problem);
            disp(tbv(i, j+1))
            diary([savedir,'NeumannRes.txt'])
        end
    end
    disp('$\mathbf{N}$ &  \textbf{tol} & & $\mathbf{q}$ & $\mathbf{\ell}$ & $\mathbf{P}$ &$\mathbf{m}$ & \textbf{error} & \textbf{time (s)} & \textbf{Speed up}\\')
    disp('\midrule')
    for i = 1:length(N)
        [ta, k] = min(tav(i,:));
        [tb, j] = min(tbv(i,:));
        disp(['{\multirow{2}{*}{$', num2str(N(i)),'$}} &  {\multirow{2}{*}{$2^{', num2str(-2*log2(N(i))),'}$}}& CPU & ',num2str(j-1),' &  ', num2str(K(i,j)),'  & ', num2str(ncontv(i,j)),' &  ', num2str(Nitersv(i,j)),' & $2^{', num2str(log2(errorbv(i,j))),'}$  &  $\mathbf{', num2str(tb),'}$& {\multirow{2}{*}{$', num2str(tb/ta),'$}}   \\'])
        disp(['\rule{0pt}{3ex} & &\shade GPU &   \shade ',num2str(k-1),'  &\shade ', num2str(K(i,k)),' &\shade  ', num2str(ncontv(i,k)),'  &\shade ', num2str(Nitersv(i,k)),' &\shade $2^{', num2str(log2(errorav(i,k))),'}$  & \shade $\mathbf{', num2str(ta),'}$&\\'])
        if i~=length(N)
            disp('\rule{0pt}{3ex} ')
        end
    end
    disp('\bottomrule')
    diary([savedir,'NeumannRes.txt'])
    save([savedir,'neumann2dtimes.mat'], 'tav','tbv', 'errorav', 'errorbv', 'Nitersv', 'ncontv', 'K')
end

if dirrun1d
    diary([savedir,'Dirichlet1DRes.txt'])
    problem = '1D Dirichlet';
    disp(problem)
    N = 2.^(10:2:18);
    tol = 1./N.^2;
    
    K = zeros(length(N), polymax+1);
    tav = inf*ones(length(N), polymax+1);
    errorav = zeros(length(N), polymax+1);
    tbv = zeros(length(N), polymax+1);
    Nitersv = zeros(length(N), polymax+1);
    errorbv = zeros(length(N), polymax+1);
    ncontv = zeros(length(N), polymax+1);
    
    for i = 1:length(N)
        filename = [savedir,'kvec' num2str(N(i)), 'dirichlet1d.mat'];
        S = load(filename);
        K(i, :) = S.Kvec(1:polymax+1);
        yncheck = S.num_matvecs;
        for j = 0:polymax
            disp([N(i), K(i, j+1), j])
            if yncheck(j+1)
                [tav(i,j+1), errorav(i, j+1)] = startup_nicksmethod_GPU(N(i), K(i, j+1), tol(i), j, problem);
                disp(tav(i, j+1))
            end
            [tbv(i,j+1), errorbv(i, j+1), Nitersv(i,j+1), ncontv(i,j+1)] = startup_nicksmethod_CPU(N(i), K(i, j+1), tol(i), j, problem);
            disp(tbv(i, j+1))
            diary([savedir,'Dirichlet1DRes.txt'])
        end
    end
    disp('$\mathbf{N}$ &  \textbf{tol} & & $\mathbf{q}$ & $\mathbf{\ell}$ & $\mathbf{P}$ &$\mathbf{m}$ & \textbf{error} & \textbf{time (s)} & \textbf{Speed up}\\')
    disp('\midrule')
    for i = 1:length(N)
        [ta, k] = min(tav(i,:));
        [tb, j] = min(tbv(i,:));
        disp(['{\multirow{2}{*}{$', num2str(N(i)),'$}} &  {\multirow{2}{*}{$2^{', num2str(-2*log2(N(i))),'}$}}& CPU & ',num2str(j-1),' &  ', num2str(K(i,j)),'  & ', num2str(ncontv(i,j)),' &  ', num2str(Nitersv(i,j)),' & $2^{', num2str(log2(errorbv(i,j))),'}$  &  $\mathbf{', num2str(tb),'}$& {\multirow{2}{*}{$', num2str(tb/ta),'$}}   \\'])
        disp(['\rule{0pt}{3ex} & &\shade GPU &   \shade ',num2str(k-1),'  &\shade ', num2str(K(i,k)),' &\shade  ', num2str(ncontv(i,k)),'  &\shade ', num2str(Nitersv(i,k)),' &\shade $2^{', num2str(log2(errorav(i,k))),'}$  & \shade $\mathbf{', num2str(ta),'}$&\\'])
        if i ~= length(N)
            disp('\rule{0pt}{3ex} ' )
        end
    end
    disp('\bottomrule')
    diary([savedir,'Dirichlet1DRes.txt'])
    save([savedir,'dirichlet1dtimes.mat'], 'tav','tbv', 'errorav', 'errorbv', 'Nitersv', 'ncontv', 'K')
end

if neurun1d
    diary([savedir,'Neumann1DRes.txt'])
    problem = '1D Neumann';
    disp(problem)
    N = 2.^(10:2:18);
    tol = 1./N.^2;
    
    K = zeros(length(N), polymax+1);
    tav = inf*ones(length(N), polymax+1);
    errorav = zeros(length(N), polymax+1);
    tbv = zeros(length(N), polymax+1);
    Nitersv = zeros(length(N), polymax+1);
    errorbv = zeros(length(N), polymax+1);
    ncontv = zeros(length(N), polymax+1);
    
    for i = 1:length(N)
        filename = [savedir,'kvec' num2str(N(i)), 'neumann1d.mat'];
        S = load(filename);
        K(i, :) = S.Kvec(1:polymax+1);
        yncheck = S.num_matvecs;
        for j = 0:polymax
            disp([N(i), K(i, j+1), j])
            if yncheck(j+1)
                [tav(i,j+1), errorav(i, j+1)] = startup_nicksmethod_GPU(N(i), K(i, j+1), tol(i), j, problem);
                disp(tav(i, j+1))
            end
            [tbv(i,j+1), errorbv(i, j+1), Nitersv(i,j+1), ncontv(i,j+1)] = startup_nicksmethod_CPU(N(i), K(i, j+1), tol(i), j, problem);
            disp(tbv(i, j+1))
            diary([savedir,'Neumann1DRes.txt'])
        end
    end
    disp('$\mathbf{N}$ &  \textbf{tol} & & $\mathbf{q}$ & $\mathbf{\ell}$ & $\mathbf{P}$ &$\mathbf{m}$ & \textbf{error} & \textbf{time (s)} & \textbf{Speed up}\\')
    disp('\midrule')
    for i = 1:length(N)
        [ta, k] = min(tav(i,:));
        [tb, j] = min(tbv(i,:));
        disp(['{\multirow{2}{*}{$', num2str(N(i)),'$}} &  {\multirow{2}{*}{$2^{', num2str(-2*log2(N(i))),'}$}}& CPU & ',num2str(j-1),' &  ', num2str(K(i,j)),'  & ', num2str(ncontv(i,j)),' &  ', num2str(Nitersv(i,j)),' & $2^{', num2str(log2(errorbv(i,j))),'}$  &  $\mathbf{', num2str(tb),'}$& {\multirow{2}{*}{$', num2str(tb/ta),'$}}   \\'])
        disp(['\rule{0pt}{3ex} & &\shade GPU &   \shade ',num2str(k-1),'  &\shade ', num2str(K(i,k)),' &\shade  ', num2str(ncontv(i,k)),'  &\shade ', num2str(Nitersv(i,k)),' &\shade $2^{', num2str(log2(errorav(i,k))),'}$  & \shade $\mathbf{', num2str(ta),'}$&\\'])
        if i ~= length(N)
            disp('\rule{0pt}{3ex} ')
        end
    end
    disp('\bottomrule')
    diary([savedir,'Neumann1DRes.txt'])
    save([savedir,'neumann1dtimes.mat'], 'tav','tbv', 'errorav', 'errorbv', 'Nitersv', 'ncontv', 'K')
end

