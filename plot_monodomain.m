%% Plots and the solutions to the monodomain problem
N = 512;
alpha = [2, 1.7, 1.5, 2, 2];
Kalpha = [1e-4, 1e-4, 1e-4, 3e-5, 1e-5];
names = cell(length(alpha),1);
for i = 1: length(alpha)
    names{i} = ['problemsol220000', num2str(alpha(i)*10), num2str(Kalpha(i)*1e5),'monodomain3', num2str(N),'.mat'];
end
count = [1,2,3,5,6];
x = linspace(0,2.5, N);
[X, Y] = meshgrid(x,x);

for i = 1:5
   load(names{i})
   u = reshape(output, N, N);
   figure;
   contourf(X, Y, u)
   view(2)
   grid off
   axis off
   axis square
   saveas(gcf, ['monodomain', num2str(count(i)),'.eps'], 'epsc2')
end