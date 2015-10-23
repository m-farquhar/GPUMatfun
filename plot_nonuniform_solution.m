%% Plots and saves the figures for the nonuniform mesh


time_steps = 2^9;
circlenum=5;
poly=2;
load (['problemsol',num2str(time_steps),'nonuniform', filename,'2.mat'])
load( ['nonuniformmatrix', filename, num2str(poly),'.mat'], 'xvec', 'yvec','tri', 'mass')


for i = 1:9
    figure
    trisurf(tri, xvec, yvec, solmat(:,i));
    shading interp
    caxis([0,1])
    axis([-1,1,-1,1,0,1])
    set(gca,'DataAspectRatio',[1 1 1])
    set(gca, 'XTick',[-1:1])
    set(gca, 'YTick',[-1:1])
    set(gca, 'ZTick',[0:0.5:1])
    set(gca, 'fontsize',24)
    set(gca, 'fontname', 'Computer Modern')
    h=colorbar('Location', 'eastoutside');
    d = get(h);
    ax = get(gca);
    axpos = ax.Position;
    cpos = d.Position;
    cpos(2) = cpos(2) + 0.25*cpos(4);
    cpos(1) = cpos(1)+0.05;
    cpos(4) = 0.5*cpos(4);
    cpos(3) = 0.5*cpos(3);
    ax.Position = axpos;
    set(h, 'Position', cpos)
    set(h, 'fontsize',24)
    set(h, 'fontname', 'Computer Modern')
    set(gca, 'position', axpos)
    saveas(gcf, ['nonuniformsol',num2str(i),'3.eps'],'epsc')
end