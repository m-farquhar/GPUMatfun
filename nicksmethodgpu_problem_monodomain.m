function  nicksmethodgpu_problem_monodomain(L)
% X = NICKSMETHODGPU(L,WTS1, WTS2, ETAXI, S) performs the operations of the
% contour integral method of Hale et al.


%% Build Subspace
L.build_subspace

%% Solve the smaller linear systems on GPU
L.lin_sys_solves

%% Scale problem back up to full size and compute solution
L.apply_V

%% Update the solution for next step
L.update_U_monodomain

