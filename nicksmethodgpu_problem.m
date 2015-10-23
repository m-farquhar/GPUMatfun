function  nicksmethodgpu_problem(L)
% X = NICKSMETHODGPU(L,WTS1, WTS2, ETAXI, S) performs the operations of the
% contour integral method of Hale et al.


%% Build Subspace
L.build_subspace

%% Solve for the smaller linear systems
L.lin_sys_solves

%% Scale problem back up to full size and compute solution
L.apply_V

%% Update the solution for next step
L.update_U

