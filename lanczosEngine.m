%% Creates an engine to use GPU Device
classdef lanczosEngine < handle
    properties
        lanczos_handle
    end

    methods
        function h = lanczosEngine(U, A, Q, lambda, N, gamma, gamma_s_real, gamma_s_imag, exp_lambda, phi_lambda, dt, ode, aode, bode, code, vrest, vamp, vth, vpeak, c1ion, c2ion, mass, wts1, wts2, shifts)
            h.lanczos_handle = lanczos_create(U, A, Q, lambda, N, gamma, gamma_s_real, gamma_s_imag, exp_lambda, phi_lambda,dt, ode, aode, bode, code, vrest, vamp, vth, vpeak, c1ion, c2ion, mass, wts1, wts2, shifts, -1:1);
        end

        function delete(h)
           lanczos_delete(h.lanczos_handle);
           h.lanczos_handle = uint64(0);
        end

        function disp(h)
            disp('  lanczosEngine');
        end

        function build_subspace(h)
            build_subspace_mex(h.lanczos_handle);
        end

        function aabb = get_aabb(h)
            aabb = get_aabb_mex(h.lanczos_handle);
        end

        function set_y(h, yreal, yimag)
            set_y_mex(h.lanczos_handle, yreal, yimag);
        end
		
		function apply_V(h)
			apply_V_mex(h.lanczos_handle);
		end

		function update_U(h)
			update_U_mex(h.lanczos_handle);
        end
        
        function update_U_monodomain(h)
			update_U_monodomain_mex(h.lanczos_handle);
		end
		
		function U = get_U(h)
            U = get_U_mex(h.lanczos_handle);
        end
        
        function X = get_xsol(h)
            X = get_xsol_mex(h.lanczos_handle);
        end
        
        function lin_sys_solves(h)
            lin_sys_solves_mex(h.lanczos_handle);
        end
    end
end