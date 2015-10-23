#!/bin/make
MATLAB = /usr/local/MATLAB/R2013a
INTEL = /opt/intel/composerxe-2011.2.137
MKLROOT = $(INTEL)/mkl/lib/intel64
OUTPUT = .

CC = icpc
#DEBUG = -DMINLIN_DEBUG 
DEBUG = -DMINLIN_NO_DEBUG
OPTS = -openmp -DMATLAB_MEX_FILE -DMKL_ILP64 -fPIC -fno-omit-frame-pointer -O3
INC = -I$(MATLAB)/extern/include -I$(MATLAB)/toolbox/distcomp/gpu/extern/include -I/usr/local/cuda/include -I.
LIBS = -shared -Wl,--version-script,$(MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined 
LIBS += -Wl,-rpath-link,$(MATLAB)/bin/glnxa64 -L$(MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lmwgpu
LIBS += -Wl,--start-group $(MKLROOT)/libmkl_intel_ilp64.a $(MKLROOT)/libmkl_sequential.a $(MKLROOT)/libmkl_core.a -Wl,--end-group
LIBS += -lpthread $(INTEL)/compiler/lib/intel64/libimf.a -L/usr/local/cuda/lib64 -lcudart -lcusparse -lcublas

all : clean lanczos_create lanczos_delete build_subspace apply_V update_U update_U_monodomain get_U get_xsol lin_sys_solves

clean :
	rm *.o

cusparse_wrapper.o : cusparse_wrapper.cpp cusparse_wrapper.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c cusparse_wrapper.cpp

lanczos_impl.o : lanczos_impl.cu lanczos.h
	nvcc $(DEBUG) -arch sm_20 -I. -c lanczos_impl.cu --compiler-options -fPIC

lanczos_create.o : lanczos_create.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c lanczos_create.cpp

lanczos_delete.o : lanczos_delete.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c lanczos_delete.cpp

lanczos_create: lanczos_create.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/lanczos_create.mexa64 lanczos_create.o cusparse_wrapper.o lanczos_impl.o $(LIBS)

lanczos_delete: lanczos_delete.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/lanczos_delete.mexa64 lanczos_delete.o cusparse_wrapper.o lanczos_impl.o $(LIBS)

build_subspace_impl.o : build_subspace.cu lanczos.h
	nvcc -arch sm_20 $(DEBUG) -I. -c build_subspace.cu -o build_subspace_impl.o --compiler-options -fPIC

build_subspace.o : build_subspace.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c build_subspace.cpp

build_subspace: build_subspace.o build_subspace_impl.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/build_subspace_mex.mexa64 build_subspace.o build_subspace_impl.o cusparse_wrapper.o lanczos_impl.o $(LIBS)
	
apply_V_impl.o : apply_V.cu lanczos.h
	nvcc -arch sm_20 $(DEBUG) -I. -c apply_V.cu -o apply_V_impl.o --compiler-options -fPIC

apply_V.o : apply_V.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c apply_V.cpp

apply_V: apply_V.o apply_V_impl.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/apply_V_mex.mexa64 apply_V.o apply_V_impl.o cusparse_wrapper.o lanczos_impl.o $(LIBS)

update_U_impl.o : update_U.cu lanczos.h
	nvcc -arch sm_20 $(DEBUG) -I. -c update_U.cu -o update_U_impl.o --compiler-options -fPIC

update_U.o : update_U.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c update_U.cpp

update_U: update_U.o update_U_impl.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/update_U_mex.mexa64 update_U.o update_U_impl.o cusparse_wrapper.o lanczos_impl.o $(LIBS)
	
update_U_monodomain_impl.o : update_U_monodomain.cu lanczos.h
	nvcc -arch sm_20 $(DEBUG) -I. -c update_U_monodomain.cu -o update_U_monodomain_impl.o --compiler-options -fPIC

update_U_monodomain.o : update_U_monodomain.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c update_U_monodomain.cpp

update_U_monodomain: update_U_monodomain.o update_U_monodomain_impl.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/update_U_monodomain_mex.mexa64 update_U_monodomain.o update_U_monodomain_impl.o cusparse_wrapper.o lanczos_impl.o $(LIBS)
	
get_U.o : get_U.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c get_U.cpp

get_U: get_U.o lanczos_impl.o cusparse_wrapper.o
	$(CC) $(OPTS) -o $(OUTPUT)/get_U_mex.mexa64 get_U.o cusparse_wrapper.o lanczos_impl.o $(LIBS)

get_xsol.o : get_xsol.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c get_xsol.cpp

get_xsol: get_xsol.o lanczos_impl.o cusparse_wrapper.o
	$(CC) $(OPTS) -o $(OUTPUT)/get_xsol_mex.mexa64 get_xsol.o cusparse_wrapper.o lanczos_impl.o $(LIBS)
	
		
lin_sys_solves_impl.o : lin_sys_solves.cu lanczos.h
	nvcc -arch sm_20 $(DEBUG) -I. -c lin_sys_solves.cu -o lin_sys_solves_impl.o --compiler-options -fPIC

lin_sys_solves.o : lin_sys_solves.cpp lanczos.h
	$(CC) $(OPTS) $(DEBUG) $(INC) -c lin_sys_solves.cpp

lin_sys_solves: lin_sys_solves.o lin_sys_solves_impl.o cusparse_wrapper.o lanczos_impl.o
	$(CC) $(OPTS) -o $(OUTPUT)/lin_sys_solves_mex.mexa64 lin_sys_solves.o lin_sys_solves_impl.o cusparse_wrapper.o lanczos_impl.o $(LIBS)
