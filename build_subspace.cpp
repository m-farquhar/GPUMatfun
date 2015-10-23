#include <mex.h>
#include <cassert>

#include "lanczos.h"

void mexFunction(
		 int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]
		 )
{
    /* Check for proper number of input and output arguments */
    if (nrhs != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_delete:invalidNumInputs",
                "Handle input argument required.");
    }

    if(nlhs != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_delete:maxlhs",
                "No output arguments are required.");
    }

    /* Check data type of input arguments  */
    if (!(mxIsUint64(prhs[0]))){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_delete:inputNotHandle",
                "First input argument must be a handle.");
    }

    // Get the handle
    lanczos_solver* lanczos_solver_p = reinterpret_cast<lanczos_solver*>(
            *reinterpret_cast<uint64_T*>(mxGetData(prhs[0]))); // Ugly, but legal

    // Do something with it
    lanczos_solver_p->build_subspace();

}
