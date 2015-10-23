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
        mexErrMsgIdAndTxt( "MATLAB:get_xsol:invalidNumInputs",
                "Handle input argument required.");
    }

    if(nlhs != 1){
        mexErrMsgIdAndTxt( "MATLAB:get_xsol:maxlhs",
                "One output arguments is required.");
    }

    /* Check data type of input arguments  */
    if (!(mxIsUint64(prhs[0]))){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_delete:inputNotHandle",
                "First input argument must be a handle.");
    }

    // Get the handle
    lanczos_solver* lanczos_solver_p = reinterpret_cast<lanczos_solver*>(
            *reinterpret_cast<uint64_T*>(mxGetData(prhs[0]))); // Ugly, but legal

    // Create output matrix
    plhs[0] = mxCreateDoubleMatrix(lanczos_solver_p->X.rows(), lanczos_solver_p->X.cols(), mxREAL);

    // Copy over the values
    lanczos_solver_p->get_xsol(mxGetPr(plhs[0]));

}
