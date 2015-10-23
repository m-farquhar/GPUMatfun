#include <mex.h>
#include <gpu/mxGPUArray.h>
#include <cassert>

#include "lanczos.h"

#define ARG_U prhs[0]
#define ARG_A prhs[1]
#define ARG_Q prhs[2]
#define ARG_lambda prhs[3]
#define ARG_N prhs[4]
#define ARG_gamma prhs[5]
#define ARG_gamma_s_real prhs[6]
#define ARG_gamma_s_imag prhs[7]
#define ARG_exp_lambda prhs[8]
#define ARG_phi_lambda prhs[9]
#define ARG_dt prhs[10]
#define ARG_ode prhs[11]
#define ARG_aode prhs[12]
#define ARG_bode prhs[13]
#define ARG_code prhs[14]
#define ARG_vrest prhs[15]
#define ARG_vamp prhs[16]
#define ARG_vth prhs[17]
#define ARG_vpeak prhs[18]
#define ARG_c1ion prhs[19]
#define ARG_c2ion prhs[20]
#define ARG_mass prhs[21]
#define ARG_wts1 prhs[22]
#define ARG_wts2 prhs[23]
#define ARG_shifts prhs[24]
#define ARG_valvec prhs[25]


void mexFunction(
		 int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]
		 )
{
    /* Check for proper number of input and output arguments */
    if (nrhs != 26) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:invalidNumInputs",
                "Twenty-five input arguments are required.");
    }

    if(nlhs != 1){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:maxlhs",
                "One output argument is required.");
    }

	// Initialise GPU as far as MATLAB is concerned
	//assert(mxInitGPU() == MX_GPU_SUCCESS);

	// Get U
	if(!mxIsDouble(ARG_U)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:badU",
                "Invalid U input argument.");
    }
	// Get U matrix dimensions
    mwSize Um  = mxGetM(ARG_U);
    mwSize Un  = mxGetN(ARG_U);

	// Get U matrix values
    const double* Uvalues = mxGetPr(ARG_U);
    const double* ui = mxGetPi(ARG_U);

    if (ui != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Second input argument must be a real matrix.");
    }

    // Get sparse matrix data

    /* Check data type of input arguments  */
    if (!(mxIsSparse(ARG_A))){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotSparse",
                "Invalid sparse matrix input argument.");
    }

	// Get sparse matrix dimensions
    mwSize m  = mxGetM(ARG_A);
    mwSize n  = mxGetN(ARG_A);
    
    if (m != n) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotSquare",
                        "Sparse matrix must be square.");
    }

	// Get sparse matrix values
    const double* pr = mxGetPr(ARG_A);
    const double* pi = mxGetPi(ARG_A);

    if (pi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "First input argument must be a real matrix.");
    }

	// Get sparse matrix fields
    mwIndex* ir = mxGetIr(ARG_A);
    mwIndex* jc = mxGetJc(ARG_A);
    mwSize nnz = mxGetNzmax(ARG_A);

	// Set sparse matrix descriptor
	int flag = 0;   // nonsymmetric
	sparse_matrix::descriptor_t descriptor = static_cast<sparse_matrix::descriptor_t>(flag);

	// Need to convert mwIndex to int for cusparse
	std::vector<int> ir_int;
	std::vector<int> jc_int;
	std::vector<double> values;  // new values array to allow for stripping of non-stored values

	// Do the conversion
        jc_int.assign(jc, jc + n + 1);
	ir_int.assign(ir, ir + nnz);
	values.assign(pr, pr + nnz);

       // Dimension
       int dimension = m;

        // Get Q 
	if(!mxIsDouble(ARG_Q)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:badQ",
                "Invalid Q input argument.");
    }
	// Get Q matrix dimensions
    mwSize Qm  = mxGetM(ARG_Q);
    mwSize Qn  = mxGetN(ARG_Q);

    int num_eigs = Qn;

	// Get Q matrix values
    const double* Qvalues = mxGetPr(ARG_Q);
    const double* qi = mxGetPi(ARG_Q);

    if (qi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Second input argument must be a real matrix.");
    }

        // Get lambda
	if(!mxIsDouble(ARG_lambda)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_lambda",
                "Invalid lambda input argument.");
    }
	// Get lambda matrix dimensions
    mwSize Lm  = mxGetM(ARG_lambda);
    mwSize Ln  = mxGetN(ARG_lambda);

    if (Lm != 1 && Ln != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_lambda",
                "Invalid lambda input argument.");
    }

	if (std::max(Lm, Ln) != num_eigs) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_lambda",
                "Invalid lambda input argument.");
    }

	// Get lambda matrix values
    const double* lambda_values = mxGetPr(ARG_lambda);
    const double* li = mxGetPi(ARG_lambda);

    if (li != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Third input argument must be a real matrix.");
    }

        // Get N
	if(!mxIsDouble(ARG_N)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_N",
                "Invalid Ninput argument.");
    }
	// Get N matrix dimensions
    mwSize Nm  = mxGetM(ARG_N);
    mwSize Nn  = mxGetN(ARG_N);

    if (Nm != 1 || Nn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_N",
                "Invalid N input argument.");
    }

	// Get N value
    const double* N = mxGetPr(ARG_N);
    const double* Ni = mxGetPi(ARG_N);

    if (Ni != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Fifth input argument must be a real matrix.");
    }

    int subspace_size = *N;


    // Get gamma
	if(!mxIsDouble(ARG_gamma)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_gamma",
                "Invalid gamma input argument.");
    }
	// Get gamma matrix dimensions
    mwSize Gm  = mxGetM(ARG_gamma);
    mwSize Gn  = mxGetN(ARG_gamma);

    if (Gm != 1 && Gn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_gamma",
                "Invalid gamma input argument.");
    }

    int gamma_size = std::max(Gm, Gn);

	// Get gamma values
    const double* gamma_values = mxGetPr(ARG_gamma);
    const double* gi = mxGetPi(ARG_gamma);

    if (gi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Sixth input argument must be a real matrix.");
    }

    // Get gamma_s_real
	if(!mxIsDouble(ARG_gamma_s_real)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_gamma_s_real",
                "Invalid gamma_s_real input argument.");
    }
	// Get gamma_s_real matrix dimensions
    mwSize Gm_real  = mxGetM(ARG_gamma_s_real);
    mwSize Gn_real  = mxGetN(ARG_gamma_s_real);

    if (Gm_real != gamma_size) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_gamma_s_real",
                "Invalid gamma_s_real input argument.");
    }

    int contour_size = Gn_real;

	// Get gamma_s_real values
    const double* gamma_s_real_values = mxGetPr(ARG_gamma_s_real);
    const double* gi_real = mxGetPi(ARG_gamma_s_real);

    if (gi_real != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Seventh input argument must be a real matrix.");
    }

    // Get gamma_s_imag
	if(!mxIsDouble(ARG_gamma_s_imag)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_gamma_s_imag",
                "Invalid gamma_s_imag input argument.");
    }
	// Get gamma_s_imag matrix dimensions
    mwSize Gm_imag  = mxGetM(ARG_gamma_s_imag);
    mwSize Gn_imag  = mxGetN(ARG_gamma_s_imag);

    if (Gm_imag != gamma_size || Gn_imag != contour_size) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_gamma_s_imag",
                "Invalid gamma_s_imag input argument.");
    }

	// Get gamma_s_imag values
    const double* gamma_s_imag_values = mxGetPr(ARG_gamma_s_imag);
    const double* gi_imag = mxGetPi(ARG_gamma_s_imag);

    if (gi_imag != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Eighth input argument must be a real matrix.");
    }
	
	// Get exp_lambda
	if(!mxIsDouble(ARG_exp_lambda)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_exp_lambda",
                "Invalid exp_lambda input argument.");
    }
	// Get exp_lambda matrix dimensions
    mwSize Elm  = mxGetM(ARG_exp_lambda);
    mwSize Eln  = mxGetN(ARG_exp_lambda);

    if (Elm != 1 && Eln != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_exp_lambda",
                "Invalid exp_lambda input argument.");
    }

    if (num_eigs != std::max(Elm, Eln)){	
		mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_exp_lambda",
                "Invalid exp_lambda input argument.");
	}

	// Get exp_lambda values
    const double* exp_lambda_values = mxGetPr(ARG_exp_lambda);
    const double* Eli = mxGetPi(ARG_exp_lambda);

    if (Eli != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Eighth input argument must be a real matrix.");
    }
	
	// Get phi_lambda
	if(!mxIsDouble(ARG_phi_lambda)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_phi_lambda",
                "Invalid phi_lambda input argument.");
    }
	// Get phi_lambda matrix dimensions
    mwSize Plm  = mxGetM(ARG_phi_lambda);
    mwSize Pln  = mxGetN(ARG_phi_lambda);

    if (Plm != 1 && Pln != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_phi_lambda",
                "Invalid phi_lambda input argument.");
    }

    if (num_eigs != std::max(Plm, Pln)){	
		mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_phi_lambda",
                "Invalid phi_lambda input argument.");
	}
	// Get phi_lambda values
    const double* phi_lambda_values = mxGetPr(ARG_phi_lambda);
    const double* Pli = mxGetPi(ARG_phi_lambda);

    if (Pli != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Ninth input argument must be a real matrix.");
    }

	
	    // Get dt
	if(!mxIsDouble(ARG_dt)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_dt",
                "Invalid dt input argument.");
    }
	// Get dt matrix dimensions
    mwSize dtm  = mxGetM(ARG_dt);
    mwSize dtn  = mxGetN(ARG_dt);

    if (dtm != 1 || dtn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_dt",
                "Invalid dt input argument.");
    }

	// Get dt value
    double* dt = mxGetPr(ARG_dt);
    const double* dti = mxGetPi(ARG_dt);

    if (dti != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Eleventh input argument must be a real matrix.");
    }

	
    const double timestep = *dt;
	
	
	    // Get ode
	if(!mxIsDouble(ARG_ode)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_ode",
                "Invalid ode input argument.");
    }
	// Get ode matrix dimensions
    mwSize odem  = mxGetM(ARG_ode);
    mwSize oden  = mxGetN(ARG_ode);

    if (odem != 1 && oden != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_ode",
                "Invalid ode input argument.");
    }

	if (std::max(odem, oden) != dimension) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_ode",
                "Invalid ode input argument.");
    }

	// Get ode matrix values
    const double* ode_values = mxGetPr(ARG_ode);
    const double* odei = mxGetPi(ARG_ode);

    if (odei != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twelfth input argument must be a real matrix.");
    }


	   // Get aode
	if(!mxIsDouble(ARG_aode)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_aode",
                "Invalid aode input argument.");
    }
	// Get aode matrix dimensions
    mwSize aodem  = mxGetM(ARG_aode);
    mwSize aoden  = mxGetN(ARG_aode);

    if (aodem != 1 || aoden != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_aode",
                "Invalid aode input argument.");
    }

	// Get aode value
    double* aode = mxGetPr(ARG_aode);
    const double* aodei = mxGetPi(ARG_aode);

    if (aodei != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Thirteenth input argument must be a real matrix.");
    }

	
    const double aode_value = *aode;
	
	   // Get bode
	if(!mxIsDouble(ARG_bode)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_bode",
                "Invalid bode input argument.");
    }
	// Get bode matrix dimensions
    mwSize bodem  = mxGetM(ARG_bode);
    mwSize boden  = mxGetN(ARG_bode);

    if (bodem != 1 || boden != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_bode",
                "Invalid bode input argument.");
    }

	// Get bode value
    double* bode = mxGetPr(ARG_bode);
    const double* bodei = mxGetPi(ARG_bode);

    if (bodei != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Fourteenth input argument must be a real matrix.");
    }

	
    const double bode_value = *bode;
	
		   // Get code
	if(!mxIsDouble(ARG_code)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_code",
                "Invalid code input argument.");
    }
	// Get code matrix dimensions
    mwSize codem  = mxGetM(ARG_code);
    mwSize coden  = mxGetN(ARG_code);

    if (codem != 1 || coden != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_code",
                "Invalid code input argument.");
    }

	// Get code value
    double* code = mxGetPr(ARG_code);
    const double* codei = mxGetPi(ARG_code);

    if (codei != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Fifteenth input argument must be a real matrix.");
    }

	
    const double code_value = *code;
	
	
		   // Get vrest
	if(!mxIsDouble(ARG_vrest)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vrest",
                "Invalid vrest input argument.");
    }
	// Get vrest matrix dimensions
    mwSize vrestm  = mxGetM(ARG_vrest);
    mwSize vrestn  = mxGetN(ARG_vrest);

    if (vrestm != 1 || vrestn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vrest",
                "Invalid vrest input argument.");
    }

	// Get vrest value
    double* vrest = mxGetPr(ARG_vrest);
    const double* vresti = mxGetPi(ARG_vrest);

    if (vresti != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Sixteenth input argument must be a real matrix.");
    }

	
    const double vrest_value = *vrest;
	
			   // Get vamp
	if(!mxIsDouble(ARG_vamp)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vamp",
                "Invalid vamp input argument.");
    }
	// Get vamp matrix dimensions
    mwSize vampm  = mxGetM(ARG_vamp);
    mwSize vampn  = mxGetN(ARG_vamp);

    if (vampm != 1 || vampn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vamp",
                "Invalid vamp input argument.");
    }

	// Get vamp value
    double* vamp = mxGetPr(ARG_vamp);
    const double* vampi = mxGetPi(ARG_vamp);

    if (vampi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Seventeenth input argument must be a real matrix.");
    }

	
    const double vamp_value = *vamp;
	
			   // Get vth
	if(!mxIsDouble(ARG_vth)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vth",
                "Invalid vth input argument.");
    }
	// Get vth matrix dimensions
    mwSize vthm  = mxGetM(ARG_vth);
    mwSize vthn  = mxGetN(ARG_vth);

    if (vthm != 1 || vthn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vth",
                "Invalid vth input argument.");
    }

	// Get vth value
    double* vth = mxGetPr(ARG_vth);
    const double* vthi = mxGetPi(ARG_vth);

    if (vthi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Eighteenth input argument must be a real matrix.");
    }

	
    const double vth_value = *vth;
	
			   // Get vpeak
	if(!mxIsDouble(ARG_vpeak)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vpeak",
                "Invalid vpeak input argument.");
    }
	// Get vpeak matrix dimensions
    mwSize vpeakm  = mxGetM(ARG_vpeak);
    mwSize vpeakn  = mxGetN(ARG_vpeak);

    if (vpeakm != 1 || vpeakn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_vpeak",
                "Invalid vpeak input argument.");
    }

	// Get vpeak value
    double* vpeak = mxGetPr(ARG_vpeak);
    const double* vpeaki = mxGetPi(ARG_vpeak);

    if (vpeaki != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Nineteenth input argument must be a real matrix.");
    }

	
    const double vpeak_value = *vpeak;
	
			   // Get c1ion
	if(!mxIsDouble(ARG_c1ion)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_c1ion",
                "Invalid c1ion input argument.");
    }
	// Get c1ion matrix dimensions
    mwSize c1ionm  = mxGetM(ARG_c1ion);
    mwSize c1ionn  = mxGetN(ARG_c1ion);

    if (c1ionm != 1 || c1ionn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_c1ion",
                "Invalid c1ion input argument.");
    }

	// Get c1ion value
    double* c1ion = mxGetPr(ARG_c1ion);
    const double* c1ioni = mxGetPi(ARG_c1ion);

    if (c1ioni != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twentieth input argument must be a real matrix.");
    }

	
    const double c1ion_value = *c1ion;
	
			   // Get c2ion
	if(!mxIsDouble(ARG_c2ion)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_c2ion",
                "Invalid c2ion input argument.");
    }
	// Get c2ion matrix dimensions
    mwSize c2ionm  = mxGetM(ARG_c2ion);
    mwSize c2ionn  = mxGetN(ARG_c2ion);

    if (c2ionm != 1 || c2ionn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_c2ion",
                "Invalid c2ion input argument.");
    }

	// Get c2ion value
    double* c2ion = mxGetPr(ARG_c2ion);
    const double* c2ioni = mxGetPi(ARG_c2ion);

    if (c2ioni != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twenty-first input argument must be a real matrix.");
    }

	
    const double c2ion_value = *c2ion;
	
	    // Get mass
	if(!mxIsDouble(ARG_mass)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_mass",
                "Invalid mass input argument.");
    }
	// Get mass matrix dimensions
    mwSize Mm  = mxGetM(ARG_mass);
    mwSize Mn  = mxGetN(ARG_mass);

    if (Mm != 1 && Mn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_mass",
                "Invalid mass input argument.");
    }

	if (Mm != m && Mn != m) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_mass",
                "Invalid mass input argument.");
    }
    

	// Get mass values
    const double* mass_values = mxGetPr(ARG_mass);
    const double* mi = mxGetPi(ARG_mass);

    if (mi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twenty-second input argument must be a real matrix.");
    }


	
	
		    // Get wts1
	if(!mxIsDouble(ARG_wts1)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_wts1",
                "Invalid wts1 input argument.");
    }
	// Get mass matrix dimensions
    mwSize wts1m  = mxGetM(ARG_wts1);
    mwSize wts1n  = mxGetN(ARG_wts1);

    if (wts1n != 2) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_wts1",
                "Invalid wts1 input argument.");
    }

	if (wts1m != contour_size) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_wts1",
                "Invalid wts1 input argument.");
    }
    

	// Get wts1 values
    const double* wts1_values = mxGetPr(ARG_wts1);
    const double* wts1i = mxGetPi(ARG_wts1);

    if (wts1i != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twenty-third input argument must be a real matrix.");
    }
	
	
			    // Get wts2
	if(!mxIsDouble(ARG_wts2)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_wts2",
                "Invalid wts2 input argument.");
    }
	// Get mass matrix dimensions
    mwSize wts2m  = mxGetM(ARG_wts2);
    mwSize wts2n  = mxGetN(ARG_wts2);

    if (wts2n != 2) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_wts2",
                "Invalid wts2 input argument.");
    }

	if (wts2m != contour_size) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_wts2",
                "Invalid wts2 input argument.");
    }
    

	// Get wts2 values
    const double* wts2_values = mxGetPr(ARG_wts2);
    const double* wts2i = mxGetPi(ARG_wts2);

    if (wts2i != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twenty-fourth input argument must be a real matrix.");
    }
	
	
		    // Get shifts
	if(!mxIsDouble(ARG_shifts)){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_shifts",
                "Invalid shifts input argument a.");
    }
	// Get mass matrix dimensions
    mwSize shiftsm  = mxGetM(ARG_shifts);
    mwSize shiftsn  = mxGetN(ARG_shifts);

    if (shiftsm != 1 && shiftsn != 1) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_shifts",
                "Invalid shifts input argument b.");
    }

	if (shiftsm != 4*subspace_size*contour_size && shiftsn != 4*subspace_size*contour_size) {
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:bad_shifts",
                "Invalid shifts input argument c.");
    }
    

	// Get shifts values
    const double* shifts_values = mxGetPr(ARG_shifts);
    const double* shiftsi = mxGetPi(ARG_shifts);

    if (shiftsi != 0){
        mexErrMsgIdAndTxt( "MATLAB:lanczos_create:inputNotReal",
                "Twenty-fifth input argument must be a real matrix.");
    }
	
 // Get valvec

    const double* valvec_val = mxGetPr(ARG_valvec);

   
	// Create the solver
    lanczos_solver* lanczos_solver_p = new lanczos_solver(
        descriptor, dimension, subspace_size, nnz, &values[0], &jc_int[0], &ir_int[0],
        num_eigs, Qvalues, lambda_values,
		Uvalues,
        gamma_size, gamma_values,
		contour_size, gamma_s_real_values, gamma_s_imag_values,
		exp_lambda_values, phi_lambda_values, timestep,
		ode_values, aode_value, bode_value, code_value, 
		vrest_value, vamp_value, vth_value, vpeak_value, 
		c1ion_value, c2ion_value, mass_values, wts1_values, wts2_values, shifts_values, valvec_val);
/*
    lanczos_solver(sparse_matrix::descriptor_t descriptor,
                          int dimension, int subspace_size,
                          int nonzeros, const double* values, const int* col_ptr, const int* row_ind,
                          int num_eigs, const double* Qvalues, const double* lambda_values,
						  const double* Uvalues,
                          int gamma_size, const double* gamma_values,
						  int contour_size, const double* gamma_s_real_values, const double* gamma_s_imag_values);
*/
        
	// Return the handle
    mwSize ndim = 2;
    mwSize dims[] = {1, 1};
    plhs[0] = mxCreateNumericArray(ndim, dims, mxUINT64_CLASS, mxREAL);
    *reinterpret_cast<uint64_T *>(mxGetData(plhs[0]))
        = reinterpret_cast<uint64_T>(lanczos_solver_p);  // Ugly, but legal

}
