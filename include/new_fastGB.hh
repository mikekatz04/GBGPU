#ifndef __NEW_FASTGB_HH__
#define __NEW_FASTGB_HH__

void get_basis_tensors_wrap(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, int num_bin);

void GenWave_wrap(double *data12, double *data21, double *data13, double *data31, double *data23, double *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int num_bin);


#endif // __NEW_FASTGB_HH__
