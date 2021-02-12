#ifndef __NEW_FASTGB_HH__
#define __NEW_FASTGB_HH__

#include "global.h"

void get_basis_tensors_wrap(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, double* e1, double* beta1, int mode_j, int num_bin);

#ifdef __THIRD__
void GenWaveThird_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* A2_all, double* omegabar_all, double* e2_all, double* n2_all, double* T2_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int mode_j, int num_bin);
#else
void GenWave_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
             double* eplus_in, double* ecross_in,
             double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
             double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
             double* k_all, double T, int N, int mode_j, int num_bin);
#endif
//void fft_data_wrap(double *data12, double *data21, double *data13, double *data31, double *data23, double *data32, int num_bin, int N);

void unpack_data_1_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                   int N, int num_bin);

void XYZ_wrap(cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32,
             double *f0_all,
             int num_bin, int N, double dt, double T, double df, int mode_j);

void fill_global_wrap(cmplx* A_glob, cmplx* E_glob, cmplx* A_template, cmplx* E_template,
                        double* A_noise_factor, double* E_noise_factor,
                        int* start_ind_all, int M, int num_bin, int per_group, int data_length);


void get_ll_wrap(double* d_h, double* h_h,
                  cmplx* A_template, cmplx* E_template,
                  cmplx* A_data, cmplx* E_data,
                  double* A_noise_factor, double* E_noise_factor,
                  int* start_ind, int M, int num_bin);

#endif // __NEW_FASTGB_HH__
