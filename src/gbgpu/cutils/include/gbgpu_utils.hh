#ifndef __NEW_FASTGB_HH__
#define __NEW_FASTGB_HH__

#include "global.h"

void fill_global_wrap(cmplx *A_glob, cmplx *E_glob, cmplx *A_template, cmplx *E_template,
                      int *start_ind_all, int M, int num_bin, int *group_index, int data_length);

void get_ll_wrap(cmplx *d_h, cmplx *h_h,
                 cmplx *A_template, cmplx *E_template,
                 cmplx *A_data, cmplx *E_data,
                 double *A_psd, double *E_psd, double df,
                 int *start_ind, int M, int num_bin, int *data_index, int *noise_index, int data_length);

void direct_like(cmplx *d_h, cmplx *h_h,
                 cmplx *A_template, cmplx *E_template,
                 cmplx *A_data, cmplx *E_data,
                 int data_length, int start_freq_ind, int nwalkers);

#endif // __NEW_FASTGB_HH__
