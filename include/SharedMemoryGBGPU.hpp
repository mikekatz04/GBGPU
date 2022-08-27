#ifndef __SHAREDMEMORY_GBGPU_HPP__
#define __SHAREDMEMORY_GBGPU_HPP__

#include "global.h"

void SharedMemoryWaveComp(
    cmplx* tdi_out,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta, 
    double T,
    double dt,
    int N, 
    int num_bin_all
);

typedef struct InputInfoTag{
    double* amp;
    double* f0;
    double* fdot0;
    double* fddot0;
    double* phi0;
    double* iota;
    double* psi;
    double* lam;
    double* theta;
    double T;
    double dt;
    int N;
    int num_bin_all;
    cmplx *tdi_out;
    cmplx *d_h;
    cmplx *h_h;
    cmplx *data_A; 
    cmplx *data_E;
    double *noise_A;
    double *noise_E;
    int start_freq_ind;
} InputInfo; 

void SharedMemoryLikeComp(
    cmplx* d_h,
    cmplx* h_h,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind
);
#endif // __SHAREDMEMORY_GBGPU_HPP__