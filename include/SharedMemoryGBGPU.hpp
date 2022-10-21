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
    int *data_index;
    int *noise_index;
    int start_freq_ind;
    int data_length;
    cmplx* d_h_remove;
    cmplx* d_h_add;
    cmplx* remove_remove;
    cmplx* add_add;
    cmplx* add_remove;
    double* amp_add;
    double* f0_add; 
    double* fdot0_add; 
    double* fddot0_add; 
    double* phi0_add; 
    double* iota_add;
    double* psi_add; 
    double* lam_add;
    double* theta_add;
    double* amp_remove; 
    double* f0_remove; 
    double* fdot0_remove; 
    double* fddot0_remove; 
    double* phi0_remove; 
    double* iota_remove;
    double* psi_remove; 
    double* lam_remove;
    double* theta_remove;
    int device;
    bool do_synchronize;
    double* factors;
} InputInfo; 

void SharedMemoryLikeComp(
    cmplx* d_h,
    cmplx* h_h,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    int* data_index,
    int* noise_index,
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
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize
);


void SharedMemorySwapLikeComp(
    cmplx* d_h_remove,
    cmplx* d_h_add,
    cmplx* remove_remove,
    cmplx* add_add,
    cmplx* add_remove,
    cmplx* data_A,
    cmplx* data_E,
    double* noise_A,
    double* noise_E,
    int* data_index,
    int* noise_index,
    double* amp_add, 
    double* f0_add, 
    double* fdot0_add, 
    double* fddot0_add, 
    double* phi0_add, 
    double* iota_add,
    double* psi_add, 
    double* lam_add,
    double* theta_add,
    double* amp_remove, 
    double* f0_remove, 
    double* fdot0_remove, 
    double* fddot0_remove, 
    double* phi0_remove, 
    double* iota_remove,
    double* psi_remove, 
    double* lam_remove,
    double* theta_remove,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize
);


void SharedMemoryGenerateGlobal(
    cmplx* data_A,
    cmplx* data_E,
    int* data_index,
    double* factors,
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
    int start_freq_ind,
    int data_length,
    int device,
    bool do_synchronize
);

#endif // __SHAREDMEMORY_GBGPU_HPP__