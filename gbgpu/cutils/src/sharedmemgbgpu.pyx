import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "SharedMemoryGBGPU.hpp":
    ctypedef void* cmplx 'cmplx'
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
    
@pointer_adjust
def SharedMemoryWaveComp_wrap(
    tdi_out,
    amp, 
    f0, 
    fdot0, 
    fddot0, 
    phi0, 
    iota, 
    psi, 
    lam, 
    theta,
    T,
    dt, 
    N,
    num_bin_all
):
    cdef size_t tdi_out_in = tdi_out
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta

    SharedMemoryWaveComp(
        <cmplx *> tdi_out_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all
    )


@pointer_adjust
def SharedMemoryLikeComp_wrap(
        d_h,
        h_h,
        data_A,
        data_E,
        noise_A,
        noise_E,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_ind
    ):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t noise_A_in = noise_A
    cdef size_t noise_E_in = noise_E
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta

    SharedMemoryLikeComp(
        <cmplx *> d_h_in,
        <cmplx *> h_h_in,
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <double *> noise_A_in,
        <double *> noise_E_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        start_freq_ind
    )

