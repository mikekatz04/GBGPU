import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust
from libcpp cimport bool
assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "ThirdSharedMemoryGBGPU.hpp":
    ctypedef void* cmplx 'cmplx'
    void ThirdSharedMemoryWaveComp(
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
        double *A2,
        double *varpi,
        double *e2,
        double *n2,
        double *T2,
        double T,
        double dt, 
        int N, 
        int num_bin_all,
        int multiply_integral_factor
    ) except+

    void ThirdSharedMemoryLikeComp(
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
        double *A2,
        double *varpi,
        double *e2,
        double *n2,
        double *T2,
        double T, 
        double dt,
        int N,
        int num_bin_all,
        int multiply_integral_factor,
        int *start_freq_ind_all, 
        int data_length,
        int device,
        bool do_synchronize
    ) except+

    void ThirdSharedMemoryGenerateGlobal(
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
        double *A2,
        double *varpi,
        double *e2,
        double *n2,
        double *T2,
        double T, 
        double dt,
        int N,
        int num_bin_all,
        int multiply_integral_factor,
        int *start_freq_ind_all,
        int data_length,
        int device,
        bool do_synchronize
    ) except+

@pointer_adjust
def ThirdSharedMemoryWaveComp_wrap(
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
    A2,
    varpi,
    e2,
    n2,
    T2,
    T,
    dt, 
    N,
    num_bin_all,
    multiply_integral_factor
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
    cdef size_t A2_in = A2
    cdef size_t varpi_in = varpi
    cdef size_t e2_in = e2
    cdef size_t n2_in = n2
    cdef size_t T2_in = T2

    ThirdSharedMemoryWaveComp(
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
        <double *>A2_in,
        <double *>varpi_in,
        <double *>e2_in,
        <double *>n2_in,
        <double *>T2_in,
        T, 
        dt,
        N,
        num_bin_all,
        multiply_integral_factor
    )


@pointer_adjust
def ThirdSharedMemoryLikeComp_wrap(
        d_h,
        h_h,
        data_A,
        data_E,
        noise_A,
        noise_E,
        data_index, 
        noise_index,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        A2,
        varpi,
        e2,
        n2,
        T2,
        T,
        dt, 
        N,
        num_bin_all,
        multiply_integral_factor,
        start_freq_ind_all,
        data_length,
        device,
        do_synchronize
    ):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t noise_A_in = noise_A
    cdef size_t noise_E_in = noise_E
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t A2_in = A2
    cdef size_t varpi_in = varpi
    cdef size_t e2_in = e2
    cdef size_t n2_in = n2
    cdef size_t T2_in = T2
    cdef size_t start_freq_ind_all_in = start_freq_ind_all

    ThirdSharedMemoryLikeComp(
        <cmplx *> d_h_in,
        <cmplx *> h_h_in,
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <double *> noise_A_in,
        <double *> noise_E_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        <double *>A2_in,
        <double *>varpi_in,
        <double *>e2_in,
        <double *>n2_in,
        <double *>T2_in,
        T, 
        dt,
        N,
        num_bin_all,
        multiply_integral_factor,
        <int*> start_freq_ind_all_in,
        data_length,
        device,
        do_synchronize
    )



@pointer_adjust
def ThirdSharedMemoryGenerateGlobal_wrap(
        data_A,
        data_E,
        data_index,
        factors,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        A2,
        varpi,
        e2,
        n2,
        T2,
        T,
        dt, 
        N,
        num_bin_all,
        multiply_integral_factor,
        start_freq_ind_all,
        data_length,
        device,
        do_synchronize
    ):

    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t data_index_in = data_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t A2_in = A2
    cdef size_t varpi_in = varpi
    cdef size_t e2_in = e2
    cdef size_t n2_in = n2
    cdef size_t T2_in = T2
    cdef size_t factors_in = factors
    cdef size_t start_freq_ind_all_in = start_freq_ind_all

    ThirdSharedMemoryGenerateGlobal(
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <int*> data_index_in,
        <double *> factors_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        <double *>A2_in,
        <double *>varpi_in,
        <double *>e2_in,
        <double *>n2_in,
        <double *>T2_in,
        T, 
        dt,
        N,
        num_bin_all,
        multiply_integral_factor,
        <int*> start_freq_ind_all_in,
        data_length,
        device,
        do_synchronize
    )
