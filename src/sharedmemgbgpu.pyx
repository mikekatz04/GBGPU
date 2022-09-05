import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust
from libcpp cimport bool
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
    ) except+

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
        int data_length
    ) except+

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
    ) except+


    void SharedMemoryGenerateGlobal(
        cmplx* data_A,
        cmplx* data_E,
        int* data_index,
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
        int data_length
    ) except+
        
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
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_ind,
        data_length
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

    SharedMemoryLikeComp(
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
        T, 
        dt,
        N,
        num_bin_all,
        start_freq_ind,
        data_length
    )



@pointer_adjust
def SharedMemorySwapLikeComp_wrap(
        d_h_remove,
        d_h_add,
        remove_remove,
        add_add,
        add_remove,
        data_A,
        data_E,
        noise_A,
        noise_E,
        data_index, 
        noise_index,
        amp_add, 
        f0_add, 
        fdot0_add, 
        fddot0_add, 
        phi0_add, 
        iota_add, 
        psi_add, 
        lam_add, 
        theta_add,
        amp_remove, 
        f0_remove, 
        fdot0_remove, 
        fddot0_remove, 
        phi0_remove, 
        iota_remove, 
        psi_remove, 
        lam_remove, 
        theta_remove,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    ):

    cdef size_t d_h_remove_in = d_h_remove
    cdef size_t d_h_add_in = d_h_add
    cdef size_t remove_remove_in = remove_remove
    cdef size_t add_add_in = add_add
    cdef size_t add_remove_in = add_remove
    cdef size_t data_A_in = data_A
    cdef size_t data_E_in = data_E
    cdef size_t noise_A_in = noise_A
    cdef size_t noise_E_in = noise_E
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_add_in = amp_add
    cdef size_t f0_add_in = f0_add
    cdef size_t fdot0_add_in = fdot0_add
    cdef size_t fddot0_add_in = fddot0_add
    cdef size_t phi0_add_in = phi0_add
    cdef size_t iota_add_in = iota_add
    cdef size_t psi_add_in = psi_add
    cdef size_t lam_add_in = lam_add
    cdef size_t theta_add_in = theta_add
    cdef size_t amp_remove_in = amp_remove
    cdef size_t f0_remove_in = f0_remove
    cdef size_t fdot0_remove_in = fdot0_remove
    cdef size_t fddot0_remove_in = fddot0_remove
    cdef size_t phi0_remove_in = phi0_remove
    cdef size_t iota_remove_in = iota_remove
    cdef size_t psi_remove_in = psi_remove
    cdef size_t lam_remove_in = lam_remove
    cdef size_t theta_remove_in = theta_remove

    SharedMemorySwapLikeComp(
        <cmplx *> d_h_remove_in,
        <cmplx *> d_h_add_in,
        <cmplx *> remove_remove_in,
        <cmplx *> add_add_in,
        <cmplx *> add_remove_in,
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <double *> noise_A_in,
        <double *> noise_E_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>amp_add_in, 
        <double *>f0_add_in, 
        <double *>fdot0_add_in, 
        <double *>fddot0_add_in, 
        <double *>phi0_add_in, 
        <double *>iota_add_in,
        <double *>psi_add_in, 
        <double *>lam_add_in,
        <double *>theta_add_in,
        <double *>amp_remove_in, 
        <double *>f0_remove_in, 
        <double *>fdot0_remove_in, 
        <double *>fddot0_remove_in, 
        <double *>phi0_remove_in, 
        <double *>iota_remove_in,
        <double *>psi_remove_in, 
        <double *>lam_remove_in,
        <double *>theta_remove_in,
        T, 
        dt,
        N,
        num_bin_all,
        start_freq_ind,
        data_length,
        device,
        do_synchronize
    )


@pointer_adjust
def SharedMemoryGenerateGlobal_wrap(
        data_A,
        data_E,
        data_index,
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
        start_freq_ind,
        data_length
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

    SharedMemoryGenerateGlobal(
        <cmplx *> data_A_in,
        <cmplx *> data_E_in,
        <int*> data_index_in,
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
        start_freq_ind,
        data_length
    )
