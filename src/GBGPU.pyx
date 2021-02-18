import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "new_fastGB.hh":
    ctypedef void* cmplx 'cmplx'
    void get_basis_tensors_wrap(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, double* e1, double* beta1, int mode_j, int num_bin);

    void GenWave_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                 double* eplus_in, double* ecross_in,
                 double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
                 double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
                 double* k_all, double T, int N, int mode_j, int num_bin);

    void unpack_data_1_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                       int N, int num_bin);

    void XYZ_wrap(cmplx *a12, cmplx *a21, cmplx *a13, cmplx *a31, cmplx *a23, cmplx *a32,
                 double *f0_all,
                 int num_bin, int N, double dt, double T, double df, int mode_j);

    void fill_global_wrap(cmplx* A_glob, cmplx* E_glob, cmplx* A_template, cmplx* E_template,
                             double* A_noise_factor, double* E_noise_factor,
                             int* start_ind_all, int M, int num_bin, int per_group, int data_length, int start_freq_ind);

    void get_ll_wrap(double* d_h_in, double* h_h_in,
                   cmplx* A_template, cmplx* E_template,
                   cmplx* A_data, cmplx* E_data,
                   double* A_noise_factor, double* E_noise_factor,
                   int* start_ind, int M, int num_bin);

    void direct_like(double* d_h, double* h_h,
                    cmplx* A_template, cmplx* E_template,
                    cmplx* A_data, cmplx* E_data,
                    int data_length, int start_freq_ind, int nwalkers);


@pointer_adjust
def get_basis_tensors(eplus, ecross, DPr, DPi, DCr, DCi, k, amp, cosiota, psi, lam, beta, e1, beta1, mode_j, num_bin):

    cdef size_t eplus_in = eplus
    cdef size_t ecross_in = ecross
    cdef size_t DPr_in = DPr
    cdef size_t DPi_in = DPi
    cdef size_t DCr_in = DCr
    cdef size_t DCi_in = DCi
    cdef size_t k_in = k
    cdef size_t amp_in = amp
    cdef size_t cosiota_in = cosiota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t beta_in = beta
    cdef size_t e1_in = e1
    cdef size_t beta1_in = beta1

    get_basis_tensors_wrap(<double*> eplus_in, <double*> ecross_in, <double*> DPr_in, <double*> DPi_in, <double*> DCr_in, <double*> DCi_in, <double*> k_in,
                            <double*> amp_in, <double*> cosiota_in, <double*> psi_in, <double*> lam_in, <double*> beta_in, <double*> e1_in, <double*> beta1_in,
                            mode_j, num_bin)


@pointer_adjust
def GenWave(data12, data21, data13, data31, data23, data32,
            eplus, ecross,
             f0_all, dfdt_all, d2fdt2_all, phi0_all,
             DPr_all, DPi_all, DCr_all, DCi_all,
             k_all, T, N, mode_j, num_bin):

    cdef size_t data12_in = data12
    cdef size_t data21_in = data21
    cdef size_t data13_in = data13
    cdef size_t data31_in = data31
    cdef size_t data23_in = data23
    cdef size_t data32_in = data32
    cdef size_t f0_all_in = f0_all
    cdef size_t dfdt_all_in = dfdt_all
    cdef size_t d2fdt2_all_in = d2fdt2_all
    cdef size_t phi0_all_in = phi0_all
    cdef size_t k_all_in = k_all
    cdef size_t eplus_in = eplus
    cdef size_t ecross_in = ecross
    cdef size_t DPr_all_in = DPr_all
    cdef size_t DPi_all_in = DPi_all
    cdef size_t DCr_all_in = DCr_all
    cdef size_t DCi_all_in = DCi_all

    GenWave_wrap(<cmplx*>data12_in, <cmplx*>data21_in, <cmplx*>data13_in, <cmplx*>data31_in, <cmplx*>data23_in, <cmplx*>data32_in,
                 <double*> eplus_in, <double*> ecross_in,
                 <double*>f0_all_in, <double*>dfdt_all_in, <double*>d2fdt2_all_in, <double*> phi0_all_in,
                 <double*>DPr_all_in, <double*>DPi_all_in, <double*>DCr_all_in, <double*>DCi_all_in,
                 <double*>k_all_in, T, N, mode_j, num_bin)


@pointer_adjust
def unpack_data_1(data12, data21, data13, data31, data23, data32,
                  N, num_bin):

    cdef size_t data12_in = data12
    cdef size_t data21_in = data21
    cdef size_t data13_in = data13
    cdef size_t data31_in = data31
    cdef size_t data23_in = data23
    cdef size_t data32_in = data32

    unpack_data_1_wrap(<cmplx*>data12_in, <cmplx*>data21_in, <cmplx*>data13_in, <cmplx*>data31_in, <cmplx*>data23_in, <cmplx*>data32_in,
                      N, num_bin)


@pointer_adjust
def XYZ(a12, a21, a13, a31, a23, a32,
        f0_all,
        num_bin, N, dt, T, df, mode_j):

    cdef size_t a12_in = a12
    cdef size_t a21_in = a21
    cdef size_t a13_in = a13
    cdef size_t a31_in = a31
    cdef size_t a23_in = a23
    cdef size_t a32_in = a32
    cdef size_t f0_all_in = f0_all

    XYZ_wrap(<cmplx *>a12_in, <cmplx *>a21_in, <cmplx *>a13_in, <cmplx *>a31_in, <cmplx *>a23_in, <cmplx *>a32_in,
                <double *>f0_all_in,
                num_bin, N, dt, T, df, mode_j)


@pointer_adjust
def get_ll(d_h, h_h,
              A_template, E_template,
              A_data, E_data,
              A_noise_factor, E_noise_factor,
              start_ind, M, num_bin):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t A_template_in = A_template
    cdef size_t E_template_in = E_template
    cdef size_t A_data_in = A_data
    cdef size_t E_data_in = E_data
    cdef size_t A_noise_factor_in = A_noise_factor
    cdef size_t E_noise_factor_in = E_noise_factor
    cdef size_t start_ind_in = start_ind

    get_ll_wrap(<double*> d_h_in, <double*> h_h_in,
            <cmplx*> A_template_in, <cmplx*> E_template_in,
            <cmplx*> A_data_in, <cmplx*> E_data_in,
            <double*> A_noise_factor_in, <double*> E_noise_factor_in,
            <int*> start_ind_in, M, num_bin);

@pointer_adjust
def fill_global(A_glob, E_glob,
              A_template, E_template,
              A_noise_factor, E_noise_factor,
              start_ind, M, num_bin, per_group, data_length, start_freq_ind):

    cdef size_t A_template_in = A_template
    cdef size_t E_template_in = E_template
    cdef size_t A_glob_in = A_glob
    cdef size_t E_glob_in = E_glob
    cdef size_t A_noise_factor_in = A_noise_factor
    cdef size_t E_noise_factor_in = E_noise_factor
    cdef size_t start_ind_in = start_ind

    fill_global_wrap(<cmplx*> A_glob_in, <cmplx*> E_glob_in,
            <cmplx*> A_template_in, <cmplx*> E_template_in,
            <double*> A_noise_factor_in, <double*> E_noise_factor_in,
            <int*> start_ind_in, M, num_bin, per_group, data_length, start_freq_ind);

@pointer_adjust
def direct_like_wrap(d_h, h_h,
                 A_template, E_template,
                 A_data, E_data,
                 data_length, start_freq_ind, nwalkers):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t A_template_in = A_template
    cdef size_t E_template_in = E_template
    cdef size_t A_data_in = A_data
    cdef size_t E_data_in = E_data

    direct_like(<double*> d_h_in, <double*> h_h_in,
                  <cmplx*> A_template_in, <cmplx*> E_template_in,
                  <cmplx*> A_data_in, <cmplx*> E_data_in,
                  data_length, start_freq_ind, nwalkers)
