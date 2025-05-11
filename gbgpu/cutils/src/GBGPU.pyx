import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gbgpu_utils.hh":
    ctypedef void* cmplx 'cmplx'

    void fill_global_wrap(cmplx* A_glob, cmplx* E_glob, cmplx* A_template, cmplx* E_template,
                             int* start_ind_all, int M, int num_bin, int* group_index, int data_length);

    void get_ll_wrap(cmplx* d_h_in, cmplx* h_h_in,
                   cmplx* A_template, cmplx* E_template,
                   cmplx* A_data, cmplx* E_data,
                   double* A_psd, double* E_psd, double df,
                   int* start_ind, int M, int num_bin, int* data_index, int* noise_index, int data_length);

    void direct_like(cmplx* d_h, cmplx* h_h,
                    cmplx* A_template, cmplx* E_template,
                    cmplx* A_data, cmplx* E_data,
                    int data_length, int start_freq_ind, int nwalkers);
    void set_threads(int num_threads);
    int get_threads();

    void swap_ll_diff_wrap(cmplx* d_h_remove, cmplx* d_h_add, cmplx* add_remove, cmplx* remove_remove, cmplx* add_add, cmplx* A_remove, cmplx* E_remove, int* start_ind_all_remove, cmplx* A_add, cmplx* E_add, int* start_ind_all_add, cmplx* A_data, cmplx* E_data, double* A_psd, double* E_psd, double df, int M, int num_bin, int* data_index, int* noise_index, int data_length);


def get_threads_wrap():
    return get_threads()


def set_threads_wrap(num_threads):
    return set_threads(num_threads)


@pointer_adjust
def get_ll(d_h, h_h,
              A_template, E_template,
              A_data, E_data,
              A_psd, E_psd, df,
              start_ind, M, num_bin, data_index, noise_index, data_length):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t A_template_in = A_template
    cdef size_t E_template_in = E_template
    cdef size_t A_data_in = A_data
    cdef size_t E_data_in = E_data
    cdef size_t A_psd_in = A_psd
    cdef size_t E_psd_in = E_psd
    cdef size_t start_ind_in = start_ind
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index

    get_ll_wrap(<cmplx*> d_h_in, <cmplx*> h_h_in,
            <cmplx*> A_template_in, <cmplx*> E_template_in,
            <cmplx*> A_data_in, <cmplx*> E_data_in,
            <double*> A_psd_in, <double*> E_psd_in, df,
            <int*> start_ind_in, M, num_bin, <int*> data_index_in, <int*> noise_index_in, data_length);

@pointer_adjust
def fill_global(A_glob, E_glob,
              A_template, E_template,
              start_ind, M, num_bin, group_index, data_length):

    cdef size_t A_template_in = A_template
    cdef size_t E_template_in = E_template
    cdef size_t A_glob_in = A_glob
    cdef size_t E_glob_in = E_glob
    cdef size_t start_ind_in = start_ind
    cdef size_t group_index_in = group_index

    fill_global_wrap(<cmplx*> A_glob_in, <cmplx*> E_glob_in,
            <cmplx*> A_template_in, <cmplx*> E_template_in,
            <int*> start_ind_in, M, num_bin, <int*>group_index_in, data_length);

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

    direct_like(<cmplx*> d_h_in, <cmplx*> h_h_in,
                  <cmplx*> A_template_in, <cmplx*> E_template_in,
                  <cmplx*> A_data_in, <cmplx*> E_data_in,
                  data_length, start_freq_ind, nwalkers)

@pointer_adjust
def swap_ll_diff(d_h_remove, d_h_add, add_remove, remove_remove, add_add, A_remove, E_remove, start_ind_all_remove, A_add, E_add, start_ind_all_add, A_data, E_data, A_psd, E_psd, df, M, num_bin, data_index, noise_index, data_length):

    cdef size_t d_h_remove_in = d_h_remove
    cdef size_t d_h_add_in = d_h_add
    cdef size_t add_remove_in = add_remove
    cdef size_t remove_remove_in = remove_remove
    cdef size_t add_add_in = add_add
    cdef size_t A_remove_in = A_remove
    cdef size_t E_remove_in = E_remove
    cdef size_t start_ind_all_remove_in = start_ind_all_remove
    cdef size_t A_add_in = A_add
    cdef size_t E_add_in = E_add
    cdef size_t start_ind_all_add_in = start_ind_all_add
    cdef size_t A_data_in = A_data
    cdef size_t E_data_in = E_data
    cdef size_t A_psd_in = A_psd
    cdef size_t E_psd_in = E_psd
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index

    swap_ll_diff_wrap(<cmplx*>d_h_remove_in, <cmplx*>d_h_add_in, <cmplx*>add_remove_in, <cmplx*>remove_remove_in, <cmplx*>add_add_in, <cmplx*>A_remove_in, <cmplx*>E_remove_in, <int*>start_ind_all_remove_in, <cmplx*>A_add_in, <cmplx*>E_add_in, <int*>start_ind_all_add_in, <cmplx*>A_data_in, <cmplx*>E_data_in, <double*>A_psd_in, <double*>E_psd_in, df, M, num_bin, <int*>data_index_in, <int*>noise_index_in, data_length)