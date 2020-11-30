import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "new_fastGB.hh":
    ctypedef void* cmplx 'cmplx'

    void GenWaveThird_wrap(cmplx *data12, cmplx *data21, cmplx *data13, cmplx *data31, cmplx *data23, cmplx *data32,
                 double* eplus_in, double* ecross_in,
                 double* f0_all, double* dfdt_all, double* d2fdt2_all, double* phi0_all,
                 double* A2_all, double* omegabar_all, double* e2_all, double* n2_all, double* T2_all,
                 double* DPr_all, double* DPi_all, double* DCr_all, double* DCi_all,
                 double* k_all, double T, int N, int mode_j, int num_bin);

@pointer_adjust
def GenWaveThird(data12, data21, data13, data31, data23, data32,
            eplus, ecross,
             f0_all, dfdt_all, d2fdt2_all, phi0_all,
             A2_all, omegabar_all, e2_all, n2_all, T2_all,
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

    cdef size_t A2_all_in = A2_all
    cdef size_t omegabar_all_in = omegabar_all
    cdef size_t e2_all_in = e2_all
    cdef size_t n2_all_in = n2_all
    cdef size_t T2_all_in = T2_all

    GenWaveThird_wrap(<cmplx*>data12_in, <cmplx*>data21_in, <cmplx*>data13_in, <cmplx*>data31_in, <cmplx*>data23_in, <cmplx*>data32_in,
                 <double*> eplus_in, <double*> ecross_in,
                 <double*>f0_all_in, <double*>dfdt_all_in, <double*>d2fdt2_all_in, <double*> phi0_all_in,
                 <double*> A2_all_in, <double*> omegabar_all_in, <double*> e2_all_in, <double*> n2_all_in, <double*> T2_all_in,
                 <double*>DPr_all_in, <double*>DPi_all_in, <double*>DCr_all_in, <double*>DCi_all_in,
                 <double*>k_all_in, T, N, mode_j, num_bin)
