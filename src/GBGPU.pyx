import numpy as np
cimport numpy as np

from gbgpu.utils.pointeradjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "new_fastGB.hh":
    void get_basis_tensors_wrap(double* eplus, double* ecross, double* DPr, double* DPi, double* DCr, double* DCi, double* k,
                            double* amp, double* cosiota, double* psi, double* lam, double* beta, int num_bin);


@pointer_adjust
def get_basis_tensors(eplus, ecross, DPr, DPi, DCr, DCi, k, amp, cosiota, psi, lam, beta, num_bin):

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


    get_basis_tensors_wrap(<double*> eplus_in, <double*> ecross_in, <double*> DPr_in, <double*> DPi_in, <double*> DCr_in, <double*> DCi_in, <double*> k_in,
                            <double*> amp_in, <double*> cosiota_in, <double*> psi_in, <double*> lam_in, <double*> beta_in, num_bin)
