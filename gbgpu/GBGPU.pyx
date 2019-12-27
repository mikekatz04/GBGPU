import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef int GetDeviceCount();

    cdef cppclass GBGPUwrap "GBGPU":
        GBGPUwrap(np.float64_t*,
        np.complex128_t *,
        np.complex128_t *,
        np.complex128_t *, int, np.float64_t*, np.float64_t*, np.float64_t*, int, int, np.float64_t, np.float64_t, int)

        void input_data(np.float64_t *data_freqs, np.complex128_t *,
                          np.complex128_t *, np.complex128_t *,
                          np.float64_t *, np.float64_t *,
                          np.float64_t *, int)

        void Fast_GB(np.float64_t *params)

cdef class GBGPU:
    cdef GBGPUwrap* g
    cdef int data_length
    cdef int nwalkers
    cdef int ndevices

    def __cinit__(self,
     np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3,
     np.ndarray[ndim=1, dtype=np.float64_t] channel1_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] channel2_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] channel3_ASDinv,
     nwalkers,
     ndevices,
     Tobs,
     dt,
     NP):

        self.nwalkers = nwalkers
        self.ndevices = ndevices
        self.data_length = len(data_channel1)
        self.g = new GBGPUwrap(&data_freqs[0],
        &data_channel1[0],
        &data_channel2[0],
        &data_channel3[0], self.data_length, &channel1_ASDinv[0], &channel2_ASDinv[0], &channel3_ASDinv[0],
        nwalkers, ndevices, Tobs, dt, NP)


    def input_data(self, np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel1_ASDinv,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel2_ASDinv,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel3_ASDinv):

        self.g.input_data(&data_freqs[0], &data_channel1[0],
                            &data_channel2[0], &data_channel3[0],
                            &channel1_ASDinv[0], &channel2_ASDinv[0],
                            &channel3_ASDinv[0], len(data_freqs))

    def FastGB(self, np.ndarray[ndim=1, dtype=np.float64_t] params):
        self.g.Fast_GB(&params[0])
        return

def getDeviceCount():
    return GetDeviceCount()
