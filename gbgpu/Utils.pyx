import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/likelihood.hh":
    cdef int GetDeviceCount();

    cdef cppclass Likelihoodwrap "Likelihood":
        Likelihoodwrap(int, np.float64_t*, np.complex128_t*, np.complex128_t*, np.complex128_t*,
        long,
        long,
        long, np.float64_t*, np.float64_t*, np.float64_t*, int, int, np.float64_t, np.float64_t)

        void input_data(np.float64_t *data_freqs, np.complex128_t *,
                          np.complex128_t *, np.complex128_t *,
                          np.float64_t *, np.float64_t *, np.float64_t *, int)

        void GetLikelihood(np.float64_t *likelihood)

        void ResetArrays()

cdef class Likelihood:
    cdef Likelihoodwrap* g
    cdef int N
    cdef int data_stream_length
    cdef int nwalkers
    cdef int ndevices
    cdef Tobs
    cdef dt
    cdef int NP

    def __cinit__(self,
     np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3,
     template_channel1, template_channel2, template_channel3,
     np.ndarray[ndim=1, dtype=np.float64_t] channel1_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] channel2_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] channel3_ASDinv,
     nwalkers,
     ndevices,
     Tobs,
     dt):

        self.Tobs = Tobs
        self.dt = dt
        self.nwalkers = nwalkers
        self.ndevices = ndevices
        self.data_stream_length = len(data_freqs)

        ptr_data_channel1 = template_channel1.data.mem.ptr
        ptr_data_channel2 = template_channel2.data.mem.ptr
        ptr_data_channel3 = template_channel3.data.mem.ptr

        self.g = new Likelihoodwrap(self.data_stream_length, &data_freqs[0],
        &data_channel1[0], &data_channel2[0], &data_channel3[0],
        ptr_data_channel1, ptr_data_channel2, ptr_data_channel3,
        &channel1_ASDinv[0], &channel2_ASDinv[0], &channel3_ASDinv[0],
        nwalkers, ndevices, Tobs, dt)


    def input_data(self, np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel1_ASDinv,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel2_ASDinv,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel3_ASDinv):

        self.g.input_data(&data_freqs[0], &data_channel1[0],
                            &data_channel2[0], &data_channel3[0],
                            &channel1_ASDinv[0], &channel2_ASDinv[0], &channel3_ASDinv[0], len(data_freqs))


    def GetLikelihood(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] like_ = np.zeros(2, dtype=np.float64)
        self.g.GetLikelihood(&like_[0])
        return like_

    def ResetArrays(self):
        self.g.ResetArrays()
        return

def getDeviceCount():
    return GetDeviceCount()
