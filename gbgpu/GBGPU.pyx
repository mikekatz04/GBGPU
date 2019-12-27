import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef int GetDeviceCount();

    cdef cppclass GBGPUwrap "GBGPU":
        GBGPUwrap(np.float64_t*,
        np.complex128_t *,
        np.complex128_t *,
        np.complex128_t *, int, np.float64_t*, np.float64_t*, np.float64_t*, int, int, np.float64_t, np.float64_t, int, int)

        void input_data(np.float64_t *data_freqs, np.complex128_t *,
                          np.complex128_t *, np.complex128_t *,
                          np.float64_t *, np.float64_t *,
                          np.float64_t *, int)

        void Fast_GB(np.float64_t *params)

cdef class GBGPU:
    cdef GBGPUwrap* g
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
     np.ndarray[ndim=1, dtype=np.float64_t] channel1_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] channel2_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] channel3_ASDinv,
     nwalkers,
     ndevices,
     Tobs,
     dt,
     NP,
     data_stream_length):

        self.NP = NP
        self.Tobs = Tobs
        self.dt = dt
        self.nwalkers = nwalkers
        self.ndevices = ndevices
        self.N = len(data_channel1)
        self.data_stream_length = data_stream_length
        self.g = new GBGPUwrap(&data_freqs[0],
        &data_channel1[0],
        &data_channel2[0],
        &data_channel3[0], self.N, &channel1_ASDinv[0], &channel2_ASDinv[0], &channel3_ASDinv[0],
        nwalkers, ndevices, Tobs, dt, NP, data_stream_length)


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
        cdef np.ndarray[ndim=1, dtype=np.float64_t] pars = np.zeros(self.NP*self.nwalkers, dtype=np.float64)

        f0 = params[0::self.NP];
        df0 = params[1::self.NP];
        lat = params[2::self.NP];
        lng = params[3::self.NP];
        Amp = params[4::self.NP];
        incl = params[5::self.NP];
        psi = params[6::self.NP];
        phi0 = params[7::self.NP];

        pars[0::self.NP] = f0*self.Tobs
        pars[1::self.NP] = np.cos(np.pi/2. - lat) # convert to spherical polar
        pars[2::self.NP] = lng
        pars[3::self.NP] = np.log(Amp)
        pars[4::self.NP] = np.cos(incl)
        pars[5::self.NP] = psi
        pars[6::self.NP] = phi0
        pars[7::self.NP] = df0*self.Tobs*self.Tobs

        self.g.Fast_GB(&pars[0])
        return

def getDeviceCount():
    return GetDeviceCount()
