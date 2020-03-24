import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef int GetDeviceCount();

    cdef cppclass GBGPUwrap "GBGPU":
        GBGPUwrap(int, np.float64_t*,
        int, int, int, np.float64_t, np.float64_t, int)

        void input_data(
                int data_stream_length_,
                long ptr_template_channel1_,
                long ptr_template_channel2_,
                long ptr_template_channel3_,
                long ptr_data_channel1_,
                long ptr_data_channel2_,
                long ptr_data_channel3_,
                long ptr_ASD_inv1_,
                long ptr_ASD_inv2_,
                long ptr_ASD_inv3_,
            )

        void Fast_GB(np.float64_t *params)

        void Likelihood(np.float64_t *likelihood)

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
     max_length_init,
     np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
     nwalkers,
     ndevices,
     Tobs,
     dt,
     NP,
     oversample=4):

        self.NP = NP
        self.Tobs = Tobs
        self.dt = dt
        self.nwalkers = nwalkers
        self.ndevices = ndevices
        self.N = max_length_init*oversample
        self.data_stream_length = len(data_freqs)

        self.g = new GBGPUwrap(self.data_stream_length, &data_freqs[0],
        self.N,
        nwalkers, ndevices, Tobs, dt, NP)


    def input_data(self,
        template_channel1,
        template_channel2,
        template_channel3,
        data_channel1,
        data_channel2,
        data_channel3,
        ASD_inv1,
        ASD_inv2,
        ASD_inv3):

        length_check = len(data_channel1)

        ptr_template_channel1 = template_channel1.data.mem.ptr
        ptr_template_channel2 = template_channel2.data.mem.ptr
        ptr_template_channel3 = template_channel3.data.mem.ptr

        ptr_data_channel1 = data_channel1.data.mem.ptr
        ptr_data_channel2 = data_channel2.data.mem.ptr
        ptr_data_channel3 = data_channel3.data.mem.ptr

        ptr_ASD_inv1 = ASD_inv1.data.mem.ptr
        ptr_ASD_inv2 = ASD_inv2.data.mem.ptr
        ptr_ASD_inv3 = ASD_inv3.data.mem.ptr

        self.g.input_data(length_check,
                            ptr_template_channel1, ptr_template_channel2, ptr_template_channel3,
                            ptr_data_channel1, ptr_data_channel2, ptr_data_channel3,
                            ptr_ASD_inv1, ptr_ASD_inv2, ptr_ASD_inv3)


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

    def Likelihood(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] like_ = np.zeros(2*self.nwalkers, dtype=np.float64)
        self.g.Likelihood(&like_[0])
        return like_

def getDeviceCount():
    return GetDeviceCount()
