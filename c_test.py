import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import tdi
import pdb

import cupy as cp

import GBGPU

# from GlobalFitUtils import Likelihood


def test():
    Tobs = 4.0 * ct.Julian_year

    df = 1.0 / Tobs
    data_freqs = np.arange(df, 1e-1 + df, df)
    data = np.random.randn(len(data_freqs)) + np.random.randn(len(data_freqs)) * 1j
    dt = 10.0
    NP = 8
    max_length_init = 2048
    oversample = 4
    N = int(Tobs / dt)

    # AE_ASDinv = np.ones_like(data_freqs)
    # T_ASDinv = np.ones_like(data_freqs)

    A_ASDinv = cp.asarray(
        1.0
        / np.sqrt(tdi.noisepsd_AE(data_freqs, model="SciRDv1", includewd=None))
        * np.sqrt(df)
    )

    E_ASDinv = cp.asarray(
        1.0
        / np.sqrt(tdi.noisepsd_AE(data_freqs, model="SciRDv1", includewd=None))
        * np.sqrt(df)
    )

    T_ASDinv = cp.asarray(
        1.0 / np.sqrt(tdi.noisepsd_T(data_freqs, model="SciRDv1")) * np.sqrt(df)
    )

    nWD = 1000
    ndevices = 1

    data_channel1 = cp.asarray(data)
    data_channel2 = data_channel1.copy()
    data_channel3 = data_channel1.copy()

    template_channel1 = cp.zeros(max_length_init * oversample * nWD * 2)
    template_channel2 = cp.zeros(max_length_init * oversample * nWD * 2)
    template_channel3 = cp.zeros(max_length_init * oversample * nWD * 2)

    import pdb

    pdb.set_trace()
    gbGPU = GBGPU.GBGPU(
        max_length_init, data_freqs, nWD, ndevices, Tobs, dt, NP, oversample=oversample
    )

    gbGPU.input_data(
        template_channel1,
        template_channel2,
        template_channel3,
        data_channel1,
        data_channel2,
        data_channel3,
        A_ASDinv,
        E_ASDinv,
        T_ASDinv,
    )

    f0 = np.random.uniform(1e-3, 2e-3, nWD)
    fdot = np.random.uniform(1e-12, 1e-11, nWD)
    beta = np.random.uniform(0.2, 1.2, nWD)
    lam = np.random.uniform(0.0, np.pi * 2, nWD)
    amp = np.random.uniform(1e-20, 1e-21, nWD)
    iota = np.random.uniform(0.0, np.pi / 2, nWD)
    psi = np.random.uniform(0.0, np.pi / 2, nWD)
    phi0 = np.random.uniform(0.0, 2 * np.pi, nWD)

    f0 = np.full(nWD, 1.35962000e-03)
    fdot = np.full(nWD, 8.94581279e-19)
    beta = np.full(nWD, 3.12414000e-01)
    lam = np.full(nWD, -2.75291000e00)
    amp = np.full(nWD, 1.07345000e-22)
    iota = np.full(nWD, 5.23599000e-01)
    psi = np.full(nWD, 0.42057295)
    phi0 = np.full(nWD, 3.05815650e00)

    nwalkers = 1

    """
    like_class = Likelihood(
        data_freqs,
        data_channel1,
        data_channel2,
        data_channel3,
        template_channel1,
        template_channel2,
        template_channel3,
        AE_ASDinv,
        AE_ASDinv,
        T_ASDinv,
        nwalkers,
        ndevices,
        Tobs,
        dt,
    )
    """
    params = np.array([f0, fdot, beta, lam, amp, iota, psi, phi0]).T.flatten()

    """
    fastGB = FB.FastGB("Test", dt=dt, Tobs=Tobs, orbit="analytic")

    num = 1
    st = time.perf_counter()
    for i in range(num):
        X, Y, Z = fastGB.onefourier(
            simulator="synthlisa",
            params=params[:8],
            buffer=None,
            T=Tobs,
            dt=dt,
            algorithm="Michele",
            oversample=4,
        )
    et = time.perf_counter()
    print("fastGB time per waveform:", (et - st) / num)
    """
    num = 1
    st = time.perf_counter()
    for i in range(num):
        # template_channel1[:] = 0.0
        # template_channel2[:] = 0.0
        # template_channel3[:] = 0.0
        gbGPU.FastGB(params)
        check = gbGPU.Likelihood()
        # like = like_class.GetLikelihood()
    et = time.perf_counter()
    print("GBGPU time per waveform:", (et - st) / (num * nWD))
    print("GBGPU time per waveform full fit:", (et - st) / num)

    # np.save('amp-phase', np.concatenate([np.array([freq]), amp, phase], axis=0))
    # np.save('TDI', np.array([data_freqs, A, E, T]))
    pdb.set_trace()


if __name__ == "__main__":
    test()
