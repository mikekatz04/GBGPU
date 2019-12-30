import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import tdi
import pdb
import FastGB as FB
import cupy as cp

import GBGPU


def test():
    Tobs = 4.0 * ct.Julian_year

    df = 1.0 / Tobs
    data_freqs = np.arange(0.0, 1e-1 + df, df)
    data = np.random.randn(len(data_freqs)) + np.random.randn(len(data_freqs)) * 1j
    dt = 10.0
    NP = 8
    max_length_init = 2048

    deltaF = np.zeros_like(data_freqs)
    deltaF[1:] = np.diff(data_freqs)
    deltaF[0] = deltaF[1]
    data_freqs[0] = data_freqs[1] / 10.0

    """AE_noise = np.genfromtxt('SnAE2017.dat').T
    T_noise = np.genfromtxt('SnAE2017.dat').T

    from scipy.interpolate import CubicSpline

    AE_noise = CubicSpline(AE_noise[0], AE_noise[1])
    T_noise = CubicSpline(T_noise[0], T_noise[1])

    AE_ASDinv = 1./np.sqrt(AE_noise(data_freqs))*np.sqrt(deltaF)
    AE_ASDinv = 1./np.sqrt(AE_noise(data_freqs))*np.sqrt(deltaF)
    T_ASDinv = 1./np.sqrt(T_noise(data_freqs))*np.sqrt(deltaF)"""

    ASDinv = (
        1.0
        / np.sqrt(tdi.noisepsd_X(data_freqs, model="SciRDv1", includewd=3))
        * np.sqrt(deltaF)
    )

    # AE_ASDinv = np.ones_like(data_freqs)
    # T_ASDinv = np.ones_like(data_freqs)

    nwalkers = 25000
    ndevices = 1

    # data_A, data_E, data_T = np.load('data_set.npy')
    data_channel1 = cp.zeros(2 * len(data))
    data_channel2 = cp.zeros(2 * len(data))
    data_channel3 = cp.zeros(2 * len(data))
    print("in py:", data_channel1.data.mem.ptr)

    data_A, data_E, data_T = data, data, data
    gbGPU = GBGPU.GBGPU(
        max_length_init,
        data_freqs,
        data_channel1,
        data_channel2,
        data_channel3,
        nwalkers,
        ndevices,
        Tobs,
        dt,
        NP,
    )

    f0 = np.random.uniform(1e-3, 2e-3, nwalkers)
    fdot = np.random.uniform(1e-12, 1e-11, nwalkers)
    beta = np.random.uniform(0.2, 1.2, nwalkers)
    lam = np.random.uniform(0.0, np.pi * 2, nwalkers)
    amp = np.random.uniform(1e-20, 1e-21, nwalkers)
    iota = np.random.uniform(0.0, np.pi / 2, nwalkers)
    psi = np.random.uniform(0.0, np.pi / 2, nwalkers)
    phi0 = np.random.uniform(0.0, 2 * np.pi, nwalkers)

    params = np.array([f0, fdot, beta, lam, amp, iota, psi, phi0]).T.flatten()
    print("st")
    gbGPU.FastGB(params)

    fastGB = FB.FastGB("Test", dt=dt, Tobs=Tobs, orbit="analytic")

    num = 1
    st = time.perf_counter()
    for i in range(num):
        fastGB.onefourier(
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

    num = 100
    st = time.perf_counter()
    for i in range(num):
        gbGPU.FastGB(params)
        # like = gbGPU.Likelihood()
    et = time.perf_counter()
    print("GBGPU time per waveform:", (et - st) / (num * nwalkers))
    print("GBGPU time per waveform full fit:", (et - st) / num)

    # np.save('amp-phase', np.concatenate([np.array([freq]), amp, phase], axis=0))
    # np.save('TDI', np.array([data_freqs, A, E, T]))
    pdb.set_trace()


if __name__ == "__main__":
    test()
