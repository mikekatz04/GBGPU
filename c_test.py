import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import tdi
import pdb
import FastGB as FB

import GBGPU


def test():
    df = 1e-5
    length = int(2 ** 11)
    data_length = int(2 * length)

    data = np.fft.rfft(np.sin(2 * np.pi * 1e-3 * np.arange(data_length) * 0.1))

    M = 4e7
    q = 0.2
    m1, m2 = (
        M / (1 + q),
        M * q / (1 + q),
    )  # cosmo.luminosity_distance(2.0).value*1e6*ct.parsec, -1.0

    Msec = (m1 + m2) * 1.989e30 * ct.G / ct.c ** 3
    upper_freq = 0.5 / Msec
    lower_freq = 1e-4 / Msec
    data_freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), length)

    data = data[:length]

    deltaF = np.zeros_like(data_freqs)
    deltaF[1:] = np.diff(data_freqs)
    deltaF[0] = deltaF[1]

    """AE_noise = np.genfromtxt('SnAE2017.dat').T
    T_noise = np.genfromtxt('SnAE2017.dat').T

    from scipy.interpolate import CubicSpline

    AE_noise = CubicSpline(AE_noise[0], AE_noise[1])
    T_noise = CubicSpline(T_noise[0], T_noise[1])

    AE_ASDinv = 1./np.sqrt(AE_noise(data_freqs))*np.sqrt(deltaF)
    AE_ASDinv = 1./np.sqrt(AE_noise(data_freqs))*np.sqrt(deltaF)
    T_ASDinv = 1./np.sqrt(T_noise(data_freqs))*np.sqrt(deltaF)"""

    AE_ASDinv = (
        1.0
        / np.sqrt(tdi.noisepsd_AE(data_freqs, model="SciRDv1", includewd=3))
        * np.sqrt(deltaF)
    )
    AE_ASDinv = (
        1.0
        / np.sqrt(tdi.noisepsd_AE(data_freqs, model="SciRDv1", includewd=3))
        * np.sqrt(deltaF)
    )
    T_ASDinv = (
        1.0 / np.sqrt(tdi.noisepsd_T(data_freqs, model="SciRDv1")) * np.sqrt(deltaF)
    )

    # AE_ASDinv = np.ones_like(data_freqs)
    # T_ASDinv = np.ones_like(data_freqs)

    nwalkers = 30000
    ndevices = 1

    Tobs = 4.0 * ct.Julian_year

    df = 1.0 / Tobs
    data_freqs = np.arange(0.0, 1e-1 + df, df)
    dt = 10.0
    NP = 8

    # data_A, data_E, data_T = np.load('data_set.npy')
    data_A, data_E, data_T = data, data, data
    gbGPU = GBGPU.GBGPU(
        data_freqs,
        data_A,
        data_E,
        data_T,
        AE_ASDinv,
        AE_ASDinv,
        T_ASDinv,
        nwalkers,
        ndevices,
        Tobs,
        dt,
        NP,
        len(data_freqs),
    )

    f0 = 1e-3
    fdot = 1e-12
    beta = 0.2
    lam = np.pi / 4.0
    amp = 1e-20
    iota = np.pi / 6.0
    psi = np.pi / 3.0
    phi0 = np.pi

    params = np.tile(np.array([f0, fdot, beta, lam, amp, iota, psi, phi0]), nwalkers)
    gbGPU.FastGB(params)

    fastGB = FB.FastGB("Test", dt=dt, Tobs=Tobs, orbit="analytic")

    num = 10
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

    num = 10
    st = time.perf_counter()
    for i in range(num):
        gbGPU.FastGB(params)
    et = time.perf_counter()
    print("GBGPU time per waveform:", (et - st) / (num * nwalkers))

    # np.save('amp-phase', np.concatenate([np.array([freq]), amp, phase], axis=0))
    # np.save('TDI', np.array([data_freqs, A, E, T]))
    pdb.set_trace()


if __name__ == "__main__":
    test()
