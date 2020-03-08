import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import tdi
import pdb
import FastGB as FB
import cupy as cp
from phenomhm.utils.convert import Converter, Recycler

import GBGPU
from GlobalFitUtils import Likelihood
import gpuPhenomHM as PhenomHM


def test():
    Tobs = 4.0 * ct.Julian_year

    df = 1.0 / Tobs
    data_freqs = np.arange(0.0, 1e-1 + df, df)
    data = np.random.randn(len(data_freqs)) + np.random.randn(len(data_freqs)) * 1j
    dt = 10.0
    NP = 8
    max_length_init = 2048

    template = np.zeros(len(data_freqs)) + np.zeros(len(data_freqs)) * 1j

    key_order = [
        "ln_mT",
        "mr",
        "a1",
        "a2",
        "ln_distance",  # Gpc z=2
        "phiRef",
        "cos_inc",
        "lam",
        "sin_beta",
        "psi",
        "ln_tRef",
    ]

    initial_point = {
        "ln_mT": np.log(2.00000000e06),
        "mr": 1 / 3.00000000e00,
        "a1": 0.0,
        "a2": 0.0,
        "ln_distance": np.log(3.65943000e01),  # Gpc z=2
        "phiRef": 2.13954125e00,
        "cos_inc": np.cos(1.04719755e00),
        "lam": -2.43647481e-02,
        "sin_beta": np.sin(6.24341583e-01),
        "psi": 2.02958790e00,
        "ln_tRef": np.log(5.02462348e01),
    }
    M = 2e6
    q = 0.2
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF = (
        np.logspace(-6, 0, int(2 ** 10)),
        3.09823412789,
        0.0,
        M / (1 + q),
        M * q / (1 + q),
        0.0,
        0.0,
        15.93461637 * 1e3 * 1e6 * ct.parsec,
        -1.0,
    )  # cosmo.luminosity_distance(2.0).value*1e6*ct.parsec, -1.0

    # freq = np.load('freqs.npy')
    t0 = 1.0
    tRef = 2.39284219993e1  # minutes to seconds
    merger_freq = 0.018 / ((m1 + m2) * 1.989e30 * ct.G / ct.c ** 3)
    Msec = (m1 + m2) * 1.989e30 * ct.G / ct.c ** 3
    f_ref = 0.0
    TDItag = 2
    l_vals = np.array([2, 3, 4, 2, 3, 4], dtype=np.uint32)  #
    m_vals = np.array([2, 3, 4, 1, 2, 3], dtype=np.uint32)  # ,

    Msec = (m1 + m2) * 1.989e30 * ct.G / ct.c ** 3
    upper_freq = 0.5 / Msec
    lower_freq = 1e-4 / Msec
    freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), len(freq))

    deltaF = np.zeros_like(data_freqs)
    deltaF[1:] = np.diff(data_freqs)
    deltaF[0] = deltaF[1]

    data_freqs[0] = data_freqs[1] / 10.0

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

    t_obs_start = 0.9
    t_obs_end = 0.0

    inc = np.cos(2.98553920)
    lam = 5.900332547
    beta = np.sin(-1.3748820938)
    psi = 0.139820023

    key_order = ["cos_inc", "lam", "sin_beta", "psi", "ln_tRef"]
    recycler = Recycler(key_order)

    converter = Converter(key_order, tLtoSSB=True)

    tRef_sampling_frame = tRef

    array = np.array([inc, lam, beta, psi, np.log(tRef)])

    array = recycler.recycle(array)
    array = converter.convert(array)
    inc, lam, beta, psi, tRef_wave_frame = array
    print("init:", inc, lam, beta, psi, tRef_wave_frame)
    nMBH = 300
    ndevices = 1

    nwalkers = nMBH * ndevices
    freqs_in = np.tile(freqs, nwalkers)
    m1_in = np.full(nwalkers, m1)
    m2_in = np.full(nwalkers, m2)
    chi1z_in = np.full(nwalkers, chi1z)
    chi2z_in = np.full(nwalkers, chi1z)
    distance_in = np.full(nwalkers, distance)
    phiRef_in = np.full(nwalkers, phiRef)
    f_ref_in = np.full(nwalkers, f_ref)
    inc_in = np.full(nwalkers, inc)
    lam_in = np.full(nwalkers, lam)
    beta_in = np.full(nwalkers, beta)
    psi_in = np.full(nwalkers, psi)
    t0_in = np.full(nwalkers, t0)
    tRef_wave_frame_in = np.full(nwalkers, tRef_wave_frame)
    tRef_sampling_frame_in = np.full(nwalkers, tRef_sampling_frame)
    merger_freq_in = np.full(nwalkers, merger_freq)

    # AE_ASDinv = np.ones_like(data_freqs)
    # T_ASDinv = np.ones_like(data_freqs)

    nWD = 25000
    ndevices = 1

    template_channel1 = cp.asarray(template)
    template_channel2 = template_channel1.copy()
    template_channel3 = template_channel1.copy()

    data_channel1 = data
    data_channel2 = data_channel1.copy()
    data_channel3 = data_channel1.copy()

    phenomHM = PhenomHM.PhenomHM(
        len(freq),
        l_vals,
        m_vals,
        data_freqs,
        template_channel1,
        template_channel2,
        template_channel3,
        TDItag,
        t_obs_start,
        t_obs_end,
        nMBH,
        ndevices,
    )

    gbGPU = GBGPU.GBGPU(
        max_length_init,
        data_freqs,
        template_channel1,
        template_channel2,
        template_channel3,
        nWD,
        ndevices,
        Tobs,
        dt,
        NP,
    )

    f0 = np.random.uniform(1e-3, 2e-3, nWD)
    fdot = np.random.uniform(1e-12, 1e-11, nWD)
    beta = np.random.uniform(0.2, 1.2, nWD)
    lam = np.random.uniform(0.0, np.pi * 2, nWD)
    amp = np.random.uniform(1e-20, 1e-21, nWD)
    iota = np.random.uniform(0.0, np.pi / 2, nWD)
    psi = np.random.uniform(0.0, np.pi / 2, nWD)
    phi0 = np.random.uniform(0.0, 2 * np.pi, nWD)

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

    params = np.array([f0, fdot, beta, lam, amp, iota, psi, phi0]).T.flatten()

    """fastGB = FB.FastGB("Test", dt=dt, Tobs=Tobs, orbit="analytic")

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
    print("fastGB time per waveform:", (et - st) / num)"""

    num = 300
    st = time.perf_counter()
    for i in range(num):
        like_class.ResetArrays()
        gbGPU.FastGB(params)
        phenomHM.Waveform(
            freqs_in,
            m1_in,
            m2_in,
            chi1z_in,
            chi2z_in,
            distance_in,
            phiRef_in,
            f_ref_in,
            inc_in,
            lam_in,
            beta_in,
            psi_in,
            t0_in,
            tRef_wave_frame_in,
            tRef_sampling_frame_in,
            merger_freq_in,
            return_TDI=False,
        )

        like = like_class.GetLikelihood()
    et = time.perf_counter()
    print("GBGPU time per waveform:", (et - st) / (num * nwalkers))
    print("GBGPU time per waveform full fit:", (et - st) / num)

    # np.save('amp-phase', np.concatenate([np.array([freq]), amp, phase], axis=0))
    # np.save('TDI', np.array([data_freqs, A, E, T]))
    pdb.set_trace()


if __name__ == "__main__":
    test()
