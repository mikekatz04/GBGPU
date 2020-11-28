import numpy as np

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False
# import matplotlib.pyplot as plt

from lisatools.sensitivity import get_sensitivity
from lisatools.diagnostic import (
    inner_product,
    snr,
    fisher,
    covariance,
    mismatch_criterion,
    cutler_vallisneri_bias,
    scale_snr,
)

from gbgpu.new_gbgpu import GBGPU

from lisatools.sampling.likelihood import Likelihood

from lisatools.sensitivity import get_sensitivity

# from lisatools.sampling.samplers.emcee import EmceeSampler
from lisatools.sampling.samplers.ptemcee import PTEmceeSampler

import warnings

YEAR = 31457280.0

warnings.filterwarnings("ignore")

use_gpu = gpu_available
gb = GBGPU(use_gpu=use_gpu)

num_bin = 400
amp = 1e-19
f0 = 2e-3
fdot = 1e-14
fddot = 0.0
phi0 = 0.1
iota = 0.2
psi = 0.3
lam = 0.4
beta_sky = 0.5
e1 = 0.3
beta1 = 0.5
A2 = 19.5
omegabar = 0.0
e2 = 0.3
P2 = 0.6
T2 = 0.0

amp_in = np.full(num_bin, amp)
f0_in = np.full(num_bin, f0)
fdot_in = np.full(num_bin, fdot)
fddot_in = np.full(num_bin, fddot)
phi0_in = np.full(num_bin, phi0)
iota_in = np.full(num_bin, iota)
psi_in = np.full(num_bin, psi)
lam_in = np.full(num_bin, lam)
beta_sky_in = np.full(num_bin, beta_sky)
e1_in = np.full(num_bin, e1)
beta1_in = np.full(num_bin, beta1)
A2_in = np.full(num_bin, A2)
P2_in = np.full(num_bin, P2)
omegabar_in = np.full(num_bin, omegabar)
e2_in = np.full(num_bin, e2)
T2_in = np.full(num_bin, T2)
N = int(128)

modes = np.array([1, 2, 3])

Tobs = 4.0 * YEAR
dt = 15.0

waveform_kwargs = dict(modes=modes, N=N, dt=dt)

like = Likelihood(
    gb, 2, frequency_domain=True, parameter_transforms={}, use_gpu=use_gpu,
)

injection_params = np.array(
    [
        amp,
        f0,
        fdot,
        fddot,
        phi0,
        iota,
        psi,
        lam,
        beta_sky,
        e1,
        beta1,
        A2,
        omegabar,
        e2,
        P2,
        T2,
    ]
)

A_inj, E_inj = gb.inject_signal(
    Tobs,
    amp,
    f0,
    fdot,
    fddot,
    phi0,
    iota,
    psi,
    lam,
    beta_sky,
    e1,
    beta1,
    A2,
    omegabar,
    e2,
    P2,
    T2,
    **waveform_kwargs,
)

like.inject_signal(
    1 / Tobs,
    data_stream=[A_inj, E_inj],
    noise_fn=get_sensitivity,
    noise_kwargs={"sens_fn": "noisepsd_AE"},
    add_noise=False,
)

amp_in[0] *= 1.1
f0_in[1] = 2.001e-3
params_test = np.array(
    [
        amp_in,
        f0_in,
        fdot_in,
        fddot_in,
        phi0_in,
        iota_in,
        psi_in,
        lam_in,
        beta_sky_in,
        e1_in,
        beta1_in,
        A2_in,
        omegabar_in,
        e2_in,
        P2_in,
        T2_in,
    ]
).T

check = like.get_ll(params_test, waveform_kwargs=waveform_kwargs)

nwalkers = 500

ndim_full = 16

test_inds = np.delete(np.arange(ndim_full), np.array([3, 12, 15]))

ndim = len(test_inds)
fill_inds = np.delete(np.arange(ndim_full), test_inds)
fill_values = injection_params[fill_inds]

prior_ranges = [
    [injection_params[i] * 0.95, injection_params[i] * 1.05] for i in test_inds
]

ntemps = 4
Tmax = np.inf

sampler = PTEmceeSampler(
    nwalkers,
    ndim,
    ndim_full,
    like,
    prior_ranges,
    subset=None,
    lnlike_kwargs={"waveform_kwargs": waveform_kwargs},
    test_inds=test_inds,
    fill_values=fill_values,
    ntemps=ntemps,
    Tmax=Tmax,
    autocorr_multiplier=500,
    fp="test_full_gb.h5",
)

"""
eps = 1e-9
cov = covariance(
    fast,
    injection_params,
    eps,
    dt,
    deriv_inds=test_inds,
    parameter_transforms=transform_fn,
    waveform_kwargs=waveform_kwargs,
    inner_product_kwargs=inner_product_kwargs,
    diagonalize=False,
)
"""

factor = 1e-2
start_points = (
    injection_params[np.newaxis, test_inds]
    + factor
    * np.random.randn(nwalkers * ntemps, ndim)
    * 0.00001
    * injection_params[np.newaxis, test_inds]
)

max_iter = 40000
sampler.sample(start_points, max_iter, show_progress=True)
