import numpy as np
import time

import unittest

try:
    import cupy as xp

    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from gbgpu.gbgpu import GBGPU

from gbgpu.utils.constants import *

dt = 15.0
Tobs = 4.0 * YEAR
gb = GBGPU(use_gpu=gpu_available)

N = int(128)
num_bin = 10
amp = 1e-22  # amplitude
f0 = 2e-3  # f0
fdot = 1e-14  # fdot
fddot = 0.0
phi0 = 0.1  # initial phase
iota = 0.2  # inclination
psi = 0.3  # polarization angle
lam = 0.4  # ecliptic longitude
beta_sky = 0.5  # ecliptic latitude
e1 = 0.2  # eccentricity of inner binary
beta1 = 0.5  # TODO: fill in
A2 = 19.5  # third body amplitude parameter
omegabar = 0.0  # omegabar parameter
e2 = 0.3  # eccentricity of third body
P2 = 0.6  # period of third body
T2 = 0.0  # time of periapsis passage of third body

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

modes = np.array([1, 2, 3])

length = int(Tobs / dt)

freqs = np.fft.rfftfreq(length, dt)
data_stream_length = len(freqs)

data = [
    1e-24 * xp.ones(data_stream_length, dtype=np.complex128),
    1e-24 * xp.ones(data_stream_length, dtype=np.complex128),
]

noise_factor = [
    xp.ones(data_stream_length, dtype=np.float64),
    xp.ones(data_stream_length, dtype=np.float64),
]

params_circ = np.array(
    [amp_in, f0_in, fdot_in, fddot_in, phi0_in, iota_in, psi_in, lam_in, beta_sky_in,]
)

num = 100

#######################
####  CIRCULAR ########
#######################
A_inj, E_inj = gb.inject_signal(
    amp,
    f0,
    fdot,
    fddot,
    phi0,
    iota,
    psi,
    lam,
    beta_sky,
    modes=np.array([2]),
    N=N,
    dt=dt,
    T=Tobs,
)

st = time.perf_counter()
for _ in range(num):
    like = gb.get_ll(
        params_circ, data, noise_factor, N=N, dt=dt, modes=np.array([2]), T=Tobs,
    )
et = time.perf_counter()
print("circ:", (et - st) / num, "per binary:", (et - st) / (num * num_bin))


#######################
####  ECCENTRIC #######
#######################


params_ecc = np.array(
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
    ]
)

modes = np.array([1, 2, 3, 4])
A_inj, E_inj = gb.inject_signal(
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
    modes=modes,
    N=N,
    dt=dt,
    T=Tobs,
)

st = time.perf_counter()
for _ in range(num):
    like = gb.get_ll(params_ecc, data, noise_factor, N=N, dt=dt, modes=modes, T=Tobs,)
et = time.perf_counter()
print(
    "ecc ({} modes):".format(len(modes)),
    (et - st) / num,
    "per binary:",
    (et - st) / (num * num_bin),
)


##############################
## CIRCULAR / THIRD BODY #####
##############################

params_circ_third = np.array(
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
        A2_in,
        omegabar_in,
        e2_in,
        P2_in,
        T2_in,
    ]
)

A_inj, E_inj = gb.inject_signal(
    amp,
    f0,
    fdot,
    fddot,
    phi0,
    iota,
    psi,
    lam,
    beta_sky,
    A2,
    omegabar,
    e2,
    P2,
    T2,
    modes=np.array([2]),
    N=N,
    dt=dt,
    T=Tobs,
)

st = time.perf_counter()
for _ in range(num):
    like = gb.get_ll(
        params_circ_third, data, noise_factor, N=N, dt=dt, modes=np.array([2]), T=Tobs,
    )
et = time.perf_counter()
print("circ / third:", (et - st) / num, "per binary:", (et - st) / (num * num_bin))

##############################
## ECCENTRIC / THIRD BODY #####
##############################

params_full = np.array(
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
)

modes = np.array([1, 2, 3, 4])
A_inj, E_inj = gb.inject_signal(
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
    modes=modes,
    N=N,
    dt=dt,
    T=Tobs,
)

st = time.perf_counter()
for _ in range(num):
    like = gb.get_ll(params_full, data, noise_factor, N=N, dt=dt, modes=modes, T=Tobs,)
et = time.perf_counter()
print(
    "ecc/third ({} modes):".format(len(modes)),
    (et - st) / num,
    "per binary:",
    (et - st) / (num * num_bin),
)

breakpoint()
