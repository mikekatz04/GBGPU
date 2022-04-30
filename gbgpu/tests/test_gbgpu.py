import numpy as np
import time

import unittest

try:
    import cupy as xp
    xp.cuda.runtime.setDevice(2)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from gbgpu.gbgpu import GBGPU

from gbgpu.utils.constants import *

import sys

sys.path.append(np.__file__[:-17])

class WaveformTest(unittest.TestCase):
    def test_circ(self):

        gb = GBGPU(use_gpu=gpu_available)

        dt = 15.0
        N = None
        Tobs = 4.0 * YEAR
        num_bin = 10
        amp = 1e-22  # amplitude
        f0 = 2e-3  # f0
        fdot = 1e-18  # fdot
        fddot = 0.0
        phi0 = 0.1  # initial phase
        iota = 0.2  # inclination
        psi = 0.3  # polarization angle
        lam = 0.4  # ecliptic longitude
        beta_sky = 0.5  # ecliptic latitude

        amp_in = np.full(num_bin, amp)
        f0_in = np.full(num_bin, f0)
        fdot_in = np.full(num_bin, fdot)
        fddot_in = np.full(num_bin, fddot)
        phi0_in = np.full(num_bin, phi0)
        iota_in = np.full(num_bin, iota)
        psi_in = np.full(num_bin, psi)
        lam_in = np.full(num_bin, lam)
        beta_sky_in = np.full(num_bin, beta_sky)

        gb.run_wave(
            amp_in,
            f0_in,
            fdot_in,
            fddot_in,
            phi0_in,
            iota_in,
            psi_in,
            lam_in,
            beta_sky_in,
            N=N,
            dt=dt,
            T=Tobs,
        )

        for i in range(len(gb.X)):
            self.assertFalse(xp.any(xp.isnan(gb.X[i])))
            self.assertFalse(xp.any(xp.isnan(gb.A[i])))
            self.assertFalse(xp.any(xp.isnan(gb.E[i])))

            self.assertFalse(xp.any(xp.isinf(gb.X[i])))
            self.assertFalse(xp.any(xp.isinf(gb.A[i])))
            self.assertFalse(xp.any(xp.isinf(gb.E[i])))

    def test_likelihood(self):

        dt = 15.0
        Tobs = 4.0 * YEAR
        gb = GBGPU(use_gpu=gpu_available)
        gb.d_d = 0.0

        N = int(256)
        num_bin = 1000
        amp = 1e-22  # amplitude
        f0 = 2e-3  # f0
        fdot = 1e-14  # fdot
        fddot = 0.0
        phi0 = 0.1  # initial phase
        iota = 0.2  # inclination
        psi = 0.3  # polarization angle
        lam = 0.4  # ecliptic longitude
        beta_sky = 0.5  # ecliptic latitude

        amp_in = np.full(num_bin, amp)
        f0_in = np.full(num_bin, f0)
        fdot_in = np.full(num_bin, fdot)
        fddot_in = np.full(num_bin, fddot)
        phi0_in = np.full(num_bin, phi0)
        iota_in = np.full(num_bin, iota)
        psi_in = np.full(num_bin, psi)
        lam_in = np.full(num_bin, lam)
        beta_sky_in = np.full(num_bin, beta_sky)

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
            N=N,
            dt=dt,
            T=Tobs,
        )

        

        params = np.array(
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
            ]
        )

        like = gb.get_ll(params, data, noise_factor, N=N, dt=dt, T=Tobs,)

        self.assertFalse(np.any(np.isnan(like)))

    def test_information_matrix(self):

        dt = 15.0
        Tobs = 4.0 * YEAR
        gb = GBGPU(use_gpu=gpu_available)
        gb.d_d = 0.0

        N = int(256)
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

        amp_in = np.full(num_bin, amp)
        f0_in = np.full(num_bin, f0)
        fdot_in = np.full(num_bin, fdot)
        fddot_in = np.full(num_bin, fddot)
        phi0_in = np.full(num_bin, phi0)
        iota_in = np.full(num_bin, iota)
        psi_in = np.full(num_bin, psi)
        lam_in = np.full(num_bin, lam)
        beta_sky_in = np.full(num_bin, beta_sky)

        length = int(Tobs / dt)

        freqs = np.fft.rfftfreq(length, dt)
        data_stream_length = len(freqs)

        params = np.array(
            [amp_in, f0_in, fdot_in, fddot_in, phi0_in, iota_in, psi_in, lam_in, beta_sky_in,]
        )

        inds = np.array([0, 1, 2, 4, 5, 6, 7, 8])

        info_matrix = gb.information_matrix(
            params,
            easy_central_difference=False,
            eps=1e-9,
            inds=inds,
            N=1024,
            dt=dt,
            T=Tobs,
        )

        cov = np.linalg.pinv(info_matrix)
        self.assertFalse(np.any(np.isnan(cov)))


