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


class WaveformTest(unittest.TestCase):
    def test_circ(self):

        gb = GBGPU(use_gpu=gpu_available)

        dt = 15.0
        N = int(128)
        Tobs = 4.0 * YEAR
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

    def test_ecc_inner(self):

        gb = GBGPU(use_gpu=gpu_available)

        dt = 15.0
        Tobs = 4.0 * YEAR
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
        e1 = 0.2  # eccentricity
        beta1 = 0.5

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

        modes = np.array([1, 2, 3])

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
            e1_in,
            beta1_in,
            N=N,
            dt=dt,
            T=Tobs,
            modes=modes,
        )

        for i in range(len(gb.X)):
            self.assertFalse(xp.any(xp.isnan(gb.X[i])))
            self.assertFalse(xp.any(xp.isnan(gb.A[i])))
            self.assertFalse(xp.any(xp.isnan(gb.E[i])))

            self.assertFalse(xp.any(xp.isinf(gb.X[i])))
            self.assertFalse(xp.any(xp.isinf(gb.A[i])))
            self.assertFalse(xp.any(xp.isinf(gb.E[i])))

    def test_circ_inner_ecc_third_body(self):

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
        A2_in = np.full(num_bin, A2)
        P2_in = np.full(num_bin, P2)
        omegabar_in = np.full(num_bin, omegabar)
        e2_in = np.full(num_bin, e2)
        T2_in = np.full(num_bin, T2)

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
            A2_in,
            P2_in,
            omegabar_in,
            e2_in,
            T2_in,
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

    def test_ecc_inner_ecc_third_body(self):

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
        e1 = 0.2
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

        modes = np.array([1, 2, 3])

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
            e1_in,
            beta1_in,
            A2_in,
            P2_in,
            omegabar_in,
            e2_in,
            T2_in,
            N=N,
            dt=dt,
            modes=modes,
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
        e1 = 0.2
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
                e1_in,
                beta1_in,
                A2_in,
                P2_in,
                omegabar_in,
                e2_in,
                T2_in,
            ]
        )

        like = gb.get_ll(params, data, noise_factor, N=N, dt=dt, modes=modes, T=Tobs,)
