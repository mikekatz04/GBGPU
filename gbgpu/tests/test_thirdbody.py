import numpy as np
import time

import unittest

try:
    import cupy as xp

    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from gbgpu.thirdbody import GBGPUThirdBody

from gbgpu.utils.constants import *


class WaveformTest(unittest.TestCase):
    def test_ecc_third_body(self):

        dt = 15.0
        Tobs = 4.0 * YEAR
        gb = GBGPUThirdBody(use_gpu=gpu_available)

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
            omegabar_in,
            e2_in,
            P2_in,
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
