import numpy as np
import time

import unittest

try:
    import cupy as xp
    xp.cuda.runtime.setDevice(7)
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
        varpi = 0.4
        e2 = 0.3
        P2 = 0.5
        T2 = 0.3

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
        varpi_in = np.full(num_bin, varpi)
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
            varpi_in,
            e2_in,
            P2_in,
            T2_in,
            N=N,
            dt=dt,
            T=Tobs,
        )

        """n2_in = 2 * np.pi / (P2_in * YEAR)
        N_integrate1 = 10000
        xi_test1 = xp.linspace(0.0, Tobs, N_integrate1)[None, None, :]
        input_tuple1 = (
            (
                xp.asarray(f0_in[:1]), 
                xp.asarray(fdot_in[:1]), 
                xp.asarray(fddot_in[:1])
            ) 
            + (
                xp.asarray(A2_in[:1]), 
                xp.asarray(varpi_in[:1]), 
                xp.asarray(e2_in[:1]), 
                xp.asarray(n2_in[:1]), 
                xp.asarray(T2_in[:1])
            ) + (xi_test1[:, :, 1:], xi_test1[:, :, :-1])
        )
        int_step1 = xp.cumsum(gb.parab_step_ET(*input_tuple1), axis=-1)

        N_integrate2 = 256
        xi_test2 = xp.linspace(0.0, Tobs, N_integrate2)[None, None, :]
        input_tuple2 = (
            (
                xp.asarray(f0_in[:1]), 
                xp.asarray(fdot_in[:1]), 
                xp.asarray(fddot_in[:1])
            ) 
            + (
                xp.asarray(A2_in[:1]), 
                xp.asarray(varpi_in[:1]), 
                xp.asarray(e2_in[:1]), 
                xp.asarray(n2_in[:1]), 
                xp.asarray(T2_in[:1])
            ) + (xi_test2[:, :, 1:], xi_test2[:, :, :-1])
        )
        int_step2 = xp.cumsum(gb.parab_step_ET(*input_tuple2), axis=-1)
        import matplotlib.pyplot as plt
        plt.plot(xi_test1.squeeze().get()[1:], int_step1.squeeze().get())
        plt.plot(xi_test2.squeeze().get()[1:], int_step2.squeeze().get(), ls="--")
        plt.savefig("plot1.png")
        breakpoint()"""
        
        for i in range(len(gb.X)):
            self.assertFalse(xp.any(xp.isnan(gb.X[i])))
            self.assertFalse(xp.any(xp.isnan(gb.A[i])))
            self.assertFalse(xp.any(xp.isnan(gb.E[i])))

            self.assertFalse(xp.any(xp.isinf(gb.X[i])))
            self.assertFalse(xp.any(xp.isinf(gb.A[i])))
            self.assertFalse(xp.any(xp.isinf(gb.E[i])))
