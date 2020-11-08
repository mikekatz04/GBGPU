import numpy as np
import time

from newfastgb_cpu import get_basis_tensors as get_basis_tensors_cpu
from newfastgb_cpu import GenWave as GenWave_cpu
from newfastgb_cpu import unpack_data_1 as unpack_data_1_cpu
from newfastgb_cpu import XYZ as XYZ_cpu

try:
    import cupy as xp
    from newfastgb import get_basis_tensors, GenWave, unpack_data_1, XYZ

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    print("no cupy")

YEAR = 31457280.0


class GBGPU(object):
    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.xp = xp
            self.get_basis_tensors = get_basis_tensors
            self.GenWave = GenWave
            self.unpack_data_1 = unpack_data_1
            self.XYZ = XYZ

        else:
            self.xp = np
            self.get_basis_tensors = get_basis_tensors_cpu
            self.GenWave = GenWave_cpu
            self.unpack_data_1 = unpack_data_1_cpu
            self.XYZ = XYZ_cpu

    def run_wave(
        self,
        amp,
        f0,
        fdot,
        fddot,
        phi0,
        iota,
        psi,
        lam,
        beta,
        N=int(2 ** 12),
        T=4 * YEAR,
        dt=10.0,
    ):

        num_bin = len(amp)

        # transform inputs
        f0 = f0 * T
        fdot = fdot * T * T
        fddot = fddot * T * T * T

        theta = np.pi / 2 - beta

        eplus = self.xp.zeros(3 * 3 * num_bin)
        ecross = self.xp.zeros(3 * 3 * num_bin)

        DPr = self.xp.zeros(num_bin)
        DPi = self.xp.zeros(num_bin)
        DCr = self.xp.zeros(num_bin)
        DCi = self.xp.zeros(num_bin)

        k = self.xp.zeros(3 * num_bin)

        amp = self.xp.asarray(amp)
        f0 = self.xp.asarray(f0)  # in mHz
        fdot = self.xp.asarray(fdot)
        fddot = self.xp.asarray(fddot)
        phi0 = self.xp.asarray(phi0)
        iota = self.xp.asarray(iota)
        psi = self.xp.asarray(psi)
        lam = self.xp.asarray(lam)
        theta = self.xp.asarray(theta)

        cosiota = self.xp.cos(iota)

        self.get_basis_tensors(
            eplus, ecross, DPr, DPi, DCr, DCi, k, amp, cosiota, psi, lam, theta, num_bin
        )

        data12 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
        data21 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
        data13 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
        data31 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
        data23 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
        data32 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)

        self.GenWave(
            data12,
            data21,
            data13,
            data31,
            data23,
            data32,
            eplus,
            ecross,
            f0,
            fdot,
            fddot,
            phi0,
            DPr,
            DPi,
            DCr,
            DCi,
            k,
            T,
            N,
            num_bin,
        )

        data12 = self.xp.fft.fft(data12.reshape(N, num_bin), axis=0).flatten()
        data21 = self.xp.fft.fft(data21.reshape(N, num_bin), axis=0).flatten()
        data13 = self.xp.fft.fft(data13.reshape(N, num_bin), axis=0).flatten()
        data31 = self.xp.fft.fft(data31.reshape(N, num_bin), axis=0).flatten()
        data23 = self.xp.fft.fft(data23.reshape(N, num_bin), axis=0).flatten()
        data32 = self.xp.fft.fft(data32.reshape(N, num_bin), axis=0).flatten()

        self.unpack_data_1(data12, data21, data13, data31, data23, data32, N, num_bin)

        df = 1 / T
        self.XYZ(
            data12, data21, data13, data31, data23, data32, f0, num_bin, N, dt, T, df
        )

        self.X = data12.reshape(N, num_bin).T
        self.Y = data21.reshape(N, num_bin).T
        self.Z = data13.reshape(N, num_bin).T
