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
        A2,
        omegabar,
        e,
        n2,
        T2,
        modes=np.array([2]),
        N=int(2 ** 12),
        T=4 * YEAR,
        dt=10.0,
    ):

        num_bin = len(amp)

        num_modes = len(modes)

        j_max = np.max(modes)

        # transform inputs
        f0 = f0 * T
        fdot = fdot * T * T
        fddot = fddot * T * T * T

        theta = np.pi / 2 - beta

        N_max = int(2 ** (j_max - 1) * N)  # get_NN(gb->params);

        eplus = self.xp.zeros(3 * 3 * num_bin)
        ecross = self.xp.zeros(3 * 3 * num_bin)

        DPr = self.xp.zeros(num_bin)
        DPi = self.xp.zeros(num_bin)
        DCr = self.xp.zeros(num_bin)
        DCi = self.xp.zeros(num_bin)

        k = self.xp.zeros(3 * num_bin)

        data12 = self.xp.zeros(num_bin * N_max, dtype=self.xp.complex128)
        data21 = self.xp.zeros(num_bin * N_max, dtype=self.xp.complex128)
        data13 = self.xp.zeros(num_bin * N_max, dtype=self.xp.complex128)
        data31 = self.xp.zeros(num_bin * N_max, dtype=self.xp.complex128)
        data23 = self.xp.zeros(num_bin * N_max, dtype=self.xp.complex128)
        data32 = self.xp.zeros(num_bin * N_max, dtype=self.xp.complex128)

        self.X = self.xp.zeros((num_bin, N_max), dtype=self.xp.complex128)
        self.Y = self.xp.zeros((num_bin, N_max), dtype=self.xp.complex128)
        self.Z = self.xp.zeros((num_bin, N_max), dtype=self.xp.complex128)

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

        A2 = self.xp.asarray(A2)
        omegabar = self.xp.asarray(omegabar)
        e = self.xp.asarray(e)
        n2 = self.xp.asarray(n2)
        T2 = self.xp.asarray(T2)

        N_base = N

        for j in modes:

            N = int(2 ** (j - 1) * N_base)

            self.get_basis_tensors(
                eplus,
                ecross,
                DPr,
                DPi,
                DCr,
                DCi,
                k,
                amp,
                cosiota,
                psi,
                lam,
                theta,
                e,
                j,
                num_bin,
            )

            breakpoint()

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
                A2,
                omegabar,
                e,
                n2,
                T2,
                DPr,
                DPi,
                DCr,
                DCi,
                k,
                T,
                N,
                j,
                num_bin,
            )

            data12 = data12.reshape(N_max, num_bin)
            data21 = data21.reshape(N_max, num_bin)
            data13 = data13.reshape(N_max, num_bin)
            data31 = data31.reshape(N_max, num_bin)
            data23 = data23.reshape(N_max, num_bin)
            data32 = data32.reshape(N_max, num_bin)

            data12[:N] = self.xp.fft.fft(data12[:N], axis=0)
            data21[:N] = self.xp.fft.fft(data21[:N], axis=0)
            data13[:N] = self.xp.fft.fft(data13[:N], axis=0)
            data31[:N] = self.xp.fft.fft(data31[:N], axis=0)
            data23[:N] = self.xp.fft.fft(data23[:N], axis=0)
            data32[:N] = self.xp.fft.fft(data32[:N], axis=0)

            data12 = data12.flatten()
            data21 = data21.flatten()
            data13 = data13.flatten()
            data31 = data31.flatten()
            data23 = data23.flatten()
            data32 = data32.flatten()

            self.unpack_data_1(
                data12, data21, data13, data31, data23, data32, N, num_bin
            )

            df = 1 / T
            self.XYZ(
                data12,
                data21,
                data13,
                data31,
                data23,
                data32,
                f0,
                num_bin,
                N,
                dt,
                T,
                df,
            )

            self.X += data12.reshape(N_max, num_bin).T
            self.Y += data21.reshape(N_max, num_bin).T
            self.Z += data13.reshape(N_max, num_bin).T
