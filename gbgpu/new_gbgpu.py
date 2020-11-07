import numpy as np

try:
    import cupy as xp
    from newfastgb import get_basis_tensors

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    print("no cupy")


class GBGPU(object):
    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu

        if self.use_gpu:
            self.xp = xp

        else:
            self.xp = np

    def run_wave(self, amp, f0, fdot, phi0, iota, psi, lam, beta):

        num_bin = len(amp)

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
        phi0 = self.xp.asarray(phi0)
        iota = self.xp.asarray(iota)
        psi = self.xp.asarray(psi)
        lam = self.xp.asarray(lam)
        beta = self.xp.asarray(beta)

        cosiota = self.xp.cos(iota)

        get_basis_tensors(
            eplus, ecross, DPr, DPi, DCr, DCi, k, amp, cosiota, psi, lam, beta, num_bin
        )

        breakpoint()
