from multiprocessing.sharedctypes import Value
import time
import warnings
from abc import ABC

import numpy as np

# import constants
from gbgpu.utils.constants import *
from gbgpu.utils.citation import *

# import Cython classes
from gbgpu_utils_cpu import get_ll as get_ll_cpu
from gbgpu_utils_cpu import fill_global as fill_global_cpu
from gbgpu_utils_cpu import direct_like_wrap as direct_like_wrap_cpu

try:
    from lisatools.sensitivity import A1TDISens

    tdi_available = True

except (ModuleNotFoundError, ImportError) as e:
    tdi_available = False
    warnings.warn("tdi module not found. No sensitivity information will be included.")

# import for GPU if available
try:
    import cupy as cp
    from gbgpu_utils import get_ll as get_ll_gpu
    from gbgpu_utils import fill_global as fill_global_gpu
    from gbgpu_utils import direct_like_wrap as direct_like_wrap_gpu

except (ModuleNotFoundError, ImportError):
    pass

from gbgpu.utils.utility import *

from lisatools.detector import EqualArmlengthOrbits, Orbits


class GBGPU(object):
    """Generate Galactic Binary Waveforms

    This class generates galactic binary waveforms in the frequency domain,
    in the form of LISA TDI channels X, A, and E. It generates waveforms in batches.
    It can also provide injection signals and calculate likelihoods in batches.
    These batches are run on GPUs or CPUs. When CPUs are used, all available threads
    are leveraged with OpenMP. To adjust the available threads, use ``OMP_NUM_THREADS``
    environmental variable or :func:`gbgpu.utils.set_omp_num_threads`.

    This class can generate waveforms for two different types of GB sources:

        * Circular Galactic binaries
        * Circular Galactic binaries with an eccentric third body (inherited)

    Args:
        use_gpu (bool, optional): If True, run on GPUs. Default is ``False``.

    Attributes:
        use_gpu (bool): Use GPU if True.
        get_basis_tensors (obj): Cython function.
        GenWave (obj): Cython function.
        GenWaveThird (obj): Cython function.
        unpack_data_1 (obj): Cython function.
        XYZ (obj): Cython function.
        num_bin (int): Number of binaries in the current calculation.
        N_max (int): Maximum points in a waveform based on maximum harmonic mode considered.
        start_inds (list of 1D int xp.ndarray): Start indices into data stream array. q - N/2.
        df (double): Fourier bin spacing.
        X_out, A_out, E_out (1D complex xp.ndarrays): X, A, or E channel TDI templates.
            Each array is a 2D complex array
            of shape (number of points, number of binaries) that is flattened. These can be
            accessed in python with the properties ``X``, ``A``, ``E``.
        N (int): Last N value used.
        d_d (double): <d|d> term in the likelihood.

    """

    def __init__(self, orbits: Orbits = None, use_gpu=False):

        self.use_gpu = use_gpu
        self.d_d = None
        self.orbits = orbits

    @property
    def xp(self):
        """CuPy or NumPy"""
        xp = np if not self.use_gpu else cp
        return xp

    @property
    def get_ll_func(self):
        """get_ll c func."""
        return get_ll_cpu if not self.use_gpu else get_ll_gpu

    @property
    def fill_global_func(self):
        """fill_global c func."""
        return fill_global_cpu if not self.use_gpu else fill_global_gpu

    @property
    def global_get_ll_func(self):
        """global_get_ll_func c func."""
        return direct_like_wrap_cpu if not self.use_gpu else direct_like_wrap_gpu

    @property
    def orbits(self) -> Orbits:
        """Orbits class."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        if orbits is None:
            self._orbits = EqualArmlengthOrbits()
        elif not isinstance(orbits, Orbits):
            raise ValueError(
                "Input orbits must be of type Orbits (from LISA Analysis Tools)"
            )
        else:
            self._orbits = orbits
        self.orbits.configure(linear_interp_setup=True)

    @property
    def citation(self):
        """Get citations for this class"""
        return zenodo + cornish_fastb + robson_triple

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
        *args,
        N=None,
        T=4 * YEAR,
        dt=10.0,
        oversample=1,
        tdi2=False,
    ):
        """Create waveforms in batches.

        This call creates the TDI templates in batches.

        The parameters and code below are based on an implementation of Fast GB
        in the LISA Data Challenges' ``ldc`` package.

        This class can be inherited to build fast waveforms for systems
        with additional astrophysical effects.

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (double or 1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (double or 1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            phi0 (double or 1D double np.ndarray): Initial phase angle of gravitational
                wave given in radians.
            iota (double or 1D double np.ndarray): Inclination of the Galactic binary
                orbit given in radians.
            psi (double or 1D double np.ndarray): Polarization angle of the Galactic
                binary orbit in radians.
            lam (double or 1D double np.ndarray): Ecliptic longitutude of the source
                given in radians.
            beta (double or 1D double np.ndarray): Ecliptic Latitude of the source
                given in radians. This is converted to the spherical polar angle.
            *args (tuple, optional): Flexible parameter to allow for a flexible
                number of argmuments when inherited by other classes.
                If running a circular Galactic binarys, ``args = ()``.
                If ``len(args) != 0``, then the inheriting class must have a
                ``prepare_additional_args`` method.
            N (int, optional): Number of points in waveform.
                This should be determined by the initial frequency, ``f0``. Default is ``None``.
                If ``None``, will use :func:`gbgpu.utils.utility.get_N` function to determine proper ``N``.
            T (double, optional): Observation time in seconds. Default is ``4 * YEAR``.
            dt (double, optional): Observation cadence in seconds. Default is ``10.0`` seconds.
            oversample(int, optional): Oversampling factor compared to the determined ``N``
                value. Final N will be ``oversample * N``. This is only used if N is
                not provided. Default is ``1``.
            tdi2 (bool, optional): If ``True``, produce the TDI channels for TDI 2nd-generation.
                If ``False``, produce TDI 1st-generation. Technically, the current TDI computation
                is not valid for generic LISA orbits, which are dealth with with 2nd-generation TDI,
                only those with an "equal-arm length" condition. Default is ``False``.

            Raises:
                ValueError: Length of ``*args`` is not 0 or 5.

        """

        # get number of observation points and adjust T accordingly
        N_obs = int(T / dt)
        T = N_obs * dt

        if T > self.orbits.t_base.max():
            raise ValueError(
                f"Observation time ({T}) is larger than length of time in orbital information ({self.orbits.t_base.max()})"
            )

        # if given scalar parameters, make sure at least 1D
        amp = np.atleast_1d(amp)
        f0 = np.atleast_1d(f0)
        fdot = np.atleast_1d(fdot)
        fddot = np.atleast_1d(fddot)
        phi0 = np.atleast_1d(phi0)
        iota = np.atleast_1d(iota)
        psi = np.atleast_1d(psi)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)

        # if circular base
        if len(args) == 0:
            add_args = ()

        else:
            if not hasattr(self, "prepare_additional_args"):
                raise ValueError(
                    "If providing more args than the base args, must be a class derived from GBGPU that has a 'prepare_additional_args' method."
                )

            add_args = self.prepare_additional_args(*args)

        # set N if it is not given based on timescales in the waveform
        if N is None:
            if hasattr(self, "special_get_N"):
                # take the original extra arguments
                N_temp = self.special_get_N(amp, f0, T, *args, oversample=oversample)
            else:
                N_temp = get_N(amp, f0, T, oversample=oversample)
            N = N_temp.max()

        # number of binaries is determined from length of amp array
        self.num_bin = num_bin = len(amp)

        # polar angle from ecliptic latitude
        theta = np.pi / 2 - beta

        # bin spacing
        self.df = df = 1 / T

        # instantiate GPU/CPU arrays

        # copy to GPU if needed
        amp = self.xp.asarray(amp.copy())
        f0 = self.xp.asarray(f0.copy())  # in mHz
        fdot = self.xp.asarray(fdot.copy())
        fddot = self.xp.asarray(fddot.copy())
        phi0 = self.xp.asarray(phi0.copy())
        iota = self.xp.asarray(iota.copy())
        psi = self.xp.asarray(psi.copy())
        lam = self.xp.asarray(lam.copy())
        theta = self.xp.asarray(theta.copy())

        cosiota = self.xp.cos(iota.copy())

        self.N = N

        # figure out start inds
        q_check = (f0 * T).astype(np.int32)
        # self.start_inds = (q_check - N / 2).astype(xp.int32)
        # self.start_inds = (q_check - N / 2).astype(xp.int32)

        cosiota = self.xp.cos(iota)

        # transfer frequency
        fstar = Clight / (self.orbits.armlength * 2 * np.pi)

        cosps, sinps = self.xp.cos(2.0 * psi), self.xp.sin(2.0 * psi)

        Aplus = amp * (1.0 + cosiota * cosiota)
        Across = -2.0 * amp * cosiota

        DP = Aplus * cosps - 1.0j * Across * sinps
        DC = -Aplus * sinps - 1.0j * Across * cosps

        # sky location basis vectors
        sinth, costh = self.xp.sin(theta), self.xp.cos(theta)
        sinph, cosph = self.xp.sin(lam), self.xp.cos(lam)
        u = self.xp.array([costh * cosph, costh * sinph, -sinth]).T[:, None, :]
        v = self.xp.array([sinph, -cosph, self.xp.zeros_like(cosph)]).T[:, None, :]
        k = self.xp.array([-sinth * cosph, -sinth * sinph, -costh]).T[:, None, :]

        # polarization tensors
        eplus = self.xp.matmul(v.transpose(0, 2, 1), v) - self.xp.matmul(
            u.transpose(0, 2, 1), u
        )
        ecross = self.xp.matmul(u.transpose(0, 2, 1), v) + self.xp.matmul(
            v.transpose(0, 2, 1), u
        )

        # time points evaluated
        tm = self.xp.linspace(0, T, num=N, endpoint=False)

        # get the spacecraft positions from orbits
        Ps = self._spacecraft(tm)

        # time domain information
        Gs, q = self._construct_slow_part(
            T,
            self.orbits.armlength,
            Ps,
            tm,
            f0,
            fdot,
            fddot,
            fstar,
            phi0,
            k,
            DP,
            DC,
            eplus,
            ecross,
            *add_args,
        )

        # transform to TDI observables
        XYZf, f_min = self._computeXYZ(T, Gs, f0, fdot, fddot, fstar, amp, q, tm)

        self.start_inds = self.kmin = self.xp.round(f_min / df).astype(int)
        self.start_inds = self.kmin = self.xp.round(f_min / df).astype(int)
        fctr = 0.5 * T / N

        # adjust for TDI2 if needed
        if tdi2:
            omegaL = 2 * np.pi * f0 * (self.orbits.armlength / Clight)
            tdi2_factor = 2.0j * self.xp.sin(2 * omegaL) * self.xp.exp(-2j * omegaL)
            fctr *= tdi2_factor

        if isinstance(fctr, float):
            fctr = self.xp.array([fctr])

        XYZf *= fctr[:, None, None]

        # we do not care about T right now
        self.AETf = self.xp.asarray(AET(XYZf[:, 0], XYZf[:, 1], XYZf[:, 2])).transpose(
            1, 0, 2
        )
        # setup waveforms for efficient GPU likelihood or global template building
        self.XYZf = XYZf

    @property
    def X_out(self):
        """X channel."""
        return self.XYZf[:, 0].T.flatten()

    @property
    def Y_out(self):
        """Y channel."""
        return self.XYZf[:, 1].T.flatten()

    @property
    def Z_out(self):
        """Z channel."""
        return self.XYZf[:, 2].T.flatten()

    @property
    def A_out(self):
        """A channel."""
        return self.AETf[:, 0].T.flatten()

    @property
    def E_out(self):
        """E channel."""
        return self.AETf[:, 1].T.flatten()

    @property
    def T_out(self):
        """T channel."""
        return self.AETf[:, 2].T.flatten()

    def _computeXYZ(self, T, Gs, f0, fdot, fddot, fstar, ampl, q, tm):
        """Compute TDI X, Y, Z from y_sr"""

        # get true frequency as a function of time
        f = (
            f0[:, None]
            + fdot[:, None] * tm[None, :]
            + 1 / 2 * fddot[:, None] * tm[None, :] ** 2
        )

        # compute transfer function
        omL = f / fstar
        SomL = self.xp.sin(omL)
        fctr = self.xp.exp(-1.0j * omL)
        fctr2 = 4.0 * omL * SomL * fctr / ampl[:, None]

        # Notes from LDC below

        ### I have factored out 1 - exp(1j*omL) and transformed to
        ### fractional frequency: those are in fctr2
        ### I have rremoved Ampl to reduce dynamical range, will restore it later

        Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        Ysl = Gs["32"] - Gs["12"] + (Gs["23"] - Gs["21"]) * fctr
        Zsl = Gs["13"] - Gs["23"] + (Gs["31"] - Gs["32"]) * fctr

        # time domain slow part
        XYZsl = fctr2[:, None, :] * self.xp.array([Xsl, Ysl, Zsl]).transpose(1, 0, 2)

        # frequency domain slow part
        XYZf_slow = ampl[:, None, None] * self.xp.fft.fft(XYZsl, axis=-1)

        # for testing
        # Xtry = 4.0*(self.G21 - self.G31 + (self.G12 - self.G13)*fctr)/self.ampl

        M = XYZf_slow.shape[2]  # len(XYZf_slow)
        XYZf = self.xp.fft.fftshift(XYZf_slow, axes=-1)

        # closest bin frequency
        f0 = (q - M / 2) / T  # freq = (q + self.xp.arange(M) - M/2)/T
        return XYZf, f0

    def _spacecraft(self, t):
        """Compute space craft positions as a function of time"""

        # output arrays
        P1 = self.orbits.get_pos(t, self.xp.full_like(t, 1).astype(self.xp.int32))
        P2 = self.orbits.get_pos(t, self.xp.full_like(t, 2).astype(self.xp.int32))
        P3 = self.orbits.get_pos(t, self.xp.full_like(t, 3).astype(self.xp.int32))
        return [P1, P2, P3]

    def _construct_slow_part(
        self,
        T,
        arm_length,
        Ps,
        tm,
        f0,
        fdot,
        fddot,
        fstar,
        phi0,
        k,
        DP,
        DC,
        eplus,
        ecross,
        *add_args,
    ):
        """Construct the time-domain function for the slow part of the waveform."""

        # these are the orbits (equal-arm lengths assumed)
        P1, P2, P3 = Ps
        r = dict()

        # unit vectors of constellation arms
        r["12"] = (P2 - P1) / arm_length  ## [3xNt]
        r["13"] = (P3 - P1) / arm_length
        r["23"] = (P3 - P2) / arm_length
        r["31"] = -r["13"]

        # wave propagation axis dotted with constellation unit vectors
        kdotr = dict()
        for ij in ["12", "13", "23"]:
            kdotr[ij] = self.xp.dot(k.squeeze(), r[ij].T)  ### should be size Nt
            kdotr[ij[-1] + ij[0]] = -kdotr[ij]

        # wave propagation axis dotted with spacecraft positions
        kdotP = self.xp.array(
            [self.xp.dot(k, P1.T), self.xp.dot(k, P2.T), self.xp.dot(k, P3.T)]
        )[:, :, 0].transpose(1, 0, 2)

        kdotP /= Clight

        Nt = len(tm)

        # delayed time at the spacecraft
        xi = tm - kdotP

        # instantaneous frequency of wave at the spacecraft at xi
        fi = (
            f0[:, None, None]
            + fdot[:, None, None] * xi
            + 1 / 2.0 * fddot[:, None, None] * xi**2
        )

        if hasattr(self, "shift_frequency"):
            # shift is performed in place to save memory
            fi[:] = self.shift_frequency(fi, xi, *add_args)

        # transfer frequency ratio
        fonfs = fi / fstar  # Ratio of true frequency to transfer frequency

        # LDC notes with '###'
        ### compute transfer f-n
        q = np.rint(f0 * T)  # index of nearest Fourier bin
        df = 2.0 * np.pi * (q / T)
        om = 2.0 * np.pi * f0

        ### The expressions below are arg2_i with om*kR_i factored out
        A = dict()
        for ij in ["12", "23", "31"]:
            aij = (
                self.xp.dot(eplus, r[ij].T) * r[ij].T * DP[:, None, None]
                + self.xp.dot(ecross, r[ij].T) * r[ij].T * DC[:, None, None]
            )
            A[ij] = aij.sum(axis=1)

        # below is information from the LDC about matching the original LDC.
        # The current code matches the time-domain-generated tempaltes in the LDC.

        # These are wfm->TR + 1j*TI in c-code

        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*fdot*(xi[0]**2)
        # arg2_2 = 2.0*np.pi*f0*xi[1] + phi0 - df*tm + np.pi*fdot*(xi[1]**2)
        # arg2_3 = 2.0*np.pi*f0*xi[2] + phi0 - df*tm + np.pi*fdot*(xi[2]**2)

        ### These (y_sr) reproduce exactly the FastGB results
        # self.y12 = 0.25*np.sin(arg12)/arg12 * np.exp(1.j*(arg12 + arg2_1)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y23 = 0.25*np.sin(arg23)/arg23 * np.exp(1.j*(arg23 + arg2_2)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y31 = 0.25*np.sin(arg31)/arg31 * np.exp(1.j*(arg31 + arg2_3)) * ( Dp31*self.DP + Dc31*self.DC )
        # self.y21 = 0.25*np.sin(arg21)/arg21 * np.exp(1.j*(arg21 + arg2_2)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y32 = 0.25*np.sin(arg32)/arg32 * np.exp(1.j*(arg32 + arg2_3)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y13 = 0.25*np.sin(arg13)/arg13 * np.exp(1.j*(arg13 + arg2_1)) * ( Dp31*self.DP + Dc31*self.DC )

        ### Those are corrected values which match the time domain results.
        ## om*kdotP_i singed out for comparison with another code.

        argS = (
            phi0[:, None, None]
            + (om[:, None, None] - df[:, None, None]) * tm[None, None, :]
            + np.pi * fdot[:, None, None] * (xi**2)
            + 1 / 3 * np.pi * fddot[:, None, None] * (xi**3)
        )

        if hasattr(self, "add_to_argS"):
            # performed in place to save memory
            argS[:] = self.add_to_argS(argS, f0, fdot, fddot, xi, *add_args)

        kdotP = om[:, None, None] * kdotP - argS

        # get Gs transfer functions
        Gs = dict()
        for ij, ij_sym, s in [
            ("12", "12", 0),
            ("23", "23", 1),
            ("31", "31", 2),
            ("21", "12", 1),
            ("32", "23", 2),
            ("13", "31", 0),
        ]:

            arg_ij = 0.5 * fonfs[:, s, :] * (1 + kdotr[ij])
            Gs[ij] = (
                0.25
                * self.xp.sin(arg_ij)
                / arg_ij
                * self.xp.exp(-1.0j * (arg_ij + kdotP[:, s]))
                * A[ij_sym]
            )
        ### Lines blow are extractions from another python code and from C-code in LDC
        # y = -0.5j*self.omL*A*sinc(args)*np.exp(-1.0j*(args + self.om*kq))
        # args = 0.5*self.omL*(1.0 - kn)
        # arg12 = 0.5*fonfs[0,:] * (1 + kdotr12)
        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*self.fdot*(xi[0]**2)  -> om*k.Ri
        # arg1 = 0.5*wfm->fonfs[i]*(1. + wfm->kdotr[i][j])
        # arg2 =  PI*2*f0*wfm->xi[i] + phi0 - df*t
        # sinc = 0.25*sin(arg1)/arg1
        # tran1r = aevol*(wfm->dplus[i][j]*wfm->DPr + wfm->dcross[i][j]*wfm->DCr)
        # tran1i = aevol*(wfm->dplus[i][j]*wfm->DPi + wfm->dcross[i][j]*wfm->DCi)
        # tran2r = cos(arg1 + arg2)
        # tran2i = sin(arg1 + arg2)
        # wfm->TR[i][j] = sinc*(tran1r*tran2r - tran1i*tran2i)
        # wfm->TI[i][j] = sinc*(tran1r*tran2i + tran1i*tran2r)
        return Gs, q

    @property
    def A(self):
        """return A channel reshaped based on number of binaries"""
        return self.A_out.reshape(self.N, self.num_bin).T

    @property
    def E(self):
        """return E channel reshaped based on number of binaries"""
        return self.E_out.reshape(self.N, self.num_bin).T

    @property
    def T(self):
        """return T channel reshaped based on number of binaries"""
        return self.T_out.reshape(self.N, self.num_bin).T

    @property
    def X(self):
        """return X channel reshaped based on number of binaries"""
        return self.X_out.reshape(self.N, self.num_bin).T

    @property
    def Y(self):
        """return Y channel reshaped based on number of binaries"""
        return self.Y_out.reshape(self.N, self.num_bin).T

    @property
    def Z(self):
        """return Z channel reshaped based on number of binaries"""
        return self.Z_out.reshape(self.N, self.num_bin).T

    @property
    def freqs(self):
        """Return frequencies associated with each signal"""
        freqs_out = (
            self.xp.arange(self.N)[None, :] + self.start_inds[:, None]
        ) * self.df
        return freqs_out

    def get_ll(
        self,
        params,
        data,
        psd,
        phase_marginalize=False,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        **kwargs,
    ):
        """Get batched log likelihood

        Generate the individual log likelihood for a batched set of Galactic binaries.
        This is also GPU/CPU agnostic.

        Args:
            params (2D double np.ndarrays): Parameters of all binaries to be calculated.
                The shape is ``(number of parameters, number of binaries)``.
            data (length 2 list of 1D or 2D complex128 xp.ndarrays): List of arrays representing the data
                stream. These should be CuPy arrays if running on the GPU, NumPy
                arrays if running on a CPU. The list should be [A channel, E channel].
                Should be 1D if only one data stream is analyzed. If 2D, shape is
                ``(number of data streams, data_length)``. If 2D,
                user must also provide ``data_index`` kwarg.
            psd (length 2 list of 1D or 2D double xp.ndarrays): List of arrays representing
                the power spectral density (PSD) in the noise.
                These should be CuPy arrays if running on the GPU, NumPy
                arrays if running on a CPU. The list should be [A channel, E channel].
                Should be 1D if only one PSD is analyzed. If 2D, shape is
                ``(number of PSDs, data_length)``. If 2D,
                user must also provide ``noise_index`` kwarg.
            phase_marginalize (bool, optional): If True, marginalize over the initial phase.
                Default is False.
            start_freq_ind (int, optional): Starting index into the frequency-domain data stream
                for the first entry of ``data``/``psd``. This is used if a subset of a full data stream
                is presented for the computation. If providing mutliple data streams in ``data``, this single
                start index value will apply to all of them.
            data_index (1D xp.int32 array, optional): If providing 2D ``data``, need to provide ``data_index``
                to indicate the data stream associated with each waveform for which the log-Likelihood
                is being computed. For example, if you have 100 binaries with 5 different data streams,
                ``data_index`` will be a length-100 xp.int32 array with values 0 to 4, indicating the specific
                data stream to use for each source.
                If ``None``, this will be filled with zeros and only analyzed with the first
                data stream given. Default is ``None``.
            noise_index (1D xp.int32 array, optional): If providing 2D ``psd``, need to provide ``noise_index``
                to indicate the PSD associated with each waveform for which the log-Likelihood
                is being computed. For example, if you have 100 binaries with 5 different PSDs,
                ``noise_index`` will be a length-100 xp.int32 array with values 0 to 4, indicating the specific
                PSD to use for each source.
                If ``None``, this will be filled with zeros and only analyzed with the first
                PSD given. Default is ``None``.
            **kwargs (dict, optional): Passes keyword arguments to the :func:`run_wave` method.

        Raises:
            TypeError: If data arrays are NumPy/CuPy while template arrays are CuPy/NumPy.

        Returns:
            1D double np.ndarray: Log likelihood values associated with each binary.

        """

        if self.d_d is None:
            raise ValueError(
                "self.d_d attribute must be set before computing log-Likelihood. This attribute is the data with data inner product (<d|d>)."
            )

        # produce TDI templates
        self.run_wave(*params, **kwargs)

        df = self.df

        # check if arrays are of same type
        if isinstance(data[0], self.xp.ndarray) is False:
            raise TypeError(
                "Make sure the data arrays are the same type as template arrays (cupy vs numpy)."
            )

        # make sure index information if provided properly
        if data[0].ndim == 1:
            if data_index is not None:
                raise ValueError("If inputing 1D data, cannot use data_index kwarg.")
            if noise_index is not None:
                raise ValueError("If inputing 1D data, cannot use noise_index kwarg.")

            data_length = data[0].shape[0]

        elif data[0].ndim == 2:
            data_length = data[0].shape[1]
            data = [dat.copy().flatten() for dat in data]

        if psd[0].ndim == 2:
            psd = [psd_i.copy().flatten() for psd_i in psd]

        # fill index values if not given
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)

        # check that index values are ready for computation
        assert len(data_index) == self.num_bin
        assert len(data_index) == len(noise_index)
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        assert data_index.max() * data_length <= len(data[0])
        assert noise_index.max() * data_length <= len(data[0])

        # initialize Likelihood terms <d|h> and <h|h>
        d_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        h_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)

        # shift start inds based on the starting index of the data stream
        start_inds_temp = (self.start_inds - start_freq_ind).astype(self.xp.int32)

        # get ll through C/CUDA
        self.get_ll_func(
            d_h,
            h_h,
            self.A_out,
            self.E_out,
            data[0],
            data[1],
            psd[0],
            psd[1],
            df,
            start_inds_temp,
            self.N,
            self.num_bin,
            data_index,
            noise_index,
            data_length,
        )

        if phase_marginalize:
            self.non_marg_d_h = d_h.copy()
            try:
                self.non_marg_d_h = self.non_marg_d_h.get()
            except AttributeError:
                pass

            d_h = self.xp.abs(d_h)

        # store these likelihood terms for later if needed
        self.h_h = h_h
        self.d_h = d_h

        # compute Likelihood
        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2 * d_h).real

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out

    def fill_global_template(
        self, group_index, templates, A, E, start_inds, N=None, start_freq_ind=0
    ):
        """Fill many global templates with waveforms

        This method takes already generated waveforms (``A, E, start_inds``)
        and their associated grouping index (``group_index``) and fills
        buffer tempalte arrays (``templates``).

        This method combines waveforms that have already been created.
        When a user does not have the waveforms in hand, they should
        use the :func:`generate_global_template` method.

        Args:
            group_index (1D double int32 xp.ndarray): Index indicating to which template each individual binary belongs.
            templates (3D complex128 xp.ndarray): Buffer array for template output to filled in place.
                The shape is ``(number of templates, 2, data_length)``. The ``2`` is
                for the ``A`` and ``E`` TDI channels in that order.
            A (1D or 2D complex128 xp.ndarray): TDI A channel template values for each individual binary.
                The shape if 2D is ``(number of binaries, N)''. In 1D, the array should be arranged so that
                it resembles ``(number of binaries, N).transpose().flatten()``.
                After running waveforms, this is how ``self.A_out`` is arranged.
            E (1D 2D complex128 xp.ndarray): TDI E channel template values for each individual binary.
                The shape if 2D is ``(number of binaries, N)''. In 1D, the array should be arranged so that
                it resembles ``(number of binaries, N).transpose().flatten()``.
                After running waveforms, this is how ``self.E_out`` is arranged.
            start_inds (1D int32 xp.ndarray): The start indices of each binary waveform
                in the full Fourier transform: ``int(f0/T) - N/2``.
            N (int, optional): The length of the A and E channels for each individual binary.
                When ``A`` and ``E`` are 1D, ``N`` must be given. Default is ``None``.
            start_freq_ind (int, optional): Starting index into the frequency-domain data stream
                for the first entry of ``templates``. This is used if a subset of a full data stream
                is presented for the computation.

        Raises:
            TypeError: If data arrays are NumPy/CuPy while tempalte arrays are CuPy/NumPy.
            ValueError: Inputs are not correctly provided.

        """

        # get shape of information
        total_groups, nchannels, data_length = templates.shape
        group_index = self.xp.asarray(group_index, dtype=self.xp.int32)
        num_bin = len(group_index)

        if nchannels < 2:
            raise ValueError("Calculates for A and E channels.")
        elif nchannels > 2:
            warnings.warn("Only calculating A and E channels here currently.")

        # check if arrays are of same type
        if isinstance(templates, self.xp.ndarray) is False:
            raise TypeError(
                "Make sure the data arrays are the same type as template arrays (cupy vs numpy)."
            )

        # prepare temporary buffers for C/CUDA
        # These are required to ensure the python memory order
        # is read properly in C/CUDA
        template_A = self.xp.zeros_like(
            templates[:, 0], dtype=self.xp.complex128
        ).flatten()
        template_E = self.xp.zeros_like(
            templates[:, 1], dtype=self.xp.complex128
        ).flatten()

        # shift start inds (see above)
        start_inds = (start_inds - start_freq_ind).astype(self.xp.int32)

        # check A, E, N inputs
        if A.ndim > 2 or E.ndim > 2:
            raise ValueError("A_in, E_in have maximum allowable dimension of 2.")
        elif A.ndim == 2:
            N = A.shape[1]
            assert E.ndim == 2
            # assumes the shape is the same as self.A
            A = A.T.flatten()
            E = E.T.flatten()

        elif A.ndim == 1:
            if N is None:
                raise ValueError(
                    "If providing a 1D flattened array for A and E, the N kwarg also needs to be provided."
                )

        # fill the templates in C/CUDA
        self.fill_global_func(
            template_A,
            template_E,
            A,
            E,
            start_inds,
            N,
            num_bin,
            group_index,
            data_length,
        )

        # read out to buffer arrays
        templates[:, 0] = template_A.reshape(total_groups, data_length)
        templates[:, 1] = template_E.reshape(total_groups, data_length)

    def generate_global_template(
        self,
        params,
        group_index,
        templates,
        start_freq_ind=0,
        **kwargs,
    ):
        """Generate global templates from binary parameters

        Generate waveforms in batches and then combine them into
        global fit templates. This method wraps :func:`fill_global_template`
        by building the waveforms first.

        Args:
            params (2D double np.ndarrays): Parameters of all binaries to be calculated.
                The shape is ``(number of parameters, number of binaries)``.
            group_index (1D double int32 xp.ndarray): Index indicating to which template each individual binary belongs.
            templates (3D complex128 xp.ndarray): Buffer array for template output to filled in place.
                The shape is ``(number of templates, 2, data_length)``. The ``2`` is
                for the ``A`` and ``E`` TDI channels in that order.
            start_freq_ind (int, optional): Starting index into the frequency-domain data stream
                for the first entry of ``templates``. This is used if a subset of a full data stream
                is presented for the computation.
            **kwargs (dict, optional): Passes keyword arguments to :func:`run_wave` function above.

        """

        # produce TDI templates
        self.run_wave(*params.T, **kwargs)
        self.fill_global_template(
            group_index,
            templates,
            self.A_out,
            self.E_out,
            self.start_inds,
            self.N,
            start_freq_ind=start_freq_ind,
        )
        return

    def inject_signal(self, *args, fmax=None, T=4.0 * YEAR, dt=10.0, **kwargs):
        """Inject a single signal

        Provides the injection of a single signal into a data stream with frequencies
        spanning from 0.0 to fmax with 1/T spacing (from Fourier transform).

        Args:
            *args (list, tuple, or 1D double np.array): Arguments to provide to
                :func:`run_wave` to build the TDI templates for injection.
            fmax (double, optional): Maximum frequency to use in data stream.
                If ``None``, will use ``1/(2 * dt)``.
                Default is ``None``.
            T (double, optional): Observation time in seconds. Default is ``4 * YEAR``.
            dt (double, optional): Observation cadence in seconds. Default is ``10.0`` seconds.
            **kwargs (dict, optional): Passes kwargs to :func:`run_wave`.

        Returns:
            Tuple of 1D np.ndarrays: NumPy arrays for the A channel and
                E channel: ``(A channel, E channel)``. Need to convert to CuPy if working
                on GPU.

        """

        # get binspacing
        if fmax is None:
            fmax = 1 / (2 * dt)

        if T > self.orbits.t_base.max():
            raise ValueError(
                f"Observation time ({T}) is larger than length of time in orbital information ({self.orbits.t_base.max()})"
            )

        # adjust inputs for run wave
        N_obs = int(T / dt)
        T = N_obs * dt
        kwargs["T"] = T
        kwargs["dt"] = dt
        self.df = df = 1 / T

        # create frequencies
        f = np.arange(0.0, fmax + df, df)
        num = len(f)

        # NumPy arrays for data streams of injections
        A_out = np.zeros(num, dtype=np.complex128)
        E_out = np.zeros(num, dtype=np.complex128)

        # build the templates
        self.run_wave(*args, **kwargs)

        # add each mode to the templates
        start = self.start_inds[0]

        # if using GPU, will return to CPU
        if self.use_gpu:
            A_temp = self.A_out.squeeze().get()
            E_temp = self.E_out.squeeze().get()

        else:
            A_temp = self.A_out.squeeze()
            E_temp = self.E_out.squeeze()

        # fill the data streams at the4 proper frqeuencies
        A_out[start.item() : start.item() + self.N] = A_temp
        E_out[start.item() : start.item() + self.N] = E_temp

        return A_out, E_out

    def _apply_parameter_transforms(self, params, parameter_transforms):
        """Apply parameter transformations to params for Information Matrix."""
        for ind_trans, trans in parameter_transforms.items():
            if isinstance(ind_trans, int):
                params[ind_trans] = trans(params[ind_trans])
            else:
                params[np.asarray(ind_trans)] = trans(*params[np.asarray(ind_trans)])
        return params

    def information_matrix(
        self,
        params,
        eps=1e-9,
        parameter_transforms={},
        inds=None,
        N=1024,
        psd_func=None,
        psd_kwargs={},
        easy_central_difference=False,
        return_gpu=False,
        **kwargs,
    ):
        """Get the information matrix for a batch.

        This function computes the Information matrix for a batch of Galactic binaries.
        It uses a 2nd order calculation for the derivative if ``easy_central_difference`` is ``False``:

        ..math:: \\frac{dh}{d\\lambda_i} = \\frac{-h(\\lambda_i + 2\\epsilon) + h(\\lambda_i - 2\\epsilon) + 8(h(\\lambda_i + \\epsilon) - h(\\lambda_i - \\epsilon))}{12\\epsilson}

        Otherwise, it will just calculate the derivate with a first-order central difference.

        This function maps all parameter values to 1. For example, if the square root of
        the diagonal of the associated covariance matrix is 1e-7 for the frequency
        parameter, then the standard deviation in the frequency is ``1e-7 * f0``. To
        properly use with covariance values not on the diagonal, they will have to be
        multipled by the parameters: :math:`C_{ij} \\vec{\\theta}_j`.

        Args:
            params (2D double np.ndarrays): Parameters of all binaries to be calculated.
                The shape is ``(number of parameters, number of binaries)``.
                See :func:`run_wave` for more information on the adjustable
                number of parameters when calculating for a third body.
            eps (double, optional): Step to take when calculating the derivative.
                The step is relative difference. Default is ``1e-9``.
            parameter_transforms (dict, optional): Dictionary containing the parameter transform
                functions. The keys in the dict should be the index associated with the parameter.
                The items should be the actual transform function. Default is no transforms (``{}``).
            inds (1D int np.ndarray, optional): Numpy array with the indexes of the parameters to
                test in the Information matrix. Default is ``None``. When it is not given, it defaults to
                all parameters.
            N (int, optional): Number of points for the waveform. Same as the ``N`` parameter in
                :func:`run_wave`. We recommend using higher ``N`` in the Information Matrix
                computation because of the numerical derivatives. Default is ``1024``.
            psd_func (object, optional): Function to compute the PSD for the A and E channels.
                Must take on argument: the frequencies as an xp.ndarray. When ``None``,
                it attemps to use the sensitivity functions from LISA Analysis Tools.
            psd_kwargs (dict, optional): Keyword arguments for the TDI noise generator. Default is ``None``.
            easy_central_difference (bool, optional): If ``True``, compute the derivatives with
                a first-order central difference computation. If ``False``, use the higher order
                derivative that computes two more waveforms during the derivative calculation.
                Default is ``False``.
            return_gpu (False, optional): If True and self.use_gpu is True, return Information
                matrices in cupy array. Default is False.

        Returns:
            3D xp.ndarray: Information Matrices for all binaries with shape: ``(number of binaries, number of parameters, number of parameters)``.

        Raises:
            ValueError: Step size issues.
            ModuleNotFoundError: LISA Analysis Tools package not available.
                Occurs when NOT providing ``psd_func`` kwarg.

        """

        if psd_func is None:
            # check if sensitivity information is available
            if not tdi_available:
                raise ModuleNotFoundError(
                    "Sensitivity curve information through LISA Analysis Tools is not available. Stock option for Information matrix will not work. Please install LISA Analysis Tools (pip install lisaanalysistools)."
                )
            psd_func = tdi.A1TDISens.get_Sn

        if N is None:
            raise ValueError("N must be provided up front for Infromation Matrix.")
        # put the N used here into kwargs
        kwargs["N"] = N

        params = np.atleast_2d(params)

        # get shape information
        num_params = len(params)
        num_bins = len(params[0])

        # fill inds if not given
        if inds is None:
            inds = np.arange(num_params)

        # setup holder arrays
        num_derivs = len(inds)
        info_matrix = self.xp.zeros((num_bins, num_derivs, num_derivs))

        # derivative buffer for waveforms
        dh = self.xp.zeros((num_bins, num_derivs, 2, N), self.xp.complex128)

        for i, ind in enumerate(inds):
            # 1 eps up derivative
            # map all the parameters to 1
            params_up_1 = np.ones_like(params)
            params_up_1[ind] += 1 * eps
            params_up_1 *= params
            params_up_1 = self._apply_parameter_transforms(
                params_up_1, parameter_transforms
            )
            self.run_wave(*params_up_1, **kwargs)
            h_I_up_eps = self.xp.asarray([self.A, self.E]).transpose((1, 0, 2))

            # 1 eps down derivative
            # map all the parameters to 1
            params_down_1 = np.ones_like(params)
            params_down_1[ind] -= 1 * eps
            params_down_1 *= params
            params_down_1 = self._apply_parameter_transforms(
                params_down_1, parameter_transforms
            )
            self.run_wave(*params_down_1, **kwargs)
            h_I_down_eps = self.xp.asarray([self.A, self.E]).transpose((1, 0, 2))

            # compute derivative and store
            if easy_central_difference:
                dh[:, i] = (h_I_up_eps - h_I_down_eps) / (2 * eps)

            else:
                # higher degree derivative computation

                # 2 eps up derivative
                # map all the parameters to 1
                params_up_2 = np.ones_like(params)
                params_up_2[ind] += 2 * eps
                params_up_2 *= params
                params_up_2 = self._apply_parameter_transforms(
                    params_up_2, parameter_transforms
                )
                self.run_wave(*params_up_2, **kwargs)
                h_I_up_2eps = self.xp.asarray([self.A, self.E]).transpose((1, 0, 2))
                if not np.all(self.start_inds == self.start_inds[0]):
                    raise ValueError(
                        "The user should decrease steps size (eps) because the frequency bins are changing during derivative calculation, which is not allowed."
                    )

                # 2 eps down derivative
                # map all the parameters to 1
                params_down_2 = np.ones_like(params)
                params_down_2[ind] -= 2 * eps
                params_down_2 *= params
                params_down_2 = self._apply_parameter_transforms(
                    params_down_2, parameter_transforms
                )
                self.run_wave(*params_down_2, **kwargs)
                h_I_down_2eps = self.xp.asarray([self.A, self.E]).transpose((1, 0, 2))

                dh[:, i] = (
                    -h_I_up_2eps + h_I_down_2eps + 8 * (h_I_up_eps - h_I_down_eps)
                ) / (12 * eps)

        # get frequencies for each binary
        freqs = self.freqs

        # get psd
        psd = self.xp.asarray(psd_func(freqs, **psd_kwargs))

        # compute Information matrix via inner products
        for i in range(num_derivs):
            for j in range(i, num_derivs):
                # innter product between derivatives
                inner_prod = (
                    4
                    * self.df
                    * self.xp.sum(
                        (dh[:, i].conj() * dh[:, j]).real / psd[:, None, :], axis=(1, 2)
                    )
                )

                # symmetry
                info_matrix[:, i, j] = inner_prod
                info_matrix[:, j, i] = info_matrix[:, i, j]

        # copy to cpu if needed
        if self.use_gpu and return_gpu is False:
            info_matrix = info_matrix.get()

        return info_matrix


class InheritGBGPU(GBGPU, ABC):
    """Inherit this class to expand on GBGPU waveforms.

    The required methods to be added are shown below.

    """

    @classmethod
    def prepare_additional_args(self, *args):
        """Prepare the arguments special to this class

        This function must take in the extra ``args`` input
        into :meth:`GBGPU.run_wave` and transform them as needed
        to input into the rest of the code. If using GPUs,
        this is where the parameters are copied to GPUs.

        Args:
            *args (tuple): Any additional args to be dealt with.

        Returns:
            Tuple: New args. In the rest of the code this is ``add_args``.

        """
        raise NotImplementedError

    @classmethod
    def special_get_N(
        self,
        amp,
        f0,
        T,
        *args,
        oversample=1,
    ):
        """Determine proper sampling rate in time domain for slow-part.

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            T (double): Observation time in seconds.
            *args (tuple): Args input for beyond-GBGPU functionality.
            oversample(int, optional): Oversampling factor compared to the determined ``N``
                value. Final N will be ``oversample * N``. This is only used if N is
                not provided. Default is ``1``.
        Returns:
            1D int32 xp.ndarray: Number of time-domain points recommended for each binary.

        """
        raise NotImplementedError

    def shift_frequency(self, fi, xi, *args):
        """Shift the evolution of the frequency in the slow part

        Args:
            fi (3D double xp.ndarray): Instantaneous frequencies of the
                wave before applying third-body effect at each spacecraft as a function of time.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            xi (3D double xp.ndarray): Time at each spacecraft.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            *args (tuple): Args returned from :meth:`prepare_additional_args`.

        Returns:
            3D double xp.ndarray: Updated frequencies with third-body effect.

        """
        raise NotImplementedError

    def add_to_argS(self, argS, f0, fdot, fddot, xi, *args):
        """Update ``argS`` in FastGB formalism for third-body effect

        ``argS`` is an effective phase that goes into ``kdotP`` in the construction
        of the slow part of the waveform. ``kdotP`` is then included directly
        in the transfer function. See :meth:`gbgpu.gbgpu.GBGPU._construct_slow_part`
        for the use of argS in the larger code.

        Args:
            argS (3D double xp.ndarray): Special phase evaluation that goes into ``kdotP``.
                Shape is ``(num binaries, 3 spacecraft, N)``.
            f0 (1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            xi (3D double xp.ndarray): Time at each spacecraft.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            T (double): Observation time in seconds.
            *args (tuple): Args returned from :meth:`prepare_additional_args`.

        Returns:
            3D double xp.ndarray: Updated ``argS`` with third-body effect

        """
        raise NotImplementedError
