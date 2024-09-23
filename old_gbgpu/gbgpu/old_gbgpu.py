import time
import warnings

import numpy as np

# import constants
from gbgpu.utils.constants import *

# import Cython classes
from newfastgb_cpu import get_basis_tensors as get_basis_tensors_cpu
from newfastgb_cpu import GenWave as GenWave_cpu
from newfastgb_cpu import unpack_data_1 as unpack_data_1_cpu
from newfastgb_cpu import XYZ as XYZ_cpu
from newfastgb_cpu import get_ll as get_ll_cpu
from newfastgbthird_cpu import GenWaveThird as GenWave_third_cpu
from newfastgb_cpu import fill_global as fill_global_cpu
from newfastgb_cpu import direct_like_wrap as direct_like_wrap_cpu

try:
    from lisatools import sensitivity as tdi

    tdi_available = True

except (ModuleNotFoundError, ImportError) as e:
    tdi_available = False
    warnings.warn("tdi module not found. No sensitivity information will be included.")

# import for GPU if available
try:
    import cupy as xp
    from newfastgb import (
        get_basis_tensors,
        GenWave,
        unpack_data_1,
        XYZ,
        get_ll,
        fill_global,
        direct_like_wrap,
    )
    from newfastgbthird import GenWaveThird as GenWave_third

except (ModuleNotFoundError, ImportError):
    import numpy as xp

from gbgpu.utils.utility import get_N


class GBGPU(object):
    """Generate Galactic Binary Waveforms

    This class generates galactic binary waveforms in the frequency domain,
    in the form of LISA TDI channels X, A, and E. It generates waveforms in batches.
    It can also provide injection signals and calculate likelihoods in batches.
    These batches are run on GPUs or CPUs. When CPUs are used, all available threads
    are leveraged with OpenMP.

    This class can generate waveforms for four different types of GB sources:

        * Circular Galactic binaries
        * Eccentric Galactic binaries (see caveats below)
        * Circular Galactic binaries with an eccentric third body
        * Eccentric Galactic binaries with an eccentric third body

    The class determines which waveform is desired based on the number of argmuments
    input by the user (see the *args description below). The eccentric inner binary
    is only roughly valid. It uses a bessel function expansion to get the relative
    amplitudes of various modes (number and index of modes is a user defined quantity).
    Therefore, the inner eccentric binaries are only valid at low eccentricities.
    The eccentricity is also not evolved over time in the current implementation.


    Args:
        shift_ind (int, optional): How many points to shift towards lower frequencies
            when calculating the likelihood. This helps to adjust for likelihoods
            that are calculated e.g. with the right summation rule and removing
            the DC component. Default is 2 for right summation and DC removal.
        use_gpu (bool, optional): If True, run on GPUs. Default is False.

    Attributes:
        xp (obj): NumPy if on CPU. CuPy if on GPU.
        use_gpu (bool): Use GPU if True.
        shift_ind (int): Indices to shift during likelihood calculation. See argument string above.
        get_basis_tensors (obj): Cython function.
        GenWave (obj): Cython function.
        GenWaveThird (obj): Cython function.
        unpack_data_1 (obj): Cython function.
        XYZ (obj): Cython function.
        get_ll_func (obj): Cython function.
        num_bin (int): Number of binaries in the current calculation.
        N_max (int): Maximum points in a waveform based on maximum harmonic mode considered.
        start_inds (list of 1D int xp.ndarray): Start indices into data stream array. q - N/2.
        df (double): Fourier bin spacing.
        X_out, A_out, E_out (list of 1D complex xp.ndarrays): X, A, or E channel TDI templates.
            This list is over the modes examined. Within each list entry is a 2D complex array
            of shape (number of points, number of binaries) that is flattened. These can be
            accessed in python with the properties :code:`X`, :code:`A`, :code:`E`.
        Ns (list): List of the number of points in each mode examined.
        d_d (double): <d|d> term in the likelihood.
        injection_params (tuple, list or 1D double array): last set of params used
            for injection (TODO: improve this method)
        running_d_d (bool): Is the likelihood currently run on injeciton. This needs to
            be improved with injection_params above.

    """

    def __init__(self, shift_ind=2, use_gpu=False):

        self.use_gpu = use_gpu
        self.shift_ind = shift_ind

        # setup Cython/C++/CUDA calls based on if using GPU
        if self.use_gpu:
            self.xp = xp
            self.get_basis_tensors = get_basis_tensors
            self.GenWave = GenWave
            self.GenWaveThird = GenWave_third
            self.unpack_data_1 = unpack_data_1
            self.XYZ = XYZ
            self.get_ll_func = get_ll
            self.fill_global_func = fill_global
            self.global_get_ll_func = direct_like_wrap

        else:
            self.xp = np
            self.get_basis_tensors = get_basis_tensors_cpu
            self.GenWave = GenWave_cpu
            self.GenWaveThird = GenWave_third_cpu
            self.unpack_data_1 = unpack_data_1_cpu
            self.XYZ = XYZ_cpu
            self.get_ll_func = get_ll_cpu
            self.fill_global_func = fill_global_cpu
            self.global_get_ll_func = direct_like_wrap_cpu

        self.d_d = None
        self.running_d_d = False

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
        modes=np.array([2]),
        N=None,
        T=4 * YEAR,
        dt=10.0,
        oversample=1,
    ):
        """Create waveforms in batches.

        This call actually creates the TDI templates in batches. It handles all
        four cases given above based on the number of *args provided by the user.

        The parameters and code below are based on an implementation by Travis Robson
        for the paper `arXiv:1806.00500 <https://arxiv.org/pdf/1806.00500.pdf>`_.

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (double or 1D double np.ndarray): Initial time derivative of the
                frequency given as Hz^2.
            fddot (double or 1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz^3.
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
                number of argmuments. If running a circular Galactic binarys, :code:`args = ()`.
                If running an eccentric binary, :code:`args = (e1, beta1)`. If running a
                circular inner binary with an eccentric third body,
                :code:`args = (A2, omegabar, e2, P2, T2)`. If running an eccentric
                inner binary and eccentric third body,
                :code:`args = (e1, beta1, A2, omegabar, e2, P2, T2)`.
            e1 (double or 1D double np.ndarray): Eccentricity of the inner binary.
                The code can handle e1 = 0.0. However, it is recommended to not input e1
                if circular orbits are desired.
            beta1 (double or 1D double np.ndarray): TODO: fill in.
            A2 (double or 1D double np.ndarray): Special amplitude parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            omegabar (double or 1D double np.ndarray): Special angular frequency parameter related to the
                line-of-site velocity for the third body orbit as defined in the paper
                given in the description above.
            e2 (double or 1D double np.ndarray): Eccentricity of the third body orbit.
            P2 (double or 1D double np.ndarray): Period of the third body orbit in Years.
            T2 (double or 1D double np.ndarray): Time of pericenter passage of the third body in Years.
                This parameter is effectively a constant of integration.
            modes (int or 1D int np.ndarray, optional): j modes to use in the bessel function expansion
                for the eccentricity of the inner binary orbit. Default is np.array([2]) for
                the main mode.
            N (int, optional): Number of points to produce for the base j=1 mode. Therefore,
                with the default j = 2 mode, the waveform will be 2 * N in length.
                This should be determined by the initial frequency, f0. Default is None.
                If None, will use a function to determine proper N.
            T (double, optional): Observation time in seconds. Default is 4 years.
            dt (double, optional): Observation cadence in seconds. Default is 10.0 seconds.
            oversample(int, optional): Oversampling factor compared to the determined :code:`N`
                value. Final N will be :code:`oversample * N`. This is only used if N is
                not provided. Default is 1.

            Raises:
                ValueError: Length of *args is not 0, 2, 5, or 7.

        """

        N_obs = int(T / dt)
        T = N_obs * dt

        # if given scalar parameters, make sure at least 1D
        modes = np.atleast_1d(modes)

        amp = np.atleast_1d(amp)
        f0 = np.atleast_1d(f0)
        fdot = np.atleast_1d(fdot)
        fddot = np.atleast_1d(fddot)
        phi0 = np.atleast_1d(phi0)
        iota = np.atleast_1d(iota)
        psi = np.atleast_1d(psi)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)

        # if eccentric
        if len(args) == 2:
            e1, beta1 = args

            run_ecc = True
            run_third = False

            P2 = None

        # if circular plus third body
        elif len(args) == 5:
            # set eccentricity to zero for inner binary
            e1 = np.full_like(amp, 0.0)
            beta1 = np.full_like(amp, 0.0)

            A2, omegabar, e2, P2, T2 = args

            run_ecc = False
            run_third = True

        # if eccentric plus third body
        elif len(args) == 7:
            e1, beta1, A2, omegabar, e2, P2, T2 = args

            run_ecc = True
            run_third = True

            P2 = None

        # if just circular
        elif len(args) == 0:
            # set eccentricity to zero
            e1 = np.full_like(amp, 0.0)
            beta1 = np.full_like(amp, 0.0)

            run_ecc = False
            run_third = False

            P2 = None
        else:
            raise ValueError(
                "Wrong number of extra arguments. Needs to be 2 for eccentric inner binary, 5 for circular inner binary and a third body, or 7 for eccentric inner and third body."
            )

        if N is None:
            N_temp = self._get_N(amp, f0, T, oversample=oversample, P2=P2)
            N = N_temp.max()

        # cast to 1D if given scalar
        e1 = np.atleast_1d(e1)
        beta1 = np.atleast_1d(beta1)

        # number of binaries is determined from length of amp array
        self.num_bin = num_bin = len(amp)

        num_modes = len(modes)

        # maximum mode index
        j_max = np.max(modes)

        # transform inputs
        f0 = f0 * T
        fdot = fdot * T * T
        fddot = fddot * T * T * T
        theta = np.pi / 2 - beta

        # maximum number of points in waveform
        self.N_max = N_max = int(2 ** (j_max - 1) * N)  # get_NN(gb->params);

        # start indices into frequency array of a full set of Fourier bins
        self.start_inds = []

        # bin spacing
        self.df = df = 1 / T

        # instantiate GPU/CPU arrays

        # polarization matrices
        eplus = self.xp.zeros(3 * 3 * num_bin)
        ecross = self.xp.zeros(3 * 3 * num_bin)

        # transfer information
        DPr = self.xp.zeros(num_bin)
        DPi = self.xp.zeros(num_bin)
        DCr = self.xp.zeros(num_bin)
        DCi = self.xp.zeros(num_bin)

        # sky location arrays
        k = self.xp.zeros(3 * num_bin)

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

        e1 = self.xp.asarray(e1.copy())
        beta1 = self.xp.asarray(beta1.copy())

        cosiota = self.xp.cos(iota.copy())

        # do things if running a third body
        if run_third:

            # cast to 1D if scalar
            A2 = np.atleast_1d(A2).copy()
            omegabar = np.atleast_1d(omegabar).copy()
            e2 = np.atleast_1d(e2).copy()
            P2 = np.atleast_1d(P2).copy()
            T2 = np.atleast_1d(T2).copy()

            # get mean anomaly
            n2 = 2 * np.pi / (P2 * YEAR)
            T2 *= YEAR

            # copy to GPU if needed
            A2 = self.xp.asarray(A2)
            omegabar = self.xp.asarray(omegabar)
            e2 = self.xp.asarray(e2)
            n2 = self.xp.asarray(n2)
            T2 = self.xp.asarray(T2)

        # base N value
        N_base = N

        # set up lists to hold output waveforms
        self.X_out = []
        self.A_out = []
        self.E_out = []

        # get lengths of the output waveforms for each mode
        self.Ns = []

        # loop over modes
        for j in modes:

            # specific number of samples for this mode
            N = int(2 ** (j - 1) * N_base)
            self.Ns.append(N)

            # allocate arrays to hold data based on N
            data12 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
            data21 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
            data13 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
            data31 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
            data23 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)
            data32 = self.xp.zeros(num_bin * N, dtype=self.xp.complex128)

            # figure out start inds
            q_check = (f0 * j / 2.0).astype(np.int32)
            self.start_inds.append((q_check - N / 2).astype(xp.int32))

            # get the basis tensors
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
                e1,
                beta1,
                j,
                num_bin,
            )

            # Generate the TD information based on using a third body or not
            if run_third:
                self.GenWaveThird(
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
                    e2,
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
            else:
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
                    j,
                    num_bin,
                )

            # prepare data for FFT
            data12 = data12.reshape(N, num_bin)
            data21 = data21.reshape(N, num_bin)
            data13 = data13.reshape(N, num_bin)
            data31 = data31.reshape(N, num_bin)
            data23 = data23.reshape(N, num_bin)
            data32 = data32.reshape(N, num_bin)

            # perform FFT
            data12[:N] = self.xp.fft.fft(data12[:N], axis=0)
            data21[:N] = self.xp.fft.fft(data21[:N], axis=0)
            data13[:N] = self.xp.fft.fft(data13[:N], axis=0)
            data31[:N] = self.xp.fft.fft(data31[:N], axis=0)
            data23[:N] = self.xp.fft.fft(data23[:N], axis=0)
            data32[:N] = self.xp.fft.fft(data32[:N], axis=0)

            # flatten data for input back in C
            data12 = data12.flatten()
            data21 = data21.flatten()
            data13 = data13.flatten()
            data31 = data31.flatten()
            data23 = data23.flatten()
            data32 = data32.flatten()

            # prepare the data for TDI calculation
            self.unpack_data_1(
                data12, data21, data13, data31, data23, data32, N, num_bin
            )

            # get TDIs
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
                j,
            )

            # add to lists
            self.X_out.append(data12)
            self.A_out.append(data21)
            self.E_out.append(data13)

    def _get_N(self, amp, f0, Tobs, oversample=1, P2=None):
        """Determine proper sampling in time domain."""
        return get_N(amp, f0, Tobs, oversample=oversample, P2=P2)

    @property
    def X(self):
        """return X channel reshaped based on number of binaries"""
        return [temp.reshape(N, self.num_bin).T for temp, N in zip(self.X_out, self.Ns)]

    @property
    def A(self):
        """return A channel reshaped based on number of binaries"""
        return [temp.reshape(N, self.num_bin).T for temp, N in zip(self.A_out, self.Ns)]

    @property
    def E(self):
        """return E channel reshaped based on number of binaries"""
        return [temp.reshape(N, self.num_bin).T for temp, N in zip(self.E_out, self.Ns)]

    @property
    def freqs(self):
        """Return frequencies associated with each signal"""
        freqs_out = []
        for start_inds, N in zip(self.start_inds, self.Ns):
            freqs_temp = self.xp.zeros((len(start_inds), N))
            for i, start_ind in enumerate(start_inds):
                if isinstance(start_ind, self.xp.ndarray):
                    start_ind = start_ind.item()

                freqs_temp[i] = self.xp.arange(start_ind, start_ind + N) * self.df
            freqs_out.append(freqs_temp)
        return freqs_out

    def get_ll(
        self,
        params,
        data,
        noise_factor,
        calc_d_d=False,
        phase_marginalize=False,
        start_freq_ind=0,
        **kwargs,
    ):
        """Get batched log likelihood

        Generate the log likelihood for a batched set of Galactic binaries. This is
        also GPU/CPU agnostic.

        Args:
            params (list, tuple or array of 1D double np.ndarrays): Array-like object containing
                the parameters of all binaries to be calculated. The shape is
                (number of parameters, number of binaries).
            data (length 2 list of 1D complex128 xp.ndarrays): List of arrays representing the data
                stream. These should be CuPy arrays if running on the GPU, NumPy
                arrays if running on a CPU. The list should be [A channel, E channel].
            noise_factor (length 2 list of 1D double xp.ndarrays): List of arrays representing
                the noise factor for weighting. This is typically something like 1/PSD(f) * sqrt(df).
                These should be CuPy arrays if running on the GPU, NumPy
                arrays if running on a CPU. The list should be [A channel, E channel].
            phase_marginalize (bool, optional): If True, marginalize over the initial phase.
                Default is False.
            **kwargs (dict, optional): Passes keyword arguments to run_wave function above.

        Raises:
            TypeError: If data arrays are NumPy/CuPy while tempalte arrays are CuPy/NumPy.

        Returns:
            1D double np.ndarray: Log likelihood values associated with each binary.

        """

        # TODO: fix how this is dealt with
        if (calc_d_d or self.d_d is None) and self.running_d_d is False:
            self.running_d_d = True
            self.get_ll(
                self.injection_params, data, noise_factor, calc_d_d=False, **kwargs
            )
            self.running_d_d = False
            # now sets self.d_d inside likelihood function

        # produce TDI templates
        self.run_wave(*params, **kwargs)

        # check if arrays are of same type
        if isinstance(data[0], self.xp.ndarray) is False:
            raise TypeError(
                "Make sure the data arrays are the same type as template arrays (cupy vs numpy)."
            )

        d_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        h_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)

        # calculate each mode separately
        # ASSUMES MODES DO NOT OVERLAP AT ALL
        # makes the inner product the sum of products over modes
        for X, A, E, start_inds, N in zip(
            self.X_out, self.A_out, self.E_out, self.start_inds, self.Ns
        ):
            d_h_temp = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
            h_h_temp = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
            # shift start inds (see above)
            start_inds = (start_inds - start_freq_ind - self.shift_ind).astype(
                self.xp.int32
            )

            # get ll
            self.get_ll_func(
                d_h_temp,
                h_h_temp,
                A,
                E,
                data[0],
                data[1],
                noise_factor[0],
                noise_factor[1],
                start_inds,
                N,
                self.num_bin,
            )

            d_h += d_h_temp
            h_h += h_h_temp

        if phase_marginalize:
            self.non_marg_d_h = d_h.copy()
            try:
                self.non_marg_d_h = self.non_marg_d_h.get()
            except AttributeError:
                pass

            d_h = self.xp.abs(d_h)

        self.h_h = h_h
        self.d_h = d_h

        if self.running_d_d:
            self.d_d = self.h_h.copy()

        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2 * d_h).real

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out

    def generate_global_template(
        self, params, group_index, templates, start_freq_ind=0, min_ind=None, **kwargs,
    ):
        """Get batched log likelihood for global fit

        Generate the log likelihood for a batched set of Galactic binaries. This is
        also GPU/CPU agnostic.

        Args:
            params (list, tuple or array of 1D double np.ndarrays): Array-like object containing
                the parameters of all binaries to be calculated. The shape is
                (number of parameters, number of binaries).
            data (length 2 list of 1D complex128 xp.ndarrays): List of arrays representing the data
                stream. These should be CuPy arrays if running on the GPU, NumPy
                arrays if running on a CPU. The list should be [A channel, E channel].
            noise_factor (length 2 list of 1D double xp.ndarrays): List of arrays representing
                the noise factor for weighting. This is typically something like 1/PSD(f) * sqrt(df).
                These should be CuPy arrays if running on the GPU, NumPy
                arrays if running on a CPU. The list should be [A channel, E channel].
            **kwargs (dict, optional): Passes keyword arguments to run_wave function above.

        Raises:
            TypeError: If data arrays are NumPy/CuPy while tempalte arrays are CuPy/NumPy.

        Returns:
            1D double np.ndarray: Log likelihood values associated with each binary.

        """

        total_groups, nchannels, data_length = templates.shape
        ndim = params.shape[1]
        group_index = self.xp.asarray(group_index, dtype=self.xp.int32)

        if nchannels < 2:
            raise ValueError("Calculates for A and E channels.")
        elif nchannels > 2:
            warnings.warn("Only calculating A and E channels here currently.")

        # produce TDI templates
        self.run_wave(*params.T, **kwargs)

        # check if arrays are of same type
        if isinstance(templates, self.xp.ndarray) is False:
            raise TypeError(
                "Make sure the data arrays are the same type as template arrays (cupy vs numpy)."
            )

        template_A = self.xp.zeros_like(
            templates[:, 0], dtype=self.xp.complex128
        ).flatten()
        template_E = self.xp.zeros_like(
            templates[:, 1], dtype=self.xp.complex128
        ).flatten()
        # calculate each mode separately
        # ASSUMES MODES DO NOT OVERLAP AT ALL
        # makes the inner product the sum of products over modes
        for X, A, E, start_inds, N in zip(
            self.X_out, self.A_out, self.E_out, self.start_inds, self.Ns
        ):
            # shift start inds (see above)
            start_inds = (start_inds - start_freq_ind - self.shift_ind).astype(
                self.xp.int32
            )

            # get ll
            self.fill_global_func(
                template_A,
                template_E,
                A,
                E,
                start_inds,
                N,
                self.num_bin,
                group_index,
                data_length,
            )

        templates[:, 0] = template_A.reshape(total_groups, data_length)
        templates[:, 1] = template_E.reshape(total_groups, data_length)

        return

    def inject_signal(
        self, *args, fmax=None, T=4.0 * YEAR, dt=10.0, noise_factor=True, **kwargs
    ):
        """Inject a single signal

        Provides the injection of a single signal into a data stream with frequencies
        spanning from 0.0 to fmax with 1/T spacing.

        Args:
            *args (list, tuple, or 1D double np.array): Arguments to provide to
                run_wave to build the TDI templates for injection.
            fmax (double, optional): Maximum frequency to use in data stream.
                Default is 1e-1.
            T (double, optional): Observation time in seconds. Default is 4 years.
            **kwargs (dict, optional): Passes kwargs to run_wave.

        Returns:
            Tuple of 1D np.ndarrays: NumPy arrays for the A channel and
                E channel: (A channel, E channel). Need to conver to CuPy if working
                on GPU.

        """

        # get binspacing
        if fmax is None:
            fmax = 1 / (2 * dt)

        N_obs = int(T / dt)
        T = N_obs * dt
        kwargs["T"] = T
        kwargs["dt"] = dt
        df = 1 / T

        # create frequencies
        f = np.arange(0.0, fmax + df, df)
        num = len(f)

        # NumPy arrays for data streams of injections
        A_out = np.zeros(num, dtype=np.complex128)
        E_out = np.zeros(num, dtype=np.complex128)

        # build the templates
        self.run_wave(*args, **kwargs)

        self.injection_params = args

        # add each mode to the templates
        for X, A, E, start_inds, N in zip(
            self.X_out, self.A_out, self.E_out, self.start_inds, self.Ns
        ):
            start = start_inds[0]

            # if using GPU, will return to CPU
            if self.use_gpu:
                A_temp = A.squeeze().get()
                E_temp = E.squeeze().get()

            else:
                A_temp = A.squeeze()
                E_temp = E.squeeze()

            # fill the data streams at the4 proper frqeuencies
            A_out[start.item() : start.item() + N] = A_temp
            E_out[start.item() : start.item() + N] = E_temp

        return A_out, E_out

    def fisher(
        self,
        params,
        eps=1e-9,
        parameter_transforms={},
        inds=None,
        N=1024,
        psd_kwargs={},
        return_gpu=False,
        **kwargs,
    ):
        """Get the fisher matrix for a batch.

        This function computes the Fisher matrix for a batch of galactic binaries.
        It cannot handle inner binary eccentricity yet. It can handle an eccentric
        third body.

        It uses a 2nd order calculation for the derivative:

        ..math:: \frac{dh}{d\lambda_i} = \frac{-h(\lambda_i + 2\epsilon) + h(\lambda_i - 2\epsilon) + 8(h(\lambda_i + \epsilon) - h(\lambda_i - \epsilon))}{12\epsilson}

        Args:
            params (2D double np.ndarray): 2D array with the parameter values of the batch.
                The shape should be (number of parameters, number of binaries).
                See :class:`gbgpu.gbgpu.GBGPU.run_Wave` for more information on the adjustable
                number of parameters when calculating for a third body.
            eps (double, optional): Step to take when calculating the derivative.
                Default is 1e-9.
            parameter_transforms (dict, optional): Dictionary containing the parameter transform
                functions. The keys in the dict should be the index associated with the parameter.
                The items should be the actual transform function. Default is no transforms ({}).
            inds (1D int np.ndarray, optional): Numpy array with the indexes of the parameters to
                test in the Fisher matrix. Default is None. When it is not given, it defaults to
                all parameters.
            N (int, optional): Number of points to produce for the base j=1 mode. Therefore,
                with the default j = 2 mode, the waveform will be 2 * N in length.
                This should be determined by the initial frequency, f0. In the future,
                this may be implemented. Default is 1024.
            psd_kwargs (dict, optional): Keyword arguments for the TDI noise generator
                from tdi.py (noisepsd_AE). Default is None.
            return_gpu (False, optional): If True and self.use_gpu is True, return fisher
                matrices in cupy array. Default is False.

        """
        # check if sensitivity information is available
        if not tdi_available:
            raise NameError(
                "tdi module is not available. Stock option for Fisher matrix will not work."
            )

        kwargs["N"] = N

        num_params = len(params)
        num_bins = len(params[0])

        # fill inds if not given
        if inds is None:
            inds = np.arange(num_params)

        # setup holder arrays
        num_derivs = len(inds)
        fish_matrix = self.xp.zeros((num_bins, num_derivs, num_derivs))

        # ECCENTRIC INNER BINARY NOT IMPLEMENTED
        dh = self.xp.zeros((num_bins, num_derivs, 2, 2 * N), self.xp.complex128)

        # assumes frequencies will be the same within a given binary and that the
        # fisher estimates will not adjust them a bin width

        for i, ind in enumerate(inds):

            # 2 eps up derivative
            params_up_2 = params.copy()
            params_up_2[ind] += 2 * eps
            for ind_trans, trans in parameter_transforms.items():
                if isinstance(ind_trans, int):
                    params_up_2[ind_trans] = trans(params_up_2[ind_trans])
                else:
                    params_up_2[np.asarray(ind_trans)] = trans(
                        *params_up_2[np.asarray(ind_trans)]
                    )

            self.run_wave(*params_up_2, **kwargs)

            h_I_up_2eps = self.xp.asarray([self.A, self.E]).squeeze()

            # 1 eps up derivative
            params_up_1 = params.copy()
            params_up_1[ind] += 1 * eps
            for ind_trans, trans in parameter_transforms.items():
                if isinstance(ind_trans, int):
                    params_up_1[ind_trans] = trans(params_up_1[ind_trans])
                else:
                    params_up_1[np.asarray(ind_trans)] = trans(
                        *params_up_1[np.asarray(ind_trans)]
                    )

            self.run_wave(*params_up_1, **kwargs)
            h_I_up_eps = self.xp.asarray([self.A, self.E]).squeeze()

            # 2 eps down derivative
            params_down_2 = params.copy()
            params_down_2[ind] -= 2 * eps
            for ind_trans, trans in parameter_transforms.items():
                if isinstance(ind_trans, int):
                    params_down_2[ind_trans] = trans(params_down_2[ind_trans])
                else:
                    params_down_2[np.asarray(ind_trans)] = trans(
                        *params_down_2[np.asarray(ind_trans)]
                    )

            self.run_wave(*params_down_2, **kwargs)
            h_I_down_2eps = self.xp.asarray([self.A, self.E]).squeeze()

            # 1 eps down derivative
            params_down_1 = params.copy()
            params_down_1[ind] -= 1 * eps
            for ind_trans, trans in parameter_transforms.items():
                if isinstance(ind_trans, int):
                    params_down_1[ind_trans] = trans(params_down_1[ind_trans])
                else:
                    params_down_1[np.asarray(ind_trans)] = trans(
                        *params_down_1[np.asarray(ind_trans)]
                    )

            self.run_wave(*params_down_1, **kwargs)
            h_I_down_eps = self.xp.asarray([self.A, self.E]).squeeze()

            # compute derivative
            dh_I = (-h_I_up_2eps + h_I_down_2eps + 8 * (h_I_up_eps - h_I_down_eps)) / (
                12 * eps
            )

            if len(dh_I.shape) == 2:
                dh_I = dh_I[:, self.xp.newaxis, :]

            # plug into derivative holder
            dh[:, i] = self.xp.transpose(dh_I, (1, 0, 2))

        # get frequencies for each binary
        freqs = self.freqs[0]

        try:
            freqs_temp = freqs.get()

        except AttributeError:
            freqs_temp = freqs

        # get PSD
        psd = self.xp.asarray(tdi.noisepsd_AE(freqs, **psd_kwargs))

        # noise factor
        noise_factor = self.xp.asarray(1.0 / psd * freqs)[:, self.xp.newaxis, :]

        # compute Fisher matrix
        for i in range(num_derivs):
            for j in range(i, num_derivs):
                # innter product between derivatives
                inner_prod = 4 * self.xp.sum(
                    (dh[:, i].conj() * dh[:, j] * noise_factor).real, axis=(1, 2)
                )

                # symmetry
                fish_matrix[:, i, j] = fish_matrix[:, j, i] = inner_prod

        # copy to cpu if needed
        if self.use_gpu and return_gpu is False:
            fish_matrix = fish_matrix.get()

        return fish_matrix
