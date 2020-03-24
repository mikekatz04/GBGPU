"""
Wrapper code for gpuPhenomHM. Helps to calculate likelihoods
for samplers. Author: Michael Katz

Calculates phenomHM waveforms, puts them through the LISA response
and calculates likelihood.
"""

import numpy as np
from scipy import constants as ct

from katzsamplertools.utils.convert import Converter
from katzsamplertools.utils.constants import *
from katzsamplertools.utils.generatenoise import (
    generate_noise_frequencies,
    generate_noise_single_channel,
)

import GBGPU
import tdi

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    print("No cupy")

import time


class pyGBGPU:
    def __init__(
        self,
        injection,
        max_length_init,
        nWD,
        ndevices,
        data_freqs,
        data_stream,
        key_order,
        Tobs,
        dt,
        **kwargs
    ):
        """
        data_stream (dict): keys X, Y, Z or A, E, T
        """
        prop_defaults = {
            "TDItag": "AET",  # AET or XYZ
            "data_stream_whitened": True,
            "data_params": {},
            "eps": 1e-6,
            "NP": 8,
            "noise_kwargs": {"model": "SciRDv1", "includewd": None},
            "add_noise": None,  # if added should be dict with fs
            "oversample": 4,
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            # TODO: check this
            kwargs[prop] = kwargs.get(prop, default)

        self.ndim = self.NP
        self.nWD, self.ndevices = nWD, ndevices
        self.Tobs = Tobs
        self.dt = dt
        self.max_length_init = max_length_init
        self.data_freqs, self.data_stream = data_freqs, data_stream
        self.injection = injection
        self.key_order = key_order
        self.num_injections = len(injection)

        self.converter = Converter("wd", key_order)

        self.determine_freqs_noise(**kwargs)

        if self.TDItag not in ["AET", "XYZ"]:
            raise ValueError("TDItag must be AET or XYZ.")

        else:
            if self.TDItag == "AET":
                self.TDItag_in = 2
            else:
                self.TDItag_in = 1

        self.generator = GBGPU.GBGPU(
            self.max_length_init,
            self.data_freqs,
            self.nWD,
            self.ndevices,
            self.Tobs,
            self.dt,
            self.NP,
            oversample=self.oversample,
        )

        if self.data_stream is {} or self.data_stream is None:
            if self.injection == {}:
                raise ValueError(
                    "If data_stream is empty dict or None,"
                    + "user must supply data_params kwarg as "
                    + "dict with params for data stream."
                )

            self.injection_array = np.zeros((self.nWD, self.NP))

            self.injection_array[: self.num_injections] = np.asarray(
                [
                    [injection_i[key] for key in self.key_order]
                    for injection_i in self.injection
                ]
            )

            self.injection_array[self.num_injections :] = self.injection_array[0]

            self.inject_signal()

            self.data_stream_whitened = False

        self.create_input_data(**kwargs)

    def create_input_data(self, **kwargs):
        for i, channel in enumerate(self.TDItag):
            if channel not in self.data_stream:
                raise KeyError("{} not in TDItag {}.".format(channel, self.TDItag))

            setattr(
                self,
                "data_channel{}".format(i + 1),
                xp.asarray(self.data_stream[channel]),
            )

        additional_factor = np.ones_like(self.data_freqs)
        df = self.data_freqs[1] - self.data_freqs[0]
        additional_factor = np.sqrt(df)

        if self.TDItag == "AET":
            self.TDItag_in = 2

            self.channel1_ASDinv = xp.asarray(
                1.0
                / np.sqrt(tdi.noisepsd_AE(self.data_freqs, **self.noise_kwargs))
                * additional_factor
            )
            self.channel2_ASDinv = xp.asarray(
                1.0
                / np.sqrt(tdi.noisepsd_AE(self.data_freqs, **self.noise_kwargs))
                * additional_factor
            )
            self.channel3_ASDinv = xp.asarray(
                1.0
                / np.sqrt(
                    tdi.noisepsd_T(
                        self.data_freqs, model=kwargs["noise_kwargs"]["model"]
                    )
                )
                * additional_factor
            )

        elif self.TDItag == "XYZ":

            self.TDItag_in = 1
            for i in range(1, 4):
                temp = (
                    1.0
                    / np.sqrt(tdi.noisepsd_XYZ(self.data_freqs, **self.noise_kwargs))
                    * additional_factor
                )
                setattr(self, "channel{}_ASDinv".format(i), temp)

        if self.data_stream_whitened is False:
            for i in range(1, 4):
                temp = getattr(self, "data_channel{}".format(i)) * getattr(
                    self, "channel{}_ASDinv".format(i)
                )
                setattr(self, "data_channel{}".format(i), temp)

        self.d_d = 4 * np.sum(
            [
                np.abs(self.data_channel1.get()) ** 2,
                np.abs(self.data_channel2.get()) ** 2,
                np.abs(self.data_channel3.get()) ** 2,
            ]
        )

        self.generator.input_data(
            self.template_channel1,
            self.template_channel2,
            self.template_channel3,
            self.data_channel1,
            self.data_channel2,
            self.data_channel3,
            self.channel1_ASDinv,
            self.channel2_ASDinv,
            self.channel3_ASDinv,
        )

    def inject_signal(self):
        data_channel = xp.asarray(np.ones_like(self.data_freqs, dtype=np.complex128))
        channel_ASDinv = xp.asarray(np.ones_like(self.data_freqs))

        self.template_channel1 = xp.zeros(
            self.max_length_init * self.oversample * self.nWD * 2
        )
        self.template_channel2 = xp.zeros(
            self.max_length_init * self.oversample * self.nWD * 2
        )
        self.template_channel3 = xp.zeros(
            self.max_length_init * self.oversample * self.nWD * 2
        )

        self.generator.input_data(
            self.template_channel1,
            self.template_channel2,
            self.template_channel3,
            data_channel,
            data_channel,
            data_channel,
            channel_ASDinv,
            channel_ASDinv,
            channel_ASDinv,
        )

        data_stream_out = self.getNLL(self.injection_array.T, return_TDI=True)

        inds_data = (
            self.injection_array.T[0][: self.num_injections] / 1e3 / self.df
        ).astype(int) - int((self.oversample * self.max_length_init) / 2)

        data_stream_out_temp = []
        for dso in data_stream_out:
            temp = xp.zeros_like(xp.asarray(self.data_freqs), dtype=xp.complex128)

            for i, ind_start in zip(range(self.num_injections), inds_data):
                ind_end = ind_start + int((self.oversample * self.max_length_init))
                temp[ind_start:ind_end] = xp.asarray(dso[i])

            data_stream_out_temp.append(temp)

        self.data_stream = {}
        for key, val, an in zip(self.TDItag, data_stream_out_temp, self.added_noise):

            if isinstance(an, np.ndarray):
                an = xp.asarray(an)
            self.data_stream[key] = val + an

    def determine_freqs_noise(self, **kwargs):
        self.added_noise = [0.0 for _ in range(3)]
        if self.data_freqs is None:
            fs = 1 / self.dt
            noise_freqs = generate_noise_frequencies(self.Tobs, fs)

            self.data_freqs = data_freqs = noise_freqs

            self.df = df = 1 / self.Tobs

            self.data_freqs[0] = self.data_freqs[1] / 100.0

            if self.add_noise is not None:

                self.added_noise[0] = generate_noise_single_channel(
                    tdi.noisepsd_AE, [], self.noise_kwargs, df, data_freqs
                )

                self.added_noise[1] = generate_noise_single_channel(
                    tdi.noisepsd_AE, [], self.noise_kwargs, df, data_freqs
                )

                self.added_noise[2] = generate_noise_single_channel(
                    tdi.noisepsd_T,
                    [],
                    dict(model=kwargs["noise_kwargs"]["model"]),
                    df,
                    data_freqs,
                )

    def NLL(
        self,
        f0,
        fdot,
        beta,
        lam,
        amp,
        iota,
        psi,
        phi0,
        return_snr=False,
        return_TDI=False,
    ):

        params = np.array([f0, fdot, beta, lam, amp, iota, psi, phi0]).T.flatten()

        self.generator.FastGB(params)
        out = self.generator.Likelihood()
        if return_TDI:
            tdis = [
                self.template_channel1.get(),
                self.template_channel2.get(),
                self.template_channel3.get(),
            ]
            tdis_real = [tdis_i[0::2] for tdis_i in tdis]
            tdis_imag = [tdis_i[1::2] for tdis_i in tdis]

            tdis_out = [
                tdis_re + tdis_im * 1j for tdis_re, tdis_im in zip(tdis_real, tdis_imag)
            ]

            tdis_out = [
                tdis_out_i.reshape(self.nWD, self.oversample * self.max_length_init)
                for tdis_out_i in tdis_out
            ]
            return tdis_out

        d_h = out[0::3]
        h_h = out[1::3]
        d_minus_h = out[2::3]

        if return_snr:
            return np.sqrt(d_h), np.sqrt(h_h)

        # 1/2<d-h|d-h> = 1/2(<d|d> + <h|h> - 2<d|h>)
        # return 1.0 / 2.0 * (self.d_d + h_h - 2 * d_h)
        return 1.0 / 2.0 * d_minus_h

    def getNLL(self, x, **kwargs):
        # changes parameters to in range in actual array (not copy)
        x = self.converter.recycle(x)

        # converts parameters in copy, not original array
        x_in = self.converter.convert(x.copy())

        f0, fdot, beta, lam, amp, iota, psi, phi0 = x_in

        return self.NLL(f0, fdot, beta, lam, amp, iota, psi, phi0, **kwargs)

    def get_Fisher(self, x):
        Mij = np.zeros((self.ndim, self.ndim), dtype=x.dtype)
        if self.nWD * self.ndevices < 2 * self.ndim:
            raise ValueError("num walkers must be greater than 2*ndim")
        x_in = np.tile(x, (self.nWD * self.ndevices, 1))

        for i in range(self.ndim):
            x_in[2 * i, i] += self.eps
            x_in[2 * i + 1, i] -= self.eps

        A, E, T = self.getNLL(x_in.T, return_TDI=True)

        for i in range(self.ndim):
            Ai_up, Ei_up, Ti_up = A[2 * i + 1], E[2 * i + 1], T[2 * i + 1]
            Ai_down, Ei_down, Ti_down = A[2 * i], E[2 * i], T[2 * i]

            hi_A = (Ai_up - Ai_down) / (2 * self.eps)
            hi_E = (Ei_up - Ei_down) / (2 * self.eps)
            hi_T = (Ti_up - Ti_down) / (2 * self.eps)

            for j in range(i, self.ndim):
                Aj_up, Ej_up, Tj_up = A[2 * j + 1], E[2 * j + 1], T[2 * j + 1]
                Aj_down, Ej_down, Tj_down = A[2 * j], E[2 * j], T[2 * j]

                hj_A = (Aj_up - Aj_down) / (2 * self.eps)
                hj_E = (Ej_up - Ej_down) / (2 * self.eps)
                hj_T = (Tj_up - Tj_down) / (2 * self.eps)

                inner_product = 4 * np.real(
                    (
                        np.dot(hi_A.conj(), hj_A)
                        + np.dot(hi_E.conj(), hj_E)
                        + np.dot(hi_T.conj(), hj_T)
                    )
                )

                Mij[i][j] = inner_product
                Mij[j][i] = inner_product

        return Mij
