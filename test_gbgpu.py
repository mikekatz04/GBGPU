from gbgpu.gbgpu import pyGBGPU
import numpy as np
import argparse
import time
import scipy.constants as ct

from katzsamplertools.utils.constants import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--time", "-t", default=0, type=int)
    args = parser.parse_args()

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

    max_length_init = 2 ** 11
    nWD = 1000
    ndevices = 1
    data_freqs = None
    data_stream = None
    key_order = [
        "milli_f0",
        "log10_fdot",
        "sin_beta",
        "lam",
        "log10_amp",
        "cos_iota",
        "psi",
        "phi0",
    ]
    Tobs = 4.0 * YRSID_SI
    dt = 10.0

    injection = [
        {
            "milli_f0": 1.35962000e-03 / 1e-3,
            "log10_fdot": np.log10(8.94581279e-19),
            "sin_beta": np.sin(3.12414000e-01),
            "lam": -2.75291000e00,
            "log10_amp": np.log10(1.07345000e-22),
            "cos_iota": np.cos(5.23599000e-01),
            "psi": 0.42057295,
            "phi0": 3.05815650e00,
        }
    ]
    """
        {
            "milli_f0": 2e-3 / 1e-3,
            "log10_fdot": np.log10(2e-18),
            "sin_beta": np.sin(0.4),
            "lam": 2.0,
            "log10_amp": np.log10(1e-21),
            "cos_iota": np.cos(1.0),
            "psi": 1.0,
            "phi0": np.pi / 2,
        },
    ]
    """

    pygbgpu = pyGBGPU(
        injection,
        max_length_init,
        nWD,
        ndevices,
        data_freqs,
        data_stream,
        key_order,
        Tobs,
        dt,
        **prop_defaults
    )

    waveform_params = np.zeros((nWD, 8))

    waveform_params[: len(injection)] = np.asarray(
        [[injection_i[key] for key in key_order] for injection_i in injection]
    )

    waveform_params[len(injection) :] = waveform_params[0]

    like = pygbgpu.getNLL(waveform_params.T)
    snr = pygbgpu.getNLL(waveform_params.T, return_snr=True)

    if args.time:
        st = time.perf_counter()
        check = pygbgpu.getNLL(waveform_params.T)
        for _ in range(args.time):
            check = pygbgpu.getNLL(waveform_params.T)
        et = time.perf_counter()

        print("Number of evals:", args.time)
        print("ndevices:", ndevices, "nWD:", nWD)
        print("total time:", et - st)
        print("time per group likelihood call:", (et - st) / args.time)
        print(
            "time per individual likelihood call:",
            (et - st) / (args.time * nWD * ndevices),
        )

    check = pygbgpu.getNLL(waveform_params.T)
    fisher = pygbgpu.get_Fisher(waveform_params[0])
    import pdb

    pdb.set_trace()
