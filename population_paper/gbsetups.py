import numpy as np


import shutil
import wave
import numpy as np
import os
from scipy import stats
import h5py
import time
from scipy.stats import rv_continuous
import kalepy as kale
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

from eryn.prior import ProbDistContainer
from eryn.ensemble import EnsembleSampler
from eryn.paraensemble import ParaEnsembleSampler
from eryn.utils import PeriodicContainer

try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False
# import matplotlib.pyplot as plt

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import (
    inner_product,
    snr,
    fisher,
    covariance,
    mismatch_criterion,
    cutler_vallisneri_bias,
    scale_snr,
)
from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.state import State, ParaState, BranchSupplimental
from eryn.backends import HDFBackend

from lisatools.sampling.stopping import SNRStopping, SearchConvergeStopping

from gbgpu.gbgpu import GBGPU
from gbgpu.thirdbody import GBGPUThirdBody

from gbgpu.utils.constants import *
from gbgpu.utils.utility import *

from lisatools.sampling.samplingguide import GBGuide

import warnings

warnings.filterwarnings("ignore")
use_gpu = gpu_available

class BaseTemplateSetup:
    def __init__(self):
        self.name = "base"
        priors_in = {
            0: uniform_dist(np.log(1e-24), np.log(1e-20)),
            1: uniform_dist(0.0, 1.0),  # uniform_dist(f0 * 0.999 * 1e3, f0 * 1.001 * 1e3),  # uniform_dist(f0, max_f)
            2: uniform_dist(0.0, 1.0),
            3: uniform_dist(0.0, 2 * np.pi),
            4: uniform_dist(-1, 1),
            5: uniform_dist(0.0, np.pi),
            6: uniform_dist(0.0, 2 * np.pi),
            7: uniform_dist(-1, 1),
        }
        
        self.priors = {"gb": ProbDistContainer(priors_in, use_cupy=True)}

        transform_fn_in = {
            0: (lambda x: np.exp(x)),
            1: (lambda x: x * 1e-3),
            5: (lambda x: np.arccos(x)),
            8: (lambda x: np.arcsin(x)),
        }

        transform_fn_in[(1, 2, 3)] = lambda f0, fdot, fddot: (
            f0,
            fdot,
            11 / 3.0 * fdot ** 2 / f0,
        )
        fill_dict = {
            "fill_inds": np.array([3]),
            "ndim_full": 9,
            "fill_values": np.array([0.0]),
        }
        self.transform_fn = {"gb": 
            TransformContainer(
                parameter_transforms=transform_fn_in, fill_dict=fill_dict
            )
        }
        self.ndim = 8
        self.periodic = PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}})




class Dist1D(rv_continuous):
    def __init__(self, x, y, **kwargs):

        a = x.min()
        b = x.max()

        super(Dist1D, self).__init__(a=a, b=b, **kwargs)
        self.pdf_spl = CubicSpline(x, y)

        out_quad = quad(self.pdf_spl, a, b)[0]
        self.pdf_spl = CubicSpline(x, y / out_quad)

        out_quad = np.zeros_like(x)
        for i, xtemp in enumerate(x[1:]):
            out_quad[i + 1] = quad(self.pdf_spl, a, xtemp)[0]
            if out_quad[i + 1] > 1.0:
                break
        out_quad = out_quad[: i + 1]
        x_ppf = x[: i + 1]
        # if b < 5:
        #   plt.plot(x_ppf, out_quad)
        self.ppf_spl = CubicSpline(out_quad, x_ppf)

    def _pdf(self, x):
        return self.pdf_spl(x)

    def _ppf(self, x):
        return self.ppf_spl(x)



class A2Dist(Dist1D):
    def __init__(self, P2, A2, P2_min=0.0, P2_max=1e3, **kwargs):

        points_in = np.log(A2[(P2 >= P2_min) & (P2 <= P2_max)])

        x, y = kale.density(points_in, points=None, reflect=True)
        super(A2Dist, self).__init__(x, y, **kwargs)


class T2Dist(Dist1D):
    def __init__(self, gb_third, e2_min=0.0, e2_max=0.999, num=int(1e7), **kwargs):

        self.gb_third = gb_third
        super(Dist1D, self).__init__(a=0.0, b=1.0, **kwargs)
        e2_vals = np.random.uniform(e2_min, e2_max, size=(num,))
        phi2_vals = np.random.uniform(0, 2 * np.pi, size=(num,))

        T2_scaled = self.gb_third.get_T2(1.0, e2_vals, phi2_vals)

        counts, bin_edges = np.histogram(T2_scaled, bins=500)
        counts = counts.astype(float)
        counts /= num
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        spl_temp = CubicSpline(bin_centers, counts)

        x_in = np.zeros(len(bin_centers) + 2)
        x_in[1:-1] = bin_centers.copy()
        x_in[0] = 0.0
        x_in[-1] = 1.0
        y_in = spl_temp(x_in)

        super(T2Dist, self).__init__(x_in, y_in)


class ThirdBodyTemplateSetup:
    def __init__(self):

        self.name = "third"
        
        # adjust P2 and A2 according to P2
        # min_P2 = P2 * 1e-1 if P2 * 1e-1 < 32.0 else 32.0
        # max_P2 = 1e1 * P2 if 1e1 * P2 > 32.0 else 32.0

        # if P2 < 2.0:
        #     max_P2 = 2.2

        # all A2 stuff
        P2_amps, A2_amps = np.load("P2_A2.npy")

        # TODO: must decide on this
        amps_dist = A2Dist(P2_amps, A2_amps, P2_min=0.0, P2_max=1000.0)  # P2_min=min_P2, P2_max=max_P2)

        priors_in = {
            0: uniform_dist(np.log(1e-24), np.log(1e-20)),
            1: uniform_dist(0.0, 1.0),  # uniform_dist(f0 * 0.999 * 1e3, f0 * 1.001 * 1e3),  # uniform_dist(f0, max_f)
            2: uniform_dist(0.0, 1.0),
            3: uniform_dist(0.0, 2 * np.pi),
            4: uniform_dist(-1, 1),
            5: uniform_dist(0.0, np.pi),
            6: uniform_dist(0.0, 2 * np.pi),
            7: uniform_dist(-1, 1),
            8: uniform_dist(np.log(1.0), np.log(1e5)),  # amps_dist,
            9: uniform_dist(0.0, 2 * np.pi),  # varpi
            10: uniform_dist(0.0, 0.985),
            11: uniform_dist(0.0, 1.0), # mapped P2
            12: uniform_dist(0.0, 1.0),
        }

        self.priors = {"gb": ProbDistContainer(priors_in, use_cupy=True)}

        transform_fn_in = {
            0: (lambda x: np.exp(x)),
            1: (lambda x: x * 1e-3),
            5: (lambda x: np.arccos(x)),
            8: (lambda x: np.arcsin(x)),
            9: (lambda x: np.exp(x)),
            (12, 13): lambda P2, T2: (P2, T2 * P2),
        }

        transform_fn_in[(1, 2, 3)] = lambda f0, fdot, fddot: (
            f0,
            fdot,
            11 / 3.0 * fdot ** 2 / f0,
        )
        
        fill_dict = {
            "fill_inds": np.array([3]),
            "ndim_full": 14,
            "fill_values": np.array([0.0]),
        }
        self.transform_fn = {"gb": 
            TransformContainer(
                parameter_transforms=transform_fn_in, fill_dict=fill_dict
            )
        }
        self.ndim = 13
        self.periodic = PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi, 9: 2 * np.pi, 12: 1.0}})


class LogLikeFn:
    def __init__(self, gb, data, psd, start_freq, df, transform_fn, N_vals, data_length, d_d_all, **waveform_kwargs):
        self.data, self.psd, self.start_freq, self.df = data, psd, start_freq, df
        self.transform_fn = transform_fn
        self.N_vals = N_vals
        self.gb = gb
        self.waveform_kwargs = waveform_kwargs
        self.data_length = data_length
        self.d_d_all = d_d_all

    def __call__(self, x, *args, branch_supps=None, **kwargs):
        if branch_supps is None:
            raise ValueError("Branch supps needed.")

        data_index = branch_supps["gb"]["data_inds"]

        x_in = self.transform_fn["gb"].both_transforms(x, xp=xp, copy=True)

        N = self.N_vals[data_index]
        
        assert data_index.dtype == np.int32

        self.gb.d_d = self.d_d_all[data_index]
        
        if "N" in self.waveform_kwargs:
            self.waveform_kwargs.pop("N")

        ll = self.gb.get_ll(x_in.get(), self.data, self.psd, data_index=data_index, data_length=self.data_length, noise_index=data_index, start_freq_ind=self.start_freq, N=N, **self.waveform_kwargs)

        return ll
