import shutil
import wave
import numpy as np
import os
from scipy import stats
import h5py
import time

from eryn.prior import ProbDistContainer
from eryn.ensemble import EnsembleSampler
from eryn.paraensemble import ParaEnsembleSampler
from eryn.utils import PeriodicContainer

from gbgpu.thirdbody import third_body_factors

try:
    import cupy as cp
    from cupy.cuda.runtime import setDevice

    gpu_available = True
except ModuleNotFoundError:
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

from gbmodels import BaseTemplateSetup, ThirdBodyTemplateSetup, PriorTransformFn, LogLikeFn, TemplateSetup
import warnings

from settings import get_settings

from typing import Callable, Union, List, Tuple

from abc import ABC


class Guide(ABC):

    def __init__(self, name, settings: dict, gpu: int=None):
        self.settings = settings
        self.name = name

        self.gpu = gpu
        if gpu is not None:
            assert gpu_available
            self.use_gpu = True
            self.xp.cuda.runtime.setDevice(gpu)
        else:
            self.use_gpu = False

        self.third_info = ThirdBodyTemplateSetup(use_gpu=self.use_gpu)
        self.base_info = BaseTemplateSetup(use_gpu=self.use_gpu)

    @property
    def xp(self):
        xp = cp if self.use_gpu else np
        return xp

    @property
    def verbose(self) -> bool:
        return self.settings["verbose"]

    @property
    def dir_info(self) -> dict:
        return self.settings["dir_info"]
    
    @property
    def waveform_kwargs(self) -> dict:
        return self.settings["waveform_kwargs"]

    def run_directory_list(self):
        for directory in self.dir_info["population_directory_list"]:
            if not os.path.exists(directory):
                raise ValueError(f"Directy in population directory list does not exist ({directory}).")

            self.run_single_directory(directory)

    def check_run_status(self, indicator: str) -> bool:

        if not os.path.exists(self.status_file):
            with open(self.status_file, "w") as fp:
                pass
        
        with open(self.status_file, "r") as fp:
            lines = fp.readlines()

        if f"{indicator}\n" in lines:
            # already run 
            return False

        else:
            return True

    def add_to_status_file(self, indicator: str):

        if not os.path.exists(self.status_file):
            with open(self.status_file, "w") as fp:
                pass
        
        with open(self.status_file, "a") as fp:
            fp.write(f"{indicator}\n")

        return

    @property
    def status_file(self) -> str:
        return self.dir_info["main_dir"] + self.dir_info["status_file_base"] + "_" + self.dir_info["base_string"] + f"_{self.name}.txt"

    def retrieve_input_files(self, directory: str):
        for fp in os.listdir(directory):
            if fp[-4:] == ".dat":
                indicator = fp.split("data_")[-1].split(".dat")[0]

                need_to_run = self.check_run_status(indicator)

                if need_to_run:
                    yield (indicator, fp)
                else:
                    if self.verbose:
                        print(f"Not running {fp}. Already in folder.")

    @classmethod
    def run_single_model(self, directory: str, indicator: str, fp: str):
        raise NotImplementedError

    def run_single_directory(self, directory: str):
        for indicator, fp in self.retrieve_input_files(directory):
            self.run_single_model(directory, indicator, fp)

    def get_input_data(self, path_to_input_file: str) -> np.ndarray:
        data = np.genfromtxt(path_to_input_file, dtype=None)
        keys = [
            'id',
            'P',
            'l_deg',
            'b_deg',
            'd_kpc',
            'wd1_mass',
            'wd1_rad',
            'wd2_mass',
            'wd2_rad',
            'age_yr',
            'wd1_cool_time',
            'wd2_cool_time',
            'cos_i',
            'SeBa_id',
            'phi',
            'theta',
            'l_ecl',
            'b_ecl',
            'M3',
            'a3',
            'Phi3',
            'iota3',
            'e3'
        ]
        dtype = np.dtype([(key, '<f8') for key in keys])
        data_new = np.asarray([tuple(data[i]) for i in range(len(data))], dtype=dtype)

        return data_new

class TriplesSetupGuide(Guide):

    def __init__(self, *args, N=16384, batch_size=1000, **kwargs):
        name = "triples_setup"
        self.batch_size = batch_size
        self.N = N
        super().__init__(name, *args, **kwargs)

    def build_sampling_parameter_set(self, data_input):
        
        m1 = data_input['wd1_mass']
        m2 = data_input['wd2_mass']
        period = data_input['P'] * 24.0 * 3600.0 # days to seconds
        d_kpc = data_input['d_kpc']
        index_out = np.arange(len(m1))

        f0 = 2 / period
        Amp = get_amplitude(m1, m2, f0, d_kpc)
        fdot = get_fdot(f0, m1=m1, m2=m2)
        fddot = 11/3 * fdot ** 2 / f0
        phi0 = np.random.uniform(0.0, 2* np.pi, size=len(f0))
        iota = np.arccos(data_input['cos_i'])
        psi = np.random.uniform(0, np.pi, size=len(f0))
        lam = data_input['l_ecl']
        beta =data_input['b_ecl']

        # third body comps
        m3 = data_input['M3']
        a2 = data_input['a3'] * AU
        phi2 = data_input['Phi3']
        i2 = data_input['iota3']
        e2 = data_input['e3']
        P2 = (2 * np.pi * np.sqrt(a2 ** 3 / (G * (m1 + m2 + m3) * MSUN))) / YEAR

        Omega2 = np.random.rand(len(f0)) * 2 * np.pi
        omega2 = np.random.rand(len(f0)) * 2 * np.pi

        A2, varpi, T2 = third_body_factors(
            m1 + m2,
            m3.copy(),
            P2.copy(),
            e2.copy(),
            i2.copy(),
            Omega2,
            omega2,
            phi2,
            lam,
            beta,
            third_mass_unit="Mjup",
            third_period_unit="yrs",
        )
        out = np.array([
            Amp, 
            f0, 
            fdot, 
            fddot,
            phi0, 
            iota, 
            psi,
            lam,
            beta,
            A2,
            varpi,
            e2,
            P2,
            T2,
            i2,
            phi2,
            Omega2,
            omega2,
        ]).T

        return out

    def build_waveforms(self, gb_gen: Callable, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if "N" in self.waveform_kwargs:
            self.waveform_kwargs.pop("N")

        gb_gen.run_wave(*params.T, N=self.N, **self.waveform_kwargs)
        return gb_gen.A, gb_gen.E, gb_gen.freqs

    def get_likelihood_information(self, params: np.ndarray):

        A_third, E_third, freqs_third = self.build_waveforms(self.third_info.template_gen, params[:, :14])
        A_base, E_base, freqs_base = self.build_waveforms(self.base_info.template_gen, params[:, :9])
        assert self.xp.all(freqs_base == freqs_third)

        freqs_third_in_psd = freqs_third if not self.use_gpu else freqs_third.get()
        psd = self.xp.asarray(get_sensitivity(freqs_third_in_psd.flatten(), sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR).reshape(freqs_third.shape))
        df = 1 / self.waveform_kwargs["T"]

        third_third = 4 * df * self.xp.sum((A_third.conj() * A_third + E_third.conj() * E_third) / psd, axis=-1)
        third_base = 4 * df * self.xp.sum((A_third.conj() * A_base + E_third.conj() * E_base) / psd, axis=-1)
        base_base = 4 * df * self.xp.sum((A_base.conj() * A_base + E_base.conj() * E_base) / psd, axis=-1)
        
        phase_change = self.xp.angle(third_base)
        opt_snr = self.xp.sqrt(third_third.real)
        ll_diff = -1/2 * (third_third + base_base - 2 * self.xp.abs(third_base))
        
        try:
            # get back from gpu if needed
            phase_change = phase_change.get()
            opt_snr = opt_snr.get()
            ll_diff = ll_diff.get()

        except AttributeError:
            pass

        return (ll_diff, opt_snr, phase_change)

    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        if self.verbose:
            print("Start triples setup:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        
        params_input = self.build_sampling_parameter_set(data_input)

        num_sources = params_input.shape[0]
        inds_tmp = np.arange(0, num_sources, self.batch_size)

        if inds_tmp[-1] != num_sources - 1:
            inds_tmp = np.concatenate([inds_tmp, np.array([num_sources - 1])])

        ll_info, snr_info, phase_change_info = np.zeros(num_sources), np.zeros(num_sources), np.zeros(num_sources)
        for iteration, (start, end) in enumerate(zip(inds_tmp[:-1], inds_tmp[1:])):
            ll_tmp, snr_tmp, phase_change_tmp = self.get_likelihood_information(params_input[start:end])

            ll_info[start:end] = ll_tmp
            snr_info[start:end] = snr_tmp
            phase_change_info[start:end] = phase_change_tmp

            if self.verbose:
                print(f"{iteration + 1} of {len(inds_tmp[:-1])}")

        original_id = data_input["id"]
        self.readout(indicator, original_id, params_input, ll_info, snr_info, phase_change_info)

        self.add_to_status_file(indicator)
        if self.verbose:
            print("End triples setup:", directory, indicator, fp)

    def readout(self, indicator: str, original_id: np.ndarray, params: np.ndarray, ll_info: np.ndarray, opt_snr: np.ndarray, phase_change: np.ndarray):

        out_fp = self.dir_info["triples_setup_directory"] + self.dir_info["base_string"] + f"_pop_for_search_{indicator}.dat"

        inds = np.where(ll_info > self.settings["cut_info"]["first_cut_ll_diff_lim"])[0]
        ll_good = ll_info[inds]
        snr_good = opt_snr[inds]
        phase_change_good = phase_change[inds]

        tmp = np.array([ll_good, snr_good, phase_change_good]).T
        params_in = params[inds]

        readout = np.concatenate([params_in, tmp, inds[:, np.newaxis], original_id[inds][:, np.newaxis]], axis=1)

        header = ""

        for key in [
            "Amp", 
            'f0', 
            'fdot', 
            'fddot', 
            'phi0', 
            'iota', 
            'psi',
            'lam',
            'beta',
            'A2',
            'omegabar',
            'e2',
            'P2',
            'T2',
            'i2',
            'phi2',
            'Omega2',
            'omega2',
            'lldiff_base_marg',
            'snr',
            'phase_shift_base',
            'index',
            "orig_id"
            ]:
            header += key + ','

        header = header[:-1]

        if self.verbose:
            print("Saving", out_fp)

        np.savetxt(out_fp, readout, header=header, delimiter=',')

        # test it immediately to make sure it reads in
        check = np.genfromtxt(out_fp, names=True, delimiter=',', dtype=None)



if __name__ == "__main__":

    settings = get_settings()
    gpu = 0
    runner = TriplesSetupGuide(settings, batch_size=1000, gpu=gpu)
    runner.run_directory_list()
    breakpoint()