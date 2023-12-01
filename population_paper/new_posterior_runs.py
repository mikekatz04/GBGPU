
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
from eryn.utils.utility import stepping_stone_log_evidence, thermodynamic_integration_log_evidence

from new_search_runs import PriorTransformFn

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

from gbsetups import LogLikeFn, ThirdBodyTemplateSetup, BaseTemplateSetup

import warnings

warnings.filterwarnings("ignore")
use_gpu = gpu_available

stop1 = SearchConvergeStopping(n_iters=20, diff=0.01, verbose=True)
def stop(iter, sample, sampler):
    if sampler.get_log_like().max() > -2.0:
        print("LL MAX:", sampler.get_log_like().max())
        return True
    temp = stop1(iter, sample, sampler)
    return temp


class ThirdPriorTransformFn:
    def __init__(self, f_min, f_max, fdot_min, fdot_max, P2_min, P2_max):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = f_min, f_max, fdot_min, fdot_max
        self.P2_min, self.P2_max = P2_min, P2_max

    def adjust_logp(self, logp, groups_running):
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1. / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1. / (fdot_max_here - fdot_min_here))

        P2_min_here = self.P2_min[groups_running]
        P2_max_here = self.P2_max[groups_running]
        P2_logpdf = np.log(1. / (P2_max_here - P2_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]
        logp[:] += P2_logpdf[:, None, None]
        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        coords[:, :, :, 1] = (coords[:, :, :, 1] - f_min_here[:, None, None]) / (f_max_here[:, None, None] - f_min_here[:, None, None])

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] - fdot_min_here[:, None, None]) / (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])
        
        P2_min_here = self.P2_min[groups_running]
        P2_max_here = self.P2_max[groups_running]
        coords[:, :, :, -2] = (coords[:, :, :, -2] - P2_min_here[:, None, None]) / (P2_max_here[:, None, None] - P2_min_here[:, None, None])
        
        return


def run_information(gb_third, nbin, data, data_orig, m3_lim, out_fp, directory_out, N_max, data_length, min_chirp_mass, max_chirp_mass, oversample, transform_fn, evidence_fp):
    for i in range(nbin):
        index = int(data["index"][i])

        orig_id = int(data["orig_id"][i])

        m3 = data_orig["M3"][int(index)]
        if m3 > m3_lim:
            continue 

        if data["e2"][i] > 0.985:
            print(f"not running {i} due to eccentricity over 0.985.")
            continue        

        with h5py.File(evidence_fp, "a") as f:
            if str(int(orig_id)) + '_third' not in list(f):
                # (f"{str(int(orig_id))} not in search.")
                continue

        with h5py.File(evidence_fp, "r") as fp_ev:
            two_logBF = fp_ev[str(int(orig_id)) + '_third']["keep_info"].attrs["2logBF"]

        if two_logBF <  two_logBF_lim:
            print(f"not running {i} due to 2logBF < {two_logBF_lim}.")
            continue

        reader_search = HDFBackend(evidence_fp, name=str(int(orig_id)) + "_third")
        last_evidence_sample = reader_search.get_last_sample()

        with h5py.File(evidence_fp, "a") as f:
            output_info = {name: f[str(int(orig_id)) + "_third"]["keep_info"].attrs[name] for name in f[str(int(orig_id)) + "_third"]["keep_info"].attrs}

        injection_params = np.array([data[key][i] for key in data.dtype.names[:14]])
        
        injection_params[7] = injection_params[7] % (2 * np.pi)

        injection_params[0] = np.log(np.exp(injection_params[0]))

        with h5py.File(directory_out + out_fp, "a") as f:
            if str(int(orig_id)) + f"_third" in list(f):
                print(f"{str(int(orig_id))}_third already in file {out_fp} so not running.")
                continue

        amp = np.exp(injection_params[0])
        f0 = injection_params[1] * 1e-3
        fdot0 = injection_params[2]

        f_min = f0 * 0.999 * 1e3
        f_max = f0 * 1.001 * 1e3
        f_lims = [f_min, f_max]

        fdot_min = get_fdot(f0, Mc=min_chirp_mass)
        fdot_max = get_fdot(f0, Mc=max_chirp_mass)

        fdot_lims = [fdot_min, fdot_max]

        N_found_base = get_N(amp, f0, Tobs, oversample=oversample).item()

        A2 = np.exp(data["A2"])[i]
        varpi  = data["omegabar"][i]
        e2  = data["e2"][i]
        P2  = data["P2"][i]
        T2  = data["T2"][i] * P2

        min_P2 = P2 * 1e-1 if P2 * 1e-1 < 32.0 else 32.0
        max_P2 = 1e1 * P2 if 1e1 * P2 > 32.0 else 32.0

        if P2 < 2.0:
            max_P2 = 2.2

        P2_lims = [min_P2, max_P2]

        # N_found_third = get_N(amp, f0, Tobs, oversample=oversample, P2=P2).item()
        N_found_third = gb_third.special_get_N(amp, f0, Tobs, A2,
                                                varpi,
                                                e2,
                                                P2,
                                                T2,oversample=oversample)

        N_found = np.max([N_found_base, N_found_third])
        waveform_kwargs["N"] = N_found
        # adjust for the few that are over 1 solar mass
        assert fdot0 >= fdot_min and fdot0 <= fdot_max

        params_inj_in = transform_fn["gb"].both_transforms(injection_params[np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])])

        if N_found > 2048:
            print(f"N_found is too high for now: {N_found}")
            continue
            waveform_kwargs["use_c_implementation"] = False

        A_inj, E_inj = gb_third.inject_signal(*params_inj_in, **waveform_kwargs)

        start_freq = int(int(f0 / df) - data_length / 2)
        fd = np.arange(start_freq, start_freq + data_length) * df

        data_channels = [A_inj[start_freq:start_freq + data_length].copy(), E_inj[start_freq:start_freq + data_length].copy()]

        AE_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model="sangria", includewd=Tobs / YEAR)
        psd = [AE_psd, AE_psd]
        
        yield (orig_id, index, injection_params, fd, data_channels, psd, start_freq, f_lims, fdot_lims, P2_lims, N_found, last_evidence_sample, output_info)


class RunPosteriorProcedure:
    def __init__(self, dt, Tobs, directory_in, directory_in2, directory_in3, seed_from_gen, directory_out, output_string, waveform_kwargs, ngroups, ntemps, nwalkers, data_length, snr_lim, m3_lim,  two_logBF_lim, oversample=4, use_gpu=True):

        self.third_info = ThirdBodyTemplateSetup()  # P2 = 2.0 for initial setup
        
        self.transform_fn = self.third_info.transform_fn
        self.snr_lim, self.m3_lim, self. two_logBF_lim = snr_lim, m3_lim,  two_logBF_lim
        self.data_length = data_length
        self.output_string = output_string
        # ## Setup all the parameters and waveform generator
        self.gb_third = GBGPUThirdBody(use_gpu=use_gpu)
        self.gb_third.get_ll = self.gb_third.get_ll_special
        self.waveform_kwargs = waveform_kwargs

        assert Tobs / dt == float(int(Tobs / dt))
        self.dt = dt
        self.Tobs = Tobs
        self.df = 1. / Tobs
        self.directory_in, self.directory_in2, self.seed_from_gen = directory_in, directory_in2, seed_from_gen
        self.directory_out = directory_out
        self.directory_in3 = directory_in3

        self.ngroups, self.ntemps, self.nwalkers = ngroups, ntemps, nwalkers
        
        self.keys = [
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

        self.max_chirp_mass = 1.05
        self.min_chirp_mass = 0.001

        self.tempering_kwargs = {"ntemps": self.ntemps, "Tmax": np.inf}
        self.oversample = oversample

        self.prior_transform_fn = ThirdPriorTransformFn(xp.zeros(ngroups), xp.ones(ngroups), xp.zeros(ngroups), xp.ones(ngroups), xp.ones(ngroups), xp.ones(ngroups))

        data_channels = [xp.zeros(self.data_length * self.ngroups, dtype=complex), xp.zeros(self.data_length * self.ngroups, dtype=complex)]
        psds = [xp.ones(self.data_length * self.ngroups, dtype=float), xp.ones(self.data_length * self.ngroups, dtype=float)]
        start_freq = xp.full(self.ngroups, int(1e-3 / self.df), dtype=np.int32)
        N_vals_in = xp.zeros(self.ngroups, dtype=int)
        d_d_all = xp.zeros(self.ngroups, dtype=float)
        self.N_max = int(self.data_length / 4)
        self.log_like_fn = LogLikeFn(self.gb_third, data_channels, psds, start_freq, self.df, self.third_info.transform_fn, N_vals_in, self.data_length, d_d_all, **waveform_kwargs)

        self.currently_running_orig_id = [None for _ in range(self.ngroups)]
        self.currently_running_output_info = [None for _ in range(self.ngroups)]
        
        self.start_state = {}
        self.sampler = {}

        info  = self.third_info
        coords = xp.zeros((self.ngroups, self.ntemps, self.nwalkers, info.ndim))

        branch_supp_base_shape = (self.ngroups, self.ntemps, self.nwalkers)

        data_inds = xp.repeat(xp.arange(self.ngroups, dtype=np.int32)[:, None], self.ntemps * self.nwalkers, axis=-1).reshape(self.ngroups, self.ntemps, self.nwalkers) 
        branch_supps = {"gb": BranchSupplimental(
            {"data_inds": data_inds}, base_shape=branch_supp_base_shape, copy=True
        )}

        groups_running = xp.zeros(self.ngroups, dtype=bool)
    
        start_state = ParaState({"gb": coords}, groups_running=groups_running, branch_supplimental=branch_supps)
        start_state.log_prior = xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        start_state.log_like = xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        start_state.betas = xp.ones((self.ngroups, self.ntemps))
        self.start_state = start_state

        # initialize sampler
        self.sampler = ParaEnsembleSampler(
            info.ndim,
            self.nwalkers,
            self.ngroups,
            self.log_like_fn,
            info.priors,
            tempering_kwargs=self.tempering_kwargs,
            args=[],
            kwargs={},
            gpu=gpu,
            periodic=info.periodic,
            backend=None,
            update_fn=None,
            update_iterations=-1,
            stopping_fn=None,
            stopping_iterations=-1,
            name="gb",
            prior_transform_fn=self.prior_transform_fn,
            provide_supplimental=True,
        )

    def run(self, nsteps):
        for fp in os.listdir(self.directory_in):
            generate_fp = f"pop_for_search_new_test_{self.seed_from_gen}_" + fp
            
            evidence_fp = f"{self.output_string}_{self.seed_from_gen}_evidence_" + fp[:-4] + ".h5"

            if evidence_fp != f"testing_new_setup_2_1010_evidence_data_C1_s433.h5":
                continue
            
            if evidence_fp not in os.listdir(self.directory_in3):
                raise FileNotFoundError(f"evidence_fp ({evidence_fp}) not found in {directory_in3}.")

            out_fp = f"{self.output_string}_{self.seed_from_gen}_PE_" + fp[:-4] + ".h5"

            data = np.genfromtxt(self.directory_in + fp, dtype=None)

            dtype = np.dtype([(key, '<f8') for key in self.keys])
            data_orig = np.asarray([tuple(data[i]) for i in range(len(data))], dtype=dtype)

            data = np.genfromtxt(
                directory_in2 + generate_fp,
                delimiter=",",
                names=True,
                dtype=None,
            )

            data["Amp"] = np.log(data["Amp"])
            data["f0"] = data["f0"] * 1e3
            data["iota"] = np.cos(data["iota"])
            data["beta"] = np.sin(data["beta"])
            data["A2"] = np.log(data["A2"])
            data["T2"] = data["T2"] / data["P2"]

            nbin = len(data)

            info_iterator = run_information(self.gb_third, nbin, data, data_orig, self.m3_lim, out_fp, self.directory_out, self.N_max, self.data_length, self.min_chirp_mass, self.max_chirp_mass, self.oversample, self.transform_fn, evidence_fp)
        
            self.run_mcmc(nbin, nsteps, info_iterator, out_fp)

    def setup_next_source(self, info_iterator):
        try:
            (orig_id, index, injection_params, fd, data_channels_tmp, psd_tmp, start_freq, f_lims, fdot_lims, P2_lims, N_val, last_evidence_sample, output_info) = next(info_iterator)
            
            d_d = 4.0 * self.df * np.sum(np.asarray(data_channels_tmp).conj() * np.asarray(data_channels_tmp) / np.asarray(psd_tmp)).item().real
            
            data_channels_tmp = [xp.asarray(tmp) for tmp in data_channels_tmp]
            psd_tmp = [xp.asarray(tmp) for tmp in psd_tmp]
                
            self.gb_third.d_d = d_d

            new_group_ind = self.start_state.groups_running.argmin().item()
            
            # data arrays are the same between third and base so no need to adjust base
            inds_slice = slice((new_group_ind) * self.data_length, (new_group_ind + 1) * self.data_length, 1)
            self.sampler.log_like_fn.data[0][inds_slice] = data_channels_tmp[0]
            self.sampler.log_like_fn.data[1][inds_slice] = data_channels_tmp[1]
            self.sampler.log_like_fn.psd[0][inds_slice] = psd_tmp[0]
            self.sampler.log_like_fn.psd[1][inds_slice] = psd_tmp[1]
            self.sampler.log_like_fn.start_freq[new_group_ind] = start_freq

            self.sampler.prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler.prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler.prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler.prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler.prior_transform_fn.P2_min[new_group_ind] = P2_lims[0]
            self.sampler.prior_transform_fn.P2_max[new_group_ind] = P2_lims[1]
            self.sampler.log_like_fn.N_vals[new_group_ind] = N_val
            self.sampler.log_like_fn.d_d_all[new_group_ind] = d_d

            # get info from search

            # get first group not running
            self.start_state.groups_running[new_group_ind] = True
            last_evidence_sample_nwalkers = last_evidence_sample.branches["gb"].shape[1]

            inds_start = np.random.choice(np.arange(last_evidence_sample_nwalkers), size=(self.ntemps, self.nwalkers), replace=True)
            self.start_state.branches["gb"].coords[new_group_ind] = xp.asarray(last_evidence_sample.branches["gb"].coords[0, inds_start, 0][None, :])

            self.currently_running_orig_id[new_group_ind] = orig_id
            self.currently_running_output_info[new_group_ind] = output_info

            return False

        except StopIteration:
            return True
            
    def run_mcmc(self, nbin, nsteps, info_iterator, out_fp):

        run = True
        finish_up = False

        while run:
            finish_up = self.setup_next_source(info_iterator)

            if not finish_up and np.any(~self.start_state.groups_running):
                continue
            # end if all are done
            if finish_up and np.all(~self.start_state.groups_running):
                run = False
                return
                
            self.start_state.log_like = None
            self.start_state.log_prior = None
            self.start_state.betas = None

            self.sampler.backend.reset(*self.sampler.backend.reset_args, **self.sampler.backend.reset_kwargs)
            self.start_state = self.sampler.run_mcmc(self.start_state, nsteps, burn=100, thin_by=25, progress=True, store=True)

            template = self.third_info

            for end_i in range(self.ngroups):
                orig_id = self.currently_running_orig_id[end_i]
                if orig_id is None:
                    continue
                group_name = str(int(orig_id)) + "_" + template.name + "_posterior"
                backend_tmp = HDFBackend(out_fp, name=group_name)
                backend_tmp.reset(
                    self.nwalkers,
                    template.ndim,
                    ntemps=self.ntemps,
                    branch_names=["gb"],
                )
                backend_tmp.grow(self.sampler.backend.iteration, None)
                with h5py.File(backend_tmp.filename, "a") as fp_save:
                    fp_save[group_name].attrs["iteration"] = self.sampler.iteration
                    fp_save[group_name]["chain"]["gb"][:] = self.sampler.get_chain()[:, end_i][:, :, :, None]
                    fp_save[group_name]["log_like"][:] = self.sampler.get_log_like()[:, end_i]
                    fp_save[group_name]["log_prior"][:] = self.sampler.get_log_prior()[:, end_i]
                    group_new = fp_save[group_name].create_group("keep_info")
                    for key, value in self.currently_running_output_info[end_i].items():
                        group_new.attrs[key] = value
                    

                self.currently_running_output_info[end_i] = None
                    
                self.start_state.groups_running[end_i] = False
                
                self.currently_running_orig_id[end_i] = None
                xp.get_default_memory_pool().free_all_blocks()

            if xp.all(~self.start_state.groups_running) and finish_up:
                run = False

    
if __name__ == "__main__":
    st = time.perf_counter()
    gpu = 7
    setDevice(gpu)
    use_gpu = True

    snr_lim = 5.0
    two_logBF_lim = -10.0
    m3_lim = 100.0

    dt = 15.0
    Tobs = 4.0 * YEAR

    N_total = int(Tobs / dt)
    Tobs = N_total * dt
    df = 1/Tobs
    convergence_iter_count = 25

    directory_in = "Realization_1/" # "Eccentric 3-body populations for Micheal/"
    directory_in2 = "populations_for_search/"
    seed_from_gen = 1010
    directory_out = "./"
    directory_in3 = "./"
    output_string = "testing_new_setup_2"
    waveform_kwargs = dict(N=None, dt=dt, T=Tobs, use_c_implementation=True)

    nwalkers = 200
    ntemps = 10
    ngroups = 200
    
    data_length = 8192
    runner = RunPosteriorProcedure(dt, Tobs, directory_in, directory_in2, directory_in3, seed_from_gen, directory_out, output_string, waveform_kwargs, ngroups, ntemps, nwalkers, data_length, snr_lim, m3_lim,  two_logBF_lim, use_gpu=use_gpu)

    nsteps = 1000
    runner.run(nsteps)

    et = time.perf_counter()
    print("TOTAL TIME:", et - st)
        
        

        

       
        