
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


def run_information(gb_third, nbin, data, data_orig, m3_lim, out_fp, directory_out, N_max, data_length, min_chirp_mass, max_chirp_mass, oversample, transform_fn, search_fp):
    for i in range(nbin):
        index = int(data["index"][i])

        orig_id = int(data["orig_id"][i])

        print(orig_id)
        m3 = data_orig["M3"][int(index)]
        if m3 > m3_lim:
            continue 

        if data["e2"][i] > 0.985:
            print(f"not running {i} due to eccentricity over 0.985.")
            continue        

        with h5py.File(search_fp, "a") as f:
            if str(int(orig_id)) not in list(f):
                print(f"{str(int(orig_id))} not in search.")
                continue

        reader_search = HDFBackend(search_fp, name=str(int(orig_id)))
        last_search_sample = reader_search.get_last_sample()
        log_like_max_search = last_search_sample.log_like.max()
        if last_search_sample.log_like.max() > -ll_diff_lim:
            print(f"not running {i} due to log prob max > -2.")
            continue

        with h5py.File(search_fp, "a") as f:
            output_info = {name: f[str(int(orig_id))]["keep_info"].attrs[name] for name in f[str(int(orig_id))]["keep_info"].attrs}
    
        # extract_snr_base and snr_base were switched
        if (
            log_like_max_search > -ll_diff_lim
            or data[f"extract_snr_base"][i] < snr_lim
        ):
            continue

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

        params_inj_in = transform_fn["third"]["gb"].both_transforms(injection_params[np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])])

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
        
        yield (orig_id, index, injection_params, fd, data_channels, psd, start_freq, f_lims, fdot_lims, P2_lims, N_found, last_search_sample, output_info)


class RunEvidenceProcedure:
    def __init__(self, dt, Tobs, directory_in, directory_in2, directory_in3, seed_from_gen, directory_out, output_string, waveform_kwargs, ngroups, ntemps, nwalkers, data_length, snr_lim, m3_lim, ll_diff_lim, oversample=4, use_gpu=True):

        self.base_info = BaseTemplateSetup()
        self.third_info = ThirdBodyTemplateSetup()  # P2 = 2.0 for initial setup
        
        self.transform_fn = {
            self.base_info.name: self.base_info.transform_fn,
            self.third_info.name: self.third_info.transform_fn,
        }
        self.snr_lim, self.m3_lim, self.ll_diff_lim = snr_lim, m3_lim, ll_diff_lim
        self.data_length = data_length
        self.output_string = output_string
        # ## Setup all the parameters and waveform generator
        self.gb = GBGPU(use_gpu=use_gpu)
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

        # adjust_betas
        ntemps_over_5 = int(ntemps / 5)

        ntemps_1 = 4 * ntemps_over_5
        ntemps_2 = ntemps - ntemps_1

        betas_1 = np.logspace(-4, 0, ntemps_1)[::-1]
        betas_2 = np.logspace(-10, -4, ntemps_2 - 1, endpoint=False)[::-1]

        betas = np.concatenate([betas_1, betas_2, np.array([0.0])])
        
        self.tempering_kwargs = {"betas": betas, "adaptive": False}
        self.oversample = oversample

        self.prior_transform_fn = {
            self.base_info.name: PriorTransformFn(xp.zeros(ngroups), xp.ones(ngroups), xp.zeros(ngroups), xp.ones(ngroups)),
            self.third_info.name: ThirdPriorTransformFn(xp.zeros(ngroups), xp.ones(ngroups), xp.zeros(ngroups), xp.ones(ngroups), xp.ones(ngroups), xp.ones(ngroups))
        }

        data_channels = [xp.zeros(self.data_length * self.ngroups, dtype=complex), xp.zeros(self.data_length * self.ngroups, dtype=complex)]
        psds = [xp.ones(self.data_length * self.ngroups, dtype=float), xp.ones(self.data_length * self.ngroups, dtype=float)]
        start_freq = xp.full(self.ngroups, int(1e-3 / self.df), dtype=np.int32)
        N_vals_in = xp.zeros(self.ngroups, dtype=int)
        d_d_all = xp.zeros(self.ngroups, dtype=float)
        self.N_max = int(self.data_length / 4)
        self.log_like_fn = {
            self.base_info.name: LogLikeFn(self.gb, data_channels, psds, start_freq, self.df, self.base_info.transform_fn, N_vals_in, self.data_length, d_d_all, **waveform_kwargs),
            self.third_info.name: LogLikeFn(self.gb_third, data_channels, psds, start_freq, self.df, self.third_info.transform_fn, N_vals_in, self.data_length, d_d_all, **waveform_kwargs)
        }

        self.currently_running_orig_id = [None for _ in range(self.ngroups)]
        self.currently_running_output_info = [None for _ in range(self.ngroups)]
        
        self.start_state = {}
        self.sampler = {}

        for info in [self.base_info, self.third_info]:
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
            self.start_state[info.name] = start_state

            # initialize sampler
            self.sampler[info.name] = ParaEnsembleSampler(
                info.ndim,
                self.nwalkers,
                self.ngroups,
                self.log_like_fn[info.name],
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
                prior_transform_fn=self.prior_transform_fn[info.name],
                provide_supplimental=True,
            )

        self.map_models = {"base": 0, "third": 1}

        self.current_model_index_product_space = xp.random.randint(2, size=(self.ngroups, self.ntemps, self.nwalkers))

    def run(self, convergence_iter_count):
        for fp in os.listdir(self.directory_in):
            generate_fp = f"pop_for_search_new_test_{self.seed_from_gen}_" + fp
            
            search_fp = f"{self.output_string}_{self.seed_from_gen}_" + fp[:-4] + ".h5"
            
            if search_fp not in os.listdir(self.directory_in3):
                raise FileNotFoundError(f"search_fp ({search_fp}) not found in {directory_in3}.")

            out_fp = f"{self.output_string}_{self.seed_from_gen}_evidence_" + fp[:-4] + ".h5"

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

            info_iterator = run_information(self.gb_third, nbin, data, data_orig, self.m3_lim, out_fp, self.directory_out, self.N_max, self.data_length, self.min_chirp_mass, self.max_chirp_mass, self.oversample, self.transform_fn, search_fp)
        
            self.run_mcmc(nbin, convergence_iter_count, info_iterator, out_fp)

    def setup_next_source(self, info_iterator):
        try:
            (orig_id, index, injection_params, fd, data_channels_tmp, psd_tmp, start_freq, f_lims, fdot_lims, P2_lims, N_val, last_search_sample, output_info) = next(info_iterator)
            
            d_d = 4.0 * self.df * np.sum(np.asarray(data_channels_tmp).conj() * np.asarray(data_channels_tmp) / np.asarray(psd_tmp)).item().real
            
            data_channels_tmp = [xp.asarray(tmp).copy() for tmp in data_channels_tmp]
            psd_tmp = [xp.asarray(tmp).copy() for tmp in psd_tmp]
                
            self.gb.d_d = d_d 
            self.gb_third.d_d = d_d

            factor = 1e-5
            cov = np.ones(self.third_info.ndim) * 1e-3
            cov[1] = 1e-7
            max_iter = 2000
            start_like = np.zeros((self.ntemps, self.nwalkers))
  
            while np.std(start_like[0]) < 7.0:
                jj = 0
                logp = np.full_like(start_like, -np.inf).flatten()
                tmp_fs = np.zeros((self.ntemps * self.nwalkers, self.third_info.ndim))
                fix = np.ones((self.ntemps * self.nwalkers), dtype=bool)
              
                while jj < max_iter and np.any(fix):
                    # left off here. need to fix 
                    # - transform function for prior needs to transform output points as well
                    tmp_fs[fix] = (injection_params[np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])] * (1. + factor * cov * np.random.randn(self.nwalkers * self.ntemps, 13)))[fix]

                    tmp = tmp_fs.copy()
                    
                    # map points
                    for ind, lims in [[1, f_lims], [2, fdot_lims], [-2, P2_lims]]:
                        tmp[:, ind] = (tmp[:, ind] - lims[0]) / (lims[1] - lims[0])

                    if np.any(tmp[:, 1] < 0.0):
                        breakpoint()

                    logp = self.third_info.priors["gb"].logpdf(tmp).get()
                    fix = np.isinf(logp)
                    jj += 1

                if "N" in self.waveform_kwargs:
                    self.waveform_kwargs.pop("N")

                tmp_fs_in = self.transform_fn["third"]["gb"].both_transforms(tmp_fs)
                xp.get_default_memory_pool().free_all_blocks()
                # tmp_fs_in[:] = tmp_fs_in[0][None, :]
                start_like = self.gb_third.get_ll(tmp_fs_in, data_channels_tmp, psd_tmp, data_length=self.data_length, start_freq_ind=start_freq, N=N_val, **waveform_kwargs)
                
                xp.cuda.runtime.deviceSynchronize()
                # print(np.std(start_like[0]))
                if np.any(np.isnan(start_like)):
                    breakpoint()
                tmp_fs = tmp_fs.reshape(ntemps, nwalkers, self.third_info.ndim)
                start_like = start_like.reshape(ntemps, nwalkers)
                logp = logp.reshape(ntemps, nwalkers)
                
                factor *= 1.5

                # print(np.std(start_like[0]))
            
            # setup in ParaState
            # get first group not running
            new_group_ind = self.start_state["third"].groups_running.argmin().item()
            self.start_state["third"].groups_running[new_group_ind] = True

            self.start_state["third"].branches["gb"].coords[new_group_ind] = xp.asarray(tmp_fs)
            self.start_state["third"].log_prior[new_group_ind] = xp.asarray(logp)
            self.start_state["third"].log_like[new_group_ind] = xp.asarray(start_like)
            # self.start_state["third"].betas[new_group_ind] = xp.asarray(betas_in_here)

            if np.any(np.isnan(self.start_state["third"].log_like)):
                breakpoint()

            # data arrays are the same between third and base so no need to adjust base
            inds_slice = slice((new_group_ind) * self.data_length, (new_group_ind + 1) * self.data_length, 1)
            self.sampler["third"].log_like_fn.data[0][inds_slice] = data_channels_tmp[0]
            self.sampler["third"].log_like_fn.data[1][inds_slice] = data_channels_tmp[1]
            self.sampler["third"].log_like_fn.psd[0][inds_slice] = psd_tmp[0]
            self.sampler["third"].log_like_fn.psd[1][inds_slice] = psd_tmp[1]
            self.sampler["third"].log_like_fn.start_freq[new_group_ind] = start_freq

            self.sampler["third"].prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler["third"].prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler["third"].prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler["third"].prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler["third"].prior_transform_fn.P2_min[new_group_ind] = P2_lims[0]
            self.sampler["third"].prior_transform_fn.P2_max[new_group_ind] = P2_lims[1]
            self.sampler["third"].log_like_fn.N_vals[new_group_ind] = N_val
            self.sampler["third"].log_like_fn.d_d_all[new_group_ind] = d_d

            self.sampler["base"].log_like_fn.start_freq[new_group_ind] = start_freq
            self.sampler["base"].prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler["base"].prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler["base"].prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler["base"].prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler["base"].log_like_fn.N_vals[new_group_ind] = N_val
            self.sampler["base"].log_like_fn.d_d_all[new_group_ind] = d_d

            # get info from search

            # get first group not running
            self.start_state["base"].groups_running[new_group_ind] = True

            self.start_state["base"].branches["gb"].coords[new_group_ind] = xp.asarray(last_search_sample.branches["gb"].coords[0, :self.nwalkers, 0][None, :])
            self.start_state["base"].log_prior[new_group_ind] = xp.asarray(last_search_sample.log_prior[0][None, :self.nwalkers])
            self.start_state["base"].log_like[new_group_ind] = xp.asarray(last_search_sample.log_like[0][None, :self.nwalkers])
            # self.start_state["base"].betas[new_group_ind] = xp.asarray(betas_in_here)

            if np.any(np.isnan(self.start_state["base"].log_like)):
                breakpoint()

            self.currently_running_orig_id[new_group_ind] = orig_id
            self.currently_running_output_info[new_group_ind] = output_info

            return False

        except StopIteration:
            return True

    def product_space_operation(self, logP_dict):

        p_base_to_third = 0.5
        p_third_to_base = 1.0 - p_base_to_third

        base_to_third_forward_indicator = (2 * (self.current_model_index_product_space != 0).astype(int) - 1)
        # forward should be minus for proposal weight
        ratio = (
            np.log(p_base_to_third) * base_to_third_forward_indicator
            + np.log(p_third_to_base) * -1 * base_to_third_forward_indicator
            + base_to_third_forward_indicator * (logP_dict["third"] - logP_dict["base"])
        )

        accept = xp.log(xp.random.rand(*ratio.shape)) < ratio

        self.current_model_index_product_space[accept] = (self.current_model_index_product_space[accept] + 1) % 2
                
    def run_mcmc(self, nbin, convergence_iter_count, info_iterator, out_fp):

        run = True
        finish_up = False

        total_steps_for_evidence = 100
        number_old_evidences = 6

        old_bf = np.zeros((self.ngroups, number_old_evidences))
        current_old_bf_ind = np.zeros((self.ngroups,), dtype=int)
        current_old_bf_count = np.zeros((self.ngroups,), dtype=int)

        current_evidence_ind = 0
        current_evidence_estimate = {name: xp.full((self.ngroups,), np.nan) for name in ["base", "third"]}
        current_evidence_diff = xp.full((self.ngroups,), np.nan)
        evidence_log_like = {name: xp.full((self.ngroups, total_steps_for_evidence, self.ntemps), xp.nan) for name in ["base", "third"]}

        while run:
            finish_up = self.setup_next_source(info_iterator)

            # end if all are done
            if finish_up and np.all(~self.start_state["third"].groups_running):
                run = False
                return

            started_run = False
            running_inner = (xp.all(self.start_state["third"].groups_running) or finish_up)

            while running_inner:
                nsteps = 10
                started_run = True
                

                assert nsteps <= total_steps_for_evidence

                current_inds_fill_evidence = (xp.arange(nsteps) + current_evidence_ind) % total_steps_for_evidence

                logP = {}
                for template in ["base", "third"]:
                    self.start_state[template].log_like = None
                    self.start_state[template].log_prior = None
                    self.start_state[template].betas = None

                    self.sampler[template].backend.reset(*self.sampler[template].backend.reset_args, **self.sampler[template].backend.reset_kwargs)
                    # if template == "third":
                    #     breakpoint()
                    #     coords_in = {"gb": self.start_state[template].branches["gb"].coords[-1:]}
                    #     check = self.sampler[template].compute_log_prior(coords_in, groups_running=xp.arange(self.ngroups)[np.array([15])])
                    #     coords_in2 = {"gb": self.start_state[template].branches["gb"].coords}
                    #     check = self.sampler[template].compute_log_prior(coords_in, groups_running=xp.arange(self.ngroups)[:])
                    #     breakpoint()

                    print(template, self.start_state[template].groups_running.sum())
                    self.start_state[template] = self.sampler[template].run_mcmc(self.start_state[template], nsteps, burn=0, thin_by=10, progress=True, store=True)
                    # np.save(f"sample_check_{template}", self.sampler[template].get_chain())
                    
                    logP[template] = self.start_state[template].log_like * self.start_state[template].betas[:, :, None] + self.start_state[template].log_prior

                    evidence_log_like[template][:, current_inds_fill_evidence] = self.sampler[template].get_log_like(discard=self.sampler[template].backend.iteration - nsteps).mean(axis=-1).transpose(1, 0, 2)  # self.start_state[template].log_like.mean(axis=-1)[:, None, :]

                    # evidence_log_like[template] = 
                self.product_space_operation(logP)
                
                current_evidence_ind += nsteps
                current_evidence_ind %= total_steps_for_evidence

                # adjust
                adjust = xp.all(~xp.isnan(evidence_log_like["third"]), axis=(1, 2))
                
                if xp.any(adjust):
                    for template in ["third", "base"]:
                        
                        assert xp.all(self.start_state["base"].betas == self.start_state["third"].betas)
                        for grp in range(self.ngroups):
                            if adjust[grp]:
                                betas = self.start_state["third"].betas[grp]
                                logls = evidence_log_like[template][grp].mean(axis=0)
                                
                                logZ, dlogZ = thermodynamic_integration_log_evidence(betas.get(), logls.get())  # , block_len=50, repeats=100)
                                current_evidence_estimate[template][grp] = logZ.item()

                    current_evidence_diff[adjust] = (current_evidence_estimate["third"] - current_evidence_estimate["base"])[adjust]
                    print(current_evidence_diff)

                    old_bf[adjust.get(), current_old_bf_ind[adjust.get()]] = 2 * current_evidence_diff[adjust.get()].get()
                    current_old_bf_ind[adjust.get()] = (current_old_bf_ind[adjust.get()] + 1) % old_bf.shape[1]
                    current_old_bf_count[adjust.get()] += 1
                    
                check_convergence = current_old_bf_count >= number_old_evidences
                end = np.full(self.ngroups, False)
                end[check_convergence] = np.all(np.abs(old_bf[check_convergence] - np.mean(old_bf[check_convergence], axis=-1)[:, None]) < 0.03, axis=-1) & (np.abs(np.sign(np.diff(old_bf[check_convergence], axis=-1)).sum(axis=-1)) < number_old_evidences - 1)
                
                print(end, current_evidence_diff)
                if np.any(end) > 0:
                    running_inner = False
                # print(iters_at_max, start_state.groups_running.sum().item(), now_max_log_like[:10])
            
            if started_run:

                # 
                # which groups ended
                end = np.where(end)[0]
                for end_i in end:
                    base_evidence = current_evidence_estimate["base"][end_i].item()
                    third_evidence = current_evidence_estimate["third"][end_i].item()

                    two_logBF = 2 * (third_evidence - base_evidence)

                    orig_id = self.currently_running_orig_id[end_i]

                    for template in [self.third_info, self.base_info]:
                        group_name = str(int(orig_id)) + "_" + template.name
                        backend_tmp = HDFBackend(out_fp, name=group_name)
                        backend_tmp.reset( self.nwalkers, template.ndim, ntemps=self.ntemps, branch_names=["gb"],)
                        backend_tmp.grow(self.sampler[template.name].backend.iteration, None)

                        with h5py.File(backend_tmp.filename, "a") as fp_save:
                            fp_save[group_name].attrs["iteration"] = self.sampler[template.name].iteration
                            fp_save[group_name]["chain"]["gb"][:] = self.sampler[template.name].get_chain()[:, end_i][:, :, :, None, :]
                            fp_save[group_name]["log_like"][:] = self.sampler[template.name].get_log_like()[:, end_i]
                            fp_save[group_name]["log_prior"][:] = self.sampler[template.name].get_log_prior()[:, end_i]
                            group_new = fp_save[group_name].create_group("keep_info")
                            for key, value in self.currently_running_output_info[end_i].items():
                                group_new.attrs[key] = value

                            group_new.attrs["base_evidence"] = base_evidence
                            group_new.attrs["third_evidence"] = third_evidence
                            group_new.attrs["2logBF"] = two_logBF
                                
                    # reset everything for the one that has finished
                    old_bf[end_i] = np.nan
                    current_old_bf_ind[end_i] = 0
                    current_old_bf_count[end_i] = 0

                    for template in ["base", "third"]:
                        current_evidence_estimate[template][end_i] = np.nan
                        evidence_log_like[template][end_i] = np.nan
                        self.start_state[template].groups_running[end_i] = False
                    
                    current_evidence_diff[end_i] = np.nan
        
                    self.currently_running_orig_id[end_i] = None
                    xp.get_default_memory_pool().free_all_blocks()

            if xp.all(~self.start_state["third"].groups_running) and finish_up:
                run = False

    
if __name__ == "__main__":
    st = time.perf_counter()
    gpu = 3
    setDevice(gpu)
    use_gpu = True

    snr_lim = 5.0
    ll_diff_lim = 2.0
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

    nwalkers = 40
    ntemps = 124
    ngroups = 500
    
    data_length = 8192
    runner = RunEvidenceProcedure(dt, Tobs, directory_in, directory_in2, directory_in3, seed_from_gen, directory_out, output_string, waveform_kwargs, ngroups, ntemps, nwalkers, data_length, snr_lim, m3_lim, ll_diff_lim, use_gpu=use_gpu)

    runner.run(convergence_iter_count)

    print("end:", fp)
    et = time.perf_counter()
    print("TOTAL TIME:", et - st)
        
        

        

       
        