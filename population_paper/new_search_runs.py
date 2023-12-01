
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

from gbsetups import BaseTemplateSetup, LogLikeFn
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


class PriorTransformFn:
    def __init__(self, f_min, f_max, fdot_min, fdot_max):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = f_min, f_max, fdot_min, fdot_max

    def adjust_logp(self, logp, groups_running):
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1. / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1. / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]
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
        return


def run_information(gb_third, nbin, data, data_orig, m3_lim, out_fp, directory_out, N_max, data_length, min_chirp_mass, max_chirp_mass, oversample, transform_fn):
    for i in range(nbin):
        index = int(data["index"][i])
        orig_id = int(data["orig_id"][i])

        #if index != 2783:
        #    continue
        m3 = data_orig["M3"][int(index)]
        if m3 > m3_lim:
            continue 

        
        template = "base"

        if out_fp in os.listdir(directory_out):
            with h5py.File(directory_out + out_fp, "a") as f:
                if str(int(orig_id)) in list(f):
                    print(f"{int(orig_id)} already in file {out_fp} so not running.")
                    continue
    
        # extract_snr_base and snr_base were switched
        
        if (
            data[f"lldiff_{template}_marg"][i] < ll_diff_lim
            or data[f"extract_snr_{template}"][i] < snr_lim
        ):
            continue

        injection_params = np.array([data[key][i] for key in data.dtype.names[:14]])
        injection_params[7] = injection_params[7] % (2 * np.pi)

        template_params = injection_params[:9].copy()
        

        # maximized phase shift
        template_params[4] = (template_params[4] + data[f"phase_shift_{template}"][i]) % (2 * np.pi)

        print("lldiff_base_marg:", data[f"lldiff_{template}_marg"][i])

        # for later
        #with h5py.File("search_output_FINAL_POPULATION_v3.hdf5") as fp2:
        #    check = str(i) in fp2["search info"]
        """
        if check is False:
            if fp in os.listdir(folder):
                print(f"{i} not in orginal run.")
                continue

        if check:
            if template == "base":
                with h5py.File("search_output_FINAL_POPULATION_v3.hdf5") as fp2:
                    lls = fp2["search info"][str(i)]["ll"][:]

                lp_max = lls.max()

                if lp_max > -2.0:
                    run_fddot = False
                else:
                    run_fddot = True

                lldiff_base_marg = -data[f"lldiff_{template}_marg"][i]

                #if False:
                if lp_max < lldiff_base_marg - 1:
                    oversample = 8
                    print("lldiff_base_marg:", lldiff_base_marg, "lp_max:", lp_max)
                    if fp in os.listdir(folder):
                        continue
                    # shutil.copy(folder + fp, folder + f"back_1_{fp}")
                else:
                    continue
            else:
                continue
        """
        amp = np.exp(injection_params[0])
        f0 = injection_params[1] * 1e-3
        fdot0 = injection_params[2]

        fdot_min = get_fdot(f0, Mc=min_chirp_mass)
        fdot_max = get_fdot(f0, Mc=max_chirp_mass)

        N_found_base = get_N(amp, f0, Tobs, oversample=oversample).item()

        A2 = data["A2"][i]
        varpi  = data["omegabar"][i]
        e2  = data["e2"][i]
        P2  = data["P2"][i]
        T2  = data["T2"][i]

        N_found_third = gb_third.special_get_N(amp, f0, Tobs, A2,
                                                varpi,
                                                e2,
                                                P2,
                                                T2,oversample=oversample)

        N_found = np.max([N_found_base, N_found_third])
        waveform_kwargs["N"] = N_found

        # adjust for the few that are over 1 solar mass
        try:
            assert fdot0 >= fdot_min and fdot0 <= fdot_max
        except AssertionError:
            fdot_max = fdot0 * 1.05
            assert fdot0 >= fdot_min and fdot0 <= fdot_max

        max_f = 20.0  # mHz
        if f0 * 1e3 > max_f:
            max_f = 30.0  # mHz

        f_min = f0 * 0.999 * 1e3
        f_max = f0 * 1.001 * 1e3
        f_lims = [f_min, f_max]

        fdot_lims = [fdot_min, fdot_max]

        params_inner = transform_fn["gb"].both_transforms(injection_params[np.array([0, 1, 2, 4, 5, 6, 7, 8])])
        params_inj_in = np.concatenate([params_inner, injection_params[9:]])
        if waveform_kwargs["N"] > N_max:
            if waveform_kwargs["N"] > N_max:  # 16384:
                print(f"ID {int(orig_id)} (index: {int(index)}) has too high of N value so not running.")
                continue
            waveform_kwargs["use_c_implementation"] = False
            if isinstance(waveform_kwargs["N"], np.ndarray):
                assert len(waveform_kwargs["N"]) == 1
                waveform_kwargs["N"] = waveform_kwargs["N"].item()

        A_inj, E_inj = gb_third.inject_signal(*params_inj_in, **waveform_kwargs)

        start_freq = int(int(f0 / df) - data_length / 2)
        fd = np.arange(start_freq, start_freq + data_length) * df

        data_channels = [A_inj[start_freq:start_freq + data_length].copy(), E_inj[start_freq:start_freq + data_length].copy()]

        AE_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model="sangria", includewd=Tobs / YEAR)
        psd = [AE_psd, AE_psd]

        info_out = {name: value for name, value in zip(data.dtype.names, data[i])}
        yield (orig_id, index, injection_params, fd, data_channels, psd, start_freq, f_lims, fdot_lims, N_found, info_out)
        

class RunSearchProcedure(BaseTemplateSetup):
    def __init__(self, dt, Tobs, directory_in, directory_in2, seed_from_gen, directory_out, output_string, waveform_kwargs, ngroups, ntemps, nwalkers, data_length, snr_lim, m3_lim, ll_diff_lim, oversample=4, use_gpu=True):

        self.snr_lim, self.m3_lim, self.ll_diff_lim = snr_lim, m3_lim, ll_diff_lim
        
        self.output_string = output_string
        # ## Setup all the parameters and waveform generator
        self.gb = GBGPU(use_gpu=use_gpu)
        self.gb_third = GBGPUThirdBody(use_gpu=use_gpu)
        self.waveform_kwargs = waveform_kwargs

        assert Tobs / dt == float(int(Tobs / dt))
        self.dt = dt
        self.Tobs = Tobs
        self.df = 1. / Tobs
        self.directory_in, self.directory_in2, self.seed_from_gen = directory_in, directory_in2, seed_from_gen
        self.directory_out = directory_out

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
        self.ngroups, self.ntemps, self.nwalkers = ngroups, ntemps, nwalkers
        self.ndim = 8
        self.data_length = data_length
        self.tempering_kwargs = {"ntemps": ntemps, "Tmax": np.inf}
        self.periodic = PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}})
        self.oversample = oversample

        self.prior_transform_fn = PriorTransformFn(xp.zeros(ngroups), xp.ones(ngroups), xp.zeros(ngroups), xp.ones(ngroups))

        data_channels = [xp.zeros(self.data_length * self.ngroups, dtype=complex), xp.zeros(self.data_length * self.ngroups, dtype=complex)]
        psds = [xp.ones(self.data_length * self.ngroups, dtype=float), xp.ones(self.data_length * self.ngroups, dtype=float)]
        start_freq = xp.full(self.ngroups, int(1e-3 / self.df), dtype=np.int32)
        N_vals_in = xp.zeros(self.ngroups, dtype=int)
        d_d_all = xp.zeros(self.ngroups, dtype=float)
        self.N_max = int(self.data_length / 4)
        self.log_like_fn = LogLikeFn(self.gb, data_channels, psds, start_freq, self.df, self.transform_fn, N_vals_in, self.data_length, d_d_all, **waveform_kwargs)
        
        self.currently_running_index_orig_id = [None for _ in range(self.ngroups)]
        
        # initialize sampler
        self.sampler = ParaEnsembleSampler(
            self.ndim,
            self.nwalkers,
            self.ngroups,
            self.log_like_fn,
            self.priors,
            tempering_kwargs=self.tempering_kwargs,
            args=[],
            kwargs={},
            gpu=gpu,
            periodic=self.periodic,
            backend=None,
            update_fn=None,
            update_iterations=-1,
            stopping_fn=None,
            stopping_iterations=-1,
            name="gb",
            prior_transform_fn=self.prior_transform_fn,
            provide_supplimental=True,
        )

        coords = xp.zeros((self.ngroups, self.ntemps, self.nwalkers, self.ndim))

        branch_supp_base_shape = (self.ngroups, self.ntemps, self.nwalkers)

        data_inds = xp.repeat(xp.arange(self.ngroups, dtype=np.int32)[:, None], self.ntemps * self.nwalkers, axis=-1).reshape(self.ngroups, self.ntemps, self.nwalkers) 
        branch_supps = {"gb": BranchSupplimental(
            {"data_inds": data_inds}, base_shape=branch_supp_base_shape, copy=True
        )}

        groups_running = xp.zeros(self.ngroups, dtype=bool)
        self.start_state = ParaState({"gb": coords}, groups_running=groups_running, branch_supplimental=branch_supps)
        self.start_state.log_prior = xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        self.start_state.log_like = xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        self.start_state.betas = xp.ones((self.ngroups, self.ntemps))

    def run(self, convergence_iter_count):

        for fp in os.listdir(self.directory_in):
            generate_fp = f"pop_for_search_new_test_{self.seed_from_gen}_" + fp
            out_fp = f"{self.output_string}_{self.seed_from_gen}_" + fp[:-4] + ".h5"
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

            nbin = len(data)

            info_iterator = run_information(self.gb_third, nbin, data, data_orig, self.m3_lim, out_fp, self.directory_out, self.N_max, self.data_length, self.min_chirp_mass, self.max_chirp_mass, self.oversample, self.transform_fn)
        
            self.run_mcmc(nbin, convergence_iter_count, info_iterator, out_fp)

    def setup_next_source(self, info_iterator):
        try:
            (orig_id, index, injection_params, fd, data_channels_tmp, psd_tmp, start_freq, f_lims, fdot_lims, N_val, keep_info) = next(info_iterator)
            
            d_d = 4.0 * self.df * np.sum(np.asarray(data_channels_tmp).conj() * np.asarray(data_channels_tmp) / np.asarray(psd_tmp)).item().real
            
            data_channels_tmp = [xp.asarray(tmp) for tmp in data_channels_tmp]
            psd_tmp = [xp.asarray(tmp) for tmp in psd_tmp]
                
            self.gb.d_d = d_d 
            factor = 1e-5
            cov = np.ones(self.ndim) * 1e-3
            cov[1] = 1e-7
            max_iter = 2000
            start_like = np.zeros((self.ntemps, self.nwalkers))
            while np.std(start_like[0]) < 7.0:
                logp = np.full_like(start_like, -np.inf).flatten()
                tmp_fs = np.zeros((self.ntemps * self.nwalkers, self.ndim))
                fix = np.ones((self.ntemps * self.nwalkers), dtype=bool)
                jj = 0
                while jj < max_iter and np.any(fix):
                    # left off here. need to fix 
                    # - transform function for prior needs to transform output points as well
                    tmp_fs[fix] = (injection_params[np.array([0, 1, 2, 4, 5, 6, 7, 8])] * (1. + factor * cov * np.random.randn(self.nwalkers * self.ntemps, 8)))[fix]

                    tmp = tmp_fs.copy()
                    
                    # map points
                    tmp[:, 1] = (tmp[:, 1] - f_lims[0]) / (f_lims[1] - f_lims[0])
                    tmp[:, 2] = (tmp[:, 2] - fdot_lims[0]) / (fdot_lims[1] - fdot_lims[0])

                    if np.any(tmp[:, 1] < 0.0):
                        breakpoint()
                    logp = self.priors["gb"].logpdf(tmp).get()
                    fix = np.isinf(logp)
                    jj += 1

                if "N" in self.waveform_kwargs:
                    self.waveform_kwargs.pop("N")

                tmp_fs_in = self.transform_fn["gb"].both_transforms(tmp_fs)
                start_like = self.gb.get_ll(tmp_fs_in, data_channels_tmp, psd_tmp, start_freq_ind=start_freq, N=N_val, **waveform_kwargs)
                if np.any(np.isnan(start_like)):
                    breakpoint()
                tmp_fs = tmp_fs.reshape(ntemps, nwalkers, 8)
                start_like = start_like.reshape(ntemps, nwalkers)
                logp = logp.reshape(ntemps, nwalkers)
                
                factor *= 1.5
                # print(np.std(start_like[0]))
            
            # setup in ParaState

            # get first group not running
            new_group_ind = self.start_state.groups_running.argmin().item()
            self.start_state.groups_running[new_group_ind] = True

            self.start_state.branches["gb"].coords[new_group_ind] = xp.asarray(tmp_fs)
            self.start_state.log_prior[new_group_ind] = xp.asarray(logp)
            self.start_state.log_like[new_group_ind] = xp.asarray(start_like)
            self.start_state.betas[new_group_ind] = xp.asarray(self.sampler.base_temperature_control.betas)

            if np.any(np.isnan(self.start_state.log_like)):
                breakpoint()

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
            self.sampler.log_like_fn.N_vals[new_group_ind] = N_val
            self.currently_running_index_orig_id[new_group_ind] = orig_id

            self.sampler.log_like_fn.d_d_all[new_group_ind] = d_d

            self.output_info_store[new_group_ind] = keep_info

            return False

        except StopIteration:
            return True
                
    def run_mcmc(self, nbin, convergence_iter_count, info_iterator, out_fp):

        max_log_like = xp.full((self.ngroups,), -np.inf)
        now_max_log_like = xp.full((self.ngroups,), -np.inf)
        iters_at_max = xp.zeros((self.ngroups,), dtype=int)
        self.output_info_store = [None for _ in range(self.ngroups)]
        
        run = True
        finish_up = False

        while run:
            finish_up = self.setup_next_source(info_iterator)

            # end if all are done
            if finish_up and np.all(~self.start_state.groups_running):
                run = False
                return

            started_run = False
            running_inner = (xp.all(self.start_state.groups_running) or finish_up)
            while running_inner:
                started_run = True
                nsteps = 20
                self.start_state.log_like = None
                self.start_state.log_prior = None
                self.start_state = self.sampler.run_mcmc(self.start_state, nsteps, progress=False, store=False)

                now_max_log_like[self.start_state.groups_running] = self.start_state.log_like.max(axis=(1, 2))[(self.start_state.groups_running)]
                improved = (now_max_log_like > max_log_like)

                iters_at_max[(improved) & (self.start_state.groups_running)] = 0
                iters_at_max[(~improved) & (self.start_state.groups_running)] += 1
                max_log_like[(improved) & (self.start_state.groups_running)] = now_max_log_like[(improved) & (self.start_state.groups_running)]

                converged = iters_at_max > convergence_iter_count

                end = converged | (now_max_log_like > -2.0)

                if np.any(end):
                    running_inner = False

                self.start_state.groups_running[end] = False
                # print(iters_at_max, start_state.groups_running.sum().item(), now_max_log_like[:10])
            
            if started_run:
                # which groups ended
                end = np.where(end.get())[0]
                for end_i in end:
                    max_log_like[end_i] = -np.inf
                    now_max_log_like[end_i] = -np.inf
                    converged[end_i] = False
                    iters_at_max[end_i] = 0

                    orig_id = self.currently_running_index_orig_id[end_i]

                    output_state = State({"gb": self.start_state.branches["gb"].coords[end_i].get()}, log_like=self.start_state.log_like[end_i].get(), log_prior=self.start_state.log_prior[end_i].get(), betas=self.start_state.betas[end_i].get(), random_state=np.random.get_state())
                    backend_tmp = HDFBackend(out_fp, name=str(int(orig_id)))
                    backend_tmp.reset(
                        self.nwalkers,
                        8,
                        ntemps=self.ntemps,
                        branch_names=["gb"],
                    )
                    backend_tmp.grow(1, None)
                    
                    accepted = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
                    
                    backend_tmp.save_step(output_state, accepted)
                    with h5py.File(backend_tmp.filename, "a") as fp:
                        group_new = fp[str(int(orig_id))].create_group("keep_info")
                        for key, value in self.output_info_store[end_i].items():
                            group_new.attrs[key] = value
                        group_new.attrs["logl_max_mcmc"] = output_state.log_like.max()

                    self.currently_running_index_orig_id[end_i] = None
                    xp.get_default_memory_pool().free_all_blocks()

            if xp.all(~self.start_state.groups_running) and finish_up:
                run = False

    
if __name__ == "__main__":
    st = time.perf_counter()
    gpu = 7
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
    output_string = "testing_new_setup_2"
    waveform_kwargs = dict(N=None, dt=dt, T=Tobs, use_c_implementation=True)

    nwalkers = 50
    ntemps = 10
    ngroups = 150
    
    data_length = 8192
    runner = RunSearchProcedure(dt, Tobs, directory_in, directory_in2, seed_from_gen, directory_out, output_string, waveform_kwargs, ngroups, ntemps, nwalkers, data_length, snr_lim, m3_lim, ll_diff_lim, use_gpu=use_gpu)
    runner.run(convergence_iter_count)

    print("end:", fp)
    et = time.perf_counter()
    print("TOTAL TIME:", et - st)
        
        

        

       
        