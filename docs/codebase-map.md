_Last mapped: dc561ce · 2026-07-10; facts re-checked against 7f53087 plus the 2026-10 docstring pass on 2026-10-22 · regenerate when structure changes_

# GBGPU codebase map

## 1. What this is

`GBGPU` (package `gbgpu`) is the LISA Analysis Tools **GB-physics owner**: it computes
Galactic-binary (GB / UCB) gravitational waveforms and their LISA TDI
likelihoods on CPU and CUDA GPUs, plus a JAX mirror. It hosts two largely
independent waveform pipelines — a classic dense "FastGB" fast/slow
decomposition (the original `GBGPU` class), and a newer TDI-on-the-fly +
WDM/FD chunked-heterodyne + polyphase signal-heterodyne stack built on
LISAanalysistools' (LAT) generic response/TDI infrastructure — and supplies
the GB-specific likelihood engines consumed by LAT's global-fit sampler.

The production GB likelihood of the 9mo / 12mo global fits is the new stack:
LAT builds a `GBWDMComputations` (chunked het), wraps it with
`GBSignalHetComputations.for_band_engine` (sig-het v5 in-model scorer, with an
internal `GBLookupComputations` for the carrier reference, anchor and
information matrix) and drives it through `gb_likelihood.WDMBandLikelihoodEngine`.
On that path the classic `GBGPU` object is a container; its `run_wave` runs
only in the warm-start referee and on the monitor (CPU). User-facing write-up:
`docs/source/user/likelihood_path.rst`.

## 2. Layout

| Path | Contents |
|---|---|
| `src/gbgpu/gbgpu.py` (~3700 lines) | `GBGPUBase`/`GBGPU` — classic dense FastGB waveform + legacy FD likelihood (fast/slow decomposition, XYZ/AET, equal-arm response approximation). `run_wave` is Python/`xp` code by default; `get_ll` / `get_fstat_ll` / `swap_likelihood_difference` / `generate_global_template` and the `use_c_implementation=True` paths run the SharedMemory CUDA/CPU kernels (`GBGPU_WITH_SHAREDMEM=ON` builds only). Constructor requires `orbits`. |
| `src/gbgpu/thirdbody.py` | `GBGPUThirdBody(GBGPUBase)` — eccentric third-body extension of the classic path. |
| `src/gbgpu/gbcomps.py` | `GBWDMComputations` (chunked-het, thin subclass of LAT's `WDMComputationsBase`) + `GBFDComputations` (FD chunked-het analog, config-only ctor). New-stack Python frontends. |
| `src/gbgpu/gbsignalhetcomputations.py` | `GBSignalHetComputations` — polyphase signal-heterodyne likelihood (v2 ... v5 scorers; CPU + CUDA); `for_band_engine()` wraps a `GBWDMComputations` delegate for in-model use (the production path; the full data-loading constructor is CPU-only offline / validation). v5 is production (LAT settings pass `v5=1`; the GBGPU argument default is 0), with the CARRIER-only reference (`SIGHET_CP_REPR` default `carrier` on v5 engines) and the carrier fold (second moment + packet first moment c1) in the SYM stash layout for any channel-symmetric invC (`SIGHET_CARRIER_LAYOUT=sym|collapsed|full`; `_collapsed_carrier_fold`, `_window_dj`, `_c1_scale`); anchor correction `SIGHET_ANCHOR_CORRECT` (default 1). LOOKUP-BUILT reference (2026-10-05): `attach_lookup_reference` / `for_band_engine(lookup_table=...)` reads the carrier c0 / c1 from the WDM lookup table (`GBLookupComputations.make_carrier_reference`, kernel `gb_lookup_carrier_ref`) instead of the FD transform (`SIGHET_REF_BUILD=auto|fd|lookup`); `anchor_engine` / `SIGHET_ANCHOR_ENGINE=chunked|lookup`. The carrier fold bins with O(Nt) segment sums. The sig-het F-stat (`setup_fstat_references` / `get_fstat_ll_wdm`, the grid-fit path) uses the same carrier reference + SYM Gram fold (`SIGHET_FSTAT_CARRIER`, default on). `information_matrix` has three routes: `SIGHET_INFOMAT_ENGINE=lookup` (2026-10-05; values unset/`default`/`lookup`, anything else raises) → the lookup Gram below (needs the attached table; no in-model block or slots); else `SIGHET_INFOMAT=1` + live in-model block + `data_index` slots → second differences of the sig-het ln L; else the chunked delegate's swap Gram. `SIGHET_SETUP_PROFILE=1` (default off): device-synced per-phase `setup_in_model` timer (`_prof_tick`; gather / checks / ref_build / fold / stash / anchor), one INFO line per `_PROF_EVERY` = 200 calls with ms/call and ms/ref. |
| `src/gbgpu/gblookupcomputations.py` | `GBLookupComputations` — the fused direct-to-WDM LOOKUP scorer (`gb_lookup_get_ll`: response at control points, Re/Im splines, one table read per pixel/layer at the common carrier + K1 term; `k_coarse` reads the table at the control points and interpolates; `GB_LOOKUP_VERBOSE=1` occupancy report); `make_carrier_reference` = the sig-het v5 carrier reference from the same kernel (unit envelope). `information_matrix` = the reference-free Fisher / Gram `<dh_a|dh_b>` (get_ll_wdm's inner product, raw physical ICRS params): central differences of `fill_lookup` into a per-source `2 num_m_layers + 4`-layer slab, one invC contraction per batch, symmetric, on the comp's device; default steps `info_matrix_param_eps()` = amp 1e-25, f0 / fdot / fddot `1e-4 / Tobs^k`, angles 1e-4 rad (`INFOMAT_PHASE_STEP`), ~1e-4 of the exact-template Gram at 30 d - 6 months. Consumed by LAT's `GB_CHOL_CACHE` (`lisatools.globalfit.moves.gb_chol_cache._CholCache`: refreshes every alive source in one batch through the sig-het `SIGHET_INFOMAT_ENGINE=lookup` route; the 9mo / 12mo launchers set it). On the production runs a `GBLookupComputations` lives INSIDE the sig-het engine (`for_band_engine(lookup_table=...)` -> `attach_lookup_reference`; `SIGHET_REF_BUILD` / `SIGHET_ANCHOR_ENGINE` / `SIGHET_INFOMAT_ENGINE` all `lookup` on the launchers). It has no `force_backend` (takes the delegate's flavor) and its methods refuse `convert_to_ra_dec=True`. `GBLookupWDMComputations` — a `GBWDMComputations` subclass with only `get_ll_wdm` swapped (the fit integration form, `GB_LIKELIHOOD=lookup` in LAT `gb_no_fg`; not set by the launchers). Needs a lookup table at the run's layer duration (LAT `wdm_lookup_store` GB recipe, `resolve_gb_lookup_table`). |
| `src/gbgpu/gb_likelihood.py` | Domain-agnostic **band likelihood engines** (`FDBandLikelihoodEngine`, `WDMBandLikelihoodEngine`, `make_band_likelihood_engine`) consumed by LAT's global-fit GB special move (via `gbbands.make_routed_band_engine`). On the 9mo / 12mo runs the WDM engine's comp is the sig-het wrapper. Moved here from `lisatools.globalfit.moves.gb_likelihood` (2026-07); that path is now a deprecation shim. `hessian` answers on no engine; `get_ll_grad` fails on the WDM C++ backends (`gb_wdm_het_get_ll_grad` is not bound). |
| `src/gbgpu/wdm_het.py` | Deprecation shim; real content lives in `lisatools.wdm_het` (moved out so BBHx/SOBBH can share it without depending on GBGPU). |
| `src/gbgpu/parallelbase.py` (+ `src/gbgpu/utils/parallelbase.py`) | `GBGPUParallelModule` (backend-prefix resolver, prefixes bare backend tags with `gbgpu_`). `utils/parallelbase.py` is now a re-export shim of it (imported by `gbgpu.py`) — see §7. |
| `src/gbgpu/utils/` | `utility.py` (physics helpers: `get_N`, `get_fdot`, `AET`, chirp-mass conversions; re-exports every name of `lisatools.utils.constants`, values from `lisaconstants`), `citation.py` (BibTeX strings of the `citation` properties), `exceptions.py` (nothing in GBGPU raises these). `config.py` (unimportable: `KNOWN_BACKENDS` / `.globals` missing) and `citations.py` (imported by nothing) are FEW-template leftovers, TODO-tagged. There is no `gbgpu.utils.constants` (LAT `globalfit/buildcatalog.py` still imports it). |
| `src/gbgpu/cutils/` | C++/CUDA source + nanobind bindings (see §5). Copies of LAT's `Detector.{hpp,cu}`/`global.hpp` land here at CMake-configure time (gitignored; LAT is the source of truth). `lisaconstants_values.h` is GENERATED by LAT's `python -m lisatools.utils.lisaconstants_header` and committed (checked by `tests/test_lisaconstants_values.py`). |
| `src/gbgpu/jax/` | JAX mirror: `sources/ucb.py` (`JaxUCBSource`), `wdm/{kernels,heterodyne_kernels,signal_het_kernels,fast_inner_heterodyne,computation_group}.py`, `tdi_on_the_fly.py`, `wrappers.py`. Gated on `import jax`; no `gbgpu_jax` backend is registered. `wrappers.py` and `computation_group.py` are unwired adapters (only a LAT dev script uses `GBComputationGroupWrapJAX`); `tdi_on_the_fly.sobbh_run_wave_tdi` / `wrappers.SOBBHTDIonTheFlyWrapJAX` import the removed `gbgpu.jax.sources.sobbh` (now `bbhx.jax.sources.sobbh`). |
| `tests/` | `test_gbgpu.py` (classic `GBGPU` waveform smoke test), `test_thirdbody.py`, and the sig-het / lookup suites: `test_sighet_cp_reim.py` (control-point representations, v5 ratio deltas), `test_sighet_carrier_collapse.py` (carrier fold: moments, B2 / c1 exactness, sym / collapsed / full layouts, negative controls), `test_sighet_fstat_carrier.py`, `test_gb_lookup_kernel.py`, `test_lookup_information_matrix.py` (lookup Gram vs the exact-template Gram, engine knob, device-array inputs, holder guards, `SIGHET_SETUP_PROFILE`; cluster-only GPU == CPU class), cluster-only GPU parity `test_sighet_carrier_gpu.py`, plus window / mirror / phase-max / budget / infomat / `test_lisaconstants_values.py` tests. GPU classes skip without CUDA; `SIGHET_GPU_TEST_BACKEND` picks the backend on the cluster. CI runs `python -m unittest tests/test*`. Partly stale on dev: `test_gbgpu.py` / `test_thirdbody.py` pass the retired `EqualArmlengthOrbits(use_gpu=...)` kwarg (swallowed by `**kwargs`) and the `force_backend="gpu"` alias, and (per the docstring pass) `(9, num_bin)` params and a list-of-channels `get_ll` data argument. |
| `examples/` | Two notebooks (copied into `docs/source/` by `conf.py` at build time): `GBGPU_tutorial.ipynb` (classic FastGB path; stale on dev: `GBGPU(force_backend="cpu")` without `orbits`, `information_matrix` without `psd` / `data_length` and with `(9, num_bin)` params, hook named `add_to_argS`) and `fast_likelihood_tutorial.ipynb` (chunked-het `GBWDMComputations`; stale where it calls the sig-het class "on the roadmap"). |
| `docs/source/` | Sphinx: `index.rst` (includes `README.md` through `readme.md`), `user/likelihood_path.rst` (production WDM / sig-het chain, `force_backend` rules, the shared-memory build option), API pages `user/{main,derivedwaves,computations,backends,jax,utils}.rst`. `README.rst` is regenerated from `README.md` by `conf.py` (pypandoc) at every build. |
| Cython | **Fully retired.** No `.pyx`/`.pxd` files remain; the old `gbgpu_utils_wrap.pyx` / `sharedmemgbgpu.pyx` CMake rules are commented out in `cutils/CMakeLists.txt`. `pyproject.toml` no longer lists Cython (build requires: `numpy`, `lisaanalysistools`, `scikit_build_core`, `nanobind==2.12.0`). |

## 3. Core abstractions

Two parallel waveform/likelihood stacks coexist:

```
CLASSIC (dense FastGB)                    NEW STACK (TDI-on-the-fly + WDM/FD/sig-het)
────────────────────────                  ──────────────────────────────────────────
GBGPUBase (abc)                           C++: LISATDIonTheFly (LAT)
 └─ GBGPU                                        └─ GBTDIonTheFly (GBGPU, cutils/gb_tdi_on_the_fly.hh)
     └─ GBGPUThirdBody                                 UCB amp/phase/f/fdot closed forms
run_wave / get_ll / fill_global_template /        GBComputationGroup — gb_fd_*, gb_wdm_het_*, gb_signal_het_*,
swap_likelihood_difference / information_matrix     gb_lookup_* kernels
  → self.backend.sharedmem.SharedMemory*_wrap    Python frontends (cutils/__init__.py backend composition):
    (GBGPUComputationWrap methods; Python          GBWDMComputations   (chunked-het, WDM domain)
     run_wave needs no kernel)                       GBFDComputations   (chunked-het, FD domain)
                                                     GBSignalHetComputations (sig-het v2..v5; production
                                                       = for_band_engine() around a GBWDMComputations)
                                                     GBLookupComputations (lookup scorer; inside sig-het
                                                       on the runs) / GBLookupWDMComputations

                          gb_likelihood.py: FDBandLikelihoodEngine / WDMBandLikelihoodEngine
                          wrap {GBFDComputations, a WDM comp: the sig-het wrapper in production,
                          or GBWDMComputations / GBLookupWDMComputations} for LAT's global-fit
                          GB special move (make_band_likelihood_engine() dispatches on the
                          basis_settings type: FDSettings → FD engine, WDMSettings → WDM engine)
```

Naming collision to watch for: `lisatools.response.tdionfly.GBTDIonTheFly` (a
**Python** class, `TDIonTheFly` subclass, lives in LAT) is a *different class*
from GBGPU's C++ `GBTDIonTheFly` (nanobind, subclass of LAT's
`LISATDIonTheFly`). `gbsignalhetcomputations.py` imports the LAT Python one
for its `for_band_engine()` reference-signal construction while
`gb_tdi_on_the_fly.hh` defines the C++ one — grep carefully.

JAX mirrors the new stack's method surface via duck-typed wrapper classes
(`GBComputationGroupWrapJAX` in `jax/wdm/computation_group.py`) so
orchestration code doesn't need per-backend branches. They are unwired: no
registered backend exposes them (see §5), and the global fit never imports
`gbgpu.jax`.

## 4. Public API / entry points

- **Classic dense waveform + likelihood** (matches the README/tutorials):
  ```python
  from gbgpu.gbgpu import GBGPU
  from lisatools.detector import EqualArmlengthOrbits
  gb = GBGPU(orbits=EqualArmlengthOrbits(...), force_backend="cpu")  # orbits REQUIRED
  gb.run_wave(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, N=N, T=T, dt=dt)
  gb.get_ll(params, data_minus_template, psd, ...)   # params (num_bin, 9); sharedmem kernels
  gb.fill_global_template(...) / gb.swap_likelihood_difference(...)
  gb.information_matrix(params, psd, ..., data_length=...)
  ```
- **New-stack chunked-het / FD likelihoods** (used by global-fit GB special move):
  ```python
  from gbgpu.gbcomps import GBWDMComputations, GBFDComputations
  comp = GBWDMComputations(wdm_settings, t_ref=..., ..., force_backend="cpu")  # or "cuda13x"; no "jax" backend
  comp.fill_global_wdm(...) / comp.get_ll_wdm(...) / comp.get_swap_ll_wdm(...)
  ```
- **Signal-heterodyne** (production: `for_band_engine`, CPU or CUDA, inherits the delegate's backend):
  ```python
  from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
  sig = GBSignalHetComputations.for_band_engine(comp, nt_layer=..., v3_n_nodes=64, v4_knots=128,
                                                v4_band=16, v5=1, lookup_table=..., anchor_engine="lookup")
  sig.setup_in_model(holder, params_ref, slots); sig.get_ll_wdm(params, holder, data_index=slots)
  # offline / validation: GBSignalHetComputations(data_td, ref_params, Nf=.., Nt=.., ..., force_backend="cpu")
  ```
- **Direct-to-WDM lookup scorer**: `gbgpu.gblookupcomputations.GBLookupComputations(comp, table)` (no `force_backend`).
- **Global-fit band-engine factory** (the layer LAT's sampler actually calls):
  ```python
  from gbgpu.gb_likelihood import make_band_likelihood_engine
  engine = make_band_likelihood_engine(basis_settings, gb_fd_comp=..., gb_wdm_comp=..., ...)
  ```
- **Physics/param helpers**: `gbgpu.utils.utility.{get_N, get_fdot, get_chirp_mass, AET, ...}`.
- **Backend introspection**: `gbgpu.get_backend("cpu"|"cuda12x"|...)`, `gbgpu.has_backend(...)`.

Note: only the classic path's per-backend low-level kernels are reachable via
`self.backend.sharedmem` / `self.backend.get_ll` / `self.backend.fill_global`
(pass-throughs onto a `GBGPUComputationWrap` instance); the new stack reaches
its native symbols via `self.backend.GBTDIonTheFlyWrap` /
`self.backend.GBComputationGroupWrap`.

## 5. Backend structure

- **CPU / CUDA C++**: single nanobind module per backend, `cgbgpu`, shipped as
  `gbgpu_backend_{cpu,cuda11x,cuda12x,cuda13x}.cgbgpu`. Despite comments
  throughout `binding_gbgpu.{hpp,cxx}` still saying "pybind11 module surface"
  (leftover from before the LISA Analysis Tools–wide Phase 3M pybind11→nanobind
  migration), the actual bindings use `#include <nanobind/nanobind.h>` and
  `NB_MODULE(cgbgpu, m)` — this is nanobind, not pybind11. Treat the
  "pybind11" wording in that file's comments as stale.
- One `.so` per backend contains: the GB-specific wrapper class
  (`GBGPUComputationWrap{CPU,GPU}` — hosts the legacy `gbgpu_utils` +
  `SharedMemoryGBGPU` methods), `GBTDIonTheFlyWrap{CPU,GPU}` +
  `GBComputationGroupWrap{CPU,GPU}` (new stack), **and** copy-compiled LAT
  sources (`Detector.cu`, `lat_tdi_on_the_fly.cu`) plus GBT's `Interpolate.cu`
  (for the LAPACKE cubic-spline solver used in orbit/signal splines) so the
  virtual-class typeinfo and LAPACKE-dependent symbols live in GBGPU's own
  `.so` — see `cutils/CMakeLists.txt`.
- CPU/GPU class-name aliasing follows the LISA Analysis Tools–wide rule: `GBTDIonTheFly` →
  `GBTDIonTheFlyGPU`/`CPU`, `GBComputationGroup` →
  `GBComputationGroupGPU`/`CPU`, and their `*Wrap` counterparts, toggled on
  `__CUDA_COMPILATION__`/`__CUDACC__` in both `gb_tdi_on_the_fly.hh` and
  `binding_gbgpu.hpp`.
- **Backend selection**: `gbgpu/__init__.py` registers `GBGPUCpuBackend`,
  `GBGPUCuda{11,12,13}xBackend` (subclasses composing LAT's
  `LISAToolsBackend`-derived methods with GB-specific ones —
  `cutils/__init__.py`) under names `gbgpu_cpu`/`gbgpu_cuda11x`/etc in
  `gpubackendtools`'s `Globals().backends_manager`. `gbgpu.get_backend("cpu")`
  auto-prefixes to `gbgpu_cpu` (and resolves `"cuda"` / `"gpu"` to the first
  available CUDA flavor); `gbgpu.has_backend` does not resolve those aliases
  (`ValueError`); `gbgpu.get_first_backend` does not work as written (use
  `gpubackendtools.get_first_backend([...prefixed names...])`). The global fit
  loads `gbgpu_cuda13x` on the compute ranks and `gbgpu_cpu` for the
  warm-start referee and the monitor.
- **`force_backend=None` trap**: `GBGPUBase.supported_backends()` (so `GBGPU`,
  `GBGPUThirdBody`) and `GBSignalHetComputations.supported_backends()` return
  BARE flavors; gpubackendtools resolves a bare flavor to the first registered
  backend of that flavor, i.e. `gbt_*`, which lacks `sharedmem` /
  `GBComputationGroupWrap`, so construction fails. Always pass a flavor.
  `GBSignalHetComputations.supported_backends()` is
  `["cpu", "cuda11x", "cuda12x", "cuda13x"]`; its full data-loading
  constructor is CPU-only (`"cpu"`; `"gbgpu_cpu"` passes the guard but fails in
  the LAT frontends), and CUDA goes through `for_band_engine`.
- **JAX**: kernel functions exist (`gbgpu.jax.wdm.*`,
  `gbgpu.jax.tdi_on_the_fly.run_wave_tdi`) and LAT's `WDMComputationsBase`
  (base of `GBWDMComputations`) lists a `gbgpu_jax` candidate and has a JAX
  branch, but **no `GBGPUJaxBackend` class is registered anywhere in
  `gbgpu/__init__.py` or `gbgpu/cutils/__init__.py`**, so `force_backend="jax"`
  does not resolve for a GB comp. `GBGPUParallelModule.GPU_RECOMMENDED_WITH_JAX`
  has no callers (TODO-tagged).

## 6. Cross-repo dependencies

**Imports (GBGPU → others):**
- `gpubackendtools` — `ParallelModuleBase`, `Globals`, `Backend`,
  `Cpu/Cuda{11,12,13}xBackend`, exceptions; GBT's `Interpolate.cu`/LAPACKE
  cubic-spline solver and `cuda_complex.hpp`/`gbt_global.h` (via
  `${GBT_CUTILS}` at CMake configure time — GBGPU's own copy of
  `cuda_complex.hpp` was deleted in the Phase 3.dedup pass).
- `lisatools` (LAT) — `Orbits`/`EqualArmlengthOrbits`/`L1Orbits`
  (`lisatools.detector`), `TDIConfig` (`lisatools.response.tdiconfig`),
  `WDMSettings`/`FDSettings`/`TDSettings`/`TDSignal`
  (`lisatools.domains`), `WDMComputationsBase` (`lisatools.chunked_het`),
  `sparse_time_grid`/`bin_fold_real` (`lisatools.signal_het`),
  `XYZ2SensitivityMatrix` (`lisatools.sensitivity`),
  `AnalysisContainer`/`AnalysisContainerArray`
  (`lisatools.analysiscontainer`), the JAX response/orbits/WDM mirrors
  (`lisatools.jax.*`), and at the C++ level: LAT's `Detector.{hpp,cu}`,
  `global.hpp`, `LISAResponse.hh`, `Interpolate.hh`, `fd_domain.hh`,
  `wdm_settings.hh`, `wdm_domain.hh`, `lat_tdi_on_the_fly.hh/.cu`,
  `lat_chunked_het_kernels.hh`, `binding_flr.hpp`,
  `lisatools_header_abi.hpp` — all resolved via `import lisatools` shell-out
  in `cutils/CMakeLists.txt` at configure time, copied into
  `src/gbgpu/cutils/` (gitignored) and compiled into GBGPU's own `.so`.

**Depends on GBGPU (others → GBGPU):**
- `LISAanalysistools` — in `src/lisatools/`: `globalfit/recipe.py` (`GBGPU`
  containers, `gbgpu.get_backend`), `globalfit/stock/erebor/{gb, vgb,
  transforms, injections, variants/gb_no_fg, variants/vgb}.py` (the comps,
  `build_sighet_engine`, `utils.utility` helpers),
  `globalfit/moves/{gbbands, gbspecialstretch, gb_likelihood}.py`,
  `globalfit/warmstart/{match_referee, opt_snr, para_search}.py`,
  `globalfit/monitor/{_generator, _short, build_truth}.py`,
  `globalfit/{gathergalaxy, postprocessing, buildcatalog, generatefuncs}.py`,
  `sampling/{fstat_gridfit, fstat_proposal, gb_observable_basis}.py` and
  `sources/gb/waveform.py`; plus ~100 dev/validation scripts under
  `LISAanalysistools/scripts/`. `lisatools.globalfit.moves.gb_likelihood` is
  now a thin deprecation shim re-exporting `gbgpu.gb_likelihood`;
  `globalfit/buildcatalog.py` imports the nonexistent `gbgpu.utils.constants`.
- `LATW` tutorials (dev branch): `02` (`import gbgpu`), `04` and `further/X1`
  (`gbgpu.gbgpu.GBGPU`), `08` (`gbgpu.jax.tdi_on_the_fly.gb_run_wave_tdi`), `09`
  (`gbgpu.gbcomps.GBWDMComputations`, `gbgpu.gbsignalhetcomputations.GBSignalHetComputations`).
- `BBHx` — no runtime import; its SOBBH kernels and `bbhx.sobbhcomps` mirror
  GBGPU's (comments only).

## 7. Non-obvious invariants / gotchas

- **SOBBH JAX pieces moved to BBHx** (`bbhx.jax.sources.sobbh`;
  `SOBBHComputationGroupWrapJAX` no longer exists in GBGPU). Two leftovers
  still import the removed `gbgpu.jax.sources.sobbh` and fail:
  `jax/tdi_on_the_fly.sobbh_run_wave_tdi` and
  `jax/wrappers.SOBBHTDIonTheFlyWrapJAX` (both TODO-tagged).
- **`GBGPUParallelModule` has one implementation**, `gbgpu/parallelbase.py`;
  `gbgpu/utils/parallelbase.py` re-exports it. `gbgpu.py` imports the `utils.`
  path; `gbcomps.py` / `gbsignalhetcomputations.py` import the top-level one
  (aliased as `FastLISAResponseParallelModule`). `GBWDMComputations` does not
  use it (LAT `WDMComputationsBase` with `_BACKEND_PREFIX = "gbgpu"`).
- **`GBGPU/CLAUDE.md` and `README.md`** were brought up to date on 2026-10-22
  (the reorg "pending arrival" items have landed; README describes both stacks).
- **`gbgpu_jax` backend is referenced but not registered** — see §5.
- **`gbgpu.__all__`** lists `get_logger`, `get_config`, `get_config_setter`,
  `get_file_manager`, which are not defined, so `from gbgpu import *` raises
  `AttributeError` (FEW-template leftover).
- **`GBGPU(...)` requires `orbits`** (`None` raises `ValueError` despite the
  `Optional` annotation); see §5 for the `force_backend=None` trap.
- **Sig-het per-device replicas drop the lookup knobs** (LAT
  `stock/erebor/source_runtime._build_gb_comp_replica` rebuilds from `_g`, which
  does not record `lookup_table` / `lookup_n_nodes` / `lookup_k_coarse` /
  `anchor_engine`): a replica would raise under the launchers' `SIGHET_*=lookup`
  env. Latent; only reached when one rank's band buffer spans several GPUs,
  which the production layouts do not do.
- LISA Analysis Tools–wide rules that apply here and are enforced/checked in this repo's
  binding code: single-registrant rule (`static_assert(!LISATOOLS_IS_WRAPPER_OWNER)`
  in `binding_gbgpu.cxx` — GBGPU must never register `OrbitsWrap`/
  `TDIConfigWrap`/`LISAResponseWrap`/WDM*/FD*/Spline* itself), host→device
  wrapper-struct upload pattern (LAT `docs/conventions.md`, "Host→device upload of
  class-wrapper objects"), CPU/GPU
  class-name aliasing (§5), and deepcopy/pickle safety (`self.xp` must stay a
  property derived from `self.backend`, never a cached module reference —
  check `gbgpu.py`/`gbcomps.py`/`gbsignalhetcomputations.py` `xp` properties
  if refactoring).
- `flip_ref_phase` on `GBGPUBase.__init__` silently negates `params[:, 4]`
  (phi0) IN PLACE on the caller's array inside `get_ll` / `get_fstat_ll` (where
  column 4 is `beta` of the 5-column layout) / `generate_global_template` /
  `swap_likelihood_difference` / `information_matrix`, to match the JaxGB / LDC
  phase convention (FastGB templates carry `exp(+i phi0)`) — easy to
  double-apply, and repeated calls flip it back and forth. `run_wave` negates a
  copy, so the Python path of `information_matrix` (flip in place, then
  `run_wave`) negates twice and builds unflipped templates. No global-fit caller
  sets it.
- **Information-matrix step sizes.** The chunked delegate's default step table
  (LAT `WDMComputationsBase._info_matrix_param_eps`: f0 2e-14 Hz, fdot 1e-21 Hz/s,
  fddot 1e-28, angles 1e-6 rad) moves the template phase by only ~1e-8..1e-6 rad
  at 30 d - 6 months; the differences are then noise-dominated (fdot-fdot 19x /
  negative on the chunked route at 30 d; negative f0 / fdot, and at 6 months
  angle, diagonals on the `SIGHET_INFOMAT` route). It is still the default of
  those two routes (production numbers; not changed). The lookup Gram uses its
  own Tobs-scaled table (`GBLookupComputations.info_matrix_param_eps`), which is
  also a good explicit `param_eps` for the other routes.
- `GBSignalHetComputations.for_band_engine()` constructs itself via
  `cls.__new__(cls)` (bypassing `__init__`) and manually re-implements the
  constructor body against a delegate `GBWDMComputations`' `wdm_settings` —
  if `__init__`'s field set changes, `for_band_engine` must be updated by
  hand (no shared helper).

## 8. Where to look for X

| Change / understand X | Start in |
|---|---|
| Classic dense FastGB waveform generation | `src/gbgpu/gbgpu.py` (`GBGPUBase._construct_slow_part`, `run_wave`) |
| Classic likelihood / global-template fill / swap proposals | `src/gbgpu/gbgpu.py` (`get_ll`, `fill_global_template`, `swap_likelihood_difference`) |
| Eccentric third-body physics | `src/gbgpu/thirdbody.py` |
| GB UCB closed-form amp/phase/freq (new stack) | `src/gbgpu/cutils/gb_tdi_on_the_fly.hh` (`GBTDIonTheFly::ucb_*`) |
| Chunked-heterodyne WDM likelihood (fill/get_ll/swap/fstat) kernels | `src/gbgpu/cutils/gb_tdi_on_the_fly.hh` (`GBComputationGroup::gb_wdm_het_*_wrap`) + `.cu` bodies |
| FD chunked-het likelihood kernels | same file, `gb_fd_*_wrap` methods; Python side `src/gbgpu/gbcomps.py::GBFDComputations` |
| Signal-heterodyne (v2 ... v5; production v5 via `for_band_engine`) | `src/gbgpu/gbsignalhetcomputations.py` + `gb_tdi_on_the_fly.hh` `gb_signal_het_*_wrap` (carrier reference producer `gb_signal_het_make_reference_carrier`; v5 layout bits `GB_SIGHET_V5_COLLAPSED` / `GB_SIGHET_V5_SYM`; F-stat bits `GB_SIGHET_FSTAT_CARRIER` / `GB_SIGHET_FSTAT_SYM`) |
| Direct-to-WDM lookup scorer | `src/gbgpu/gblookupcomputations.py` + `gb_tdi_on_the_fly.hh` (`GBLookupArgs`, `gb_lookup_get_ll_wrap`; table reads from LAT `wdm_lookup_kernels.hh`) |
| GB in-model proposal information matrix (lookup Gram / sig-het second differences / chunked swap Gram) | `GBSignalHetComputations.information_matrix` (route choice, `SIGHET_INFOMAT_ENGINE` / `SIGHET_INFOMAT`) → `GBLookupComputations.information_matrix`; caller: LAT `gbspecialstretch._compute_proposal_cholesky` / `_CholCache` (`GB_CHOL_CACHE`) via `gbbands._RoutedBandEngine.route_information_matrix` |
| Global-fit GB special-move likelihood engine | `src/gbgpu/gb_likelihood.py` (`make_band_likelihood_engine`, `FDBandLikelihoodEngine`, `WDMBandLikelihoodEngine`) |
| nanobind module registration / new native method exposure | `src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}` |
| CMake build wiring (LAT/GBT source copy-in, LAPACKE, mathdx/cuFFTDx) | `src/gbgpu/cutils/CMakeLists.txt` |
| Backend composition (which native symbols a backend carries) | `src/gbgpu/cutils/__init__.py` |
| Backend registration / `gbgpu.get_backend` | `src/gbgpu/__init__.py` |
| JAX mirror of chunked-het kernels | `src/gbgpu/jax/wdm/{kernels,heterodyne_kernels}.py` |
| JAX mirror of signal-het kernels | `src/gbgpu/jax/wdm/signal_het_kernels.py` |
| JAX duck-typed native-class wrappers | `src/gbgpu/jax/wdm/computation_group.py`, `src/gbgpu/jax/wrappers.py` |
| Param/physics helper functions (get_N, get_fdot, chirp mass, AET) | `src/gbgpu/utils/utility.py` |
| Backend-prefix / `force_backend` resolution | `src/gbgpu/parallelbase.py` (`src/gbgpu/utils/parallelbase.py` re-exports it, see §7) |
| Citations / bibliography metadata | `src/gbgpu/utils/citation.py` (`citations.py` is a dead FEW-template leftover) |
| Sphinx docs (user guide + API pages) | `docs/source/index.rst`, `docs/source/user/*.rst` |
