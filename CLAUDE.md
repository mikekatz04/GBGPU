# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

**LISA Analysis Tools–wide conventions:** [`../LISAanalysistools/docs/conventions.md`](../LISAanalysistools/docs/conventions.md) (canonical). **This repo's map:** [`docs/codebase-map.md`](docs/codebase-map.md).

## Role and current state (dev branch, checked 2026-10-22)

GBGPU is the **GB-physics owner** in the LISA Analysis Tools layered architecture.
LISAanalysistools (LAT) owns generic LISA infrastructure; GBGPU owns
GB-specific physics on top of it.

The reorg work items this file used to list as "pending arrival" (the C++
`GBTDIonTheFly` / `GBComputationGroup` carve-out, `gbcomps.py`, the
signal-heterodyne port) have all landed. Current layout (details in
`docs/codebase-map.md`):

- **C++ / CUDA** (`src/gbgpu/cutils/`): `gb_tdi_on_the_fly.{hh,cu}` holds
  `GBTDIonTheFly` (UCB amplitude / phase on top of LAT's `LISATDIonTheFly`)
  and `GBComputationGroup` with every modern kernel family: `gb_fd_*`,
  chunked-heterodyne `gb_wdm_het_*`, signal-heterodyne `gb_signal_het_*`
  (v2 ... v5 scorers, carrier reference, sig-het F-stat) and the direct-to-WDM
  lookup `gb_lookup_*`. The legacy FastGB kernels are `gbgpu_utils.cu` and
  `SharedMemoryGBGPU.cu`. There are no separate `GBSignalHet.*` /
  `GBAbsoluteFD.*` / `binding_gbsignalhet.cxx` files.
- **Bindings**: `binding_gbgpu.{hpp,cxx}` is a **nanobind** module
  (`NB_MODULE(cgbgpu, m)`; the "pybind11" wording in its comments predates the
  Phase 3M pybind11 -> nanobind migration). It exposes
  `GBGPUComputationWrap{CPU,GPU}` (legacy kernels), `GBTDIonTheFlyWrap{CPU,GPU}`
  and `GBComputationGroupWrap{CPU,GPU}`, shipped as
  `gbgpu_backend_{cpu,cuda11x,cuda12x,cuda13x}.cgbgpu`. `pyproject.toml`
  pins `nanobind==2.12.0` in `build-system.requires` (sprint-wide ABI pin).
  Cython is fully retired (no `.pyx`; the old rules are commented out in
  `cutils/CMakeLists.txt`).
- **Python**: `gbgpu.py` (`GBGPUBase` / `GBGPU`, FastGB), `thirdbody.py`,
  `gbcomps.py` (`GBWDMComputations`, a thin subclass of LAT
  `chunked_het.WDMComputationsBase`; `GBFDComputations`),
  `gbsignalhetcomputations.py` (`GBSignalHetComputations`; production path
  `for_band_engine`, CPU and CUDA), `gblookupcomputations.py`
  (`GBLookupComputations`, `GBLookupWDMComputations`), `gb_likelihood.py`
  (band engines), `cutils/__init__.py` (backends), `parallelbase.py`
  (`GBGPUParallelModule`; `utils/parallelbase.py` and `wdm_het.py` are
  re-export shims).
- **JAX** (`gbgpu/jax/`): `sources.ucb.JaxUCBSource`, `tdi_on_the_fly`,
  `wdm.{kernels, heterodyne_kernels, fast_inner_heterodyne, signal_het_kernels}`.
  These import generic infrastructure absolutely from
  `lisatools.jax.{response,wdm,orbits}` -- there is NO `gbgpu.jax.base` etc.
  No `gbgpu_jax` backend is registered; `jax/wrappers.py` and
  `jax/wdm/computation_group.py` (`GBComputationGroupWrapJAX`) are unwired
  adapters, and the SOBBH JAX pieces moved to BBHx
  (`bbhx.jax.sources.sobbh`; `gbgpu.jax.tdi_on_the_fly.sobbh_run_wave_tdi`
  still imports the removed `gbgpu.jax.sources.sobbh`).

**Production GB likelihood (9mo / 12mo global fits).** LAT builds a
`GBWDMComputations` and wraps it with `GBSignalHetComputations.for_band_engine`
(LAT `stock/erebor/gb.build_sighet_engine`, with a GB WDM lookup table);
inside it a `GBLookupComputations` builds the carrier references, the anchor
offsets and the information matrices; `gb_likelihood.WDMBandLikelihoodEngine`
drives it. The `GBGPU` object the recipe builds is a container on the WDM path;
`GBGPU.run_wave` runs only in the warm-start referee and on the monitor (CPU).
The user-facing write-up is `docs/source/user/likelihood_path.rst`.

**API traps to remember.**
- Pass `force_backend` explicitly. `None` is broken for `GBGPU`,
  `GBGPUThirdBody` and the `GBSignalHetComputations` constructor: their
  `supported_backends` return bare flavors, which resolve to a `gbt_*`
  backend without the GB wraps.
- `GBGPU(...)` requires `orbits` (`None` raises `ValueError`).
- The `GBSignalHetComputations` constructor is CPU only (`"cpu"`); CUDA goes
  through `for_band_engine`, which inherits the delegate's backend.
- `GBGPU.get_chi_sqared`, `hessian_wdm`, `utils/config.py` and
  `utils/citations.py` are broken / dead (TODO-tagged, excluded from the API
  docs); `gbgpu.__all__` lists names that do not exist.
- `examples/GBGPU_tutorial.ipynb` (copied into `docs/source/` by `conf.py`)
  still builds `GBGPU` without `orbits` and calls `information_matrix` with an
  outdated signature; `examples/fast_likelihood_tutorial.ipynb` still calls
  `GBSignalHetComputations` "on the roadmap". `tests/test_gbgpu.py` /
  `tests/test_thirdbody.py` use the retired `EqualArmlengthOrbits(use_gpu=...)`
  kwarg. Notebooks and tests are stale on dev.

**Build infrastructure:**
- GBGPU's CMakeLists consumes LAT's `Detector.cu`, `Detector.hpp`, `global.hpp`
  via a `python -c "import lisatools"` shell-out (`LISATOOLS_CUTILS` variable,
  then `file(COPY ${LISATOOLS_CUTILS}/...)`) and copy-compiles LAT's
  `lat_tdi_on_the_fly.cu`. This replaced an older
  `file(DOWNLOAD https://raw.githubusercontent.com/.../refs/heads/main/...)`
  pattern that pulled stale snapshots ignoring the version-pinned
  lisaanalysistools build dep. The copied files live under
  `src/gbgpu/cutils/` and are gitignored — source of truth is LAT.
- Same flow for `${GBT_CUTILS}` (gpubackendtools cutils dir): GBT's
  `Interpolate.cu` (LAPACKE cubic-spline solver) is compiled into `cgbgpu`.
- `cutils/lisaconstants_values.h` is GENERATED by LAT's
  `python -m lisatools.utils.lisaconstants_header` (committed so the copied LAT
  headers resolve it); `tests/test_lisaconstants_values.py` checks it against
  the installed `lisaconstants` (pinned `==2.0.2`). Never hand-edit it.
- **`GBGPU_CUDA_ARCH` defaults to `80;90` (2026-07-28)** — was `"native"`, which
  needs CMake to see a GPU at *configure* time. On a GPU-less cluster
  build/login node that detection falls through to nvcc's own default (sm_52),
  which is below the **sm_60 floor for `atomicAdd(double*, double)`** — so
  `gb_tdi_on_the_fly.cu` (`gb_fd_fill_global`) and LAT's
  `lat_chunked_het_kernels.hh` (`wdm_het_fill_global_impl`) fail with *"no
  instance of overloaded function atomicAdd matches (double*, double)"*. LAT
  never hit this because its own CMakeLists hardcodes `70;75;80;86;90`, and
  `install.sh` passes no arch flag — so each repo used its own default.
  A configure-time `FATAL_ERROR` now rejects any arch < 60 up front.

  `80;90` is deliberately narrower than LAT's list: CUDA guarantees binary
  compatibility forward across *minor* revisions within a major generation, so
  the sm_80 cubin also covers sm_86 (A40/A6000/RTX30) and sm_89 (Ada/L40).
  **Not** covered: sm_70 (V100), sm_75 (Turing) — those need
  `-DGBGPU_CUDA_ARCH=70` or they fail at *runtime* with "no kernel image is
  available for execution on the device". No PTX entry is emitted, so future
  majors need an explicit value too. Note GPUBackendTools still defaults to
  `native` and carries the same latent exposure.
- **`GBGPU_WITH_SHAREDMEM` (CMake default `ON`, 2026-07-28)** — compiles the legacy
  `SharedMemoryGBGPU.cu` kernel family into `cgbgpu`. It is 4k lines of
  heavily-templated CUDA and the repo's **only** `#include <cufftdx.hpp>`, so
  `-DGBGPU_WITH_SHAREDMEM=OFF` also removes the configure-time
  `nvidia-mathdx` tarball download + extract + `find_package(mathdx)` — a
  GPU configure with it OFF needs **no network access** and compiles
  substantially faster. `LISAanalysistools/install.sh` builds the dev stack
  with `OFF` by default (its `GBGPU_WITH_SHAREDMEM` variable).

  ```sh
  pip install -e . --config-settings=cmake.define.GBGPU_WITH_SHAREDMEM=OFF
  ```

  The seven `SharedMemory*_wrap` methods stay **bound** (identical Python
  surface) but raise `RuntimeError` with a rebuild instruction — the guard is
  `GBGPU_NO_SHAREDMEM` / `GBGPU_SHAREDMEM_UNAVAILABLE` in `binding_gbgpu.hpp`.
  Unaffected: `GBTDIonTheFly`, the chunked-heterodyne kernels, the
  signal-heterodyne (`gb_signal_het_*`) and lookup families, the gbgpu_utils
  `fill_global` (`GBGPU.fill_global_template`) and the Python path of
  `GBGPU.run_wave` / `information_matrix` — i.e. an OFF build fully supports
  the WDM global fits (sig-het in-model, F-stat grid births through the
  sig-het / chunked F-stat) and their parity gates.

  **Still needs ON:** the legacy FD methods `GBGPU.get_ll`, `get_fstat_ll`,
  `swap_likelihood_difference`, `generate_global_template` and the
  `use_c_implementation=True` paths. LAT callers: FD-basis code only
  (`gbspecialstretch.para_log_like`, which refuses non-FD bases since the
  2026-08-12 ruling; the FD branch of `recipe.py`; `gathergalaxy`;
  `sources/gb/waveform.GBXYZTDIWaveform`). Synthetic GB injections
  (`stock/erebor/injections.py`) moved to `GBFDComputations` in 2026-08 and no
  longer need it.

**Single-registrant rule (LISA Analysis Tools–wide)**: GBGPU's binding TUs MUST NOT
register `OrbitsWrap`, `LISAResponseWrap`, `TDIConfigWrap`, or
`CubicSplineWrap`. The first three are owned by LAT's `pycppdetector`;
`CubicSplineWrap` is owned by GBT's `interp` module (2026-06-10). GBGPU's
`binding_gbgpu.cxx` enforces it with `#include "lisatools_header_abi.hpp"`
(via `binding_gbgpu.hpp`) + `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)`,
and consumes LAT's canonical `ReturnPointerBase` through `binding_flr.hpp`;
the backend loaders in `cutils/__init__.py` take the shared wraps from
`lisatools_backend_<flavor>.pycppdetector` and `gbt_backend_<flavor>.interp`.
`LISAanalysistools/tools/check_single_registrant.sh` is the CI grep
complement.

## Backend implementation hierarchy (LISA Analysis Tools–wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.
