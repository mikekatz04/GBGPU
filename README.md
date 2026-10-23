# GBGPU

[![Doc badge](https://img.shields.io/badge/Docs-master-brightgreen)](https://lisa-analysis-tools.github.io/GBGPU/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138723.svg)](https://doi.org/10.5281/zenodo.17138723)

`GBGPU` computes gravitational waveforms and LISA likelihoods of Galactic binaries on CPUs and NVIDIA GPUs. It is the Galactic-binary package of LISA Analysis Tools, built on [LISAanalysistools](https://github.com/lisa-analysis-tools/lisa-analysis-tools) (`lisatools`: orbits, TDI, domains, the chunked-heterodyne base) and `gpubackendtools` (backend machinery). It holds two stacks:

- `gbgpu.gbgpu.GBGPU`: a GPU-accelerated version of the `FastGB` waveform, which has been developed by Neil Cornish, Tyson Littenberg, Travis Robson, and Stas Babak. It computes the waveforms with a fast/slow-type decomposition and is very closely related to the implementation of `FastGB` in the LISA Data Challenges' Python code package. For more details on the original construction of `FastGB` see [arXiv:0704.1808](https://arxiv.org/abs/0704.1808). The waveform itself (`run_wave`) is array code on `numpy` or `cupy`; batched frequency-domain likelihoods, F-statistics, swap likelihoods and global templates run in C++/CUDA kernels. Its response uses an equal-arm approximation. `gbgpu.thirdbody.GBGPUThirdBody` adds a third body on a long outer orbit.
- The WDM / FD computation objects: `gbgpu.gbcomps.GBWDMComputations` and `GBFDComputations` (chunked heterodyne), `gbgpu.gbsignalhetcomputations.GBSignalHetComputations` (signal heterodyne), `gbgpu.gblookupcomputations.GBLookupComputations` (direct-to-WDM lookup) and the band likelihood engines of `gbgpu.gb_likelihood`. They run C++/CUDA kernels on the `lisatools` orbits (unequal arms with `L1Orbits` / `ESAOrbits`) and WDM / FD grids, and they are the GB likelihood of the LISA Analysis Tools global fit.

`gbgpu.jax` holds JAX versions of the kernels (validation and autograd; there is no JAX backend).

See the [documentation](https://lisa-analysis-tools.github.io/GBGPU/) for more details. This code was designed for [arXiv:2205.03461](https://arxiv.org/abs/2205.03461). If you use any part of this code, please cite [arXiv:2205.03461](https://arxiv.org/abs/2205.03461), its [Zenodo page](https://zenodo.org/records/16999246), [arXiv:0704.1808](https://arxiv.org/abs/0704.1808), and [arXiv:1806.00500](https://arxiv.org/abs/1806.00500).

To install the latest release of `gbgpu` using `pip`, simply run:

```sh
# For CPU-only version
pip install gbgpu

# For GPU-enabled versions with CUDA 11.Y.Z
pip install gbgpu-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install gbgpu-cuda12x

# For GPU-enabled versions with CUDA 13.Y.Z
pip install gbgpu-cuda13x
```

The pip releases follow the `master` branch. The `dev` branch (and the documentation built from it) describes the development API, which is installed from source together with the rest of the development stack (see `LISAanalysistools/install.sh`).

To know your CUDA version, run the tool `nvidia-smi` in a terminal and check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

Now, in a python file or notebook:

```py3
import gbgpu
```

Importing `gbgpu` registers the backends `gbgpu_cpu`, `gbgpu_cuda11x`, `gbgpu_cuda12x` and `gbgpu_cuda13x`. You may check which ones are available:

```py3
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda13x"]:
...     status = "available" if gbgpu.has_backend(backend) else "unavailable"
...     print(f" - Backend '{backend}': {status}")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda13x': unavailable
```

`gbgpu.get_backend("cuda")` returns the first available of `cuda13x`, `cuda12x` and `cuda11x`, and `"gpu"` is (for now) an alias for `"cuda"`. `gbgpu.has_backend` does not resolve these two aliases (it raises `ValueError`), so test the flavors one by one as above.

If you expected a backend to be available but it is not, ask for it with `gbgpu.get_backend` to obtain an error message which can guide you to fix this issue:

```py3
>>> import gbgpu
>>> gbgpu.get_backend("cuda12x")
...
gpubackendtools.gpubackendtools.BackendAccessException: Backend 'gbgpu_cuda12x' is unavailable. See previous error messages.
```

The exceptions chained above the last one give the reason (plugin not installed, no CUDA device or driver, missing `cupy`, ...).

Once GBGPU is working, generate a first batch of waveforms. Every GBGPU class takes its backend from `force_backend`; pass it explicitly. `GBGPU` also requires an orbits object from `lisatools`:

```py3
import numpy as np
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from gbgpu.gbgpu import GBGPU

gb = GBGPU(orbits=EqualArmlengthOrbits(force_backend="cpu"), force_backend="cpu")
# columns: amp, f0 [Hz], fdot [Hz/s], fddot [Hz/s^2], phi0, iota, psi, lam, beta
params = np.tile([2e-23, 2e-3, 7.5e-18, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], (10, 1))
gb.run_wave(*params.T, T=1.0 * YRSID_SI, dt=10.0)
gb.XYZf.shape  # (10, 3, gb.N)
```

Then check out the [examples notebooks](https://github.com/lisa-analysis-tools/GBGPU/tree/master/examples/). Tutorials for the wider LISA Analysis Tools stack live in [LATW](https://github.com/lisa-analysis-tools/LATW).

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- Python >= 3.12 (`requires-python` of `pyproject.toml`)
- `lisaanalysistools` and `gpubackendtools`: the build copies C++ sources and CMake helpers from both installed packages at configure time. With build isolation, `pip` installs the build requirements itself (`scikit-build-core`, `nanobind`, `lisaanalysistools`, `numpy`); the development stack builds without isolation against its editable installs.

If you want to enable GPU support in GBGPU, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`). The CUDA code is built for compute capabilities 8.0 and 9.0 by default; pass `--config-settings=cmake.define.GBGPU_CUDA_ARCH=70` (for example) for other GPUs.

The legacy shared-memory kernels of `GBGPU` (`SharedMemoryGBGPU.cu`, which need cuFFTDx on GPU and download it at configure time) are controlled by the CMake option `GBGPU_WITH_SHAREDMEM` (default `ON`). With `OFF` the build is faster and needs no network access; `GBGPU.get_ll`, `get_fstat_ll`, `swap_likelihood_difference`, `generate_global_template` and the `use_c_implementation=True` paths then raise `RuntimeError`, while `run_wave` and all WDM / FD computation objects work:

```sh
pip install -e . --config-settings=cmake.define.GBGPU_WITH_SHAREDMEM=OFF
```

### Installation instructions using conda

We recommend to install GBGPU using conda in order to have the compilers all within an environment.
First clone the repo

```
git clone https://github.com/lisa-analysis-tools/GBGPU.git
cd GBGPU
```

Now create an environment (these instructions work for all platforms but some
adjustements can be needed, refer to the
[detailed installation documentation](https://lisa-analysis-tools.github.io/GBGPU/) for more information):

```
conda create -n gbgpu_env -y -c conda-forge --override-channels \
    python=3.12 cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate gbgpu_env
```

Then we can install locally for development:
```
pip install -e '.[doc,testing]'
```

### Installation instructions using conda on GPUs and linux
Below is a quick set of instructions to install the GBGPU package on GPUs and linux.

```sh
conda create -n gbgpu_env -c conda-forge gbgpu-cuda12x python=3.12
conda activate gbgpu_env
```

Test the installation device by running python
```python
import gbgpu
gbgpu.get_backend("cuda12x")
```

### Running the installation

To start the from-source installation, ensure the pre-requisite are met, clone
the repository, and then simply run a `pip install` command:

```sh
# Clone the repository
git clone https://github.com/lisa-analysis-tools/GBGPU.git
cd GBGPU

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://lisa-analysis-tools.github.io/GBGPU/). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/lisa-analysis-tools/GBGPU/issues)
or contact the developers through other means.



### Running the Tests

The tests live in the `tests/` directory of the repository (they are not installed with the package). Install the `[testing]` extra, then run the `unittest` module from the repository root:

```sh
$ git clone https://github.com/lisa-analysis-tools/GBGPU.git
$ cd GBGPU
$ pip install '.[testing]'
$ python -m unittest discover   # or "python -m unittest tests/test*", as the CI does
```

The GPU test classes skip without CUDA.

## Contributing

Please open an issue or a pull request on the [GitHub repository](https://github.com/lisa-analysis-tools/GBGPU).

If you want to develop GBGPU and produce documentation, install `gbgpu` from source with the `[doc]` label and in `editable` mode:

```
$ git clone https://github.com/lisa-analysis-tools/GBGPU.git
$ cd GBGPU
$ pip install -e '.[doc,testing]'
```

This will install necessary packages for building the documentation (`sphinx`, `myst-parser`, `nbsphinx`, `pypandoc`, `sphinx-rtd-theme`, `sphinx-tippy`, ...) and to run the tests. `pypandoc` also needs the `pandoc` program.

The documentation source files are in `docs/source`. To compile the documentation locally, change to the `docs` directory and run `make html` (the CI runs `sphinx-build -M html docs/source docs/build --define nbsphinx_execute=never` from the repository root).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/lisa-analysis-tools/GBGPU/tags).

## Contributors

A (non-exhaustive) list of contributors to the GBGPU code: Michael Katz (author), Travis Robson, Neil Cornish, Tyson Littenberg, Stas Babak, Mathieu Dubois, Maxime Pigou.

## License

This project is licensed under the Apache License - see the [LICENSE](https://github.com/lisa-analysis-tools/GBGPU/blob/master/LICENSE) file for details.

## Citation

Please make sure to cite GBGPU papers and the GBGPU software on [Zenodo](https://zenodo.org/records/17138723).
There are other papers that require citation based on the classes used: the `citation` property of `GBGPU` and `GBGPUThirdBody` returns the BibTeX entries of the GBGPU Zenodo record, FastGB ([arXiv:0704.1808](https://arxiv.org/abs/0704.1808)) and the third-body waveform ([arXiv:1806.00500](https://arxiv.org/abs/1806.00500)).
