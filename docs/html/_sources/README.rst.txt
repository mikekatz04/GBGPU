gbgpu: GPU/CPU Galactic Binary Waveforms
========================================

``GBGPU`` is a GPU-accelerated version of the ``FastGB`` waveform which
has been developed by Neil Cornish, Tyson Littenberg, Travis Robson, and
Stas Babak. It computes gravitational waveforms for Galactic binary
systems observable by LISA using a fast/slow-type decomposition. For
more details on the original construction of ``FastGB`` see
`arXiv:0704.1808 <https://arxiv.org/abs/0704.1808>`__.

The current version of the code is very closely related to the
implementation of ``FastGB`` in the LISA Data Challenges’ Python code
package. The waveform code is entirely Python-based. It is about 1/2 the
speed of the full C version, but much simpler in Python for right now.
There are also many additional functions including fast likelihood
computations for individual Galactic binaries, as well as fast C-based
methods to combine waveforms into global fitting templates.

The code is CPU/GPU agnostic. CUDA and NVIDIA GPUs are required to run
these codes for GPUs.

See the
`documentation <https://mikekatz04.github.io/GBGPU/html/index.html>`__
for more details. This code was designed for
`arXiv:2205.03461 <https://arxiv.org/abs/2205.03461>`__. If you use any
part of this code, please cite
`arXiv:2205.03461 <https://arxiv.org/abs/2205.03461>`__, its `Zenodo
page <https://zenodo.org/record/6500434#.YmpofxNBzlw>`__,
`arXiv:0704.1808 <https://arxiv.org/abs/0704.1808>`__, and
`arXiv:1806.00500 <https://arxiv.org/abs/1806.00500>`__.

Getting Started
---------------

Below is a quick set of instructions to get you started with ``gbgpu``.

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Create a virtual environment. **Note**: There is no available
   ``conda`` compiler for Windows. If you want to install for Windows,
   you will probably need to add libraries and include paths to the
   ``setup.py`` file.

::

   conda create -n gbgpu_env -c conda-forge gcc_linux-64 gxx_linux-64 gsl numpy Cython scipy jupyter ipython h5py matplotlib python=3.8
   conda activate gbgpu_env

::

   If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.

2) Clone the repository.

::

   git clone https://github.com/mikekatz04/GBGPU.git
   cd GBGPU

3) Run install. Make sure CUDA is on your PATH if installing for GPU.

::

   python setup.py install

4) To import gbgpu:

::

   from gbgpu.gbgpu import GBGPU

Prerequisites
~~~~~~~~~~~~~

To install this software for CPU usage, you need `gsl
>2.0 <https://www.gnu.org/software/gsl/>`__, Python >3.4, and NumPy. We
generally recommend installing everything, including gcc and g++
compilers, in the conda environment as is shown in the examples here.
This generally helps avoid compilation and linking issues. If you use
your own chosen compiler, you may need to add information to the
``setup.py`` file.

To install this software for use with NVIDIA GPUs (compute capability
>2.0), you need the `CUDA
toolkit <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__
and `CuPy <https://cupy.chainer.org/>`__. The CUDA toolkit must have
cuda version >8.0. Be sure to properly install CuPy within the correct
CUDA toolkit version. Make sure the nvcc binary is on ``$PATH`` or set
it as the ``CUDAHOME`` environment variable.

Installing
~~~~~~~~~~

0) `Install Anaconda <https://docs.anaconda.com/anaconda/install/>`__ if
   you do not have it.

1) Create a virtual environment. **Note**: There is no available
   ``conda`` compiler for Windows. If you want to install for Windows,
   you will probably need to add libraries and include paths to the
   ``setup.py`` file.

::

   conda create -n gbgpu_env -c conda-forge gcc_linux-64 gxx_linux-64 gsl numpy Cython scipy jupyter ipython h5py matplotlib python=3.8
   conda activate gbgpu_env

::

   If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.

2) If using GPUs, use pip to `install
   cupy <https://docs-cupy.chainer.org/en/stable/install.html>`__. If
   you have cuda version 9.2, for example:

::

   pip install cupy-cuda92

3) Clone the repository.

::

   git clone https://github.com/mikekatz04/GBGPU.git
   cd GBGPU

4) Run install. Make sure CUDA is on your PATH.

::

   python setup.py install

Running the Tests
-----------------

Change to the testing directory:

::

   cd gbgpu/tests

Run in the terminal:

::

   python -m unittest discover

Versioning
----------

We use `SemVer <http://semver.org/>`__ for versioning. For the versions
available, see the `tags on this
repository <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags>`__.

Current Version: 1.0.0

Authors
-------

-  **Michael Katz**
-  Travis Robson
-  Neil Cornish
-  Tyson Littenberg
-  Stas Babak

License
-------

This project is licensed under the GNU License - see the
`LICENSE.md <LICENSE.md>`__ file for details.
