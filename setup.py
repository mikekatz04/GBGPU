# from future.utils import iteritems
import os
import shutil
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    elif "CUDA_HOME" in os.environ:
        home = os.environ["CUDA_HOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be " "located in %s" % (k, v)
            )

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    CUDA = locate_cuda()
    run_cuda_install = True
except OSError:
    run_cuda_install = False

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


if run_cuda_install:
    ext_gpu_dict = dict(
        sources=[
            "src/gbgpu/cutils/src/gbgpu_utils.cu",
            "src/gbgpu/cutils/src/GBGPU.pyx",
        ],
        library_dirs=[CUDA["lib64"]],
        libraries=["cudart", "cublas", "cufft"],
        language="c++",
        runtime_library_dirs=[CUDA["lib64"]],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": [],  # '-g'],
            "nvcc": [
                "-arch=sm_80",
                "-gencode=arch=compute_50,code=sm_50",
                "-gencode=arch=compute_52,code=sm_52",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_80,code=sm_80",
                "--default-stream=per-thread",
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "'-fPIC'",
            ],  # ,"-G", "-g"] # for debugging
        },
        include_dirs=[
            numpy_include,
            CUDA["include"],
            "src/gbgpucutils/include",
        ],
    )
    ext_gpu = Extension("gbgpu.cutils.gbgpu_utils", **ext_gpu_dict)

ext_cpu_dict = dict(
    sources=[
        "src/gbgpu/cutils/src/gbgpu_utils.cpp",
        "src/gbgpu/cutils/src/GBGPU_cpu.pyx",
    ],
    library_dirs=[],
    libraries=[],
    language="c++",
    extra_compile_args={
        "gcc": [],  # "-std=c++11"
    },  # '-g'],
    include_dirs=[numpy_include, "src/gbgpu/cutils/include"],
)
ext_cpu = Extension("gbgpu.cutils.gbgpu_utils_cpu", **ext_cpu_dict)

if run_cuda_install:
    extensions = [ext_gpu, ext_cpu]

else:
    extensions = [ext_cpu]

setup(
    name="gbgpu",
    # Random metadata. there's more you can supply
    author="Michael Katz",
    package_dir={"gbgpu": "src/gbgpu/"},
    packages=[
        "gbgpu",
        "gbgpu.utils",
        "gbgpu.cutils",
        "gbgpu.cutils.src",
        "gbgpu.cutils.include",
    ],
    py_modules=["gbgpu.gbgpu", "gbgpu.thirdbody"],
    ext_modules=extensions,
    # Inject our custom trigger
    cmdclass={"build_ext": custom_build_ext},
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
    package_data={
        "gbgpu.cutils.src": ["gbgpu_utils.cu", "gbgpu_utils.cpp", "GBGPU.pyx"],
        "gbgpu.cutils.include": [
            "gbgpu_utils.hh",
            "cuda_complex.hpp",
            "LISA.h",
            "global.h",
        ],
    },
    python_requires=">=3.6",
)
