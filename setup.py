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


lib_gsl_dir = "/opt/local/lib"
include_gsl_dir = "/opt/local/include"

if run_cuda_install:

    ext_gpu_dict = dict(
        sources=["src/new_fastGB.cu", "src/GBGPU.pyx"],
        library_dirs=[lib_gsl_dir, CUDA["lib64"]],
        libraries=["cudart", "cublas", "cufft", "gsl", "gslcblas", "gomp"],
        language="c++",
        runtime_library_dirs=[CUDA["lib64"]],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c99"],  # '-g'],
            "nvcc": [
                "-arch=sm_70",
                "-gencode=arch=compute_35,code=sm_35",
                "-gencode=arch=compute_50,code=sm_50",
                "-gencode=arch=compute_52,code=sm_52",
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                "--default-stream=per-thread",
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "-Xcompiler",
                "-fopenmp",
            ],  # ,"-G", "-g"] # for debugging
        },
        include_dirs=[numpy_include, include_gsl_dir, CUDA["include"], "include"],
    )
    ext_gpu = Extension("newfastgb", **ext_gpu_dict)

    ext_gpu_third_dict = dict(
        sources=["src/new_fastGB.cu", "src/GBGPU_third.pyx"],
        library_dirs=[lib_gsl_dir, CUDA["lib64"]],
        libraries=["cudart", "cublas", "cufft", "gsl", "gslcblas", "gomp"],
        language="c++",
        runtime_library_dirs=[CUDA["lib64"]],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c99", "-D__THIRD__"],  # '-g'],
            "nvcc": [
                "-arch=sm_70",
                "-gencode=arch=compute_35,code=sm_35",
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
                "-lineinfo",
                "-Xcompiler",
                "-fopenmp",
                "-D__THIRD__",
            ],  # ,"-G", "-g"] # for debugging
        },
        include_dirs=[numpy_include, include_gsl_dir, CUDA["include"], "include"],
    )

    ext_third_gpu = Extension("newfastgbthird", **ext_gpu_third_dict)

cu_files = ["new_fastGB"]
pyx_files = ["GBGPU", "GBGPU_third"]
for fp in cu_files:
    shutil.copy("src/" + fp + ".cu", "src/" + fp + ".cpp")

for fp in pyx_files:
    shutil.copy("src/" + fp + ".pyx", "src/" + fp + "_cpu.pyx")

ext_cpu_dict = dict(
    sources=["src/new_fastGB.cpp", "src/GBGPU_cpu.pyx"],
    library_dirs=[lib_gsl_dir],
    libraries=["gsl", "gslcblas", "gomp"],
    language="c++",
    extra_compile_args={"gcc": ["-std=c++11", "-fopenmp", "-fPIC"],},  # '-g'],
    include_dirs=[numpy_include, include_gsl_dir, "include"],
)
ext_cpu = Extension("newfastgb_cpu", **ext_cpu_dict)

ext_cpu_third_dict = dict(
    sources=["src/new_fastGB.cpp", "src/GBGPU_cpu.pyx"],
    library_dirs=[lib_gsl_dir],
    libraries=["gsl", "gslcblas", "gomp"],
    language="c++",
    extra_compile_args={"gcc": ["-std=c++11", "-fopenmp", "-fPIC"],},  # '-g'],
    include_dirs=[numpy_include, include_gsl_dir, "include"],
)

ext_cpu_third_dict["sources"] = ["src/new_fastGB.cpp", "src/GBGPU_third_cpu.pyx"]
ext_cpu_third_dict["extra_compile_args"]["gcc"].append("-D__THIRD__")

ext_third_cpu = Extension("newfastgbthird_cpu", **ext_cpu_third_dict)

if run_cuda_install:
    extensions = [ext_gpu, ext_third_gpu, ext_cpu, ext_third_cpu]

else:
    extensions = [ext_cpu, ext_third_cpu]

fp_out_name = "old_gbgpu/utils/constants.py"
fp_in_name = "include/Constants.h"

# develop few.utils.constants.py
with open(fp_out_name, "w") as fp_out:
    with open(fp_in_name, "r") as fp_in:
        lines = fp_in.readlines()
        for line in lines:
            if len(line.split()) == 3:
                if line.split()[0] == "#define":
                    try:
                        _ = float(line.split()[2])
                        string_out = line.split()[1] + " = " + line.split()[2] + "\n"
                        fp_out.write(string_out)

                    except (ValueError) as e:
                        continue


setup(
    name="old_gbgpu",
    # Random metadata. there's more you can supply
    author="Michael Katz",
    version="0.1",
    packages=["old_gbgpu", "old_gbgpu.utils"],
    py_modules=["old_gbgpu.old_gbgpu", "old_gbgpu.utils.pointeradjust", "old_gbgpu.utils.constants",],
    ext_modules=extensions,
    # Inject our custom trigger
    cmdclass={"build_ext": custom_build_ext},
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
)

for fp in cu_files:
    os.remove("src/" + fp + ".cpp")

for fp in pyx_files:
    os.remove("src/" + fp + "_cpu.pyx")
