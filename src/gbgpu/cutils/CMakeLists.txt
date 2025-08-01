# =============================
# ==== BACKEND COMPILATION ====
# =============================

# TODO: how can we avoid moving source and header files in this directory ?
#include_directories(BEFORE "include")
#include_directories(BEFORE "src")

# In the project root CMakeLists.txt, we defined a "gbgpu" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(GBGPU_WITH_CPU gbgpu WITH_CPU)
get_target_property(GBGPU_WITH_GPU gbgpu WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
# TODO: is this needed ?
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${gbgpu_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# Declare the CPU backend
# We compile the files in a module called gbgpu_utils_cpu
if(GBGPU_WITH_CPU)

  set(module_name "gbgpu_utils_cpu")
  set(cythonized_file "gbgpu_utils_wrap_cpu.cxx")

  # Process gbgpu_utils_wrap.pyx into a C++ file
  add_custom_command(
    OUTPUT "${cythonized_file}"
    COMMENT "Cythonize gbgpu_utils_wrap.pyx into ${cythonized_file}"
    COMMAND
      Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
      -3 -+
      --output-file "${CMAKE_CURRENT_BINARY_DIR}/${cythonized_file}"
      --module-name "${module_name}"
      -I "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "gbgpu_utils_wrap.pyx"
    VERBATIM)

  # Copy CUDA file into a C++
  # TODO: use cmake -E create_hardlink ?
  add_custom_command(
    OUTPUT "gbgpu_utils.cxx"
    COMMENT "Copy gbgpu_utils.cu to gbgpu_utils.cxx"
    COMMAND
      ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils.cu"
      "${CMAKE_CURRENT_BINARY_DIR}/gbgpu_utils.cxx"
    DEPENDS "gbgpu_utils.cu"
    VERBATIM)

  # Declare the python compiled module
  python_add_library(
    "${module_name}"
    MODULE WITH_SOABI "${cythonized_file}" gbgpu_utils.cxx)

  # Compilation options
  target_include_directories(
    "${module_name}"
    PRIVATE ${Python_NumPy_INCLUDE_DIR})

  target_compile_definitions(
    "${module_name}"
    PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)

  target_sources(
    "${module_name}" PUBLIC FILE_SET HEADERS
    FILES cuda_complex.hpp global.h gbgpu_utils.hh)

  get_target_property(GBGPU_CXX_MARCH_OPT gbgpu CXX_MARCH)
  if(GBGPU_CXX_MARCH_OPT)
    target_compile_options(
      "${module_name}"
      PRIVATE "${GBGPU_CXX_MARCH_OPT}")
  endif()

  # Installation options
  install(TARGETS "${module_name}" DESTINATION "gbgpu/cutils")

endif()

# III. Declare the GPU backend
# We compile the files in a module called gbgpu_utils_gpu
if(GBGPU_WITH_GPU)

  set(module_name "gbgpu_utils_gpu")
  set(cythonized_file "gbgpu_utils_wrap_gpu.cxx")

  # Process gbgpu_utils_wrap.pyx into a C++ file
  add_custom_command(
    OUTPUT "${cythonized_file}"
    COMMENT "Cythonize gbgpu_utils_wrap.pyx into ${cythonized_file}"
    COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
    -3 -+
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/${cythonized_file}"
    --module-name "${module_name}"
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS "gbgpu_utils_wrap.pyx"
    VERBATIM)

  # Declare the python compiled module
  python_add_library(
    "${module_name}"
    MODULE WITH_SOABI "${cythonized_file}" gbgpu_utils.cu)

  # Compilation options
  target_include_directories(
    "${module_name}"
    PRIVATE ${Python_NumPy_INCLUDE_DIR})

  target_compile_definitions(
    "${module_name}"
    PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)

  target_sources(
    "${module_name}" PUBLIC FILE_SET HEADERS
    FILES cuda_complex.hpp global.h gbgpu_utils.hh)

  # Explicitly require dynamic linking of cudart
  set_property(
    TARGET "${module_name}"
    PROPERTY CUDA_RUNTIME_LIBRARY "Shared")

  set_property(
    TARGET "${module_name}"
    PROPERTY CUDA_ARCHITECTURES ${GBGPU_CUDA_ARCH})

  # Installation options
  install(TARGETS "${module_name}" DESTINATION "gbgpu/cutils")

endif()
