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

# apply_gpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to GPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "gbgpu_gpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the GPU backend directory (e.g.
#   'gbgpu_backend_cuda12x')
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
# * Ensures the target links against CUDA libraries (cuBLAS, cuSPARSE, ...)
# * Defines the CUDA_ARCHITECTURE property
#
# Usage example: apply_gpu_backend_common_options(gbgpu_utils_wrap)
function(apply_gpu_backend_common_options libname)
  set(target_name "gbgpu_gpu_${libname}")
  set(backend_name "gbgpu_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${GBGPU_CUDA_ARCH})
endfunction()

function(apply_cpu_backend_common_options libname)
  set(target_name "gbgpu_cpu_${libname}")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/gbgpu_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION gbgpu_backend_cpu)

  get_target_property(GBGPU_CXX_MARCH_OPT gbgpu CXX_MARCH)
  if(GBGPU_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${GBGPU_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- utils ---
# ----------------

# I. Process gbgpu_utils_wrap.pyx into a C++ file
add_custom_command(
  OUTPUT "utils.cxx"
  COMMENT "Cythonize gbgpu_utils_wrap.pyx into utils.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/utils.cxx" -3 -+ --module-name
    "utils" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "gbgpu_utils_wrap.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(GBGPU_WITH_CPU)
  add_custom_command(
    OUTPUT "gbgpu_utils.cxx"
    COMMENT "Copy gbgpu_utils.cu to gbgpu_utils.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gbgpu_utils.cxx"
    DEPENDS "gbgpu_utils.cu"
    VERBATIM)

  python_add_library(gbgpu_cpu_utils MODULE WITH_SOABI utils.cxx gbgpu_utils.cxx)
  apply_cpu_backend_common_options(utils)

  target_sources(gbgpu_cpu_utils PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h gbgpu_utils.hh)
endif()

# III. Declare the GPU backend
if(GBGPU_WITH_GPU)
  python_add_library(gbgpu_gpu_utils MODULE WITH_SOABI utils.cxx gbgpu_utils.cu)
  apply_gpu_backend_common_options(utils)
  target_sources(gbgpu_gpu_utils PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h gbgpu_utils.hh)
endif()


# Declare the CPU backend
# We compile the files in a module called gbgpu_utils_cpu
# if(GBGPU_WITH_CPU)

#   set(module_name "gbgpu_cpu_utils")
#   set(cythonized_file "gbgpu_utils_wrap_cpu.cxx")

#   # Process gbgpu_utils_wrap.pyx into a C++ file
#   add_custom_command(
#     OUTPUT "${cythonized_file}"
#     COMMENT "Cythonize gbgpu_utils_wrap.pyx into ${cythonized_file}"
#     COMMAND
#       Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
#       -3 -+
#       --output-file "${CMAKE_CURRENT_BINARY_DIR}/${cythonized_file}"
#       --module-name "${module_name}"
#       -I "${CMAKE_CURRENT_SOURCE_DIR}"
#     DEPENDS "gbgpu_utils_wrap.pyx"
#     VERBATIM)

#   # Copy CUDA file into a C++
#   # TODO: use cmake -E create_hardlink ?
#   add_custom_command(
#     OUTPUT "gbgpu_utils.cxx"
#     COMMENT "Copy gbgpu_utils.cu to gbgpu_utils.cxx"
#     COMMAND
#       ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils.cu"
#       "${CMAKE_CURRENT_BINARY_DIR}/gbgpu_utils.cxx"
#     DEPENDS "gbgpu_utils.cu"
#     VERBATIM)

#   # Declare the python compiled module
#   python_add_library(
#     "${module_name}"
#     MODULE WITH_SOABI "${cythonized_file}" gbgpu_utils.cxx)

#   # Compilation options
#   target_include_directories(
#     "${module_name}"
#     PRIVATE ${Python_NumPy_INCLUDE_DIR})

#   target_compile_definitions(
#     "${module_name}"
#     PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)

#   target_sources(
#     "${module_name}" PUBLIC FILE_SET HEADERS
#     FILES cuda_complex.hpp global.h gbgpu_utils.hh)

#   get_target_property(GBGPU_CXX_MARCH_OPT gbgpu CXX_MARCH)
#   if(GBGPU_CXX_MARCH_OPT)
#     target_compile_options(
#       "${module_name}"
#       PRIVATE "${GBGPU_CXX_MARCH_OPT}")
#   endif()

#   # Installation options
#   install(TARGETS "${module_name}" DESTINATION gbgpu_backend_cpu)

# endif()

# # III. Declare the GPU backend
# # We compile the files in a module called gbgpu_utils_gpu
# if(GBGPU_WITH_GPU)

#   set(module_name "gbgpu_utils")
#   set(cythonized_file "gbgpu_utils_wrap_gpu.cxx")
#   set(backend_name "gbgpu_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")

#   # Process gbgpu_utils_wrap.pyx into a C++ file
#   add_custom_command(
#     OUTPUT "${cythonized_file}"
#     COMMENT "Cythonize gbgpu_utils_wrap.pyx into ${cythonized_file}"
#     COMMAND
#     Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
#     -3 -+
#     --output-file "${CMAKE_CURRENT_BINARY_DIR}/${cythonized_file}"
#     --module-name "${module_name}"
#     -I "${CMAKE_CURRENT_SOURCE_DIR}"
#     DEPENDS "gbgpu_utils_wrap.pyx"
#     VERBATIM)

#   # Declare the python compiled module
#   python_add_library(
#     "${module_name}"
#     MODULE WITH_SOABI "${cythonized_file}" gbgpu_utils.cu)

#   # Compilation options
#   target_include_directories(
#     "${module_name}"
#     PRIVATE ${Python_NumPy_INCLUDE_DIR})

#   target_compile_definitions(
#     "${module_name}"
#     PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)

#   target_sources(
#     "${module_name}" PUBLIC FILE_SET HEADERS
#     FILES cuda_complex.hpp global.h gbgpu_utils.hh)

#   # Explicitly require dynamic linking of cudart
#   set_property(
#     TARGET "${module_name}"
#     PROPERTY CUDA_RUNTIME_LIBRARY "Shared")

#   set_property(
#     TARGET "${module_name}"
#     PROPERTY CUDA_ARCHITECTURES ${GBGPU_CUDA_ARCH})

#   # Installation options
#   install(TARGETS "${module_name}" DESTINATION ${backend_name})


