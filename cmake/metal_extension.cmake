# SPDX-License-Identifier: Apache-2.0

if(NOT APPLE)
  message(FATAL_ERROR "The Metal extension can only be built on macOS.")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "The Metal extension requires Apple Silicon.")
endif()

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import pathlib, nanobind; print(pathlib.Path(nanobind.__file__).resolve().parent)"
  OUTPUT_VARIABLE NANOBIND_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import pathlib, mlx.core; print(pathlib.Path(mlx.core.__file__).resolve().parent)"
  OUTPUT_VARIABLE MLX_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

set(NANOBIND_SRC "${NANOBIND_PACKAGE_DIR}/src/nb_combined.cpp")
set(MLX_INCLUDE_DIR "${MLX_PACKAGE_DIR}/include")
set(MLX_LIB_DIR "${MLX_PACKAGE_DIR}/lib")
set(MLX_METAL_CPP_DIR "${MLX_INCLUDE_DIR}/metal_cpp")

foreach(path IN ITEMS
    "${NANOBIND_PACKAGE_DIR}/include"
    "${NANOBIND_PACKAGE_DIR}/src"
    "${NANOBIND_PACKAGE_DIR}/ext/robin_map/include"
    "${NANOBIND_SRC}"
    "${MLX_INCLUDE_DIR}"
    "${MLX_METAL_CPP_DIR}"
    "${MLX_LIB_DIR}/libmlx.dylib")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Required Metal build dependency path does not exist: ${path}")
  endif()
endforeach()

Python_add_library(_paged_ops MODULE WITH_SOABI
  "${NANOBIND_SRC}"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal/paged_ops.cpp")

target_include_directories(_paged_ops PRIVATE
  "${NANOBIND_PACKAGE_DIR}/include"
  "${NANOBIND_PACKAGE_DIR}/src"
  "${NANOBIND_PACKAGE_DIR}/ext/robin_map/include"
  "${MLX_INCLUDE_DIR}"
  "${MLX_METAL_CPP_DIR}")

target_compile_definitions(_paged_ops PRIVATE
  _METAL_
  ACCELERATE_NEW_LAPACK
  APHRODITE_METAL_PARTITION_SIZE=512)

target_compile_options(_paged_ops PRIVATE
  -fvisibility=default)

target_link_libraries(_paged_ops PRIVATE
  "${MLX_LIB_DIR}/libmlx.dylib"
  "-framework Metal"
  "-framework Foundation")

target_link_options(_paged_ops PRIVATE
  "-undefined"
  "dynamic_lookup")

set_target_properties(_paged_ops PROPERTIES
  BUILD_RPATH "${MLX_LIB_DIR}"
  INSTALL_RPATH "${MLX_LIB_DIR}")

install(TARGETS _paged_ops
  LIBRARY DESTINATION aphrodite/metal/metal
  COMPONENT _paged_ops)
