# Sonar flash attention requires SONAR_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set SONAR_GPU_ARCHES here.
if(SONAR_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND SONAR_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

#
# Build Sonar flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because sonar-flash-attn uses the same macros/functions as Sonar.
# Because functions all belong to the global scope, sonar-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The sonar-flash-attn install rules are nested under sonar to make sure the library gets installed in the correct place.
# To only install sonar-flash-attn, use --component _vllm_fa2_C (for FA2), --component _vllm_fa3_C (for FA3),
# or --component _vllm_fa4_cutedsl_C (for FA4 CuteDSL Python files).
# If no component is specified, sonar-flash-attn is still installed.

# If SONAR_FLASH_ATTN_SRC_DIR is set, sonar-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of sonar-flash-attn within Sonar.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{SONAR_FLASH_ATTN_SRC_DIR})
  set(SONAR_FLASH_ATTN_SRC_DIR $ENV{SONAR_FLASH_ATTN_SRC_DIR})
endif()

if(SONAR_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          sonar-flash-attn SOURCE_DIR
          ${SONAR_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/sonar-flash-attn
  )
else()
  FetchContent_Declare(
          sonar-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG af5422c5c24e2f464d394295c5b2f659f9332217
          GIT_PROGRESS TRUE
          # Don't share the sonar-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/sonar-flash-attn
  )
endif()

# Make sure sonar-flash-attn install rules are nested under sonar/
# ALL_COMPONENTS ensures the save/modify/restore runs exactly once regardless
# of how many components are being installed, avoiding double-append of /sonar/.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" ALL_COMPONENTS)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/sonar/\")" ALL_COMPONENTS)

# Fetch the sonar-flash-attn library
FetchContent_MakeAvailable(sonar-flash-attn)
message(STATUS "sonar-flash-attn is available at ${sonar-flash-attn_SOURCE_DIR}")

# Restore the install prefix after FA's install rules
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# Install shared Python files for both FA2 and FA3 components
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  # Ensure the sonar/vllm_flash_attn directory exists before installation
  install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/sonar/vllm_flash_attn\")"
    COMPONENT ${_FA_COMPONENT})

  # Copy vllm_flash_attn python files (except __init__.py and flash_attn_interface.py
  # which are source-controlled in sonar)
  install(
    DIRECTORY ${sonar-flash-attn_SOURCE_DIR}/vllm_flash_attn/
    DESTINATION sonar/vllm_flash_attn
    COMPONENT ${_FA_COMPONENT}
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__init__.py" EXCLUDE
    PATTERN "flash_attn_interface.py" EXCLUDE
  )

endforeach()

#
# FA4 CuteDSL component
# This is a Python-only component that copies the flash_attn/cute directory
# and transforms imports to match our package structure.
#
add_custom_target(_vllm_fa4_cutedsl_C)

# Install flash_attn/cute directory (needed for FA4).
# When using a local source dir (SONAR_FLASH_ATTN_SRC_DIR), create a symlink
# so edits to cute-dsl Python files take effect immediately without rebuilding.
# Otherwise, copy files and transform flash_attn.cute imports to
# sonar.vllm_flash_attn.cute to match our package structure.
if(SONAR_FLASH_ATTN_SRC_DIR)
  install(CODE "
    set(LINK_TARGET \"${sonar-flash-attn_SOURCE_DIR}/flash_attn/cute\")
    set(LINK_NAME \"\${CMAKE_INSTALL_PREFIX}/sonar/vllm_flash_attn/cute\")
    file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/sonar/vllm_flash_attn\")
    file(REMOVE_RECURSE \"\${LINK_NAME}\")
    file(CREATE_LINK \"\${LINK_TARGET}\" \"\${LINK_NAME}\" SYMBOLIC)
  " COMPONENT _vllm_fa4_cutedsl_C)
else()
  install(CODE "
    file(GLOB_RECURSE CUTE_PY_FILES \"${sonar-flash-attn_SOURCE_DIR}/flash_attn/cute/*.py\")
    foreach(SRC_FILE \${CUTE_PY_FILES})
      file(RELATIVE_PATH REL_PATH \"${sonar-flash-attn_SOURCE_DIR}/flash_attn/cute\" \${SRC_FILE})
      set(DST_FILE \"\${CMAKE_INSTALL_PREFIX}/sonar/vllm_flash_attn/cute/\${REL_PATH}\")
      get_filename_component(DST_DIR \${DST_FILE} DIRECTORY)
      file(MAKE_DIRECTORY \${DST_DIR})
      file(READ \${SRC_FILE} FILE_CONTENTS)
      string(REPLACE \"flash_attn.cute\" \"sonar.vllm_flash_attn.cute\" FILE_CONTENTS \"\${FILE_CONTENTS}\")
      file(WRITE \${DST_FILE} \"\${FILE_CONTENTS}\")
    endforeach()
  " COMPONENT _vllm_fa4_cutedsl_C)
endif()
