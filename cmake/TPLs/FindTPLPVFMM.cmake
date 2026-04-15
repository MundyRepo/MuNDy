tribits_tpl_allow_pre_find_package(pvfmm pvfmm_ALLOW_PREFIND)

if (pvfmm_ALLOW_PREFIND)

  if (DEFINED TPL_PVFMM_DIR)
    find_package(pvfmm)
  else()
    message(FATAL_ERROR "TPL_PVFMM_DIR must be defined before calling find_package(pvfmm).")
  endif()

  if (pvfmm_FOUND)
    # Tell TriBITS that we found PVFMM and there no need to look any further
    set(TPL_PVFMM_INCLUDE_DIRS ${PVFMM_INCLUDE_DIR}/pvfmm ${PVFMM_DEP_INCLUDE_DIR} CACHE PATH "...")
    set(TPL_PVFMM_LIBRARIES ${PVFMM_LIB_DIR}/${PVFMM_STATIC_LIB} ${PVFMM_DEP_LIB} CACHE FILEPATH "...")
    set(TPL_PVFMM_LIBRARY_DIRS ${PVFMM_LIB_DIR} CACHE PATH "...")

    # Print out the results
    message(STATUS "Found PVFMM: ${PVFMM_VERSION}")
    message(STATUS "  PVFMM_LIB_DIR: ${PVFMM_LIB_DIR}")
    message(STATUS "  PVFMM_INCLUDE_DIR: ${PVFMM_INCLUDE_DIR}")
    message(STATUS "  TPL_PVFMM_INCLUDE_DIRS: ${TPL_PVFMM_INCLUDE_DIRS}")
    message(STATUS "  TPL_PVFMM_LIBRARIES: ${TPL_PVFMM_LIBRARIES}")
    message(STATUS "  TPL_PVFMM_LIBRARY_DIRS: ${TPL_PVFMM_LIBRARY_DIRS}")
  endif()

endif()


set(REQUIRED_HEADERS pvfmm.hpp pvfmm.h)
set(REQUIRED_LIBS_NAMES pvfmm)
tribits_tpl_find_include_dirs_and_libraries( PVFMM
  REQUIRED_HEADERS ${REQUIRED_HEADERS}
  REQUIRED_LIBS_NAMES ${REQUIRED_LIBS_NAMES} )