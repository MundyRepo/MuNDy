# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

include_guard(GLOBAL)

# This is a thin shim. Its filename is dictated by TriBITS: TriBITS loads this
# module via include(TribitsInternalPackageWriteConfigFile) over
# CMAKE_MODULE_PATH, so to shadow the upstream module our file must carry the
# exact same name and live in ${PROJECT_SOURCE_DIR}/cmake (TriBITS resets
# CMAKE_MODULE_PATH inside tribits_project_impl()). All real override logic
# lives in the Mundy-named module that we include below.
#
# Public API: downstream users call find_package(Mundy) or
# find_package(Mundy<Subpackage>). MundyPackageConfig.cmake and MundyPackage_*
# are private implementation details used only to break the self-name collision
# between the Mundy project and the Mundy package.
include("${CMAKE_CURRENT_LIST_DIR}/MundyTribitsInternalPackageWriteConfigFile.cmake")
