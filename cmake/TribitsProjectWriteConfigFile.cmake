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
# module via include(TribitsProjectWriteConfigFile) over CMAKE_MODULE_PATH, so
# to shadow the upstream module our file must carry the exact same name and
# stay in ${PROJECT_SOURCE_DIR}/cmake so TriBITS resolves it first. All real
# override logic lives in the Mundy-named module that we include below.
#
# Public API: find_package(Mundy) remains the project-level entry point.
# MundyPackageConfig.cmake is a private helper that the public project config
# may include internally when the project and one package share the same name.
include("${CMAKE_CURRENT_LIST_DIR}/MundyTribitsProjectWriteConfigFile.cmake")
