# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

include_guard(GLOBAL)

# This same-name module intentionally shadows the upstream TriBITS module of
# the same name. The thin shim stays in ${PROJECT_SOURCE_DIR}/cmake so TriBITS
# resolves it first, while the real implementation lives in a dedicated
# overrides directory.
#
# Public API: find_package(Mundy) remains the project-level entry point.
# MundyPackageConfig.cmake is a private helper that the public project config
# may include internally when the project and one package share the same name.
include("${CMAKE_CURRENT_LIST_DIR}/tribits_overrides/TribitsProjectWriteConfigFile.cmake")
