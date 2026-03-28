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
# the same name. TriBITS resets CMAKE_MODULE_PATH inside tribits_project_impl(),
# so the shadow must live in ${PROJECT_SOURCE_DIR}/cmake. The real
# implementation is kept in a dedicated overrides directory to make that design
# explicit.
#
# Public API: downstream users call find_package(Mundy) or
# find_package(Mundy<Subpackage>). MundyPackageConfig.cmake and MundyPackage_*
# are private implementation details used only to break the self-name collision
# between the Mundy project and the Mundy package.
include("${CMAKE_CURRENT_LIST_DIR}/tribits_overrides/TribitsInternalPackageWriteConfigFile.cmake")
