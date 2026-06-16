# Mundy TriBITS customizations

This directory holds Mundy's local TriBITS customizations (alongside the
vendored `TriBITS/`, the `TPLs/` find-modules, and the ctest/valgrind helpers).
All of Mundy's own customization modules are `Mundy*`-named and live flat here.

There are two distinct kinds of customization:

## 1. Additive modules (new `mundy_tribits_*` functions)

Included explicitly by the top-level `CMakeLists.txt`; they define new functions
the project calls instead of the stock TriBITS ones. They do not shadow anything.

- `MundyTribitsAddLibrary.cmake` — `mundy_tribits_add_library()`.
- `MundyTribitsAddExecutableAndTest.cmake` — `mundy_tribits_add_executable_and_test()`
  (adds `MPI_PROC_RANGE`).
- `MundyTribitsInstallDirectories.cmake` — `mundy_tribits_install_directories()`.
- `MundyTribitsProject.cmake` — `mundy_tribits_project()` and the public-config
  contract test registration.

## 2. Override modules (shadow upstream package/project config generation)

- `MundyTribitsInternalPackageWriteConfigFile.cmake` — package-config writer overrides.
- `MundyTribitsProjectWriteConfigFile.cmake` — project-config writer override.
- `MundyTribitsPackageConfigTemplate.cmake.in` / `MundyTribitsProjectConfigTemplate.cmake.in`
  — local copies of the TriBITS config templates.
- `MundyPublicConfigContract.cmake` — `cmake -P` test (registered as
  `MundyPublicConfigContract`) that pins the public contract below.

### Public contract
- Downstream users call `find_package(Mundy)` for the top-level project.
- Downstream users call `find_package(MundyUtils)`, `find_package(MundyMath)`,
  `find_package(MundyMesh)`, etc. for the readable subpackages.
- Downstream users link with `Mundy_LIBRARIES`, `Mundy::all_libs`,
  `MundyMath_LIBRARIES`, `MundyMath::all_libs`, and similar public names.

### Private implementation detail
- `MundyPackageConfig.cmake` and the `MundyPackage_*` variable namespace exist
  only to avoid the self-name collision between the `Mundy` project and the
  `Mundy` package.
- Downstream code must never reference `MundyPackage_*` directly.

## The two non-`Mundy`-named shim files

`TribitsInternalPackageWriteConfigFile.cmake` and
`TribitsProjectWriteConfigFile.cmake` are the only customization files here that
are not `Mundy`-prefixed, because their names are dictated by TriBITS:

- TriBITS loads them via `include(TribitsInternalPackageWriteConfigFile)` /
  `include(TribitsProjectWriteConfigFile)` over `CMAKE_MODULE_PATH`, and resets
  `CMAKE_MODULE_PATH` inside `tribits_project_impl()`.
- To shadow the upstream modules, the override entry points must therefore be
  reached through files with those exact upstream names, sitting in this
  directory. Their filenames cannot be `Mundy`-prefixed.
- Each shim is thin and simply delegates into its `Mundy*` module here, so the
  real override logic stays easy to find.
