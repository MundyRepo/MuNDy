# TriBITS Overrides

This directory contains Mundy's explicit local overrides for a small subset of
TriBITS package-config generation logic.

Public contract:
- Downstream users call `find_package(Mundy)` for the top-level project.
- Downstream users call `find_package(MundyUtils)`, `find_package(MundyMath)`,
  `find_package(MundyMesh)`, etc. for the readable subpackages.
- Downstream users link with `Mundy_LIBRARIES`, `Mundy::all_libs`,
  `MundyMath_LIBRARIES`, `MundyMath::all_libs`, and similar public names.

Private implementation detail:
- `MundyPackageConfig.cmake` and the `MundyPackage_*` variable namespace exist
  only to avoid the self-name collision between the `Mundy` project and the
  `Mundy` package.
- Downstream code must never reference `MundyPackage_*` directly.

Why the top-level same-name files still exist:
- TriBITS resets `CMAKE_MODULE_PATH` inside `tribits_project_impl()`.
- Because of that, the actual override points must still be reached through
  same-name shim files in `${PROJECT_SOURCE_DIR}/cmake`.
- Those shims delegate here so the real override logic stays easy to find.
