![MuNDy](images/mundy_banner_v2_t.png)

# MuNDy: Multibody Nonlocal Dynamics

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Backend-Kokkos](https://img.shields.io/badge/backend-Kokkos-1E88E5.svg)
![Mesh-STK](https://img.shields.io/badge/mesh-Trilinos%2FSTK-4CAF50.svg)

MuNDy is C++ infrastructure for building scalable, biologically grounded microscale multibody dynamics software.

MuNDy supports models with evolving mechanical and relational structure: heterogeneous rigid and flexible bodies; constraints, motors, and contact; growth, division, death, and bonds that form, break, and reorganize; and long-range interactions mediated through a shared medium. Rather than a monolithic simulator, MuNDy builds upon Trilinos/STK’s runtime-extensible entity/part/field data model to provide reusable abstractions and data structures for this problem class. It targets research software developers building domain-specific applications across deployment scales, from laptops and workstations to multi-GPU clusters.

```{toctree}
:hidden:
:maxdepth: 2

primers/index
api
```

```{important}
**Project status (6/12/2026):**  
MuNDy is under active development and rapidly approaching our first formal release. Currently in the "polishing" phase: we have a complete set of core utilities, mathematical tools, geometric and mechanical primitives, and mesh integration machinery. We are now focused on hardening the API, expanding documentation and examples, and building out the Python interface.
```

---

## Table of Contents

- [Documentation](#documentation)
- [Primers](primers/index)
- [C++ API Reference](api)
- [Installation via Spack](#installation-via-spack)
- [Organizational Overview](#organizational-overview)
- [Subpackages](#subpackages)
  - [MundyUtils: Centralized reusable utilities](#mundyutils-centralized-reusable-utilities)
  - [MundyMath: Constexpr, inline mathematics](#mundymath-constexpr-inline-mathematics)
  - [MundyGeom: Geometric primitives and utilities](#mundygeom-geometric-primitives-and-utilities)
  - [MundyMech: Mechanical primitives and utilities](#mundymech-mechanical-primitives-and-utilities-under-construction)
  - [MundyMesh: MuNDy’s extension to Trilinos/STK](#mundymesh-mundys-extension-to-trilinosstk)
  - [MundySearch: Neighbor-list construction and iteration](#mundysearch-neighbor-list-construction-and-iteration)
  - [Standalone Offshoots](#standalone-offshoots)
- [Release Roadmap](#release-roadmap)

---

## Documentation

Use the [hosted Doxygen documentation](https://mundyrepo.github.io/MuNDy/) for public user-facing documentation, design notes, primers, and generated API reference pages. This Sphinx build also provides local [Primers](primers/index) and a local [C++ API Reference](api). You can generate the local Doxygen documentation from the source tree with:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r doc/requirements.txt

module load doxygen
doxygen doc/Doxyfile
```
## Installation via Spack

If you already have Spack installed, start at `spack repo add`. Otherwise, from the MuNDy source tree:

```bash
git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ./spack
. ./spack/share/spack/setup-env.sh

# Add MuNDy's Spack package repository | Tells Spack where to find the MuNDy package recipes
spack repo add ./dep/our_spack_packages
```

You only need to add the MuNDy package repository once per Spack installation. After that, install the configuration you want:

```bash
# MuNDy's reusable header-only core
spack install --add mundy +core

# Core + mesh support
spack install --add mundy +core +mesh

# Core + mesh support with CUDA (example: sm_90 / CUDA 12.3.107)
spack install --add mundy +core +mesh +cuda cuda_arch=90 ^cuda@12.3.107
```

`+mesh` automatically pulls in the required Trilinos `Teuchos` and `STK` support. MuNDy's current subpackages also require `OpenRAND`, which is enabled by default in the Spack recipe.

## Organizational Overview

MuNDy adopts a **Trilinos-style subpackage stack**:
- Lower-level packages provide core infrastructure.
- Higher-level packages may depend on any number of layers beneath them (never the reverse).
- Users can enable only the portions they need by disabling higher-level packages during configuration.

This structure is intended to keep:
- **Core utilities** small, reusable, and dependency-light. (utils, math, geom, mech)
- **Simulation layers** configurable, so applications can opt into only what they need (mesh, search, mbody)

### Code Statistics (via cloc)
```text
cloc-1.96.pl --exclude-dir=TriBITS,ci,doc,scrap ./MuNDy
     424 text files.
     384 unique files.                                          
      46 files ignored.

github.com/AlDanial/cloc v 1.96  T=1.18 s (324.5 files/s, 91526.3 lines/s)
------------------------------------------------------------------
Language            files        blank        comment         code
------------------------------------------------------------------
C/C++ Header          181         8542          15906        34395
C++                    86         5730           6410        26960
Markdown                9          896              0         2787
CMake                  79          530           1638         2430
Python                  5          161            196          659
Bourne Shell           18          130            142          547
JSON                    2           10              0          137
YAML                    4           21              0           87
------------------------------------------------------------------
SUM:                  384        16020          24292        68002
------------------------------------------------------------------
```

---

## Subpackages

### [**MundyUtils**](primers/utils): Centralized reusable utilities

API directory reference: {ref}`mundy/core/utils <dir_mundy_core_utils>`.

Centralized, Kokkos-friendly building blocks for type-level plumbing, error handling, and device-aware data management.
- **{ref}`MUNDY_THROW_ASSERT() <file_mundy_core_utils_src_mundy_utils_throw_assert.hpp>` / {ref}`MUNDY_THROW_REQUIRE() <file_mundy_core_utils_src_mundy_utils_throw_assert.hpp>`**  
  Kokkos-compatible throw/assert helpers with diagnostics and detailed error context 
  - On-device: abort  |  On-host: throw
  - Can be called within constexpr contexts

- **{ref}`mundy::tuple <exhale_struct_structmundy_1_1tuple>` / {ref}`mundy::variant <exhale_struct_structmundy_1_1variant>`**  
  Reduced, Kokkos-compatible analogs of `std::tuple` / `std::variant` for default-constructible types.  
  - {ref}`mundy::tuple <exhale_struct_structmundy_1_1tuple>` is NTTP-compatible and `constexpr`-friendly  

- **{ref}`mundy::reference_wrapper <exhale_class_classmundy_1_1reference__wrapper>`**  
  Kokkos-compatible analog of `std::reference_wrapper` for non-const references.  

- **{ref}`mundy::storage <exhale_class_classmundy_1_1storage>`**
  Unified means to maybe own or maybe view a value using a simple normalized storage policy.

- **{ref}`mundy::aggregate <exhale_class_classmundy_1_1aggregate>`**  
  Compile-time extensible “tagged bag of types” (conceptually similar to `boost::hana::map`).  
  - Kokkos-compatible  
  - `constexpr` and NTTP-compatible  

- **{ref}`mundy::StringLiteral <exhale_struct_structmundy_1_1StringLiteral>`**  
  `constexpr` string literals that are NTTP-compatible.  
  - Supports `constexpr` concatenation

- **{ref}`mundy::StringSink <exhale_struct_structmundy_1_1StringSink>`**  
  A stream-like (`<<`) utility for constructing compile-time and runtime strings.  
  - Kokkos-compatible/constexpr-friendly
  - Automatically produces compile-time strings when possible

- **{ref}`mundy::NgpPool <exhale_class_classmundy_1_1NgpPoolT>` / {ref}`mundy::NgpView <exhale_class_classmundy_1_1NgpViewT>`**  
  Dual-view abstractions that follow MuNDy’s sync semantics plus a dual-view push/pop pool.  
  - Designed to integrate cleanly with Kokkos’ NGP (Next Generation Parallelism) model  

---

### [**MundyMath**](primers/math): `constexpr`, inline mathematics

API directory reference: {ref}`mundy/core/math <dir_mundy_core_math>`.

Small, composable math utilities with view semantics that integrate naturally into Kokkos-based code.
- **{ref}`mundy::Matrix <file_mundy_core_math_src_mundy_math_Matrix.hpp>` / {ref}`mundy::Vector <file_mundy_core_math_src_mundy_math_Vector.hpp>` / {ref}`mundy::Quaternion <file_mundy_core_math_src_mundy_math_Quaternion.hpp>`**  
  Kokkos-compatible, `constexpr` inline linear algebra for small matrix/vector sizes.  
  - NTTP-compatible  
  - View semantics for arbitrary accessors  

- **{ref}`mundy::minimize(...) <file_mundy_core_math_src_mundy_math_minimize.hpp>`** 
  Kokkos-compatible analog of dlib’s `minimize` (L-BFGS) with no dynamic memory allocation.  
  - Callable inside kernels or from host drivers  

- **{ref}`mundy::convex <namespace_mundy__convex>`**  
  Linear complementarity problem (LCP) and constrained convex quadratic programming (QP) solver.  
  - Kokkos-compatible  
  - Can run inside a kernel or orchestrate kernel launches
  - Supports Mixed LCP/QP problems with equality and inequality constraints

- **{ref}`mundy::Hilbert <file_mundy_core_math_src_mundy_math_Hilbert.hpp>` / {ref}`mundy::zmort <file_mundy_core_math_src_mundy_math_zmort.hpp>`**  
  Domain decomposition helpers for Hilbert space-filling curves and Z-morton ordering.  
  - Useful for load balancing and locality-aware particle/domain layout  

---

### [**MundyGeom**](primers/geom): Geometric primitives and utilities

API directory reference: {ref}`mundy/core/geom <dir_mundy_core_geom>`.

Foundational geometric abstractions for multibody dynamics and contact mechanics.
- **Primitives**  
  {ref}`mundy::Point <file_mundy_core_geom_src_mundy_geom_primitives_Point.hpp>`, {ref}`mundy::Line <exhale_class_classmundy_1_1Line>`, {ref}`mundy::LineSegment <exhale_class_classmundy_1_1LineSegment>`, {ref}`mundy::VSegment <exhale_class_classmundy_1_1VSegment>`, {ref}`mundy::Ring <exhale_class_classmundy_1_1Ring>`, {ref}`mundy::Sphere <exhale_class_classmundy_1_1Sphere>`, {ref}`mundy::Spherocylinder <exhale_class_classmundy_1_1Spherocylinder>`, {ref}`mundy::SpherocylinderSegment <exhale_class_classmundy_1_1SpherocylinderSegment>`, {ref}`mundy::Circle3D <exhale_class_classmundy_1_1Circle3D>`, {ref}`mundy::Ellipsoid <exhale_class_classmundy_1_1Ellipsoid>`.

- **{ref}`mundy::distance <file_mundy_core_geom_src_mundy_geom_distance.hpp>`**  
  Utilities for computing:
  - Euclidean separation distances  
  - Shared-normal signed separation distances between primitives  

- **{ref}`mundy::compute_aabb <file_mundy_core_geom_src_mundy_geom_compute_aabb.hpp>` / {ref}`mundy::compute_bounding_radius <file_mundy_core_geom_src_mundy_geom_compute_bounding_radius.hpp>`**  
  Helpers to compute axis-aligned bounding boxes (AABB) and bounding radii for each primitive.

- **{ref}`mundy::transform <file_mundy_core_geom_src_mundy_geom_transform.hpp>` / {ref}`mundy::randomize <file_mundy_core_geom_src_mundy_geom_randomize.hpp>`**  
  Utilities for:
  - Translation  
  - Rotation  
  - Randomization of primitive configurations  

- **{ref}`mundy::periodicity <file_mundy_core_geom_src_mundy_geom_periodicity.hpp>`**  
  Utilities for handling distances and interactions in periodic domains.

---

### [**MundyMech**](primers/mech): Mechanical primitives and utilities (under construction)

API directory reference: {ref}`mundy/core/mech <dir_mundy_core_mech>`.

Mechanical elements and force laws for building multibody models.

- **Primitives**  
  {ref}`mundy::mech::BallJoint <exhale_class_classmundy_1_1BallJoint>`, {ref}`mundy::mech::HookeanSpring <exhale_class_classmundy_1_1HookeanSpring>`, {ref}`mundy::mech::FeneSpring <exhale_class_classmundy_1_1FeneSpring>`, {ref}`mundy::mech::TorsionalSpring <exhale_class_classmundy_1_1TorsionalSpring>`.

Further mechanical models and integration hooks will be added as the library matures.

---

### [**MundyMesh**](primers/mesh): MuNDy’s extension to Trilinos/STK

API directory reference: {ref}`mundy/mesh <dir_mundy_mesh>`.

Helpers and abstractions for integrating MuNDy with Trilinos/STK meshes and fields.
- **{ref}`mundy::mesh::string_to_selector <file_mundy_mesh_src_mundy_mesh_StringToSelector.hpp>` / {ref}`mundy::mesh::string_to_topology <file_mundy_mesh_src_mundy_mesh_StringToTopology.hpp>` / {ref}`mundy::mesh::string_to_rank <file_mundy_mesh_src_mundy_mesh_StringToSelector.hpp>`**  
  Map string descriptions like:
  - Selector expressions: `"(partA | partB) & !partC"`  
  - Topology: `"HEX_8"`  
  - Rank: `"ELEM_RANK"`  
  to their corresponding STK objects.

- **{ref}`mundy::mesh::DeclareEntities <file_mundy_mesh_src_mundy_mesh_DeclareEntities.hpp>` / {ref}`mundy::mesh::DeclareField <file_mundy_mesh_src_mundy_mesh_DeclareField.hpp>` / {ref}`mundy::mesh::DeclarePart <file_mundy_mesh_src_mundy_mesh_DeclarePart.hpp>` / {ref}`mundy::mesh::DeclareClass <file_mundy_mesh_src_mundy_mesh_DeclareClass.hpp>`**  
  Helper functions that streamline common STK mesh setup tasks, such as declaring fields and parts with the correct properties and parallel consistency.

* **{ref}`mundy::mesh::NgpModRequests <file_mundy_mesh_src_mundy_mesh_NgpModRequests.hpp>`**
  A ticket-based framework for staging mesh modification requests from the device and processing them on the host.
    - Requesting new entities (with known or generated Ids)
    - Requesting new connectivity (e.g. element-to-node relations) involving existing or future entities
    - Requesting deletion of existing entities or connectivity
    - Safe and efficient in the face of concurrent requests from multiple threads on the device

- **{ref}`mundy::mesh::FieldViews <file_mundy_mesh_src_mundy_mesh_FieldViews.hpp>`**  
  Helpers for extracting mathematical views into STK field types, both on host and device.  

- **{ref}`mundy::mesh::Classes <exhale_class_classmundy_1_1mesh_1_1Class>`**  
  A utility for mapping a deconstructed class hierarchy (e.g. rod segments, rod ends, contacts) onto STK parts and fields in a consistent IO-compliant way with enforced invariants.

- **{ref}`mundy::mesh::Aggregate <exhale_class_classmundy_1_1mesh_1_1Aggregate>`**  
  Wraps STK fields in their underlying view type, enabling clean code such as  
  ```cpp
  center_accessor(e) += dt * velocity_accessor(e);
  ```
  and aggregation of these accessors to avoid function bloat.

* **{ref}`mundy::mesh::LinkData <exhale_class_classmundy_1_1mesh_1_1LinkData>` / {ref}`mundy::mesh::LinkCOOData <exhale_class_classmundy_1_1mesh_1_1LinkCOOData>` / {ref}`mundy::mesh::LinkCSRData <file_mundy_mesh_src_mundy_mesh_LinkCSRData.hpp>`**
  Kokkos-compatible dynamic connectivity constructs (ghosting contrasts that are themselves entities).
  * Supports dynamically updating COO connectivity
  * Allows on-device sparse updates to CSR structures
  * Follows dual-view-like semantics aligned with STK’s NGP design
  * Automatic synchronization tracking during mesh modification cycles

* **{ref}`mundy::mesh::NgpFieldBLAS <file_mundy_mesh_src_mundy_mesh_NgpFieldBLAS.hpp>`**
  Reimplementation of STK’s field BLAS routines with unified host/device syntax.

* **{ref}`mundy::mesh::NgpAccessorExpr <file_mundy_mesh_src_mundy_mesh_NgpAccessorExpr.hpp>`**
  MuNDy’s usability layer: a templated expression system with:
  * Automatic pruning of reused branches
  * Automatic synchronization of read fields
  * Automatic marking of modified fields as dirty

  This lets users write expressions like:
  ```cpp
  x(rods) += dt * vel(rods);
  ```
  and have them executed on the device without manual synchronization bookkeeping.

---

### [**MundySearch**](https://mundyrepo.github.io/MuNDy/MundySearch.html): Neighbor-list construction and iteration

MundySearch builds and iterates neighbor lists over STK mesh entities.
- **Search inputs** pair selectors with geometry components so builders know which entities to search and how to read their detection regions.
- **NeighborListBuilder** provides a fluent, type-safe interface for constructing backend-specific lists.
- **Excluders** filter candidate pairs during the build, including self-interaction, symmetric duplicate, and narrow-phase filters.
- **ManagedNeighborList** caches neighbor lists across time steps and rebuilds them only when selected policies require it.
- **for_each_neighbor_pair / for_each_target_with_neighbors** expose backend-independent parallel iteration over stored neighbor relationships.

---

### Standalone Offshoots

Independent projects that emerged from MuNDy’s infrastructure and are usable on their own.
- **[OpenRAND](https://github.com/msu-sparta/OpenRAND)**
  Performance-portable, counter-based random number generation that is stupid simple to use.
  - Designed to easily fit in GPU registers
  - Makes reproducibility in spite of varied parallelism possible
  - Now used by HOOMD-Blue
  
- **[alsous_gigantism_2025](https://github.com/flatironinstitute/alsous_gigantism_2025)**
  A discrete elastic rod model implemented using MuNDy. 

- **[mundy_mock_app](https://github.com/MundyRepo/mundy_mock_app)** /
  **[mundy_mock_app_tribits](https://github.com/MundyRepo/mundy_mock_app_tribits)**
  Helper applications for bootstrapping MuNDy-based codes:
  - CMake-based or TriBITS+CMake templates
  - Intended as starting points for internal and external applications that depend on MuNDy

---

## Release Roadmap

Planned steps toward the first public release (estimated summer 2026):
- Python API mirroring accessor expressions
- Tutorial + Example applications
