![MuNDy](doc/images/mundy_banner_v2_t.png)

# MuNDy: Multibody Nonlocal Dynamics

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Backend-Kokkos](https://img.shields.io/badge/backend-Kokkos-1E88E5.svg)
![Mesh-STK](https://img.shields.io/badge/mesh-Trilinos%2FSTK-4CAF50.svg)

MuNDy is a C++ infrastructure for building scalable, biologically grounded microscale multibody dynamics software.

MuNDy supports models with evolving mechanical and relational structure: heterogeneous rigid and flexible bodies; constraints, motors, and contacts; growth, division, death, and bonds that form, break, and reorganize; and long-range interactions mediated through a shared medium. Rather than a monolithic simulator, MuNDy builds upon Trilinos/STK's runtime-extensible entity/part/field data model to provide reusable abstractions and data structures for this problem class. It is designed for research software developers building domain-specific applications across deployment scales, from laptops and workstations to multi-GPU clusters.

> [!IMPORTANT]
> **Project status (6/12/2026):**  
> MuNDy is under active development and rapidly approaching our first formal release. Currently in the "polishing" phase: we have a complete set of core utilities, mathematical tools, geometric and mechanical primitives, and mesh integration machinery. We are now focused on hardening the API, expanding documentation and examples, and building out the Python interface.

---

## Table of Contents

- [Documentation](#documentation)
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

Use the [hosted Doxygen documentation](https://mundyrepo.github.io/MuNDy/) for user-facing documentation, design notes, primers, and generated API reference pages. Start with the [Primers](https://mundyrepo.github.io/MuNDy/pages.html), then use the generated [class](https://mundyrepo.github.io/MuNDy/classes.html), [file](https://mundyrepo.github.io/MuNDy/files.html), and [namespace](https://mundyrepo.github.io/MuNDy/namespaces.html) indexes as needed. You can also generate the local Doxygen documentation from the source tree with:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r doc/requirements.txt

# Assumes doxygen is in PATH and has version >= 1.9 | version >= 1.11 recommended for best results
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

### [**MundyUtils**](https://mundyrepo.github.io/MuNDy/MundyUtils.html): Centralized reusable utilities

Centralized, Kokkos-friendly building blocks for type-level plumbing, error handling, and device-aware data management.
- **MUNDY_THROW_ASSERT() / MUNDY_THROW_REQUIRE()**  
  Kokkos-compatible throw/assert helpers with diagnostics and detailed error context 
  - On-device: abort  |  On-host: throw
  - Can be called within constexpr contexts

- **mundy::tuple / mundy::variant**  
  Reduced, Kokkos-compatible analogs of `std::tuple` / `std::variant` for default-constructible types.  
  - mundy::tuple is NTTP-compatible and `constexpr`-friendly  

- **mundy::reference_wrapper**  
  Kokkos-compatible analog of `std::reference_wrapper` for non-const references.  

- **mundy::storage**
  Unified means to maybe own or maybe view a value using a simple normalized storage policy.

- **mundy::aggregate**  
  Compile-time extensible “tagged bag of types” (conceptually similar to `boost::hana::map`).  
  - Kokkos-compatible  
  - `constexpr` and NTTP-compatible  

- **mundy::StringLiteral**  
  `constexpr` string literals that are NTTP-compatible.  
  - Supports `constexpr` concatenation

- **mundy::StringSink**  
  A stream-like (`<<`) utility for constructing compile-time and runtime strings.  
  - Kokkos-compatible/constexpr-friendly
  - Automatically produces compile-time strings when possible

- **mundy::NgpPool / mundy::NgpView**  
  Dual-view abstractions that follow MuNDy’s sync semantics plus a dual-view push/pop pool.  
  - Designed to integrate cleanly with Kokkos’ NGP (Next Generation Parallelism) model  

---

### [**MundyMath**](https://mundyrepo.github.io/MuNDy/MundyMath.html): `constexpr`, inline mathematics

Small, composable math utilities with view semantics that integrate naturally into Kokkos-based code.
- **mundy::Matrix / mundy::Vector / mundy::Quaternion**  
  Kokkos-compatible, `constexpr` inline linear algebra for small matrix/vector sizes.  
  - NTTP-compatible  
  - View semantics for arbitrary accessors  

- **mundy::minimize(...)** 
  Kokkos-compatible analog of dlib’s `minimize` (L-BFGS) with no dynamic memory allocation.  
  - Callable inside kernels or from host drivers  

- **mundy::convex**  
  Linear complementarity problem (LCP) and constrained convex quadratic programming (QP) solver.
  - Kokkos-compatible  
  - Can run inside a kernel or orchestrate kernel launches
  - Supports Mixed LCP/QP problems with equality and inequality constraints

- **mundy::Hilbert / mundy::zmort**  
  Domain decomposition helpers for Hilbert space-filling curves and Z-morton ordering.  
  - Useful for load balancing and locality-aware particle/domain layout  

---

### [**MundyGeom**](https://mundyrepo.github.io/MuNDy/MundyGeom.html): Geometric primitives and utilities

Foundational geometric abstractions for multibody dynamics and contact mechanics.
- **Primitives**  
  mundy::Point, mundy::Line, mundy::LineSegment, mundy::VSegment, mundy::Ring, mundy::Sphere, mundy::Spherocylinder, mundy::SpherocylinderSegment, mundy::Circle3D, mundy::Ellipsoid.

- **mundy::distance**  
  Utilities for computing:
  - Euclidean separation distances  
  - Shared-normal signed separation distances between primitives  

- **mundy::compute_aabb / mundy::compute_bounding_radius**  
  Helpers to compute axis-aligned bounding boxes (AABB) and bounding radii for each primitive.

- **mundy::transform / mundy::randomize**  
  Utilities for:
  - Translation  
  - Rotation  
  - Randomization of primitive configurations  

- **mundy::periodicity**  
  Utilities for handling distances and interactions in periodic domains.

---

### [**MundyMech**](https://mundyrepo.github.io/MuNDy/MundyMech.html): Mechanical primitives and utilities (under construction)

Mechanical elements and force laws for building multibody models.

- **Primitives**  
  mundy::mech::BallJoint, mundy::mech::HookeanSpring, mundy::mech::FeneSpring, mundy::mech::TorsionalSpring.

Further mechanical models and integration hooks will be added as the library matures.

---

### [**MundyMesh**](https://mundyrepo.github.io/MuNDy/MundyMesh.html): MuNDy’s extension to Trilinos/STK

Helpers and abstractions for integrating MuNDy with Trilinos/STK meshes and fields.
- **mundy::mesh::string_to_selector / mundy::mesh::string_to_topology / mundy::mesh::string_to_rank**  
  Map string descriptions like:
  - Selector expressions: `"(partA | partB) & !partC"`  
  - Topology: `"HEX_8"`  
  - Rank: `"ELEM_RANK"`  
  to their corresponding STK objects.

- **mundy::mesh::EntityDeclaration / mundy::mesh::FieldDeclaration mundy::mesh::ComponentDeclaration / mundy::mesh::PartDeclaration / mundy::mesh::ClassDeclaration**  
  Helper functions that streamline common STK mesh setup tasks, such as declaring fields and parts with the correct properties and parallel consistency.

* **mundy::mesh::NgpModRequests**
  A ticket-based framework for staging mesh modification requests from the device and processing them on the host.
    - Requesting new entities (with known or generated Ids)
    - Requesting new connectivity (e.g. element-to-node relations) involving existing or future entities
    - Requesting deletion of existing entities or connectivity
    - Safe and efficient in the face of concurrent requests from multiple threads on the device

- **mundy::mesh::FieldViews**  
  Helpers for extracting mathematical views into STK field types, both on host and device.  

- **mundy::mesh::Classes**  
  A utility for mapping a deconstructed class hierarchy (e.g. rod segments, rod ends, contacts) onto STK parts and fields in a consistent IO-compliant way with enforced invariants.

- **mundy::mesh::Aggregate**  
  Wraps STK fields in their underlying view type, enabling clean code such as  
  ```cpp
  center_accessor(e) += dt * velocity_accessor(e);
  ```
  and aggregation of these accessors to avoid function bloat.

* **mundy::mesh::LinkData / mundy::mesh::LinkCOOData / mundy::mesh::LinkCSRData**
  Kokkos-compatible dynamic connectivity constructs (ghosting contrasts that are themselves entities).
  * Supports dynamically updating COO connectivity
  * Allows on-device sparse updates to CSR structures
  * Follows dual-view-like semantics aligned with STK’s NGP design
  * Automatic synchronization tracking during mesh modification cycles

* **mundy::mesh::NgpFieldBLAS**
  Reimplementation of STK’s field BLAS routines with unified host/device syntax.

* **mundy::mesh::NgpAccessorExpr**
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

---
