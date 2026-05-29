![MuNDy](doc/images/mundy_banner_v2_t.png)

# MuNDy: Multibody Nonlocal Dynamics

MuNDy is a C++ framework for high-performance simulation of **multibody nonlocal dynamics** on modern CPU and GPU architectures.

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Backend-Kokkos](https://img.shields.io/badge/backend-Kokkos-1E88E5.svg)
![Mesh-STK](https://img.shields.io/badge/mesh-Trilinos%2FSTK-4CAF50.svg)

> [!IMPORTANT]
> **Project status (3/18/2026):**  
> MuNDy is under active development. We have chosen to make development public as we move toward a first formal release targeted for **Late June 2026**.

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
  - [Standalone Offshoots](#standalone-offshoots)
- [Release Roadmap](#release-roadmap)

---

## Documentation

Check our our [Wiki](https://github.com/MundyRepo/MuNDy/wiki) for user-facing documentation, design notes, and tutorials. The Wiki is a living document that we will continue to expand and refine as the library matures. You can also generate the local Doxygen documentation from the source tree with:

```bash
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
- **Simulation layers** configurable, so applications can opt into only what they need (mesh, mbody)

### Code Statistics (via cloc)
```text
cloc-1.96.pl --exclude-dir=TriBITS,ci,doc,scrap ./MuNDy
     384 text files.
     345 unique files.                                          
      52 files ignored.

github.com/AlDanial/cloc v 1.96  T=1.71 s (201.6 files/s, 53673.7 lines/s)
------------------------------------------------------------------
Language            files        blank        comment         code
------------------------------------------------------------------
C/C++ Header          142         7071          12382        30540
C++                    81         4823           4428        22397
CMake                  88          620           2046         2824
Markdown                8          832              0         2220
Python                  5          119            161          522
Bourne Shell           15           98             91          329
Text                    1           25              0          172
JSON                    2           10              0          137
YAML                    2            0              0            6
CSV                     1            0              0            1
------------------------------------------------------------------
SUM:                  345        13598          19108        59148
------------------------------------------------------------------
```

---

## Subpackages

### [**MundyUtils**](https://github.com/MundyRepo/MuNDy/wiki/1.-MundyUtils): Centralized reusable utilities

Doxygen directory reference: \ref mundy/core/utils "MundyUtils".

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

### [**MundyMath**](https://github.com/MundyRepo/MuNDy/wiki/2.-MundyMath): `constexpr`, inline mathematics

Doxygen directory reference: \ref mundy/core/math "MundyMath".

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

### [**MundyGeom**](https://github.com/MundyRepo/MuNDy/wiki/3.-MundyGeom): Geometric primitives and utilities

Doxygen directory reference: \ref mundy/core/geom "MundyGeom".

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

### [**MundyMech**](https://github.com/MundyRepo/MuNDy/wiki/4.-MundyMech): Mechanical primitives and utilities (under construction)

Doxygen directory reference: \ref mundy/core/mech "MundyMech".

Mechanical elements and force laws for building multibody models.

- **Primitives**  
  mundy::mech::BallJoint, mundy::mech::HookeanSpring, mundy::mech::FeneSpring, mundy::mech::TorsionalSpring.

Further mechanical models and integration hooks will be added as the library matures.

---

### [**MundyMesh**](https://github.com/MundyRepo/MuNDy/wiki/4.-MundyMesh): MuNDy’s extension to Trilinos/STK

Doxygen directory reference: \ref mundy/mesh "MundyMesh".

Helpers and abstractions for integrating MuNDy with Trilinos/STK meshes and fields.
- **mundy::mesh::string_to_selector / mundy::mesh::string_to_topology / mundy::mesh::string_to_rank**  
  Map string descriptions like:
  - Selector expressions: `"(partA | partB) & !partC"`  
  - Topology: `"HEX_8"`  
  - Rank: `"ELEM_RANK"`  
  to their corresponding STK objects.

- **mundy::mesh::DeclareEntitiesHelper / mundy::mesh::FieldDeclarationHelper / mundy::mesh::PartDeclarationHelper / mundy::mesh::ClassDeclarationHelper**  
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
