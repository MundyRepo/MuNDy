![MuNDy](doc/images/mundy_banner_v2_t.png)

# MuNDy: Multibody Nonlocal Dynamics

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
[![Backend-Kokkos](https://img.shields.io/badge/backend-Kokkos-1E88E5.svg)](https://kokkos.org/)
[![Mesh-STK](https://img.shields.io/badge/mesh-Trilinos%2FSTK-4CAF50.svg)](https://trilinos.github.io/stk.html)
[![CDash](https://img.shields.io/badge/CDash-Mundy-2EA44F.svg)](https://my.cdash.org/index.php?project=Mundy)

MuNDy is a C++ infrastructure for building scalable, biologically grounded microscale multibody dynamics software.

MuNDy supports models with evolving mechanical and relational structure: heterogeneous rigid and flexible bodies; constraints, motors, and contacts; growth, division, death, and bonds that form, break, and reorganize; and long-range interactions mediated through a shared medium. Rather than a monolithic simulator, MuNDy builds upon Trilinos/STK's runtime-extensible entity/part/field data model to provide reusable abstractions and data structures for this problem class. It is designed for research software developers building domain-specific applications across deployment scales, from laptops and workstations to multi-GPU clusters.

> [!IMPORTANT]
> **Project status (7/17/2026):**  
> MuNDy is under active development and rapidly approaching our first formal release. Most of MuNDy has finished its "polishing" phase and has a mostly stable API + dev-docs in the form of Primers and Doxygen generated documentation. The remaining work is focused on the multibody dynamics solver, Python interface, and mini-apps.

---

## A note from the author
I'm pleased to see that you're interested in using my library. I treasure Mundy, so if you have questions, suggested improvements, or issues (even something as simple as installation problems), please don't hesitate to make a GitHub issue or pull request.

I built Mundy to be academic infrastructure for the type of wheels I saw repeatedly reinvented across the active/soft/granular matter community and their biological applications. 

Mundy is a different library depending on where in the package hierarchy you enter. 
  - At the level of MundyMesh and MundySearch, you'll find that Mundy _feels_ like an entity component system library (offering many of the same features as packages like Entt) except with the added flexibility to organize entities into a class hierarchy with data inheritance and polymorphic-like features. This is the "domain model" we inherited from STK, whereby standard class-based polymorphism is replaced with a deconstructed class hierarchy built at runtime. There is no Spheres class--you create a particle topology collection of entities and endow them with a center location and radius. Any such collection that meets the requirements to act like spheres may be used in methods meant to act on spheres.
  - At the level of MundyCore (utils, math, geom, mech), Mundy _feels_ like a header-only utilities library. 
    - Utils should _feel_ like STD, containing Kokkos-compatible versions of some core STD structures (tuple, variant, reference_wrapper) while also adding carefully-designed, custom, STD-like structures. Features like aggregate--a tagged bag of types, which should feel like boost::hana::Map. Or host_ptr, which should feel like std::shared_ptr but with a device compatible copy constructor. Or MUNDY_THROW_ASSERT/REQUIRE, which should feel like assert/throw but with more expressive errors and device compatibility.
    - Math should _feel_ like a Kokkos-based, constexpr-compatible Eigen (only the staticly sized small matrix/vector/quaternion parts but with the same performance and auto-diff support) plus a few extra features. Features like accessor-based views and backend-independent numerics (CG, L-BFGS, convex QP/LCP) that can both drive kernels and be called on device. The same LCP that solves contact problems for N bodies can be used to solve tessellated body-to-body signed separation distance on the device.
    - Geom should _feel_ like a tiny, auto-differentiable geometry library (nothing close to the breadth of David Eberly's GeometricTools) with view-semantics compatible primitives, MD-like periodicity support, and the requirement that Graphics-community shortcuts/approximations that sacrifice accuracy for speed are not allowed (regular square roots, stable signed distance calculations even under double precision, no hard-coded magic numbers, etc).

What I am working on now is MundyMbody. This is the final piece: a proper multibody dynamics solver with composable mechanical elements, constraints, and contact laws. This is the part of Mundy that will _feel_ like a multibody dynamics library (like Chrono or JADE) but with full device compatibility, custom user-defined types/fields + operators that act on them. Indeed, Mbody is really just a refactored version of Project Chrono's Core with their polymorphic class hierarchy inverted and deconstructed according to Mundy's philosophy. This is the part of Mundy that will be most useful to non-specialists and will be my target for a Python API. For now, I anticipate the Mbody campaign will wrap up late Fall 2026.

(The proper capitalization is MuNDy, but I prefer name-case Mundy. Use either as you see fit.)

Best of luck,
-Bryce Palmer

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
git clone --depth=2 --branch=releases/v1.2.0 https://github.com/spack/spack.git ./spack
. ./spack/share/spack/setup-env.sh

# Add MuNDy's Spack package repository | Tells Spack where to find the MuNDy's dependency recipes
spack repo add ./dep/mundy_spack_repo
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
     384 text files.
     345 unique files.                                          
      52 files ignored.

github.com/AlDanial/cloc v 1.96  T=1.85 s (186.5 files/s, 49656.7 lines/s)
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

### [**MundyUtils**](https://mundyrepo.github.io/MuNDy/MundyUtils.html): Centralized reusable utilities

Centralized, Kokkos-friendly building blocks for type-level plumbing, error handling, and device-aware data management.
- **MUNDY_THROW_ASSERT() / MUNDY_THROW_REQUIRE()**  
  Kokkos-compatible throw/assert helpers with diagnostics and detailed error context 
  * On-device: abort  |  On-host: throw
  * Can be called within constexpr contexts

- **mundy::tuple / mundy::variant**  
  Reduced, Kokkos-compatible analogs of `std::tuple` / `std::variant` for default-constructible types.  
  * mundy::tuple is NTTP-compatible and `constexpr`-friendly  

- **mundy::reference_wrapper**  
  Kokkos-compatible analog of `std::reference_wrapper` for non-const references.  

- **mundy::storage**
  Unified means to maybe own or maybe view a value using a simple normalized storage policy.

- **mundy::aggregate**  
  Compile-time extensible “tagged bag of types” (conceptually similar to `boost::hana::map`).  
  * Kokkos-compatible  
  * `constexpr` and NTTP-compatible  

- **mundy::StringLiteral**  
  `constexpr` string literals that are NTTP-compatible.  
  * Supports `constexpr` concatenation

- **mundy::StringSink**  
  A stream-like (`<<`) utility for constructing compile-time and runtime strings.  
  * Kokkos-compatible/constexpr-friendly
  * Automatically produces compile-time strings when possible

- **mundy::NgpPool / mundy::NgpView**  
  Dual-view abstractions that follow MuNDy’s sync semantics plus a dual-view push/pop pool.  
  * Designed to integrate cleanly with Kokkos’ NGP (Next Generation Parallelism) model  

---

### [**MundyMath**](https://mundyrepo.github.io/MuNDy/MundyMath.html): `constexpr`, inline mathematics

Small, composable math utilities with view semantics that integrate naturally into Kokkos-based code.
- **mundy::Matrix / mundy::Vector / mundy::Quaternion**  
  Kokkos-compatible, `constexpr` inline linear algebra for small matrix/vector sizes.  
  * NTTP-compatible  
  * View semantics for arbitrary accessors  

- **mundy::minimize(...)** 
  Kokkos-compatible analog of dlib’s `minimize` (L-BFGS) with no dynamic memory allocation.  
  * Callable inside kernels or from host drivers  

- **mundy::convex**  
  Linear complementarity problem (LCP) and constrained convex quadratic programming (QP) solver.
  * Kokkos-compatible  
  * Can run inside a kernel or orchestrate kernel launches
  * Supports Mixed LCP/QP problems with equality and inequality constraints

- **mundy::Hilbert / mundy::zmort**  
  Domain decomposition helpers for Hilbert space-filling curves and Z-morton ordering.  
  * Useful for load balancing and locality-aware particle/domain layout  

---

### [**MundyGeom**](https://mundyrepo.github.io/MuNDy/MundyGeom.html): Geometric primitives and utilities

Foundational geometric abstractions for multibody dynamics and contact mechanics.
- **Primitives**  
  mundy::Point, mundy::Line, mundy::LineSegment, mundy::VSegment, mundy::Ring, mundy::Sphere, mundy::Spherocylinder, mundy::SpherocylinderSegment, mundy::Circle3D, mundy::Ellipsoid.

- **mundy::distance**  
  Utilities for computing:
  * Euclidean separation distances  
  * Shared-normal signed separation distances between primitives  

- **mundy::compute_aabb / mundy::compute_bounding_radius**  
  Helpers to compute axis-aligned bounding boxes (AABB) and bounding radii for each primitive.

- **mundy::transform / mundy::randomize**  
  Utilities for:
  * Translation  
  * Rotation  
  * Randomization of primitive configurations  

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
  * Selector expressions: `"(partA | partB) & !partC"`  
  * Topology: `"HEX_8"`  
  * Rank: `"ELEM_RANK"`  
  to their corresponding STK objects.

- **mundy::mesh::EntityDeclaration / mundy::mesh::FieldDeclaration mundy::mesh::ComponentDeclaration / mundy::mesh::PartDeclaration / mundy::mesh::ClassDeclaration**  
  Helper functions that streamline common STK mesh setup tasks, such as declaring fields and parts with the correct properties and parallel consistency.

- **mundy::mesh::NgpModRequests**
  A ticket-based framework for staging mesh modification requests from the device and processing them on the host.
  * Requesting new entities (with known or generated Ids)
  * Requesting new connectivity (e.g. element-to-node relations) involving existing or future entities
  * Requesting deletion of existing entities or connectivity
  * Safe and efficient in the face of concurrent requests from multiple threads on the device

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

- **mundy::mesh::LinkData / mundy::mesh::LinkCOOData / mundy::mesh::LinkCSRData**
  Kokkos-compatible dynamic connectivity constructs (ghosting contrasts that are themselves entities).
  * Supports dynamically updating COO connectivity
  * Allows on-device sparse updates to CSR structures
  * Follows dual-view-like semantics aligned with STK’s NGP design
  * Automatic synchronization tracking during mesh modification cycles

- **mundy::mesh::NgpFieldBLAS**
  Reimplementation of STK’s field BLAS routines with unified host/device syntax.

- **mundy::mesh::NgpAccessorExpr**
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

Neighbor-list construction and iteration over STK mesh entities, backed by ArborX BVH or STK distributed coarse search.

- **mundy::search::SearchInput / mundy::search::PeriodicSearchInput**  
  Binds a `stk::mesh::Selector` to a geometry component (AABB or OBB), encoding which class of entities are searched and how to access their geometry.

- **mundy::search::NeighborListBuilder**  
  The canonical builder for all neighbor lists, allowing for the specification of source/target inputs, broad/narrow-phase refinements, and rebuild policies.
  * mundy::search::ArborX1dNeighborList — single-rank ArborX BVH, compressed CSR storage; lower memory, suited for sparse neighbor lists
  * mundy::search::ArborX2dNeighborList — single-rank ArborX BVH, dense 2D per-target storage; suited for GPU pair-parallel dispatch
  * mundy::search::STKSearchNeighborList — STK MORTON_LBVH, MPI-distributed
  * Periodic variants (PeriodicArborX1dNeighborList, PeriodicArborX2dNeighborList, PeriodicSTKSearchNeighborList) carry per-object image shifts alongside stored pairs

- **Excluders**  
  Build-time predicates that reject candidate target/source pairs before they enter the stored list: mundy::search::ExcludeSelfInteraction, mundy::search::ExcludeSymmetricDuplicates, mundy::search::ExcludeConnectedEntities, mundy::search::ExcludeNonIntersectingOBBs.

- **Rebuilders**  
  Policies that determine when a cached neighbor list should be rebuilt based on changes in the underlying mesh or geometry: mundy::search::RebuildOnEntityChange, mundy::search::RebuildOnAABBDisplacement, mundy::search::RebuildOnOBBDisplacement, mundy::search::AlwaysRebuild, mundy::search::NeverRebuild.

- **Iteration and reduction**  
  Parallel iteration/reduction over stored pairs or per-target neighbor rows; works with all concrete list types: mundy::search::for_each_neighbor_pair, mundy::search::for_each_target_with_neighbors, mundy::search::for_each_neighbor_pair_reduce, mundy::search::for_each_target_with_neighbors_reduce.

---

### Standalone Offshoots

Independent projects that emerged from MuNDy’s infrastructure and are usable on their own.
- **[OpenRAND](https://github.com/msu-sparta/OpenRAND)**
  Performance-portable, counter-based random number generation that is stupid simple to use.
  * Designed to easily fit in GPU registers
  * Makes reproducibility in spite of varied parallelism possible
  * Now used by HOOMD-Blue
  
- **[alsous_gigantism_2025](https://github.com/flatironinstitute/alsous_gigantism_2025)**
  A discrete elastic rod model implemented using MuNDy. 

- **[mundy_mock_app](https://github.com/MundyRepo/mundy_mock_app)** /
  **[mundy_mock_app_tribits](https://github.com/MundyRepo/mundy_mock_app_tribits)**
  Helper applications for bootstrapping MuNDy-based codes:
  * CMake-based or TriBITS+CMake templates
  * Intended as starting points for internal and external applications that depend on MuNDy

---

## Release Roadmap

Planned steps toward the first public release (estimated summer 2026):
- Python API mirroring accessor expressions
- Tutorial + Example applications

---
