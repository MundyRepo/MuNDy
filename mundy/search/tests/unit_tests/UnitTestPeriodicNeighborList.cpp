// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
//
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

/// \file UnitTestPeriodicNeighborList.cpp
/// \brief Unit tests for PeriodicArborX1dNeighborList and PeriodicArborX2dNeighborList.
///
/// Test structure:
///   Group 1 — Construction and access: default construction, copy/move semantics.
///   Group 2 — Deterministic 2-particle periodic test:
///     Two particles near opposite sides of a periodic domain; their periodic images
///     overlap through the boundary, producing a known pair with a known image shift.
///     The exact relative shift value and source ordinal are checked for both 1D and 2D.
///   Group 3 — Random N² single-image validation:
///     N particles, each with a single image (zero shift). The periodic infrastructure
///     must produce the same pairs as the non-periodic oracle. This validates that owner
///     mapping, shift storage, and the build pipeline are correct end-to-end.
///   Group 4 — Debug-only bound-check tests (NDEBUG guard).
///
/// Periodic geometry used in Group 2:
///   Domain length L = 10, particle radius r = 1.5.
///
///   Owner A (node 1): x = 0.5, y = z = 0.
///   Owner B (node 2): x = 9.5, y = z = 0.
///
///   Images for A (2 images):
///     A_im0: shift = (0, 0, 0), center = (0.5, 0, 0), box x = [-1, 2]
///     A_im1: shift = (-10, 0, 0), center = (-9.5, 0, 0), box x = [-11, -8]
///
///   Images for B (2 images):
///     B_im0: shift = (0, 0, 0), center = (9.5, 0, 0), box x = [8, 11]
///     B_im1: shift = (-10, 0, 0), center = (-0.5, 0, 0), box x = [-2, 1]
///
///   Intersections after ExcludeSelfInteraction (source-image shift ≠ target-image shift):
///     A_im0 (target owner A, shift 0) queries B_im1 (source owner B, shift -10):
///       x = [-1,2] ∩ [-2,1] = [-1,1] — overlap.  Relative shift = -10 - 0 = -10.
///     B_im1 (target owner B, shift -10) queries A_im0 (source owner A, shift 0):
///       x = [-2,1] ∩ [-1,2] = [-1,1] — overlap.  Relative shift = 0 - (-10) = +10.
///
///   Expected result:
///     Target owner A (idx 0): 1 neighbor → source owner B (idx 1), shift (-10, 0, 0).
///     Target owner B (idx 1): 1 neighbor → source owner A (idx 0), shift (+10, 0, 0).

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// External
#include <gtest/gtest.h>

// C++ core
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// ArborX
#include <ArborX.hpp>

// STK mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>      // for stk::mesh::Field
#include <stk_mesh/base/FieldBase.hpp>  // for stk::mesh::field_data
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for declare_field, put_field_on_mesh
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy search
#include <mundy_search/ArborX1dNeighborList.hpp>
#include <mundy_search/ArborX2dNeighborList.hpp>
#include <mundy_search/Excluder.hpp>
#include <mundy_search/ForEach.hpp>
#include <mundy_search/NeighborListBuilder.hpp>
#include <mundy_search/Neighbors.hpp>
#include <mundy_search/STKSearchNeighborList.hpp>  // for PeriodicSTKSearchNeighborList
#include <mundy_search/SearchInput.hpp>            // for mundy::search::PeriodicSearchInput (component input)
#include <mundy_search/impl/ArborXSearchBoxes.hpp>

// Mundy math / geom / mesh / utils
#include <mundy_geom/periodicity.hpp>     // for mundy::OrthorhombicMetric, AXIS_XYZ
#include <mundy_math/Vector3.hpp>         // for mundy::Vector3
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::AABBFieldComponent
#include <mundy_utils/rng.hpp>            // for mundy::make_philox

namespace mundy {
namespace search {
namespace {

// =============================================================================
// Compile-time concept checks
// =============================================================================

static_assert(NeighborListType<PeriodicArborX1dNeighborList<Kokkos::HostSpace>>,
              "PeriodicArborX1dNeighborList<HostSpace> must satisfy NeighborListType.");
static_assert(NeighborListType<PeriodicArborX2dNeighborList<Kokkos::HostSpace>>,
              "PeriodicArborX2dNeighborList<HostSpace> must satisfy NeighborListType.");

// =============================================================================
// Type aliases
// =============================================================================

using TestMemSpace = Kokkos::HostSpace;
using TestExecSpace = Kokkos::DefaultHostExecutionSpace;
using ImageShiftType = mundy::Vector3<float>;

// Self-contained host oracle-reference boxes: per-image (box, owner ordinal, shift) plus the dense owner entities.
// Deliberately independent of the production search-box wrappers so the gold standard never depends on the
// implementation it validates; it just exposes the per-image accessors the N^2 oracle loops read.
struct PeriodicBoxes {
  std::vector<ArborX::Box> image_boxes;
  std::vector<size_t> image_owner_indices;
  std::vector<ImageShiftType> image_shift_vectors;
  std::vector<stk::mesh::Entity> dense_owner_entities;

  size_t size() const {
    return image_boxes.size();
  }
  size_t num_owners() const {
    return dense_owner_entities.size();
  }
  const ArborX::Box& box(size_t image_index) const {
    return image_boxes[image_index];
  }
  size_t owner_index(size_t image_index) const {
    return image_owner_indices[image_index];
  }
  ImageShiftType image_shift(size_t image_index) const {
    return image_shift_vectors[image_index];
  }
  stk::mesh::Entity owner_entity(size_t owner_index) const {
    return dense_owner_entities[owner_index];
  }
};

using PerList1d = PeriodicArborX1dNeighborList<TestMemSpace, float>;
using PerList2d = PeriodicArborX2dNeighborList<TestMemSpace, float>;
using PerSTKList = PeriodicSTKSearchNeighborList<TestMemSpace, float>;

// Component-backed periodic input consumed by the builds (which generate the images on device).
using PerComponent = mundy::mesh::AABBFieldComponent<double>;
using PerMetric = mundy::OrthorhombicMetric<mundy::AXIS_XYZ, float>;
using PerInput = mundy::search::PeriodicSearchInput<PerComponent, PerMetric>;

// Reuse non-periodic search boxes for the single-image N² test.

// =============================================================================
// Box helpers
// =============================================================================

ArborX::Box make_arborx_box(float cx, float cy, float cz, float hx, float hy, float hz) {
  return ArborX::Box{ArborX::Point{cx - hx, cy - hy, cz - hz}, ArborX::Point{cx + hx, cy + hy, cz + hz}};
}

ArborX::Box make_arborx_box(float cx, float cy, float cz, float h) {
  return make_arborx_box(cx, cy, cz, h, h, h);
}

// Build a mundy::AABB<float> centered at (cx,cy,cz) with isotropic half-extent r.
AABB<float> make_aabb(float cx, float cy, float cz, float r) {
  return AABB<float>{Point<float>{cx - r, cy - r, cz - r}, Point<float>{cx + r, cy + r, cz + r}};
}

// Oracle image shift for a lattice-neighbour offset nb (host mirror of the device generator): the exact displacement
// from the owner's original reference point to its wrapped-then-shifted reference point, via the same impl helpers the
// generator uses (integer-image recovery in fractional space + from_fractional).  Its formula is pinned independently
// by TargetAndSourceBoxHelperShifts; here it lets the N^2 oracle reproduce the build's shifts exactly.
template <typename Metric>
ImageShiftType oracle_image_shift(const AABB<float>& aabb, const mundy::Vector3<int>& nb, const Metric& metric) {
  const auto k = mundy::image_index(reference_point(aabb), metric);
  const auto shift = mundy::lattice_displacement(mundy::Vector3<int>{nb[0] - k[0], nb[1] - k[1], nb[2] - k[2]}, metric);
  return ImageShiftType{shift[0], shift[1], shift[2]};
}

bool boxes_overlap(const ArborX::Box& a, const ArborX::Box& b) {
  for (int d = 0; d < 3; ++d) {
    if (a.maxCorner()[d] < b.minCorner()[d]) return false;
    if (b.maxCorner()[d] < a.minCorner()[d]) return false;
  }
  return true;
}

// =============================================================================
// STK mesh helpers
// =============================================================================

std::pair<std::shared_ptr<stk::mesh::MetaData>, std::unique_ptr<stk::mesh::BulkData>> make_node_mesh(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta_ptr = builder.create_meta_data();
  meta_ptr->use_simple_fields();
  auto bulk_ptr = builder.create(meta_ptr);
  meta_ptr->commit();
  bulk_ptr->modification_begin();
  for (int id = 1; id <= num_nodes; ++id) {
    bulk_ptr->declare_node(id);
  }
  bulk_ptr->modification_end();
  return {std::move(meta_ptr), std::move(bulk_ptr)};
}

// A node-only mesh with a 6-scalar `aabb` node field declared (pre-commit) for component-backed periodic inputs.
struct NodeMeshWithAABB {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Field<double>* aabb_field = nullptr;
};

NodeMeshWithAABB make_node_mesh_with_aabb(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta_ptr = builder.create_meta_data();
  meta_ptr->use_simple_fields();
  auto& aabb_field = meta_ptr->declare_field<double>(stk::topology::NODE_RANK, "aabb_periodic_field");
  stk::mesh::put_field_on_mesh(aabb_field, meta_ptr->universal_part(), 6, nullptr);
  auto bulk_ptr = builder.create(meta_ptr);
  meta_ptr->commit();
  bulk_ptr->modification_begin();
  for (int id = 1; id <= num_nodes; ++id) bulk_ptr->declare_node(id);
  bulk_ptr->modification_end();
  return {std::move(meta_ptr), std::move(bulk_ptr), &aabb_field};
}

// Distributed node-only mesh with an `aabb` node field: node id `i+1` (global id `i`) is owned by rank `i % nprocs`
// (round-robin), declared only on its owning rank. Remote nodes are absent locally until the neighbor-list build
// ghosts them — the multi-rank path the cross-rank validation exercises.
NodeMeshWithAABB make_distributed_node_mesh_with_aabb(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta_ptr = builder.create_meta_data();
  meta_ptr->use_simple_fields();
  auto& aabb_field = meta_ptr->declare_field<double>(stk::topology::NODE_RANK, "aabb_periodic_field");
  stk::mesh::put_field_on_mesh(aabb_field, meta_ptr->universal_part(), 6, nullptr);
  auto bulk_ptr = builder.create(meta_ptr);
  meta_ptr->commit();
  const int my_rank = bulk_ptr->parallel_rank();
  const int nprocs = bulk_ptr->parallel_size();
  bulk_ptr->modification_begin();
  for (int id = 1; id <= num_nodes; ++id) {
    if ((id - 1) % nprocs == my_rank) bulk_ptr->declare_node(id);
  }
  bulk_ptr->modification_end();
  return {std::move(meta_ptr), std::move(bulk_ptr), &aabb_field};
}

// Write one node's AABB (center ± half-extents) into the field (layout: min xyz 0-2, max xyz 3-5).
void store_aabb(stk::mesh::Field<double>& field, stk::mesh::Entity node, float cx, float cy, float cz, float hx,
                float hy, float hz) {
  double* d = stk::mesh::field_data(field, node);
  d[0] = cx - hx;
  d[1] = cy - hy;
  d[2] = cz - hz;
  d[3] = cx + hx;
  d[4] = cy + hy;
  d[5] = cz + hz;
}

// Populate the `aabb` field (center = position, given half-extents) and return a component `PeriodicSearchInput`
// over `selector` with an orthorhombic metric of cell length L.  The build generates the periodic images.
PerInput make_periodic_input(stk::mesh::Field<double>& aabb_field, const stk::mesh::Selector& selector,
                             const std::vector<stk::mesh::Entity>& nodes,
                             const std::vector<std::array<float, 3>>& positions, float hx, float hy, float hz,
                             float L) {
  for (size_t i = 0; i < nodes.size(); ++i)
    store_aabb(aabb_field, nodes[i], positions[i][0], positions[i][1], positions[i][2], hx, hy, hz);
  PerComponent component(aabb_field);
  component.modify_on_host();
  return PerInput{selector, component, PerMetric{mundy::Vector3<float>{L, L, L}}};
}

// Isotropic convenience overload (half-extent r in all axes).
PerInput make_periodic_input(stk::mesh::Field<double>& aabb_field, const stk::mesh::Selector& selector,
                             const std::vector<stk::mesh::Entity>& nodes,
                             const std::vector<std::array<float, 3>>& positions, float r, float L) {
  return make_periodic_input(aabb_field, selector, nodes, positions, r, r, r, L);
}

// =============================================================================
// Oracle-reference box construction helper
// =============================================================================

// Build the local PeriodicBoxes oracle from pre-populated host arrays.
// owner_entities[i] = owner STK entity for owner ordinal i.
// image_boxes[k]    = ArborX::Box for image k.
// owner_indices[k]  = owner ordinal for image k.
// image_shifts[k]   = shift vector applied to the owner to produce image k.
PeriodicBoxes make_periodic_boxes(const stk::mesh::Selector& /*selector*/,
                                  const std::vector<stk::mesh::Entity>& owner_entities,
                                  const std::vector<ArborX::Box>& image_boxes, const std::vector<size_t>& owner_indices,
                                  const std::vector<ImageShiftType>& image_shifts) {
  EXPECT_EQ(image_boxes.size(), owner_indices.size());
  EXPECT_EQ(image_boxes.size(), image_shifts.size());
  return PeriodicBoxes{image_boxes, owner_indices, image_shifts, owner_entities};
}

// =============================================================================
// Physically-correct periodic image construction
//
// The periodic neighbor list separates targets from sources and builds their
// image boxes differently.
//
// TARGETS — each owner contributes exactly ONE image:
//   The original object may have its reference point anywhere.  It is wrapped
//   rigidly into the primary cell [0,L)^3.  The stored shift is:
//       target_shift = wrapped_position - original_position
//   (the vector that translates the original into its wrapped image).
//
// SOURCES — each owner contributes up to 27 images (one per lattice neighbor):
//   The original source is first wrapped into [0,L)^3, giving wrapped_position.
//   Then candidate images are stamped at wrapped_position + n*L for n ∈ {-1,0,+1}^3.
//   The stored shift for image n is:
//       source_shift = (wrapped_position + n*L) - original_position
//
//   AABB pruning (optional): compute the union AABB of all target images with
//   compute_target_bbox(), then pass it to make_source_periodic_boxes().  Any
//   candidate source image that does not intersect the target AABB cannot produce
//   a neighbor with ANY target, so it is discarded.
//
// The list reports the RELATIVE SHIFT for each pair:
//       relative_shift = source_shift - target_shift
//
// Using 1 target image and (up to) 27 source images guarantees that for any given
// (target_owner, source_owner, relative_shift) triple there is at most one
// (target_image, source_image) pair that produces it, eliminating duplicates.
// Using 27 images for BOTH targets and sources breaks this guarantee: two
// distinct image pairs can share the same relative shift, flooding the list
// with duplicates for the same logical interaction.
// =============================================================================

// Build target boxes: 1 image per owner at the wrapped position.
// shift = wrapped_position - original_position.
static PeriodicBoxes make_target_periodic_boxes(const stk::mesh::Selector& selector,
                                                const std::vector<stk::mesh::Entity>& nodes,
                                                const std::vector<std::array<float, 3>>& positions, float L, float r) {
  const OrthorhombicMetric<AXIS_XYZ, float> metric{Vector3<float>{L, L, L}};
  const size_t N = positions.size();
  std::vector<ArborX::Box> img_boxes(N);
  std::vector<size_t> img_owners(N);
  std::vector<ImageShiftType> img_shifts(N);
  for (size_t i = 0; i < N; ++i) {
    // Mirror the device generator: rigidly wrap the owner AABB (reference point = its min corner) into [0,L)^3.
    const AABB<float> aabb = make_aabb(positions[i][0], positions[i][1], positions[i][2], r);
    const ImageShiftType shift = oracle_image_shift(aabb, mundy::Vector3<int>{0, 0, 0}, metric);
    img_boxes[i] = impl::pack_arborx_box(translate(aabb, shift));
    img_owners[i] = i;
    img_shifts[i] = shift;
  }
  return make_periodic_boxes(selector, nodes, img_boxes, img_owners, img_shifts);
}

// Compute the union AABB of all image boxes in a PeriodicBoxes set.
// Pass the result to make_source_periodic_boxes() to prune candidate images
// that cannot possibly overlap any target.
static ArborX::Box compute_target_bbox(const PeriodicBoxes& target_boxes) {
  MUNDY_THROW_REQUIRE(target_boxes.size() > 0, std::invalid_argument,
                      "compute_target_bbox: target_boxes must be non-empty.");
  float lo[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
  float hi[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max()};
  for (size_t k = 0; k < target_boxes.size(); ++k) {
    const auto& b = target_boxes.box(k);
    for (int d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], b.minCorner()[d]);
      hi[d] = std::max(hi[d], b.maxCorner()[d]);
    }
  }
  return ArborX::Box{ArborX::Point{lo[0], lo[1], lo[2]}, ArborX::Point{hi[0], hi[1], hi[2]}};
}

// Build source boxes: up to 27 images per owner (wrapped + n*L for n∈{-1,0,1}^3).
// If target_bbox is provided, candidate images that don't intersect it are skipped —
// they cannot produce a pair with any target and including them wastes memory and
// ArborX build time.  Omit target_bbox (or pass std::nullopt) to get all 27 images.
// shift for surviving image n = (wrapped_position + n*L) - original_position.
static PeriodicBoxes make_source_periodic_boxes(const stk::mesh::Selector& selector,
                                                const std::vector<stk::mesh::Entity>& nodes,
                                                const std::vector<std::array<float, 3>>& positions, float L, float r,
                                                std::optional<ArborX::Box> target_bbox = std::nullopt) {
  const OrthorhombicMetric<AXIS_XYZ, float> metric{Vector3<float>{L, L, L}};
  std::vector<ArborX::Box> img_boxes;
  std::vector<size_t> img_owners;
  std::vector<ImageShiftType> img_shifts;
  img_boxes.reserve(positions.size() * 27);
  img_owners.reserve(positions.size() * 27);
  img_shifts.reserve(positions.size() * 27);
  const mundy::Vector3<int> nb_bound = impl::lattice_neighbour_bound(metric);
  for (size_t i = 0; i < positions.size(); ++i) {
    // Mirror the device generator: per periodic lattice neighbour, the owner box translated by the exact image shift.
    const AABB<float> aabb = make_aabb(positions[i][0], positions[i][1], positions[i][2], r);
    for (int nx = -nb_bound[0]; nx <= nb_bound[0]; ++nx) {
      for (int ny = -nb_bound[1]; ny <= nb_bound[1]; ++ny) {
        for (int nz = -nb_bound[2]; nz <= nb_bound[2]; ++nz) {
          const ImageShiftType shift = oracle_image_shift(aabb, mundy::Vector3<int>{nx, ny, nz}, metric);
          const ArborX::Box image_box = impl::pack_arborx_box(translate(aabb, shift));
          if (target_bbox.has_value() && !boxes_overlap(image_box, *target_bbox)) continue;
          img_boxes.push_back(image_box);
          img_owners.push_back(i);
          img_shifts.push_back(shift);
        }
      }
    }
  }
  return make_periodic_boxes(selector, nodes, img_boxes, img_owners, img_shifts);
}

// =============================================================================
// Oracle helpers for periodic tests
// =============================================================================

// Pair type for periodic validation: (target_owner, source_owner, relative_shift).
struct PeriodicPair {
  size_t target_owner;
  size_t source_owner;
  ImageShiftType relative_shift;

  bool operator<(const PeriodicPair& o) const {
    if (target_owner != o.target_owner) return target_owner < o.target_owner;
    if (source_owner != o.source_owner) return source_owner < o.source_owner;
    for (int d = 0; d < 3; ++d) {
      if (relative_shift[d] != o.relative_shift[d]) return relative_shift[d] < o.relative_shift[d];
    }
    return false;
  }

  bool operator==(const PeriodicPair& o) const {
    return target_owner == o.target_owner && source_owner == o.source_owner &&
           relative_shift[0] == o.relative_shift[0] && relative_shift[1] == o.relative_shift[1] &&
           relative_shift[2] == o.relative_shift[2];
  }
};

// The lists expose only the two per-object shifts; a consumer derives the pairwise relative shift itself.
template <typename ListType>
ImageShiftType list_relative_shift(const ListType& list, size_t t, size_t k) {
  return list.source_image_shift(t, k) - list.target_image_shift(t);
}

// Collect periodic pairs from a built list via direct iteration.
template <typename ListType>
std::vector<PeriodicPair> collect_periodic_pairs(const ListType& list) {
  std::vector<PeriodicPair> result;
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      result.push_back({t, list.source_index(t, k), list_relative_shift(list, t, k)});
    }
  }
  return result;
}

// For the single-image N² test: (target_ordinal, source_ordinal) set with self-excluded.
std::set<std::pair<size_t, size_t>> oracle_pairs_no_self(const std::vector<ArborX::Box>& boxes) {
  std::set<std::pair<size_t, size_t>> pairs;
  for (size_t t = 0; t < boxes.size(); ++t) {
    for (size_t s = 0; s < boxes.size(); ++s) {
      if (t != s && boxes_overlap(boxes[t], boxes[s])) pairs.insert({t, s});
    }
  }
  return pairs;
}

// Collect (target_owner, source_owner) pairs from a periodic list, ignoring shifts.
template <typename ListType>
std::set<std::pair<size_t, size_t>> collect_index_pairs(const ListType& list) {
  std::set<std::pair<size_t, size_t>> result;
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      result.insert({t, list.source_index(t, k)});
    }
  }
  return result;
}

bool shifts_approx_eq(const ImageShiftType& a, const ImageShiftType& b) {
  return std::abs(a[0] - b[0]) < 1e-4f && std::abs(a[1] - b[1]) < 1e-4f && std::abs(a[2] - b[2]) < 1e-4f;
}

// All valid image shifts for an owner at position p (half-extent r) under the metric — i.e. the owner's possible
// absolute image shifts (original reference point → each lattice-neighbour image).
template <typename Metric>
std::vector<ImageShiftType> owner_image_shifts(const std::array<float, 3>& p, float r, const Metric& metric) {
  const AABB<float> aabb = make_aabb(p[0], p[1], p[2], r);
  const mundy::Vector3<int> nb = impl::lattice_neighbour_bound(metric);
  std::vector<ImageShiftType> out;
  for (int nx = -nb[0]; nx <= nb[0]; ++nx)
    for (int ny = -nb[1]; ny <= nb[1]; ++ny)
      for (int nz = -nb[2]; nz <= nb[2]; ++nz)
        out.push_back(oracle_image_shift(aabb, mundy::Vector3<int>{nx, ny, nz}, metric));
  return out;
}

// Validate the per-owner target_image_shift interface of a built periodic list:
//   (1) target_image_shift(t) equals the oracle target shift (original → wrapped reference point) for owner t, and
//   (2) for every stored pair, target_image_shift(t) + (source_image_shift(t,k) − target_image_shift(t)) is a valid
//       image shift of the neighbour source owner — so a kernel can place the source's contacting image.
// positions[i] is owner ordinal i's position (the build's dense ordinal follows entity-id order for these meshes).
template <typename ListType, typename Metric>
void check_target_image_shifts(const ListType& list, const std::vector<std::array<float, 3>>& positions, float r,
                               const Metric& metric) {
  for (size_t t = 0; t < list.num_targets(); ++t) {
    const AABB<float> taabb = make_aabb(positions[t][0], positions[t][1], positions[t][2], r);
    const ImageShiftType expected = oracle_image_shift(taabb, mundy::Vector3<int>{0, 0, 0}, metric);
    const ImageShiftType got = list.target_image_shift(t);
    EXPECT_TRUE(shifts_approx_eq(got, expected)) << "target_image_shift mismatch for owner " << t;
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t s = list.source_index(t, k);
      const ImageShiftType rel = list_relative_shift(list, t, k);
      const ImageShiftType abs_src{got[0] + rel[0], got[1] + rel[1], got[2] + rel[2]};
      const auto candidates = owner_image_shifts(positions[s], r, metric);
      bool found = false;
      for (const auto& c : candidates)
        if (shifts_approx_eq(abs_src, c)) {
          found = true;
          break;
        }
      EXPECT_TRUE(found) << "target_image_shift+relative is not a valid source image for pair (" << t << "," << s
                         << ")";
    }
  }
}

// =============================================================================
// Build helpers
// =============================================================================

// One component `PeriodicSearchInput` drives the build: it generates the target images (1 per owner) and the
// source images (≤27 per owner, pruned) internally, so the same input is used for both sides.
PerList1d build_per1d_list(const stk::mesh::BulkData& bulk, const PerInput& input) {
  return make_neighbor_list_builder<PerList1d>()
      .exec_space(TestExecSpace{})
      .target_input(input)
      .source_input(input)
      .broad_phase(ExcludeSelfInteraction{})
      .build(bulk);
}

PerList2d build_per2d_list(const stk::mesh::BulkData& bulk, const PerInput& input) {
  return make_neighbor_list_builder<PerList2d>()
      .exec_space(TestExecSpace{})
      .target_input(input)
      .source_input(input)
      .broad_phase(ExcludeSelfInteraction{})
      .build(bulk);
}

PerSTKList build_per_stk_list(const stk::mesh::BulkData& bulk, const PerInput& input) {
  return make_neighbor_list_builder<PerSTKList>()
      .exec_space(TestExecSpace{})
      .target_input(input)
      .source_input(input)
      .broad_phase(ExcludeSelfInteraction{})
      .build(bulk);
}

// =============================================================================
// Shared iteration-count checks
// =============================================================================

template <typename ListType>
void check_for_each_pair_count(const ListType& list) {
  Kokkos::View<size_t*, TestMemSpace> count("fep_count", 1);
  Kokkos::deep_copy(count, size_t(0));
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const NeighborPair<ListType>&) { Kokkos::atomic_inc(&count(0)); });
  Kokkos::fence();
  EXPECT_EQ(count(0), list.size()) << "for_each_neighbor_pair count does not match list.size().";
}

template <typename ListType>
void check_for_each_target_count(const ListType& list) {
  Kokkos::View<size_t*, TestMemSpace> count("fet_count", 1);
  Kokkos::deep_copy(count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<ListType>&) { Kokkos::atomic_inc(&count(0)); });
  Kokkos::fence();
  EXPECT_EQ(count(0), list.num_targets()) << "for_each_target_with_neighbors count does not match list.num_targets().";
}

// =============================================================================
// Test fixture — 2-particle periodic geometry
// =============================================================================

// Two STK nodes (A and B) near opposite sides of a domain of length 10.
// Each owner has two images, as described in the file-level comment.
// See the comment block at the top of this file for the full overlap analysis.
class PeriodicFixture : public ::testing::Test {
 protected:
  // Particle geometry.
  static constexpr float kRadius = 1.5f;
  static constexpr float kL = 10.0f;  // periodic domain length
  static constexpr float kXa = 0.5f;  // owner A x-position
  static constexpr float kXb = 9.5f;  // owner B x-position

  // Image box half-widths.
  static constexpr float kHx = kRadius;  // 1.5
  static constexpr float kHyz = 1.0f;    // deliberately narrower in y,z to isolate x-axis overlaps

  void SetUp() override {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
      GTEST_SKIP();
    }
    auto mesh = make_node_mesh_with_aabb(2);
    meta_ = std::move(mesh.meta);
    bulk_ = std::move(mesh.bulk);
    aabb_field_ = mesh.aabb_field;

    node_a_ = bulk_->get_entity(stk::topology::NODE_RANK, 1);
    node_b_ = bulk_->get_entity(stk::topology::NODE_RANK, 2);
    ASSERT_TRUE(bulk_->is_valid(node_a_));
    ASSERT_TRUE(bulk_->is_valid(node_b_));

    selector_ = meta_->universal_part();

    // Owner A at x=0.5, owner B at x=9.5, in a periodic box of length L=10.  The boxes are anisotropic (wide in x,
    // half = kHx = 1.5; narrow in y,z, half = kHyz = 1.0) to isolate x-axis boundary overlaps.  The build wraps each
    // owner's center into [0,L) for its single target image and stamps 27 source images; the resulting wrap-around
    // overlap (A's left source image with B's target, and vice-versa) is verified by `verify_periodic_2particle`.
    const std::vector<std::array<float, 3>> positions = {{kXa, 0.0f, 0.0f}, {kXb, 0.0f, 0.0f}};
    periodic_input_ = make_periodic_input(*aabb_field_, selector_, {node_a_, node_b_}, positions, kHx, kHyz, kHyz, kL);
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Entity node_a_;
  stk::mesh::Entity node_b_;
  stk::mesh::Selector selector_;
  stk::mesh::Field<double>* aabb_field_ = nullptr;
  PerInput periodic_input_;
};

// =============================================================================
// Group 0 — Test-infrastructure self-checks
//
// Verifies that PeriodicPair comparison, boxes_overlap, and
// collect_periodic_pairs are all correct before trusting them in the oracle.
// =============================================================================

// Exercises equality, all-field ordering (target > source > shift xyz), set
// deduplication, and set lookup — including the shifts that appear in real
// boundary-crossing tests (face ±10, edge ±10 in two axes, corner ±10 all).
TEST(TestInfra, PeriodicPairComparisonAndSetBehavior) {
  // Equality: same fields → equal, reflexive, not less-than.
  PeriodicPair p{1, 2, ImageShiftType{-10.0f, 0.0f, 0.0f}};
  EXPECT_TRUE(p == p);
  EXPECT_FALSE(p < p);
  EXPECT_TRUE(p == (PeriodicPair{1, 2, ImageShiftType{-10.0f, 0.0f, 0.0f}}));

  // Each field individually breaks equality.
  EXPECT_FALSE(p == (PeriodicPair{9, 2, ImageShiftType{-10.0f, 0.0f, 0.0f}}));    // target
  EXPECT_FALSE(p == (PeriodicPair{1, 9, ImageShiftType{-10.0f, 0.0f, 0.0f}}));    // source
  EXPECT_FALSE(p == (PeriodicPair{1, 2, ImageShiftType{10.0f, 0.0f, 0.0f}}));     // shift x
  EXPECT_FALSE(p == (PeriodicPair{1, 2, ImageShiftType{-10.0f, -10.0f, 0.0f}}));  // shift y
  EXPECT_FALSE(p == (PeriodicPair{1, 2, ImageShiftType{-10.0f, 0.0f, -10.0f}}));  // shift z

  // Ordering: target is primary key, source secondary, shift x/y/z tertiary.
  EXPECT_TRUE((PeriodicPair{0, 9, {0, 0, 0}}) < (PeriodicPair{1, 0, {0, 0, 0}}));     // target
  EXPECT_TRUE((PeriodicPair{2, 0, {0, 0, 0}}) < (PeriodicPair{2, 1, {0, 0, 0}}));     // source
  EXPECT_TRUE((PeriodicPair{1, 2, {-10, 0, 0}}) < (PeriodicPair{1, 2, {10, 0, 0}}));  // shift x
  EXPECT_TRUE((PeriodicPair{1, 2, {0, -10, 0}}) < (PeriodicPair{1, 2, {0, 10, 0}}));  // shift y
  EXPECT_TRUE((PeriodicPair{1, 2, {0, 0, -10}}) < (PeriodicPair{1, 2, {0, 0, 10}}));  // shift z

  // std::set deduplicates identical pairs and distinguishes distinct ones.
  std::set<PeriodicPair> s;
  s.insert({0, 1, {-10.0f, 0.0f, 0.0f}});      // face x
  s.insert({0, 1, {0.0f, -10.0f, 0.0f}});      // face y
  s.insert({0, 1, {0.0f, 0.0f, -10.0f}});      // face z
  s.insert({0, 1, {-10.0f, -10.0f, -10.0f}});  // corner xyz
  s.insert({1, 0, {10.0f, 0.0f, 0.0f}});       // reverse face x
  EXPECT_EQ(s.size(), 5u);
  // Re-inserting the same entries must not grow the set.
  s.insert({0, 1, {-10.0f, 0.0f, 0.0f}});
  s.insert({0, 1, {-10.0f, -10.0f, -10.0f}});
  EXPECT_EQ(s.size(), 5u);
  // Lookup: the corner pair must be found; a near-miss (wrong sign on one axis) must not.
  EXPECT_EQ(s.count({0, 1, {-10.0f, -10.0f, -10.0f}}), 1u);
  EXPECT_EQ(s.count({0, 1, {10.0f, -10.0f, -10.0f}}), 0u);
}

// Covers overlap, disjoint, single-axis separation, touching edge, and gap.
TEST(TestInfra, BoxesOverlapCases) {
  auto a = make_arborx_box(0.0f, 0.0f, 0.0f, 1.0f);                          // [-1,1]^3
  EXPECT_TRUE(boxes_overlap(a, make_arborx_box(1.0f, 0.0f, 0.0f, 1.0f)));    // overlap in x
  EXPECT_FALSE(boxes_overlap(a, make_arborx_box(5.0f, 0.0f, 0.0f, 1.0f)));   // gap in x
  EXPECT_FALSE(boxes_overlap(a, make_arborx_box(0.0f, 3.0f, 0.0f, 1.0f)));   // gap in y only
  EXPECT_TRUE(boxes_overlap(a, make_arborx_box(2.0f, 0.0f, 0.0f, 1.0f)));    // touch at x=1 (not strictly separated)
  EXPECT_FALSE(boxes_overlap(a, make_arborx_box(2.01f, 0.0f, 0.0f, 1.0f)));  // epsilon gap
}

// Verifies collect_periodic_pairs returns the right pairs for the known 2-particle geometry: two owners near
// opposite x-faces whose images overlap only through the periodic boundary.  The build generates the images from
// the component input; the two expected wrap-around pairs (with ±L relative shifts) must be reported.
TEST(TestInfra, CollectPeriodicPairsMatchesKnown2ParticleGeometry) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_node_mesh_with_aabb(2);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  auto node_a = mesh.bulk->get_entity(stk::topology::NODE_RANK, 1);
  auto node_b = mesh.bulk->get_entity(stk::topology::NODE_RANK, 2);
  constexpr float kL = 10.0f, kXa = 0.5f, kXb = 9.5f, kHx = 1.5f, kHyz = 1.0f;
  const std::vector<std::array<float, 3>> positions = {{kXa, 0, 0}, {kXb, 0, 0}};
  auto input = make_periodic_input(*mesh.aabb_field, selector, {node_a, node_b}, positions, kHx, kHyz, kHyz, kL);

  auto pairs = collect_periodic_pairs(build_per1d_list(*mesh.bulk, input));
  ASSERT_EQ(pairs.size(), 2u);
  std::set<PeriodicPair> ps(pairs.begin(), pairs.end());
  EXPECT_EQ(ps.count({0, 1, {-kL, 0, 0}}), 1u);
  EXPECT_EQ(ps.count({1, 0, {kL, 0, 0}}), 1u);
}

// Verifies the per-owner target_image_shift interface with an out-of-cell target (shift ≠ 0): owner 0 sits one cell
// past +x and wraps back by -L, owner 1 is already in-cell (shift 0); both wrap near x≈1 and are neighbors. Checks the
// reported per-owner shift and the identity target_image_shift + (source−target relative) = a valid source image shift.
template <typename ListType>
void run_target_image_shift_test() {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_node_mesh_with_aabb(2);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  auto node_a = mesh.bulk->get_entity(stk::topology::NODE_RANK, 1);
  auto node_b = mesh.bulk->get_entity(stk::topology::NODE_RANK, 2);
  constexpr float kL = 10.0f, kR = 0.5f;
  const std::vector<std::array<float, 3>> positions = {{kL + 1.0f, 5.0f, 5.0f}, {1.4f, 5.0f, 5.0f}};
  const PerMetric metric{mundy::Vector3<float>{kL, kL, kL}};
  auto input = make_periodic_input(*mesh.aabb_field, selector, {node_a, node_b}, positions, kR, kL);

  auto list = mundy::search::make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(input)
                  .source_input(input)
                  .broad_phase(ExcludeSelfInteraction{})
                  .build(*mesh.bulk);

  ASSERT_EQ(list.num_targets(), 2u);
  EXPECT_TRUE(shifts_approx_eq(list.target_image_shift(0), ImageShiftType{-kL, 0.0f, 0.0f}))
      << "owner 0 (out of cell, +x) should report wrap shift -L.";
  EXPECT_TRUE(shifts_approx_eq(list.target_image_shift(1), ImageShiftType{0.0f, 0.0f, 0.0f}))
      << "owner 1 (in cell) should report zero wrap shift.";
  check_target_image_shifts(list, positions, kR, metric);
}

TEST(PeriodicArborX1dNeighborList, TargetImageShiftReportedPerOwner) {
  run_target_image_shift_test<PerList1d>();
}

TEST(PeriodicArborX2dNeighborList, TargetImageShiftReportedPerOwner) {
  run_target_image_shift_test<PerList2d>();
}

// Verifies make_target_periodic_boxes and make_source_periodic_boxes produce
// the correct image counts and shifts for three representative positions:
//   A: inside domain (x=2, y=3, z=4)  — wraps trivially, target shift = (0,0,0)
//   B: outside in +x (x=L+2)          — wraps to x=2,   target shift = (-L,0,0)
//   C: outside in -x (x=-1)           — wraps to x=L-1, target shift = (+L,0,0)
TEST(TestInfra, TargetAndSourceBoxHelperShifts) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr float L = 10.0f, r = 0.5f;
  auto [meta, bulk] = make_node_mesh(3);
  const stk::mesh::Selector sel = meta->universal_part();
  auto n0 = bulk->get_entity(stk::topology::NODE_RANK, 1);
  auto n1 = bulk->get_entity(stk::topology::NODE_RANK, 2);
  auto n2 = bulk->get_entity(stk::topology::NODE_RANK, 3);

  const std::vector<std::array<float, 3>> pos = {
      {2.0f, 3.0f, 4.0f},      // A: inside
      {L + 2.0f, 3.0f, 4.0f},  // B: +x outside
      {-1.0f, 3.0f, 4.0f},     // C: -x outside
  };

  // --- Targets: 1 image per owner, shift = wrapped - original ---
  {
    auto tboxes = make_target_periodic_boxes(sel, {n0, n1, n2}, pos, L, r);
    ASSERT_EQ(tboxes.size(), 3u);

    // A inside: metric.wrap({2,3,4}) = {2,3,4}, shift = (0,0,0)
    EXPECT_FLOAT_EQ(tboxes.image_shift(0)[0], 0.0f);
    // B outside +x: metric.wrap({12,3,4}) = {2,3,4}, shift = 2-12 = -L
    EXPECT_FLOAT_EQ(tboxes.image_shift(1)[0], -L);
    EXPECT_FLOAT_EQ(tboxes.image_shift(1)[1], 0.0f);
    EXPECT_FLOAT_EQ(tboxes.image_shift(1)[2], 0.0f);
    // C outside -x: metric.wrap({-1,3,4}) = {9,3,4}, shift = 9-(-1) = +L
    EXPECT_FLOAT_EQ(tboxes.image_shift(2)[0], L);

    // All target image centers must lie in [0, L).
    for (size_t k = 0; k < tboxes.size(); ++k) {
      const auto& b = tboxes.box(k);
      const float cx = (b.minCorner()[0] + b.maxCorner()[0]) * 0.5f;
      EXPECT_GE(cx, 0.0f) << "target image " << k << " center outside [0,L)";
      EXPECT_LT(cx, L) << "target image " << k << " center outside [0,L)";
    }
  }

  // --- Sources: 27 images per owner, shift = (wrapped + n*L) - original ---
  {
    auto sboxes = make_source_periodic_boxes(sel, {n0, n1, n2}, pos, L, r);
    ASSERT_EQ(sboxes.size(), 3u * 27u);

    // For owner A (images 0..26): wrapped = (2,3,4), original = (2,3,4).
    // All shifts are n*L exactly.  Find the n=(0,0,0) image (shift=(0,0,0)).
    bool found_zero_shift_A = false;
    for (size_t k = 0; k < 27; ++k) {
      EXPECT_EQ(sboxes.owner_index(k), 0u);
      const auto s = sboxes.image_shift(k);
      if (s[0] == 0.0f && s[1] == 0.0f && s[2] == 0.0f) found_zero_shift_A = true;
      // Each shift component must be a multiple of L.
      EXPECT_FLOAT_EQ(std::fmod(std::abs(s[0]), L), 0.0f) << "k=" << k;
      EXPECT_FLOAT_EQ(std::fmod(std::abs(s[1]), L), 0.0f) << "k=" << k;
      EXPECT_FLOAT_EQ(std::fmod(std::abs(s[2]), L), 0.0f) << "k=" << k;
    }
    EXPECT_TRUE(found_zero_shift_A);

    // For owner B (images 27..53): original = (L+2, 3, 4), wrapped = (2, 3, 4).
    // The n=(1,0,0) image sits at (12, 3, 4), so shift = 12 - 12 = 0 in x.
    bool found_zero_shift_B = false;
    for (size_t k = 27; k < 54; ++k) {
      EXPECT_EQ(sboxes.owner_index(k), 1u);
      const auto s = sboxes.image_shift(k);
      if (s[0] == 0.0f && s[1] == 0.0f && s[2] == 0.0f) found_zero_shift_B = true;
    }
    EXPECT_TRUE(found_zero_shift_B);
  }
}

// Verifies that compute_target_bbox + make_source_periodic_boxes pruning:
//   (a) produces strictly fewer images than the full 27-per-owner layout, and
//   (b) produces exactly the same neighbor pairs as the N² oracle built from the
//       FULL (unpruned) 27-image source set.
TEST(TestInfra, SourceBoxAabbPruningReducesImagesAndPreservesCorrectness) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 20;
  constexpr size_t kSeed = 77;
  constexpr float L = 10.0f, r = 1.5f;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, L), rng.uniform<float>(0.0f, L), rng.uniform<float>(0.0f, L)};
  }
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector sel = mesh.meta->universal_part();
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i)
    nodes[i] = mesh.bulk->get_entity(stk::topology::NODE_RANK, static_cast<stk::mesh::EntityId>(i + 1));

  const auto tboxes = make_target_periodic_boxes(sel, nodes, positions, L, r);
  const auto sboxes_full = make_source_periodic_boxes(sel, nodes, positions, L, r);
  const auto sboxes_pruned = make_source_periodic_boxes(sel, nodes, positions, L, r, compute_target_bbox(tboxes));

  // (a) Pruning reduces image count vs the naive 27*N ceiling.
  EXPECT_LT(sboxes_pruned.size(), static_cast<size_t>(kN) * 27u)
      << "Pruned source count=" << sboxes_pruned.size() << " should be < 27*N=" << kN * 27;
  EXPECT_LT(sboxes_pruned.size(), sboxes_full.size())
      << "Pruned count should be strictly less than full count=" << sboxes_full.size();

  // (b) Oracle from FULL (unpruned) source boxes — independent of compute_target_bbox.
  std::set<PeriodicPair> oracle;
  for (size_t ti = 0; ti < tboxes.size(); ++ti) {
    const size_t t_own = tboxes.owner_index(ti);
    for (size_t si = 0; si < sboxes_full.size(); ++si) {
      const size_t s_own = sboxes_full.owner_index(si);
      if (t_own == s_own) continue;
      if (!boxes_overlap(tboxes.box(ti), sboxes_full.box(si))) continue;
      oracle.insert({t_own, s_own, sboxes_full.image_shift(si) - tboxes.image_shift(ti)});
    }
  }

  // List built via the component input: the build generates and prunes the source images on device; the result
  // must match the full-image oracle above (pruning preserves correctness).
  auto input = make_periodic_input(*mesh.aabb_field, sel, nodes, positions, r, L);
  auto list = build_per1d_list(*mesh.bulk, input);
  const auto actual_vec = collect_periodic_pairs(list);
  const std::set<PeriodicPair> actual(actual_vec.begin(), actual_vec.end());

  EXPECT_EQ(actual_vec.size(), actual.size()) << "List has duplicate entries after pruning.";
  EXPECT_EQ(actual.size(), oracle.size()) << "list=" << actual.size() << " oracle=" << oracle.size();
  for (const auto& p : oracle)
    EXPECT_TRUE(actual.count(p)) << "MISSING t=" << p.target_owner << " s=" << p.source_owner << " shift=("
                                 << p.relative_shift[0] << "," << p.relative_shift[1] << "," << p.relative_shift[2]
                                 << ")";
  for (const auto& p : actual)
    EXPECT_TRUE(oracle.count(p)) << "SPURIOUS t=" << p.target_owner << " s=" << p.source_owner << " shift=("
                                 << p.relative_shift[0] << "," << p.relative_shift[1] << "," << p.relative_shift[2]
                                 << ")";
}

// =============================================================================
// Group 1 — Construction and access
// =============================================================================

TEST(PeriodicArborX1dNeighborList, DefaultConstruct) {
  PerList1d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(), 0u);
}

TEST(PeriodicArborX2dNeighborList, DefaultConstruct) {
  PerList2d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(), 0u);
}

TEST_F(PeriodicFixture, CopyMove_1d) {
  auto original = build_per1d_list(*bulk_, periodic_input_);
  const size_t nt = original.num_targets();
  const size_t ns = original.num_sources();
  const size_t size = original.size();

  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(), size);

  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(), size);

  PerList1d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.size(), size);

  PerList1d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.size(), size);
}

TEST_F(PeriodicFixture, CopyMove_2d) {
  auto original = build_per2d_list(*bulk_, periodic_input_);
  const size_t nt = original.num_targets();
  const size_t ns = original.num_sources();
  const size_t size = original.size();

  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(), size);

  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(), size);

  PerList2d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.size(), size);

  PerList2d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.size(), size);
}

// =============================================================================
// Group 2 — Deterministic 2-particle periodic test
// =============================================================================

// Shared body verifying the deterministic periodic-boundary result.
// For the 2-owner layout, the expected pairs are:
//   (owner 0=A, owner 1=B, relative shift (-10,0,0))
//   (owner 1=B, owner 0=A, relative shift (+10,0,0))
template <typename ListType>
void verify_periodic_2particle(const ListType& list, stk::mesh::Entity node_a, stk::mesh::Entity node_b, float kL) {
  ASSERT_EQ(list.num_targets(), 2u);
  ASSERT_EQ(list.num_sources(), 2u);
  EXPECT_EQ(list.size(), 2u);

  EXPECT_EQ(list.num_neighbors(0), 1u);  // owner A sees owner B
  EXPECT_EQ(list.num_neighbors(1), 1u);  // owner B sees owner A

  // Entity accessors.
  EXPECT_EQ(list.target_entity(0), node_a);
  EXPECT_EQ(list.target_entity(1), node_b);

  // --- Owner A (idx 0) ---
  // Source owner ordinal must be 1 (owner B), entity must be node_b.
  EXPECT_EQ(list.source_index(0, 0), size_t(1));
  EXPECT_EQ(list.get_neighbor(0, 0), node_b);
  // Relative shift: source B_im1 shift (-10) minus target A_im0 shift (0) = -10.
  const ImageShiftType shift_A = list_relative_shift(list, 0, 0);
  EXPECT_FLOAT_EQ(shift_A[0], -kL);
  EXPECT_FLOAT_EQ(shift_A[1], 0.0f);
  EXPECT_FLOAT_EQ(shift_A[2], 0.0f);

  // --- Owner B (idx 1) ---
  // Source owner ordinal must be 0 (owner A), entity must be node_a.
  EXPECT_EQ(list.source_index(1, 0), size_t(0));
  EXPECT_EQ(list.get_neighbor(1, 0), node_a);
  // Relative shift: source A_im0 shift (0) minus target B_im1 shift (-10) = +10.
  const ImageShiftType shift_B = list_relative_shift(list, 1, 0);
  EXPECT_FLOAT_EQ(shift_B[0], kL);
  EXPECT_FLOAT_EQ(shift_B[1], 0.0f);
  EXPECT_FLOAT_EQ(shift_B[2], 0.0f);

  // Iteration helpers.
  check_for_each_pair_count(list);
  check_for_each_target_count(list);
}

TEST_F(PeriodicFixture, Deterministic_1d) {
  auto list = build_per1d_list(*bulk_, periodic_input_);
  verify_periodic_2particle(list, node_a_, node_b_, kL);
}

TEST_F(PeriodicFixture, Deterministic_2d) {
  auto list = build_per2d_list(*bulk_, periodic_input_);
  verify_periodic_2particle(list, node_a_, node_b_, kL);
}

// =============================================================================
// Group 3 — Random N² single-image validation
// =============================================================================

// With a periodic cell far larger than the populated domain, the build behaves non-periodically: each owner's
// center already lies in [0,L) (no wrap), and its n≠0 lattice images sit ~L away and are pruned, leaving one
// effective image per owner with zero shift.  The resulting pairs must match the non-periodic oracle, and every
// stored relative image shift must be zero.
template <typename ListType, typename BuildFn>
void run_single_image_n2_validation(BuildFn build_fn, NodeMeshWithAABB& mesh, const stk::mesh::Selector& selector,
                                    int num_nodes) {
  constexpr size_t kSeed = 137;
  constexpr float kDomainSize = 10.0f;
  constexpr float kRadius = 0.9f;
  constexpr float kHugeL = 1000.0f;  // ≫ domain → no wrap, distant images pruned → non-periodic behavior

  std::vector<stk::mesh::Entity> nodes(num_nodes);
  std::vector<std::array<float, 3>> positions(num_nodes);
  std::vector<ArborX::Box> boxes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i] = mesh.bulk->get_entity(stk::topology::NODE_RANK, i + 1);
    ASSERT_TRUE(mesh.bulk->is_valid(nodes[i])) << "Node " << (i + 1) << " not found.";
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kDomainSize), rng.uniform<float>(0.0f, kDomainSize),
                    rng.uniform<float>(0.0f, kDomainSize)};
    boxes[i] = make_arborx_box(positions[i][0], positions[i][1], positions[i][2], kRadius);
  }

  auto input = make_periodic_input(*mesh.aabb_field, selector, nodes, positions, kRadius, kHugeL);
  ListType list = build_fn(*mesh.bulk, input);

  // The index-pair set must match the non-periodic oracle.
  const auto expected = oracle_pairs_no_self(boxes);
  const auto actual = collect_index_pairs(list);
  EXPECT_EQ(actual, expected) << "Large-cell periodic list does not match non-periodic oracle.";

  // All relative image shifts must be (0,0,0): the only surviving image is the n=0 (unshifted) one.
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const ImageShiftType shift = list_relative_shift(list, t, k);
      EXPECT_FLOAT_EQ(shift[0], 0.0f) << "Non-zero x-shift for pair (target=" << t << ", neighbor=" << k << ").";
      EXPECT_FLOAT_EQ(shift[1], 0.0f) << "Non-zero y-shift for pair (target=" << t << ", neighbor=" << k << ").";
      EXPECT_FLOAT_EQ(shift[2], 0.0f) << "Non-zero z-shift for pair (target=" << t << ", neighbor=" << k << ").";
    }
  }
}

TEST(PeriodicArborX1dNeighborList, SingleImageN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 50;
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_single_image_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per1d_list(b, in); }, mesh, selector, kN);
}

TEST(PeriodicArborX2dNeighborList, SingleImageN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 50;
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_single_image_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per2d_list(b, in); }, mesh, selector, kN);
}

// =============================================================================
// Group 5 — Full periodic N² validation with boundary-crossing pairs
//
// For N particles placed at known positions near every type of periodic boundary
// (face, edge, corner), we generate all 27 images per particle (shifts in
// {-L,0,+L}^3) and run a brute-force oracle that checks every (target-image,
// source-image) pair for overlap.  The oracle produces the set of
// (target_owner, source_owner, relative_shift) triples; the periodic neighbor
// list must produce exactly the same set.
//
// Unlike the existing SingleImageN2Validation test, this actually exercises the
// boundary-crossing logic: the oracle will find pairs that only intersect via a
// non-zero image shift, and the list must report the correct relative shift.
//
// Test particles (r=1.5, L=10) intentionally placed so that:
//   - Every face boundary (±x, ±y, ±z) has at least one crossing pair.
//   - Every edge boundary (±xy, ±xz, ±yz) has at least one crossing pair.
//   - The corner boundary (±xyz) has a crossing pair.
//   - Some interior particles produce direct (zero-shift) pairs.
// =============================================================================

static constexpr float kValidL = 10.0f;
static constexpr float kValidR = 1.5f;

// clang-format off
// Positions of test particles in [0, L).
// Each "near-X" particle is paired with a "near-L-X" particle whose AABB
// overlaps only through the periodic image, not directly.
static const std::vector<std::array<float, 3>> kBoundaryPositions = {
    {0.5f, 5.0f, 5.0f},  //  0: near x=0 face
    {9.5f, 5.0f, 5.0f},  //  1: near x=L face     (overlaps 0 through x-boundary)
    {5.0f, 0.5f, 5.0f},  //  2: near y=0 face
    {5.0f, 9.5f, 5.0f},  //  3: near y=L face     (overlaps 2 through y-boundary)
    {5.0f, 5.0f, 0.5f},  //  4: near z=0 face
    {5.0f, 5.0f, 9.5f},  //  5: near z=L face     (overlaps 4 through z-boundary)
    {0.5f, 0.5f, 5.0f},  //  6: near xy edge (x=0,y=0)
    {9.5f, 9.5f, 5.0f},  //  7: near xy edge (x=L,y=L)  (overlaps 6 through xy)
    {0.5f, 5.0f, 0.5f},  //  8: near xz edge (x=0,z=0)
    {9.5f, 5.0f, 9.5f},  //  9: near xz edge (x=L,z=L)  (overlaps 8 through xz)
    {5.0f, 0.5f, 0.5f},  // 10: near yz edge (y=0,z=0)
    {5.0f, 9.5f, 9.5f},  // 11: near yz edge (y=L,z=L)  (overlaps 10 through yz)
    {0.5f, 0.5f, 0.5f},  // 12: near (0,0,0) corner
    {9.5f, 9.5f, 9.5f},  // 13: near (L,L,L) corner     (overlaps 12 through xyz)
    {2.5f, 5.0f, 5.0f},  // 14: interior, direct overlap with 0
    {5.0f, 2.5f, 5.0f},  // 15: interior, direct overlap with 2
    {5.0f, 5.0f, 2.5f},  // 16: interior, direct overlap with 4
};
// clang-format on

// Run the full periodic N² oracle and compare against the built list.
//
// The list is built from a component `PeriodicSearchInput` (the build generates the target/source images on
// device, with source-image pruning).  The independent host oracle uses make_target_periodic_boxes (1 image/owner)
// and the full unpruned 27-image make_source_periodic_boxes, so it does not depend on the build's pruning: matching
// the two proves the device generation + pruning is correct, including every boundary-crossing relative shift.
template <typename ListType, typename BuildFn>
void run_full_periodic_n2_validation(BuildFn build_fn, NodeMeshWithAABB& mesh, const stk::mesh::Selector& selector,
                                     const std::vector<std::array<float, 3>>& positions, float L, float r) {
  const size_t N = positions.size();
  std::vector<stk::mesh::Entity> nodes(N);
  for (size_t i = 0; i < N; ++i) {
    nodes[i] = mesh.bulk->get_entity(stk::topology::NODE_RANK, static_cast<stk::mesh::EntityId>(i + 1));
    ASSERT_TRUE(mesh.bulk->is_valid(nodes[i])) << "node " << (i + 1) << " missing.";
  }

  // Independent host oracle (full, unpruned images).
  const auto tboxes = make_target_periodic_boxes(selector, nodes, positions, L, r);
  const auto sboxes = make_source_periodic_boxes(selector, nodes, positions, L, r);

  // List built via the component input (device image generation + pruning).
  auto input = make_periodic_input(*mesh.aabb_field, selector, nodes, positions, r, L);
  ListType list = build_fn(*mesh.bulk, input);

  // Brute-force oracle: iterate every (target-image, source-image) pair.
  std::set<PeriodicPair> oracle;
  for (size_t ti = 0; ti < tboxes.size(); ++ti) {
    const size_t t_own = tboxes.owner_index(ti);
    for (size_t si = 0; si < sboxes.size(); ++si) {
      const size_t s_own = sboxes.owner_index(si);
      if (t_own == s_own) continue;
      if (!boxes_overlap(tboxes.box(ti), sboxes.box(si))) continue;
      oracle.insert({t_own, s_own, sboxes.image_shift(si) - tboxes.image_shift(ti)});
    }
  }

  const std::vector<PeriodicPair> actual_vec = collect_periodic_pairs(list);
  const std::set<PeriodicPair> actual(actual_vec.begin(), actual_vec.end());

  EXPECT_EQ(actual_vec.size(), actual.size()) << "List contains duplicate (target, source, shift) entries.";
  EXPECT_EQ(actual.size(), oracle.size()) << "Pair count: list=" << actual.size() << " oracle=" << oracle.size();
  for (const auto& p : oracle)
    EXPECT_TRUE(actual.count(p)) << "MISSING: target=" << p.target_owner << " source=" << p.source_owner << " shift=("
                                 << p.relative_shift[0] << "," << p.relative_shift[1] << "," << p.relative_shift[2]
                                 << ")";
  for (const auto& p : actual)
    EXPECT_TRUE(oracle.count(p)) << "SPURIOUS: target=" << p.target_owner << " source=" << p.source_owner << " shift=("
                                 << p.relative_shift[0] << "," << p.relative_shift[1] << "," << p.relative_shift[2]
                                 << ")";

  check_for_each_pair_count(list);
  check_for_each_target_count(list);
}

TEST(PeriodicArborX1dNeighborList, BoundaryN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_node_mesh_with_aabb(static_cast<int>(kBoundaryPositions.size()));
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_full_periodic_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per1d_list(b, in); }, mesh, selector,
      kBoundaryPositions, kValidL, kValidR);
}

TEST(PeriodicArborX2dNeighborList, BoundaryN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_node_mesh_with_aabb(static_cast<int>(kBoundaryPositions.size()));
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_full_periodic_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per2d_list(b, in); }, mesh, selector,
      kBoundaryPositions, kValidL, kValidR);
}

// Random positions.
TEST(PeriodicArborX1dNeighborList, RandomFullPeriodicN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 40;
  constexpr size_t kSeed = 42;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kValidL), rng.uniform<float>(0.0f, kValidL),
                    rng.uniform<float>(0.0f, kValidL)};
  }
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_full_periodic_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per1d_list(b, in); }, mesh, selector, positions,
      kValidL, kValidR);
}

TEST(PeriodicArborX2dNeighborList, RandomFullPeriodicN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 40;
  constexpr size_t kSeed = 42;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kValidL), rng.uniform<float>(0.0f, kValidL),
                    rng.uniform<float>(0.0f, kValidL)};
  }
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_full_periodic_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per2d_list(b, in); }, mesh, selector, positions,
      kValidL, kValidR);
}

// ---- Periodic STK backend: reuse the same validations (single-rank; multi-rank periodic validation deferred). ----

TEST(PeriodicSTKSearchNeighborList, TargetImageShiftReportedPerOwner) {
  run_target_image_shift_test<PerSTKList>();
}

TEST(PeriodicSTKSearchNeighborList, SingleImageN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 50;
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_single_image_n2_validation<PerSTKList>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per_stk_list(b, in); }, mesh, selector, kN);
}

TEST(PeriodicSTKSearchNeighborList, BoundaryN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_node_mesh_with_aabb(static_cast<int>(kBoundaryPositions.size()));
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_full_periodic_n2_validation<PerSTKList>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per_stk_list(b, in); }, mesh, selector,
      kBoundaryPositions, kValidL, kValidR);
}

TEST(PeriodicSTKSearchNeighborList, RandomFullPeriodicN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 40;
  constexpr size_t kSeed = 42;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kValidL), rng.uniform<float>(0.0f, kValidL),
                    rng.uniform<float>(0.0f, kValidL)};
  }
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector selector = mesh.meta->universal_part();
  run_full_periodic_n2_validation<PerSTKList>(
      [](stk::mesh::BulkData& b, const PerInput& in) { return build_per_stk_list(b, in); }, mesh, selector, positions,
      kValidL, kValidR);
}

// =============================================================================
// Cross-rank validation of the periodic STK multi-rank build.
//
// The full point set is replicated on every rank; point i has global id i and is owned by rank i % nprocs, declared
// only on its owning rank. The build ghosts remote source owners. Each rank checks ONLY its owned target rows against
// the replicated N² gid-oracle (filtered to targets it owns), mapping the list's entities back to global ids — so no
// cross-rank gather is needed and the union over ranks is the full global oracle. Runs at any rank count (np=1 is the
// degenerate single-rank case; np=2,4 exercise ghosting + the global-bbox source prune).
// =============================================================================
void run_multirank_periodic_stk_validation(const std::vector<std::array<float, 3>>& positions, float L, float r) {
  const int N = static_cast<int>(positions.size());
  auto mesh = make_distributed_node_mesh_with_aabb(N);
  auto& bulk = *mesh.bulk;
  const int my_rank = bulk.parallel_rank();
  const int nprocs = bulk.parallel_size();
  const stk::mesh::Selector owned = mesh.meta->locally_owned_part();

  // AABBs on owned nodes only (center = position, isotropic half-extent r).
  for (int i = 0; i < N; ++i) {
    if (i % nprocs != my_rank) continue;
    auto node = bulk.get_entity(stk::topology::NODE_RANK, static_cast<stk::mesh::EntityId>(i + 1));
    ASSERT_TRUE(bulk.is_valid(node)) << "owned node " << (i + 1) << " missing on rank " << my_rank;
    store_aabb(*mesh.aabb_field, node, positions[i][0], positions[i][1], positions[i][2], r, r, r);
  }
  PerComponent component(*mesh.aabb_field);
  component.modify_on_host();
  PerInput input{owned, component, PerMetric{mundy::Vector3<float>{L, L, L}}};

  PerSTKList list = mundy::search::make_neighbor_list_builder<PerSTKList>()
                        .exec_space(TestExecSpace{})
                        .target_input(input)
                        .source_input(input)
                        .broad_phase(ExcludeSelfInteraction{})
                        .build(bulk);

  // Replicated gid-oracle (positions index == global id); keep only rows this rank owns. Dummy owner entities are
  // unused by the pair set. Same-owner pairs are skipped (a particle's own image is L away at these radii).
  std::vector<stk::mesh::Entity> dummy(N);
  const auto tboxes = make_target_periodic_boxes(owned, dummy, positions, L, r);
  const auto sboxes = make_source_periodic_boxes(owned, dummy, positions, L, r);
  std::set<PeriodicPair> expected;
  for (size_t ti = 0; ti < tboxes.size(); ++ti) {
    const size_t tgid = tboxes.owner_index(ti);
    if (static_cast<int>(tgid % static_cast<size_t>(nprocs)) != my_rank) continue;
    for (size_t si = 0; si < sboxes.size(); ++si) {
      const size_t sgid = sboxes.owner_index(si);
      if (tgid == sgid) continue;
      if (!boxes_overlap(tboxes.box(ti), sboxes.box(si))) continue;
      expected.insert({tgid, sgid, sboxes.image_shift(si) - tboxes.image_shift(ti)});
    }
  }

  // The list's rows for this rank's owned targets, mapped to global ids (valid for ghosted sources too).
  std::vector<PeriodicPair> actual_vec;
  for (size_t t = 0; t < list.num_targets(); ++t) {
    const size_t tgid = static_cast<size_t>(bulk.identifier(list.target_entity(t))) - 1;
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t sgid = static_cast<size_t>(bulk.identifier(list.get_neighbor(t, k))) - 1;
      actual_vec.push_back({tgid, sgid, list_relative_shift(list, t, k)});
    }
  }
  const std::set<PeriodicPair> actual(actual_vec.begin(), actual_vec.end());

  EXPECT_EQ(actual_vec.size(), actual.size()) << "rank " << my_rank << ": duplicate (target, source, shift) entries.";
  EXPECT_EQ(actual.size(), expected.size())
      << "rank " << my_rank << ": list=" << actual.size() << " oracle=" << expected.size();
  for (const auto& p : expected)
    EXPECT_TRUE(actual.count(p)) << "rank " << my_rank << " MISSING: target=" << p.target_owner
                                 << " source=" << p.source_owner;
  for (const auto& p : actual)
    EXPECT_TRUE(expected.count(p)) << "rank " << my_rank << " SPURIOUS: target=" << p.target_owner
                                   << " source=" << p.source_owner;
}

TEST(PeriodicSTKSearchNeighborList, MultiRankBoundaryN2Validation) {
  run_multirank_periodic_stk_validation(kBoundaryPositions, kValidL, kValidR);
}

TEST(PeriodicSTKSearchNeighborList, MultiRankRandomN2Validation) {
  constexpr int kN = 60;
  constexpr size_t kSeed = 7;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kValidL), rng.uniform<float>(0.0f, kValidL),
                    rng.uniform<float>(0.0f, kValidL)};
  }
  run_multirank_periodic_stk_validation(positions, kValidL, kValidR);
}

// =============================================================================
// Group 4 — Debug-only bound-check tests
// =============================================================================

#ifndef NDEBUG

TEST_F(PeriodicFixture, OutOfBounds_1d) {
  auto list = build_per1d_list(*bulk_, periodic_input_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
  EXPECT_THROW(list.source_image_shift(0, list.num_neighbors(0)), std::out_of_range);
}

TEST_F(PeriodicFixture, OutOfBounds_2d) {
  auto list = build_per2d_list(*bulk_, periodic_input_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
  EXPECT_THROW(list.source_image_shift(0, list.num_neighbors(0)), std::out_of_range);
}

#endif  // NDEBUG

}  // namespace
}  // namespace search
}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX
