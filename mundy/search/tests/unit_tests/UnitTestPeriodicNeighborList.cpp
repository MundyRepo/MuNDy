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
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
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
#include <mundy_search/impl/ArborXSearchBoxes.hpp>

// Mundy math / geom / utils
#include <mundy_geom/periodicity.hpp>  // for mundy::OrthorhombicMetric, AXIS_XYZ
#include <mundy_math/Vector3.hpp>      // for mundy::Vector3
#include <mundy_utils/rng.hpp>         // for mundy::make_philox

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
using PeriodicBoxes = impl::PeriodicArborXSearchBoxesT<TestMemSpace, float>;
using PerList1d = PeriodicArborX1dNeighborList<TestMemSpace, float>;
using PerList2d = PeriodicArborX2dNeighborList<TestMemSpace, float>;

// Reuse non-periodic search boxes for the single-image N² test.
using SearchBoxes = impl::ArborXSearchBoxesT<TestMemSpace>;

// =============================================================================
// Box helpers
// =============================================================================

ArborX::Box make_arborx_box(float cx, float cy, float cz, float hx, float hy, float hz) {
  return ArborX::Box{ArborX::Point{cx - hx, cy - hy, cz - hz}, ArborX::Point{cx + hx, cy + hy, cz + hz}};
}

ArborX::Box make_arborx_box(float cx, float cy, float cz, float h) {
  return make_arborx_box(cx, cy, cz, h, h, h);
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

// =============================================================================
// PeriodicArborXSearchBoxesT construction helper
// =============================================================================

// Build a PeriodicArborXSearchBoxesT<HostSpace, float> from pre-populated host arrays.
// owner_entities[i] = owner STK entity for owner ordinal i.
// image_boxes[k]    = ArborX::Box for image k.
// owner_indices[k]  = owner ordinal for image k.
// image_shifts[k]   = shift vector applied to the owner to produce image k.
PeriodicBoxes make_periodic_boxes(const stk::mesh::Selector& selector,
                                  const std::vector<stk::mesh::Entity>& owner_entities,
                                  const std::vector<ArborX::Box>& image_boxes, const std::vector<size_t>& owner_indices,
                                  const std::vector<ImageShiftType>& image_shifts) {
  const size_t num_owners = owner_entities.size();
  const size_t num_images = image_boxes.size();
  EXPECT_EQ(num_images, owner_indices.size());
  EXPECT_EQ(num_images, image_shifts.size());

  Kokkos::View<ArborX::Box*, TestMemSpace> boxes("per_boxes", num_images);
  Kokkos::View<stk::mesh::Entity*, TestMemSpace> owners("per_owners", num_owners);
  Kokkos::View<size_t*, TestMemSpace> oi("per_oi", num_images);
  Kokkos::View<ImageShiftType*, TestMemSpace> shifts("per_shifts", num_images);

  for (size_t i = 0; i < num_owners; ++i) {
    owners(i) = owner_entities[i];
  }

  for (size_t k = 0; k < num_images; ++k) {
    boxes(k) = image_boxes[k];
    oi(k) = owner_indices[k];
    shifts(k) = image_shifts[k];
  }
  return PeriodicBoxes{selector, boxes, owners, oi, shifts};
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
                                                const std::vector<std::array<float, 3>>& positions,
                                                float L, float r) {
  const OrthorhombicMetric<AXIS_XYZ, float> metric{Vector3<float>{L, L, L}};
  const size_t N = positions.size();
  std::vector<ArborX::Box>    img_boxes(N);
  std::vector<size_t>         img_owners(N);
  std::vector<ImageShiftType> img_shifts(N);
  for (size_t i = 0; i < N; ++i) {
    const Point<float> orig{positions[i][0], positions[i][1], positions[i][2]};
    const Point<float> wrapped = metric.wrap(orig);
    img_boxes[i]  = make_arborx_box(wrapped[0], wrapped[1], wrapped[2], r);
    img_owners[i] = i;
    img_shifts[i] = wrapped - orig;
  }
  return make_periodic_boxes(selector, nodes, img_boxes, img_owners, img_shifts);
}

// Compute the union AABB of all image boxes in a PeriodicBoxes set.
// Pass the result to make_source_periodic_boxes() to prune candidate images
// that cannot possibly overlap any target.
static ArborX::Box compute_target_bbox(const PeriodicBoxes& target_boxes) {
  MUNDY_THROW_REQUIRE(target_boxes.size() > 0, std::invalid_argument,
                      "compute_target_bbox: target_boxes must be non-empty.");
  float lo[3] = { std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max()};
  float hi[3] = {-std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max()};
  for (size_t k = 0; k < target_boxes.size(); ++k) {
    const auto& b = target_boxes.box(k);
    for (int d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], b.minCorner()[d]);
      hi[d] = std::max(hi[d], b.maxCorner()[d]);
    }
  }
  return ArborX::Box{ArborX::Point{lo[0], lo[1], lo[2]},
                     ArborX::Point{hi[0], hi[1], hi[2]}};
}

// Build source boxes: up to 27 images per owner (wrapped + n*L for n∈{-1,0,1}^3).
// If target_bbox is provided, candidate images that don't intersect it are skipped —
// they cannot produce a pair with any target and including them wastes memory and
// ArborX build time.  Omit target_bbox (or pass std::nullopt) to get all 27 images.
// shift for surviving image n = (wrapped_position + n*L) - original_position.
static PeriodicBoxes make_source_periodic_boxes(const stk::mesh::Selector& selector,
                                                const std::vector<stk::mesh::Entity>& nodes,
                                                const std::vector<std::array<float, 3>>& positions,
                                                float L, float r,
                                                std::optional<ArborX::Box> target_bbox = std::nullopt) {
  const OrthorhombicMetric<AXIS_XYZ, float> metric{Vector3<float>{L, L, L}};
  std::vector<ArborX::Box>    img_boxes;
  std::vector<size_t>         img_owners;
  std::vector<ImageShiftType> img_shifts;
  img_boxes.reserve(positions.size() * 27);
  img_owners.reserve(positions.size() * 27);
  img_shifts.reserve(positions.size() * 27);
  for (size_t i = 0; i < positions.size(); ++i) {
    const Point<float> orig{positions[i][0], positions[i][1], positions[i][2]};
    const Point<float> wrapped = metric.wrap(orig);
    for (int nx : {-1, 0, 1}) {
      for (int ny : {-1, 0, 1}) {
        for (int nz : {-1, 0, 1}) {
          const auto image_pos = metric.shift_image(wrapped, Vector3<int>{nx, ny, nz});
          const ArborX::Box image_box = make_arborx_box(image_pos[0], image_pos[1], image_pos[2], r);
          if (target_bbox.has_value() && !boxes_overlap(image_box, *target_bbox)) continue;
          img_boxes.push_back(image_box);
          img_owners.push_back(i);
          img_shifts.push_back(image_pos - orig);
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

// Collect periodic pairs from a built list via direct iteration.
template <typename ListType>
std::vector<PeriodicPair> collect_periodic_pairs(const ListType& list) {
  std::vector<PeriodicPair> result;
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      result.push_back({t, list.source_index(t, k), list.relative_image_shift(t, k)});
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

// =============================================================================
// Build helpers
// =============================================================================

// Single-box overloads (for the deterministic 2-particle and single-image tests).
PerList1d build_per1d_list(const stk::mesh::BulkData& bulk, const PeriodicBoxes& boxes) {
  return make_neighbor_list_builder<PerList1d>()
      .exec_space(TestExecSpace{})
      .target_input(boxes)
      .source_input(boxes)
      .exclude(ExcludeSelfInteraction{})
      .build(bulk);
}

PerList2d build_per2d_list(const stk::mesh::BulkData& bulk, const PeriodicBoxes& boxes) {
  return make_neighbor_list_builder<PerList2d>()
      .exec_space(TestExecSpace{})
      .target_input(boxes)
      .source_input(boxes)
      .exclude(ExcludeSelfInteraction{})
      .build(bulk);
}

// Separate target/source overloads (correct form: 1 target image, 27 source images per owner).
PerList1d build_per1d_list(const stk::mesh::BulkData& bulk, const PeriodicBoxes& targets,
                           const PeriodicBoxes& sources) {
  return make_neighbor_list_builder<PerList1d>()
      .exec_space(TestExecSpace{})
      .target_input(targets)
      .source_input(sources)
      .exclude(ExcludeSelfInteraction{})
      .build(bulk);
}

PerList2d build_per2d_list(const stk::mesh::BulkData& bulk, const PeriodicBoxes& targets,
                           const PeriodicBoxes& sources) {
  return make_neighbor_list_builder<PerList2d>()
      .exec_space(TestExecSpace{})
      .target_input(targets)
      .source_input(sources)
      .exclude(ExcludeSelfInteraction{})
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
    auto [meta, bulk] = make_node_mesh(2);
    meta_ = std::move(meta);
    bulk_ = std::move(bulk);

    node_a_ = bulk_->get_entity(stk::topology::NODE_RANK, 1);
    node_b_ = bulk_->get_entity(stk::topology::NODE_RANK, 2);
    ASSERT_TRUE(bulk_->is_valid(node_a_));
    ASSERT_TRUE(bulk_->is_valid(node_b_));

    selector_ = meta_->universal_part();

    // Image boxes.
    // A images (owner ordinal 0):
    //   A_im0: center (0.5,0,0),  shift (0,0,0)   → x∈[-1, 2]
    //   A_im1: center (-9.5,0,0), shift (-10,0,0)  → x∈[-11,-8]
    // B images (owner ordinal 1):
    //   B_im0: center (9.5,0,0),  shift (0,0,0)   → x∈[8, 11]
    //   B_im1: center (-0.5,0,0), shift (-10,0,0)  → x∈[-2, 1]
    const std::vector<ArborX::Box> image_boxes = {
        make_arborx_box(kXa, 0.0f, 0.0f, kHx, kHyz, kHyz),       // A_im0
        make_arborx_box(kXa - kL, 0.0f, 0.0f, kHx, kHyz, kHyz),  // A_im1
        make_arborx_box(kXb, 0.0f, 0.0f, kHx, kHyz, kHyz),       // B_im0
        make_arborx_box(kXb - kL, 0.0f, 0.0f, kHx, kHyz, kHyz),  // B_im1
    };
    const std::vector<size_t> owner_indices = {0, 0, 1, 1};
    const std::vector<ImageShiftType> image_shifts = {
        ImageShiftType{0.0f, 0.0f, 0.0f},  // A_im0: zero shift
        ImageShiftType{-kL, 0.0f, 0.0f},   // A_im1: shifted left by L
        ImageShiftType{0.0f, 0.0f, 0.0f},  // B_im0: zero shift
        ImageShiftType{-kL, 0.0f, 0.0f},   // B_im1: shifted left by L
    };

    periodic_boxes_ = make_periodic_boxes(selector_, {node_a_, node_b_}, image_boxes, owner_indices, image_shifts);
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Entity node_a_;
  stk::mesh::Entity node_b_;
  stk::mesh::Selector selector_;
  PeriodicBoxes periodic_boxes_;
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

// Verifies collect_periodic_pairs returns the right pairs for the known 2-particle geometry.
TEST(TestInfra, CollectPeriodicPairsMatchesKnown2ParticleGeometry) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto [meta, bulk] = make_node_mesh(2);
  const stk::mesh::Selector selector = meta->universal_part();
  auto node_a = bulk->get_entity(stk::topology::NODE_RANK, 1);
  auto node_b = bulk->get_entity(stk::topology::NODE_RANK, 2);
  constexpr float kL = 10.0f, kXa = 0.5f, kXb = 9.5f, kHx = 1.5f, kHyz = 1.0f;
  auto pboxes = make_periodic_boxes(
      selector, {node_a, node_b},
      {make_arborx_box(kXa, 0, 0, kHx, kHyz, kHyz), make_arborx_box(kXa - kL, 0, 0, kHx, kHyz, kHyz),
       make_arborx_box(kXb, 0, 0, kHx, kHyz, kHyz), make_arborx_box(kXb - kL, 0, 0, kHx, kHyz, kHyz)},
      {0, 0, 1, 1}, {{0, 0, 0}, {-kL, 0, 0}, {0, 0, 0}, {-kL, 0, 0}});
  auto pairs = collect_periodic_pairs(build_per1d_list(*bulk, pboxes));
  ASSERT_EQ(pairs.size(), 2u);
  std::set<PeriodicPair> ps(pairs.begin(), pairs.end());
  EXPECT_EQ(ps.count({0, 1, {-kL, 0, 0}}), 1u);
  EXPECT_EQ(ps.count({1, 0, {kL, 0, 0}}), 1u);
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
  constexpr int kN = 20; constexpr size_t kSeed = 77;
  constexpr float L = 10.0f, r = 1.5f;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, L),
                    rng.uniform<float>(0.0f, L),
                    rng.uniform<float>(0.0f, L)};
  }
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i)
    nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, static_cast<stk::mesh::EntityId>(i + 1));

  const auto tboxes       = make_target_periodic_boxes(sel, nodes, positions, L, r);
  const auto sboxes_full  = make_source_periodic_boxes(sel, nodes, positions, L, r);
  const auto sboxes_pruned = make_source_periodic_boxes(sel, nodes, positions, L, r,
                                                         compute_target_bbox(tboxes));

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

  // List built from PRUNED source boxes — one ArborX build, no stall.
  auto list = build_per1d_list(*bulk, tboxes, sboxes_pruned);
  const auto actual_vec = collect_periodic_pairs(list);
  const std::set<PeriodicPair> actual(actual_vec.begin(), actual_vec.end());

  EXPECT_EQ(actual_vec.size(), actual.size()) << "List has duplicate entries after pruning.";
  EXPECT_EQ(actual.size(), oracle.size())
      << "list=" << actual.size() << " oracle=" << oracle.size();
  for (const auto& p : oracle)
    EXPECT_TRUE(actual.count(p))
        << "MISSING t=" << p.target_owner << " s=" << p.source_owner
        << " shift=(" << p.relative_shift[0] << "," << p.relative_shift[1]
        << "," << p.relative_shift[2] << ")";
  for (const auto& p : actual)
    EXPECT_TRUE(oracle.count(p))
        << "SPURIOUS t=" << p.target_owner << " s=" << p.source_owner
        << " shift=(" << p.relative_shift[0] << "," << p.relative_shift[1]
        << "," << p.relative_shift[2] << ")";
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
  auto original = build_per1d_list(*bulk_, periodic_boxes_);
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
  auto original = build_per2d_list(*bulk_, periodic_boxes_);
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
  const ImageShiftType shift_A = list.relative_image_shift(0, 0);
  EXPECT_FLOAT_EQ(shift_A[0], -kL);
  EXPECT_FLOAT_EQ(shift_A[1], 0.0f);
  EXPECT_FLOAT_EQ(shift_A[2], 0.0f);

  // --- Owner B (idx 1) ---
  // Source owner ordinal must be 0 (owner A), entity must be node_a.
  EXPECT_EQ(list.source_index(1, 0), size_t(0));
  EXPECT_EQ(list.get_neighbor(1, 0), node_a);
  // Relative shift: source A_im0 shift (0) minus target B_im1 shift (-10) = +10.
  const ImageShiftType shift_B = list.relative_image_shift(1, 0);
  EXPECT_FLOAT_EQ(shift_B[0], kL);
  EXPECT_FLOAT_EQ(shift_B[1], 0.0f);
  EXPECT_FLOAT_EQ(shift_B[2], 0.0f);

  // Iteration helpers.
  check_for_each_pair_count(list);
  check_for_each_target_count(list);
}

TEST_F(PeriodicFixture, Deterministic_1d) {
  auto list = build_per1d_list(*bulk_, periodic_boxes_);
  verify_periodic_2particle(list, node_a_, node_b_, kL);
}

TEST_F(PeriodicFixture, Deterministic_2d) {
  auto list = build_per2d_list(*bulk_, periodic_boxes_);
  verify_periodic_2particle(list, node_a_, node_b_, kL);
}

// =============================================================================
// Group 3 — Random N² single-image validation
// =============================================================================

// Each owner gets exactly one image with zero shift (owner_index[k] = k).
// The periodic infrastructure must then produce the same pairs as the non-periodic
// oracle. Additionally, all stored relative image shifts must be zero.
template <typename ListType, typename BuildFn>
void run_single_image_n2_validation(BuildFn build_fn, stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector,
                                    int num_nodes) {
  constexpr size_t kSeed = 137;
  constexpr float kDomainSize = 10.0f;
  constexpr float kRadius = 0.9f;

  std::vector<stk::mesh::Entity> nodes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i] = bulk.get_entity(stk::topology::NODE_RANK, i + 1);
    ASSERT_TRUE(bulk.is_valid(nodes[i])) << "Node " << (i + 1) << " not found.";
  }

  // Generate sphere boxes with Philox (counter = particle ordinal).
  std::vector<ArborX::Box> boxes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    const float cx = rng.uniform<float>(0.0f, kDomainSize);
    const float cy = rng.uniform<float>(0.0f, kDomainSize);
    const float cz = rng.uniform<float>(0.0f, kDomainSize);
    boxes[i] = make_arborx_box(cx, cy, cz, kRadius);
  }

  // One image per owner, all with zero shift.
  // image_index k → owner_index k (bijection).
  std::vector<size_t> owner_indices(num_nodes);
  std::vector<ImageShiftType> image_shifts(num_nodes, ImageShiftType{0.0f, 0.0f, 0.0f});
  for (int i = 0; i < num_nodes; ++i) owner_indices[i] = static_cast<size_t>(i);

  PeriodicBoxes periodic_boxes = make_periodic_boxes(selector, nodes, boxes, owner_indices, image_shifts);
  ListType list = build_fn(bulk, periodic_boxes);

  // The index-pair set must match the non-periodic oracle.
  const auto expected = oracle_pairs_no_self(boxes);
  const auto actual = collect_index_pairs(list);
  EXPECT_EQ(actual, expected) << "Single-image periodic list does not match non-periodic oracle.";

  // All relative image shifts must be (0,0,0): source shift (0) - target shift (0) = 0.
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const ImageShiftType shift = list.relative_image_shift(t, k);
      EXPECT_FLOAT_EQ(shift[0], 0.0f) << "Non-zero x-shift for pair (target=" << t << ", neighbor=" << k << ").";
      EXPECT_FLOAT_EQ(shift[1], 0.0f) << "Non-zero y-shift for pair (target=" << t << ", neighbor=" << k << ").";
      EXPECT_FLOAT_EQ(shift[2], 0.0f) << "Non-zero z-shift for pair (target=" << t << ", neighbor=" << k << ").";
    }
  }
}

TEST(PeriodicArborX1dNeighborList, SingleImageN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_single_image_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& boxes) { return build_per1d_list(b, boxes); }, *bulk, selector,
      kN);
}

TEST(PeriodicArborX2dNeighborList, SingleImageN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_single_image_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& boxes) { return build_per2d_list(b, boxes); }, *bulk, selector,
      kN);
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
// Uses make_target_periodic_boxes (1 image/owner) and the full-27-image
// make_source_periodic_boxes (no AABB pruning) so the oracle is independent of
// compute_target_bbox. With 1 target image per owner and distinct per-owner source
// shifts, each (target_owner, source_owner, relative_shift) triple is produced by
// exactly one image pair — no duplicates possible.
template <typename ListType, typename BuildFn>
void run_full_periodic_n2_validation(BuildFn build_fn, stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector,
                                     const std::vector<std::array<float, 3>>& positions, float L, float r) {
  const size_t N = positions.size();
  std::vector<stk::mesh::Entity> nodes(N);
  for (size_t i = 0; i < N; ++i) {
    nodes[i] = bulk.get_entity(stk::topology::NODE_RANK, static_cast<stk::mesh::EntityId>(i + 1));
    ASSERT_TRUE(bulk.is_valid(nodes[i])) << "node " << (i + 1) << " missing.";
  }

  const auto tboxes = make_target_periodic_boxes(selector, nodes, positions, L, r);
  const auto sboxes = make_source_periodic_boxes(selector, nodes, positions, L, r);
  ListType list = build_fn(bulk, tboxes, sboxes);

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
  auto [meta, bulk] = make_node_mesh(static_cast<int>(kBoundaryPositions.size()));
  const stk::mesh::Selector selector = meta->universal_part();
  run_full_periodic_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& t, const PeriodicBoxes& s) { return build_per1d_list(b, t, s); },
      *bulk, selector, kBoundaryPositions, kValidL, kValidR);
}

TEST(PeriodicArborX2dNeighborList, BoundaryN2Validation) {
  auto [meta, bulk] = make_node_mesh(static_cast<int>(kBoundaryPositions.size()));
  const stk::mesh::Selector selector = meta->universal_part();
  run_full_periodic_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& t, const PeriodicBoxes& s) { return build_per2d_list(b, t, s); },
      *bulk, selector, kBoundaryPositions, kValidL, kValidR);
}

// Random positions.
TEST(PeriodicArborX1dNeighborList, RandomFullPeriodicN2Validation) {
  constexpr int kN = 40;
  constexpr size_t kSeed = 42;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kValidL), rng.uniform<float>(0.0f, kValidL),
                    rng.uniform<float>(0.0f, kValidL)};
  }
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_full_periodic_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& t, const PeriodicBoxes& s) { return build_per1d_list(b, t, s); },
      *bulk, selector, positions, kValidL, kValidR);
}

TEST(PeriodicArborX2dNeighborList, RandomFullPeriodicN2Validation) {
  constexpr int kN = 40;
  constexpr size_t kSeed = 42;
  std::vector<std::array<float, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<float>(0.0f, kValidL), rng.uniform<float>(0.0f, kValidL),
                    rng.uniform<float>(0.0f, kValidL)};
  }
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_full_periodic_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& t, const PeriodicBoxes& s) { return build_per2d_list(b, t, s); },
      *bulk, selector, positions, kValidL, kValidR);
}

// =============================================================================
// Group 4 — Debug-only bound-check tests
// =============================================================================

#ifndef NDEBUG

TEST_F(PeriodicFixture, OutOfBounds_1d) {
  auto list = build_per1d_list(*bulk_, periodic_boxes_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
  EXPECT_THROW(list.relative_image_shift(0, list.num_neighbors(0)), std::out_of_range);
}

TEST_F(PeriodicFixture, OutOfBounds_2d) {
  auto list = build_per2d_list(*bulk_, periodic_boxes_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
  EXPECT_THROW(list.relative_image_shift(0, list.num_neighbors(0)), std::out_of_range);
}

#endif  // NDEBUG

}  // namespace
}  // namespace search
}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX
