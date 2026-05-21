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
#include <memory>
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

// Mundy math / utils
#include <mundy_math/Vector3.hpp>   // for mundy::Vector3
#include <mundy_utils/rng.hpp>      // for mundy::make_philox

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

using TestMemSpace    = Kokkos::HostSpace;
using TestExecSpace   = Kokkos::DefaultHostExecutionSpace;
using ImageShiftType  = mundy::Vector3<float>;
using PeriodicBoxes   = impl::PeriodicArborXSearchBoxesT<TestMemSpace, float>;
using PerList1d       = PeriodicArborX1dNeighborList<TestMemSpace, float>;
using PerList2d       = PeriodicArborX2dNeighborList<TestMemSpace, float>;

// Reuse non-periodic search boxes for the single-image N² test.
using SearchBoxes     = impl::ArborXSearchBoxesT<TestMemSpace>;

// =============================================================================
// Box helpers
// =============================================================================

ArborX::Box make_arborx_box(float cx, float cy, float cz, float hx, float hy, float hz) {
  return ArborX::Box{ArborX::Point{cx - hx, cy - hy, cz - hz},
                     ArborX::Point{cx + hx, cy + hy, cz + hz}};
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

std::pair<std::shared_ptr<stk::mesh::MetaData>, std::unique_ptr<stk::mesh::BulkData>>
make_node_mesh(int num_nodes) {
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
PeriodicBoxes make_periodic_boxes(
    const stk::mesh::Selector&              selector,
    const std::vector<stk::mesh::Entity>&   owner_entities,
    const std::vector<ArborX::Box>&         image_boxes,
    const std::vector<size_t>&              owner_indices,
    const std::vector<ImageShiftType>&      image_shifts) {
  const size_t num_owners = owner_entities.size();
  const size_t num_images = image_boxes.size();
  EXPECT_EQ(num_images, owner_indices.size());
  EXPECT_EQ(num_images, image_shifts.size());

  Kokkos::View<ArborX::Box*, TestMemSpace>       boxes("per_boxes", num_images);
  Kokkos::View<stk::mesh::Entity*, TestMemSpace> owners("per_owners", num_owners);
  Kokkos::View<size_t*, TestMemSpace>            oi("per_oi", num_images);
  Kokkos::View<ImageShiftType*, TestMemSpace>    shifts("per_shifts", num_images);

  for (size_t i = 0; i < num_owners; ++i) owners(i) = owner_entities[i];
  for (size_t k = 0; k < num_images; ++k) {
    boxes(k)  = image_boxes[k];
    oi(k)     = owner_indices[k];
    shifts(k) = image_shifts[k];
  }
  return PeriodicBoxes{selector, boxes, owners, oi, shifts};
}

// =============================================================================
// Oracle helpers for periodic tests
// =============================================================================

// Pair type for periodic validation: (target_owner, source_owner, relative_shift).
struct PeriodicPair {
  size_t        target_owner;
  size_t        source_owner;
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
           relative_shift[0] == o.relative_shift[0] &&
           relative_shift[1] == o.relative_shift[1] &&
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

// =============================================================================
// Shared iteration-count checks
// =============================================================================

template <typename ListType>
void check_for_each_pair_count(const ListType& list) {
  Kokkos::View<size_t*, TestMemSpace> count("fep_count", 1);
  Kokkos::deep_copy(count, size_t(0));
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const NeighborPair<ListType>&) { Kokkos::atomic_inc(&count(0)); });
  Kokkos::fence();
  EXPECT_EQ(count(0), list.size())
      << "for_each_neighbor_pair count does not match list.size().";
}

template <typename ListType>
void check_for_each_target_count(const ListType& list) {
  Kokkos::View<size_t*, TestMemSpace> count("fet_count", 1);
  Kokkos::deep_copy(count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const Neighbors<ListType>&) { Kokkos::atomic_inc(&count(0)); });
  Kokkos::fence();
  EXPECT_EQ(count(0), list.num_targets())
      << "for_each_target_with_neighbors count does not match list.num_targets().";
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
  static constexpr float kL      = 10.0f;  // periodic domain length
  static constexpr float kXa     = 0.5f;   // owner A x-position
  static constexpr float kXb     = 9.5f;   // owner B x-position

  // Image box half-widths.
  static constexpr float kHx = kRadius;   // 1.5
  static constexpr float kHyz = 1.0f;     // deliberately narrower in y,z to isolate x-axis overlaps

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
        make_arborx_box(kXa,       0.0f, 0.0f, kHx, kHyz, kHyz),  // A_im0
        make_arborx_box(kXa - kL,  0.0f, 0.0f, kHx, kHyz, kHyz),  // A_im1
        make_arborx_box(kXb,       0.0f, 0.0f, kHx, kHyz, kHyz),  // B_im0
        make_arborx_box(kXb - kL,  0.0f, 0.0f, kHx, kHyz, kHyz),  // B_im1
    };
    const std::vector<size_t> owner_indices = {0, 0, 1, 1};
    const std::vector<ImageShiftType> image_shifts = {
        ImageShiftType{  0.0f, 0.0f, 0.0f},   // A_im0: zero shift
        ImageShiftType{-kL,   0.0f, 0.0f},     // A_im1: shifted left by L
        ImageShiftType{  0.0f, 0.0f, 0.0f},   // B_im0: zero shift
        ImageShiftType{-kL,   0.0f, 0.0f},     // B_im1: shifted left by L
    };

    periodic_boxes_ = make_periodic_boxes(selector_, {node_a_, node_b_},
                                          image_boxes, owner_indices, image_shifts);
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Entity                    node_a_;
  stk::mesh::Entity                    node_b_;
  stk::mesh::Selector                  selector_;
  PeriodicBoxes                        periodic_boxes_;
};

// =============================================================================
// Group 1 — Construction and access
// =============================================================================

TEST(PeriodicArborX1dNeighborList, DefaultConstruct) {
  PerList1d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(),        0u);
}

TEST(PeriodicArborX2dNeighborList, DefaultConstruct) {
  PerList2d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(),        0u);
}

TEST_F(PeriodicFixture, CopyMove_1d) {
  auto original = build_per1d_list(*bulk_, periodic_boxes_);
  const size_t nt   = original.num_targets();
  const size_t ns   = original.num_sources();
  const size_t size = original.size();

  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(),        size);

  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(),        size);

  PerList1d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.size(),        size);

  PerList1d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.size(),        size);
}

TEST_F(PeriodicFixture, CopyMove_2d) {
  auto original = build_per2d_list(*bulk_, periodic_boxes_);
  const size_t nt   = original.num_targets();
  const size_t ns   = original.num_sources();
  const size_t size = original.size();

  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(),        size);

  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(),        size);

  PerList2d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.size(),        size);

  PerList2d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.size(),        size);
}

// =============================================================================
// Group 2 — Deterministic 2-particle periodic test
// =============================================================================

// Shared body verifying the deterministic periodic-boundary result.
// For the 2-owner layout, the expected pairs are:
//   (owner 0=A, owner 1=B, relative shift (-10,0,0))
//   (owner 1=B, owner 0=A, relative shift (+10,0,0))
template <typename ListType>
void verify_periodic_2particle(const ListType& list,
                               stk::mesh::Entity node_a,
                               stk::mesh::Entity node_b,
                               float kL) {
  ASSERT_EQ(list.num_targets(), 2u);
  ASSERT_EQ(list.num_sources(), 2u);
  EXPECT_EQ(list.size(),        2u);

  EXPECT_EQ(list.num_neighbors(0), 1u);  // owner A sees owner B
  EXPECT_EQ(list.num_neighbors(1), 1u);  // owner B sees owner A

  // Entity accessors.
  EXPECT_EQ(list.target_entity(0), node_a);
  EXPECT_EQ(list.target_entity(1), node_b);

  // --- Owner A (idx 0) ---
  // Source owner ordinal must be 1 (owner B), entity must be node_b.
  EXPECT_EQ(list.source_index(0, 0),   size_t(1));
  EXPECT_EQ(list.get_neighbor(0, 0),   node_b);
  // Relative shift: source B_im1 shift (-10) minus target A_im0 shift (0) = -10.
  const ImageShiftType shift_A = list.relative_image_shift(0, 0);
  EXPECT_FLOAT_EQ(shift_A[0], -kL);
  EXPECT_FLOAT_EQ(shift_A[1], 0.0f);
  EXPECT_FLOAT_EQ(shift_A[2], 0.0f);

  // --- Owner B (idx 1) ---
  // Source owner ordinal must be 0 (owner A), entity must be node_a.
  EXPECT_EQ(list.source_index(1, 0),   size_t(0));
  EXPECT_EQ(list.get_neighbor(1, 0),   node_a);
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
void run_single_image_n2_validation(BuildFn build_fn,
                                    stk::mesh::BulkData& bulk,
                                    const stk::mesh::Selector& selector,
                                    int num_nodes) {
  constexpr size_t kSeed       = 137;
  constexpr float  kDomainSize = 10.0f;
  constexpr float  kRadius     = 0.9f;

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
  std::vector<size_t>        owner_indices(num_nodes);
  std::vector<ImageShiftType> image_shifts(num_nodes, ImageShiftType{0.0f, 0.0f, 0.0f});
  for (int i = 0; i < num_nodes; ++i) owner_indices[i] = static_cast<size_t>(i);

  PeriodicBoxes periodic_boxes = make_periodic_boxes(selector, nodes, boxes, owner_indices, image_shifts);
  ListType list = build_fn(bulk, periodic_boxes);

  // The index-pair set must match the non-periodic oracle.
  const auto expected = oracle_pairs_no_self(boxes);
  const auto actual   = collect_index_pairs(list);
  EXPECT_EQ(actual, expected)
      << "Single-image periodic list does not match non-periodic oracle.";

  // All relative image shifts must be (0,0,0): source shift (0) - target shift (0) = 0.
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const ImageShiftType shift = list.relative_image_shift(t, k);
      EXPECT_FLOAT_EQ(shift[0], 0.0f)
          << "Non-zero x-shift for pair (target=" << t << ", neighbor=" << k << ").";
      EXPECT_FLOAT_EQ(shift[1], 0.0f)
          << "Non-zero y-shift for pair (target=" << t << ", neighbor=" << k << ").";
      EXPECT_FLOAT_EQ(shift[2], 0.0f)
          << "Non-zero z-shift for pair (target=" << t << ", neighbor=" << k << ").";
    }
  }
}

TEST(PeriodicArborX1dNeighborList, SingleImageN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_single_image_n2_validation<PerList1d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& boxes) {
        return build_per1d_list(b, boxes);
      },
      *bulk, selector, kN);
}

TEST(PeriodicArborX2dNeighborList, SingleImageN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_single_image_n2_validation<PerList2d>(
      [](stk::mesh::BulkData& b, const PeriodicBoxes& boxes) {
        return build_per2d_list(b, boxes);
      },
      *bulk, selector, kN);
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
