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

/// \file UnitTestNeighborList.cpp
/// \brief Unit tests for ArborX1dNeighborList and ArborX2dNeighborList.
///
/// Test structure:
///   Group 1 — Construction and access:
///     default construction, copy/move semantics (all four operations).
///   Group 2 — Deterministic 4-box test:
///     4 nodes with known AABB boxes; ExcludeSelfInteraction removes self-pairs.
///     Exact expected pair structure is verified, along with for_each iteration counts.
///   Group 3 — Random N² validation:
///     N spheres at random Philox-generated positions; every pair found in the list
///     must correspond to overlapping boxes, and every overlapping oracle pair must
///     appear in the list.
///   Group 4 — Debug-only bound-check tests (NDEBUG guard):
///     MUNDY_THROW_ASSERT fires only in debug builds; tests wrap expected-throw cases
///     in #ifndef NDEBUG.

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

// Mundy utils
#include <mundy_utils/rng.hpp>  // for mundy::make_philox

namespace mundy {
namespace search {
namespace {

// =============================================================================
// Compile-time concept checks
// =============================================================================

// These fire at translation-unit scope: the test binary won't link if broken.
static_assert(NeighborListType<ArborX1dNeighborList<Kokkos::HostSpace>>,
              "ArborX1dNeighborList<HostSpace> must satisfy NeighborListType.");
static_assert(NeighborListType<ArborX2dNeighborList<Kokkos::HostSpace>>,
              "ArborX2dNeighborList<HostSpace> must satisfy NeighborListType.");

// =============================================================================
// Type aliases
// =============================================================================

// All tests run on the host so we can iterate over list views directly without
// device-side deep_copy round-trips.
using TestMemSpace = Kokkos::HostSpace;
using TestExecSpace = Kokkos::DefaultHostExecutionSpace;
using SearchBoxes = impl::ArborXSearchBoxesT<TestMemSpace>;
using List1d = ArborX1dNeighborList<TestMemSpace>;
using List2d = ArborX2dNeighborList<TestMemSpace>;

// =============================================================================
// Box helpers
// =============================================================================

// Create an ArborX::Box from center + per-axis half-widths.
ArborX::Box make_arborx_box(float cx, float cy, float cz, float hx, float hy, float hz) {
  return ArborX::Box{ArborX::Point{cx - hx, cy - hy, cz - hz}, ArborX::Point{cx + hx, cy + hy, cz + hz}};
}

// Overload for uniform half-width (sphere-like box).
ArborX::Box make_arborx_box(float cx, float cy, float cz, float h) {
  return make_arborx_box(cx, cy, cz, h, h, h);
}

// Two boxes intersect when their intervals are non-empty in all three dimensions.
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

// Create a minimal node-only STK mesh with nodes numbered 1..num_nodes.
// No fields are declared; only the universal part exists.
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
// Search-box construction helper
// =============================================================================

// Wrap pre-populated host arrays into an ArborXSearchBoxesT<HostSpace>.
// boxes_h[i] and entities_h[i] must correspond to the same search object.
SearchBoxes make_search_boxes(const stk::mesh::Selector& selector, const std::vector<ArborX::Box>& boxes_h,
                              const std::vector<stk::mesh::Entity>& entities_h) {
  const size_t n = boxes_h.size();
  EXPECT_EQ(n, entities_h.size());  // sanity: equal-length arrays
  Kokkos::View<ArborX::Box*, TestMemSpace> boxes("unit_test_boxes", n);
  Kokkos::View<stk::mesh::Entity*, TestMemSpace> entities("unit_test_entities", n);
  for (size_t i = 0; i < n; ++i) {
    boxes(i) = boxes_h[i];
    entities(i) = entities_h[i];
  }
  return SearchBoxes{selector, boxes, entities};
}

// =============================================================================
// Oracle
// =============================================================================

// Compute the expected (target_ordinal, source_ordinal) pair set: all pairs where
// boxes overlap, excluding t==s (mirrors ExcludeSelfInteraction with identical
// target and source box sets).
std::set<std::pair<size_t, size_t>> oracle_pairs(const std::vector<ArborX::Box>& target_boxes,
                                                 const std::vector<ArborX::Box>& source_boxes) {
  std::set<std::pair<size_t, size_t>> pairs;
  for (size_t t = 0; t < target_boxes.size(); ++t) {
    for (size_t s = 0; s < source_boxes.size(); ++s) {
      if (t == s) continue;  // ExcludeSelfInteraction removes same-entity pairs
      if (boxes_overlap(target_boxes[t], source_boxes[s])) pairs.insert({t, s});
    }
  }
  return pairs;
}

// =============================================================================
// Direct pair extraction (host-side, no parallel overhead)
// =============================================================================

// Collect all (target_ordinal, source_ordinal) pairs from a built list by
// walking the list's own accessors.  Correct only for HostSpace lists where
// views are accessible without a device round-trip.
template <typename ListType>
std::set<std::pair<size_t, size_t>> collect_pairs(const ListType& list) {
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

List1d build_1d_list(const stk::mesh::BulkData& bulk, const SearchBoxes& target_boxes,
                     const SearchBoxes& source_boxes) {
  return make_neighbor_list_builder<List1d>()
      .exec_space(TestExecSpace{})
      .target_input(target_boxes)
      .source_input(source_boxes)
      .exclude(ExcludeSelfInteraction{})
      .build(bulk);
}

List2d build_2d_list(const stk::mesh::BulkData& bulk, const SearchBoxes& target_boxes,
                     const SearchBoxes& source_boxes) {
  return make_neighbor_list_builder<List2d>()
      .exec_space(TestExecSpace{})
      .target_input(target_boxes)
      .source_input(source_boxes)
      .exclude(ExcludeSelfInteraction{})
      .build(bulk);
}

// =============================================================================
// Shared list-validity checks
// =============================================================================

// Verify that for_each_neighbor_pair visits exactly list.size() pairs.
// Uses an atomic counter so the test is correct even under OpenMP host execution.
template <typename ListType>
void check_for_each_pair_count(const ListType& list) {
  Kokkos::View<size_t*, TestMemSpace> count("fep_count", 1);
  Kokkos::deep_copy(count, size_t(0));
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const NeighborPair<ListType>&) { Kokkos::atomic_inc(&count(0)); });
  Kokkos::fence();
  EXPECT_EQ(count(0), list.size()) << "for_each_neighbor_pair count does not match list.size().";
}

// Verify that for_each_target_with_neighbors visits exactly list.num_targets() targets.
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
// Test Fixture — 4-node mesh with known AABB geometry
// =============================================================================

// Four AABB boxes assigned one-to-one to STK nodes 1..4 (ordinals 0..3):
//
//   Box 0 (node 1): center (0,0,0) half-size 2.0  →  [-2,2]³
//   Box 1 (node 2): center (3,0,0) half (1.5,1,1) →  [1.5,4.5]×[-1,1]×[-1,1]
//   Box 2 (node 3): center (0,3,0) half (1,1.5,1) →  [-1,1]×[1.5,4.5]×[-1,1]
//   Box 3 (node 4): center (100.5,100.5,100.5) h=0.5 →  isolated
//
// Box overlaps after ExcludeSelfInteraction:
//   Box 0 ↔ Box 1  (x-overlap: [1.5,2])
//   Box 0 ↔ Box 2  (y-overlap: [1.5,2])
//   Box 1 ↔ Box 2  NO  (x=[1.5,4.5] ∩ [-1,1] is empty)
//   Box 3           no neighbors
//
// Expected pairs: (0,1),(0,2),(1,0),(2,0) — 4 total.
class NonPeriodicFixture : public ::testing::Test {
 protected:
  static constexpr int kN = 4;

  void SetUp() override {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
      GTEST_SKIP();
    }
    auto [meta, bulk] = make_node_mesh(kN);
    meta_ = std::move(meta);
    bulk_ = std::move(bulk);

    // Retrieve nodes in explicit ID order: ordinal i = node ID (i+1).
    nodes_.resize(kN);
    for (int id = 1; id <= kN; ++id) {
      nodes_[id - 1] = bulk_->get_entity(stk::topology::NODE_RANK, id);
      ASSERT_TRUE(bulk_->is_valid(nodes_[id - 1])) << "Node " << id << " not found.";
    }

    selector_ = meta_->universal_part();

    boxes_ = {
        make_arborx_box(0.0f, 0.0f, 0.0f, 2.0f),              // box 0
        make_arborx_box(3.0f, 0.0f, 0.0f, 1.5f, 1.0f, 1.0f),  // box 1
        make_arborx_box(0.0f, 3.0f, 0.0f, 1.0f, 1.5f, 1.0f),  // box 2
        make_arborx_box(100.5f, 100.5f, 100.5f, 0.5f),        // box 3 (isolated)
    };

    search_boxes_ = make_search_boxes(selector_, boxes_, nodes_);
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  std::vector<stk::mesh::Entity> nodes_;
  stk::mesh::Selector selector_;
  std::vector<ArborX::Box> boxes_;
  SearchBoxes search_boxes_;
};

// =============================================================================
// Group 1 — Construction and access
// =============================================================================

TEST(ArborX1dNeighborList, DefaultConstruct) {
  // Default-constructed list must report zero extents without crashing.
  List1d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(), 0u);
}

TEST(ArborX2dNeighborList, DefaultConstruct) {
  List2d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(), 0u);
}

// Verify all four copy/move operations for the 1D list.
// Kokkos views are reference-counted, so copy yields a shallow copy that shares
// the same underlying data and reports identical counts.
TEST_F(NonPeriodicFixture, CopyMove_1d) {
  auto original = build_1d_list(*bulk_, search_boxes_, search_boxes_);
  const size_t nt = original.num_targets();
  const size_t ns = original.num_sources();
  const size_t size = original.size();

  // Copy construction.
  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(), size);

  // Move construction.
  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(), size);

  // Copy assignment.
  List1d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.num_sources(), ns);
  EXPECT_EQ(copy_assign.size(), size);

  // Move assignment.
  List1d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.num_sources(), ns);
  EXPECT_EQ(move_assign.size(), size);
}

TEST_F(NonPeriodicFixture, CopyMove_2d) {
  auto original = build_2d_list(*bulk_, search_boxes_, search_boxes_);
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

  List2d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.num_sources(), ns);
  EXPECT_EQ(copy_assign.size(), size);

  List2d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.num_sources(), ns);
  EXPECT_EQ(move_assign.size(), size);
}

// =============================================================================
// Group 2 — Deterministic 4-box test
// =============================================================================

// Shared body: verify the structure and iteration behavior of a list built from
// the 4-box fixture. Called for both 1D and 2D to avoid duplicating assertions.
template <typename ListType>
void verify_deterministic_4box(const ListType& list, const std::vector<stk::mesh::Entity>& nodes,
                               const std::vector<ArborX::Box>& boxes) {
  // Counts.
  ASSERT_EQ(list.num_targets(), 4u);
  ASSERT_EQ(list.num_sources(), 4u);
  EXPECT_EQ(list.size(), 4u);  // (0,1),(0,2),(1,0),(2,0)

  // Per-target neighbor counts.
  EXPECT_EQ(list.num_neighbors(0), 2u);  // box 0 overlaps boxes 1 and 2
  EXPECT_EQ(list.num_neighbors(1), 1u);  // box 1 overlaps box 0 only
  EXPECT_EQ(list.num_neighbors(2), 1u);  // box 2 overlaps box 0 only
  EXPECT_EQ(list.num_neighbors(3), 0u);  // box 3 is isolated

  // Target entity accessors: ordinal i must return nodes[i].
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(list.target_entity(i), nodes[i]) << "target_entity(" << i << ") mismatch.";
  }

  // Neighbors of target 0 must be exactly {nodes[1], nodes[2]} in any order.
  {
    std::set<stk::mesh::Entity> nbrs_of_0;
    for (size_t k = 0; k < list.num_neighbors(0); ++k) {
      nbrs_of_0.insert(list.get_neighbor(0, k));
    }
    EXPECT_EQ(nbrs_of_0, (std::set<stk::mesh::Entity>{nodes[1], nodes[2]}));
  }

  // Each of targets 1 and 2 has one neighbor: nodes[0].
  EXPECT_EQ(list.get_neighbor(1, 0), nodes[0]);
  EXPECT_EQ(list.get_neighbor(2, 0), nodes[0]);

  // Oracle-level pair set check.
  const auto expected = oracle_pairs(boxes, boxes);
  const auto actual = collect_pairs(list);
  EXPECT_EQ(actual, expected) << "Neighbor list pair set does not match oracle.";

  // Iteration helpers: count visited pairs and targets.
  check_for_each_pair_count(list);
  check_for_each_target_count(list);
}

TEST_F(NonPeriodicFixture, Deterministic_1d) {
  auto list = build_1d_list(*bulk_, search_boxes_, search_boxes_);
  verify_deterministic_4box(list, nodes_, boxes_);
}

TEST_F(NonPeriodicFixture, Deterministic_2d) {
  auto list = build_2d_list(*bulk_, search_boxes_, search_boxes_);
  verify_deterministic_4box(list, nodes_, boxes_);
}

// =============================================================================
// Group 3 — Random N² correctness validation
// =============================================================================

// Place N spheres at random Philox-generated centers (seed=42, counter=ordinal).
// Build the list, then:
//   (a) Verify every list pair corresponds to actually-overlapping boxes.
//   (b) Verify every oracle-overlapping pair appears in the list.
// The same oracle comparison used for the deterministic test is re-used here.
template <typename ListType, typename BuildFn>
void run_random_n2_validation(BuildFn build_fn, stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector,
                              int num_nodes) {
  constexpr size_t kSeed = 42;
  constexpr float kDomainSize = 10.0f;
  constexpr float kRadius = 0.9f;

  std::vector<stk::mesh::Entity> nodes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i] = bulk.get_entity(stk::topology::NODE_RANK, i + 1);
    ASSERT_TRUE(bulk.is_valid(nodes[i])) << "Node " << (i + 1) << " not found.";
  }

  // Generate sphere boxes: counter = particle ordinal so each particle gets an
  // independent random stream regardless of thread scheduling.
  std::vector<ArborX::Box> boxes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    const float cx = rng.uniform<float>(0.0f, kDomainSize);
    const float cy = rng.uniform<float>(0.0f, kDomainSize);
    const float cz = rng.uniform<float>(0.0f, kDomainSize);
    boxes[i] = make_arborx_box(cx, cy, cz, kRadius);
  }

  SearchBoxes search_boxes = make_search_boxes(selector, boxes, nodes);
  ListType list = build_fn(bulk, search_boxes, search_boxes);

  // (a) Every pair in the list must correspond to overlapping boxes.
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t s = list.source_index(t, k);
      EXPECT_TRUE(boxes_overlap(boxes[t], boxes[s]))
          << "Spurious pair (target=" << t << ", source=" << s << "): boxes do not overlap.";
      EXPECT_NE(t, s) << "Self-pair (target==source==" << t << ") found despite ExcludeSelfInteraction.";
    }
  }

  // (b) Oracle pairs must all be present.
  const auto expected = oracle_pairs(boxes, boxes);
  const auto actual = collect_pairs(list);
  EXPECT_EQ(actual, expected) << "Neighbor list is missing oracle pairs or contains extra pairs.";
}

TEST(ArborX1dNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_random_n2_validation<List1d>(
      [](stk::mesh::BulkData& b, const SearchBoxes& tgt, const SearchBoxes& src) { return build_1d_list(b, tgt, src); },
      *bulk, selector, kN);
}

TEST(ArborX2dNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector selector = meta->universal_part();
  run_random_n2_validation<List2d>(
      [](stk::mesh::BulkData& b, const SearchBoxes& tgt, const SearchBoxes& src) { return build_2d_list(b, tgt, src); },
      *bulk, selector, kN);
}

// =============================================================================
// Group 4 — Debug-only bound-check tests
// =============================================================================

// MUNDY_THROW_ASSERT fires only when NDEBUG is not defined. Tests that expect a
// throw are wrapped in #ifndef NDEBUG so they compile and pass in release builds.

#ifndef NDEBUG

TEST_F(NonPeriodicFixture, OutOfBounds_1d) {
  auto list = build_1d_list(*bulk_, search_boxes_, search_boxes_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  // target 0 has 2 neighbors; neighbor ordinal 2 is out of range.
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
}

TEST_F(NonPeriodicFixture, OutOfBounds_2d) {
  auto list = build_2d_list(*bulk_, search_boxes_, search_boxes_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
}

#endif  // NDEBUG

// =============================================================================
// Group 5 — Adversarial edge-case and invariant probes
// =============================================================================
//
// These tests target invariants and code paths not covered by Groups 1-4.
// Each probe is designed so that a real implementation defect causes a test
// failure rather than a silent pass.

// ---- Shared adversarial helpers (not used in Groups 1-4) ----

// Verify list.size() == sum of num_neighbors over all targets.
// For List1d, size() reads source_indices_.extent(0); for List2d it reads total_pairs_
// computed by parallel_reduce at construction.  Both must equal the sum of per-target
// counts accessed through num_neighbors().
template <typename ListType>
void check_size_equals_neighbor_sum(const ListType& list) {
  size_t manual_sum = 0;
  for (size_t t = 0; t < list.num_targets(); ++t) manual_sum += list.num_neighbors(t);
  EXPECT_EQ(list.size(), manual_sum) << "list.size() != sum of per-target neighbor counts.";
}

// Directly construct NeighborPair for every (target, neighbor_ordinal) and verify
// that every accessor returns a value consistent with the list's direct accessors.
template <typename ListType>
void check_neighbor_pair_accessors(const ListType& list) {
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      NeighborPair<ListType> pair(list, t, k);
      EXPECT_EQ(pair.target_index(), t);
      const size_t si = list.source_index(t, k);
      EXPECT_EQ(pair.source_index(), si) << "target " << t << " neighbor " << k;
      EXPECT_EQ(pair.target_entity(), list.target_entity(t)) << "target " << t << " neighbor " << k;
      EXPECT_EQ(pair.source_entity(), list.source_entity(si)) << "target " << t << " neighbor " << k;
      EXPECT_LT(si, list.num_sources()) << "source_index out of range at target " << t << " neighbor " << k;
    }
  }
}

// Directly construct Neighbors for every target and verify all accessors.
template <typename ListType>
void check_neighbors_accessors(const ListType& list) {
  for (size_t t = 0; t < list.num_targets(); ++t) {
    Neighbors<ListType> nbrs(list, t);
    EXPECT_EQ(nbrs.size(), list.num_neighbors(t)) << "target " << t;
    EXPECT_EQ(nbrs.target_entity(), list.target_entity(t)) << "target " << t;
    EXPECT_EQ(nbrs.target_index(), t) << "target " << t;
    for (size_t k = 0; k < nbrs.size(); ++k) {
      EXPECT_EQ(nbrs[k], list.get_neighbor(t, k)) << "target " << t << " neighbor " << k;
      EXPECT_EQ(nbrs(k), list.get_neighbor(t, k)) << "target " << t << " neighbor " << k;
      EXPECT_EQ(nbrs.source_index(k), list.source_index(t, k)) << "target " << t << " neighbor " << k;
    }
  }
}

// Use for_each_neighbor_pair to build a per-(target,source) visit-count matrix,
// then compare it against the direct-accessor pair set from collect_pairs.
// This verifies (a) no pair is duplicated by the iterator, (b) no pair is skipped,
// and (c) target_index()/source_index() on NeighborPair return the correct ordinals.
template <typename ListType>
void check_foreach_pair_matches_direct(const ListType& list) {
  const size_t nt = list.num_targets();
  const size_t ns = list.num_sources();
  Kokkos::View<int**, TestMemSpace> visit_count("adv_visit", nt, ns);
  Kokkos::deep_copy(visit_count, 0);
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const NeighborPair<ListType>& pair) {
        Kokkos::atomic_inc(&visit_count(pair.target_index(), pair.source_index()));
      });
  Kokkos::fence();
  std::set<std::pair<size_t, size_t>> fe_set;
  for (size_t t = 0; t < nt; ++t) {
    for (size_t s = 0; s < ns; ++s) {
      const int cnt = visit_count(t, s);
      if (cnt > 0) {
        EXPECT_EQ(cnt, 1) << "Pair (" << t << "," << s << ") visited " << cnt << " times (expected 1).";
        fe_set.insert({t, s});
      }
    }
  }
  EXPECT_EQ(fe_set, collect_pairs(list)) << "for_each pair set doesn't match direct-accessor pair set.";
}

// ---- Test 5.1: NoExcluder includes self-pairs ----
//
// All builds in Groups 1-4 apply ExcludeSelfInteraction.  The NoExcluder path
// is never exercised.  If the ArborX callback accidentally applies exclusion
// even without an excluder, self-pairs would be absent and size() would be 4
// instead of 8.
//
// Expected pairs with no excluder on the 4-box fixture (see fixture comment):
//   target 0: sources {0,1,2}  (self + two overlapping boxes)
//   target 1: sources {0,1}    (overlaps box 0 and self; box 2 is disjoint in x)
//   target 2: sources {0,2}    (overlaps box 0 and self; box 1 is disjoint in x)
//   target 3: sources {3}      (isolated — only overlaps itself)
// Total: 3+2+2+1 = 8 pairs.

TEST_F(NonPeriodicFixture, NoExcluder_IncludesSelfPairs_1d) {
  auto list = make_neighbor_list_builder<List1d>()
      .exec_space(TestExecSpace{})
      .target_input(search_boxes_)
      .source_input(search_boxes_)
      .build(*bulk_);
  EXPECT_EQ(list.size(), 8u) << "NoExcluder should include all 8 pairs (4 self + 4 cross).";
  EXPECT_EQ(list.num_neighbors(3), 1u) << "Isolated box must retain its self-pair without an excluder.";
  EXPECT_EQ(list.source_index(3, 0), 3u) << "Self-pair source ordinal for target 3 must be 3.";
  check_size_equals_neighbor_sum(list);
}

TEST_F(NonPeriodicFixture, NoExcluder_IncludesSelfPairs_2d) {
  auto list = make_neighbor_list_builder<List2d>()
      .exec_space(TestExecSpace{})
      .target_input(search_boxes_)
      .source_input(search_boxes_)
      .build(*bulk_);
  EXPECT_EQ(list.size(), 8u);
  EXPECT_EQ(list.num_neighbors(3), 1u);
  EXPECT_EQ(list.source_index(3, 0), 3u);
  check_size_equals_neighbor_sum(list);
}

// ---- Test 5.2: all-isolated geometry — empty pair set, full target visit ----
//
// for_each_target_with_neighbors must visit ALL targets even when every target has
// zero neighbors.  If the implementation changes its semantics to skip zero-neighbor
// targets (matching the misleading "with_neighbors" name literally), this test fails.

TEST(AdversarialSearch, AllIsolated_EmptyPairs_1d) {
  constexpr int kN = 3;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i) nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, i + 1);
  const std::vector<ArborX::Box> far_boxes = {
      make_arborx_box(0.0f, 0.0f, 0.0f, 0.1f),
      make_arborx_box(1000.0f, 0.0f, 0.0f, 0.1f),
      make_arborx_box(0.0f, 1000.0f, 0.0f, 0.1f),
  };
  const SearchBoxes sb = make_search_boxes(sel, far_boxes, nodes);
  const auto list = build_1d_list(*bulk, sb, sb);
  ASSERT_EQ(list.size(), 0u);
  for (size_t t = 0; t < list.num_targets(); ++t)
    EXPECT_EQ(list.num_neighbors(t), 0u) << "target " << t << " should be isolated.";
  Kokkos::View<size_t*, TestMemSpace> pair_count("iso_pairs", 1);
  Kokkos::deep_copy(pair_count, size_t(0));
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const NeighborPair<List1d>&) { Kokkos::atomic_inc(&pair_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(pair_count(0), 0u) << "for_each_neighbor_pair should fire zero times for an empty list.";
  Kokkos::View<size_t*, TestMemSpace> tgt_count("iso_tgts", 1);
  Kokkos::deep_copy(tgt_count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<List1d>&) { Kokkos::atomic_inc(&tgt_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(tgt_count(0), static_cast<size_t>(kN))
      << "for_each_target_with_neighbors must visit all targets even when they have zero neighbors.";
}

TEST(AdversarialSearch, AllIsolated_EmptyPairs_2d) {
  constexpr int kN = 3;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i) nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, i + 1);
  const std::vector<ArborX::Box> far_boxes = {
      make_arborx_box(0.0f, 0.0f, 0.0f, 0.1f),
      make_arborx_box(1000.0f, 0.0f, 0.0f, 0.1f),
      make_arborx_box(0.0f, 1000.0f, 0.0f, 0.1f),
  };
  const SearchBoxes sb = make_search_boxes(sel, far_boxes, nodes);
  const auto list = build_2d_list(*bulk, sb, sb);
  ASSERT_EQ(list.size(), 0u);
  for (size_t t = 0; t < list.num_targets(); ++t)
    EXPECT_EQ(list.num_neighbors(t), 0u) << "target " << t << " should be isolated.";
  Kokkos::View<size_t*, TestMemSpace> pair_count("iso2_pairs", 1);
  Kokkos::deep_copy(pair_count, size_t(0));
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const NeighborPair<List2d>&) { Kokkos::atomic_inc(&pair_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(pair_count(0), 0u);
  Kokkos::View<size_t*, TestMemSpace> tgt_count("iso2_tgts", 1);
  Kokkos::deep_copy(tgt_count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<List2d>&) { Kokkos::atomic_inc(&tgt_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(tgt_count(0), static_cast<size_t>(kN));
}

// ---- Test 5.3: NeighborPair accessor chain consistency ----
//
// Groups 1-4 only COUNT pairs via for_each; they never verify that the NeighborPair
// object returns correct entity/index values.  A bug in the offset arithmetic
// (e.g., off-by-one in pair_index()) would produce wrong source_entity values
// without affecting the pair count.

TEST_F(NonPeriodicFixture, NeighborPairAccessorChain_1d) {
  auto list = build_1d_list(*bulk_, search_boxes_, search_boxes_);
  check_neighbor_pair_accessors(list);
}

TEST_F(NonPeriodicFixture, NeighborPairAccessorChain_2d) {
  auto list = build_2d_list(*bulk_, search_boxes_, search_boxes_);
  check_neighbor_pair_accessors(list);
}

// ---- Test 5.4: Neighbors accessor consistency ----
//
// The Neighbors wrapper is constructed inside for_each_target_with_neighbors but
// is never directly tested.  A misalignment between Neighbors::operator[] and
// list.get_neighbor() would go undetected by the existing tests.

TEST_F(NonPeriodicFixture, NeighborsAccessorConsistency_1d) {
  auto list = build_1d_list(*bulk_, search_boxes_, search_boxes_);
  check_neighbors_accessors(list);
}

TEST_F(NonPeriodicFixture, NeighborsAccessorConsistency_2d) {
  auto list = build_2d_list(*bulk_, search_boxes_, search_boxes_);
  check_neighbors_accessors(list);
}

// ---- Test 5.5: for_each_neighbor_pair visits correct pairs, each exactly once ----
//
// The existing check_for_each_pair_count only verifies the total count.  This test
// also verifies which (target_idx, source_idx) pairs are visited, catching bugs
// where the iterator emits duplicate or swapped indices.

TEST_F(NonPeriodicFixture, ForEachPairMatchesDirectAccess_1d) {
  auto list = build_1d_list(*bulk_, search_boxes_, search_boxes_);
  check_foreach_pair_matches_direct(list);
}

TEST_F(NonPeriodicFixture, ForEachPairMatchesDirectAccess_2d) {
  auto list = build_2d_list(*bulk_, search_boxes_, search_boxes_);
  check_foreach_pair_matches_direct(list);
}

// ---- Test 5.6: size() equals the sum of per-target neighbor counts ----
//
// List1d returns size() from source_indices_.extent(0); List2d from total_pairs_
// computed via parallel_reduce in the constructor.  Both must agree with summing
// num_neighbors(t) for all t.  Also verified on a copy of each list to confirm
// that the plain-int total_pairs_ copy-construct path for List2d is correct.

TEST_F(NonPeriodicFixture, SizeEqualsNeighborSum_1d) {
  auto list = build_1d_list(*bulk_, search_boxes_, search_boxes_);
  check_size_equals_neighbor_sum(list);
  auto copy = list;
  check_size_equals_neighbor_sum(copy);
}

TEST_F(NonPeriodicFixture, SizeEqualsNeighborSum_2d) {
  auto list = build_2d_list(*bulk_, search_boxes_, search_boxes_);
  check_size_equals_neighbor_sum(list);
  auto copy = list;
  check_size_equals_neighbor_sum(copy);
}

// ---- Test 5.7: ExcludeSymmetricDuplicates suppresses exactly one orientation ----
//
// ExcludeSymmetricDuplicates is completely absent from Groups 1-4.  The excluder
// marks buckets in the target ∩ source selector intersection, then suppresses the
// pair (t,s) when src_entity < trg_entity.  With the universal selector the
// intersection is the full mesh, so the rule applies to every pair.
//
// 4-box fixture, ExcludeSelf + ExcludeSymDup:
//   Full pair set (ExcludeSelf only): {(0,1),(1,0),(0,2),(2,0)}
//   Suppressed (src < trg):           {(1,0),(2,0)}  — source nodes 1,2 < target nodes 2,3
//   Retained (src > trg):             {(0,1),(0,2)}  — source nodes 2,3 > target node 1
//
// The test verifies size()==2 and, for every retained pair, that
// source_entity > target_entity (the property guaranteed by the excluder).

TEST_F(NonPeriodicFixture, ExcludeSymmetricDuplicates_1d) {
  auto list = make_neighbor_list_builder<List1d>()
      .exec_space(TestExecSpace{})
      .target_input(search_boxes_)
      .source_input(search_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*bulk_);
  EXPECT_EQ(list.size(), 2u) << "ExcludeSymmetricDuplicates should halve the 4-pair set to 2.";
  check_size_equals_neighbor_sum(list);
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const stk::mesh::Entity trg = list.target_entity(t);
      const stk::mesh::Entity src = list.source_entity(list.source_index(t, k));
      EXPECT_GT(src, trg) << "ExcludeSymmetricDuplicates must retain only pairs where src_entity > trg_entity.";
    }
  }
}

TEST_F(NonPeriodicFixture, ExcludeSymmetricDuplicates_2d) {
  auto list = make_neighbor_list_builder<List2d>()
      .exec_space(TestExecSpace{})
      .target_input(search_boxes_)
      .source_input(search_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*bulk_);
  EXPECT_EQ(list.size(), 2u);
  check_size_equals_neighbor_sum(list);
  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const stk::mesh::Entity trg = list.target_entity(t);
      const stk::mesh::Entity src = list.source_entity(list.source_index(t, k));
      EXPECT_GT(src, trg) << "ExcludeSymmetricDuplicates must retain only pairs where src_entity > trg_entity.";
    }
  }
}

}  // namespace
}  // namespace search
}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX
