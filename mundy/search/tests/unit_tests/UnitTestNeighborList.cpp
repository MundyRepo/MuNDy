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
///   Group 1 — Construction: default construction, copy/move semantics.
///   Group 2 — Deterministic geometry: 6-node mesh with 3 STK parts enables 4 selector
///     configurations (Universal, Disjoint, Overlapping, IdSubset) each tested against
///     4 excluder variants (NoExcluder, ExcludeSelf, ExcludeSymDup, ExcludeSelf+ExcludeSymDup)
///     for both List1d and List2d — 32 tests total.  Every test calls verify_exact_pair_set,
///     which checks the exact pair set AND five structural invariants.
///   Group 3 — Random N²: oracle comparison on 50 Philox-generated random spheres.
///   Group 4 — Iteration protocol: for_each_target_with_neighbors must visit every target
///     even when it has zero neighbors.
///   Group 5 — Debug bounds: MUNDY_THROW_ASSERT out-of-range checks (guarded by #ifndef NDEBUG).

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// External
#include <gtest/gtest.h>

// C++ core
#include <cstddef>
#include <initializer_list>
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

static_assert(NeighborListType<ArborX1dNeighborList<Kokkos::HostSpace>>,
              "ArborX1dNeighborList<HostSpace> must satisfy NeighborListType.");
static_assert(NeighborListType<ArborX2dNeighborList<Kokkos::HostSpace>>,
              "ArborX2dNeighborList<HostSpace> must satisfy NeighborListType.");

// =============================================================================
// Type aliases
// =============================================================================

using TestMemSpace = Kokkos::HostSpace;
using TestExecSpace = Kokkos::DefaultHostExecutionSpace;
using SearchBoxes = impl::ArborXSearchBoxesT<TestMemSpace>;
using List1d = ArborX1dNeighborList<TestMemSpace>;
using List2d = ArborX2dNeighborList<TestMemSpace>;
using PairSet = std::set<std::pair<size_t, size_t>>;

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

// Create a minimal node-only mesh with nodes numbered 1..num_nodes, no extra parts.
std::pair<std::shared_ptr<stk::mesh::MetaData>, std::unique_ptr<stk::mesh::BulkData>> make_node_mesh(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta_ptr = builder.create_meta_data();
  meta_ptr->use_simple_fields();
  auto bulk_ptr = builder.create(meta_ptr);
  meta_ptr->commit();
  bulk_ptr->modification_begin();
  for (int id = 1; id <= num_nodes; ++id) bulk_ptr->declare_node(id);
  bulk_ptr->modification_end();
  return {std::move(meta_ptr), std::move(bulk_ptr)};
}

// =============================================================================
// Search-box construction
// =============================================================================

SearchBoxes make_search_boxes(const stk::mesh::Selector& selector, const std::vector<ArborX::Box>& boxes_h,
                              const std::vector<stk::mesh::Entity>& entities_h) {
  const size_t n = boxes_h.size();
  EXPECT_EQ(n, entities_h.size());
  Kokkos::View<ArborX::Box*, TestMemSpace> boxes("boxes", n);
  Kokkos::View<stk::mesh::Entity*, TestMemSpace> entities("entities", n);
  for (size_t i = 0; i < n; ++i) {
    boxes(i) = boxes_h[i];
    entities(i) = entities_h[i];
  }
  return SearchBoxes{selector, boxes, entities};
}

// =============================================================================
// Pair collection
// =============================================================================

template <typename ListType>
PairSet collect_pairs(const ListType& list) {
  PairSet result;
  for (size_t t = 0; t < list.num_targets(); ++t)
    for (size_t k = 0; k < list.num_neighbors(t); ++k)
      result.insert({t, list.source_index(t, k)});
  return result;
}

// Oracle for the random-N² test: overlapping boxes, excluding self (ordinal equality = entity equality
// when the target and source arrays hold the same node sequence).
PairSet oracle_pairs_no_self(const std::vector<ArborX::Box>& target_boxes,
                             const std::vector<ArborX::Box>& source_boxes) {
  PairSet pairs;
  for (size_t t = 0; t < target_boxes.size(); ++t)
    for (size_t s = 0; s < source_boxes.size(); ++s)
      if (t != s && boxes_overlap(target_boxes[t], source_boxes[s]))
        pairs.insert({t, s});
  return pairs;
}

// =============================================================================
// Shared invariant checks (also used standalone in Group 4/5)
// =============================================================================

template <typename ListType>
void check_size_equals_neighbor_sum(const ListType& list) {
  size_t manual_sum = 0;
  for (size_t t = 0; t < list.num_targets(); ++t) manual_sum += list.num_neighbors(t);
  EXPECT_EQ(list.size(), manual_sum) << "list.size() != sum of per-target neighbor counts.";
}

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

// Verifies that for_each_neighbor_pair emits exactly the same pairs as the direct accessor,
// each exactly once.
template <typename ListType>
void check_foreach_pair_matches_direct(const ListType& list) {
  const size_t nt = list.num_targets();
  const size_t ns = list.num_sources();
  if (nt == 0 || ns == 0) {
    EXPECT_EQ(list.size(), 0u);
    return;
  }
  Kokkos::View<int**, TestMemSpace> visit_count("visit_count", nt, ns);
  Kokkos::deep_copy(visit_count, 0);
  mundy::search::for_each_neighbor_pair(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const NeighborPair<ListType>& pair) {
        Kokkos::atomic_inc(&visit_count(pair.target_index(), pair.source_index()));
      });
  Kokkos::fence();
  PairSet fe_set;
  for (size_t t = 0; t < nt; ++t)
    for (size_t s = 0; s < ns; ++s) {
      const int cnt = visit_count(t, s);
      if (cnt > 0) {
        EXPECT_EQ(cnt, 1) << "Pair (" << t << "," << s << ") visited " << cnt << " times (expected 1).";
        fe_set.insert({t, s});
      }
    }
  EXPECT_EQ(fe_set, collect_pairs(list)) << "for_each pair set doesn't match direct-accessor pair set.";
}

// =============================================================================
// Canonical verifier: exact pair set + all structural invariants.
// =============================================================================

template <typename ListType>
void verify_exact_pair_set(const ListType& list, const PairSet& expected) {
  EXPECT_EQ(list.size(), expected.size());
  EXPECT_EQ(collect_pairs(list), expected);
  check_size_equals_neighbor_sum(list);
  check_neighbor_pair_accessors(list);
  check_neighbors_accessors(list);
  check_foreach_pair_matches_direct(list);
}

// =============================================================================
// DeterministicFixture — 6-node mesh with 3 STK parts, 4 selector configurations
// =============================================================================
//
// Nodes and boxes (0-indexed array ordinals):
//   Ord 0  (node 1, target_part):  box0 = [-2,   2]³       (center 0,    half 2.0)
//   Ord 1  (node 2, target_part):  box1 = isolated @ 100   (center 100,  half 0.5)
//   Ord 2  (node 3, source_part):  box2 = [ 1.5, 3.5]³     (center 2.5,  half 1.0)
//   Ord 3  (node 4, source_part):  box3 = isolated @ 200   (center 200,  half 0.5)
//   Ord 4  (node 5, shared_part):  box4 = [-0.5, 2.5]³     (center 1.0,  half 1.5)
//   Ord 5  (node 6, shared_part):  box5 = [ 0.5, 3.5]³     (center 2.0,  half 1.5)
//
// Non-self overlapping undirected pairs (global ordinals): {0↔2, 0↔4, 0↔5, 2↔4, 2↔5, 4↔5}
// Isolated nodes (no cross-overlaps): ordinals 1 and 3.
//
// Selector configurations and their array ordinals:
//   Universal:   target = {0,1,2,3,4,5},  source = {0,1,2,3,4,5}
//   Disjoint:    target = {0,1},           source = {2,3}
//   Overlapping: target = {0,1,4,5},       source = {2,3,4,5}
//   IdSubset:    target = {4,5},           source = {4,5}
//
// For Overlapping and IdSubset the source and target arrays are different, so
// local array ordinals differ from global mesh ordinals.  Expected pair comments
// annotate each entry with the global node IDs involved.
class DeterministicFixture : public ::testing::Test {
 public:
  void SetUp() override {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();

    stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    meta_ = builder.create_meta_data();
    meta_->use_simple_fields();
    target_part_ = &meta_->declare_part("target_part", stk::topology::NODE_RANK);
    source_part_ = &meta_->declare_part("source_part", stk::topology::NODE_RANK);
    shared_part_ = &meta_->declare_part("shared_part", stk::topology::NODE_RANK);
    bulk_ = builder.create(meta_);
    meta_->commit();

    bulk_->modification_begin();
    for (int id = 1; id <= 6; ++id) bulk_->declare_node(id);
    for (int id : {1, 2}) {
      auto n = bulk_->get_entity(stk::topology::NODE_RANK, id);
      bulk_->change_entity_parts(n, stk::mesh::PartVector{target_part_}, stk::mesh::PartVector{});
    }
    for (int id : {3, 4}) {
      auto n = bulk_->get_entity(stk::topology::NODE_RANK, id);
      bulk_->change_entity_parts(n, stk::mesh::PartVector{source_part_}, stk::mesh::PartVector{});
    }
    for (int id : {5, 6}) {
      auto n = bulk_->get_entity(stk::topology::NODE_RANK, id);
      bulk_->change_entity_parts(n, stk::mesh::PartVector{shared_part_}, stk::mesh::PartVector{});
    }
    bulk_->modification_end();

    nodes_.resize(6);
    for (int id = 1; id <= 6; ++id) {
      nodes_[id - 1] = bulk_->get_entity(stk::topology::NODE_RANK, id);
      ASSERT_TRUE(bulk_->is_valid(nodes_[id - 1])) << "Node " << id << " not found.";
    }

    all_boxes_ = {
        make_arborx_box(0.0f, 0.0f, 0.0f, 2.0f),       // ord 0: [-2, 2]³
        make_arborx_box(100.f, 100.f, 100.f, 0.5f),     // ord 1: isolated
        make_arborx_box(2.5f, 2.5f, 2.5f, 1.0f),        // ord 2: [1.5, 3.5]³
        make_arborx_box(200.f, 200.f, 200.f, 0.5f),     // ord 3: isolated
        make_arborx_box(1.0f, 1.0f, 1.0f, 1.5f),        // ord 4: [-0.5, 2.5]³
        make_arborx_box(2.0f, 2.0f, 2.0f, 1.5f),        // ord 5: [0.5, 3.5]³
    };

    const stk::mesh::Selector sel_all     = meta_->universal_part();
    const stk::mesh::Selector sel_tgt     = *target_part_;
    const stk::mesh::Selector sel_src     = *source_part_;
    const stk::mesh::Selector sel_shr     = *shared_part_;
    const stk::mesh::Selector sel_tgt_shr = *target_part_ | *shared_part_;
    const stk::mesh::Selector sel_src_shr = *source_part_ | *shared_part_;

    // Universal: all 6 boxes, all 6 nodes, universal selector.
    universal_boxes_ = make_search_boxes(sel_all, all_boxes_, nodes_);

    // Disjoint: target={node1,node2}, source={node3,node4}.
    disjoint_target_boxes_ = make_search_boxes(
        sel_tgt, {all_boxes_[0], all_boxes_[1]}, {nodes_[0], nodes_[1]});
    disjoint_source_boxes_ = make_search_boxes(
        sel_src, {all_boxes_[2], all_boxes_[3]}, {nodes_[2], nodes_[3]});

    // Overlapping: target={node1,node2,node5,node6}, source={node3,node4,node5,node6}.
    overlapping_target_boxes_ = make_search_boxes(
        sel_tgt_shr,
        {all_boxes_[0], all_boxes_[1], all_boxes_[4], all_boxes_[5]},
        {nodes_[0], nodes_[1], nodes_[4], nodes_[5]});
    overlapping_source_boxes_ = make_search_boxes(
        sel_src_shr,
        {all_boxes_[2], all_boxes_[3], all_boxes_[4], all_boxes_[5]},
        {nodes_[2], nodes_[3], nodes_[4], nodes_[5]});

    // IdSubset: target={node5,node6}, source={node5,node6} — identical subset.
    idsubset_boxes_ = make_search_boxes(
        sel_shr, {all_boxes_[4], all_boxes_[5]}, {nodes_[4], nodes_[5]});
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Part* target_part_ = nullptr;
  stk::mesh::Part* source_part_ = nullptr;
  stk::mesh::Part* shared_part_ = nullptr;
  std::vector<stk::mesh::Entity> nodes_;
  std::vector<ArborX::Box> all_boxes_;
  SearchBoxes universal_boxes_;
  SearchBoxes disjoint_target_boxes_;
  SearchBoxes disjoint_source_boxes_;
  SearchBoxes overlapping_target_boxes_;
  SearchBoxes overlapping_source_boxes_;
  SearchBoxes idsubset_boxes_;
};

// =============================================================================
// Group 1 — Construction and access
// =============================================================================

TEST(ArborX1dNeighborList, DefaultConstruct) {
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

// Kokkos views are reference-counted: copy yields a shallow copy sharing the same data.
TEST_F(DeterministicFixture, CopyMove_1d) {
  auto original = make_neighbor_list_builder<List1d>()
      .exec_space(TestExecSpace{})
      .target_input(universal_boxes_)
      .source_input(universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk_);
  const size_t nt = original.num_targets();
  const size_t ns = original.num_sources();
  const size_t sz = original.size();

  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(), sz);

  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(), sz);

  List1d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.num_sources(), ns);
  EXPECT_EQ(copy_assign.size(), sz);

  List1d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.num_sources(), ns);
  EXPECT_EQ(move_assign.size(), sz);
}

TEST_F(DeterministicFixture, CopyMove_2d) {
  auto original = make_neighbor_list_builder<List2d>()
      .exec_space(TestExecSpace{})
      .target_input(universal_boxes_)
      .source_input(universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk_);
  const size_t nt = original.num_targets();
  const size_t ns = original.num_sources();
  const size_t sz = original.size();

  auto copy_ctor = original;
  EXPECT_EQ(copy_ctor.num_targets(), nt);
  EXPECT_EQ(copy_ctor.num_sources(), ns);
  EXPECT_EQ(copy_ctor.size(), sz);

  auto move_ctor = std::move(copy_ctor);
  EXPECT_EQ(move_ctor.num_targets(), nt);
  EXPECT_EQ(move_ctor.num_sources(), ns);
  EXPECT_EQ(move_ctor.size(), sz);

  List2d copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.num_sources(), ns);
  EXPECT_EQ(copy_assign.size(), sz);

  List2d move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.num_sources(), ns);
  EXPECT_EQ(move_assign.size(), sz);
}

// =============================================================================
// Group 2 — Deterministic pair-set tests (4 selectors × 4 excluders × 2 list types)
// =============================================================================
//
// Expected pair notation: (target_array_ordinal, source_array_ordinal).
//
// Universal (target={0..5}, source={0..5}):
//   Cross overlapping undirected: {0↔2, 0↔4, 0↔5, 2↔4, 2↔5, 4↔5}
//   NoExcluder        (18): 6 self + 12 directed cross
//   ExcludeSelf       (12): 12 directed cross
//   ExcludeSymDup     (12): 6 self + 6 cross where s_ord > t_ord
//   ExcludeSelf+SymDup (6): 6 cross where s_ord > t_ord
//
// Disjoint (target={node1,node2} ords {0,1}, source={node3,node4} ords {0,1}):
//   Only box0↔box2 overlaps (node1↔node3).  Intersection of selectors is empty.
//   All 4 excluder variants: {(0,0)} — 1 pair.
//
// Overlapping (target={node1,node2,node5,node6} ords {0,1,2,3},
//              source={node3,node4,node5,node6} ords {0,1,2,3}):
//   Intersection = shared_part = {node5,node6} = target ords {2,3} / source ords {2,3}.
//   NoExcluder        (9): {(0,0),(0,2),(0,3),(2,0),(2,2),(2,3),(3,0),(3,2),(3,3)}
//   ExcludeSelf       (7): remove self-pairs (2,2),(3,3)
//   ExcludeSymDup     (8): remove (3,2) [trg=node6, src=node5, src<trg]
//   ExcludeSelf+SymDup(6): remove (2,2),(3,3),(3,2)
//
// IdSubset (target={node5,node6} ords {0,1}, source={node5,node6} ords {0,1}):
//   Intersection = shared_part = all nodes in both arrays.
//   NoExcluder        (4): {(0,0),(0,1),(1,0),(1,1)}
//   ExcludeSelf       (2): {(0,1),(1,0)}
//   ExcludeSymDup     (3): remove (1,0) [trg=node6, src=node5, src<trg]
//   ExcludeSelf+SymDup(1): {(0,1)}

// ---- Universal ----

template <typename ListType>
void test_universal_no_excluder(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
      {0, 2}, {2, 0}, {0, 4}, {4, 0}, {0, 5}, {5, 0},
      {2, 4}, {4, 2}, {2, 5}, {5, 2}, {4, 5}, {5, 4}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_universal_self(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {
      {0, 2}, {2, 0}, {0, 4}, {4, 0}, {0, 5}, {5, 0},
      {2, 4}, {4, 2}, {2, 5}, {5, 2}, {4, 5}, {5, 4}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_universal_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  // Universal intersection = all nodes; suppress (t,s) where src_entity < trg_entity.
  // For sequential node creation, entity(n1) < entity(n2) iff node-id(n1) < node-id(n2),
  // which equals s_ord < t_ord in the universal array.
  const PairSet expected = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
      {0, 2}, {0, 4}, {0, 5}, {2, 4}, {2, 5}, {4, 5}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_universal_self_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 2}, {0, 4}, {0, 5}, {2, 4}, {2, 5}, {4, 5}};
  verify_exact_pair_set(list, expected);
}

// ---- Disjoint ----

template <typename ListType>
void test_disjoint_no_excluder(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};  // node1 (box0) ↔ node3 (box2)
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_disjoint_self(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};  // disjoint nodes: no self-pairs to remove
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_disjoint_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};  // intersection = empty; ExcludeSymDup never fires
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_disjoint_self_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};
  verify_exact_pair_set(list, expected);
}

// ---- Overlapping ----
// Target local ords: 0=node1, 1=node2, 2=node5, 3=node6
// Source local ords: 0=node3, 1=node4, 2=node5, 3=node6

template <typename ListType>
void test_overlapping_no_excluder(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {
      {0, 0},  // node1↔node3
      {0, 2},  // node1↔node5
      {0, 3},  // node1↔node6
      {2, 0},  // node5↔node3
      {2, 2},  // node5 self
      {2, 3},  // node5↔node6
      {3, 0},  // node6↔node3
      {3, 2},  // node6↔node5
      {3, 3},  // node6 self
  };
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_overlapping_self(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}, {0, 2}, {0, 3}, {2, 0}, {2, 3}, {3, 0}, {3, 2}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_overlapping_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  // Intersection = shared_part = {node5, node6}.
  // Only pair where BOTH entities are in shared_part AND src<trg: (3,2) [trg=node6, src=node5].
  const PairSet expected = {{0, 0}, {0, 2}, {0, 3}, {2, 0}, {2, 2}, {2, 3}, {3, 0}, {3, 3}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_overlapping_self_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}, {0, 2}, {0, 3}, {2, 0}, {2, 3}, {3, 0}};
  verify_exact_pair_set(list, expected);
}

// ---- IdSubset ----
// Target local ords: 0=node5, 1=node6
// Source local ords: 0=node5, 1=node6

template <typename ListType>
void test_idsubset_no_excluder(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_idsubset_self(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 1}, {1, 0}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_idsubset_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  // Both arrays are shared_part; intersection = shared_part = all nodes here.
  // Suppress (1,0): trg=node6, src=node5, src_entity < trg_entity.
  const PairSet expected = {{0, 0}, {0, 1}, {1, 1}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType>
void test_idsubset_self_symdups(DeterministicFixture& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 1}};
  verify_exact_pair_set(list, expected);
}

// TEST_F wrappers — one 1d/2d pair per template helper (32 tests total).

// clang-format off
#define MAKE_DET_TESTS(selector, excluder_tag)                                              \
  TEST_F(DeterministicFixture, selector##_##excluder_tag##_1d) {                           \
    test_##selector##_##excluder_tag<List1d>(*this);                                        \
  }                                                                                         \
  TEST_F(DeterministicFixture, selector##_##excluder_tag##_2d) {                           \
    test_##selector##_##excluder_tag<List2d>(*this);                                        \
  }

MAKE_DET_TESTS(universal,   no_excluder)
MAKE_DET_TESTS(universal,   self)
MAKE_DET_TESTS(universal,   symdups)
MAKE_DET_TESTS(universal,   self_symdups)
MAKE_DET_TESTS(disjoint,    no_excluder)
MAKE_DET_TESTS(disjoint,    self)
MAKE_DET_TESTS(disjoint,    symdups)
MAKE_DET_TESTS(disjoint,    self_symdups)
MAKE_DET_TESTS(overlapping, no_excluder)
MAKE_DET_TESTS(overlapping, self)
MAKE_DET_TESTS(overlapping, symdups)
MAKE_DET_TESTS(overlapping, self_symdups)
MAKE_DET_TESTS(idsubset,    no_excluder)
MAKE_DET_TESTS(idsubset,    self)
MAKE_DET_TESTS(idsubset,    symdups)
MAKE_DET_TESTS(idsubset,    self_symdups)

#undef MAKE_DET_TESTS
// clang-format on

// =============================================================================
// Group 3 — Random N² correctness validation
// =============================================================================

// Place N spheres at Philox-generated centers; build with ExcludeSelfInteraction;
// verify every list pair corresponds to overlapping boxes and every oracle pair appears.
template <typename ListType, typename BuildFn>
void run_random_n2_validation(BuildFn build_fn, stk::mesh::BulkData& bulk,
                              const stk::mesh::Selector& selector, int num_nodes) {
  constexpr size_t kSeed = 42;
  constexpr float kDomainSize = 10.0f;
  constexpr float kRadius = 0.9f;

  std::vector<stk::mesh::Entity> nodes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i] = bulk.get_entity(stk::topology::NODE_RANK, i + 1);
    ASSERT_TRUE(bulk.is_valid(nodes[i])) << "Node " << (i + 1) << " not found.";
  }

  std::vector<ArborX::Box> boxes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    const float cx = rng.uniform<float>(0.0f, kDomainSize);
    const float cy = rng.uniform<float>(0.0f, kDomainSize);
    const float cz = rng.uniform<float>(0.0f, kDomainSize);
    boxes[i] = make_arborx_box(cx, cy, cz, kRadius);
  }

  SearchBoxes sb = make_search_boxes(selector, boxes, nodes);
  ListType list = build_fn(bulk, sb, sb);

  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t s = list.source_index(t, k);
      EXPECT_TRUE(boxes_overlap(boxes[t], boxes[s]))
          << "Spurious pair (target=" << t << ", source=" << s << "): boxes do not overlap.";
      EXPECT_NE(t, s) << "Self-pair (target==source==" << t << ") despite ExcludeSelfInteraction.";
    }
  }

  const auto expected = oracle_pairs_no_self(boxes, boxes);
  const auto actual = collect_pairs(list);
  EXPECT_EQ(actual, expected) << "Neighbor list is missing oracle pairs or contains extra pairs.";
}

TEST(ArborX1dNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  run_random_n2_validation<List1d>(
      [](stk::mesh::BulkData& b, const SearchBoxes& tgt, const SearchBoxes& src) {
        return make_neighbor_list_builder<List1d>()
            .exec_space(TestExecSpace{})
            .target_input(tgt)
            .source_input(src)
            .exclude(ExcludeSelfInteraction{})
            .build(b);
      },
      *bulk, meta->universal_part(), kN);
}

TEST(ArborX2dNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  run_random_n2_validation<List2d>(
      [](stk::mesh::BulkData& b, const SearchBoxes& tgt, const SearchBoxes& src) {
        return make_neighbor_list_builder<List2d>()
            .exec_space(TestExecSpace{})
            .target_input(tgt)
            .source_input(src)
            .exclude(ExcludeSelfInteraction{})
            .build(b);
      },
      *bulk, meta->universal_part(), kN);
}

// =============================================================================
// Group 4 — Iteration protocol
// =============================================================================
//
// for_each_target_with_neighbors must visit ALL targets, including those with zero
// neighbors.  A literal reading of "with_neighbors" might suggest skipping
// zero-neighbor targets; this test guards against that regression.

TEST(IterationProtocol, AllIsolated_VisitsAllTargets_1d) {
  constexpr int kN = 3;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i) nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, i + 1);
  const std::vector<ArborX::Box> far_boxes = {
      make_arborx_box(0.f, 0.f, 0.f, 0.1f),
      make_arborx_box(1000.f, 0.f, 0.f, 0.1f),
      make_arborx_box(0.f, 1000.f, 0.f, 0.1f),
  };
  const SearchBoxes sb = make_search_boxes(sel, far_boxes, nodes);
  const auto list = make_neighbor_list_builder<List1d>()
      .exec_space(TestExecSpace{})
      .target_input(sb)
      .source_input(sb)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk);

  ASSERT_EQ(list.size(), 0u);

  Kokkos::View<size_t*, TestMemSpace> tgt_count("tgt_count", 1);
  Kokkos::deep_copy(tgt_count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const Neighbors<List1d>&) { Kokkos::atomic_inc(&tgt_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(tgt_count(0), static_cast<size_t>(kN))
      << "for_each_target_with_neighbors must visit all " << kN << " targets even with zero neighbors.";
}

TEST(IterationProtocol, AllIsolated_VisitsAllTargets_2d) {
  constexpr int kN = 3;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i) nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, i + 1);
  const std::vector<ArborX::Box> far_boxes = {
      make_arborx_box(0.f, 0.f, 0.f, 0.1f),
      make_arborx_box(1000.f, 0.f, 0.f, 0.1f),
      make_arborx_box(0.f, 1000.f, 0.f, 0.1f),
  };
  const SearchBoxes sb = make_search_boxes(sel, far_boxes, nodes);
  const auto list = make_neighbor_list_builder<List2d>()
      .exec_space(TestExecSpace{})
      .target_input(sb)
      .source_input(sb)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk);

  ASSERT_EQ(list.size(), 0u);

  Kokkos::View<size_t*, TestMemSpace> tgt_count("tgt_count2", 1);
  Kokkos::deep_copy(tgt_count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const Neighbors<List2d>&) { Kokkos::atomic_inc(&tgt_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(tgt_count(0), static_cast<size_t>(kN));
}

// =============================================================================
// Group 5 — Debug bounds
// =============================================================================
//
// MUNDY_THROW_ASSERT fires only when NDEBUG is not defined.

#ifndef NDEBUG

TEST_F(DeterministicFixture, OutOfBounds_1d) {
  auto list = make_neighbor_list_builder<List1d>()
      .exec_space(TestExecSpace{})
      .target_input(universal_boxes_)
      .source_input(universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
}

TEST_F(DeterministicFixture, OutOfBounds_2d) {
  auto list = make_neighbor_list_builder<List2d>()
      .exec_space(TestExecSpace{})
      .target_input(universal_boxes_)
      .source_input(universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
}

#endif  // NDEBUG

}  // namespace
}  // namespace search
}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX
