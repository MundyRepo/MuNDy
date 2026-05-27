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
/// \brief Unit tests for ArborX1dNeighborList, ArborX2dNeighborList, and STKSearchNeighborList.
///
/// The test infrastructure is parameterized via a BoxTrait template argument so that
/// ArborX and STK list types share the same deterministic fixture, helper functions, and
/// macro-generated test cases without duplication.
///
/// Test structure:
///   Group 0 — Compile-time concept checks (NeighborListType concept).
///   Group 1 — Construction: default construction, copy/move semantics.
///   Group 2 — Deterministic geometry: 6-node mesh with 3 STK parts enables 4 selector
///     configurations (Universal, Disjoint, Overlapping, IdSubset) each tested against
///     4 excluder variants (NoExcluder, ExcludeSelf, ExcludeSymDup, ExcludeSelf+ExcludeSymDup)
///     for each list type (1d, 2d, stk) — 48 tests total.  Every test calls
///     verify_exact_pair_set, which checks the exact pair set AND five structural invariants.
///   Group 3 — Random N²: oracle comparison on 50 Philox-generated random spheres.
///   Group 4 — Iteration protocol: for_each_target_with_neighbors must visit every target
///     even when it has zero neighbors.
///   Group 5 — Debug bounds: MUNDY_THROW_ASSERT out-of-range checks (guarded by #ifndef NDEBUG).
///   Group 6 — Reduction functions: for_each_neighbor_pair_reduce and
///     for_each_target_with_neighbors_reduce.

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

// STK mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>

// STK search
#include <stk_search/Box.hpp>

// Mundy config
#include <MundySearch_config.hpp>

// Mundy search — common
#include <mundy_search/Excluder.hpp>
#include <mundy_search/ForEach.hpp>
#include <mundy_search/NeighborListBuilder.hpp>
#include <mundy_search/Neighbors.hpp>

// Mundy search — STK (always available)
#include <mundy_search/STKSearchNeighborList.hpp>
#include <mundy_search/impl/STKSearchBoxes.hpp>

// Mundy utils
#include <mundy_utils/rng.hpp>  // for mundy::make_philox

#ifdef HAVE_MUNDYSEARCH_ARBORX
// ArborX
#include <ArborX.hpp>
// Mundy search — ArborX
#include <mundy_search/ArborX1dNeighborList.hpp>
#include <mundy_search/ArborX2dNeighborList.hpp>
#include <mundy_search/impl/ArborXSearchBoxes.hpp>
#endif

namespace mundy {
namespace search {
namespace {

// =============================================================================
// Execution / memory space aliases
// =============================================================================

using TestMemSpace  = Kokkos::HostSpace;
using TestExecSpace = Kokkos::DefaultHostExecutionSpace;

// =============================================================================
// List type aliases and compile-time concept checks (Group 0)
// =============================================================================

using STKList = STKSearchNeighborList<TestMemSpace>;
static_assert(NeighborListType<STKList>, "STKSearchNeighborList<HostSpace> must satisfy NeighborListType.");

#ifdef HAVE_MUNDYSEARCH_ARBORX
using List1d = ArborX1dNeighborList<TestMemSpace>;
using List2d = ArborX2dNeighborList<TestMemSpace>;
static_assert(NeighborListType<List1d>, "ArborX1dNeighborList<HostSpace> must satisfy NeighborListType.");
static_assert(NeighborListType<List2d>, "ArborX2dNeighborList<HostSpace> must satisfy NeighborListType.");
#endif

using PairSet = std::set<std::pair<size_t, size_t>>;

// =============================================================================
// Box traits
//
// Each trait bundles the native box type, the matching SearchBoxes container,
// a box factory, an overlap predicate, and a search-box constructor.  All test
// helpers are parameterized via these traits so no logic is duplicated between
// the ArborX and STK paths.
// =============================================================================

#ifdef HAVE_MUNDYSEARCH_ARBORX
/// Trait for ArborX-backed tests (ArborX::Box, impl::ArborXSearchBoxesT).
struct ArborXBoxTrait {
  using box_type         = ArborX::Box;
  using search_boxes_type = impl::ArborXSearchBoxesT<TestMemSpace>;

  static box_type make(float cx, float cy, float cz, float hx, float hy, float hz) {
    return ArborX::Box{ArborX::Point{cx - hx, cy - hy, cz - hz},
                       ArborX::Point{cx + hx, cy + hy, cz + hz}};
  }
  static box_type make(float cx, float cy, float cz, float h) { return make(cx, cy, cz, h, h, h); }

  static bool overlap(const box_type& a, const box_type& b) {
    for (int d = 0; d < 3; ++d) {
      if (a.maxCorner()[d] < b.minCorner()[d]) return false;
      if (b.maxCorner()[d] < a.minCorner()[d]) return false;
    }
    return true;
  }

  static search_boxes_type make_search_boxes(const stk::mesh::Selector&          sel,
                                             const std::vector<box_type>&         bh,
                                             const std::vector<stk::mesh::Entity>& eh) {
    const size_t n = bh.size();
    EXPECT_EQ(n, eh.size());
    Kokkos::View<box_type*, TestMemSpace>          bv("boxes", n);
    Kokkos::View<stk::mesh::Entity*, TestMemSpace> ev("entities", n);
    for (size_t i = 0; i < n; ++i) { bv(i) = bh[i]; ev(i) = eh[i]; }
    return search_boxes_type{sel, bv, ev};
  }
};
#endif  // HAVE_MUNDYSEARCH_ARBORX

/// Trait for STK-backed tests (impl::STKSearchBoxesT).
struct STKBoxTrait {
  using search_boxes_type = impl::STKSearchBoxesT<TestMemSpace>;
  using box_type          = typename search_boxes_type::box_type;

  static box_type make(float cx, float cy, float cz, float hx, float hy, float hz) {
    return box_type{cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz};
  }
  static box_type make(float cx, float cy, float cz, float h) { return make(cx, cy, cz, h, h, h); }

  static bool overlap(const box_type& a, const box_type& b) {
    if (a.get_x_max() < b.get_x_min() || b.get_x_max() < a.get_x_min()) return false;
    if (a.get_y_max() < b.get_y_min() || b.get_y_max() < a.get_y_min()) return false;
    if (a.get_z_max() < b.get_z_min() || b.get_z_max() < a.get_z_min()) return false;
    return true;
  }

  static search_boxes_type make_search_boxes(const stk::mesh::Selector&          sel,
                                             const std::vector<box_type>&         bh,
                                             const std::vector<stk::mesh::Entity>& eh) {
    const size_t n = bh.size();
    EXPECT_EQ(n, eh.size());
    Kokkos::View<box_type*, TestMemSpace>          bv("boxes", n);
    Kokkos::View<stk::mesh::Entity*, TestMemSpace> ev("entities", n);
    for (size_t i = 0; i < n; ++i) { bv(i) = bh[i]; ev(i) = eh[i]; }
    return search_boxes_type{sel, bv, ev};
  }
};

// =============================================================================
// STK mesh helper
// =============================================================================

/// Create a minimal node-only mesh with nodes numbered 1..num_nodes.
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
  for (int id = 1; id <= num_nodes; ++id) bulk_ptr->declare_node(id);
  bulk_ptr->modification_end();
  return {std::move(meta_ptr), std::move(bulk_ptr)};
}

// =============================================================================
// Pair collection and oracle
// =============================================================================

template <typename ListType>
PairSet collect_pairs(const ListType& list) {
  PairSet result;
  for (size_t t = 0; t < list.num_targets(); ++t)
    for (size_t k = 0; k < list.num_neighbors(t); ++k)
      result.insert({t, list.source_index(t, k)});
  return result;
}

/// N² oracle: all overlapping pairs with t != s (no self).
template <typename Trait>
PairSet oracle_pairs_no_self(const std::vector<typename Trait::box_type>& boxes) {
  PairSet pairs;
  for (size_t t = 0; t < boxes.size(); ++t)
    for (size_t s = 0; s < boxes.size(); ++s)
      if (t != s && Trait::overlap(boxes[t], boxes[s]))
        pairs.insert({t, s});
  return pairs;
}

// =============================================================================
// Shared structural invariant checks
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

/// Verifies that for_each_neighbor_pair emits exactly the same pairs as the direct accessor,
/// each exactly once.
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

/// Canonical verifier: exact pair set + all structural invariants.
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
// DeterministicFixtureT<Trait>
//
// Trait-parameterized fixture.  The mesh geometry is fixed; each trait converts
// the geometry to its native box type and search-box container.
//
// Nodes and boxes (0-indexed array ordinals):
//   Ord 0  (node 1, target_part):  box0 = [-2,   2]³       (center 0,    half 2.0)
//   Ord 1  (node 2, target_part):  box1 = isolated @ 100   (center 100,  half 0.5)
//   Ord 2  (node 3, source_part):  box2 = [ 1.5, 3.5]³     (center 2.5,  half 1.0)
//   Ord 3  (node 4, source_part):  box3 = isolated @ 200   (center 200,  half 0.5)
//   Ord 4  (node 5, shared_part):  box4 = [-0.5, 2.5]³     (center 1.0,  half 1.5)
//   Ord 5  (node 6, shared_part):  box5 = [ 0.5, 3.5]³     (center 2.0,  half 1.5)
//
// Selector configurations:
//   Universal:   target = {0..5},      source = {0..5}
//   Disjoint:    target = {0,1},       source = {2,3}
//   Overlapping: target = {0,1,4,5},   source = {2,3,4,5}
//   IdSubset:    target = {4,5},       source = {4,5}
// =============================================================================

template <typename Trait>
class DeterministicFixtureT : public ::testing::Test {
 public:
  using box_type          = typename Trait::box_type;
  using search_boxes_type = typename Trait::search_boxes_type;

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
        Trait::make(0.0f, 0.0f, 0.0f, 2.0f),        // ord 0: [-2, 2]³
        Trait::make(100.f, 100.f, 100.f, 0.5f),      // ord 1: isolated
        Trait::make(2.5f, 2.5f, 2.5f, 1.0f),         // ord 2: [1.5, 3.5]³
        Trait::make(200.f, 200.f, 200.f, 0.5f),      // ord 3: isolated
        Trait::make(1.0f, 1.0f, 1.0f, 1.5f),         // ord 4: [-0.5, 2.5]³
        Trait::make(2.0f, 2.0f, 2.0f, 1.5f),         // ord 5: [0.5, 3.5]³
    };

    const stk::mesh::Selector sel_all     = meta_->universal_part();
    const stk::mesh::Selector sel_tgt     = *target_part_;
    const stk::mesh::Selector sel_src     = *source_part_;
    const stk::mesh::Selector sel_shr     = *shared_part_;
    const stk::mesh::Selector sel_tgt_shr = *target_part_ | *shared_part_;
    const stk::mesh::Selector sel_src_shr = *source_part_ | *shared_part_;

    universal_boxes_ = Trait::make_search_boxes(sel_all, all_boxes_, nodes_);

    disjoint_target_boxes_ = Trait::make_search_boxes(
        sel_tgt, {all_boxes_[0], all_boxes_[1]}, {nodes_[0], nodes_[1]});
    disjoint_source_boxes_ = Trait::make_search_boxes(
        sel_src, {all_boxes_[2], all_boxes_[3]}, {nodes_[2], nodes_[3]});

    overlapping_target_boxes_ = Trait::make_search_boxes(
        sel_tgt_shr,
        {all_boxes_[0], all_boxes_[1], all_boxes_[4], all_boxes_[5]},
        {nodes_[0], nodes_[1], nodes_[4], nodes_[5]});
    overlapping_source_boxes_ = Trait::make_search_boxes(
        sel_src_shr,
        {all_boxes_[2], all_boxes_[3], all_boxes_[4], all_boxes_[5]},
        {nodes_[2], nodes_[3], nodes_[4], nodes_[5]});

    idsubset_boxes_ = Trait::make_search_boxes(
        sel_shr, {all_boxes_[4], all_boxes_[5]}, {nodes_[4], nodes_[5]});
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Part* target_part_ = nullptr;
  stk::mesh::Part* source_part_ = nullptr;
  stk::mesh::Part* shared_part_ = nullptr;
  std::vector<stk::mesh::Entity> nodes_;
  std::vector<box_type> all_boxes_;
  search_boxes_type universal_boxes_;
  search_boxes_type disjoint_target_boxes_;
  search_boxes_type disjoint_source_boxes_;
  search_boxes_type overlapping_target_boxes_;
  search_boxes_type overlapping_source_boxes_;
  search_boxes_type idsubset_boxes_;
};

using STKDeterministicFixture = DeterministicFixtureT<STKBoxTrait>;

#ifdef HAVE_MUNDYSEARCH_ARBORX
using DeterministicFixture = DeterministicFixtureT<ArborXBoxTrait>;
#endif

// =============================================================================
// Group 2 test helper implementations (template on ListType + FixtureType)
//
// All helpers call the builder via make_neighbor_list_builder<ListType>() and access
// the fixture's typed search-box members (which are the right type for each list via the
// Trait).  The expected pair sets are geometry-derived and backend-agnostic: on a single
// process, all list types produce the same results for the same geometry.
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
//   Only box0↔box2 overlaps.  All 4 excluder variants: {(0,0)}.
//
// Overlapping (target={node1,node2,node5,node6} ords {0,1,2,3},
//              source={node3,node4,node5,node6} ords {0,1,2,3}):
//   Intersection = shared_part = {node5,node6} → target ords {2,3} / source ords {2,3}.
//
// IdSubset (target={node5,node6} ords {0,1}, source={node5,node6} ords {0,1}):
//   All four nodes mutually overlapping.
// =============================================================================

// ---- Universal ----

template <typename ListType, typename FixtureType>
void test_universal_no_excluder(FixtureType& f) {
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

template <typename ListType, typename FixtureType>
void test_universal_self(FixtureType& f) {
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

template <typename ListType, typename FixtureType>
void test_universal_symdups(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  // Suppress (t,s) where src_entity < trg_entity; sequential node creation means
  // entity handle ordering == node-ID ordering == array ordinal ordering.
  const PairSet expected = {
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
      {0, 2}, {0, 4}, {0, 5}, {2, 4}, {2, 5}, {4, 5}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_universal_self_symdups(FixtureType& f) {
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

template <typename ListType, typename FixtureType>
void test_disjoint_no_excluder(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};  // node1 (box0) ↔ node3 (box2)
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_disjoint_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};  // disjoint parts: no self-pairs to remove
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_disjoint_symdups(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.disjoint_target_boxes_)
      .source_input(f.disjoint_source_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}};  // selector intersection is empty; ExcludeSymDup never fires
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_disjoint_self_symdups(FixtureType& f) {
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

template <typename ListType, typename FixtureType>
void test_overlapping_no_excluder(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {
      {0, 0},   // node1↔node3
      {0, 2},   // node1↔node5
      {0, 3},   // node1↔node6
      {2, 0},   // node5↔node3
      {2, 2},   // node5 self
      {2, 3},   // node5↔node6
      {3, 0},   // node6↔node3
      {3, 2},   // node6↔node5
      {3, 3},   // node6 self
  };
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_overlapping_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}, {0, 2}, {0, 3}, {2, 0}, {2, 3}, {3, 0}, {3, 2}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_overlapping_symdups(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.overlapping_target_boxes_)
      .source_input(f.overlapping_source_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  // Intersection = shared_part = {node5, node6}.
  // Only pair where BOTH entities are in intersection AND src < trg: (3,2) [trg=node6, src=node5].
  const PairSet expected = {{0, 0}, {0, 2}, {0, 3}, {2, 0}, {2, 2}, {2, 3}, {3, 0}, {3, 3}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_overlapping_self_symdups(FixtureType& f) {
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

template <typename ListType, typename FixtureType>
void test_idsubset_no_excluder(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .build(*f.bulk_);
  const PairSet expected = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_idsubset_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  const PairSet expected = {{0, 1}, {1, 0}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_idsubset_symdups(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.idsubset_boxes_)
      .source_input(f.idsubset_boxes_)
      .exclude(ExcludeSymmetricDuplicates{})
      .build(*f.bulk_);
  // Both arrays are shared_part; intersection = all nodes here.
  // Suppress (1,0): trg=node6, src=node5, src_entity < trg_entity.
  const PairSet expected = {{0, 0}, {0, 1}, {1, 1}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_idsubset_self_symdups(FixtureType& f) {
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

// =============================================================================
// Macros for Group 2 test registration
//
// MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, selector, excluder_tag)
//   — registers one TEST_F for the given (selector × excluder_tag) combination.
//
// MAKE_ALL_DET_TESTS(FIXTURE, SUFFIX, LIST_TYPE)
//   — registers all 16 (4 selectors × 4 excluder_tags) combinations at once.
//
// Usage:
//   MAKE_ALL_DET_TESTS(DeterministicFixture,    1d,  List1d)  → 16 ArborX-1d tests
//   MAKE_ALL_DET_TESTS(DeterministicFixture,    2d,  List2d)  → 16 ArborX-2d tests
//   MAKE_ALL_DET_TESTS(STKDeterministicFixture, stk, STKList) → 16 STK tests
// =============================================================================

// clang-format off
#define MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, selector, excluder_tag)     \
  TEST_F(FIXTURE, selector##_##excluder_tag##_##SUFFIX) {                     \
    test_##selector##_##excluder_tag<LIST_TYPE, FIXTURE>(*this);              \
  }

#define MAKE_ALL_DET_TESTS(FIXTURE, SUFFIX, LIST_TYPE)                        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   no_excluder)         \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   self)                \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   symdups)             \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   self_symdups)        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    no_excluder)         \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    self)                \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    symdups)             \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    self_symdups)        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, no_excluder)         \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, self)                \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, symdups)             \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, self_symdups)        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    no_excluder)         \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    self)                \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    symdups)             \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    self_symdups)
// clang-format on

// =============================================================================
// Group 1 — Construction and access
// =============================================================================

TEST(STKSearchNeighborList, DefaultConstruct) {
  STKList list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(), 0u);
}

TEST_F(STKDeterministicFixture, CopyMove_stk) {
  auto original = make_neighbor_list_builder<STKList>()
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

  STKList copy_assign;
  copy_assign = original;
  EXPECT_EQ(copy_assign.num_targets(), nt);
  EXPECT_EQ(copy_assign.num_sources(), ns);
  EXPECT_EQ(copy_assign.size(), sz);

  STKList move_assign;
  move_assign = std::move(copy_assign);
  EXPECT_EQ(move_assign.num_targets(), nt);
  EXPECT_EQ(move_assign.num_sources(), ns);
  EXPECT_EQ(move_assign.size(), sz);
}

#ifdef HAVE_MUNDYSEARCH_ARBORX

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

#endif  // HAVE_MUNDYSEARCH_ARBORX

// =============================================================================
// Group 2 — Deterministic pair-set tests
//
// 16 tests per list type (4 selectors × 4 excluder variants):
//   ArborX-1d: DeterministicFixture.{selector}_{excluder}_1d   (16 tests)
//   ArborX-2d: DeterministicFixture.{selector}_{excluder}_2d   (16 tests)
//   STK:       STKDeterministicFixture.{selector}_{excluder}_stk (16 tests)
// =============================================================================

MAKE_ALL_DET_TESTS(STKDeterministicFixture, stk, STKList)

#ifdef HAVE_MUNDYSEARCH_ARBORX
MAKE_ALL_DET_TESTS(DeterministicFixture, 1d, List1d)
MAKE_ALL_DET_TESTS(DeterministicFixture, 2d, List2d)
#endif

#undef MAKE_DET_TEST
#undef MAKE_ALL_DET_TESTS

// =============================================================================
// Group 3 — Random N² correctness validation
//
// Place N spheres at Philox-generated centers; build with ExcludeSelfInteraction;
// verify every list pair corresponds to overlapping boxes and the set equals the
// brute-force N² oracle.
// =============================================================================

template <typename ListType, typename Trait>
void run_random_n2_validation(stk::mesh::BulkData& bulk, const stk::mesh::Selector& selector,
                              int num_nodes) {
  using box_type = typename Trait::box_type;

  constexpr size_t kSeed       = 42;
  constexpr float  kDomainSize = 10.0f;
  constexpr float  kRadius     = 0.9f;

  std::vector<stk::mesh::Entity> nodes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i] = bulk.get_entity(stk::topology::NODE_RANK, i + 1);
    ASSERT_TRUE(bulk.is_valid(nodes[i])) << "Node " << (i + 1) << " not found.";
  }

  std::vector<box_type> boxes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    const float cx = rng.uniform<float>(0.0f, kDomainSize);
    const float cy = rng.uniform<float>(0.0f, kDomainSize);
    const float cz = rng.uniform<float>(0.0f, kDomainSize);
    boxes[i] = Trait::make(cx, cy, cz, kRadius);
  }

  auto sb   = Trait::make_search_boxes(selector, boxes, nodes);
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(sb)
      .source_input(sb)
      .exclude(ExcludeSelfInteraction{})
      .build(bulk);

  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t s = list.source_index(t, k);
      EXPECT_TRUE(Trait::overlap(boxes[t], boxes[s]))
          << "Spurious pair (target=" << t << ", source=" << s << "): boxes do not overlap.";
      EXPECT_NE(t, s) << "Self-pair (target==source==" << t << ") despite ExcludeSelfInteraction.";
    }
  }

  EXPECT_EQ(collect_pairs(list), oracle_pairs_no_self<Trait>(boxes))
      << "Neighbor list is missing oracle pairs or contains extra pairs.";
}

TEST(STKSearchNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  run_random_n2_validation<STKList, STKBoxTrait>(*bulk, meta->universal_part(), kN);
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST(ArborX1dNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  run_random_n2_validation<List1d, ArborXBoxTrait>(*bulk, meta->universal_part(), kN);
}

TEST(ArborX2dNeighborList, RandomN2Validation) {
  constexpr int kN = 50;
  auto [meta, bulk] = make_node_mesh(kN);
  run_random_n2_validation<List2d, ArborXBoxTrait>(*bulk, meta->universal_part(), kN);
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

// =============================================================================
// Group 4 — Iteration protocol
//
// for_each_target_with_neighbors must visit ALL targets, including those with zero
// neighbors.  A literal reading of "with_neighbors" might suggest skipping
// zero-neighbor targets; this test guards against that regression.
// =============================================================================

template <typename ListType, typename Trait>
void test_all_isolated_visits_all_targets() {
  constexpr int kN = 3;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();

  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i)
    nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, i + 1);

  // Three mutually non-overlapping boxes (isolated).
  auto sb = Trait::make_search_boxes(sel,
      {Trait::make(0.f, 0.f, 0.f, 0.1f),
       Trait::make(1000.f, 0.f, 0.f, 0.1f),
       Trait::make(0.f, 1000.f, 0.f, 0.1f)},
      nodes);

  const auto list = make_neighbor_list_builder<ListType>()
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
      KOKKOS_LAMBDA(const Neighbors<ListType>&) { Kokkos::atomic_inc(&tgt_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(tgt_count(0), static_cast<size_t>(kN))
      << "for_each_target_with_neighbors must visit all " << kN
      << " targets even with zero neighbors.";
}

TEST(IterationProtocol, AllIsolated_VisitsAllTargets_stk) {
  test_all_isolated_visits_all_targets<STKList, STKBoxTrait>();
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST(IterationProtocol, AllIsolated_VisitsAllTargets_1d) {
  test_all_isolated_visits_all_targets<List1d, ArborXBoxTrait>();
}

TEST(IterationProtocol, AllIsolated_VisitsAllTargets_2d) {
  test_all_isolated_visits_all_targets<List2d, ArborXBoxTrait>();
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

// =============================================================================
// Group 5 — Debug bounds
//
// MUNDY_THROW_ASSERT fires only when NDEBUG is not defined.
// =============================================================================

#ifndef NDEBUG

template <typename ListType, typename FixtureType>
void test_out_of_bounds(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()),  std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()),  std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()),  std::out_of_range);
  EXPECT_THROW(list.source_index(0, list.num_neighbors(0)), std::out_of_range);
}

TEST_F(STKDeterministicFixture, OutOfBounds_stk) {
  test_out_of_bounds<STKList, STKDeterministicFixture>(*this);
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST_F(DeterministicFixture, OutOfBounds_1d) {
  test_out_of_bounds<List1d, DeterministicFixture>(*this);
}

TEST_F(DeterministicFixture, OutOfBounds_2d) {
  test_out_of_bounds<List2d, DeterministicFixture>(*this);
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

#endif  // NDEBUG

// =============================================================================
// Group 6 — Reduction functions
//
// Verifies for_each_neighbor_pair_reduce and for_each_target_with_neighbors_reduce
// using Kokkos::Sum<size_t>.  All results are cross-checked against the direct
// pair accessor.
//
// Four invariants tested on the universal+ExcludeSelf list:
//   1. for_each_neighbor_pair_reduce Sum(1 per pair)             == list.size()
//   2. for_each_neighbor_pair_reduce Sum(source_index per pair)  == direct sum
//   3. for_each_target_with_neighbors_reduce Sum(nbrs.size())    == list.size()
//   4. for_each_target_with_neighbors_reduce Sum(1 per target)   == num_targets()
// =============================================================================

template <typename ListType>
void check_reduce_functions(const ListType& list) {
  // 1. Pair count via Sum reducer must equal list.size().
  {
    size_t count = 0;
    Kokkos::Sum<size_t> reducer(count);
    mundy::search::for_each_neighbor_pair_reduce(
        TestExecSpace{}, list,
        KOKKOS_LAMBDA(const NeighborPair<ListType>&, size_t& n) { ++n; },
        reducer);
    EXPECT_EQ(count, list.size())
        << "for_each_neighbor_pair_reduce Sum(1) != list.size()";
  }

  // 2. Sum of source ordinals via pair reducer must match direct iteration.
  {
    size_t direct_sum = 0;
    for (size_t t = 0; t < list.num_targets(); ++t)
      for (size_t k = 0; k < list.num_neighbors(t); ++k)
        direct_sum += list.source_index(t, k);

    size_t reduce_sum = 0;
    Kokkos::Sum<size_t> reducer(reduce_sum);
    mundy::search::for_each_neighbor_pair_reduce(
        TestExecSpace{}, list,
        KOKKOS_LAMBDA(const NeighborPair<ListType>& pair, size_t& s) {
          s += pair.source_index();
        },
        reducer);
    EXPECT_EQ(reduce_sum, direct_sum)
        << "for_each_neighbor_pair_reduce Sum(source_index) != direct sum";
  }

  // 3. Sum of per-target neighbor counts must equal list.size().
  {
    size_t total = 0;
    Kokkos::Sum<size_t> reducer(total);
    mundy::search::for_each_target_with_neighbors_reduce(
        TestExecSpace{}, list,
        KOKKOS_LAMBDA(const Neighbors<ListType>& nbrs, size_t& n) {
          n += nbrs.size();
        },
        reducer);
    EXPECT_EQ(total, list.size())
        << "for_each_target_with_neighbors_reduce Sum(nbrs.size) != list.size()";
  }

  // 4. Every target is visited exactly once (including zero-neighbor targets).
  {
    size_t count = 0;
    Kokkos::Sum<size_t> reducer(count);
    mundy::search::for_each_target_with_neighbors_reduce(
        TestExecSpace{}, list,
        KOKKOS_LAMBDA(const Neighbors<ListType>&, size_t& n) { ++n; },
        reducer);
    EXPECT_EQ(count, list.num_targets())
        << "for_each_target_with_neighbors_reduce Sum(1) != num_targets()";
  }
}

template <typename ListType, typename FixtureType>
void test_reduce_universal_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(f.universal_boxes_)
      .source_input(f.universal_boxes_)
      .exclude(ExcludeSelfInteraction{})
      .build(*f.bulk_);
  check_reduce_functions(list);
}

TEST_F(STKDeterministicFixture, ReduceFunctions_stk) {
  test_reduce_universal_self<STKList, STKDeterministicFixture>(*this);
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST_F(DeterministicFixture, ReduceFunctions_1d) {
  test_reduce_universal_self<List1d, DeterministicFixture>(*this);
}
TEST_F(DeterministicFixture, ReduceFunctions_2d) {
  test_reduce_universal_self<List2d, DeterministicFixture>(*this);
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

// for_each_target_with_neighbors_reduce must visit every target even when every
// target has zero neighbors — regression parallel to Group 4 IterationProtocol.

template <typename ListType, typename Trait>
void test_reduce_all_isolated_visits_all_targets() {
  constexpr int kN = 3;
  auto [meta, bulk] = make_node_mesh(kN);
  const stk::mesh::Selector sel = meta->universal_part();

  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i)
    nodes[i] = bulk->get_entity(stk::topology::NODE_RANK, i + 1);

  auto sb = Trait::make_search_boxes(sel,
      {Trait::make(0.f, 0.f, 0.f, 0.1f),
       Trait::make(1000.f, 0.f, 0.f, 0.1f),
       Trait::make(0.f, 1000.f, 0.f, 0.1f)},
      nodes);

  const auto list = make_neighbor_list_builder<ListType>()
      .exec_space(TestExecSpace{})
      .target_input(sb)
      .source_input(sb)
      .exclude(ExcludeSelfInteraction{})
      .build(*bulk);

  ASSERT_EQ(list.size(), 0u);

  size_t count = 0;
  Kokkos::Sum<size_t> reducer(count);
  mundy::search::for_each_target_with_neighbors_reduce(
      TestExecSpace{}, list,
      KOKKOS_LAMBDA(const Neighbors<ListType>&, size_t& n) { ++n; },
      reducer);
  EXPECT_EQ(count, static_cast<size_t>(kN))
      << "for_each_target_with_neighbors_reduce must visit all " << kN
      << " targets even when all have zero neighbors.";
}

TEST(ReduceProtocol, AllIsolated_VisitsAllTargets_stk) {
  test_reduce_all_isolated_visits_all_targets<STKList, STKBoxTrait>();
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST(ReduceProtocol, AllIsolated_VisitsAllTargets_1d) {
  test_reduce_all_isolated_visits_all_targets<List1d, ArborXBoxTrait>();
}

TEST(ReduceProtocol, AllIsolated_VisitsAllTargets_2d) {
  test_reduce_all_isolated_visits_all_targets<List2d, ArborXBoxTrait>();
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

}  // namespace
}  // namespace search
}  // namespace mundy
