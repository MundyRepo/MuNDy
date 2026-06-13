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
///   Group 7 — Rebuilder system: RebuilderType concept checks, AlwaysRebuild, NeverRebuild,
///     RebuildOnEntityChange (no-change / increase / decrease box count), RebuildOnAABBDisplacement
///     threshold behavior, RebuildOnOBBDisplacement threshold behavior (translation-only and
///     rotation-only), RebuilderChain OR logic via operator|, and ManagedNeighborList lifecycle
///     (has_valid_list, invalidate, current).

// External
#include <gtest/gtest.h>

// C++ core
#include <cmath>
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
#include <stk_mesh/base/Field.hpp>      // for stk::mesh::Field
#include <stk_mesh/base/FieldBase.hpp>  // for stk::mesh::field_data
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for declare_field, put_field_on_mesh
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
#include <mundy_search/ManagedNeighborList.hpp>
#include <mundy_search/NeighborListBuilder.hpp>
#include <mundy_search/NeighborListRebuilder.hpp>
#include <mundy_search/Neighbors.hpp>

// Mundy search — STK (always available)
#include <mundy_search/STKSearchNeighborList.hpp>
#include <mundy_search/SearchInput.hpp>  // for SearchInput (component-backed inputs)
#include <mundy_search/impl/STKSearchBoxes.hpp>

// Mundy mesh / geom
#include <mundy_geom/primitives/AABB.hpp>  // for mundy::AABB (test geometry + oracle)
#include <mundy_geom/primitives/OBB.hpp>   // for mundy::OBB (OBB-rebuilder test geometry)
#include <mundy_math/Quaternion.hpp>       // for mundy::Quaternion
#include <mundy_math/Vector3.hpp>          // for mundy::Vector3, mundy::Point
#include <mundy_mesh/FieldComponent.hpp>   // for mundy::mesh::AABBFieldComponent, OBBFieldComponent

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

using TestMemSpace = Kokkos::HostSpace;
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
// Test geometry helpers
//
// The search API consumes components, so all list types (STK, ArborX 1d/2d) take the same component-backed
// `SearchInput` built from a per-node `aabb` field.  Geometry, the overlap oracle, and field population are
// therefore backend-agnostic — no per-backend trait is needed.
// =============================================================================

// Backend-agnostic test geometry: an axis-aligned box per node, stored in an `aabb` field and read through an
// AABBFieldComponent.  Both the STK and ArborX lists consume the same component input, so a single representation
// (mundy::AABB<double>) serves both input construction and the overlap oracle.

using TestAABB = mundy::AABB<double>;
using TestComponent = mundy::mesh::AABBFieldComponent<double>;
using TestInput = SearchInput<TestComponent>;

/// Build an AABB from a center and (possibly anisotropic) half-extents.
inline TestAABB make_aabb(double cx, double cy, double cz, double hx, double hy, double hz) {
  return TestAABB(cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz);
}
inline TestAABB make_aabb(double cx, double cy, double cz, double h) {
  return make_aabb(cx, cy, cz, h, h, h);
}

/// AABB overlap predicate (the N^2 oracle's comparison).
inline bool aabb_overlap(const TestAABB& a, const TestAABB& b) {
  for (int d = 0; d < 3; ++d) {
    if (a.max_corner()[d] < b.min_corner()[d]) return false;
    if (b.max_corner()[d] < a.min_corner()[d]) return false;
  }
  return true;
}

/// Declare a 6-scalar `aabb` node field (min xyz, max xyz). Must be called before commit.
inline stk::mesh::Field<double>& declare_aabb_field(stk::mesh::MetaData& meta) {
  auto& field = meta.declare_field<double>(stk::topology::NODE_RANK, "aabb_test_field");
  stk::mesh::put_field_on_mesh(field, meta.universal_part(), 6, nullptr);
  return field;
}

/// Write one node's AABB into the field (layout: min xyz at 0-2, max xyz at 3-5).
inline void store_aabb(stk::mesh::Field<double>& field, stk::mesh::Entity node, const TestAABB& box) {
  double* data = stk::mesh::field_data(field, node);
  const auto& lo = box.min_corner();
  const auto& hi = box.max_corner();
  data[0] = lo[0];
  data[1] = lo[1];
  data[2] = lo[2];
  data[3] = hi[0];
  data[4] = hi[1];
  data[5] = hi[2];
}

using TestOBB = mundy::OBB<double>;
using TestOBBComponent = mundy::mesh::OBBFieldComponent<double>;

/// Declare a 10-scalar `obb` node field (center 0-2, quaternion wxyz 3-6, half-extents 7-9). Pre-commit only.
inline stk::mesh::Field<double>& declare_obb_field(stk::mesh::MetaData& meta) {
  auto& field = meta.declare_field<double>(stk::topology::NODE_RANK, "obb_test_field");
  stk::mesh::put_field_on_mesh(field, meta.universal_part(), 10, nullptr);
  return field;
}

/// Write one node's OBB into the field (layout: center 0-2, quat wxyz 3-6, half-extents 7-9).
inline void store_obb(stk::mesh::Field<double>& field, stk::mesh::Entity node, const TestOBB& obb) {
  double* data = stk::mesh::field_data(field, node);
  const auto& c = obb.center();
  const auto& q = obb.orientation();
  const auto& h = obb.half_extents();
  data[0] = c[0];
  data[1] = c[1];
  data[2] = c[2];
  data[3] = q.w();
  data[4] = q.x();
  data[5] = q.y();
  data[6] = q.z();
  data[7] = h[0];
  data[8] = h[1];
  data[9] = h[2];
}

// =============================================================================
// STK mesh helper
// =============================================================================

/// A node-only mesh with an `aabb` node field declared (pre-commit) for component-backed search inputs.
struct NodeMeshWithAABB {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Field<double>* aabb_field = nullptr;
};

/// Create a minimal node-only mesh with nodes numbered 1..num_nodes and a 6-scalar `aabb` node field.
inline NodeMeshWithAABB make_node_mesh_with_aabb(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta_ptr = builder.create_meta_data();
  meta_ptr->use_simple_fields();
  auto& aabb_field = declare_aabb_field(*meta_ptr);
  auto bulk_ptr = builder.create(meta_ptr);
  meta_ptr->commit();
  bulk_ptr->modification_begin();
  for (int id = 1; id <= num_nodes; ++id) bulk_ptr->declare_node(id);
  bulk_ptr->modification_end();
  return {std::move(meta_ptr), std::move(bulk_ptr), &aabb_field};
}

/// Distributed node-only mesh: node id `i+1` (global id `i`) is owned by rank `i % nprocs` and declared only there.
/// Remote nodes are absent locally until the STK build ghosts them — the multi-rank path the cross-rank test exercises.
inline NodeMeshWithAABB make_distributed_node_mesh_with_aabb(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta_ptr = builder.create_meta_data();
  meta_ptr->use_simple_fields();
  auto& aabb_field = declare_aabb_field(*meta_ptr);
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

// =============================================================================
// Pair collection and oracle
// =============================================================================

template <typename ListType>
PairSet collect_pairs(const ListType& list) {
  PairSet result;
  for (size_t t = 0; t < list.num_targets(); ++t)
    for (size_t k = 0; k < list.num_neighbors(t); ++k) result.insert({t, list.source_index(t, k)});
  return result;
}

/// N² oracle: all overlapping pairs with t != s (no self).
inline PairSet oracle_pairs_no_self(const std::vector<TestAABB>& boxes) {
  PairSet pairs;
  for (size_t t = 0; t < boxes.size(); ++t)
    for (size_t s = 0; s < boxes.size(); ++s)
      if (t != s && aabb_overlap(boxes[t], boxes[s])) pairs.insert({t, s});
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
      TestExecSpace{}, list, KOKKOS_LAMBDA(const NeighborPair<ListType>& pair) {
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
    aabb_field_ = &declare_aabb_field(*meta_);  // declared before commit
    obb_field_ = &declare_obb_field(*meta_);    // declared before commit (for OBB-rebuilder managed test)
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

    // One AABB per node, written into the shared `aabb` field; selectors below pick which subset each input uses.
    const std::vector<TestAABB> geom = {
        make_aabb(0.0, 0.0, 0.0, 2.0),     // ord 0: [-2, 2]³
        make_aabb(100., 100., 100., 0.5),  // ord 1: isolated
        make_aabb(2.5, 2.5, 2.5, 1.0),     // ord 2: [1.5, 3.5]³
        make_aabb(200., 200., 200., 0.5),  // ord 3: isolated
        make_aabb(1.0, 1.0, 1.0, 1.5),     // ord 4: [-0.5, 2.5]³
        make_aabb(2.0, 2.0, 2.0, 1.5),     // ord 5: [0.5, 3.5]³
    };
    for (size_t i = 0; i < 6; ++i) store_aabb(*aabb_field_, nodes_[i], geom[i]);

    aabb_component_ = TestComponent(*aabb_field_);
    aabb_component_.modify_on_host();  // host data is current; the build syncs it to device

    // One OBB per node (axis-aligned unit-ish box at each AABB's center); used by the managed OBB-rebuilder test.
    for (size_t i = 0; i < 6; ++i) {
      const auto& lo = geom[i].min_corner();
      const auto& hi = geom[i].max_corner();
      const Point<double> center{0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
      const Vector3<double> half{0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1]), 0.5 * (hi[2] - lo[2])};
      store_obb(*obb_field_, nodes_[i], TestOBB{center, Quaternion<double>::identity(), half[0], half[1], half[2]});
    }
    obb_component_ = TestOBBComponent(*obb_field_);
    obb_component_.modify_on_host();

    const stk::mesh::Selector sel_all = meta_->universal_part();
    const stk::mesh::Selector sel_tgt = *target_part_;
    const stk::mesh::Selector sel_src = *source_part_;
    const stk::mesh::Selector sel_shr = *shared_part_;
    const stk::mesh::Selector sel_tgt_shr = *target_part_ | *shared_part_;
    const stk::mesh::Selector sel_src_shr = *source_part_ | *shared_part_;

    // All inputs share one component; they differ only in selector (which entities, and their ordering).
    universal_boxes_ = TestInput(sel_all, aabb_component_);
    disjoint_target_boxes_ = TestInput(sel_tgt, aabb_component_);
    disjoint_source_boxes_ = TestInput(sel_src, aabb_component_);
    overlapping_target_boxes_ = TestInput(sel_tgt_shr, aabb_component_);
    overlapping_source_boxes_ = TestInput(sel_src_shr, aabb_component_);
    idsubset_boxes_ = TestInput(sel_shr, aabb_component_);
  }

  // Move `ids` into part `add` (if non-null) and out of part `remove` (if non-null) via a mesh modification.
  // This advances `synchronized_count()`, mutating the entity set WITHIN a fixed selector — the supported way an
  // entity set changes (the selector itself, which defines the list's identity, stays fixed).
  void move_nodes(const std::vector<int>& ids, stk::mesh::Part* add, stk::mesh::Part* remove) {
    const stk::mesh::PartVector add_parts = add ? stk::mesh::PartVector{add} : stk::mesh::PartVector{};
    const stk::mesh::PartVector remove_parts = remove ? stk::mesh::PartVector{remove} : stk::mesh::PartVector{};
    bulk_->modification_begin();
    for (int id : ids) {
      bulk_->change_entity_parts(bulk_->get_entity(stk::topology::NODE_RANK, id), add_parts, remove_parts);
    }
    bulk_->modification_end();
  }

  std::shared_ptr<stk::mesh::MetaData> meta_;
  std::unique_ptr<stk::mesh::BulkData> bulk_;
  stk::mesh::Part* target_part_ = nullptr;
  stk::mesh::Part* source_part_ = nullptr;
  stk::mesh::Part* shared_part_ = nullptr;
  stk::mesh::Field<double>* aabb_field_ = nullptr;
  stk::mesh::Field<double>* obb_field_ = nullptr;
  std::vector<stk::mesh::Entity> nodes_;
  TestComponent aabb_component_;
  TestOBBComponent obb_component_;
  TestInput universal_boxes_;
  TestInput disjoint_target_boxes_;
  TestInput disjoint_source_boxes_;
  TestInput overlapping_target_boxes_;
  TestInput overlapping_source_boxes_;
  TestInput idsubset_boxes_;
};

// Single component-backed fixture serves every list type; these aliases keep the existing test-suite names.
using STKDeterministicFixture = DeterministicFixture;

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
  const PairSet expected = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {0, 2}, {2, 0}, {0, 4},
                            {4, 0}, {0, 5}, {5, 0}, {2, 4}, {4, 2}, {2, 5}, {5, 2}, {4, 5}, {5, 4}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_universal_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.universal_boxes_)
                  .source_input(f.universal_boxes_)
                  .broad_phase(ExcludeSelfInteraction{})
                  .build(*f.bulk_);
  const PairSet expected = {{0, 2}, {2, 0}, {0, 4}, {4, 0}, {0, 5}, {5, 0},
                            {2, 4}, {4, 2}, {2, 5}, {5, 2}, {4, 5}, {5, 4}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_universal_symdups(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.universal_boxes_)
                  .source_input(f.universal_boxes_)
                  .broad_phase(ExcludeSymmetricDuplicates{})
                  .build(*f.bulk_);
  // Suppress (t,s) where src_entity < trg_entity; sequential node creation means
  // entity handle ordering == node-ID ordering == array ordinal ordering.
  const PairSet expected = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5},
                            {0, 2}, {0, 4}, {0, 5}, {2, 4}, {2, 5}, {4, 5}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_universal_self_symdups(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.universal_boxes_)
                  .source_input(f.universal_boxes_)
                  .broad_phase(ExcludeSelfInteraction{})
                  .broad_phase(ExcludeSymmetricDuplicates{})
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
                  .broad_phase(ExcludeSelfInteraction{})
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
                  .broad_phase(ExcludeSymmetricDuplicates{})
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
                  .broad_phase(ExcludeSelfInteraction{})
                  .broad_phase(ExcludeSymmetricDuplicates{})
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

template <typename ListType, typename FixtureType>
void test_overlapping_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.overlapping_target_boxes_)
                  .source_input(f.overlapping_source_boxes_)
                  .broad_phase(ExcludeSelfInteraction{})
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
                  .broad_phase(ExcludeSymmetricDuplicates{})
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
                  .broad_phase(ExcludeSelfInteraction{})
                  .broad_phase(ExcludeSymmetricDuplicates{})
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
                  .broad_phase(ExcludeSelfInteraction{})
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
                  .broad_phase(ExcludeSymmetricDuplicates{})
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
                  .broad_phase(ExcludeSelfInteraction{})
                  .broad_phase(ExcludeSymmetricDuplicates{})
                  .build(*f.bulk_);
  const PairSet expected = {{0, 1}};
  verify_exact_pair_set(list, expected);
}

// Narrow-phase-only: ExcludeSelfInteraction runs after the spatial query (narrow path).
// The pair set must be identical to the broad-phase-only self-exclusion tests — the
// phase a filter runs in must not affect which pairs survive.
template <typename ListType, typename FixtureType>
void test_universal_narrow_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.universal_boxes_)
                  .source_input(f.universal_boxes_)
                  .narrow_phase(ExcludeSelfInteraction{})
                  .build(*f.bulk_);
  const PairSet expected = {{0, 2}, {2, 0}, {0, 4}, {4, 0}, {0, 5}, {5, 0},
                            {2, 4}, {4, 2}, {2, 5}, {5, 2}, {4, 5}, {5, 4}};
  verify_exact_pair_set(list, expected);
}

template <typename ListType, typename FixtureType>
void test_disjoint_narrow_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.disjoint_target_boxes_)
                  .source_input(f.disjoint_source_boxes_)
                  .narrow_phase(ExcludeSelfInteraction{})
                  .build(*f.bulk_);
  const PairSet expected = {{0, 0}};
  verify_exact_pair_set(list, expected);
}

// Mixed broad + narrow: ExcludeSymmetricDuplicates in broad, ExcludeSelfInteraction in
// narrow.  Running both phases must produce the intersection of both filter sets — the
// result is the half-list with self-pairs also removed.
template <typename ListType, typename FixtureType>
void test_universal_broad_symdups_narrow_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.universal_boxes_)
                  .source_input(f.universal_boxes_)
                  .broad_phase(ExcludeSymmetricDuplicates{})
                  .narrow_phase(ExcludeSelfInteraction{})
                  .build(*f.bulk_);
  // ExcludeSymmetricDuplicates keeps only (t,s) where src_entity >= trg_entity (including
  // self-pairs); the narrow-phase ExcludeSelfInteraction then removes those self-pairs.
  const PairSet expected = {{0, 2}, {0, 4}, {0, 5}, {2, 4}, {2, 5}, {4, 5}};
  verify_exact_pair_set(list, expected);
}

// =============================================================================
// Macros for Group 2 test registration
//
// MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, selector, excluder_tag)
//   — registers one TEST_F for the given (selector × excluder_tag) combination.
//
// MAKE_ALL_DET_TESTS(FIXTURE, SUFFIX, LIST_TYPE)
//   — registers all combinations at once.
//
// Broad-phase-only (4 selectors × 4 tags = 16 tests), plus 3 narrow/mixed tests:
//   MAKE_ALL_DET_TESTS(DeterministicFixture,    1d,  List1d)  → 19 ArborX-1d tests
//   MAKE_ALL_DET_TESTS(DeterministicFixture,    2d,  List2d)  → 19 ArborX-2d tests
//   MAKE_ALL_DET_TESTS(STKDeterministicFixture, stk, STKList) → 19 STK tests
// =============================================================================

// clang-format off
#define MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, selector, excluder_tag)     \
  TEST_F(FIXTURE, selector##_##excluder_tag##_##SUFFIX) {                     \
    test_##selector##_##excluder_tag<LIST_TYPE, FIXTURE>(*this);              \
  }

#define MAKE_ALL_DET_TESTS(FIXTURE, SUFFIX, LIST_TYPE)                                    \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   no_excluder)                    \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   self)                           \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   symdups)                        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   self_symdups)                   \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    no_excluder)                    \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    self)                           \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    symdups)                        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    self_symdups)                   \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, no_excluder)                    \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, self)                           \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, symdups)                        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, overlapping, self_symdups)                   \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    no_excluder)                    \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    self)                           \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    symdups)                        \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, idsubset,    self_symdups)                   \
  /* narrow-phase-only: filter runs after spatial query, result must match broad-phase */ \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   narrow_self)                    \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, disjoint,    narrow_self)                    \
  /* mixed broad + narrow: each phase contributes independent filtering */                \
  MAKE_DET_TEST(FIXTURE, SUFFIX, LIST_TYPE, universal,   broad_symdups_narrow_self)
// clang-format on

// =============================================================================
// Group 1 — Construction and access
// =============================================================================

TEST(STKSearchNeighborList, DefaultConstruct) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
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
                      .broad_phase(ExcludeSelfInteraction{})
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
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  List1d list;
  EXPECT_EQ(list.num_targets(), 0u);
  EXPECT_EQ(list.num_sources(), 0u);
  EXPECT_EQ(list.size(), 0u);
}

TEST(ArborX2dNeighborList, DefaultConstruct) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
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
                      .broad_phase(ExcludeSelfInteraction{})
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
                      .broad_phase(ExcludeSelfInteraction{})
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

template <typename ListType>
void run_random_n2_validation(stk::mesh::BulkData& bulk, stk::mesh::Field<double>& aabb_field,
                              const stk::mesh::Selector& selector, int num_nodes) {
  constexpr size_t kSeed = 42;
  constexpr float kDomainSize = 10.0f;
  constexpr double kRadius = 0.9;

  std::vector<stk::mesh::Entity> nodes(num_nodes);
  std::vector<TestAABB> boxes(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    nodes[i] = bulk.get_entity(stk::topology::NODE_RANK, i + 1);
    ASSERT_TRUE(bulk.is_valid(nodes[i])) << "Node " << (i + 1) << " not found.";
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    const double cx = rng.uniform<float>(0.0f, kDomainSize);
    const double cy = rng.uniform<float>(0.0f, kDomainSize);
    const double cz = rng.uniform<float>(0.0f, kDomainSize);
    boxes[i] = make_aabb(cx, cy, cz, kRadius);
    store_aabb(aabb_field, nodes[i], boxes[i]);
  }

  TestComponent component(aabb_field);
  component.modify_on_host();
  TestInput input(selector, component);
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(input)
                  .source_input(input)
                  .broad_phase(ExcludeSelfInteraction{})
                  .build(bulk);

  for (size_t t = 0; t < list.num_targets(); ++t) {
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t s = list.source_index(t, k);
      EXPECT_TRUE(aabb_overlap(boxes[t], boxes[s]))
          << "Spurious pair (target=" << t << ", source=" << s << "): boxes do not overlap.";
      EXPECT_NE(t, s) << "Self-pair (target==source==" << t << ") despite ExcludeSelfInteraction.";
    }
  }

  EXPECT_EQ(collect_pairs(list), oracle_pairs_no_self(boxes))
      << "Neighbor list is missing oracle pairs or contains extra pairs.";
}

TEST(STKSearchNeighborList, RandomN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 50;
  auto mesh = make_node_mesh_with_aabb(kN);
  run_random_n2_validation<STKList>(*mesh.bulk, *mesh.aabb_field, mesh.meta->universal_part(), kN);
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST(ArborX1dNeighborList, RandomN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 50;
  auto mesh = make_node_mesh_with_aabb(kN);
  run_random_n2_validation<List1d>(*mesh.bulk, *mesh.aabb_field, mesh.meta->universal_part(), kN);
}

TEST(ArborX2dNeighborList, RandomN2Validation) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  constexpr int kN = 50;
  auto mesh = make_node_mesh_with_aabb(kN);
  run_random_n2_validation<List2d>(*mesh.bulk, *mesh.aabb_field, mesh.meta->universal_part(), kN);
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

// =============================================================================
// Group 4 — Iteration protocol
//
// for_each_target_with_neighbors must visit ALL targets, including those with zero
// neighbors.  A literal reading of "with_neighbors" might suggest skipping
// zero-neighbor targets; this test guards against that regression.
// =============================================================================

template <typename ListType>
void test_all_isolated_visits_all_targets() {
  constexpr int kN = 3;
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector sel = mesh.meta->universal_part();

  // Three mutually non-overlapping boxes (isolated).
  const TestAABB geom[kN] = {make_aabb(0., 0., 0., 0.1), make_aabb(1000., 0., 0., 0.1), make_aabb(0., 1000., 0., 0.1)};
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i) {
    nodes[i] = mesh.bulk->get_entity(stk::topology::NODE_RANK, i + 1);
    store_aabb(*mesh.aabb_field, nodes[i], geom[i]);
  }

  TestComponent component(*mesh.aabb_field);
  component.modify_on_host();
  TestInput input(sel, component);
  const auto list = make_neighbor_list_builder<ListType>()
                        .exec_space(TestExecSpace{})
                        .target_input(input)
                        .source_input(input)
                        .broad_phase(ExcludeSelfInteraction{})
                        .build(*mesh.bulk);

  ASSERT_EQ(list.size(), 0u);

  Kokkos::View<size_t*, TestMemSpace> tgt_count("tgt_count", 1);
  Kokkos::deep_copy(tgt_count, size_t(0));
  mundy::search::for_each_target_with_neighbors(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<ListType>&) { Kokkos::atomic_inc(&tgt_count(0)); });
  Kokkos::fence();
  EXPECT_EQ(tgt_count(0), static_cast<size_t>(kN))
      << "for_each_target_with_neighbors must visit all " << kN << " targets even with zero neighbors.";
}

TEST(IterationProtocol, AllIsolated_VisitsAllTargets_stk) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  test_all_isolated_visits_all_targets<STKList>();
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST(IterationProtocol, AllIsolated_VisitsAllTargets_1d) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  test_all_isolated_visits_all_targets<List1d>();
}

TEST(IterationProtocol, AllIsolated_VisitsAllTargets_2d) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  test_all_isolated_visits_all_targets<List2d>();
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
                  .broad_phase(ExcludeSelfInteraction{})
                  .build(*f.bulk_);

  EXPECT_THROW(list.num_neighbors(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.target_entity(list.num_targets()), std::out_of_range);
  EXPECT_THROW(list.source_entity(list.num_sources()), std::out_of_range);
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
        TestExecSpace{}, list, KOKKOS_LAMBDA(const NeighborPair<ListType>&, size_t& n) { ++n; }, reducer);
    EXPECT_EQ(count, list.size()) << "for_each_neighbor_pair_reduce Sum(1) != list.size()";
  }

  // 2. Sum of source ordinals via pair reducer must match direct iteration.
  {
    size_t direct_sum = 0;
    for (size_t t = 0; t < list.num_targets(); ++t)
      for (size_t k = 0; k < list.num_neighbors(t); ++k) direct_sum += list.source_index(t, k);

    size_t reduce_sum = 0;
    Kokkos::Sum<size_t> reducer(reduce_sum);
    mundy::search::for_each_neighbor_pair_reduce(
        TestExecSpace{}, list,
        KOKKOS_LAMBDA(const NeighborPair<ListType>& pair, size_t& s) { s += pair.source_index(); }, reducer);
    EXPECT_EQ(reduce_sum, direct_sum) << "for_each_neighbor_pair_reduce Sum(source_index) != direct sum";
  }

  // 3. Sum of per-target neighbor counts must equal list.size().
  {
    size_t total = 0;
    Kokkos::Sum<size_t> reducer(total);
    mundy::search::for_each_target_with_neighbors_reduce(
        TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<ListType>& nbrs, size_t& n) { n += nbrs.size(); },
        reducer);
    EXPECT_EQ(total, list.size()) << "for_each_target_with_neighbors_reduce Sum(nbrs.size) != list.size()";
  }

  // 4. Every target is visited exactly once (including zero-neighbor targets).
  {
    size_t count = 0;
    Kokkos::Sum<size_t> reducer(count);
    mundy::search::for_each_target_with_neighbors_reduce(
        TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<ListType>&, size_t& n) { ++n; }, reducer);
    EXPECT_EQ(count, list.num_targets()) << "for_each_target_with_neighbors_reduce Sum(1) != num_targets()";
  }
}

template <typename ListType, typename FixtureType>
void test_reduce_universal_self(FixtureType& f) {
  auto list = make_neighbor_list_builder<ListType>()
                  .exec_space(TestExecSpace{})
                  .target_input(f.universal_boxes_)
                  .source_input(f.universal_boxes_)
                  .broad_phase(ExcludeSelfInteraction{})
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

template <typename ListType>
void test_reduce_all_isolated_visits_all_targets() {
  constexpr int kN = 3;
  auto mesh = make_node_mesh_with_aabb(kN);
  const stk::mesh::Selector sel = mesh.meta->universal_part();

  const TestAABB geom[kN] = {make_aabb(0., 0., 0., 0.1), make_aabb(1000., 0., 0., 0.1), make_aabb(0., 1000., 0., 0.1)};
  std::vector<stk::mesh::Entity> nodes(kN);
  for (int i = 0; i < kN; ++i) {
    nodes[i] = mesh.bulk->get_entity(stk::topology::NODE_RANK, i + 1);
    store_aabb(*mesh.aabb_field, nodes[i], geom[i]);
  }

  TestComponent component(*mesh.aabb_field);
  component.modify_on_host();
  TestInput input(sel, component);
  const auto list = make_neighbor_list_builder<ListType>()
                        .exec_space(TestExecSpace{})
                        .target_input(input)
                        .source_input(input)
                        .broad_phase(ExcludeSelfInteraction{})
                        .build(*mesh.bulk);

  ASSERT_EQ(list.size(), 0u);

  size_t count = 0;
  Kokkos::Sum<size_t> reducer(count);
  mundy::search::for_each_target_with_neighbors_reduce(
      TestExecSpace{}, list, KOKKOS_LAMBDA(const Neighbors<ListType>&, size_t& n) { ++n; }, reducer);
  EXPECT_EQ(count, static_cast<size_t>(kN))
      << "for_each_target_with_neighbors_reduce must visit all " << kN << " targets even when all have zero neighbors.";
}

TEST(ReduceProtocol, AllIsolated_VisitsAllTargets_stk) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  test_reduce_all_isolated_visits_all_targets<STKList>();
}

#ifdef HAVE_MUNDYSEARCH_ARBORX
TEST(ReduceProtocol, AllIsolated_VisitsAllTargets_1d) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  test_reduce_all_isolated_visits_all_targets<List1d>();
}

TEST(ReduceProtocol, AllIsolated_VisitsAllTargets_2d) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  test_reduce_all_isolated_visits_all_targets<List2d>();
}
#endif  // HAVE_MUNDYSEARCH_ARBORX

// =============================================================================
// Group 7 — Rebuilder system
//
// Behavioral verification strategy: when no rebuild occurs the managed list
// returns the same pair set from the initial build even when the caller passes
// new (non-overlapping) input boxes.  When a rebuild does occur the pair set
// reflects the new geometry.  The direct-API tests for RebuildOnAABBDisplacement
// call needs_rebuild / snapshot directly to verify threshold logic without
// going through ManagedNeighborList.
// =============================================================================

// Compile-time concept checks
static_assert(RebuilderType<AlwaysRebuild>, "AlwaysRebuild must satisfy RebuilderType.");
static_assert(RebuilderType<NeverRebuild>, "NeverRebuild must satisfy RebuilderType.");
static_assert(RebuilderType<RebuildOnEntityChange<TestMemSpace>>,
              "RebuildOnEntityChange<HostSpace> must satisfy RebuilderType.");
static_assert(RebuilderType<RebuildOnAABBDisplacement<double, TestMemSpace>>,
              "RebuildOnAABBDisplacement<double, HostSpace> must satisfy RebuilderType.");
static_assert(RebuilderType<RebuildOnOBBDisplacement<double, TestMemSpace>>,
              "RebuildOnOBBDisplacement<double, HostSpace> must satisfy RebuilderType.");
static_assert(RebuilderType<RebuilderChain<AlwaysRebuild, NeverRebuild>>,
              "RebuilderChain<AlwaysRebuild,NeverRebuild> must satisfy RebuilderType.");

// Move the disjoint-target nodes (1,2) far from all sources by mutating the `aabb` field; returns the target input.
// Targets {1,2} are disjoint from the disjoint-source nodes {3,4}, so this does not perturb the source geometry.
TestInput make_far_target_boxes(STKDeterministicFixture& f) {
  store_aabb(*f.aabb_field_, f.nodes_[0], make_aabb(500., 500., 500., 0.5));
  store_aabb(*f.aabb_field_, f.nodes_[1], make_aabb(600., 600., 600., 0.5));
  f.aabb_component_.modify_on_host();
  return f.disjoint_target_boxes_;
}

// A standalone node-only mesh with `aabb` + `obb` fields and target/source parts, for the direct rebuilder tests.
// node 1 → target part; node 2 (if present) → source part.  Geometry is written per-test via store_aabb/store_obb.
struct GeomMesh {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Field<double>* aabb_field = nullptr;
  stk::mesh::Field<double>* obb_field = nullptr;
  stk::mesh::Part* target_part = nullptr;
  stk::mesh::Part* source_part = nullptr;
  std::vector<stk::mesh::Entity> nodes;
};
inline GeomMesh make_geom_mesh(int num_nodes) {
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta = builder.create_meta_data();
  meta->use_simple_fields();
  GeomMesh m;
  m.target_part = &meta->declare_part("rb_target", stk::topology::NODE_RANK);
  m.source_part = &meta->declare_part("rb_source", stk::topology::NODE_RANK);
  m.aabb_field = &declare_aabb_field(*meta);
  m.obb_field = &declare_obb_field(*meta);
  m.bulk = builder.create(meta);
  meta->commit();
  m.bulk->modification_begin();
  for (int id = 1; id <= num_nodes; ++id) m.bulk->declare_node(id);
  m.bulk->change_entity_parts(m.bulk->get_entity(stk::topology::NODE_RANK, 1), stk::mesh::PartVector{m.target_part},
                              stk::mesh::PartVector{});
  if (num_nodes >= 2)
    m.bulk->change_entity_parts(m.bulk->get_entity(stk::topology::NODE_RANK, 2), stk::mesh::PartVector{m.source_part},
                                stk::mesh::PartVector{});
  m.bulk->modification_end();
  m.nodes.resize(num_nodes);
  for (int id = 1; id <= num_nodes; ++id) m.nodes[id - 1] = m.bulk->get_entity(stk::topology::NODE_RANK, id);
  m.meta = meta;
  return m;
}

// ---- ManagedNeighborList lifecycle ----

TEST(ManagedNeighborList, HasNoValidListBeforeFirstUpdate) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{});
  EXPECT_FALSE(managed.has_valid_list());
}

TEST(ManagedNeighborList, CurrentBeforeFirstUpdateThrows) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{});
  EXPECT_THROW(managed.current(), std::runtime_error);
}

TEST_F(STKDeterministicFixture, ManagedNeighborList_HasValidListAfterUpdate) {
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{});
  EXPECT_FALSE(managed.has_valid_list());
  managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(managed.has_valid_list());
}

TEST_F(STKDeterministicFixture, ManagedNeighborList_InvalidateClearsCache) {
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{});
  managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(managed.has_valid_list());
  managed.invalidate();
  EXPECT_FALSE(managed.has_valid_list());
}

// Even NeverRebuild must build on the first update() call after invalidate(),
// because the cache is empty (not because needs_rebuild() fired).
TEST_F(STKDeterministicFixture, ManagedNeighborList_InvalidateForcesBuildOnNextUpdate) {
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  managed.invalidate();
  // Update with geometry that produces zero pairs.
  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_TRUE(managed.has_valid_list());
  EXPECT_EQ(collect_pairs(r2.list), PairSet{});  // rebuilt with new geometry
}

// ---- AlwaysRebuild ----

// On each update() call AlwaysRebuild forces a fresh build, so passing different
// input geometry produces a different pair set.
TEST_F(STKDeterministicFixture, Rebuilder_AlwaysRebuild_RebuildsEveryUpdate) {
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(AlwaysRebuild{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  // AlwaysRebuild fires on the second call → result reflects the new geometry.
  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), PairSet{});
}

// ---- NeverRebuild ----

// After the first build NeverRebuild never fires, so passing different input
// geometry still returns the original pair set from the cache.
TEST_F(STKDeterministicFixture, Rebuilder_NeverRebuild_CachesAfterFirstBuild) {
  auto managed = make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_FALSE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}}));  // old pairs = cache was used
}

// ---- RebuildOnEntityChange ----

// Same geometry, same box count → cache is reused even though boxes moved.
TEST_F(STKDeterministicFixture, Rebuilder_EntityChange_NoRebuildOnUnchangedCount) {
  auto managed =
      make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(RebuildOnEntityChange<TestMemSpace>{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_FALSE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}}));  // cache: box count unchanged (still 2)
}

// Target entity count increases (2 → 4) → rebuild → pair set reflects the larger target set.
// Both target selectors are disjoint from the source nodes {3,4}, so the source geometry is untouched.
// The list's target selector is FIXED (target_part); the entity count grows from 2 to 4 via a mesh modification
// (nodes {5,6} move into target_part) → rebuild.
TEST_F(STKDeterministicFixture, Rebuilder_EntityChange_RebuildOnIncrease) {
  auto managed =
      make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(RebuildOnEntityChange<TestMemSpace>{});

  // 2-entity target_part {1,2} vs source {3,4}: node1 overlaps node3 → {(0,0)}.
  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  // Move nodes {5,6} into target_part (mesh modification): target_part now enumerates {1,2,5,6} (ords 0,1,2,3),
  // count 2 → 4 → rebuild. node1(0), node5(2), node6(3) each overlap node3(0); node4 overlaps nothing.
  move_nodes({5, 6}, target_part_, shared_part_);
  auto r2 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}, {2, 0}, {3, 0}}));
}

// The list's target selector is FIXED (target_part); the entity count shrinks from 4 to 2 via a mesh modification
// (nodes {5,6} leave target_part) → rebuild → pair set reflects the smaller target set.
TEST_F(STKDeterministicFixture, Rebuilder_EntityChange_RebuildOnDecrease) {
  auto managed =
      make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(RebuildOnEntityChange<TestMemSpace>{});

  // Start with nodes {5,6} in target_part: 4-entity target {1,2,5,6} vs source {3,4}.
  move_nodes({5, 6}, target_part_, shared_part_);
  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}, {2, 0}, {3, 0}}));

  // Move {5,6} back out of target_part: count 4 → 2 → rebuild → pairs reduced.
  move_nodes({5, 6}, shared_part_, target_part_);
  auto r2 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}}));
}

// ---- RebuildOnAABBDisplacement ----

// Directly exercise needs_rebuild / snapshot to verify the threshold without
// going through ManagedNeighborList.  One box with center shifted along x.
//
//   Snapshot at cx = 0 → min_x = -1, max_x = 1.
//   cx = 0.2: corner displacement 0.2 < threshold 0.5 → no rebuild.
//   cx = 1.0: corner displacement 1.0 > threshold 0.5 → rebuild.
//   New snapshot at cx = 1 → min_x = 0, max_x = 2.
//   cx = 0.6: displacement 0.4 < 0.5 → no rebuild.
//   cx = 0.4: displacement 0.6 > 0.5 → rebuild.
TEST(RebuildOnAABBDisplacement, ThresholdBehavior) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_geom_mesh(1);
  auto node = mesh.nodes[0];
  const stk::mesh::Selector sel = mesh.meta->universal_part();
  TestComponent component(*mesh.aabb_field);
  TestInput input(sel, component);

  constexpr float kThreshold = 0.5f;
  RebuildOnAABBDisplacement<double, TestMemSpace> rebuilder(kThreshold);

  // Write the single box centered at (cx,0,0), half-extent 1: min=(cx-1,-1,-1), max=(cx+1,1,1).
  auto set_cx = [&](double cx) {
    store_aabb(*mesh.aabb_field, node, make_aabb(cx, 0.0, 0.0, 1.0));
    component.modify_on_host();
  };

  set_cx(0.0);
  // No snapshot yet: always needs rebuild.
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));
  rebuilder.snapshot(*mesh.bulk, input, input);

  // Identical geometry: no rebuild.
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // cx=0.2: corner displacement 0.2 < 0.5 → no rebuild.
  set_cx(0.2);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // cx=1.0: corner displacement 1.0 > 0.5 → rebuild.
  set_cx(1.0);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Take new snapshot at cx=1.
  rebuilder.snapshot(*mesh.bulk, input, input);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // cx=0.6: displacement from cx=1 is 0.4 < 0.5 → no rebuild.
  set_cx(0.6);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // cx=0.4: displacement from cx=1 is 0.6 > 0.5 → rebuild.
  set_cx(0.4);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));
}

// Separate target and source thresholds: the tighter side fires independently.
//
//   target threshold = 0.3, source threshold = 0.8
//   Snapshot at cx=0 for both.
//
//   Target moves to cx=0.4 (disp 0.4 > 0.3) → target side fires.
//   Source stays at cx=0    → source side quiet.
//   Expected: rebuild triggered by target alone.
//
//   After new snapshot at cx=0.4 / cx=0:
//   Target moves to cx=0.6 (disp 0.2 < 0.3) → target quiet.
//   Source moves to cx=0.9 (disp 0.9 > 0.8) → source side fires.
//   Expected: rebuild triggered by source alone.
TEST(RebuildOnAABBDisplacement, SeparateTargetAndSourceThresholds) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  // 2-node mesh: node 1 = target (selector target_part), node 2 = source (selector source_part), so target and
  // source geometry are independently controllable through one shared `aabb` field.
  auto mesh = make_geom_mesh(2);
  TestComponent component(*mesh.aabb_field);
  TestInput tgt_input(*mesh.target_part, component);
  TestInput src_input(*mesh.source_part, component);

  constexpr float kTargetThreshold = 0.3f;
  constexpr float kSourceThreshold = 0.8f;
  RebuildOnAABBDisplacement<double, TestMemSpace> rebuilder(kTargetThreshold, kSourceThreshold);

  auto set_tgt = [&](double cx) {
    store_aabb(*mesh.aabb_field, mesh.nodes[0], make_aabb(cx, 0.0, 0.0, 1.0));
    component.modify_on_host();
  };
  auto set_src = [&](double cx) {
    store_aabb(*mesh.aabb_field, mesh.nodes[1], make_aabb(cx, 0.0, 0.0, 1.0));
    component.modify_on_host();
  };

  set_tgt(0.0);
  set_src(0.0);
  // No snapshot yet: always needs rebuild.
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));
  rebuilder.snapshot(*mesh.bulk, tgt_input, src_input);

  // Both unchanged: no rebuild.
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Target moves 0.4 > 0.3, source stays: target side fires.
  set_tgt(0.4);
  set_src(0.0);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Target moves 0.2 < 0.3, source moves 0.5 < 0.8: neither fires.
  set_tgt(0.2);
  set_src(0.5);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Source moves 0.9 > 0.8, target stays at 0: source side fires even though target is quiet.
  set_tgt(0.0);
  set_src(0.9);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Take snapshot with target at 0.4 and source at 0.
  set_tgt(0.4);
  set_src(0.0);
  rebuilder.snapshot(*mesh.bulk, tgt_input, src_input);

  // Target moves from 0.4 to 0.6 (disp 0.2 < 0.3): quiet.  Source stays at 0: quiet.
  set_tgt(0.6);
  set_src(0.0);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Source now moves 0.9 > 0.8: source fires, target still quiet.
  set_src(0.9);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));
}

// ---- RebuildOnOBBDisplacement ----

// Pure-translation threshold test.
//
//   Unit cube OBBs (half-extent 0.5) at origin; threshold 0.5.
//   With R_rel = I (no rotation change), the escape condition on axis x reduces to
//     |T[x]| > threshold = 0.5.
//
//   cx=0.3 → displacement 0.3 < 0.5 → no rebuild.
//   cx=0.7 → displacement 0.7 > 0.5 → rebuild.
TEST(RebuildOnOBBDisplacement, ThresholdBehavior_TranslationOnly) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_geom_mesh(1);
  auto node = mesh.nodes[0];
  const stk::mesh::Selector sel = mesh.meta->universal_part();
  // `input` only supplies the selector/rank; the OBB geometry comes from the rebuilder's own OBB component.
  TestComponent aabb_component(*mesh.aabb_field);
  TestInput input(sel, aabb_component);

  constexpr double kHalf = 0.5;
  constexpr double kThreshold = 0.5;
  TestOBBComponent obb_component(*mesh.obb_field);

  auto set_cx = [&](double cx) {
    store_obb(*mesh.obb_field, node,
              TestOBB{Point<double>{cx, 0.0, 0.0}, Quaternion<double>::identity(), kHalf, kHalf, kHalf});
    obb_component.modify_on_host();
  };
  set_cx(0.0);

  RebuildOnOBBDisplacement<double, TestMemSpace> rebuilder(obb_component, kThreshold);

  // No snapshot yet: always needs rebuild.
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));
  rebuilder.snapshot(*mesh.bulk, input, input);

  // Unchanged: no rebuild.
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Displacement 0.3 < 0.5: no rebuild.
  set_cx(0.3);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Displacement 0.7 > 0.5: rebuild.
  set_cx(0.7);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // New snapshot at cx=0.7.
  rebuilder.snapshot(*mesh.bulk, input, input);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Displacement from 0.7: |0.7-0.4|=0.3 < 0.5 → no rebuild.
  set_cx(0.4);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Displacement from 0.7: |0.7-0.1|=0.6 > 0.5 → rebuild.
  set_cx(0.1);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));
}

// Rotation-only threshold test.
//
//   Unit cube OBBs (half-extent 0.5) at origin; threshold 0.1.
//   For a rotation θ around z with no translation the escape condition on axis k=0 is:
//     (|cos θ| + |sin θ|) * 0.5 > 0.5 + 0.1 = 0.6
//     |cos θ| + |sin θ|          > 1.2
//   This triggers at θ ≈ 17°; all three axes must pass.
//
//   Relative θ=5°:  (cos5°+sin5°)*0.5 ≈ 0.542 ≤ 0.6 → no rebuild.
//   Relative θ=25°: (cos25°+sin25°)*0.5 ≈ 0.664 > 0.6 → rebuild.
TEST(RebuildOnOBBDisplacement, ThresholdBehavior_RotationOnly) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto mesh = make_geom_mesh(1);
  auto node = mesh.nodes[0];
  const stk::mesh::Selector sel = mesh.meta->universal_part();
  TestComponent aabb_component(*mesh.aabb_field);
  TestInput input(sel, aabb_component);  // supplies the selector/rank only

  constexpr double kHalf = 0.5;
  constexpr double kThreshold = 0.1;
  const double pi = Kokkos::numbers::pi_v<double>;
  TestOBBComponent obb_component(*mesh.obb_field);

  // Rotation by angle theta around z: q = {cos(θ/2), 0, 0, sin(θ/2)}.
  auto set_theta = [&](double theta) {
    const Quaternion<double> q{std::cos(theta / 2.0), 0.0, 0.0, std::sin(theta / 2.0)};
    store_obb(*mesh.obb_field, node, TestOBB{Point<double>{0.0, 0.0, 0.0}, q, kHalf, kHalf, kHalf});
    obb_component.modify_on_host();
  };
  set_theta(0.0);

  RebuildOnOBBDisplacement<double, TestMemSpace> rebuilder(obb_component, kThreshold);

  // No snapshot yet: always needs rebuild.
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));
  rebuilder.snapshot(*mesh.bulk, input, input);

  // θ=0 (identity): no rebuild.
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Relative θ=5° from snapshot: (cos5°+sin5°)*0.5 ≈ 0.542 ≤ 0.6 → no rebuild.
  set_theta(5.0 * pi / 180.0);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Relative θ=30° from snapshot: (cos30°+sin30°)*0.5 ≈ 0.683 > 0.6 → rebuild.
  set_theta(30.0 * pi / 180.0);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Snapshot at θ=30°; subsequent checks are relative to this orientation.
  rebuilder.snapshot(*mesh.bulk, input, input);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Relative rotation from snapshot: 35°−30°=5° → (cos5°+sin5°)*0.5 ≈ 0.542 ≤ 0.6 → no rebuild.
  set_theta(35.0 * pi / 180.0);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, input, input));

  // Relative rotation from snapshot: 55°−30°=25° → (cos25°+sin25°)*0.5 ≈ 0.664 > 0.6 → rebuild.
  set_theta(55.0 * pi / 180.0);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, input, input));
}

// Separate target/source thresholds: each side fires independently.
//
//   target threshold=0.3, source threshold=0.8.
//   Snapshot both at origin.
//
//   Target → 0.4 (0.4 > 0.3): target fires.  Source unchanged: quiet.  Expected: rebuild.
//   Target → 0.2 (0.2 < 0.3), source → 0.5 (0.5 < 0.8): neither fires.
//   Source → 0.9 (0.9 > 0.8), target quiet: source fires.  Expected: rebuild.
//
//   After snapshot at target=0.4, source=0:
//   Target → 0.55 (disp 0.15 < 0.3): quiet.
//   Source → 0.9 (0.9 > 0.8): source fires.  Expected: rebuild.
TEST(RebuildOnOBBDisplacement, SeparateTargetAndSourceThresholds) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  // 2-node mesh: node 1 = target, node 2 = source; one shared `obb` field, two selectors → independent geometry.
  auto mesh = make_geom_mesh(2);
  TestComponent aabb_component(*mesh.aabb_field);
  TestInput tgt_input(*mesh.target_part, aabb_component);  // supply selectors/rank only
  TestInput src_input(*mesh.source_part, aabb_component);

  constexpr double kHalf = 0.5;
  constexpr double kTargetThreshold = 0.3;
  constexpr double kSourceThreshold = 0.8;
  TestOBBComponent obb_component(*mesh.obb_field);

  auto set_tgt = [&](double cx) {
    store_obb(*mesh.obb_field, mesh.nodes[0],
              TestOBB{Point<double>{cx, 0.0, 0.0}, Quaternion<double>::identity(), kHalf, kHalf, kHalf});
    obb_component.modify_on_host();
  };
  auto set_src = [&](double cx) {
    store_obb(*mesh.obb_field, mesh.nodes[1],
              TestOBB{Point<double>{cx, 0.0, 0.0}, Quaternion<double>::identity(), kHalf, kHalf, kHalf});
    obb_component.modify_on_host();
  };

  set_tgt(0.0);
  set_src(0.0);

  // Same OBB component read over each side's selector: target → node 1, source → node 2.
  RebuildOnOBBDisplacement<double, TestMemSpace> rebuilder(obb_component, obb_component, kTargetThreshold,
                                                           kSourceThreshold);

  // No snapshot yet: always needs rebuild.
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));
  rebuilder.snapshot(*mesh.bulk, tgt_input, src_input);

  // Both unchanged: no rebuild.
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Target moves 0.4 > 0.3, source stays: target fires.
  set_tgt(0.4);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Target 0.2 < 0.3, source 0.5 < 0.8: neither fires.
  set_tgt(0.2);
  set_src(0.5);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Source 0.9 > 0.8, target at 0.2: source fires.
  set_src(0.9);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Snapshot with target=0.4, source=0.
  set_tgt(0.4);
  set_src(0.0);
  rebuilder.snapshot(*mesh.bulk, tgt_input, src_input);

  // Target to 0.55: disp=|0.55−0.4|=0.15 < 0.3 → quiet.  Source at 0: quiet.
  set_tgt(0.55);
  EXPECT_FALSE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));

  // Source to 0.9: disp=0.9 > 0.8 → source fires.
  set_src(0.9);
  EXPECT_TRUE(rebuilder.needs_rebuild(*mesh.bulk, tgt_input, src_input));
}

// End-to-end test through ManagedNeighborList: large displacement forces rebuild
// and updates the snapshot; a subsequent same-geometry call returns the cache.
TEST_F(STKDeterministicFixture, Rebuilder_AABBDisplacement_EndToEnd) {
  constexpr float kThreshold = 0.3f;
  auto managed = make_neighbor_list_builder<STKList>()
                     .exec_space(TestExecSpace{})
                     .manage(RebuildOnAABBDisplacement<double, TestMemSpace>{kThreshold});

  // Initial: 1 overlapping pair; snapshot taken.
  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  // Far-away targets: displacement >> threshold → rebuild → 0 pairs; snapshot updated.
  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), PairSet{});

  // Same geometry as r2: displacement == 0 < threshold → cache → still 0 pairs.
  auto r3 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_FALSE(r3.rebuilt);
  EXPECT_EQ(collect_pairs(r3.list), PairSet{});
}

// End-to-end through ManagedNeighborList with an OBB-displacement rebuilder.  The OBB rebuilder watches the
// fixture's `obb_component_` (over the search input's selector) and drives rebuild decisions, even though the
// search itself runs on the AABB inputs.  This exercises the managed + OBB-rebuilder integration path.
TEST_F(STKDeterministicFixture, Rebuilder_OBBDisplacement_EndToEnd) {
  constexpr double kThreshold = 0.5;
  auto managed = make_neighbor_list_builder<STKList>()
                     .exec_space(TestExecSpace{})
                     .manage(RebuildOnOBBDisplacement<double, TestMemSpace>{obb_component_, kThreshold});

  // Initial build: OBB snapshot taken; the AABB geometry yields 1 pair.
  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  // Displace target node 1's OBB far beyond the threshold → OBB rebuilder fires → rebuild.
  store_obb(*obb_field_, nodes_[0],
            TestOBB{Point<double>{50.0, 0.0, 0.0}, Quaternion<double>::identity(), 0.5, 0.5, 0.5});
  obb_component_.modify_on_host();
  auto r2 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);

  // No further OBB change → contained within the inflated snapshot → cache reused.
  auto r3 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_FALSE(r3.rebuilt);
}

// ---- RebuilderChain via operator| ----

// (NeverRebuild | AlwaysRebuild): prior returns false, so next is always evaluated
// and returns true → chain always rebuilds.
TEST_F(STKDeterministicFixture, Rebuilder_Chain_NeverOrAlways_BehavesLikeAlways) {
  auto managed =
      make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{} | AlwaysRebuild{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), PairSet{});  // AlwaysRebuild forced rebuild
}

// (NeverRebuild | NeverRebuild): both return false → chain never rebuilds after first.
TEST_F(STKDeterministicFixture, Rebuilder_Chain_NeverOrNever_BehavesLikeNever) {
  auto managed =
      make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(NeverRebuild{} | NeverRebuild{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_FALSE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}}));  // cache: both returned false
}

// (RebuildOnEntityChange | AlwaysRebuild): entity count unchanged (static mesh),
// so RebuildOnEntityChange returns false; AlwaysRebuild is then evaluated and
// returns true → chain always rebuilds.
TEST_F(STKDeterministicFixture, Rebuilder_Chain_EntityChangeOrAlways_BehavesLikeAlways) {
  auto managed = make_neighbor_list_builder<STKList>()
                     .exec_space(TestExecSpace{})
                     .manage(RebuildOnEntityChange<TestMemSpace>{} | AlwaysRebuild{});

  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  auto far_tgt = make_far_target_boxes(*this);
  auto r2 = managed.update(*bulk_, far_tgt, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), PairSet{});  // AlwaysRebuild forced rebuild
}

// ---- RebuildOnAABBDisplacement | RebuildOnEntityChange combined chain ----
//
// AABB is prior_ (evaluated first); EntityChange is next_ (evaluated only when AABB returns
// false).  The count guard added to RebuildOnAABBDisplacement::needs_rebuild ensures it
// returns true immediately when the box count changes, without accessing out-of-bounds in
// the snapshot view.  The three tests below cover: entity add (2→6), entity remove (6→2),
// and entity swap (same count, different identity).

// Entity add: target count increases 2→4.  AABB count guard fires and returns true (prior_),
// so EntityChange is never evaluated.  A rebuild occurs; the pair set reflects the larger target set.
TEST_F(STKDeterministicFixture, CombinedRebuilder_AABBSafeOnEntityAdd) {
  constexpr float kThreshold = 0.3f;
  auto managed =
      make_neighbor_list_builder<STKList>()
          .exec_space(TestExecSpace{})
          .manage(RebuildOnAABBDisplacement<double, TestMemSpace>{kThreshold} | RebuildOnEntityChange<TestMemSpace>{});

  // Initial build: 2 targets {1,2}, 2 sources {3,4} → 1 pair; both snapshots recorded.
  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}}));

  // 4-entity target {1,2,5,6}: count 2→4 → AABB count guard fires → rebuild.
  auto r2 = managed.update(*bulk_, overlapping_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}, {2, 0}, {3, 0}}));
}

// Entity remove: target count decreases 4→2.  AABB count guard fires and returns true,
// so EntityChange is never evaluated.  A rebuild occurs; the pair set shrinks accordingly.
TEST_F(STKDeterministicFixture, CombinedRebuilder_AABBSafeOnEntityRemove) {
  constexpr float kThreshold = 0.3f;
  auto managed =
      make_neighbor_list_builder<STKList>()
          .exec_space(TestExecSpace{})
          .manage(RebuildOnAABBDisplacement<double, TestMemSpace>{kThreshold} | RebuildOnEntityChange<TestMemSpace>{});

  // Start with nodes {5,6} in target_part: 4-entity target {1,2,5,6} vs source {3,4}.
  move_nodes({5, 6}, target_part_, shared_part_);
  auto r1 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), (PairSet{{0, 0}, {2, 0}, {3, 0}}));

  // Remove {5,6} from target_part (mesh modification → synchronized_count advances): count 4→2. Both the
  // displacement rebuilder (conservatively, on the mesh mod) and EntityChange fire; the rebuilt list has 1 pair
  // and the displacement check never reads stale geometry for the removed nodes.
  move_nodes({5, 6}, shared_part_, target_part_);
  auto r2 = managed.update(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  EXPECT_TRUE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), (PairSet{{0, 0}}));
}

// Entity swap WITHIN the fixed target selector: nodes {1,2} leave target_part and nodes {5,6} enter it, carrying
// the same coordinates — same count (2→2), same coordinates, different entity identities.  Because a swap is a mesh
// modification, synchronized_count() advances, so the displacement rebuilder (prior_) conservatively rebuilds, and
// EntityChange (next_) independently detects the identity change.  Both fire.  (Entity-aligned displacement is
// defined only on a stable entity set; any entity-set change rides in on a mesh modification, which it treats as
// rebuild-worthy — so the displacement rebuilder no longer "ignores" identity swaps.)
TEST_F(STKDeterministicFixture, CombinedRebuilder_EntityChangeFiresOnEntitySwap) {
  constexpr float kThreshold = 0.3f;
  auto chain = RebuildOnAABBDisplacement<double, TestMemSpace>{kThreshold} | RebuildOnEntityChange<TestMemSpace>{};

  // Snapshot the initial state (target_part = {1,2}).
  chain.snapshot(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);

  // node5 ← node1's coords, node6 ← node2's coords; then swap {1,2} out of target_part and {5,6} in.
  store_aabb(*aabb_field_, nodes_[4], make_aabb(0.0, 0.0, 0.0, 2.0));
  store_aabb(*aabb_field_, nodes_[5], make_aabb(100., 100., 100., 0.5));
  aabb_component_.modify_on_host();
  move_nodes({1, 2}, shared_part_, target_part_);
  move_nodes({5, 6}, target_part_, shared_part_);

  // A mesh modification advances synchronized_count(), so displacement (prior_) conservatively rebuilds...
  EXPECT_TRUE(chain.prior().needs_rebuild(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_));
  // ...and the full chain rebuilds (EntityChange also detects the entity identity change).
  EXPECT_TRUE(chain.needs_rebuild(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_));
}

// =============================================================================
// Group 7 (continued) — Zero-target / zero-source edge cases
//
// When the target or source input is empty the correct result is trivially the
// empty list.  Rebuilders must not fire spurious rebuilds in this state, and
// their internal Kokkos reductions must not produce garbage from identity values
// over empty ranges (Kokkos::Max identity is INT_MIN, which != 0 and would
// incorrectly report a change without the explicit n==0 guard).
// =============================================================================

// Helper: an input over an empty selector (no node is in both target and source parts) → enumerates 0 entities.
inline TestInput make_empty_boxes(STKDeterministicFixture& f) {
  return TestInput(*f.target_part_ & *f.source_part_, f.aabb_component_);
}

// --- RebuildOnEntityChange with zero entities ---

// entities_changed on two empty views must return false.
// Without the n==0 guard, Kokkos::Max over an empty range returns INT_MIN
// (the identity), which is != 0 and would falsely signal a change.
TEST_F(STKDeterministicFixture, EntityChange_EmptyTargets_DoesNotSignalChange) {
  RebuildOnEntityChange<TestMemSpace> rebuilder;
  auto empty = make_empty_boxes(*this);

  // No snapshot yet → always rebuilds on first call.
  EXPECT_TRUE(rebuilder.needs_rebuild(*bulk_, empty, empty));

  // After snapshot of empty inputs, subsequent call with still-empty inputs → no change.
  rebuilder.snapshot(*bulk_, empty, empty);
  EXPECT_FALSE(rebuilder.needs_rebuild(*bulk_, empty, empty));
}

// Snapshot with real entities, then empty the SAME (fixed) target selector via a mesh modification (its nodes leave
// target_part) → count drops 2→0 → rebuild (the entity sequence did change).
TEST_F(STKDeterministicFixture, EntityChange_TransitionToEmpty_SignalsRebuild) {
  RebuildOnEntityChange<TestMemSpace> rebuilder;

  rebuilder.snapshot(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);
  // Remove nodes {1,2} from target_part: the target selector now enumerates 0 entities → count 2 → 0.
  move_nodes({1, 2}, nullptr, target_part_);
  EXPECT_TRUE(rebuilder.needs_rebuild(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_));
}

// --- RebuildOnAABBDisplacement with zero targets/sources ---

// After a snapshot with non-empty boxes, presenting zero-target boxes must
// suppress the displacement check entirely and return false.  Without the guard,
// the size mismatch (snapshot stores 2 boxes, current has 0) would trigger a
// count-change rebuild; but an empty target set trivially produces an empty list
// regardless of how far boxes "moved".
TEST_F(STKDeterministicFixture, AABBDisplacement_EmptyTargets_DoesNotFireAfterSnapshot) {
  RebuildOnAABBDisplacement<double, TestMemSpace> rebuilder(0.01f);
  auto empty = make_empty_boxes(*this);

  // First call always rebuilds (no snapshot yet).
  EXPECT_TRUE(rebuilder.needs_rebuild(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_));
  rebuilder.snapshot(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_);

  // Empty target set → result is trivially empty; displacement check suppressed.
  EXPECT_FALSE(rebuilder.needs_rebuild(*bulk_, empty, disjoint_source_boxes_));

  // Symmetric: empty source set → also suppressed.
  EXPECT_FALSE(rebuilder.needs_rebuild(*bulk_, disjoint_target_boxes_, empty));

  // Both empty → also suppressed.
  EXPECT_FALSE(rebuilder.needs_rebuild(*bulk_, empty, empty));
}

// With zero targets and zero sources from the very start, corners_moved over
// an empty range must return false (not crash or return garbage from the reducer).
TEST_F(STKDeterministicFixture, AABBDisplacement_EmptyFromStart_NoRebuildAfterFirstBuild) {
  RebuildOnAABBDisplacement<double, TestMemSpace> rebuilder(0.01f);
  auto empty = make_empty_boxes(*this);

  // First call: no snapshot → rebuilds.
  EXPECT_TRUE(rebuilder.needs_rebuild(*bulk_, empty, empty));
  rebuilder.snapshot(*bulk_, empty, empty);

  // Same empty inputs → nothing changed, no rebuild.
  EXPECT_FALSE(rebuilder.needs_rebuild(*bulk_, empty, empty));

  // Presenting non-empty boxes after an empty snapshot → count change, rebuilds.
  EXPECT_TRUE(rebuilder.needs_rebuild(*bulk_, disjoint_target_boxes_, disjoint_source_boxes_));
}

// --- ManagedNeighborList with zero targets end-to-end ---

// With zero targets the managed list should build once and then never rebuild
// (there is nothing to search; the result is always empty).
TEST_F(STKDeterministicFixture, ManagedList_ZeroTargets_NeverRebuildsAfterFirstBuild) {
  auto empty = make_empty_boxes(*this);
  auto managed = make_neighbor_list_builder<STKList>()
                     .exec_space(TestExecSpace{})
                     .manage(RebuildOnAABBDisplacement<double, TestMemSpace>{0.01f});

  // First update: list doesn't exist yet, must build.
  auto r1 = managed.update(*bulk_, empty, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);
  EXPECT_EQ(collect_pairs(r1.list), PairSet{});

  // Second update with same empty targets: displacement check suppressed → no rebuild.
  auto r2 = managed.update(*bulk_, empty, disjoint_source_boxes_);
  EXPECT_FALSE(r2.rebuilt);
  EXPECT_EQ(collect_pairs(r2.list), PairSet{});

  // Even with very different source geometry, zero targets → no rebuild needed.
  auto r3 = managed.update(*bulk_, empty, overlapping_source_boxes_);
  EXPECT_FALSE(r3.rebuilt);
  EXPECT_EQ(collect_pairs(r3.list), PairSet{});
}

// EntityChange rebuilder also must not fire for empty→empty transitions inside
// a managed list.
TEST_F(STKDeterministicFixture, ManagedList_ZeroTargets_EntityChangeDoesNotFire) {
  auto empty = make_empty_boxes(*this);
  auto managed =
      make_neighbor_list_builder<STKList>().exec_space(TestExecSpace{}).manage(RebuildOnEntityChange<TestMemSpace>{});

  auto r1 = managed.update(*bulk_, empty, disjoint_source_boxes_);
  EXPECT_TRUE(r1.rebuilt);

  // Entity snapshot is empty; presenting empty targets again → no entity change.
  auto r2 = managed.update(*bulk_, empty, disjoint_source_boxes_);
  EXPECT_FALSE(r2.rebuilt);
}

// =============================================================================
// Cross-rank validation of the non-periodic STK multi-rank build (coarse_search + cooperative ghosting).
//
// Replicated point set; point i has global id i and is owned by rank i % nprocs, declared only on its owning rank.
// The build ghosts remote source owners. Each rank checks ONLY its owned target rows against the replicated N²
// gid-oracle (filtered to targets it owns), mapping the list's entities back to global ids — no cross-rank gather,
// and the union over ranks is the full global oracle. Runs at any rank count (np=1 degenerate; np=2,4 exercise
// ghosting).
// =============================================================================
void run_multirank_stk_validation(const std::vector<std::array<double, 3>>& positions, double r) {
  const int N = static_cast<int>(positions.size());
  auto mesh = make_distributed_node_mesh_with_aabb(N);
  auto& bulk = *mesh.bulk;
  const int my_rank = bulk.parallel_rank();
  const int nprocs = bulk.parallel_size();
  const stk::mesh::Selector owned = mesh.meta->locally_owned_part();

  // Replicated boxes (index == global id); store on owned nodes only.
  std::vector<TestAABB> boxes(N);
  for (int i = 0; i < N; ++i) {
    boxes[i] = make_aabb(positions[i][0], positions[i][1], positions[i][2], r);
    if (i % nprocs != my_rank) continue;
    auto node = bulk.get_entity(stk::topology::NODE_RANK, static_cast<stk::mesh::EntityId>(i + 1));
    ASSERT_TRUE(bulk.is_valid(node)) << "owned node " << (i + 1) << " missing on rank " << my_rank;
    store_aabb(*mesh.aabb_field, node, boxes[i]);
  }
  TestComponent component(*mesh.aabb_field);
  component.modify_on_host();
  TestInput input{owned, component};

  STKList list = make_neighbor_list_builder<STKList>()
                     .exec_space(TestExecSpace{})
                     .target_input(input)
                     .source_input(input)
                     .broad_phase(ExcludeSelfInteraction{})
                     .build(bulk);

  // Replicated gid-oracle, kept only for rows this rank owns.
  const PairSet full = oracle_pairs_no_self(boxes);
  PairSet expected;
  for (const auto& p : full)
    if (static_cast<int>(p.first % static_cast<size_t>(nprocs)) == my_rank) expected.insert(p);

  // The list's rows for this rank's owned targets, mapped to global ids (valid for ghosted sources too).
  PairSet actual;
  for (size_t t = 0; t < list.num_targets(); ++t) {
    const size_t tgid = static_cast<size_t>(bulk.identifier(list.target_entity(t))) - 1;
    for (size_t k = 0; k < list.num_neighbors(t); ++k) {
      const size_t sgid = static_cast<size_t>(bulk.identifier(list.get_neighbor(t, k))) - 1;
      actual.insert({tgid, sgid});
    }
  }

  EXPECT_EQ(actual.size(), expected.size())
      << "rank " << my_rank << ": list=" << actual.size() << " oracle=" << expected.size();
  for (const auto& p : expected)
    EXPECT_TRUE(actual.count(p)) << "rank " << my_rank << " MISSING: target=" << p.first << " source=" << p.second;
  for (const auto& p : actual)
    EXPECT_TRUE(expected.count(p)) << "rank " << my_rank << " SPURIOUS: target=" << p.first << " source=" << p.second;
}

TEST(STKSearchNeighborList, MultiRankRandomN2Validation) {
  constexpr int kN = 60;
  constexpr size_t kSeed = 11;
  constexpr double kDomain = 10.0;
  constexpr double kRadius = 1.0;
  std::vector<std::array<double, 3>> positions(kN);
  for (int i = 0; i < kN; ++i) {
    openrand::Philox rng = mundy::make_philox(kSeed, static_cast<uint32_t>(i));
    positions[i] = {rng.uniform<double>(0.0, kDomain), rng.uniform<double>(0.0, kDomain),
                    rng.uniform<double>(0.0, kDomain)};
  }
  run_multirank_stk_validation(positions, kRadius);
}

}  // namespace
}  // namespace search
}  // namespace mundy
