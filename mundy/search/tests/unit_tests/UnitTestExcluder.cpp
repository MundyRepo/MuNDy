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

/// \file UnitTestExcluder.cpp
/// \brief Unit tests for Excluder.hpp: concept checks, per-excluder behavior, and chain semantics.
///
/// Test structure:
///   Group 1 — Concept checks: compile-time static_asserts that all excluder types satisfy ExcluderType.
///   Group 2 — NoExcluder: never excludes any candidate.
///   Group 3 — ExcludeSelfInteraction: entity-based self-exclusion, zero-shift semantics.
///   Group 4 — ExcluderChain: OR semantics, multi-level chaining, setup propagation.
///   Group 5 — ExcludeSymmetricDuplicates: selector intersection logic and directional suppression.
///   Group 6 — ExcludeConnectedEntities: excludes element pairs sharing a connected node.
///   Group 7 — ExcludeNonIntersectingOBBs: ordinal-indexed OBB views, separated vs intersecting pairs.

// Mundy
#include <MundySearch_config.hpp>  // for HAVE_MUNDYSEARCH_*

#ifdef HAVE_MUNDYSEARCH_ARBORX

// External
#include <gtest/gtest.h>

// C++ core
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

// STK mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy geom
#include <mundy_geom/primitives/OBB.hpp>  // for mundy::OBB, mundy::Point, mundy::intersects

// Mundy math
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3

// Mundy mesh
#include <mundy_mesh/FieldComponent.hpp>  // for mundy::mesh::OBBFieldComponent

// Mundy search
#include <mundy_search/Excluder.hpp>
#include <mundy_search/SearchCandidate.hpp>

namespace mundy {
namespace search {
namespace {

// =============================================================================
// Aliases
// =============================================================================

using Cand = NeighborSearchCandidate<size_t>;
using Vec3f = mundy::Vector3<float>;
using PeriodicCand = PeriodicNeighborSearchCandidate<Vec3f, size_t>;

// =============================================================================
// Group 1 — Compile-time concept checks
// =============================================================================

static_assert(ExcluderType<NoExcluder>);
static_assert(ExcluderType<ExcludeSelfInteraction>);
static_assert(ExcluderType<ExcludeConnectedEntities>);
static_assert(ExcluderType<ExcludeSymmetricDuplicates>);
static_assert(ExcluderType<ExcludeNonIntersectingOBBs<double>>);
static_assert(ExcluderType<ExcludeNonIntersectingOBBs<float>>);
static_assert(ExcluderType<ExcluderChain<NoExcluder, ExcludeSelfInteraction>>);
static_assert(ExcluderType<ExcluderChain<NoExcluder, ExcludeSymmetricDuplicates>>);
static_assert(ExcluderType<ExcluderChain<NoExcluder, ExcludeNonIntersectingOBBs<double>>>);
static_assert(
    ExcluderType<ExcluderChain<ExcluderChain<NoExcluder, ExcludeSelfInteraction>, ExcludeSymmetricDuplicates>>);
// const-qualified types must NOT satisfy ExcluderType because setup() is non-const.
static_assert(!ExcluderType<const NoExcluder>);
static_assert(!ExcluderType<const ExcludeSelfInteraction>);
static_assert(!ExcluderType<const ExcludeNonIntersectingOBBs<double>>);

// =============================================================================
// Minimal STK mesh factory for excluder tests
// =============================================================================

// 4 nodes: nodes 1,2 -> part_a; nodes 3,4 -> part_b.
struct TwoPartMesh {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Part* part_a = nullptr;
  stk::mesh::Part* part_b = nullptr;
  stk::mesh::Entity node[5];  // 1-indexed: node[1]..node[4]
};

TwoPartMesh make_two_part_mesh() {
  TwoPartMesh m;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  m.meta = builder.create_meta_data();
  m.meta->use_simple_fields();
  m.part_a = &m.meta->declare_part("part_a", stk::topology::NODE_RANK);
  m.part_b = &m.meta->declare_part("part_b", stk::topology::NODE_RANK);
  m.bulk = builder.create(m.meta);
  m.meta->commit();

  m.bulk->modification_begin();
  for (int id = 1; id <= 4; ++id) m.bulk->declare_node(id);
  for (int id : {1, 2}) {
    auto n = m.bulk->get_entity(stk::topology::NODE_RANK, id);
    m.bulk->change_entity_parts(n, stk::mesh::PartVector{m.part_a}, stk::mesh::PartVector{});
  }
  for (int id : {3, 4}) {
    auto n = m.bulk->get_entity(stk::topology::NODE_RANK, id);
    m.bulk->change_entity_parts(n, stk::mesh::PartVector{m.part_b}, stk::mesh::PartVector{});
  }
  m.bulk->modification_end();

  for (int id = 1; id <= 4; ++id) m.node[id] = m.bulk->get_entity(stk::topology::NODE_RANK, id);
  return m;
}

// Build a NeighborSearchCandidate from array ordinals and explicit entities.
Cand make_cand(size_t t_ord, size_t s_ord, stk::mesh::Entity t_ent, stk::mesh::Entity s_ent) {
  return Cand(t_ord, s_ord, t_ent, s_ent);
}

// Write one node's OBB into a 10-scalar field (center 0-2, quat wxyz 3-6, half-extents 7-9).
void store_obb(stk::mesh::Field<double>& field, stk::mesh::Entity node, const OBB<double>& obb) {
  double* d = stk::mesh::field_data(field, node);
  const auto& c = obb.center();
  const auto& q = obb.orientation();
  const auto& h = obb.half_extents();
  d[0] = c[0];
  d[1] = c[1];
  d[2] = c[2];
  d[3] = q.w();
  d[4] = q.x();
  d[5] = q.y();
  d[6] = q.z();
  d[7] = h[0];
  d[8] = h[1];
  d[9] = h[2];
}

// Mesh with an `obb` node field; node id i+1 carries obbs[i] (so the universal-selector enumeration gives dense
// ordinal i -> obbs[i]). The first `split` nodes go to part_a, the rest to part_b, for asymmetric target/source tests.
struct ObbMesh {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Field<double>* obb_field = nullptr;
  stk::mesh::Part* part_a = nullptr;
  stk::mesh::Part* part_b = nullptr;
  std::vector<stk::mesh::Entity> node;  // 0-indexed: node[i] is entity id i+1

  mundy::mesh::OBBFieldComponent<double> component() const {
    return mundy::mesh::OBBFieldComponent<double>(*obb_field);
  }
};

ObbMesh make_obb_mesh(const std::vector<OBB<double>>& obbs, size_t split = 0) {
  ObbMesh m;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  m.meta = builder.create_meta_data();
  m.meta->use_simple_fields();
  m.part_a = &m.meta->declare_part("part_a", stk::topology::NODE_RANK);
  m.part_b = &m.meta->declare_part("part_b", stk::topology::NODE_RANK);
  auto& f = m.meta->declare_field<double>(stk::topology::NODE_RANK, "obb_field");
  stk::mesh::put_field_on_mesh(f, m.meta->universal_part(), 10, nullptr);
  m.obb_field = &f;
  m.bulk = builder.create(m.meta);
  m.meta->commit();

  const int n = static_cast<int>(obbs.size());
  m.bulk->modification_begin();
  for (int id = 1; id <= n; ++id) m.bulk->declare_node(id);
  for (int id = 1; id <= n; ++id) {
    auto node = m.bulk->get_entity(stk::topology::NODE_RANK, id);
    stk::mesh::Part* part = (static_cast<size_t>(id - 1) < split) ? m.part_a : m.part_b;
    m.bulk->change_entity_parts(node, stk::mesh::PartVector{part}, stk::mesh::PartVector{});
  }
  m.bulk->modification_end();

  m.node.resize(n);
  for (int id = 1; id <= n; ++id) {
    m.node[id - 1] = m.bulk->get_entity(stk::topology::NODE_RANK, id);
    store_obb(f, m.node[id - 1], obbs[id - 1]);
  }
  return m;
}

// =============================================================================
// Group 2 — NoExcluder
// =============================================================================

TEST(NoExcluderTest, NeverExcludes) {
  auto m = make_two_part_mesh();
  NoExcluder ex;
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // Same entity (would be self-interaction).
  EXPECT_FALSE(ex(make_cand(0, 0, m.node[1], m.node[1])));
  // Different entities.
  EXPECT_FALSE(ex(make_cand(0, 1, m.node[1], m.node[2])));
  EXPECT_FALSE(ex(make_cand(1, 0, m.node[2], m.node[1])));
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
}

TEST(NoExcluderTest, ChainReturnsSelf) {
  NoExcluder base;
  auto chain = base.exclude(NoExcluder{});
  auto prior = chain.prior_excluder();
  auto appended = chain.appended_excluder();
  (void)prior;
  (void)appended;
  // Chain compiles and exposes accessor types.
  static_assert(std::is_same_v<decltype(chain), ExcluderChain<NoExcluder, NoExcluder>>);
}

// =============================================================================
// Group 3 — ExcludeSelfInteraction
// =============================================================================

TEST(ExcludeSelfInteractionTest, ExcludesSameEntity) {
  auto m = make_two_part_mesh();
  ExcludeSelfInteraction ex;
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_TRUE(ex(make_cand(0, 0, m.node[1], m.node[1]))) << "same entity should be excluded.";
  EXPECT_TRUE(ex(make_cand(2, 2, m.node[3], m.node[3])));
}

TEST(ExcludeSelfInteractionTest, RetainsDifferentEntities) {
  auto m = make_two_part_mesh();
  ExcludeSelfInteraction ex;
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_FALSE(ex(make_cand(0, 1, m.node[1], m.node[2])));
  EXPECT_FALSE(ex(make_cand(1, 0, m.node[2], m.node[1])));
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
}

TEST(ExcludeSelfInteractionTest, SetupIsIdempotent) {
  auto m = make_two_part_mesh();
  ExcludeSelfInteraction ex;
  // Calling setup twice should not change behavior.
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  ex.setup(*m.bulk, *m.part_a, *m.part_b);
  EXPECT_TRUE(ex(make_cand(0, 0, m.node[1], m.node[1])));
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
}

// Periodic self: same entity AND zero shift -> excluded.
// Same entity but nonzero shift -> retained (genuine periodic image interaction).
TEST(ExcludeSelfInteractionTest, PeriodicSelfRequiresZeroShift) {
  auto m = make_two_part_mesh();
  ExcludeSelfInteraction ex;
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // zero relative shift (target shift == source shift): self -> exclude
  PeriodicCand self_zero(0, 0, m.node[1], m.node[1], Vec3f{0.f, 0.f, 0.f}, Vec3f{0.f, 0.f, 0.f});
  EXPECT_TRUE(ex(self_zero));

  // nonzero relative shift (source imaged away from target): same entity but different image -> retain
  PeriodicCand self_nonzero(0, 0, m.node[1], m.node[1], Vec3f{0.f, 0.f, 0.f}, Vec3f{1.f, 0.f, 0.f});
  EXPECT_FALSE(ex(self_nonzero));

  // different entity, zero relative shift -> not self -> retain
  PeriodicCand cross_zero(0, 1, m.node[1], m.node[2], Vec3f{0.f, 0.f, 0.f}, Vec3f{0.f, 0.f, 0.f});
  EXPECT_FALSE(ex(cross_zero));
}

// =============================================================================
// Group 4 — ExcluderChain
// =============================================================================

TEST(ExcluderChainTest, ORSemantics_NoExcluderPlusSelf) {
  auto m = make_two_part_mesh();
  auto chain = NoExcluder{}.exclude(ExcludeSelfInteraction{});
  chain.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_TRUE(chain(make_cand(0, 0, m.node[1], m.node[1]))) << "self should be excluded.";
  EXPECT_FALSE(chain(make_cand(0, 1, m.node[1], m.node[2])));
  EXPECT_FALSE(chain(make_cand(1, 0, m.node[2], m.node[1])));
}

TEST(ExcluderChainTest, ORSemantics_BothExclude) {
  // A chain where the appended excluder also excludes: result is still true (OR).
  auto m = make_two_part_mesh();
  auto chain = NoExcluder{}.exclude(ExcludeSelfInteraction{}).exclude(ExcludeSelfInteraction{});
  chain.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_TRUE(chain(make_cand(0, 0, m.node[1], m.node[1])));
  EXPECT_FALSE(chain(make_cand(0, 1, m.node[1], m.node[2])));
}

TEST(ExcluderChainTest, ORSemantics_NeitherExcludes) {
  auto m = make_two_part_mesh();
  auto chain = NoExcluder{}.exclude(NoExcluder{});
  chain.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_FALSE(chain(make_cand(0, 0, m.node[1], m.node[1])));
  EXPECT_FALSE(chain(make_cand(0, 1, m.node[1], m.node[2])));
}

TEST(ExcluderChainTest, MultiLevelChaining) {
  // Three-level chain: NoExcluder -> ExcludeSelf -> ExcludeSymDup (universal).
  // Self pairs excluded by level 2; directional pairs excluded by level 3.
  auto m = make_two_part_mesh();
  const stk::mesh::Selector universal = m.meta->universal_part();
  auto chain = NoExcluder{}.exclude(ExcludeSelfInteraction{}).exclude(ExcludeSymmetricDuplicates{});
  chain.setup(*m.bulk, universal, universal);

  // Self pair: excluded by ExcludeSelf.
  EXPECT_TRUE(chain(make_cand(0, 0, m.node[1], m.node[1])));
  // (t=node2, s=node1): src=node1 < trg=node2 and both in intersection -> excluded by ExcludeSymDup.
  EXPECT_TRUE(chain(make_cand(1, 0, m.node[2], m.node[1])));
  // (t=node1, s=node2): src=node2 > trg=node1 -> retained.
  EXPECT_FALSE(chain(make_cand(0, 1, m.node[1], m.node[2])));
}

TEST(ExcluderChainTest, SetupPropagatesIntoChain) {
  // Verify that setup() reaches the appended excluder by testing ExcludeSymDup behavior
  // under two different selector configurations on the same chain instance.
  auto m = make_two_part_mesh();
  auto chain = NoExcluder{}.exclude(ExcludeSymmetricDuplicates{});

  // Setup 1: disjoint selectors -> intersection empty -> ExcludeSymDup never fires.
  chain.setup(*m.bulk, *m.part_a, *m.part_b);
  // (t=node2, s=node1): even though src<trg, neither is in intersection -> NOT excluded.
  EXPECT_FALSE(chain(make_cand(1, 0, m.node[2], m.node[1])));

  // Setup 2: universal selector -> all nodes in intersection.
  chain.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  // Same pair now should be excluded.
  EXPECT_TRUE(chain(make_cand(1, 0, m.node[2], m.node[1])));
}

// =============================================================================
// Group 5 — ExcludeSymmetricDuplicates
// =============================================================================

// Helper: create an excluder already set up for a given selector pair.
ExcludeSymmetricDuplicates make_symdups(stk::mesh::BulkData& bulk, const stk::mesh::Selector& tgt_sel,
                                        const stk::mesh::Selector& src_sel) {
  ExcludeSymmetricDuplicates ex;
  ex.setup(bulk, tgt_sel, src_sel);
  return ex;
}

// Universal selectors: intersection = all 4 nodes.
// Suppress when BOTH entities in intersection AND src_entity < trg_entity.
TEST(ExcludeSymmetricDuplicatesTest, Universal_SuppressesLowerSrcEntity) {
  auto m = make_two_part_mesh();
  auto ex = make_symdups(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // src < trg -> suppressed.
  EXPECT_TRUE(ex(make_cand(1, 0, m.node[2], m.node[1])));
  EXPECT_TRUE(ex(make_cand(2, 0, m.node[3], m.node[1])));
  EXPECT_TRUE(ex(make_cand(3, 0, m.node[4], m.node[1])));
  EXPECT_TRUE(ex(make_cand(2, 1, m.node[3], m.node[2])));
  EXPECT_TRUE(ex(make_cand(3, 1, m.node[4], m.node[2])));
  EXPECT_TRUE(ex(make_cand(3, 2, m.node[4], m.node[3])));
}

TEST(ExcludeSymmetricDuplicatesTest, Universal_RetainsHigherSrcEntity) {
  auto m = make_two_part_mesh();
  auto ex = make_symdups(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // src > trg -> retained.
  EXPECT_FALSE(ex(make_cand(0, 1, m.node[1], m.node[2])));
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
  EXPECT_FALSE(ex(make_cand(0, 3, m.node[1], m.node[4])));
  EXPECT_FALSE(ex(make_cand(1, 2, m.node[2], m.node[3])));
  EXPECT_FALSE(ex(make_cand(1, 3, m.node[2], m.node[4])));
  EXPECT_FALSE(ex(make_cand(2, 3, m.node[3], m.node[4])));
}

TEST(ExcludeSymmetricDuplicatesTest, Universal_RetainsSelfPairs) {
  // Self pairs (src == trg) are NOT suppressed — src_entity < trg_entity is false.
  auto m = make_two_part_mesh();
  auto ex = make_symdups(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_FALSE(ex(make_cand(0, 0, m.node[1], m.node[1])));
  EXPECT_FALSE(ex(make_cand(1, 1, m.node[2], m.node[2])));
  EXPECT_FALSE(ex(make_cand(2, 2, m.node[3], m.node[3])));
  EXPECT_FALSE(ex(make_cand(3, 3, m.node[4], m.node[4])));
}

// Disjoint selectors: target=part_a, source=part_b -> intersection empty -> never suppresses.
TEST(ExcludeSymmetricDuplicatesTest, Disjoint_NeverSuppresses) {
  auto m = make_two_part_mesh();
  auto ex = make_symdups(*m.bulk, *m.part_a, *m.part_b);

  // These would be suppressed under universal, but not with disjoint intersection.
  EXPECT_FALSE(ex(make_cand(1, 0, m.node[2], m.node[1])));
  EXPECT_FALSE(ex(make_cand(0, 1, m.node[1], m.node[2])));
  // Cross-part pairs: target from part_a, source from part_b.
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
  EXPECT_FALSE(ex(make_cand(0, 3, m.node[1], m.node[4])));
  EXPECT_FALSE(ex(make_cand(1, 2, m.node[2], m.node[3])));
  EXPECT_FALSE(ex(make_cand(1, 3, m.node[2], m.node[4])));
}

// Overlapping selectors: target=part_a|part_b, source=part_b -> intersection=part_b={node3,node4}.
// Suppresses only when BOTH entities in part_b AND src<trg.
TEST(ExcludeSymmetricDuplicatesTest, Overlapping_SuppressesOnlyWithinIntersection) {
  auto m = make_two_part_mesh();
  const stk::mesh::Selector tgt_sel = *m.part_a | *m.part_b;
  const stk::mesh::Selector src_sel = *m.part_b;
  auto ex = make_symdups(*m.bulk, tgt_sel, src_sel);

  // Both in part_b, src < trg -> suppressed.
  EXPECT_TRUE(ex(make_cand(2, 1, m.node[4], m.node[3])));

  // Both in part_b, src > trg -> retained.
  EXPECT_FALSE(ex(make_cand(1, 2, m.node[3], m.node[4])));

  // trg in part_a (not in intersection) -> not suppressed regardless of direction.
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
  EXPECT_FALSE(ex(make_cand(0, 3, m.node[1], m.node[4])));
  EXPECT_FALSE(ex(make_cand(1, 2, m.node[2], m.node[3])));
}

// Identical-subset selectors: target=part_b, source=part_b -> intersection=part_b.
// Only pairs involving {node3, node4} considered; suppress src < trg.
TEST(ExcludeSymmetricDuplicatesTest, IdenticalSubset_SymmetricSuppression) {
  auto m = make_two_part_mesh();
  auto ex = make_symdups(*m.bulk, *m.part_b, *m.part_b);

  EXPECT_TRUE(ex(make_cand(1, 0, m.node[4], m.node[3])));   // src=node3 < trg=node4
  EXPECT_FALSE(ex(make_cand(0, 1, m.node[3], m.node[4])));  // src=node4 > trg=node3
  EXPECT_FALSE(ex(make_cand(0, 0, m.node[3], m.node[3])));  // self: src == trg
  EXPECT_FALSE(ex(make_cand(1, 1, m.node[4], m.node[4])));  // self: src == trg
}

TEST(ExcludeSymmetricDuplicatesTest, ResetOnSetup) {
  // Calling setup again with a different selector should update the intersection mask.
  auto m = make_two_part_mesh();
  ExcludeSymmetricDuplicates ex;

  // First setup: universal -> all nodes in intersection.
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  EXPECT_TRUE(ex(make_cand(1, 0, m.node[2], m.node[1])));

  // Second setup: disjoint -> empty intersection -> no suppression.
  ex.setup(*m.bulk, *m.part_a, *m.part_b);
  EXPECT_FALSE(ex(make_cand(1, 0, m.node[2], m.node[1])));

  // Third setup: universal again -> back to suppressing.
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  EXPECT_TRUE(ex(make_cand(1, 0, m.node[2], m.node[1])));
}

// =============================================================================
// Group 6 — ExcludeConnectedEntities
// =============================================================================
//
// Three BEAM_2 elements sharing nodes as follows:
//   elem[1]: nodes 1, 2
//   elem[2]: nodes 2, 3   (shares node 2 with elem[1])
//   elem[3]: nodes 4, 5   (no shared nodes with elem[1] or elem[2])
//
// ExcludeConnectedEntities(NODE_RANK) must exclude (elem[1], elem[2]) and
// retain (elem[1], elem[3]) and (elem[2], elem[3]).

struct ThreeBeamMesh {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::unique_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Part* beam_part = nullptr;
  stk::mesh::Entity node[6];  // 1-indexed: node[1]..node[5]
  stk::mesh::Entity elem[4];  // 1-indexed: elem[1]..elem[3]
};

ThreeBeamMesh make_three_beam_mesh() {
  ThreeBeamMesh m;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  m.meta = builder.create_meta_data();
  m.meta->use_simple_fields();
  m.beam_part = &m.meta->declare_part_with_topology("beams", stk::topology::BEAM_2);
  m.bulk = builder.create(m.meta);
  m.meta->commit();

  m.bulk->modification_begin();
  for (int id = 1; id <= 5; ++id) m.node[id] = m.bulk->declare_node(id);

  auto declare_beam = [&](int eid, int n0, int n1) {
    m.elem[eid] = m.bulk->declare_element(eid, stk::mesh::PartVector{m.beam_part});
    m.bulk->declare_relation(m.elem[eid], m.node[n0], 0);
    m.bulk->declare_relation(m.elem[eid], m.node[n1], 1);
  };
  declare_beam(1, 1, 2);
  declare_beam(2, 2, 3);
  declare_beam(3, 4, 5);
  m.bulk->modification_end();
  return m;
}

TEST(ExcludeConnectedEntitiesTest, ExcludesElementsPairSharingANode) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto m = make_three_beam_mesh();
  ExcludeConnectedEntities ex(stk::topology::NODE_RANK);
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // elem[1] and elem[2] share node[2].
  EXPECT_TRUE(ex(make_cand(0, 1, m.elem[1], m.elem[2])));
  // Symmetric.
  EXPECT_TRUE(ex(make_cand(1, 0, m.elem[2], m.elem[1])));
}

TEST(ExcludeConnectedEntitiesTest, RetainsElementPairsWithNoSharedNode) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto m = make_three_beam_mesh();
  ExcludeConnectedEntities ex(stk::topology::NODE_RANK);
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // elem[1]{1,2} and elem[3]{4,5} share nothing.
  EXPECT_FALSE(ex(make_cand(0, 2, m.elem[1], m.elem[3])));
  EXPECT_FALSE(ex(make_cand(2, 0, m.elem[3], m.elem[1])));
  // elem[2]{2,3} and elem[3]{4,5} share nothing.
  EXPECT_FALSE(ex(make_cand(1, 2, m.elem[2], m.elem[3])));
  EXPECT_FALSE(ex(make_cand(2, 1, m.elem[3], m.elem[2])));
}

TEST(ExcludeConnectedEntitiesTest, SelfPairIsExcluded) {
  // An element shares all its nodes with itself — degenerate case, excluded.
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto m = make_three_beam_mesh();
  ExcludeConnectedEntities ex(stk::topology::NODE_RANK);
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_TRUE(ex(make_cand(0, 0, m.elem[1], m.elem[1])));
  EXPECT_TRUE(ex(make_cand(2, 2, m.elem[3], m.elem[3])));
}

TEST(ExcludeConnectedEntitiesTest, ReflectsNewConnectivityAfterSetup) {
  // Calling setup again after modifying the mesh updates the NGP mesh snapshot.
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) GTEST_SKIP();
  auto m = make_three_beam_mesh();
  ExcludeConnectedEntities ex(stk::topology::NODE_RANK);
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // Before modification: elem[1] and elem[3] do not share nodes.
  EXPECT_FALSE(ex(make_cand(0, 2, m.elem[1], m.elem[3])));

  // Reconnect elem[3] to share node[2] with elem[1].
  m.bulk->modification_begin();
  m.bulk->destroy_relation(m.elem[3], m.node[4], 0);
  m.bulk->declare_relation(m.elem[3], m.node[2], 0);
  m.bulk->modification_end();

  // Re-snapshot the NGP mesh.
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_TRUE(ex(make_cand(0, 2, m.elem[1], m.elem[3])));
}

// =============================================================================
// Group 7 — ExcludeNonIntersectingOBBs
// =============================================================================
//
// Three axis-aligned unit-cube OBBs:
//   obb_origin: center=(0,0,0),  half=(0.5,0.5,0.5)
//   obb_close:  center=(0.8,0,0), half=(0.5,0.5,0.5) -> separation=0.8 < 1.0 -> intersects obb_origin
//   obb_far:    center=(2.0,0,0), half=(0.5,0.5,0.5) -> separation=2.0 > 1.0 -> separated from obb_origin

static OBB<double> make_unit_cube(double cx, double cy, double cz) {
  return OBB<double>{Point<double>{cx, cy, cz}, Quaternion<double>::identity(), 0.5, 0.5, 0.5};
}

TEST(ExcludeNonIntersectingOBBsTest, SetupMaterializesFromComponent) {
  // setup() reads the OBBs from the component over the selector; re-running it is consistent.
  auto m = make_obb_mesh({make_unit_cube(0.0, 0.0, 0.0), make_unit_cube(2.0, 0.0, 0.0)});  // origin, far
  ExcludeNonIntersectingOBBs<double> ex{m.component()};

  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  EXPECT_TRUE(ex(make_cand(0, 1, m.node[0], m.node[1])));  // separated -> excluded
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  EXPECT_TRUE(ex(make_cand(0, 1, m.node[0], m.node[1])));  // re-materialized, unchanged
}

TEST(ExcludeNonIntersectingOBBsTest, ExcludesSeparatedPair) {
  auto m = make_obb_mesh({make_unit_cube(0.0, 0.0, 0.0),    // ordinal 0 — origin
                          make_unit_cube(2.0, 0.0, 0.0)});  // ordinal 1 — separated
  ExcludeNonIntersectingOBBs<double> ex{m.component()};
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_TRUE(ex(make_cand(0, 1, m.node[0], m.node[1])));  // origin vs far -> excluded
  EXPECT_TRUE(ex(make_cand(1, 0, m.node[1], m.node[0])));  // far vs origin -> excluded (symmetric)
}

TEST(ExcludeNonIntersectingOBBsTest, RetainsIntersectingPair) {
  auto m = make_obb_mesh({make_unit_cube(0.0, 0.0, 0.0),    // ordinal 0 — origin
                          make_unit_cube(0.8, 0.0, 0.0)});  // ordinal 1 — overlapping
  ExcludeNonIntersectingOBBs<double> ex{m.component()};
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_FALSE(ex(make_cand(0, 1, m.node[0], m.node[1])));  // origin vs close -> retained
  EXPECT_FALSE(ex(make_cand(1, 0, m.node[1], m.node[0])));  // close vs origin -> retained
}

TEST(ExcludeNonIntersectingOBBsTest, AsymmetricTargetSourceSelectors) {
  // One OBB field, asymmetric via per-side selectors: part_a = {origin}; part_b = {far, close}.
  auto m = make_obb_mesh({make_unit_cube(0.0, 0.0, 0.0),   // node 1 -> part_a, target ordinal 0 — origin
                          make_unit_cube(2.0, 0.0, 0.0),   // node 2 -> part_b, source ordinal 0 — separated
                          make_unit_cube(0.8, 0.0, 0.0)},  // node 3 -> part_b, source ordinal 1 — overlapping
                         /*split=*/1);
  ExcludeNonIntersectingOBBs<double> ex{m.component(), m.component()};
  ex.setup(*m.bulk, *m.part_a, *m.part_b);

  EXPECT_TRUE(ex(make_cand(0, 0, m.node[0], m.node[1])));   // origin vs far   -> excluded
  EXPECT_FALSE(ex(make_cand(0, 1, m.node[0], m.node[2])));  // origin vs close -> retained
}

TEST(ExcludeNonIntersectingOBBsTest, SymmetricSingleComponentConstructor) {
  // Single component used for both target and source sides.
  auto m = make_obb_mesh({make_unit_cube(0.0, 0.0, 0.0),    // ordinal 0 — origin
                          make_unit_cube(0.8, 0.0, 0.0),    // ordinal 1 — close
                          make_unit_cube(2.0, 0.0, 0.0)});  // ordinal 2 — far
  ExcludeNonIntersectingOBBs<double> ex{m.component()};
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_FALSE(ex(make_cand(0, 0, m.node[0], m.node[0])));  // origin vs origin -> retained
  EXPECT_FALSE(ex(make_cand(0, 1, m.node[0], m.node[1])));  // origin vs close  -> retained
  EXPECT_TRUE(ex(make_cand(0, 2, m.node[0], m.node[2])));   // origin vs far    -> excluded
}

TEST(ExcludeNonIntersectingOBBsTest, ChainCompatibility) {
  // ExcludeNonIntersectingOBBs must compose correctly in an ExcluderChain.
  auto m = make_obb_mesh({make_unit_cube(0.0, 0.0, 0.0),    // ordinal 0 — origin
                          make_unit_cube(0.8, 0.0, 0.0),    // ordinal 1 — close
                          make_unit_cube(2.0, 0.0, 0.0)});  // ordinal 2 — far
  auto chain = NoExcluder{}.exclude(ExcludeNonIntersectingOBBs<double>{m.component()});
  chain.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  EXPECT_FALSE(chain(make_cand(0, 1, m.node[0], m.node[1])));  // origin vs close -> retained
  EXPECT_TRUE(chain(make_cand(0, 2, m.node[0], m.node[2])));   // origin vs far   -> excluded
}

}  // namespace
}  // namespace search
}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX
