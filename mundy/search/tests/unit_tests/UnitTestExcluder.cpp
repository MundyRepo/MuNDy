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

// Mundy math
#include <mundy_math/Vector3.hpp>  // for mundy::Vector3

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
static_assert(ExcluderType<ExcludeSymmetricDuplicates>);
static_assert(ExcluderType<ExcluderChain<NoExcluder, ExcludeSelfInteraction>>);
static_assert(ExcluderType<ExcluderChain<NoExcluder, ExcludeSymmetricDuplicates>>);
static_assert(ExcluderType<ExcluderChain<ExcluderChain<NoExcluder, ExcludeSelfInteraction>, ExcludeSymmetricDuplicates>>);
// const-qualified types must NOT satisfy ExcluderType because setup() is non-const.
static_assert(!ExcluderType<const NoExcluder>);
static_assert(!ExcluderType<const ExcludeSelfInteraction>);

// =============================================================================
// Minimal STK mesh factory for excluder tests
// =============================================================================

// 4 nodes: nodes 1,2 → part_a; nodes 3,4 → part_b.
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

  for (int id = 1; id <= 4; ++id)
    m.node[id] = m.bulk->get_entity(stk::topology::NODE_RANK, id);
  return m;
}

// Build a NeighborSearchCandidate from array ordinals and explicit entities.
Cand make_cand(size_t t_ord, size_t s_ord, stk::mesh::Entity t_ent, stk::mesh::Entity s_ent) {
  return Cand(t_ord, s_ord, t_ent, s_ent);
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

// Periodic self: same entity AND zero shift → excluded.
// Same entity but nonzero shift → retained (genuine periodic image interaction).
TEST(ExcludeSelfInteractionTest, PeriodicSelfRequiresZeroShift) {
  auto m = make_two_part_mesh();
  ExcludeSelfInteraction ex;
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());

  // zero-shift periodic candidate: self → exclude
  PeriodicCand self_zero(0, 0, m.node[1], m.node[1], Vec3f{0.f, 0.f, 0.f});
  EXPECT_TRUE(ex(self_zero));

  // nonzero-shift periodic candidate: same entity but different image → retain
  PeriodicCand self_nonzero(0, 0, m.node[1], m.node[1], Vec3f{1.f, 0.f, 0.f});
  EXPECT_FALSE(ex(self_nonzero));

  // different entity, zero shift → not self → retain
  PeriodicCand cross_zero(0, 1, m.node[1], m.node[2], Vec3f{0.f, 0.f, 0.f});
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
  // Three-level chain: NoExcluder → ExcludeSelf → ExcludeSymDup (universal).
  // Self pairs excluded by level 2; directional pairs excluded by level 3.
  auto m = make_two_part_mesh();
  const stk::mesh::Selector universal = m.meta->universal_part();
  auto chain = NoExcluder{}.exclude(ExcludeSelfInteraction{}).exclude(ExcludeSymmetricDuplicates{});
  chain.setup(*m.bulk, universal, universal);

  // Self pair: excluded by ExcludeSelf.
  EXPECT_TRUE(chain(make_cand(0, 0, m.node[1], m.node[1])));
  // (t=node2, s=node1): src=node1 < trg=node2 and both in intersection → excluded by ExcludeSymDup.
  EXPECT_TRUE(chain(make_cand(1, 0, m.node[2], m.node[1])));
  // (t=node1, s=node2): src=node2 > trg=node1 → retained.
  EXPECT_FALSE(chain(make_cand(0, 1, m.node[1], m.node[2])));
}

TEST(ExcluderChainTest, SetupPropagatesIntoChain) {
  // Verify that setup() reaches the appended excluder by testing ExcludeSymDup behavior
  // under two different selector configurations on the same chain instance.
  auto m = make_two_part_mesh();
  auto chain = NoExcluder{}.exclude(ExcludeSymmetricDuplicates{});

  // Setup 1: disjoint selectors → intersection empty → ExcludeSymDup never fires.
  chain.setup(*m.bulk, *m.part_a, *m.part_b);
  // (t=node2, s=node1): even though src<trg, neither is in intersection → NOT excluded.
  EXPECT_FALSE(chain(make_cand(1, 0, m.node[2], m.node[1])));

  // Setup 2: universal selector → all nodes in intersection.
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

  // src < trg → suppressed.
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

  // src > trg → retained.
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

// Disjoint selectors: target=part_a, source=part_b → intersection empty → never suppresses.
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

// Overlapping selectors: target=part_a|part_b, source=part_b → intersection=part_b={node3,node4}.
// Suppresses only when BOTH entities in part_b AND src<trg.
TEST(ExcludeSymmetricDuplicatesTest, Overlapping_SuppressesOnlyWithinIntersection) {
  auto m = make_two_part_mesh();
  const stk::mesh::Selector tgt_sel = *m.part_a | *m.part_b;
  const stk::mesh::Selector src_sel = *m.part_b;
  auto ex = make_symdups(*m.bulk, tgt_sel, src_sel);

  // Both in part_b, src < trg → suppressed.
  EXPECT_TRUE(ex(make_cand(2, 1, m.node[4], m.node[3])));

  // Both in part_b, src > trg → retained.
  EXPECT_FALSE(ex(make_cand(1, 2, m.node[3], m.node[4])));

  // trg in part_a (not in intersection) → not suppressed regardless of direction.
  EXPECT_FALSE(ex(make_cand(0, 2, m.node[1], m.node[3])));
  EXPECT_FALSE(ex(make_cand(0, 3, m.node[1], m.node[4])));
  EXPECT_FALSE(ex(make_cand(1, 2, m.node[2], m.node[3])));
}

// Identical-subset selectors: target=part_b, source=part_b → intersection=part_b.
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

  // First setup: universal → all nodes in intersection.
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  EXPECT_TRUE(ex(make_cand(1, 0, m.node[2], m.node[1])));

  // Second setup: disjoint → empty intersection → no suppression.
  ex.setup(*m.bulk, *m.part_a, *m.part_b);
  EXPECT_FALSE(ex(make_cand(1, 0, m.node[2], m.node[1])));

  // Third setup: universal again → back to suppressing.
  ex.setup(*m.bulk, m.meta->universal_part(), m.meta->universal_part());
  EXPECT_TRUE(ex(make_cand(1, 0, m.node[2], m.node[1])));
}

}  // namespace
}  // namespace search
}  // namespace mundy

#endif  // HAVE_MUNDYSEARCH_ARBORX
