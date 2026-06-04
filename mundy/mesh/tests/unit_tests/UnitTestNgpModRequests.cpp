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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/Class.hpp>         // for mundy::mesh::declare_class, mundy::mesh::Class
#include <mundy_mesh/LinkData.hpp>      // for mundy::mesh::LinkData, declare_link_data
#include <mundy_mesh/LinkMetaData.hpp>  // for mundy::mesh::declare_link_meta_data
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/NgpModRequests.hpp>  // for mundy::mesh::NgpModRequests

namespace mundy {

namespace mesh {

namespace {

struct NgpModRequestsRawFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta_data_ptr;
  std::shared_ptr<BulkData> bulk_data_ptr;
  stk::mesh::Part* elem_part_1{nullptr};
  stk::mesh::Part* elem_part_2{nullptr};
  stk::mesh::Part* particle_part{nullptr};

  NgpModRequestsRawFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    meta_data_ptr = builder.create_meta_data();
    meta_data_ptr->use_simple_fields();
    bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
    elem_part_1 = &meta_data_ptr->declare_part("PART_1", stk::topology::ELEM_RANK);
    elem_part_2 = &meta_data_ptr->declare_part("PART_2", stk::topology::ELEM_RANK);
    particle_part = &meta_data_ptr->declare_part_with_topology("PARTICLE_PART", stk::topology::PARTICLE);
    meta_data_ptr->commit();
  }
};

TEST(NgpModRequests, TicketIssuerHostClaimFinalizeAndReset) {
  using issuer_t = TicketIssuer<stk::ngp::MemSpace>;

  issuer_t issuer;
  issuer.initialize(true);

  ASSERT_NO_THROW(issuer.activate_host());

  TicketRange range = issuer.claim(3);
  EXPECT_EQ(range.begin(), 0u);
  EXPECT_EQ(range.end(), 3u);
  EXPECT_EQ(range.count(), 3u);

  size_t ticket = issuer.claim();
  EXPECT_EQ(ticket, 3u);
  EXPECT_EQ(issuer.count(), 4u);
  EXPECT_EQ(issuer.finalize_count(), 4u);

  issuer.reset();
  EXPECT_EQ(issuer.count(), 0u);
}

TEST(NgpModRequests, TicketIssuerInitializeHostActive) {
  using issuer_t = TicketIssuer<stk::ngp::MemSpace>;

  issuer_t issuer;
  issuer.initialize(false);

  size_t t0 = issuer.claim();
  TicketRange range = issuer.claim(2);
  EXPECT_EQ(t0, 0u);
  EXPECT_EQ(range.begin(), 1u);
  EXPECT_EQ(range.end(), 3u);
  EXPECT_EQ(issuer.count(), 3u);
}

TEST(NgpModRequests, TicketIssuerWithinEachRequestClass) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  auto req_new_ids1 = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto req_new_ids2 = reqs.request_entities_new_ids(*fixture.elem_part_2);

  auto req_known_ids1 = reqs.request_entities_known_ids(*fixture.elem_part_1);
  auto req_known_ids2 = reqs.request_entities_known_ids(*fixture.elem_part_2);

  auto req_conns = reqs.request_connections();
  auto req_destroy_entities = reqs.destroy_entities();
  auto req_destroy_conns = reqs.destroy_connections();

  reqs.activate_host();

  // Validate that we can claim tickets for each class and have them properly generate the ticket ids and increment the
  // counts
  auto claim_three_test = [](auto issuer, const std::string& message) {
    ASSERT_EQ(issuer.count(), 0u) << message;
    size_t t0 = issuer.claim();
    TicketRange range = issuer.claim(2);
    EXPECT_EQ(t0, 0u) << message;
    EXPECT_EQ(range.begin(), 1u) << message;
    EXPECT_EQ(range.end(), 3u) << message;
    EXPECT_EQ(issuer.count(), 3u) << message;
  };

  for (stk::mesh::EntityRank rank = stk::topology::BEGIN_RANK; rank < stk::topology::NUM_RANKS; ++rank) {
    claim_three_test(req_new_ids1.tickets(rank), "req_new_ids1.tickets(" + std::to_string(rank) + ") failed");
    claim_three_test(req_new_ids2.tickets(rank), "req_new_ids2.tickets(" + std::to_string(rank) + ") failed");

    claim_three_test(req_known_ids1.tickets(rank), "req_known_ids1.tickets(" + std::to_string(rank) + ") failed");
    claim_three_test(req_known_ids2.tickets(rank), "req_known_ids2.tickets(" + std::to_string(rank) + ") failed");
  }

  claim_three_test(req_conns.tickets(), "req_conns failed");
  claim_three_test(req_destroy_entities.tickets(), "req_destroy_entities failed");
  claim_three_test(req_destroy_conns.tickets(), "req_destroy_conns failed");
}

TEST(NgpModRequests, RequestEntitiesMemoizeByPartitionKey) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  stk::mesh::PartVector parts_1{fixture.elem_part_1, fixture.elem_part_2};
  stk::mesh::PartVector parts_2{fixture.elem_part_2, fixture.elem_part_1, fixture.elem_part_1};

  auto new_ids_a = reqs.request_entities_new_ids(parts_1);
  auto new_ids_b = reqs.request_entities_new_ids(parts_2);
  EXPECT_EQ(new_ids_a.id(), new_ids_b.id());

  auto known_ids_a = reqs.request_entities_known_ids(parts_1);
  auto known_ids_b = reqs.request_entities_known_ids(parts_2);
  EXPECT_EQ(known_ids_a.id(), known_ids_b.id());
}

TEST(NgpModRequests, CopyIsViewLikeForSharedState) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  NgpModRequests reqs_copy = reqs;

  auto helper_from_original = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto helper_from_copy = reqs_copy.request_entities_new_ids(*fixture.elem_part_1);
  EXPECT_EQ(helper_from_original.id(), helper_from_copy.id());

  reqs.activate_host();
  helper_from_original.element_tickets().claim(2);
  EXPECT_EQ(helper_from_original.element_tickets().count(), 2u);

  reqs_copy.activate_host();
  EXPECT_EQ(helper_from_copy.element_tickets().count(), 2u);

  helper_from_copy.element_tickets().claim(1);
  EXPECT_EQ(helper_from_original.element_tickets().count(), 3u);
}

TEST(NgpModRequests, EntityHelperTicketAccessorActsLikeView) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  auto helper = reqs.request_entities_new_ids(*fixture.elem_part_1);

  reqs.activate_host();
  auto issuer_copy = helper.element_tickets();
  issuer_copy.claim();

  EXPECT_EQ(helper.element_tickets().count(), 1u)
      << "Claiming through a returned ticket issuer must mutate helper state.";
}

TEST(NgpModRequests, ReaccessedHelperSeesPriorMutations) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  auto helper_a = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto helper_b = reqs.request_entities_new_ids(*fixture.elem_part_1);

  reqs.activate_host();
  helper_a.element_tickets().claim(2);

  EXPECT_EQ(helper_b.element_tickets().count(), 2u)
      << "Two returned helpers for the same partition key should share mutable state.";
}

TEST(NgpModRequests, SeparateConnectionHelperHandlesShareState) {
  NgpModRequests reqs;
  auto conn_a = reqs.request_connections();
  auto conn_b = reqs.request_connections();

  reqs.activate_host();
  conn_a.tickets().claim(3);

  EXPECT_EQ(conn_b.tickets().count(), 3u)
      << "Connection helper accessor should return a lightweight view into shared state.";
}

TEST(NgpModRequests, FinalizeCountsDoesNotInvalidateHelpers) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  auto known_ids_pre_finalize = reqs.request_entities_known_ids(*fixture.elem_part_1);

  reqs.activate_host();
  size_t ticket = known_ids_pre_finalize.element_tickets().claim();
  reqs.finalize_counts();

  known_ids_pre_finalize.request_element(ticket, 4242);

  auto known_ids_post_finalize = reqs.request_entities_known_ids(*fixture.elem_part_1);

  // Both pre- and post-finalize helpers should see the same mutual state:
  EXPECT_EQ(known_ids_pre_finalize.get_entity_id(ticket, stk::topology::ELEMENT_RANK), 4242u)
      << "Pre-finalize helpers should 'see' all requests.";
  EXPECT_EQ(known_ids_post_finalize.get_entity_id(ticket, stk::topology::ELEMENT_RANK), 4242u)
      << "Post-finalize helpers should 'see' all requests.";

  EXPECT_EQ(known_ids_pre_finalize.id(), known_ids_post_finalize.id());
  EXPECT_EQ(known_ids_pre_finalize.element_tickets().count(), known_ids_post_finalize.element_tickets().count())
      << "Pre- and post-finalize helpers should 'see' the same ticket state.";
  EXPECT_EQ(known_ids_pre_finalize.element_tickets().count(), 1u);
}

TEST(NgpModRequests, HigherLevelCreateEntitiesAndConnectExistingToNew) {
  NgpModRequestsRawFixture fixture;
  BulkData& bulk = *fixture.bulk_data_ptr;

  bulk.modification_begin();
  stk::mesh::Entity existing_elem =
      bulk.declare_entity(stk::topology::ELEMENT_RANK, /*id*/ 1, stk::mesh::PartVector{fixture.elem_part_1});
  bulk.modification_end();
  ASSERT_TRUE(bulk.is_valid(existing_elem));

  NgpModRequests reqs;
  stk::mesh::PartVector no_parts{};
  auto req_new_elems = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto req_known_elems = reqs.request_entities_known_ids(*fixture.elem_part_2);
  auto req_new_nodes = reqs.request_entities_new_ids(no_parts);
  auto req_conns = reqs.request_connections();

  reqs.activate_host();
  size_t new_elem_ticket = req_new_elems.element_tickets().claim();
  size_t known_elem_ticket = req_known_elems.element_tickets().claim();
  size_t new_node_ticket = req_new_nodes.node_tickets().claim();
  size_t conn_ticket = req_conns.tickets().claim();
  reqs.finalize_counts();

  ASSERT_EQ(new_elem_ticket, 0u);
  ASSERT_EQ(known_elem_ticket, 0u);
  ASSERT_EQ(new_node_ticket, 0u);
  ASSERT_EQ(conn_ticket, 0u);

  req_new_elems.request_element(new_elem_ticket);
  req_known_elems.request_element(known_elem_ticket, 2001);
  FutureEntity future_new_node = req_new_nodes.request_node(new_node_ticket);
  req_conns.request(conn_ticket, existing_elem, future_new_node, 0);

  ASSERT_NO_THROW(reqs.process_requests(bulk));

  stk::mesh::Entity created_new_elem = req_new_elems.get_entity(new_elem_ticket, stk::topology::ELEMENT_RANK);
  stk::mesh::Entity created_known_elem = req_known_elems.get_entity(known_elem_ticket, stk::topology::ELEMENT_RANK);
  stk::mesh::Entity created_new_node = req_new_nodes.get_entity(new_node_ticket, stk::topology::NODE_RANK);

  EXPECT_TRUE(bulk.is_valid(created_new_elem));
  EXPECT_TRUE(bulk.is_valid(created_known_elem));
  EXPECT_TRUE(bulk.is_valid(created_new_node));
  EXPECT_EQ(bulk.identifier(created_known_elem), 2001u);
  EXPECT_EQ(bulk.begin_nodes(existing_elem)[0], created_new_node);
}

// ============================================================
// Fixture and tests for the Class-based API
// ============================================================

struct NgpModRequestsClassFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta_data_ptr;
  std::shared_ptr<BulkData> bulk_data_ptr;
  Class* particle_class{nullptr};
  Class* element_class{nullptr};

  NgpModRequestsClassFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    meta_data_ptr = builder.create_meta_data();
    meta_data_ptr->use_simple_fields();

    // Declare classes before committing meta_data — analogous to declaring parts.
    particle_class = &declare_class(*meta_data_ptr, "PARTICLE_CLASS", stk::topology::PARTICLE,
                                    /*disable_io_support=*/true);
    element_class = &declare_class(*meta_data_ptr, "ELEMENT_CLASS", stk::topology::HEX_8,
                                   /*disable_io_support=*/true);

    bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
    meta_data_ptr->commit();
  }

  BulkData& bulk() {
    return *bulk_data_ptr;
  }
};

TEST(NgpModRequests, ClassBasedRequestEntitiesNewIds_SingleClass) {
  // Entities requested via a single Class must be created and reside in that class.
  NgpModRequestsClassFixture fixture;
  BulkData& bulk = fixture.bulk();

  NgpModRequests reqs;
  auto req = reqs.request_entities_new_ids(*fixture.particle_class);

  reqs.activate_host();
  size_t ticket = req.element_tickets().claim();
  reqs.finalize_counts();
  req.request_element(ticket);

  ASSERT_NO_THROW(reqs.process_requests(bulk));

  stk::mesh::Entity entity = req.get_entity(ticket, stk::topology::ELEMENT_RANK);
  ASSERT_TRUE(bulk.is_valid(entity));

  // Entity must be a member of the class's leaf_part (direct membership) and
  // assembly_part (membership visible through the class hierarchy).
  EXPECT_TRUE(bulk.bucket(entity).member(fixture.particle_class->leaf_part()))
      << "Entity should reside in the class's leaf_part.";
  EXPECT_TRUE(bulk.bucket(entity).member(fixture.particle_class->assembly_part()))
      << "Entity should be visible through the class's assembly_part.";
}

TEST(NgpModRequests, ClassBasedRequestEntitiesNewIds_ClassVector) {
  // Requesting with a ClassVector must produce the same memoized helper as a single-Class call
  // for the same set of classes, and entities must land in all specified classes.
  NgpModRequestsClassFixture fixture;
  BulkData& bulk = fixture.bulk();

  NgpModRequests reqs;
  ClassVector classes{fixture.particle_class};

  auto req_vec = reqs.request_entities_new_ids(classes);
  auto req_single = reqs.request_entities_new_ids(*fixture.particle_class);

  // Same partition key → same memoized helper.
  EXPECT_EQ(req_vec.id(), req_single.id())
      << "ClassVector{cls} and single Class& must return the same memoized helper.";

  reqs.activate_host();
  size_t ticket = req_vec.element_tickets().claim();
  reqs.finalize_counts();
  req_vec.request_element(ticket);

  ASSERT_NO_THROW(reqs.process_requests(bulk));

  stk::mesh::Entity entity = req_vec.get_entity(ticket, stk::topology::ELEMENT_RANK);
  ASSERT_TRUE(bulk.is_valid(entity));
  EXPECT_TRUE(bulk.bucket(entity).member(fixture.particle_class->leaf_part()));
}

TEST(NgpModRequests, ClassBasedRequestEntitiesKnownIds) {
  // The known-ID class overload must create the entity with the specified ID in the class.
  NgpModRequestsClassFixture fixture;
  BulkData& bulk = fixture.bulk();

  NgpModRequests reqs;
  auto req = reqs.request_entities_known_ids(*fixture.particle_class);

  reqs.activate_host();
  size_t ticket = req.element_tickets().claim();
  reqs.finalize_counts();
  req.request_element(ticket, /*id=*/9999);

  ASSERT_NO_THROW(reqs.process_requests(bulk));

  stk::mesh::Entity entity = req.get_entity(ticket, stk::topology::ELEMENT_RANK);
  ASSERT_TRUE(bulk.is_valid(entity));
  EXPECT_EQ(bulk.identifier(entity), 9999u);
  EXPECT_TRUE(bulk.bucket(entity).member(fixture.particle_class->leaf_part()));
}

TEST(NgpModRequests, HigherLevelDestroyConnectionAndEntity) {
  NgpModRequestsRawFixture fixture;
  BulkData& bulk = *fixture.bulk_data_ptr;

  bulk.modification_begin();
  stk::mesh::Entity elem =
      bulk.declare_entity(stk::topology::ELEMENT_RANK, /*id*/ 1, stk::mesh::PartVector{fixture.elem_part_1});
  stk::mesh::Entity node = bulk.declare_entity(stk::topology::NODE_RANK, /*id*/ 1, stk::mesh::PartVector{});
  bulk.declare_relation(elem, node, 0);
  bulk.modification_end();
  ASSERT_TRUE(bulk.is_valid(elem));
  ASSERT_TRUE(bulk.is_valid(node));

  NgpModRequests reqs;
  auto destroy_conns = reqs.destroy_connections();
  auto destroy_entities = reqs.destroy_entities();

  reqs.activate_host();
  destroy_conns.tickets().claim(1);
  destroy_entities.tickets().claim(1);
  reqs.finalize_counts();

  destroy_conns.request(0, elem, node, 0);
  destroy_entities.destroy(0, elem);

  ASSERT_NO_THROW(reqs.process_requests(bulk));
  EXPECT_FALSE(bulk.is_valid(elem));
  EXPECT_TRUE(bulk.is_valid(node));
}

// ============================================================
// Fixture and tests for request_links / NgpRequestLinkRelationsT
// ============================================================

/// Fixture: mesh with a constraint-rank link part (dimensionality 2) and a declared LinkData.
struct NgpModRequestsLinkFixture {
  MeshBuilder builder{MPI_COMM_WORLD};
  std::shared_ptr<MetaData> meta;
  std::shared_ptr<BulkData> bulk;
  LinkMetaData* link_meta{nullptr};
  stk::mesh::Part* link_part{nullptr};
  LinkData* link_data{nullptr};
  static constexpr stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;

  NgpModRequestsLinkFixture() {
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    meta = builder.create_meta_data();
    meta->use_simple_fields();
    bulk = builder.create_bulk_data(meta);
    link_meta = &declare_link_meta_data(*meta, "MOD_REQ_LINKS", link_rank);
    link_part = &link_meta->declare_link_part("MOD_REQ_LINK_PART", 2u);
    meta->commit();
    link_data = &declare_link_data(*bulk, *link_meta);
  }
};

// Two calls to request_links for the same LinkData must return the same memoized helper.
TEST(NgpModRequests, RequestLinks_Memoized) {
  NgpModRequestsLinkFixture f;

  NgpModRequests reqs;
  auto req_a = reqs.request_links(*f.link_data);
  auto req_b = reqs.request_links(*f.link_data);
  EXPECT_EQ(req_a.id(), req_b.id())
      << "Two request_links() calls for the same LinkData must return the same memoized helper.";
}

// Two handles from request_links(same_link_data) share the underlying ticket state.
TEST(NgpModRequests, RequestLinks_SharedState) {
  NgpModRequestsLinkFixture f;

  NgpModRequests reqs;
  auto req_a = reqs.request_links(*f.link_data);
  auto req_b = reqs.request_links(*f.link_data);

  reqs.activate_host();
  req_a.tickets().claim(3);

  EXPECT_EQ(req_b.tickets().count(), 3u)
      << "Claiming through one handle must be visible through any other handle for the same LinkData.";
}

// request_links with existing (non-future) linker and linked entity writes COO fields directly.
TEST(NgpModRequests, RequestLinks_WritesRelationForExistingLinker) {
  NgpModRequestsLinkFixture f;

  f.bulk->modification_begin();
  stk::mesh::Entity link = f.bulk->declare_entity(f.link_rank, 1u, stk::mesh::PartVector{f.link_part});
  stk::mesh::Entity node0 = f.bulk->declare_node(1u);
  stk::mesh::Entity node1 = f.bulk->declare_node(2u);
  f.bulk->modification_end();
  ASSERT_TRUE(f.bulk->is_valid(link));

  NgpModRequests reqs;
  auto req_links = reqs.request_links(*f.link_data);

  reqs.activate_host();
  size_t t0 = req_links.tickets().claim();
  size_t t1 = req_links.tickets().claim();
  reqs.finalize_counts();

  req_links.request(t0, link, node0, 0u);
  req_links.request(t1, link, node1, 1u);

  ASSERT_NO_THROW(reqs.process_requests(*f.bulk));

  EXPECT_EQ(f.link_data->coo_data().get_linked_entity(link, 0u), node0);
  EXPECT_EQ(f.link_data->coo_data().get_linked_entity(link, 1u), node1);
  EXPECT_EQ(f.link_data->coo_data().get_linked_entity_id(link, 0u), f.bulk->identifier(node0));
}

// Primary use case: request a new link entity and simultaneously request link relations whose
// linker is the resulting FutureEntity.  After process_requests the link entity must exist and
// both COO slots must point to the correct nodes.
TEST(NgpModRequests, RequestLinks_WritesRelationForFutureLinker) {
  NgpModRequestsLinkFixture f;

  f.bulk->modification_begin();
  stk::mesh::Entity node0 = f.bulk->declare_node(1u);
  stk::mesh::Entity node1 = f.bulk->declare_node(2u);
  f.bulk->modification_end();

  NgpModRequests reqs;
  auto req_link_entities = reqs.request_entities_new_ids(*f.link_part);
  auto req_links = reqs.request_links(*f.link_data);

  reqs.activate_host();
  size_t entity_ticket = req_link_entities.constraint_tickets().claim();
  size_t rel_ticket_0 = req_links.tickets().claim();
  size_t rel_ticket_1 = req_links.tickets().claim();
  reqs.finalize_counts();

  FutureEntity future_link = req_link_entities.request_constraint(entity_ticket);
  req_links.request(rel_ticket_0, future_link, node0, 0u);
  req_links.request(rel_ticket_1, future_link, node1, 1u);

  ASSERT_NO_THROW(reqs.process_requests(*f.bulk));

  stk::mesh::Entity created_link = req_link_entities.get_entity(entity_ticket, f.link_rank);
  ASSERT_TRUE(f.bulk->is_valid(created_link)) << "Link entity must have been created by process_requests.";
  EXPECT_EQ(f.link_data->coo_data().get_linked_entity(created_link, 0u), node0)
      << "COO ordinal 0 must point to node0 after process_requests.";
  EXPECT_EQ(f.link_data->coo_data().get_linked_entity(created_link, 1u), node1)
      << "COO ordinal 1 must point to node1 after process_requests.";
}

// Two distinct LinkData objects in a single process_requests call: each gets its own
// NgpRequestLinkRelationsT (keyed by LinkData*), both COO writes land in the right object,
// and coo_modify_on_host() is called once for each.
TEST(NgpModRequests, RequestLinks_TwoLinkDataObjects) {
  // Build a mesh with two independent link meta-datas at the same rank.
  MeshBuilder builder{MPI_COMM_WORLD};
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  auto meta = builder.create_meta_data();
  meta->use_simple_fields();
  auto bulk = builder.create_bulk_data(meta);

  static constexpr stk::mesh::EntityRank link_rank = stk::topology::CONSTRAINT_RANK;

  LinkMetaData& link_meta_a = declare_link_meta_data(*meta, "LINKS_A", link_rank);
  stk::mesh::Part* link_part_a = &link_meta_a.declare_link_part("LINK_PART_A", 2u);

  LinkMetaData& link_meta_b = declare_link_meta_data(*meta, "LINKS_B", link_rank);
  stk::mesh::Part* link_part_b = &link_meta_b.declare_link_part("LINK_PART_B", 2u);

  meta->commit();
  LinkData& link_data_a = declare_link_data(*bulk, link_meta_a);
  LinkData& link_data_b = declare_link_data(*bulk, link_meta_b);

  // Declare shared target nodes.
  bulk->modification_begin();
  stk::mesh::Entity node0 = bulk->declare_node(1u);
  stk::mesh::Entity node1 = bulk->declare_node(2u);
  bulk->modification_end();

  NgpModRequests reqs;
  auto req_links_a = reqs.request_entities_new_ids(*link_part_a);
  auto req_links_b = reqs.request_entities_new_ids(*link_part_b);
  auto req_rel_a   = reqs.request_links(link_data_a);
  auto req_rel_b   = reqs.request_links(link_data_b);

  // request_links for two different LinkData must return helpers with different ids.
  EXPECT_NE(req_rel_a.id(), req_rel_b.id());

  reqs.activate_host();
  size_t ta  = req_links_a.constraint_tickets().claim();
  size_t tb  = req_links_b.constraint_tickets().claim();
  size_t ra0 = req_rel_a.tickets().claim();
  size_t ra1 = req_rel_a.tickets().claim();
  size_t rb0 = req_rel_b.tickets().claim();
  size_t rb1 = req_rel_b.tickets().claim();
  reqs.finalize_counts();

  FutureEntity future_a = req_links_a.request_constraint(ta);
  FutureEntity future_b = req_links_b.request_constraint(tb);
  req_rel_a.request(ra0, future_a, node0, 0u);
  req_rel_a.request(ra1, future_a, node1, 1u);
  req_rel_b.request(rb0, future_b, node1, 0u);  // reversed: b links to node1 first
  req_rel_b.request(rb1, future_b, node0, 1u);

  ASSERT_NO_THROW(reqs.process_requests(*bulk));

  stk::mesh::Entity link_a = req_links_a.get_entity(ta, link_rank);
  stk::mesh::Entity link_b = req_links_b.get_entity(tb, link_rank);

  ASSERT_TRUE(bulk->is_valid(link_a));
  ASSERT_TRUE(bulk->is_valid(link_b));

  // link_data_a COO must hold node0 at ordinal 0, node1 at ordinal 1.
  EXPECT_EQ(link_data_a.coo_data().get_linked_entity(link_a, 0u), node0);
  EXPECT_EQ(link_data_a.coo_data().get_linked_entity(link_a, 1u), node1);

  // link_data_b COO must hold node1 at ordinal 0, node0 at ordinal 1.
  EXPECT_EQ(link_data_b.coo_data().get_linked_entity(link_b, 0u), node1);
  EXPECT_EQ(link_data_b.coo_data().get_linked_entity(link_b, 1u), node0);

  // coo_modify_on_host() was called for both — both should need a device sync.
  EXPECT_TRUE(link_data_a.coo_need_sync_to_device());
  EXPECT_TRUE(link_data_b.coo_need_sync_to_device());
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
