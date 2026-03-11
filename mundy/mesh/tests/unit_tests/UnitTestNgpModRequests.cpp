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
  auto& req_new_ids1 = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto& req_new_ids2 = reqs.request_entities_new_ids(*fixture.elem_part_2);

  auto& req_known_ids1 = reqs.request_entities_known_ids(*fixture.elem_part_1);
  auto& req_known_ids2 = reqs.request_entities_known_ids(*fixture.elem_part_2);

  auto& req_conns = reqs.request_connections();
  auto& req_destroy_entities = reqs.destroy_entities();
  auto& req_destroy_conns = reqs.destroy_connections();

  reqs.activate_host();

  // Validate that we can claim tickets for each class and have them properly generate the ticket ids and increment the counts
  auto claim_three_test = [](auto& issuer, const std::string &message) {
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

  auto& new_ids_a = reqs.request_entities_new_ids(parts_1);
  auto& new_ids_b = reqs.request_entities_new_ids(parts_2);
  EXPECT_EQ(new_ids_a.id(), new_ids_b.id());

  auto& known_ids_a = reqs.request_entities_known_ids(parts_1);
  auto& known_ids_b = reqs.request_entities_known_ids(parts_2);
  EXPECT_EQ(known_ids_a.id(), known_ids_b.id());
}

TEST(NgpModRequests, CopyIsViewLikeForSharedState) {
  NgpModRequestsRawFixture fixture;

  NgpModRequests reqs;
  NgpModRequests reqs_copy = reqs;

  auto& helper_from_original = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto& helper_from_copy = reqs_copy.request_entities_new_ids(*fixture.elem_part_1);
  EXPECT_EQ(helper_from_original.id(), helper_from_copy.id());

  reqs.activate_host();
  helper_from_original.element_tickets().claim(2);
  EXPECT_EQ(helper_from_original.element_tickets().count(), 2u);

  reqs_copy.activate_host();
  EXPECT_EQ(helper_from_copy.element_tickets().count(), 2u);

  helper_from_copy.element_tickets().claim(1);
  EXPECT_EQ(helper_from_original.element_tickets().count(), 3u);
}

TEST(NgpModRequests, HigherLevelCreateEntitiesAndConnectExistingToNew) {
  NgpModRequestsRawFixture fixture;
  BulkData& bulk = *fixture.bulk_data_ptr;

  bulk.modification_begin();
  stk::mesh::Entity existing_elem = bulk.declare_entity(stk::topology::ELEMENT_RANK, /*id*/ 1, stk::mesh::PartVector{fixture.elem_part_1});
  bulk.modification_end();
  ASSERT_TRUE(bulk.is_valid(existing_elem));

  NgpModRequests reqs;
  stk::mesh::PartVector no_parts{};
  auto& req_new_elems = reqs.request_entities_new_ids(*fixture.elem_part_1);
  auto& req_known_elems = reqs.request_entities_known_ids(*fixture.elem_part_2);
  auto& req_new_nodes = reqs.request_entities_new_ids(no_parts);
  auto& req_conns = reqs.request_connections();

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

TEST(NgpModRequests, HigherLevelDestroyConnectionAndEntity) {
  NgpModRequestsRawFixture fixture;
  BulkData& bulk = *fixture.bulk_data_ptr;

  bulk.modification_begin();
  stk::mesh::Entity elem = bulk.declare_entity(stk::topology::ELEMENT_RANK, /*id*/ 1, stk::mesh::PartVector{fixture.elem_part_1});
  stk::mesh::Entity node = bulk.declare_entity(stk::topology::NODE_RANK, /*id*/ 1, stk::mesh::PartVector{});
  bulk.declare_relation(elem, node, 0);
  bulk.modification_end();
  ASSERT_TRUE(bulk.is_valid(elem));
  ASSERT_TRUE(bulk.is_valid(node));

  NgpModRequests reqs;
  auto& destroy_conns = reqs.destroy_connections();
  auto& destroy_entities = reqs.destroy_entities();

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

/*
Stuff to test:

 Raw functionality:
  - TicketIssuer functions as desired
    initialize(activate_device)
    activate_host()
    activate_device()
    reset()
    finalize_count()
    claim(N)
    claim()
  - NgpModRequests should be a "view"-like class (cheap to copy where modifying a copy modifies the underlying data)
  - request_entities_new_ids and request_entities_known_ids should memoize their return based on the given key (sorted
and uniqued set of parts)

 Higher level functionality:
  - Can successfully request entities with new ids, request entities with known ids, and request connections between
existing and new entities, and then fetch the new entities after processing requests
  - Can successfully destroy entities and connections
*/

// TEST(NgpModRequests, BasicUsage) {
//   // Setup
//   MeshBuilder builder(MPI_COMM_WORLD);
//   builder.set_spatial_dimension(3);
//   builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
//   std::shared_ptr<MetaData> meta_data_ptr = builder.create_meta_data();
//   MetaData& meta_data = *meta_data_ptr;
//   meta_data.use_simple_fields();
//   std::shared_ptr<BulkData> bulk_data_ptr = builder.create_bulk_data(meta_data_ptr);
//   BulkData& bulk_data = *bulk_data_ptr;
//   stk::mesh::Part& elem_part = meta_data.declare_part("PART_1", stk::topology::ELEM_RANK);
//   stk::mesh::Part& particle_part = meta_data.declare_part_with_topology("PART_2", stk::topology::PARTICLE);
//   meta_data.commit();

//   // Declare two particles
//   // bulk_data.modification_begin();
//   // Entity sphere1 = bulk_data.declare_element(1, stk::mesh::ConstPartVector{&spheres_part});
//   // Entity sphere2 = bulk_data.declare_element(2, stk::mesh::ConstPartVector{&spheres_part});
//   // Entity node1 = bulk_data.declare_node(1);
//   // Entity node2 = bulk_data.declare_node(2);
//   // bulk_data.declare_relation(sphere1, node1, 0);
//   // bulk_data.declare_relation(sphere1, node2, 1);
//   // bulk_data.modification_end();

//   // Run the test
// }

}  // namespace

}  // namespace mesh

}  // namespace mundy
