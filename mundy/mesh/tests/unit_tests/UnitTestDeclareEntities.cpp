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

// C++ core
#include <memory>  // for std::unique_ptr
#include <vector>  // for std::vector

// STK
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_topology/topology.hpp>

// Mundy
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/LinkMetaData.hpp>     // for mundy::mesh::LinkMetaData
#include <mundy_mesh/LinkData.hpp>        // for mundy::mesh::LinkData

namespace mundy {

namespace mesh {

namespace {

static const double initial_value[3] = {1.1, 2.2, 3.3};

class UnitTestDeclareEntities : public ::testing::Test {
 public:
  using DoubleField = stk::mesh::Field<double>;

  UnitTestDeclareEntities()
      : communicator_(MPI_COMM_WORLD),
        spatial_dimension_(3),
        entity_rank_names_({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"}) {
  }

  virtual ~UnitTestDeclareEntities() {
    reset_mesh();
  }

  void reset_mesh() {
    bulk_data_ptr_.reset();
    meta_data_ptr_.reset();
  }

  virtual stk::mesh::BulkData& get_bulk() {
    EXPECT_NE(bulk_data_ptr_, nullptr) << "Trying to get bulk data before it has been initialized.";
    return *bulk_data_ptr_;
  }

  DoubleField* create_field_on_parts(const std::string& field_name, const stk::mesh::EntityRank& entity_rank,
                                     const int& num_components, const stk::mesh::PartVector& parts) {
    DoubleField& field = meta_data_ptr_->declare_field<double>(entity_rank, field_name);
    for (stk::mesh::Part* part : parts) {
      stk::mesh::put_field_on_mesh(field, *part, num_components, initial_value);
    }
    return &field;
  }

  void setup_mesh(stk::mesh::BulkData::AutomaticAuraOption aura_option,
                  std::unique_ptr<stk::mesh::FieldDataManager> field_data_manager,
                  unsigned initial_bucket_capacity = stk::mesh::get_default_initial_bucket_capacity(),
                  unsigned maximum_bucket_capacity = stk::mesh::get_default_maximum_bucket_capacity()) {
    stk::mesh::MeshBuilder builder(communicator_);
    builder.set_spatial_dimension(spatial_dimension_);
    builder.set_entity_rank_names(entity_rank_names_);
    builder.set_aura_option(aura_option);
#if TRILINOS_MAJOR_MINOR_VERSION >= 160000
    builder.set_field_data_manager(std::move(field_data_manager));
#else
    builder.set_field_data_manager(field_data_manager.get());
#endif
    builder.set_initial_bucket_capacity(initial_bucket_capacity);
    builder.set_maximum_bucket_capacity(maximum_bucket_capacity);

    meta_data_ptr_ = builder.create_meta_data();
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->set_coordinate_field_name("coordinates");

    link_meta_data_nodes_ptr_ = declare_link_meta_data_ptr(*meta_data_ptr_, "ALL_LINKS", stk::topology::NODE_RANK);
    link_meta_data_elems_ptr_ = declare_link_meta_data_ptr(*meta_data_ptr_, "ALL_LINKS", stk::topology::ELEM_RANK);

    ASSERT_TRUE(link_meta_data_nodes_ptr_ != nullptr);
    ASSERT_TRUE(link_meta_data_elems_ptr_ != nullptr);
    ASSERT_TRUE(link_meta_data_nodes_ptr_ != link_meta_data_elems_ptr_);  // Just like fields, you can replicate names as long as the rank is different

    bulk_data_ptr_ = builder.create(meta_data_ptr_);
    link_data_nodes_ptr_ = declare_link_data_ptr(*bulk_data_ptr_, *link_meta_data_nodes_ptr_);
    link_data_elems_ptr_ = declare_link_data_ptr(*bulk_data_ptr_, *link_meta_data_elems_ptr_);
    aura_option_ = aura_option;
    initial_bucket_capacity_ = initial_bucket_capacity;
    maximum_bucket_capacity_ = maximum_bucket_capacity;

    ASSERT_TRUE(bulk_data_ptr_ != nullptr);
    ASSERT_TRUE(link_data_nodes_ptr_ != nullptr);
    ASSERT_TRUE(link_data_elems_ptr_ != nullptr);

    // Declare parts and fields
    /* Method:
      - One part with topology (BEAM_2) and two fields (one node rank and one element rank)
      - One part with topology (PARTICLE) and one different element rank field
      - One element-rank part without a topology and two fields (one node rank and one element rank)
      - One part with topology (NODE) and no fields
    */

    beam_2_part_ptr_ = &meta_data_ptr_->declare_part_with_topology("beam_2_part", stk::topology::BEAM_2);
    particle_part_ptr_ = &meta_data_ptr_->declare_part_with_topology("particle_part", stk::topology::PARTICLE);
    element_rank_part_ptr_ = &meta_data_ptr_->declare_part("element_rank_part", stk::topology::ELEMENT_RANK);
    node_part_ptr_ = &meta_data_ptr_->declare_part_with_topology("node_part", stk::topology::NODE);

    elem_field_ptr_ =
        create_field_on_parts("elem_field", stk::topology::ELEMENT_RANK, 1, {beam_2_part_ptr_, element_rank_part_ptr_});
    other_elem_field_ptr_ =
        create_field_on_parts("other_elem_field", stk::topology::ELEMENT_RANK, 1, {particle_part_ptr_});
    node_field_ptr_ = create_field_on_parts("node_field", stk::topology::NODE_RANK, 1, {beam_2_part_ptr_});
    other_node_field_ptr_ =
        create_field_on_parts("other_node_field", stk::topology::NODE_RANK, 1, {element_rank_part_ptr_});

    // Sanity check field rank and part membership
    EXPECT_EQ(elem_field_ptr_->entity_rank(), stk::topology::ELEMENT_RANK);
    EXPECT_EQ(other_elem_field_ptr_->entity_rank(), stk::topology::ELEMENT_RANK);
    EXPECT_EQ(node_field_ptr_->entity_rank(), stk::topology::NODE_RANK);
    EXPECT_EQ(other_node_field_ptr_->entity_rank(), stk::topology::NODE_RANK);

    EXPECT_TRUE(elem_field_ptr_->defined_on(*beam_2_part_ptr_));
    EXPECT_TRUE(elem_field_ptr_->defined_on(*element_rank_part_ptr_));
    EXPECT_TRUE(other_elem_field_ptr_->defined_on(*particle_part_ptr_));
    EXPECT_TRUE(node_field_ptr_->defined_on(*beam_2_part_ptr_));

    // Check that the parts have the correct topology
    EXPECT_EQ(beam_2_part_ptr_->topology(), stk::topology::BEAM_2);
    EXPECT_EQ(particle_part_ptr_->topology(), stk::topology::PARTICLE);
    EXPECT_EQ(element_rank_part_ptr_->topology(), stk::topology::INVALID_TOPOLOGY);
    EXPECT_EQ(node_part_ptr_->topology(), stk::topology::NODE);

    // Setup the links
    unsigned dimensionality = 2;
    node_slinks_part_ptr_ = &link_meta_data_nodes_ptr_->declare_link_part("NODE_SURFACE_LINKS", dimensionality);
    elem_slinks_part_ptr_ = &link_meta_data_elems_ptr_->declare_link_part("ELEM_SURFACE_LINKS", dimensionality);

    meta_data_ptr_->commit();
  }

  void setup() {
    const int we_know_there_are_five_ranks = 5;
    auto field_data_manager = std::make_unique<stk::mesh::DefaultFieldDataManager>(we_know_there_are_five_ranks);
    setup_mesh(stk::mesh::BulkData::AUTO_AURA, std::move(field_data_manager));
  }

 protected:
  MPI_Comm communicator_;
  unsigned spatial_dimension_;
  std::vector<std::string> entity_rank_names_;

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<LinkMetaData> link_meta_data_nodes_ptr_;
  std::shared_ptr<LinkMetaData> link_meta_data_elems_ptr_;

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<LinkData> link_data_nodes_ptr_;
  std::shared_ptr<LinkData> link_data_elems_ptr_;

  stk::mesh::BulkData::AutomaticAuraOption aura_option_{stk::mesh::BulkData::AUTO_AURA};
  unsigned initial_bucket_capacity_ = 0;
  unsigned maximum_bucket_capacity_ = 0;

  DoubleField* elem_field_ptr_;
  DoubleField* other_elem_field_ptr_;
  DoubleField* node_field_ptr_;
  DoubleField* other_node_field_ptr_;

  stk::mesh::Part* beam_2_part_ptr_;
  stk::mesh::Part* particle_part_ptr_;
  stk::mesh::Part* element_rank_part_ptr_;
  stk::mesh::Part* node_part_ptr_;
  stk::mesh::Part* node_slinks_part_ptr_;
  stk::mesh::Part* elem_slinks_part_ptr_;
};  // class UnitTestDeclareEntities

TEST_F(UnitTestDeclareEntities, Fixture) {
  ASSERT_NO_THROW(setup());

  // None of the fields or parts should be null
  EXPECT_NE(elem_field_ptr_, nullptr);
  EXPECT_NE(other_elem_field_ptr_, nullptr);
  EXPECT_NE(node_field_ptr_, nullptr);
  EXPECT_NE(other_node_field_ptr_, nullptr);

  EXPECT_NE(beam_2_part_ptr_, nullptr);
  EXPECT_NE(particle_part_ptr_, nullptr);
  EXPECT_NE(element_rank_part_ptr_, nullptr);
  EXPECT_NE(node_part_ptr_, nullptr);
}

TEST_F(UnitTestDeclareEntities, DeclareEntities) {
  if (stk::parallel_machine_size(communicator_) != 1) {
    GTEST_SKIP() << "This test is only designed to run with 1 MPI rank.";
  }

  setup();

  /* Example usage declaring a pearl necklace (a chain of particles connected by springs):
   n1       n3
    \      /  \
     s1   s2   s3
      \  /      \
       n2        n4
  */

  const size_t num_nodes = 4;
  const size_t num_edges = num_nodes - 1;
  DeclareEntitiesHelper builder;
  for (size_t i = 0; i < num_nodes; ++i) {
    builder.create_node()
        .id(i + 1)
        .owning_proc(0)
        .add_part(node_part_ptr_)
        .add_field_data<double>(node_field_ptr_, {2.71 + i});
  }

  for (size_t i = 0; i < num_edges; ++i) {
    auto spring = builder.create_element();
    spring.id(i + 1)
        .owning_proc(0)
        .topology(stk::topology::BEAM_2)
        .nodes({i + 1, i + 2})
        .add_parts({beam_2_part_ptr_, element_rank_part_ptr_})
        .add_field_data<double>(elem_field_ptr_, 3.14 + i);
  }

  for (size_t i = 0; i < num_nodes; ++i) {
    auto particle = builder.create_element();
    particle.id(i + 1 + num_edges)
        .owning_proc(0)
        .topology(stk::topology::PARTICLE)
        .nodes({i + 1})
        .add_part(particle_part_ptr_)
        .add_field_data<double>(other_elem_field_ptr_, 1.23 + i);
  }

  EXPECT_NO_THROW(builder.check_consistency(*bulk_data_ptr_)) << "Builder consistency check failed.";

  bulk_data_ptr_->modification_begin();
  builder.declare_entities(*bulk_data_ptr_);
  bulk_data_ptr_->modification_end();

  // Check that the nodes and elements were created
  auto node1 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 1);
  auto node2 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 2);
  auto node3 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 3);
  auto node4 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 4);
  stk::mesh::EntityVector nodes = {node1, node2, node3, node4};

  auto spring1 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 1);
  auto spring2 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 2);
  auto spring3 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 3);
  stk::mesh::EntityVector springs = {spring1, spring2, spring3};

  auto particle1 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 4);
  auto particle2 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 5);
  auto particle3 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 6);
  auto particle4 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 7);
  stk::mesh::EntityVector particles = {particle1, particle2, particle3, particle4};

  for (const auto& node : nodes) {
    EXPECT_TRUE(bulk_data_ptr_->is_valid(node)) << "Node " << bulk_data_ptr_->identifier(node) << " is not valid.";
  }

  for (const auto& spring : springs) {
    EXPECT_TRUE(bulk_data_ptr_->is_valid(spring))
        << "Spring " << bulk_data_ptr_->identifier(spring) << " is not valid.";
  }

  for (const auto& particle : particles) {
    EXPECT_TRUE(bulk_data_ptr_->is_valid(particle))
        << "Particle " << bulk_data_ptr_->identifier(particle) << " is not valid.";
  }

  // Check that the elements and nodes are in the correct parts
  for (const auto& node : nodes) {
    EXPECT_TRUE(bulk_data_ptr_->bucket(node).member(*node_part_ptr_));
  }

  for (const auto& spring : springs) {
    EXPECT_TRUE(bulk_data_ptr_->bucket(spring).member(*beam_2_part_ptr_));
    EXPECT_TRUE(bulk_data_ptr_->bucket(spring).member(*element_rank_part_ptr_));
  }

  for (const auto& particle : particles) {
    EXPECT_TRUE(bulk_data_ptr_->bucket(particle).member(*particle_part_ptr_));
  }

  // Check that the elements and nodes have the correct topology
  for (const auto& node : nodes) {
    EXPECT_EQ(bulk_data_ptr_->bucket(node).topology(), stk::topology::NODE);
  }

  for (const auto& spring : springs) {
    EXPECT_EQ(bulk_data_ptr_->bucket(spring).topology(), stk::topology::BEAM_2);
  }

  for (const auto& particle : particles) {
    EXPECT_EQ(bulk_data_ptr_->bucket(particle).topology(), stk::topology::PARTICLE);
  }

  // Check that the nodes are connected to the correct elements
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(spring1)[0], node1);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(spring1)[1], node2);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(spring2)[0], node2);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(spring2)[1], node3);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(spring3)[0], node3);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(spring3)[1], node4);

  EXPECT_EQ(bulk_data_ptr_->begin_nodes(particle1)[0], node1);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(particle2)[0], node2);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(particle3)[0], node3);
  EXPECT_EQ(bulk_data_ptr_->begin_nodes(particle4)[0], node4);

  // Check that the elements have the correct field data
  for (const auto& spring : springs) {
    EXPECT_DOUBLE_EQ(stk::mesh::field_data(*elem_field_ptr_, spring)[0], 3.14 + bulk_data_ptr_->identifier(spring) - 1);

    const stk::mesh::Entity* spring_nodes = bulk_data_ptr_->begin_nodes(spring);
    EXPECT_DOUBLE_EQ(stk::mesh::field_data(*node_field_ptr_, spring_nodes[0])[0],
                     2.71 + bulk_data_ptr_->identifier(spring_nodes[0]) - 1);
    EXPECT_DOUBLE_EQ(stk::mesh::field_data(*node_field_ptr_, spring_nodes[1])[0],
                     2.71 + bulk_data_ptr_->identifier(spring_nodes[1]) - 1);
  }

  for (const auto& particle : particles) {
    EXPECT_DOUBLE_EQ(stk::mesh::field_data(*other_elem_field_ptr_, particle)[0],
                     1.23 + bulk_data_ptr_->identifier(particle) - 1 - num_edges);
  }
}

TEST_F(UnitTestDeclareEntities, DeclareLinks) {
  if (stk::parallel_machine_size(communicator_) != 1) {
    GTEST_SKIP() << "This test is only designed to run with 1 MPI rank.";
  }

  // Setup: Two particles (elem + node) with a spring (elem + 2 nodes) attached to their surface via links
  // The links are members of the link meta data with name "LINKS"
  setup();

  DeclareEntitiesHelper builder;
  
  // Create two particles
  for (size_t i = 0; i < 2; ++i) {
    builder.create_node()
        .id(i + 1)
        .owning_proc(0)
        .add_part(node_part_ptr_);
    builder.create_element()
        .id(i + 1)
        .owning_proc(0)
        .topology(stk::topology::PARTICLE)
        .nodes({i + 1})
        .add_part(particle_part_ptr_);
  }

  // Create a spring
  builder.create_node()
      .id(3)
      .owning_proc(0)
      .add_part(node_part_ptr_);
  builder.create_node()
      .id(4)
      .owning_proc(0)
      .add_part(node_part_ptr_);
  builder.create_element()
      .id(3)
      .owning_proc(0)
      .topology(stk::topology::BEAM_2)
      .nodes({3, 4})
      .add_part(beam_2_part_ptr_);

  // Create two node rank links
  builder.create_node()
      .id(5)
      .owning_proc(0)
      .add_part(node_slinks_part_ptr_)  // Endows this node with link dimensionality 2 within the node rank "ALL_LINKS" link data
      .links_to(link_data_nodes_ptr_.get(), 1, stk::topology::ELEM_RANK, 0)  // Link to particle 1
      .links_to(link_data_nodes_ptr_.get(), 3, stk::topology::NODE_RANK, 1); // Link to spring node 1
  builder.create_node()
      .id(6)
      .owning_proc(0)
      .add_part(node_slinks_part_ptr_)  // Endows this node with link dimensionality 2 within the node rank "ALL_LINKS" link data
      .links_to(link_data_nodes_ptr_.get(), 2, stk::topology::ELEM_RANK, 0)  // Link to particle 2
      .links_to(link_data_nodes_ptr_.get(), 4, stk::topology::NODE_RANK, 1); // Link to spring node 2

  EXPECT_NO_THROW(builder.check_consistency(*bulk_data_ptr_)) << "Builder consistency check failed.";
  bulk_data_ptr_->modification_begin();
  builder.declare_entities(*bulk_data_ptr_);
  bulk_data_ptr_->modification_end();

  // Validate that the two particles (elem + node), a spring (elem + 2 nodes), and two node-rank linkers were created
  // Particles
  stk::mesh::Entity particle_node1 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 1);
  stk::mesh::Entity particle_node2 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 2);
  stk::mesh::Entity particle1 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 1);
  stk::mesh::Entity particle2 = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 2);
  EXPECT_TRUE(bulk_data_ptr_->is_valid(particle_node1))
      << "Node " << bulk_data_ptr_->identifier(particle_node1) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(particle_node2))
      << "Node " << bulk_data_ptr_->identifier(particle_node2) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(particle1)) << "Particle " << bulk_data_ptr_->identifier(particle1) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(particle2)) << "Particle " << bulk_data_ptr_->identifier(particle2) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->bucket(particle_node1).member(*node_part_ptr_));
  EXPECT_TRUE(bulk_data_ptr_->bucket(particle_node2).member(*node_part_ptr_));
  EXPECT_TRUE(bulk_data_ptr_->bucket(particle_node1).member(*particle_part_ptr_)) << "Inherited part membership failed";
  EXPECT_TRUE(bulk_data_ptr_->bucket(particle_node2).member(*particle_part_ptr_)) << "Inherited part membership failed";
  EXPECT_TRUE(bulk_data_ptr_->bucket(particle1).member(*particle_part_ptr_));
  EXPECT_TRUE(bulk_data_ptr_->bucket(particle2).member(*particle_part_ptr_));

  // Spring
  stk::mesh::Entity spring_node1 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 3);
  stk::mesh::Entity spring_node2 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 4);
  stk::mesh::Entity spring = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 3);
  EXPECT_TRUE(bulk_data_ptr_->is_valid(spring_node1))
      << "Node " << bulk_data_ptr_->identifier(spring_node1) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(spring_node2))
      << "Node " << bulk_data_ptr_->identifier(spring_node2) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(spring)) << "Spring " << bulk_data_ptr_->identifier(spring) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->bucket(spring_node1).member(*node_part_ptr_));
  EXPECT_TRUE(bulk_data_ptr_->bucket(spring_node2).member(*node_part_ptr_));
  EXPECT_TRUE(bulk_data_ptr_->bucket(spring_node1).member(*beam_2_part_ptr_)) << "Inherited part membership failed";
  EXPECT_TRUE(bulk_data_ptr_->bucket(spring_node2).member(*beam_2_part_ptr_)) << "Inherited part membership failed";
  EXPECT_TRUE(bulk_data_ptr_->bucket(spring).member(*beam_2_part_ptr_));

  // Node-rank linkers
  stk::mesh::Entity link_node1 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 5);
  stk::mesh::Entity link_node2 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 6);
  EXPECT_TRUE(bulk_data_ptr_->is_valid(link_node1))
      << "Link Node " << bulk_data_ptr_->identifier(link_node1) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(link_node2))
      << "Link Node " << bulk_data_ptr_->identifier(link_node2) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->bucket(link_node1).member(*node_slinks_part_ptr_));
  EXPECT_TRUE(bulk_data_ptr_->bucket(link_node2).member(*node_slinks_part_ptr_));
    
  // Verify the links | node_coo_data.get_linked_entity(
  LinkCOOData &node_coo_data = link_data_nodes_ptr_->coo_data();
  ASSERT_TRUE(node_coo_data.is_valid());
  stk::mesh::Entity linked_entity0 = node_coo_data.get_linked_entity(link_node1, 0);
  stk::mesh::Entity linked_entity1 = node_coo_data.get_linked_entity(link_node1, 1);
  stk::mesh::Entity linked_entity2 = node_coo_data.get_linked_entity(link_node2, 0);
  stk::mesh::Entity linked_entity3 = node_coo_data.get_linked_entity(link_node2, 1);
  EXPECT_TRUE(bulk_data_ptr_->is_valid(linked_entity0))
      << "Linked entity 0 from link node " << bulk_data_ptr_->identifier(link_node1) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(linked_entity1))
      << "Linked entity 1 from link node " << bulk_data_ptr_->identifier(link_node1) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(linked_entity2))
      << "Linked entity 0 from link node " << bulk_data_ptr_->identifier(link_node2) << " is not valid.";
  EXPECT_TRUE(bulk_data_ptr_->is_valid(linked_entity3))
      << "Linked entity 1 from link node " << bulk_data_ptr_->identifier(link_node2) << " is not valid.";

  EXPECT_TRUE(node_coo_data.get_linked_entity(link_node1, 0) == particle1);
  EXPECT_TRUE(node_coo_data.get_linked_entity(link_node1, 1) == spring_node1);
  EXPECT_TRUE(node_coo_data.get_linked_entity(link_node2, 0) == particle2);
  EXPECT_TRUE(node_coo_data.get_linked_entity(link_node2, 1) == spring_node2);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
