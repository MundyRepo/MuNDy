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
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/DumpMeshInfo.hpp>  // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_mesh/base/MeshUtils.hpp>     // for stk::mesh::fixup_ghosted_to_shared_nodes
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/Linkers.hpp>          // for mundy::linkers::connect_linker_to_entitys_nodes
#include <mundy_linkers/NeighborLinkers.hpp>  // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderLinkers
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderSegmentLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSpherocylinderLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSpherocylinderSegmentLinkers
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>    // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_mesh/MeshBuilder.hpp>    // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>      // for mundy::meta::FieldReqs
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/Spheres.hpp>                 // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments
#include <mundy_shapes/Spherocylinders.hpp>         // for mundy::shapes::Spherocylinders

namespace mundy {

namespace linkers {

namespace {

//! \name ComputeSignedSeparationDistanceAndContactNormal functionality unit tests
//@{

TEST(ComputeSignedSeparationDistanceAndContactNormal,
     PerformsCalculationCorrectlyForPairsOfSpheresSpherocylindersAndSpherocylinderSegments) {
  /* Check that ComputeSignedSeparationDistanceAndContactNormal works correctly for each type of pair of spheres,
  spherocylinders, and spherocylinder segments.

  This is not a rigorous test, but it is a good starting point to ensure that the class is functioning correctly. Later
  tests will generate more complex configurations.

  This test is designed for either 1 or 2 ranks. For the 1 rank case, rank 0 will own all the entities. For the 2 rank
  case, rank 0 will own the first half of the entities and rank 1 will own the second half of the entities and linkers
  will span the two ranks.

  The configuration of shapes is as follows:
    Sp1: radius = 1.5, position = (0.0, 0.0, 0.0)
    Sp2: radius = 2.0, position = (1.0, 2.0, 2.0)

    Sy1: radius = 2.0, length = 5.0,  position = (-1.0, 0.0, 0.0), orientated to align with the y-axis
    Sy2: radius = 1.5, length = 5.0,  position = (-2.0, 0.0, 0.0), orientated to align with the z-axis

    Seg1: left_endpt = (0.0, 1.5, 0.0), right_endpt = (1.0, 1.5, 0.0), radius = 1.5
    Seg2: left_endpt = (0.0, 2.0, 0.0), right_endpt = (1.0, 2.0, 0.0), radius = 2.0

  The expected results are thus:
    Sp1-Sp2:   ssd = -0.5, cn = (1/3, 2/3, 2/3)
    Sp1-Sy1:   ssd = -2.5, cn = (-1, 0, 0)
    Sp1-Seg1:  ssd = -1.5, cn = (0, 1, 0)
    Sy1-Seg1:  ssd = -2.5, cn = (1, 0, 0)
    Sy1-Sy2:   ssd = -2.5, cn = (-1, 0, 0)
    Seg1-Seg2: ssd = -3.0, cn = (0, 1, 0)
  */

  // Create an instance of ComputeSignedSeparationDistanceAndContactNormal based on committed mesh that meets the
  // default requirements for ComputeSignedSeparationDistanceAndContactNormal.
  auto [compute_ssd_and_cn_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          ComputeSignedSeparationDistanceAndContactNormal>();
  ASSERT_TRUE(compute_ssd_and_cn_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // This test is designed for either 1 or 2 ranks.
  const int num_ranks = bulk_data_ptr->parallel_size();
  const int rank = bulk_data_ptr->parallel_rank();
  MUNDY_THROW_REQUIRE(num_ranks == 1 || num_ranks == 2, std::logic_error, "This test is designed for 1 or 2 ranks.");

  // Fetch the parts.
  stk::mesh::Part *sphere_part_ptr = meta_data_ptr->get_part(mundy::shapes::Spheres::get_name());
  stk::mesh::Part *spherocylinder_part_ptr = meta_data_ptr->get_part(mundy::shapes::Spherocylinders::get_name());
  stk::mesh::Part *spherocylinder_segment_part_ptr =
      meta_data_ptr->get_part(mundy::shapes::SpherocylinderSegments::get_name());

  stk::mesh::Part *neighbor_linker_part_ptr = meta_data_ptr->get_part(NeighborLinkers::get_name());
  stk::mesh::Part *sphere_sphere_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SphereSphereLinkers::get_name());
  stk::mesh::Part *sphere_spherocylinder_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SphereSpherocylinderLinkers::get_name());
  stk::mesh::Part *sphere_spherocylinder_segment_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SphereSpherocylinderSegmentLinkers::get_name());
  stk::mesh::Part *spherocylinder_segment_spherocylinder_segment_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::get_name());
  stk::mesh::Part *spherocylinder_spherocylinder_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SpherocylinderSpherocylinderLinkers::get_name());
  stk::mesh::Part *spherocylinder_spherocylinder_segment_linker_part_ptr =
      meta_data_ptr->get_part(neighbor_linkers::SpherocylinderSpherocylinderSegmentLinkers::get_name());

  for (auto part_ptr :
       {sphere_part_ptr, spherocylinder_part_ptr, spherocylinder_segment_part_ptr, neighbor_linker_part_ptr,
        sphere_sphere_linker_part_ptr, sphere_spherocylinder_linker_part_ptr,
        sphere_spherocylinder_segment_linker_part_ptr, spherocylinder_segment_spherocylinder_segment_linker_part_ptr,
        spherocylinder_spherocylinder_linker_part_ptr}) {
    ASSERT_TRUE(part_ptr != nullptr);
  }

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<double> *radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *length_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_LENGTH");
  stk::mesh::Field<double> *orientation_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_ORIENTATION");
  stk::mesh::Field<double> *linker_ssd_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_SIGNED_SEPARATION_DISTANCE");
  stk::mesh::Field<double> *linker_cn_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_CONTACT_NORMAL");
  LinkedEntitiesFieldType *linked_entities_field_ptr = meta_data_ptr->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, "LINKED_NEIGHBOR_ENTITIES");
  stk::mesh::Field<int> *linked_entity_owners_field_ptr =
      meta_data_ptr->get_field<int>(stk::topology::CONSTRAINT_RANK, "LINKED_NEIGHBOR_ENTITY_OWNERS");

  ASSERT_TRUE(node_coord_field_ptr != nullptr);
  ASSERT_TRUE(radius_field_ptr != nullptr);
  ASSERT_TRUE(length_field_ptr != nullptr);
  ASSERT_TRUE(orientation_field_ptr != nullptr);
  ASSERT_TRUE(linker_ssd_field_ptr != nullptr);
  ASSERT_TRUE(linker_cn_field_ptr != nullptr);
  ASSERT_TRUE(linked_entities_field_ptr != nullptr);
  ASSERT_TRUE(linked_entity_owners_field_ptr != nullptr);

  // Declare two of each type of shape for 3 * 2 total shapes and generate 6 linkers between the unqiue shape pairings.
  // Typically linker generation would be done via a generator class, but we'll do it by hand here to keep the test
  // simple.
  //
  // Each rank declares one of each shape and Rank 0 declares the linkers between the shapes. Rank 1 will ghost its
  // shapes to Rank 0.
  bulk_data_ptr->modification_begin();
  std::vector<stk::mesh::EntityProc> send_shapes_to_rank0;

  // Declare the spheres.
  if (rank == 0) {
    stk::mesh::Entity sp1_element = bulk_data_ptr->declare_element(1, stk::mesh::ConstPartVector{sphere_part_ptr});
    stk::mesh::Entity sp1_node = bulk_data_ptr->declare_node(1);
    bulk_data_ptr->declare_relation(sp1_element, sp1_node, 0);
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sp2_element = bulk_data_ptr->declare_element(2, stk::mesh::ConstPartVector{sphere_part_ptr});
    stk::mesh::Entity sp2_node = bulk_data_ptr->declare_node(2);
    bulk_data_ptr->declare_relation(sp2_element, sp2_node, 0);
    send_shapes_to_rank0.push_back(std::make_pair(sp2_element, 0));
  }

  // Declare the spherocylinders
  if (rank == 0) {
    stk::mesh::Entity sy1_element =
        bulk_data_ptr->declare_element(3, stk::mesh::ConstPartVector{spherocylinder_part_ptr});
    stk::mesh::Entity sy1_node = bulk_data_ptr->declare_node(3);
    bulk_data_ptr->declare_relation(sy1_element, sy1_node, 0);
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sy2_element =
        bulk_data_ptr->declare_element(4, stk::mesh::ConstPartVector{spherocylinder_part_ptr});
    stk::mesh::Entity sy2_node = bulk_data_ptr->declare_node(4);
    bulk_data_ptr->declare_relation(sy2_element, sy2_node, 0);
    send_shapes_to_rank0.push_back(std::make_pair(sy2_element, 0));
  }

  // Declare the spherocylinder segments
  if (rank == 0) {
    stk::mesh::Entity seg1_element =
        bulk_data_ptr->declare_element(5, stk::mesh::ConstPartVector{spherocylinder_segment_part_ptr});
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->declare_node(5);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->declare_node(6);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node1, 0);
    bulk_data_ptr->declare_relation(seg1_element, seg1_node2, 1);
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity seg2_element =
        bulk_data_ptr->declare_element(6, stk::mesh::ConstPartVector{spherocylinder_segment_part_ptr});
    stk::mesh::Entity seg2_node1 = bulk_data_ptr->declare_node(7);
    stk::mesh::Entity seg2_node2 = bulk_data_ptr->declare_node(8);
    bulk_data_ptr->declare_relation(seg2_element, seg2_node1, 0);
    bulk_data_ptr->declare_relation(seg2_element, seg2_node2, 1);
    send_shapes_to_rank0.push_back(std::make_pair(seg2_element, 0));
  }
  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting("GHOST_SHAPES_TO_RANK0");
  bulk_data_ptr->change_ghosting(ghosting, send_shapes_to_rank0);
  bulk_data_ptr->modification_end();

  bulk_data_ptr->modification_begin();
  if (rank == 0) {
    // Fatch the declared entities
    stk::mesh::Entity sp1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    stk::mesh::Entity sp2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
    stk::mesh::Entity sy1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
    stk::mesh::Entity sy2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 4);
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 5);
    stk::mesh::Entity seg2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 6);

    for (auto &entity : {sp1_element, sp2_element, sy1_element, sy2_element, seg1_element, seg2_element}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(entity));
    }

    // Declare the linkers
    // one sphere-sphere linker, one sphere-spherocylinder linker, one sphere-spherocylinder segment linker, and so on
    stk::mesh::Entity sp1_sp2_linker =
        bulk_data_ptr->declare_constraint(1, stk::mesh::ConstPartVector{sphere_sphere_linker_part_ptr});
    mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr, sp1_sp2_linker, sp1_element, sp2_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sp1_sp2_linker)[0] = bulk_data_ptr->entity_key(sp1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sp1_sp2_linker)[1] = bulk_data_ptr->entity_key(sp2_element);

    stk::mesh::Entity sp1_sy1_linker =
        bulk_data_ptr->declare_constraint(2, stk::mesh::ConstPartVector{sphere_spherocylinder_linker_part_ptr});
    mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr, sp1_sy1_linker, sp1_element, sy1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sp1_sy1_linker)[0] = bulk_data_ptr->entity_key(sp1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sp1_sy1_linker)[1] = bulk_data_ptr->entity_key(sy1_element);

    stk::mesh::Entity sp1_seg1_linker =
        bulk_data_ptr->declare_constraint(3, stk::mesh::ConstPartVector{sphere_spherocylinder_segment_linker_part_ptr});
    mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr, sp1_seg1_linker, sp1_element, seg1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sp1_seg1_linker)[0] = bulk_data_ptr->entity_key(sp1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sp1_seg1_linker)[1] = bulk_data_ptr->entity_key(seg1_element);

    stk::mesh::Entity seg1_seg2_linker = bulk_data_ptr->declare_constraint(
        4, stk::mesh::ConstPartVector{spherocylinder_segment_spherocylinder_segment_linker_part_ptr});
    mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr, seg1_seg2_linker, seg1_element, seg2_element);
    stk::mesh::field_data(*linked_entities_field_ptr, seg1_seg2_linker)[0] = bulk_data_ptr->entity_key(seg1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, seg1_seg2_linker)[1] = bulk_data_ptr->entity_key(seg2_element);

    stk::mesh::Entity sy1_sy2_linker =
        bulk_data_ptr->declare_constraint(5, stk::mesh::ConstPartVector{spherocylinder_spherocylinder_linker_part_ptr});
    mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr, sy1_sy2_linker, sy1_element, sy2_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sy1_sy2_linker)[0] = bulk_data_ptr->entity_key(sy1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sy1_sy2_linker)[1] = bulk_data_ptr->entity_key(sy2_element);

    stk::mesh::Entity sy1_seg1_linker = bulk_data_ptr->declare_constraint(
        6, stk::mesh::ConstPartVector{spherocylinder_spherocylinder_segment_linker_part_ptr});
    mundy::linkers::connect_linker_to_entitys_nodes(*bulk_data_ptr, sy1_seg1_linker, sy1_element, seg1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sy1_seg1_linker)[0] = bulk_data_ptr->entity_key(sy1_element);
    stk::mesh::field_data(*linked_entities_field_ptr, sy1_seg1_linker)[1] = bulk_data_ptr->entity_key(seg1_element);
  }

  // We're using one-sided linker creation, so we need to fixup the ghosted to shared nodes.
  stk::mesh::fixup_ghosted_to_shared_nodes(*bulk_data_ptr);
  bulk_data_ptr->modification_end();

  // Ghost the linked entities to any process that owns any of the other linked entities.
  bulk_data_ptr->modification_begin();
  const stk::mesh::Selector linker_parts_selector = stk::mesh::selectUnion(stk::mesh::ConstPartVector{
      sphere_sphere_linker_part_ptr, sphere_spherocylinder_linker_part_ptr,
      sphere_spherocylinder_segment_linker_part_ptr, spherocylinder_segment_spherocylinder_segment_linker_part_ptr,
      spherocylinder_spherocylinder_linker_part_ptr, spherocylinder_spherocylinder_segment_linker_part_ptr});
  mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr, *linked_entities_field_ptr,
                                               *linked_entity_owners_field_ptr, linker_parts_selector);
  bulk_data_ptr->modification_end();

  // Initialize the spheres. Only performed form local entities.
  if (rank == 0) {
    stk::mesh::Entity sp1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    stk::mesh::Entity sp1_node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 1);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp1_element));
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp1_node));
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, sp1_node).set(0.0, 0.0, 0.0);
    stk::mesh::field_data(*radius_field_ptr, sp1_element)[0] = 1.5;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sp2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
    stk::mesh::Entity sp2_node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 2);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp2_element));
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp2_node));
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, sp2_node).set(1.0, 2.0, 2.0);
    stk::mesh::field_data(*radius_field_ptr, sp2_element)[0] = 2.0;
  }

  // Initialize the spherocylinders. Only performed form local entities.
  // As a reminder, the quaternion orientation of the sylinder represents the rotation from the local frame (in which
  // the rode is aligned with the x-axis) to the global frame.
  if (rank == 0) {
    stk::mesh::Entity sy1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
    stk::mesh::Entity sy1_node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 3);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy1_element));
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy1_node));
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, sy1_node).set(-1.0, 0.0, 0.0);
    stk::mesh::field_data(*radius_field_ptr, sy1_element)[0] = 2.0;
    stk::mesh::field_data(*length_field_ptr, sy1_element)[0] = 5.0;

    // Orientate the spherocylinder to align with the y-axis.
    auto orientation = mundy::math::euler_to_quat(0.0, 0.0, M_PI / 2.0);
    mundy::mesh::quaternion_field_data(*orientation_field_ptr, sy1_element) = orientation;
    ASSERT_TRUE(
        is_approx_close(orientation * mundy::math::Vector3d{1.0, 0.0, 0.0}, mundy::math::Vector3d{0.0, 1.0, 0.0}))
        << "Failed to orient sy1 correctly";
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sy2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 4);
    stk::mesh::Entity sy2_node = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 4);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy2_element));
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy2_node));
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, sy2_node).set(-2.0, 0.0, 0.0);
    stk::mesh::field_data(*radius_field_ptr, sy2_element)[0] = 1.5;
    stk::mesh::field_data(*length_field_ptr, sy2_element)[0] = 5.0;

    // Orientate the spherocylinder to align with the z-axis.
    auto orientation = mundy::math::euler_to_quat(0.0, -M_PI / 2.0, 0.0);
    mundy::mesh::quaternion_field_data(*orientation_field_ptr, sy2_element) = orientation;
    ASSERT_TRUE(
        is_approx_close(orientation * mundy::math::Vector3d{1.0, 0.0, 0.0}, mundy::math::Vector3d{0.0, 0.0, 1.0}))
        << "Failed to orient sy2 correctly";
  }

  // Initialize the spherocylinder segments. Only performed form local entities.
  if (rank == 0) {
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 5);
    stk::mesh::Entity seg1_node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 5);
    stk::mesh::Entity seg1_node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 6);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_element));
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_node1));
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_node2));
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg1_node1).set(0.0, 1.5, 0.0);
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg1_node2).set(1.0, 1.5, 0.0);
    stk::mesh::field_data(*radius_field_ptr, seg1_element)[0] = 1.5;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity seg2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 6);
    stk::mesh::Entity seg2_node1 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 7);
    stk::mesh::Entity seg2_node2 = bulk_data_ptr->get_entity(stk::topology::NODE_RANK, 8);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg2_element));
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg2_node1));
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg2_node2));
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg2_node1).set(0.0, 2.0, 0.0);
    mundy::mesh::vector3_field_data(*node_coord_field_ptr, seg2_node2).set(1.0, 2.0, 0.0);
    stk::mesh::field_data(*radius_field_ptr, seg2_element)[0] = 2.0;
  }

  // Compute the signed separation distance and contact normal for all neighbor linkers.
  ASSERT_NO_THROW(compute_ssd_and_cn_ptr->execute(*neighbor_linker_part_ptr)) << "Failed to compute ssd and cn.";

  // Only rank 0 will check the results since it owns the linkers.
  if (rank == 0) {
    // Fetch the linkers
    stk::mesh::Entity sp1_sp2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 1);
    stk::mesh::Entity sp1_sy1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 2);
    stk::mesh::Entity sp1_seg1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 3);
    stk::mesh::Entity seg1_seg2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 4);
    stk::mesh::Entity sy1_sy2_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 5);
    stk::mesh::Entity sy1_seg1_linker = bulk_data_ptr->get_entity(stk::topology::CONSTRAINT_RANK, 6);
    for (auto &linker :
         {sp1_sp2_linker, sp1_sy1_linker, sp1_seg1_linker, seg1_seg2_linker, sy1_sy2_linker, sy1_seg1_linker}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker));
    }

    // Sphere-Sphere
    const auto cn_sp1_sp2 = mundy::mesh::vector3_field_data(*linker_cn_field_ptr, sp1_sp2_linker);
    const double ssd_sp1_sp2 = stk::mesh::field_data(*linker_ssd_field_ptr, sp1_sp2_linker)[0];
    EXPECT_TRUE(is_approx_close(cn_sp1_sp2, mundy::math::Vector3d{1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0}))
        << "cn_sp1_sp2 = " << cn_sp1_sp2;
    EXPECT_DOUBLE_EQ(ssd_sp1_sp2, -0.5);

    // Sphere-Spherocylinder
    const auto cn_sp1_sy1 = mundy::mesh::vector3_field_data(*linker_cn_field_ptr, sp1_sy1_linker);
    const double ssd_sp1_sy1 = stk::mesh::field_data(*linker_ssd_field_ptr, sp1_sy1_linker)[0];
    EXPECT_TRUE(is_approx_close(cn_sp1_sy1, mundy::math::Vector3d{-1.0, 0.0, 0.0})) << "cn_sp1_sy1 = " << cn_sp1_sy1;
    EXPECT_DOUBLE_EQ(ssd_sp1_sy1, -2.5);

    // Sphere-Spherocylinder Segment
    const auto cn_sp1_seg1 = mundy::mesh::vector3_field_data(*linker_cn_field_ptr, sp1_seg1_linker);
    const double ssd_sp1_seg1 = stk::mesh::field_data(*linker_ssd_field_ptr, sp1_seg1_linker)[0];
    EXPECT_TRUE(is_approx_close(cn_sp1_seg1, mundy::math::Vector3d{0.0, 1.0, 0.0})) << "cn_sp1_seg1 = " << cn_sp1_seg1;
    EXPECT_DOUBLE_EQ(ssd_sp1_seg1, -1.5);

    // Spherocylinder Segment-Spherocylinder Segment
    const auto cn_seg1_seg2 = mundy::mesh::vector3_field_data(*linker_cn_field_ptr, seg1_seg2_linker);
    const double ssd_seg1_seg2 = stk::mesh::field_data(*linker_ssd_field_ptr, seg1_seg2_linker)[0];
    EXPECT_TRUE(is_approx_close(cn_seg1_seg2, mundy::math::Vector3d{0.0, 1.0, 0.0}))
        << "cn_seg1_seg2 = " << cn_seg1_seg2;
    EXPECT_DOUBLE_EQ(ssd_seg1_seg2, -3.0);

    // Spherocylinder-Spherocylinder
    const auto cn_sy1_sy2 = mundy::mesh::vector3_field_data(*linker_cn_field_ptr, sy1_sy2_linker);
    const double ssd_sy1_sy2 = stk::mesh::field_data(*linker_ssd_field_ptr, sy1_sy2_linker)[0];
    EXPECT_TRUE(is_approx_close(cn_sy1_sy2, mundy::math::Vector3d{-1.0, 0.0, 0.0})) << "cn_sy1_sy2 = " << cn_sy1_sy2;
    EXPECT_DOUBLE_EQ(ssd_sy1_sy2, -2.5);

    // Spherocylinder-Spherocylinder Segment
    const auto cn_sy1_seg1 = mundy::mesh::vector3_field_data(*linker_cn_field_ptr, sy1_seg1_linker);
    const double ssd_sy1_seg1 = stk::mesh::field_data(*linker_ssd_field_ptr, sy1_seg1_linker)[0];
    EXPECT_TRUE(is_approx_close(cn_sy1_seg1, mundy::math::Vector3d{1.0, 0.0, 0.0})) << "cn_sy1_seg1 = " << cn_sy1_seg1;
    EXPECT_DOUBLE_EQ(ssd_sy1_seg1, -2.5);
  }
}
//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
