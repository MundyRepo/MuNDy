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
#include <stk_mesh/base/Field.hpp>         // for stk::mesh::Field
#include <stk_mesh/base/MeshUtils.hpp>     // for stk::mesh::fixup_ghosted_to_shared_nodes
#include <stk_mesh/base/Types.hpp>         // for stk::mesh::ConstPartVector
#include <stk_topology/topology.hpp>       // for stk::topology
#include <stk_util/parallel/Parallel.hpp>  // for stk::ParallelMachine

// Mundy libs
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>  // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/Linkers.hpp>                   // for mundy::linkers::connect_linker_to_entitys_nodes
#include <mundy_linkers/neighbor_linkers/SphereSphereLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSphereLinkers
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderLinkers
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderSegmentLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSpherocylinderLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSpherocylinderSegmentLinkers
#include <mundy_mesh/BulkData.hpp>                                                        // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
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

//! \name EvaluateLinkerPotentials functionality unit tests
//@{

TEST(EvaluateLinkerPotentials, PerformsHertzianContactCalculationCorrectlyForSpheresSimple) {
  /* Check that EvaluateLinkerPotentials evaluates the Hertzian contact correctly for each type of pair of spheres,
  spherocylinders, and spherocylinder segments.

  For this test, we generate a 2 spheres, 2 spherocylinders, and 2 spherocylinder segments. generate neighbor linkers
  between unique pairs of these objects. For simplicity, we directly assign the signed separation distances and contact
  normals for these pairs. We then pass these fields to the EvaluateLinkerPotentials kernel, compute the potential
  force, and check it against the expected value.

  This test is designed for either 1 or 2 ranks. For the 1 rank case, rank 0 will own all the entities. For the 2 rank
  case, rank 0 will own the first half of the entities and rank 1 will own the second half of the entities and linkers
  will span the two ranks.

  Our configuration is as follows:
    Sp1-Sp2:   r1 = 1.5, r2 = 2.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sp1-Sy1:   r1 = 1.5, r2 = 2.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sp1-Seg1:  r1 = 1.5, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sy1-Seg1:  r1 = 2.0, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Sy1-Sy2:   r1 = 2.0, r2 = 3.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    Seg1-Seg2: r1 = 3.5, r2 = 4.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
  */

  if (stk::parallel_machine_size(MPI_COMM_WORLD) > 2) {
    GTEST_SKIP() << "This test is designed for 1 or 2 ranks.";
  }

  // Create an instance of EvaluateLinkerPotentials based on committed mesh that meets the
  // default requirements for EvaluateLinkerPotentials.
  auto hertzian_contact_fixed_params = Teuchos::ParameterList().set(
      "enabled_kernel_names",
      mundy::core::make_string_array("SPHERE_SPHERE_HERTZIAN_CONTACT", "SPHERE_SPHEROCYLINDER_HERTZIAN_CONTACT",
                                     "SPHERE_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT",
                                     "SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT",
                                     "SPHEROCYLINDER_SPHEROCYLINDER_HERTZIAN_CONTACT",
                                     "SPHEROCYLINDER_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
  auto [evaluate_linker_potentials_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<EvaluateLinkerPotentials>(
          {hertzian_contact_fixed_params});
  ASSERT_TRUE(evaluate_linker_potentials_ptr != nullptr);
  ASSERT_TRUE(bulk_data_ptr != nullptr);
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr);

  // This test is designed for either 1 or 2 ranks.
  const int num_ranks = bulk_data_ptr->parallel_size();
  const int rank = bulk_data_ptr->parallel_rank();

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

  ASSERT_TRUE(sphere_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_segment_part_ptr != nullptr);
  ASSERT_TRUE(neighbor_linker_part_ptr != nullptr);
  ASSERT_TRUE(sphere_sphere_linker_part_ptr != nullptr);
  ASSERT_TRUE(sphere_spherocylinder_linker_part_ptr != nullptr);
  ASSERT_TRUE(sphere_spherocylinder_segment_linker_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_segment_spherocylinder_segment_linker_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_spherocylinder_linker_part_ptr != nullptr);
  ASSERT_TRUE(spherocylinder_spherocylinder_segment_linker_part_ptr != nullptr);

  // Fetch the required fields.
  stk::mesh::Field<double> *linker_contact_normal_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_CONTACT_NORMAL");
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_YOUNGS_MODULUS");
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_POISSONS_RATIO");
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_SIGNED_SEPARATION_DISTANCE");
  stk::mesh::Field<double> *linker_potential_force_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::CONSTRAINT_RANK, "LINKER_POTENTIAL_FORCE");
  LinkedEntitiesFieldType *linked_entities_field_ptr = meta_data_ptr->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, "LINKED_NEIGHBOR_ENTITIES");
  stk::mesh::Field<int> *linked_entity_owners_field_ptr =
      meta_data_ptr->get_field<int>(stk::topology::CONSTRAINT_RANK, "LINKED_NEIGHBOR_ENTITY_OWNERS");

  ASSERT_TRUE(element_radius_field_ptr != nullptr);
  ASSERT_TRUE(element_youngs_modulus_field_ptr != nullptr);
  ASSERT_TRUE(element_poissons_ratio_field_ptr != nullptr);
  ASSERT_TRUE(linker_signed_separation_distance_field_ptr != nullptr);
  ASSERT_TRUE(linker_potential_force_field_ptr != nullptr);
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

    // Fill the linker contact normal with an arbitrary unit vector, we'll use [1, 2, 3] / sqrt(14).
    ASSERT_TRUE(linker_contact_normal_field_ptr != nullptr);
    for (auto &linker :
         {sp1_sp2_linker, sp1_sy1_linker, sp1_seg1_linker, seg1_seg2_linker, sy1_sy2_linker, sy1_seg1_linker}) {
      ASSERT_TRUE(bulk_data_ptr->is_valid(linker));
      mundy::mesh::vector3_field_data(*linker_contact_normal_field_ptr, linker)
          .set(1.0 / std::sqrt(14.0), 2.0 / std::sqrt(14.0), 3.0 / std::sqrt(14.0));
    }
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

  // Initialize the spheres. Only performed for local entities.
  // sp1 radius: 1.5, sp2 radius: 2.0, sy1 radius: 2.5, sy2 radius: 3.0, seg1 radius: 3.5, seg2 radius: 4.0
  if (rank == 0) {
    stk::mesh::Entity sp1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 1);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp1_element));
    stk::mesh::field_data(*element_radius_field_ptr, sp1_element)[0] = 1.5;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sp1_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sp1_element)[0] = 0.3;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sp2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 2);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sp2_element));
    stk::mesh::field_data(*element_radius_field_ptr, sp2_element)[0] = 2.0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sp2_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sp2_element)[0] = 0.3;
  }

  // Initialize the spherocylinders. Only performed form local entities.
  // As a reminder, the quaternion orientation of the sylinder represents the rotation from the local frame (in which
  // the rode is aligned with the x-axis) to the global frame.
  if (rank == 0) {
    stk::mesh::Entity sy1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 3);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy1_element));
    stk::mesh::field_data(*element_radius_field_ptr, sy1_element)[0] = 2.5;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sy1_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sy1_element)[0] = 0.3;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity sy2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 4);
    ASSERT_TRUE(bulk_data_ptr->is_valid(sy2_element));
    stk::mesh::field_data(*element_radius_field_ptr, sy2_element)[0] = 3.0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, sy2_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, sy2_element)[0] = 0.3;
  }

  // Initialize the spherocylinder segments. Only performed form local entities.
  if (rank == 0) {
    stk::mesh::Entity seg1_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 5);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg1_element));
    stk::mesh::field_data(*element_radius_field_ptr, seg1_element)[0] = 3.5;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, seg1_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, seg1_element)[0] = 0.3;
  }
  if (rank == 1 || num_ranks == 1) {
    stk::mesh::Entity seg2_element = bulk_data_ptr->get_entity(stk::topology::ELEMENT_RANK, 6);
    ASSERT_TRUE(bulk_data_ptr->is_valid(seg2_element));
    stk::mesh::field_data(*element_radius_field_ptr, seg2_element)[0] = 4.0;
    stk::mesh::field_data(*element_youngs_modulus_field_ptr, seg2_element)[0] = 1.0e6;
    stk::mesh::field_data(*element_poissons_ratio_field_ptr, seg2_element)[0] = 0.3;
  }

  // Initialize the linker signed separation distances. Only performed for local entities.
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
      stk::mesh::field_data(*linker_signed_separation_distance_field_ptr, linker)[0] = -0.5;
    }
  }

  // Execute the EvaluateLinkerPotentials kernel for all neighbor linkers.
  ASSERT_NO_THROW(evaluate_linker_potentials_ptr->execute(*neighbor_linker_part_ptr))
      << "Failed to evaluate linker potentials.";

  auto check_potential_force_magnitude = [&linker_potential_force_field_ptr](
                                             stk::mesh::Entity linker, const double &radius1, const double &radius2,
                                             const double &ssd, const double &poissons_ratio1,
                                             const double &youngs_modulus1, const double &poissons_ratio2,
                                             const double &youngs_modulus2, const std::string &message) {
    // Check that the result is as expected.
    const auto potential_force = mundy::mesh::vector3_field_data(*linker_potential_force_field_ptr, linker);
    const double potential_force_magnitude = mundy::math::norm(potential_force);

    // The expected potential force is computed using the Hertzian contact model.
    // F = \frac{4}{3} E \sqrt{R} \delta^{3/2}
    // where:
    // - F is the contact force,
    // - E is the effective modulus of elasticity, calculated as
    //   E = \left( \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2} \right)^{-1},
    // - R is the effective radius of contact, defined as
    //   \frac{1}{R} = \frac{1}{R_1} + \frac{1}{R_2},
    // - \delta is the deformation at the contact point,
    // - R_1 and R_2 are the radii of the two spheres,
    // - E_1 and E_2 are the Young's moduli of the materials,
    // - \nu_1 and \nu_2 are the Poisson's ratios of the materials.
    // Delta is the amount of overlap between the two spheres.
    const double delta = -std::min(ssd, 0.0);
    const double E = 1.0 / ((1.0 - poissons_ratio1 * poissons_ratio1) / youngs_modulus1 +
                            (1.0 - poissons_ratio2 * poissons_ratio2) / youngs_modulus2);
    const double R = 1.0 / (1.0 / radius1 + 1.0 / radius2);
    const double expected_potential_force_magnitude = 4.0 / 3.0 * E * std::sqrt(R) * std::pow(delta, 1.5);
    EXPECT_DOUBLE_EQ(potential_force_magnitude, expected_potential_force_magnitude) << message;
  };

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

    // Sp1-Sp2:   r1 = 1.5, r2 = 2.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sp1-Sy1:   r1 = 1.5, r2 = 2.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sp1-Seg1:  r1 = 1.5, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sy1-Sy2:   r1 = 2.5, r2 = 3.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Sy1-Seg1:  r1 = 2.5, r2 = 3.5, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    // Seg1-Seg2: r1 = 3.5, r2 = 4.0, ssd = -0.5, poissons_ratio = 0.3, youngs_modulus = 1e6
    check_potential_force_magnitude(sp1_sp2_linker, 1.5, 2.0, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sp1_sp2_linker");
    check_potential_force_magnitude(sp1_sy1_linker, 1.5, 2.5, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sp1_sy1_linker");
    check_potential_force_magnitude(sp1_seg1_linker, 1.5, 3.5, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sp1_seg1_linker");
    check_potential_force_magnitude(sy1_sy2_linker, 2.5, 3.0, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sy1_sy2_linker");
    check_potential_force_magnitude(sy1_seg1_linker, 2.5, 3.5, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "sy1_seg1_linker");
    check_potential_force_magnitude(seg1_seg2_linker, 3.5, 4.0, -0.5, 0.3, 1.0e6, 0.3, 1.0e6, "seg1_seg2_linker");
  }
}

TEST(EvaluateLinkerPotentials, FrictionalHertzianContactSlideSlipSphere) {
  // There exists an expected value for the transition between sliding and rolling
  // sphere shot at a table such that it contacts with the table exactly along a line
  //
  //  o ->    _________
  //         |
  //
  //       o->_________
  //         |
  //
  //          __o->_____
  //         |
  //
  // The sphere is traveling to the right at an initial velocity of v0, experiences
  // a downward gravitational force of g, and the ball/sphere have a friction factor of mu.
  // Trickle 2003 states that the ball will initially slide and, but that at a time
  // t = 2 v0 / (7 mu g), it will transition to rolling.
  //
  // For this problem, we will simulate all the dynamics ourselves with EvaluateLinkerPotentials providing the
  // frictional hertzian contact force.
  //
  // Because this is a 2D problem, we'll represent the sphere's orientation using its azimuthal angle with the table.
  //
  // r = 1.0, rho = 1.0
  // F = m a, m = 4/3 pi r^3 rho
  // tau = I alpha, I = 2/5 m r^2
  //
  // For simplicity, we'll discretize this using velocity Verlet integration.
  //   v(t + dt/2) = v(t) + 0.5 a(t) dt
  //   x(t + dt) = x(t) + v(t + dt/2) dt
  //   a(t + dt) = F(x(t + dt)) / m
  //   v(t + dt) = v(t + dt/2) + 0.5 a(t + dt) dt
  //
  // Here F and tau are center of mass force and torque, respectively. The hertzian contact force is applied at the
  // contact point, which is at the bottom of the sphere. Hence, F = F_contact - m g and tau = (-r \hat{k}) x F_contact.

  // Replicate the EvaluateLinkerPotentials kernel for the frictional hertzian contact force and simplify it for our
  // needs.
  using Vector3 = mundy::math::Vector3d;
  auto eval_hertz_with_friction =
      [](const double &time_step_size, const double &density, const double &normal_spring_coeff,
         const double &tang_spring_coeff, const double &normal_damping_coeff, const double &tang_damping_coeff,
         const double &friction_coeff, const double &left_radius, const double &right_radius,
         const double &signed_sep_dist, const Vector3 &left_cp_vel, const Vector3 &right_cp_vel,
         const Vector3 &left_contact_normal, Vector3 &tang_disp, Vector3 &potential_force_field) {
        if (signed_sep_dist > 0) {
          // No contact, reset the tangential displacement
          tang_disp.set(0.0, 0.0, 0.0);
        } else {
          // Compute the relative normal and tangential velocities
          const auto rel_cp_vel = right_cp_vel - left_cp_vel;
          const auto rel_vel_normal = mundy::math::dot(rel_cp_vel, left_contact_normal) * left_contact_normal;
          const auto rel_vel_tang = rel_cp_vel - rel_vel_normal;

          // Compute the tangential displacement (history variable)
          // First add on the current tangential displacement, then project onto the tangent plane.
          tang_disp += rel_vel_tang * time_step_size;
          tang_disp -= mundy::math::dot(tang_disp, left_contact_normal) * left_contact_normal;
          const double tang_disp_mag = mundy::math::norm(tang_disp);

          // Compute the contact force
          // Note, for LAMMPS' delta is the negative of our signed separation distance.
          // As well, they compute the force on the RIGHT particle. We compute the force on the LEFT, introducing a
          // negative sign.
          const double left_mass = 4.0 / 3.0 * M_PI * left_radius * left_radius * left_radius * density;
          const double right_mass = 4.0 / 3.0 * M_PI * right_radius * right_radius * right_radius * density;
          [[maybe_unused]] const double effective_radius = (left_radius * right_radius) / (left_radius + right_radius);
          const double effective_mass = (left_mass * right_mass) / (left_mass + right_mass);
          // const double hertz_poly = std::sqrt(-effective_radius * signed_sep_dist);
          const double hertz_poly = 1.0;  // Hookean contact model
          auto normal_force = hertz_poly * (normal_spring_coeff * signed_sep_dist * left_contact_normal +
                                            effective_mass * normal_damping_coeff * rel_vel_normal);
          auto tang_force =
              hertz_poly * (tang_spring_coeff * tang_disp + effective_mass * tang_damping_coeff * rel_vel_tang);

          // Rescale frictional displacements and forces if needed to satisfy the Coulomb friction law
          // Ft = min(friction_coeff*Fn, Ft)
          const double normal_force_mag = mundy::math::norm(normal_force);
          const double tang_force_mag = mundy::math::norm(tang_force);
          const double scaled_normal_force_mag = friction_coeff * normal_force_mag;
          if (tang_force_mag > scaled_normal_force_mag) {
            if (tang_disp_mag != 0.0) {  // TODO(palmerb4): Exact comparison to 0.0 is bad. Use a tol.
              tang_disp = (scaled_normal_force_mag / tang_force_mag) *
                              (tang_disp + effective_mass * tang_damping_coeff * rel_vel_tang / tang_spring_coeff) -
                          effective_mass * tang_damping_coeff * rel_vel_tang / tang_spring_coeff;
              tang_force *= scaled_normal_force_mag / tang_force_mag;
            } else {
              tang_force.set(0.0, 0.0, 0.0);
            }
          }

          // Save the contact force (Forces are equal and opposite, so we only save the left force)
          potential_force_field += normal_force + tang_force;
        }
      };  // eval_hertz_with_friction

  // Simulation params
  const double time_step_size = 1.0e-4;
  const double radius = 1.0;
  const double density = 1.0;
  const double friction_coeff = 0.2;
  const double g = 9.81;
  const double initial_vel = 2.0;
  const double mass = 4.0 / 3.0 * M_PI * radius * radius * radius * density;
  const double inertia = 2.0 / 5.0 * mass * radius * radius;

  // Hertzian contact params
  const double youngs_modulus = 10000.0;
  const double poissons_ratio = 0.3;
  const double shear_modulus = 0.5 * youngs_modulus / (1.0 + poissons_ratio);
  const double normal_spring_coeff = 4.0 / 3.0 * shear_modulus / (1.0 - poissons_ratio);
  const double tang_spring_coeff = 4.0 * shear_modulus / (2.0 - poissons_ratio);
  const double normal_damping_coeff = 1.0;
  const double tang_damping_coeff = 0.5;

  // Initialize the sphere
  mundy::math::Vector3d position(0.0, radius, 0.0);
  mundy::math::Vector3d orientation(0.0, 0.0, 0.0);
  mundy::math::Vector3d velocity(initial_vel, 0.0, 0.0);
  mundy::math::Vector3d omega(0.0, 0.0, 0.0);
  mundy::math::Vector3d acceleration(0.0, 0.0, 0.0);
  mundy::math::Vector3d alpha(0.0, 0.0, 0.0);
  mundy::math::Vector3d tang_disp(0.0, 0.0, 0.0);
  mundy::math::Vector3d potential_force_field(0.0, 0.0, 0.0);

  // Left is the sphere, right is the table, so the normal points down and the right velocity is zero.
  const auto left_contact_normal = mundy::math::Vector3d(0.0, -1.0, 0.0);
  const auto right_cp_vel = mundy::math::Vector3d(0.0, 0.0, 0.0);
  double signed_separation_distance = position[1] - radius;

  // Save the x-velocity of the sphere, the signed separation distance, and the time
  const double max_time = 2.0;
  const int num_steps = static_cast<int>(max_time / time_step_size);
  std::vector<double> x_velocities(num_steps);
  std::vector<double> signed_separation_distances(num_steps);
  std::vector<double> times(num_steps);

  double time = 0.0;
  for (int i = 0; i < num_steps; i++) {
    // Print the result for time t^k every 0.1 seconds
    // if (std::fmod(time, 0.01) < time_step_size) {
    //   std::cout << "Time: " << time << std::endl;
    //   std::cout << "  Position: " << position << "\t Orientation: " << orientation << std::endl;
    //   std::cout << "  Velocity: " << velocity << "\t Omega: " << omega << std::endl;
    //   std::cout << "  Acceleration: " << acceleration << "\t Alpha: " << alpha << std::endl;
    //   std::cout << "  Potential Force: " << potential_force_field << "\t Tangential Displacement: " << tang_disp
    //             << std::endl;
    //   std::cout << "  signed_separation_distance: " << signed_separation_distance << std::endl;
    // }

    // Save the x-velocity, signed separation distance, and time for time t^k
    x_velocities[i] = velocity[0];
    signed_separation_distances[i] = signed_separation_distance;
    times[i] = time;

    // Update the time
    time += time_step_size;

    // Get the new position and orientation
    //   x(t + dt) = x(t) + v(t) dt + 0.5 a(t) dt^2
    //   orientation(t + dt) = orientation(t) + omega(t) dt + 0.5 alpha(t) dt^2
    position += velocity * time_step_size + 0.5 * acceleration * time_step_size * time_step_size;
    orientation += omega * time_step_size + 0.5 * alpha * time_step_size * time_step_size;

    // Setup the current timestep
    signed_separation_distance = position[1] - radius;
    potential_force_field.set(0.0, 0.0, 0.0);

    // We are at time t^{k+1}. Compute F^{k+1} and tau^{k+1}
    mundy::math::Vector3d left_cp_vel = velocity + mundy::math::cross(omega, radius * left_contact_normal);
    eval_hertz_with_friction(time_step_size, density, normal_spring_coeff, tang_spring_coeff, normal_damping_coeff,
                             tang_damping_coeff, friction_coeff, radius, radius, signed_separation_distance,
                             left_cp_vel, right_cp_vel, left_contact_normal, tang_disp, potential_force_field);

    // Update the velocity and angular velocity
    //  v(t + dt) = v(t) + 0.5 (a(t) + a(t + dt)) dt
    //  omega(t + dt) = omega(t) + 0.5 (alpha(t) + alpha(t + dt)) dt
    //
    // Use the forces and torques to get a(t + dt) and alpha(t + dt) but stored in temporary variables to avoid
    // overwriting a(t) and alpha(t)
    auto acceleration_next = potential_force_field / mass - mundy::math::Vector3d(0.0, g, 0.0);
    auto alpha_next = mundy::math::cross(radius * left_contact_normal, potential_force_field) / inertia;
    velocity += 0.5 * (acceleration + acceleration_next) * time_step_size;
    omega += 0.5 * (alpha + alpha_next) * time_step_size;
    acceleration = acceleration_next;
    alpha = alpha_next;
  }

  // Write the results to a file. Use a buffer to speed this up.
  std::ostringstream buffer;
  buffer << "Time,X-Velocity,Signed_Separation_Distance" << std::endl;
  for (int i = 0; i < num_steps; i++) {
    buffer << times[i] << "," << x_velocities[i] << "," << signed_separation_distances[i] << std::endl;
  }
  std::ofstream file("frictional_hertzian_contact_slide_slip_sphere.csv");
  file << buffer.str();
  file.close();
  file.close();
}

//@}

}  // namespace

}  // namespace linkers

}  // namespace mundy
