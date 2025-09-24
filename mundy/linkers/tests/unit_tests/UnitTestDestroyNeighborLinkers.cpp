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
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core libs
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/BulkData.hpp>       // for stk::mesh::BulkData
#include <stk_mesh/base/Comm.hpp>           // for comm_mesh_counts
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/FieldParallel.hpp>  // for stk:::mesh::communicate_field_data
#include <stk_mesh/base/GetEntities.hpp>    // for stk::mesh::get_selected_entities
#include <stk_mesh/base/MeshBuilder.hpp>    // for stk::mesh::MeshBuilder
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_mesh/base/Types.hpp>          // for stk::mesh::EntityProc, EntityVector, etc

// Mundy libs
#include <MundyLinkers_config.hpp>                    // for HAVE_MUNDYLINKERS_MUNDYSHAPES
#include <mundy_core/MakeStringArray.hpp>             // for mundy::core::make_string_array
#include <mundy_linkers/DestroyNeighborLinkers.hpp>   // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/GenerateNeighborLinkers.hpp>  // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/Linkers.hpp>   // for mundy::linkers::Linker and  mundy::linkers::declare_family_tree_relation
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>    // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>  // for mundy::meta::MetaMethodFactory and mundy::meta::HasMeshReqsAndIsRegisterable
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

namespace mundy {

namespace linkers {

namespace {

//! \name GenerateNeighborLinkers functionality unit tests
//@{

#ifdef HAVE_MUNDYLINKERS_MUNDYSHAPES

bool aabbs_overlap(const double *aabb_i, const double *aabb_j) {
  // Check overlap in the x dimension
  bool overlap_in_x = aabb_i[3] >= aabb_j[0] && aabb_j[3] >= aabb_i[0];

  // Check overlap in the y dimension
  bool overlap_in_y = aabb_i[4] >= aabb_j[1] && aabb_j[4] >= aabb_i[1];

  // Check overlap in the z dimension
  bool overlap_in_z = aabb_i[5] >= aabb_j[2] && aabb_j[5] >= aabb_i[2];

  // If there is overlap in all three dimensions, the AABBs overlap
  return overlap_in_x && overlap_in_y && overlap_in_z;
}

void direct_compare_neighbor_linkers(const std::shared_ptr<mundy::mesh::BulkData> &bulk_data_ptr,
                                     const std::vector<stk::mesh::Entity> &elements_to_check,
                                     const size_t num_elements_per_process, const int num_nodes_per_element,
                                     const stk::mesh::Part *elements_part_ptr,
                                     const stk::mesh::Part *neighbor_linkers_part_ptr,
                                     const stk::mesh::Part *element_elements_linkers_part_ptr,
                                     const stk::mesh::Field<double> *element_aabb_field_ptr,
                                     const LinkedEntitiesFieldType *linked_entities_field_ptr) {
  // Get the total number of elements and linkers. Must be called parallel synchronously.
  std::vector<size_t> entity_counts;
  stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
  const size_t total_num_elements = entity_counts[stk::topology::ELEMENT_RANK];
  const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
  ASSERT_EQ(total_num_elements, bulk_data_ptr->parallel_size() * num_elements_per_process);

  // Ghost all particles and linkers to process rank 0.
  bulk_data_ptr->modification_begin();
  std::vector<stk::mesh::EntityProc> send_all_entities;
  if (bulk_data_ptr->parallel_rank() != 0) {
    for (size_t i = 0; i < num_elements_per_process; i++) {
      stk::mesh::Entity element_i = elements_to_check[i];
      send_all_entities.push_back(std::make_pair(element_i, 0));
    }

    const stk::mesh::BucketVector &neighbor_linker_buckets = bulk_data_ptr->get_buckets(
        stk::topology::CONSTRAINT_RANK, stk::mesh::Selector(*neighbor_linkers_part_ptr) &
                                            stk::mesh::Selector(bulk_data_ptr->mesh_meta_data().locally_owned_part()));
    for (size_t bucket_idx = 0; bucket_idx < neighbor_linker_buckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &neighbor_linker_bucket = *neighbor_linker_buckets[bucket_idx];
      for (size_t neighbor_linker_idx = 0; neighbor_linker_idx < neighbor_linker_bucket.size(); ++neighbor_linker_idx) {
        stk::mesh::Entity const &neighbor_linker = neighbor_linker_bucket[neighbor_linker_idx];
        send_all_entities.push_back(std::make_pair(neighbor_linker, 0));
      }
    }
  }
  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting("GHOST_ALL_TO_RANK0");
  bulk_data_ptr->change_ghosting(ghosting, send_all_entities);
  bulk_data_ptr->modification_end();

  // Communicate the AABB field to process 0. Must be called parallel synchronously.
  stk::mesh::communicate_field_data(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()),
                                    std::vector<const stk::mesh::FieldBase *>{element_aabb_field_ptr});

  // Perform the direct N^2 neighbor comparison on process 0.
  if (bulk_data_ptr->parallel_rank() == 0) {
    // Construct the adjacency matrix for each element in the element part.
    // NOTE: GIDs are 1-indexed and the aabb field needs communicated to the ghosted elements.
    std::vector<std::vector<int>> adjacency_matrix(total_num_elements, std::vector<int>(total_num_elements, false));
    const stk::mesh::BucketVector &element_buckets =
        bulk_data_ptr->get_buckets(stk::topology::ELEMENT_RANK, *elements_part_ptr);
    size_t local_num_elements = 0;
    for (size_t bucket_idx_i = 0; bucket_idx_i < element_buckets.size(); ++bucket_idx_i) {
      stk::mesh::Bucket &element_bucket_i = *element_buckets[bucket_idx_i];
      for (size_t element_idx_i = 0; element_idx_i < element_bucket_i.size(); ++element_idx_i) {
        stk::mesh::Entity const &element_i = element_bucket_i[element_idx_i];
        const stk::mesh::EntityId element_i_gid = bulk_data_ptr->identifier(element_i);
        EXPECT_LT(element_i_gid - 1, total_num_elements);
        local_num_elements++;

        double *aabb_i = stk::mesh::field_data(*element_aabb_field_ptr, element_i);
        for (size_t bucket_idx_j = 0; bucket_idx_j < element_buckets.size(); ++bucket_idx_j) {
          stk::mesh::Bucket &element_bucket_j = *element_buckets[bucket_idx_j];
          for (size_t element_idx_j = 0; element_idx_j < element_bucket_j.size(); ++element_idx_j) {
            stk::mesh::Entity const &element_j = element_bucket_j[element_idx_j];
            const stk::mesh::EntityId element_j_gid = bulk_data_ptr->identifier(element_j);
            EXPECT_LT(element_j_gid - 1, total_num_elements);

            double *aabb_j = stk::mesh::field_data(*element_aabb_field_ptr, element_j);

            // Check if the AABBs overlap so long as the elements are not the same.
            adjacency_matrix[element_i_gid - 1][element_j_gid - 1] =
                aabbs_overlap(aabb_i, aabb_j) * (element_i_gid != element_j_gid);
          }
        }
      }
    }
    EXPECT_EQ(local_num_elements, total_num_elements) << "The ghosting should have brought all elements to process 0.";

    // Half the sum of the adjacency matrix should be equal to the number of neighbor linkers.
    size_t expected_num_neighbor_linkers = std::accumulate(adjacency_matrix.begin(), adjacency_matrix.end(), 0,
                                                           [](int acc, const std::vector<int> &vec) {
                                                             return acc + std::accumulate(vec.begin(), vec.end(), 0);
                                                           }) /
                                           2;
    EXPECT_EQ(expected_num_neighbor_linkers, total_num_linkers);

    // Loop over each neighbor linker and check if its connected elements are neighbors.
    size_t local_num_linkers = 0;
    const stk::mesh::BucketVector &neighbor_linker_buckets =
        bulk_data_ptr->get_buckets(stk::topology::CONSTRAINT_RANK, *neighbor_linkers_part_ptr);
    for (size_t bucket_idx = 0; bucket_idx < neighbor_linker_buckets.size(); ++bucket_idx) {
      stk::mesh::Bucket &neighbor_linker_bucket = *neighbor_linker_buckets[bucket_idx];
      for (size_t neighbor_linker_idx = 0; neighbor_linker_idx < neighbor_linker_bucket.size(); ++neighbor_linker_idx) {
        stk::mesh::Entity const &neighbor_linker = neighbor_linker_bucket[neighbor_linker_idx];

        // Get the linked entities from the neighbor linker.
        stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(*linked_entities_field_ptr, neighbor_linker));
        stk::mesh::Entity left_element = bulk_data_ptr->get_entity(key_t_ptr[0]);
        stk::mesh::Entity right_element = bulk_data_ptr->get_entity(key_t_ptr[1]);

        EXPECT_TRUE(bulk_data_ptr->bucket(neighbor_linker).member(*element_elements_linkers_part_ptr))
            << "Neighbor linkers should have the element-element specialization.";
        EXPECT_TRUE(bulk_data_ptr->is_valid(left_element) && bulk_data_ptr->is_valid(right_element))
            << "Neighbor linkers should connect valid elements.";
        EXPECT_EQ(bulk_data_ptr->num_nodes(neighbor_linker), 2 * num_nodes_per_element)
            << "Neighbor linkers should connect exactly " << 2 * num_nodes_per_element << " nodes.";

        const stk::mesh::EntityId element_i_gid = bulk_data_ptr->identifier(left_element);
        const stk::mesh::EntityId element_j_gid = bulk_data_ptr->identifier(right_element);
        EXPECT_TRUE(adjacency_matrix[element_i_gid - 1][element_j_gid - 1])
            << "Neighbor linkers should connect neighboring elements.";

        EXPECT_TRUE(element_i_gid != element_j_gid) << "Neighbor linkers should not connect a element to itself.";

        local_num_linkers++;
      }
    }
    EXPECT_EQ(expected_num_neighbor_linkers, local_num_linkers);
  }

  // Destroy the ghosting.
  bulk_data_ptr->modification_begin();
  bulk_data_ptr->destroy_ghosting(ghosting);
  bulk_data_ptr->modification_end();

  // Wait for all processes to finish before continuing.
  stk::parallel_machine_barrier(bulk_data_ptr->parallel());
}

TEST(GenerateNeighborLinkers, PerformsNeighborLinkerDestructionCorrectlyForSpheresSimple) {
  /* Outline
    For this test, we generate a 1 sphere per process of equal radii along a line with overlap, compute their AABBs, and
    connect them with linkers. For example, for 3 processes, we would have 3 spheres along the x-axis. Process 0
    would own sphere 0, process 1 would own sphere 1, and process 2 would own sphere 2. Process 0 would generate a
    linker between spheres 0 and 1 and process 1 would generate a linker between spheres 1 and 2.

    We will run the destroy distant neighbor linkers routine on the linkers and validate that none are deleted since
    they all overlap. We'll then move the sphere on process 0 far from the others and rerun the routine. We'll validate
    that the linker between spheres 0 and 1 is deleted.
  */

  // Only valid for multiple processes
  if (stk::parallel_machine_size(MPI_COMM_WORLD) == 1) {
    GTEST_SKIP() << "This test is only valid for multiple processes.";
    return;
  }

  // Free variables
  const double overlap = 0.1;

  // Create an instance of DestroyNeighborLinkers based on committed mesh that meets both of their
  // default requirements.
  Teuchos::ParameterList generate_neighbor_linkers_fixed_params = Teuchos::ParameterList();

  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
  destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
      .sublist("DESTROY_DISTANT_NEIGHBORS")
      .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
      .set("valid_connected_source_and_target_part_names", mundy::core::make_string_array("SPHERES"))
      .set("linker_destroy_flag_field_name", "LINKER_DESTROY_FLAG")
      .set("element_aabb_field_name", "ELEMENT_AABB");

  auto [generate_neighbor_linkers_ptr, destroy_neighbor_linkers_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<GenerateNeighborLinkers,
                                                                                        DestroyNeighborLinkers>(
          std::array{generate_neighbor_linkers_fixed_params, destroy_neighbor_linkers_fixed_params});
  ASSERT_TRUE(bulk_data_ptr != nullptr) << "bulk_data_ptr cannot be null";
  ASSERT_TRUE(generate_neighbor_linkers_ptr != nullptr) << "generate_neighbor_linkers_ptr cannot be null";
  ASSERT_TRUE(destroy_neighbor_linkers_ptr != nullptr) << "destroy_neighbor_linkers_ptr cannot be null";
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();
  ASSERT_TRUE(meta_data_ptr != nullptr) << "meta_data_ptr cannot be null";

  // Fetch the required fields.
  stk::mesh::Field<double> *element_aabb_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_AABB");
  ASSERT_TRUE(element_aabb_field_ptr != nullptr) << "element_aabb_field_ptr cannot be null";

  // Fetch the requested parts.
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  stk::mesh::Part *neighbor_linkers_part_ptr = meta_data_ptr->get_part("NEIGHBOR_LINKERS");
  ASSERT_TRUE(spheres_part_ptr != nullptr) << "spheres_part_ptr cannot be null";
  ASSERT_TRUE(neighbor_linkers_part_ptr != nullptr) << "neighbor_linkers_part_ptr cannot be null";

  // Add 1 sphere and 1 node to the mesh per process using the process rank + 1 as the sphere and node's ID.
  bulk_data_ptr->modification_begin();
  const int process_rank = bulk_data_ptr->parallel_rank();
  stk::mesh::Entity node = bulk_data_ptr->declare_node(process_rank + 1);
  stk::mesh::Entity sphere =
      bulk_data_ptr->declare_element(process_rank + 1, stk::mesh::ConstPartVector{spheres_part_ptr});
  bulk_data_ptr->declare_relation(sphere, node, 0);
  bulk_data_ptr->modification_end();

  // Set the sphere's AABB
  {
    double *element_aabb = stk::mesh::field_data(*element_aabb_field_ptr, sphere);
    element_aabb[0] = process_rank - 0.5 - 0.5 * overlap;
    element_aabb[1] = -1.0;
    element_aabb[2] = -1.0;
    element_aabb[3] = process_rank + 0.5 + 0.5 * overlap;
    element_aabb[4] = 1.0;
    element_aabb[5] = 1.0;
  }

  // At this point we have 1 sphere per process. We will now generate the neighbor linkers between the spheres.
  generate_neighbor_linkers_ptr->execute(*spheres_part_ptr, *spheres_part_ptr);

  // Get the total number of spheres and linkers. Must be called parallel synchronously.
  {
    std::vector<size_t> entity_counts;
    stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
    const size_t total_num_spheres = entity_counts[stk::topology::ELEMENT_RANK];
    const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
    ASSERT_EQ(total_num_spheres, bulk_data_ptr->parallel_size());
    EXPECT_EQ(total_num_linkers, bulk_data_ptr->parallel_size() - 1);
  }

  // Attempt to destroy the distant neighbor linkers. None should be destroyed since they all overlap.
  destroy_neighbor_linkers_ptr->execute(*neighbor_linkers_part_ptr);
  {
    std::vector<size_t> entity_counts;
    stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
    const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
    EXPECT_EQ(total_num_linkers, bulk_data_ptr->parallel_size() - 1) << "No linkers should have been destroyed.";
  }

  // Move the sphere on process 0 far from the others and rerun the routine. The linker between spheres 0 and 1 should
  // be deleted.
  if (process_rank == 0) {
    double *element_aabb = stk::mesh::field_data(*element_aabb_field_ptr, sphere);
    element_aabb[0] = -100.0;
    element_aabb[1] = -1.0;
    element_aabb[2] = -1.0;
    element_aabb[3] = -99.0;
    element_aabb[4] = 1.0;
    element_aabb[5] = 1.0;
  }

  destroy_neighbor_linkers_ptr->execute(*neighbor_linkers_part_ptr);
  {
    std::vector<size_t> entity_counts;
    stk::mesh::comm_mesh_counts(*bulk_data_ptr, entity_counts);
    const size_t total_num_linkers = entity_counts[stk::topology::CONSTRAINT_RANK];
    EXPECT_EQ(total_num_linkers, bulk_data_ptr->parallel_size() - 2) << "One linker should have been destroyed.";

    // As a sanity check, loop over all linkers and validate that they are valid.
    mundy::mesh::for_each_entity_run(*static_cast<stk::mesh::BulkData *>(bulk_data_ptr.get()),
                                     stk::topology::CONSTRAINT_RANK, *neighbor_linkers_part_ptr,
                                     []([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                        const stk::mesh::Entity &linker) { EXPECT_TRUE(bulk_data.is_valid(linker)); });
  }
}

// TEST(DestroyNeighborLinkers, RepeatedNeighborLinkerGenerationAndDestructionForSpheres) {
//   /* Outline
//     For this test, we generate a N spheres of equal radii isotropically distributed within a box and split across all
//     processes, compute their AABBs, and generate their neighbor linkers. We then communicate all spheres to process 0
//     and use a direct N^2 neighbor search to create a matrix of booleans (adjacency list) stating if particle (i,j)
//     are neighbors or not. We use this matrix to validate the neighbor linkers. The sum of the matrix should be equal
//     to the number of neighbor linkers and, for each linker connecting particles i and j, the matrix should be true at
//     (i,j) and (j,i).

//     We let the volume fraction of spheres be large enough to guarantee that some neighbor linkers span multiple
//     processes.
//   */

//   // Only valid for multiple processes
//   if (stk::parallel_machine_size(MPI_COMM_WORLD) == 1) {
//     GTEST_SKIP() << "This test is only valid for multiple processes.";
//     return;
//   }

//   // Free variables
//   const int num_spheres_per_process = 10;
//   const double volume_fraction = 0.4;
//   const double sphere_radius = 1.0;
//   const double sphere_volume = (4.0 / 3.0) * M_PI * std::pow(sphere_radius, 3);
//   const double length_of_domain = std::cbrt(num_spheres_per_process * sphere_volume / volume_fraction);

//   // Create an instance of GenerateNeighborLinkers and ComputeAABB based on committed mesh that meets both of their
//   // default requirements.
//   Teuchos::ParameterList compute_aabb_fixed_params = Teuchos::ParameterList();  // Use default parameters.

//   Teuchos::ParameterList neighbor_linkers_fixed_params = Teuchos::ParameterList();

//   Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
//   destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
//       .sublist("DESTROY_DISTANT_NEIGHBORS")
//       .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
//       .set("valid_connected_source_and_target_part_names", mundy::core::make_string_array("SPHERES"))
//       .set("linker_destroy_flag_field_name", "LINKER_DESTROY_FLAG")
//       .set("element_aabb_field_name", "ELEMENT_AABB");

//   neighbor_linkers_fixed_params.set("specialized_neighbor_linkers_part_names",
//                                     mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"));
//   auto [compute_aabb_ptr, generate_neighbor_linkers_ptr, destroy_neighbor_linkers_ptr, bulk_data_ptr] =
//       mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
//           mundy::shapes::ComputeAABB, GenerateNeighborLinkers, DestroyNeighborLinkers>(
//           std::array{compute_aabb_fixed_params, neighbor_linkers_fixed_params,
//           destroy_neighbor_linkers_fixed_params});

//   ASSERT_TRUE(bulk_data_ptr != nullptr) << "bulk_data_ptr cannot be null";
//   ASSERT_TRUE(compute_aabb_ptr != nullptr) << "compute_aabb_ptr cannot be null";
//   ASSERT_TRUE(generate_neighbor_linkers_ptr != nullptr) << "generate_neighbor_linkers_ptr cannot be null";
//   ASSERT_TRUE(destroy_neighbor_linkers_ptr != nullptr) << "destroy_neighbor_linkers_ptr cannot be null";
//   auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

//   // Fetch the required fields.
//   stk::mesh::Field<double> *node_coord_field_ptr =
//       meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
//   stk::mesh::Field<double> *element_radius_field_ptr =
//       meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
//   stk::mesh::Field<double> *element_aabb_field_ptr =
//       meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_AABB");
//   stk::mesh::Field<LinkedEntitiesFieldType::value_type> *linked_entities_field_ptr =
//       meta_data_ptr->get_field<LinkedEntitiesFieldType::value_type>(stk::topology::CONSTRAINT_RANK,
//                                                                     "LINKED_NEIGHBOR_ENTITIES");
//   ASSERT_TRUE(node_coord_field_ptr != nullptr) << "node_coord_field_ptr cannot be null";
//   ASSERT_TRUE(element_aabb_field_ptr != nullptr) << "element_aabb_field_ptr cannot be null";
//   ASSERT_TRUE(element_radius_field_ptr != nullptr) << "element_radius_field_ptr cannot be null";
//   ASSERT_TRUE(linked_entities_field_ptr != nullptr) << "linked_entities_field_ptr cannot be null";

//   // Fetch the requested parts.
//   stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
//   stk::mesh::Part *neighbor_linkers_part_ptr = meta_data_ptr->get_part("NEIGHBOR_LINKERS");
//   stk::mesh::Part *sphere_spheres_linkers_part_ptr = meta_data_ptr->get_part("SPHERE_SPHERE_LINKERS");
//   ASSERT_TRUE(spheres_part_ptr != nullptr) << "spheres_part_ptr cannot be null";
//   ASSERT_TRUE(neighbor_linkers_part_ptr != nullptr) << "neighbor_linkers_part_ptr cannot be null";
//   ASSERT_TRUE(sphere_spheres_linkers_part_ptr != nullptr) << "sphere_spheres_linkers_part_ptr cannot be null";

//   // Add n spheres to the mesh per process.
//   bulk_data_ptr->modification_begin();
//   std::vector<size_t> requests(meta_data_ptr->entity_rank_count(), 0);
//   requests[stk::topology::NODE_RANK] = num_spheres_per_process;
//   requests[stk::topology::ELEMENT_RANK] = num_spheres_per_process;

//   std::vector<stk::mesh::Entity> requested_entities;
//   bulk_data_ptr->generate_new_entities(requests, requested_entities);

//   // Associate each segments with the sphere part and connect them to their nodes.
//   std::vector<stk::mesh::Part *> add_spheres_part = {spheres_part_ptr};
//   for (int i = 0; i < num_spheres_per_process; i++) {
//     stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
//     stk::mesh::Entity node_i = requested_entities[i];
//     bulk_data_ptr->change_entity_parts(sphere_i, add_spheres_part);
//     bulk_data_ptr->declare_relation(sphere_i, node_i, 0);
//   }
//   bulk_data_ptr->modification_end();

//   // Perform repeated creation and deletion cycles
//   const int num_cycles = 10;
//   const int log_interval = 1;
//   Kokkos::Timer timer;
//   for (int c = 0; c < num_cycles; c++) {
//     if (bulk_data_ptr->parallel_rank() == 0 && c % log_interval == 0) {
//       std::cout << "Cycle " << c << " of " << num_cycles << std::endl;
//       std::cout << "Running at " << log_interval/ timer.seconds() << " cycles per second." << std::endl;
//     }
//     // Set the sphere's position and radius
//     openrand::Philox rng(bulk_data_ptr->parallel_rank(), 0);
//     for (int i = 0; i < num_spheres_per_process; i++) {
//       stk::mesh::Entity node_i = requested_entities[i];
//       stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
//       double *node_coords = stk::mesh::field_data(*node_coord_field_ptr, node_i);
//       double *element_radius = stk::mesh::field_data(*element_radius_field_ptr, sphere_i);

//       node_coords[0] = length_of_domain * rng.rand<double>();
//       node_coords[1] = length_of_domain * rng.rand<double>();
//       node_coords[2] = length_of_domain * rng.rand<double>();

//       element_radius[0] = sphere_radius;
//     }

//     // Compute the AABB for all the spheres. By default this writes to the ELEMENT_AABB field.
//     compute_aabb_ptr->execute(*spheres_part_ptr);

//     // Destroy neighbor linkers that connect spheres that are too far apart.
//     destroy_neighbor_linkers_ptr->execute(*neighbor_linkers_part_ptr);

//     // Compute the neighbor linkers. Between neighboring spheres.
//     generate_neighbor_linkers_ptr->execute(*spheres_part_ptr, *spheres_part_ptr);

//     // Run the direct neighbor comparison on process 0.
//     const int num_nodes_per_sphere = 1;
//     std::vector<stk::mesh::Entity> elements_to_check;
//     std::copy(requested_entities.begin() + num_spheres_per_process, requested_entities.end(),
//               std::back_inserter(elements_to_check));
//     direct_compare_neighbor_linkers(bulk_data_ptr, elements_to_check, num_spheres_per_process, num_nodes_per_sphere,
//                                     spheres_part_ptr, neighbor_linkers_part_ptr, sphere_spheres_linkers_part_ptr,
//                                     element_aabb_field_ptr, linked_entities_field_ptr);
//   }
// }

TEST(DestroyNeighborLinkers, RepeatedNeighborLinkerGenerationAndDestructionForSpheres) {
  /* Outline
    For this test, we generate a N spheres of equal radii isotropically distributed within a box and split across all
    processes, compute their AABBs, and generate their neighbor linkers. We then communicate all spheres to process 0
    and use a direct N^2 neighbor search to create a matrix of booleans (adjacency list) stating if particle (i,j) are
    neighbors or not. We use this matrix to validate the neighbor linkers. The sum of the matrix should be equal to the
    number of neighbor linkers and, for each linker connecting particles i and j, the matrix should be true at (i,j) and
    (j,i).

    We let the volume fraction of spheres be large enough to guarantee that some neighbor linkers span multiple
    processes.
  */

  // Only valid for multiple processes
  if (stk::parallel_machine_size(MPI_COMM_WORLD) == 1) {
    GTEST_SKIP() << "This test is only valid for multiple processes.";
    return;
  }

  // Free variables
  const int num_spheres_per_process = 100;
  const double volume_fraction = 0.4;
  const double sphere_radius = 1.0;
  const double sphere_volume = (4.0 / 3.0) * M_PI * std::pow(sphere_radius, 3);
  const double length_of_domain = std::cbrt(num_spheres_per_process * sphere_volume / volume_fraction);

  // Create an instance of GenerateNeighborLinkers and ComputeAABB based on committed mesh that meets both of their
  // default requirements.
  Teuchos::ParameterList compute_aabb_fixed_params = Teuchos::ParameterList();  // Use default parameters.

  Teuchos::ParameterList neighbor_linkers_fixed_params = Teuchos::ParameterList();

  Teuchos::ParameterList destroy_neighbor_linkers_fixed_params = Teuchos::ParameterList();
  destroy_neighbor_linkers_fixed_params.set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS")
      .sublist("DESTROY_DISTANT_NEIGHBORS")
      .set("valid_entity_part_names", mundy::core::make_string_array("NEIGHBOR_LINKERS"))
      .set("valid_connected_source_and_target_part_names", mundy::core::make_string_array("SPHERES"))
      .set("linker_destroy_flag_field_name", "LINKER_DESTROY_FLAG")
      .set("element_aabb_field_name", "ELEMENT_AABB");

  neighbor_linkers_fixed_params.set("specialized_neighbor_linkers_part_names",
                                    mundy::core::make_string_array("SPHERE_SPHERE_LINKERS"));
  auto [compute_aabb_ptr, generate_neighbor_linkers_ptr, destroy_neighbor_linkers_ptr, bulk_data_ptr] =
      mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements<
          mundy::shapes::ComputeAABB, GenerateNeighborLinkers, DestroyNeighborLinkers>(
          std::array{compute_aabb_fixed_params, neighbor_linkers_fixed_params, destroy_neighbor_linkers_fixed_params});

  ASSERT_TRUE(bulk_data_ptr != nullptr) << "bulk_data_ptr cannot be null";
  ASSERT_TRUE(compute_aabb_ptr != nullptr) << "compute_aabb_ptr cannot be null";
  ASSERT_TRUE(generate_neighbor_linkers_ptr != nullptr) << "generate_neighbor_linkers_ptr cannot be null";
  ASSERT_TRUE(destroy_neighbor_linkers_ptr != nullptr) << "destroy_neighbor_linkers_ptr cannot be null";
  auto meta_data_ptr = bulk_data_ptr->mesh_meta_data_ptr();

  // Fetch the required fields.
  stk::mesh::Field<double> *node_coord_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
  stk::mesh::Field<double> *element_radius_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_RADIUS");
  stk::mesh::Field<double> *element_aabb_field_ptr =
      meta_data_ptr->get_field<double>(stk::topology::ELEMENT_RANK, "ELEMENT_AABB");
  stk::mesh::Field<LinkedEntitiesFieldType::value_type> *linked_entities_field_ptr =
      meta_data_ptr->get_field<LinkedEntitiesFieldType::value_type>(stk::topology::CONSTRAINT_RANK,
                                                                    "LINKED_NEIGHBOR_ENTITIES");
  ASSERT_TRUE(node_coord_field_ptr != nullptr) << "node_coord_field_ptr cannot be null";
  ASSERT_TRUE(element_aabb_field_ptr != nullptr) << "element_aabb_field_ptr cannot be null";
  ASSERT_TRUE(element_radius_field_ptr != nullptr) << "element_radius_field_ptr cannot be null";
  ASSERT_TRUE(linked_entities_field_ptr != nullptr) << "linked_entities_field_ptr cannot be null";

  // Fetch the requested parts.
  stk::mesh::Part *spheres_part_ptr = meta_data_ptr->get_part("SPHERES");
  stk::mesh::Part *neighbor_linkers_part_ptr = meta_data_ptr->get_part("NEIGHBOR_LINKERS");
  stk::mesh::Part *sphere_spheres_linkers_part_ptr = meta_data_ptr->get_part("SPHERE_SPHERE_LINKERS");
  ASSERT_TRUE(spheres_part_ptr != nullptr) << "spheres_part_ptr cannot be null";
  ASSERT_TRUE(neighbor_linkers_part_ptr != nullptr) << "neighbor_linkers_part_ptr cannot be null";
  ASSERT_TRUE(sphere_spheres_linkers_part_ptr != nullptr) << "sphere_spheres_linkers_part_ptr cannot be null";

  // Add n spheres to the mesh per process.
  bulk_data_ptr->modification_begin();
  std::vector<size_t> requests(meta_data_ptr->entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_spheres_per_process;
  requests[stk::topology::ELEMENT_RANK] = num_spheres_per_process;

  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr->generate_new_entities(requests, requested_entities);

  // Associate each segments with the sphere part and connect them to their nodes.
  std::vector<stk::mesh::Part *> add_spheres_part = {spheres_part_ptr};
  for (int i = 0; i < num_spheres_per_process; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
    stk::mesh::Entity node_i = requested_entities[i];
    bulk_data_ptr->change_entity_parts(sphere_i, add_spheres_part);
    bulk_data_ptr->declare_relation(sphere_i, node_i, 0);
  }
  bulk_data_ptr->modification_end();

  // Perform repeated creation and deletion cycles
  const int num_cycles = 10;
  const int log_interval = 1;
  Kokkos::Timer timer;
  for (int c = 0; c < num_cycles; c++) {
    if ((bulk_data_ptr->parallel_rank() == 0) && (c % log_interval == 0) && (c != 0)) {
      std::cout << "Cycle " << c << " of " << num_cycles << " Running at " << log_interval / timer.seconds()
                << " cycles per second." << std::endl;
      timer.reset();
    }
    // Set the sphere's position and radius
    openrand::Philox rng(bulk_data_ptr->parallel_rank(), 0);
    for (int i = 0; i < num_spheres_per_process; i++) {
      stk::mesh::Entity node_i = requested_entities[i];
      stk::mesh::Entity sphere_i = requested_entities[num_spheres_per_process + i];
      double *node_coords = stk::mesh::field_data(*node_coord_field_ptr, node_i);
      double *element_radius = stk::mesh::field_data(*element_radius_field_ptr, sphere_i);

      node_coords[0] = length_of_domain * rng.rand<double>();
      node_coords[1] = length_of_domain * rng.rand<double>();
      node_coords[2] = length_of_domain * rng.rand<double>();

      element_radius[0] = sphere_radius;
    }

    // Compute the AABB for all the spheres. By default this writes to the ELEMENT_AABB field.
    compute_aabb_ptr->execute(*spheres_part_ptr);

    // Destroy neighbor linkers that connect spheres that are too far apart.
    destroy_neighbor_linkers_ptr->execute(*neighbor_linkers_part_ptr);

    // Compute the neighbor linkers. Between neighboring spheres.
    generate_neighbor_linkers_ptr->execute(*spheres_part_ptr, *spheres_part_ptr);

    // Run the direct neighbor comparison on process 0.
    const int num_nodes_per_sphere = 1;
    std::vector<stk::mesh::Entity> elements_to_check;
    std::copy(requested_entities.begin() + num_spheres_per_process, requested_entities.end(),
              std::back_inserter(elements_to_check));
    direct_compare_neighbor_linkers(bulk_data_ptr, elements_to_check, num_spheres_per_process, num_nodes_per_sphere,
                                    spheres_part_ptr, neighbor_linkers_part_ptr, sphere_spheres_linkers_part_ptr,
                                    element_aabb_field_ptr, linked_entities_field_ptr);
  }
}
#endif  // HAVE_MUNDYLINKERS_MUNDYSHAPES

}  // namespace

}  // namespace linkers

}  // namespace mundy
