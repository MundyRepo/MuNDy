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

/// \file GenerateCollisionConstraints.cpp
/// \brief Definition of the GenerateCollisionConstraints class

// External
#include <fmt/format.h>  // for fmt::format

// C++ core libs
#include <algorithm>
#include <memory>     // for std::shared_ptr, std::unique_ptr
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector

// Mundy libs
#include <mundy_mesh/fmt_stk_types.hpp>                                     // adds fmt::format for stk types
#include <mundy_core/throw_assert.hpp>                             // for MUNDY_THROW_ASSERT
#include <mundy_constraints/GenerateCollisionConstraints.hpp>  // for mundy::constraint::GenerateCollisionConstraints
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>                          // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartReqs.hpp>                    // for mundy::meta::PartReqs

namespace mundy {

namespace constraints {

// \name Constructors and destructor
//{

GenerateCollisionConstraints::GenerateCollisionConstraints(mundy::mesh::BulkData *const bulk_data_ptr,
                                                           const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "GenerateCollisionConstraints: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_fixed_params.sublist("kernels", true);
  num_multibody_types_ = kernels_sublist.get<int>("count");
  multibody_part_ptr_vector_.reserve(num_multibody_types_);
  multibody_kernel_ptrs_.reserve(num_multibody_types_);
  for (size_t i = 0; i < num_multibody_types_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    const std::string kernel_name = kernel_params.get<std::string>("name");
    multibody_part_ptr_vector_.push_back(meta_data_ptr_->get_part(kernel_name));
    multibody_kernel_ptrs_.push_back(
        OurThreeWayKernelFactory::create_new_instance(kernel_name, bulk_data_ptr_, kernel_params));
  }
}
//}

// \name MetaFactory static interface implementation
//{

void GenerateCollisionConstraints::set_mutable_params([[maybe_unused]] const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  validate_mutable_parameters_and_set_defaults(&valid_mutable_params);

  // Parse the parameters
  Teuchos::ParameterList &kernels_sublist = valid_mutable_params.sublist("kernels", true);
  MUNDY_THROW_REQUIRE(num_multibody_types_ == kernels_sublist.get<int>("count"), std::invalid_argument,
                     "GenerateCollisionConstraints: Internal error. Mismatch between the stored kernel count "
                     "and the parameter list kernel count.\n"
                      "Odd... Please contact the development team.");
  for (size_t i = 0; i < num_multibody_types_; i++) {
    Teuchos::ParameterList &kernel_params = kernels_sublist.sublist("kernel_" + std::to_string(i));
    multibody_kernel_ptrs_[i]->set_mutable_params(kernel_params);
  }
}
//}

// \name Actions
//{

void GenerateCollisionConstraints::execute([[maybe_unused]] const stk::mesh::Selector &input_selector) {
  // Two words of word of warning:
  //   1. This method is programmed with care to avoid generating duplicative constraints. To do so, we only generate a
  //      collision constraint if the entity_key of the source particle is less than that of the target particle.
  //   2. Parts of this method require mesh modification and are, therefore, incompatible with the GPU. We do our best
  //   to isolate these sections.

  // If we make some basic assumptions then it's possible to break this process into a (CPU-based) constraint generation
  // routine and a (potentially GPU-based) constraint update routine.
  //
  // The core assumption is this: every pair of neighbors in the neighbor list will induce 1 collision constraint and 2
  // nodes. The nodes will connect to the dynamic connectivity (linkers) of the colliding pair. Under these assumptions
  // GenerateCollisionConstraints should
  //   0. Parse the parameter list and fetch our old and new neighbor list. We don't maintain the two lists, the user
  //   does. This makes clear that we only care about the difference between the two lists.
  //   1. Compare the neighbor list to our existing neighbor list to get the pairs_for_deletion and pairs_for_creation.
  //   2. Ghost the pairs_for_creation, their downward connectivity, AND their linkers.
  //   3. Call begin_modification()
  //   4. (On the CPU) Generate a constraint pool containing a new collision entity and two new nodes for each element
  //      of the neighbor list.
  //   5. (On the CPU) Generate a relation between the collision entity and its nodes as well as those nodes and the
  //   linker for the pair of particles. This step is independent of the multibody type associated with the elements.
  //   6. Call end_modification()
  //   7. (On the CPU or GPU) Loops over the generated collision constraints and call a a populate constraint kernel.
  // That kernel should
  //   0. Take in a collision constraint and its left and right bodies.
  //   1. Fetch the linker's connected nodes and any of its fields necessary to compute the contact locations, contact
  //   normal, and signed separation distance.

  // Some issues:
  // - Issue: If attributes are fetched via type then how are we given the neighbor list? We make strong assumptions
  //     about our neighbor list. For example, complementarity collision constraints want the neighbor detection to have
  //     a buffer distance but potential-based shape typically want the neighbor detection to be as tight as possible.
  //     This class can't be the one to generate the neighbors tho since other classes may wish to loop over the
  //     neighbors. Yeah, but will those shape want to loop over the neighbors with our specific buffer distance?
  //     That doesn't seem unreasonable.
  //   Solution: the neighbor list can be passed in as a mutable parameter since Teuchos::ParameterList can legit
  //     hold any variable type.
  // - Issue: Who gets linkers and will they only every store the surface connectivity? What if I wanted to connect a
  //     sphere to another sphere?
  //   Solution: For now, only elements will receive linkers to encode their
  //     dynamic surface connectivity. Once we build up a higher level of abstraction, we can break dynamic
  //     connectivity into subsets.
  // - Issue: If linkers are generated on the fly, then GenerateCollisionConstraints is one of the classes that should
  //     generate linkers. I really don't want to generate a linker for every edge, face, and element.
  //   Solution: Polytopes are either represented as a super-element with its own linker or as a collection of linked
  //     elements. Either way, we only consider element-to-element neighbor detection, and one linker per element.
  // - Issue: Setting up this kernel in such a way that the node positions and their fields can be updated in a way that
  //     satisfies sharing/ghosting is hard. If we loop over the collision constraints, then how do we choose the
  //     correct kernel. If we loop over the neighbors, how can we guarantee that the sphere, linker, and its nodes are
  //     ghosted or shared?
  //   Solution: In the current design, the process that generated the collision constraint should have access to the
  //     linkers and spheres but not the node of the ghosted sphere. We need to ghost the downward connectivity
  //     of our neighbors.
  // - Issue: Given two spheres, how do we fetch the collision constraint that links them?
  //   Solution: Once GenerateCollisionConstraints generates the collision constraints, it should store them with the
  //     neighbor list such that we can pass this kernel the constraint and the two spheres without needing to perform
  //     complicated lookups. This will require modifying mundy's data structors to better accommodate KWay kernels
  //     without code repetition (done).
  // - Issue: We currently assume that every pair of nearby particles generates a collision constraint, this is fine
  //     from a pool perspective but not fine when we consider that most nearby particles will already have collision
  //     constraints. As a result, we can't just preconnect the collision constraints to the linkers; we need to check
  //     if the two spheres share a collision constraint. The previous neighbor list will tell us this information! If
  //     we store the old neighbor list and take their set difference, then we can easily see the elements whose current
  //     collision constraints should be deleted and the elements which need collision constraints!!!!

  // What's the point in having this as a MetaMethodSubsetExecutionInterface? Well, the actual requirements are in how we populate the
  // constraints. Otherwise, we simply act on the neighbor list, not the parts.
  // I changed the name to Update collision constraints. This makes it more clear that the neighbor list is a mutable
  // parameter. Users may leave the lists the same and simply wish to repopulate the  existing collision constraints
  // using the given neighbor list. Users may even keep the neighbor list fixed for a couple execute calls and then
  // update it later on.

  // Find the set of neighbors that need collision constraints and ghost them and their connectivity.
  bulk_data_ptr_->modification_begin();
  auto pairs_to_generate = find_our_gained_neighbor_pairs(*old_neighbor_pairs_ptr_, *current_neighbor_pairs_ptr_);
  bool ghost_downward_connectivity = true;
  bool ghost_upward_connectivity = true;
  std::string name_of_ghosting = "geometric_ghosts";
  ghost_neighbors(bulk_data_ptr_, pairs_to_generate, name_of_ghosting, ghost_downward_connectivity,
                  ghost_upward_connectivity);
  bulk_data_ptr_->modification_end();

  // TODO(palmerb4): Perform some communication with the ghosts.

  bulk_data_ptr_->modification_begin();
  generate_empty_collision_constraints_between_pairs(bulk_data_ptr_, collision_part_ptr_, pairs_to_generate);
  bulk_data_ptr_->modification_end();

  // Populate the empty collision constraints.
  // The population procedure differs based on the multibody types of the connected bodies.
  // We should simply use UpdateConstraints to populate the constraints.
  // TODO(palmerb4): Write UpdateConstraints.
}
//}

// \name Internal helper functions
//{

IdentProcPairVector GenerateCollisionConstraints::find_our_gained_neighbor_pairs(
    const IdentProcPairVector &old_neighbor_pairs, const IdentProcPairVector &new_neighbor_pairs) {
  // Gained neighbors are those that are in the new neighbor list but not in the old list.
  IdentProcPairVector gained_neighbor_pairs;
  std::set_difference(std::begin(new_neighbor_pairs), std::end(new_neighbor_pairs), std::begin(old_neighbor_pairs),
                      std::end(old_neighbor_pairs), std::back_inserter(gained_neighbor_pairs));

  // TODO(palmerb4): Correct for invalid entities.

  return gained_neighbor_pairs;
}

std::vector<stk::mesh::Entity> GenerateCollisionConstraints::get_connected_lower_rank_entities(
    mundy::mesh::BulkData *const bulk_data_ptr, const stk::mesh::Entity &entity,
    const stk::topology::rank_t &entity_rank) {
  // For all ranks less than the current rank, fetch the connected entities of that rank and add them to the output.
  std::vector<stk::mesh::Entity> connected_lower_rank_entities;
  for (stk::topology::rank_t rank :
       {stk::topology::NODE_RANK, stk::topology::EDGE_RANK, stk::topology::FACE_RANK, stk::topology::ELEMENT_RANK}) {
    bool is_lower_rank = rank < entity_rank;
    if (is_lower_rank) {
      const unsigned num_conn_for_rank = bulk_data_ptr->num_connectivity(entity, rank);
      const stk::mesh::Entity *conn = bulk_data_ptr->begin(entity, rank);
      for (unsigned i = 0; i < num_conn_for_rank; ++i) {
        connected_lower_rank_entities.push_back(conn[i]);
      }
    }
  }
  return connected_lower_rank_entities;
}

std::vector<stk::mesh::Entity> GenerateCollisionConstraints::get_connected_higher_rank_entities(
    mundy::mesh::BulkData *const bulk_data_ptr, const stk::mesh::Entity &entity,
    const stk::topology::rank_t &entity_rank) {
  // For all ranks higher than the current rank, fetch the connected entities of that rank and add them to the output.
  std::vector<stk::mesh::Entity> connected_higher_rank_entities;
  for (stk::topology::rank_t rank : {stk::topology::EDGE_RANK, stk::topology::FACE_RANK, stk::topology::ELEMENT_RANK,
                                     stk::topology::CONSTRAINT_RANK}) {
    bool is_higher_rank = rank > entity_rank;
    if (is_higher_rank) {
      const unsigned num_conn_for_rank = bulk_data_ptr->num_connectivity(entity, rank);
      const stk::mesh::Entity *conn = bulk_data_ptr->begin(entity, rank);
      for (unsigned i = 0; i < num_conn_for_rank; ++i) {
        connected_higher_rank_entities.push_back(conn[i]);
      }
    }
  }
  return connected_higher_rank_entities;
}

stk::mesh::Ghosting &GenerateCollisionConstraints::ghost_neighbors(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                   const IdentProcPairVector &pairs_to_ghost,
                                                                   const std::string &name_of_ghosting,
                                                                   bool ghost_downward_connectivity,
                                                                   bool ghost_upward_connectivity) {
  MUNDY_THROW_REQUIRE(bulk_data_ptr->in_modifiable_state(), std::invalid_argument,
                     "GenerateCollisionConstraints: The provided bulk data is not in a modified state. \n"
                         "Be sure to run modificiation_begin() before running this routine.");

  std::vector<stk::mesh::EntityProc> entities_to_ghost;
  for (size_t i = 0; i < pairs_to_ghost.size(); ++i) {
    // Get the state information about the neighbor pair.
    stk::mesh::Entity source_entity = bulk_data_ptr->get_entity(pairs_to_ghost[i].first.id());
    stk::mesh::Entity target_entity = bulk_data_ptr->get_entity(pairs_to_ghost[i].second.id());
    int source_proc = pairs_to_ghost[i].first.proc();
    int target_proc = pairs_to_ghost[i].second.proc();

    // If either of the pair is invalid (typically because it was deleted) ignore that pair.
    bool is_source_valid = bulk_data_ptr->is_valid(source_entity);
    bool is_target_valid = bulk_data_ptr->is_valid(target_entity);
    if (is_source_valid && is_target_valid) {
      stk::mesh::Bucket &source_bucket = bulk_data_ptr->bucket(source_entity);
      stk::mesh::Bucket &target_bucket = bulk_data_ptr->bucket(target_entity);

      bool is_source_owned = source_bucket.owned();
      bool is_target_owned = target_bucket.owned();

      // There are two ways to check if the entities are owned. Assert that they are consistent.
      bool is_source_proc_consistent =
          (bulk_data_ptr->parallel_owner_rank(source_entity) == source_proc) == is_source_owned;
      bool is_target_proc_consistent =
          (bulk_data_ptr->parallel_owner_rank(target_entity) == target_proc) == is_target_owned;
      MUNDY_THROW_ASSERT(is_source_proc_consistent, std::invalid_argument,
                          fmt::format("GenerateCollisionConstraints: The source proc for pair {} gives inconsistent "
                                       "ownership.\nMake sure that the entity proc of your source is correct.",
                                       i));
      MUNDY_THROW_ASSERT(is_target_proc_consistent, std::invalid_argument,
                        fmt::format("GenerateCollisionConstraints: The target proc for pair {} gives inconsistent "
                                     "ownership.\nMake sure that the entity proc of your target is correct.",
                                     i));

      if (is_source_owned) {
        if (is_target_owned) {
          // The current process owns both source and target; no ghosting necessary.
          continue;
        } else {
          // The current process owns the source but not the target; send source entity to the target proc.
          entities_to_ghost.emplace_back(source_entity, target_proc);

          // Optionaly, send the downward connectivity of the source entity to the target proc.
          if (ghost_downward_connectivity) {
            auto downward_connected_entities =
                get_connected_lower_rank_entities(bulk_data_ptr, source_entity, source_bucket.entity_rank());
            entities_to_ghost.reserve(entities_to_ghost.size() + downward_connected_entities.size());
            std::transform(std::make_move_iterator(downward_connected_entities.begin()),
                           std::make_move_iterator(downward_connected_entities.end()),
                           std::back_inserter(entities_to_ghost), [target_proc](stk::mesh::Entity entity) {
                             return std::make_pair(entity, target_proc);
                           });
          }

          // Optionaly, send the upward connectivity of the source entity to the target proc.
          if (ghost_upward_connectivity) {
            auto upward_connected_entities =
                get_connected_higher_rank_entities(bulk_data_ptr, source_entity, source_bucket.entity_rank());
            entities_to_ghost.reserve(entities_to_ghost.size() + upward_connected_entities.size());
            std::transform(std::make_move_iterator(upward_connected_entities.begin()),
                           std::make_move_iterator(upward_connected_entities.end()),
                           std::back_inserter(entities_to_ghost), [target_proc](stk::mesh::Entity entity) {
                             return std::make_pair(entity, target_proc);
                           });
          }
        }
      } else if (is_target_owned) {
        // The current process owns the target but not the source; send target entity to the source proc.
        entities_to_ghost.emplace_back(target_entity, source_proc);

        // Optionaly, send the downward connectivity of the target entity to the source proc.
        if (ghost_downward_connectivity) {
          auto downward_connected_entities =
              get_connected_lower_rank_entities(bulk_data_ptr, target_entity, target_bucket.entity_rank());
          entities_to_ghost.reserve(entities_to_ghost.size() + downward_connected_entities.size());
          std::transform(std::make_move_iterator(downward_connected_entities.begin()),
                         std::make_move_iterator(downward_connected_entities.end()),
                         std::back_inserter(entities_to_ghost), [source_proc](stk::mesh::Entity entity) {
                           return std::make_pair(entity, source_proc);
                         });
        }

        // Optionaly, send the upward connectivity of the target entity to the source proc.
        if (ghost_upward_connectivity) {
          auto upward_connected_entities =
              get_connected_higher_rank_entities(bulk_data_ptr, target_entity, target_bucket.entity_rank());
          entities_to_ghost.reserve(entities_to_ghost.size() + upward_connected_entities.size());
          std::transform(std::make_move_iterator(upward_connected_entities.begin()),
                         std::make_move_iterator(upward_connected_entities.end()),
                         std::back_inserter(entities_to_ghost), [source_proc](stk::mesh::Entity entity) {
                           return std::make_pair(entity, source_proc);
                         });
        }
      }
    }
  }

  stk::mesh::Ghosting &ghosting = bulk_data_ptr->create_ghosting(name_of_ghosting);
  bulk_data_ptr->change_ghosting(ghosting, entities_to_ghost);
  return ghosting;
}

void GenerateCollisionConstraints::generate_empty_collision_constraints_between_pairs(
    mundy::mesh::BulkData *const bulk_data_ptr, stk::mesh::Part *const collision_part_ptr,
    const IdentProcPairVector &pairs_to_connect) {
  // A word of warning: This method is programmed with care to avoid generating duplicative constraints. To do so, we
  // only generate a collision constraint if the entity_key of the source body is less than that of the target body.

  // This function can be broken into steps:
  //   0. Count the number of collision constraints C that need generated.
  //   1. For each pair, generate C elements and C*P nodes.
  //   2. Change the topology of the c elements to be collision constraints.
  //   3. For each collision constraint, attach two unique nodes.
  //   4. For each pair, fetch their linkers and attach the respective nodes.
  MUNDY_THROW_REQUIRE(bulk_data_ptr->in_modifiable_state(), std::invalid_argument,
                     "GenerateCollisionConstraints: The provided bulk data is not in a modified state. \n"
                         "Be sure to run modificiation_begin() before running this routine.");

  // All pairs must be valid.
  size_t num_pairs = pairs_to_connect.size();
  for (size_t i = 0; i < num_pairs; ++i) {
    stk::mesh::Entity source_entity = bulk_data_ptr->get_entity(pairs_to_connect[i].first.id());
    stk::mesh::Entity target_entity = bulk_data_ptr->get_entity(pairs_to_connect[i].second.id());
    bool is_pair_valid = bulk_data_ptr->is_valid(source_entity) && bulk_data_ptr->is_valid(target_entity);
    MUNDY_THROW_ASSERT(is_pair_valid, std::invalid_argument,
                      fmt::format("GenerateCollisionConstraints: Constraint generation failed. Pair i = {} is invalid.", i));
  }

  // Step 0. Count the number of collision constraints C that need generated.
  const size_t num_collisions = std::count_if(pairs_to_connect.begin(), pairs_to_connect.end(),
                                              [](const std::pair<SearchIdentProc, SearchIdentProc> &neighbor_pair) {
                                                return neighbor_pair.first.id() < neighbor_pair.second.id();
                                              });

  // Step 1. Generate C elements and 2C nodes
  // Note, generate_new_entities has a very particular input and output. To aid in understanding the following code,
  // consider the following pseudo-example:
  //    If requests = { 0, 4,  8}, then we are requesting 0 entites of rank 0, 4 entites of rank 1, and 8 entites of
  //    rank 2. The resulting requested_entities is therefore requested_entities = {0 entites of rank 0, 4 entites of
  //    rank 1, 8 entites of rank 2}.
  std::vector<size_t> requests(bulk_data_ptr->mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::ELEMENT_RANK] = num_collisions;
  requests[stk::topology::NODE_RANK] = 2 * num_collisions;
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr->generate_new_entities(requests, requested_entities);

  size_t count = 0;
  for (size_t i = 0; i < num_pairs; i++) {
    // Only generate collision constraints if the source body's id is less than the id of the target body. This prevents
    // duplicate constraints.
    if (pairs_to_connect[i].first.id() < pairs_to_connect[i].second.id()) {
      // Step 2. Associate each element with the collision constraint part.
      stk::mesh::Entity collision_element = requested_entities[2 * num_collisions + count];
      bulk_data_ptr->change_entity_parts(collision_element, stk::mesh::ConstPartVector{collision_part_ptr});

      // Step 3. Set the downward relations from the elements to the nodes.
      stk::mesh::Entity left_node = requested_entities[2 * count + 0];
      stk::mesh::Entity right_node = requested_entities[2 * count + 1];
      bulk_data_ptr->declare_relation(collision_element, left_node, 0);
      bulk_data_ptr->declare_relation(collision_element, right_node, 1);

      // Step 4. Attach the nodes to their respective linker.
      // These linkers may have existing nodes, so we tack the new ones onto the end of their dynamic connectivity.
      stk::mesh::Entity source_entity = bulk_data_ptr->get_entity(pairs_to_connect[i].first.id());
      stk::mesh::Entity target_entity = bulk_data_ptr->get_entity(pairs_to_connect[i].second.id());
      MUNDY_THROW_ASSERT(bulk_data_ptr->num_connectivity(source_entity, stk::topology::CONSTRAINT_RANK) == 1,
                         std::invalid_argument,
                         fmt::format("GenerateCollisionConstraints: The source entity within Pair i = {} doesn't have a "
                                      "linker (or has multiple linkers).",
                                      i));
      MUNDY_THROW_ASSERT(bulk_data_ptr->num_connectivity(target_entity, stk::topology::CONSTRAINT_RANK) == 1,
                         std::invalid_argument,
                          fmt::format("GenerateCollisionConstraints: The target entity within Pair i = {} doesn't have a "
                                        "linker (or has multiple linkers).",
                                        i));
      stk::mesh::Entity const source_linker = bulk_data_ptr_->begin(source_entity, stk::topology::CONSTRAINT_RANK)[0];
      stk::mesh::Entity const target_linker = bulk_data_ptr_->begin(target_entity, stk::topology::CONSTRAINT_RANK)[0];
      const size_t num_nodes_in_source_linker = bulk_data_ptr_->num_nodes(source_linker);
      const size_t num_nodes_in_target_linker = bulk_data_ptr_->num_nodes(target_linker);
      bulk_data_ptr->declare_relation(source_linker, left_node, num_nodes_in_source_linker);
      bulk_data_ptr->declare_relation(target_linker, right_node, num_nodes_in_target_linker);

      count++;
    }
  }
}
//}

}  // namespace constraints

}  // namespace mundy
