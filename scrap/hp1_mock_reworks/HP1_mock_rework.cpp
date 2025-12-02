// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If
// not, see <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

// External libs
#include <openrand/philox.h>

// C++ core
#include <algorithm>   // for std::transform
#include <filesystem>  // for std::filesystem::path
#include <fstream>     // for std::ofstream
#include <iostream>    // for std::cout, std::endl
#include <memory>      // for std::shared_ptr, std::unique_ptr
#include <numeric>     // for std::accumulate
#include <regex>       // for std::regex
#include <string>      // for std::string
#include <vector>      // for std::vector

// Trilinos
#include <Kokkos_Core.hpp>                   // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList
#include <stk_balance/balance.hpp>           // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>        // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>            // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>    // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>          // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>   // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>            // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>        // for stk::mesh::Selector
#include <stk_topology/topology.hpp>         // for stk::topology
#include <stk_util/parallel/Parallel.hpp>    // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_alens/actions_crosslinkers.hpp>                // for mundy::alens::crosslinkers...
#include <mundy_alens/periphery/FastDirectPeriphery.hpp>       // for gen_sphere_quadrature
#include <mundy_constraints/AngularSprings.hpp>                // for mundy::constraints::AngularSprings
#include <mundy_constraints/ComputeConstraintForcing.hpp>      // for mundy::constraints::ComputeConstraintForcing
#include <mundy_constraints/DeclareAndInitConstraints.hpp>     // for mundy::constraints::DeclareAndInitConstraints
#include <mundy_constraints/HookeanSprings.hpp>                // for mundy::constraints::HookeanSprings
#include <mundy_core/MakeStringArray.hpp>                      // for mundy::core::make_string_array
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_io/IOBroker.hpp>         // for mundy::io::IOBroker
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers
#include <mundy_math/Hilbert.hpp>                           // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>                           // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>       // for mundy::math::distance::ellipsoid_ellipsoid
#include <mundy_mesh/BulkData.hpp>                          // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>    // for mundy::mesh::MetaData
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>        // for mundy::mesh::utils::destroy_flagged_entities
#include <mundy_mesh/utils/FillFieldWithValue.hpp>            // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaKernelDispatcher.hpp>                // for mundy::meta::MetaKernelDispatcher
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs
#include <mundy_meta/utils/MeshGeneration.hpp>  // for mundy::meta::utils::generate_class_instance_and_mesh_from_meta_class_requirements
#include <mundy_shapes/ComputeAABB.hpp>  // for mundy::shapes::ComputeAABB
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

/*
I think users should define what they expect the internal variable to be called.
It's their responsibility to avoid conflicts. This puts more burden on the user, but it
does wonders to decrease verbosity and to increase readability. What you ask for is what
you get. For example:

class ExampleMetaMethod : public mundy::meta::MetaMethod {
  MUNDY_METHOD_HAS_BULKDATA(ExampleMetaMethod);
  MUNDY_METHOD_HAS_FIELD(ExampleMetaMethod, double, node, coordinates);
  MUNDY_METHOD_HAS_OBJECT(ExampleMetaMethod, bool, enable_print);

  void run {
    if (*enable_print_object_ptr_) {
      std:: cout << node_coordinates_field_ptr_->name() << std::endl;
    }
  }
};

vs

class ExampleMetaMethod : public mundy::meta::MetaMethod {
  MUNDY_METHOD_HAS_BULKDATA(ExampleMetaMethod);
  MUNDY_METHOD_HAS_FIELD(ExampleMetaMethod, double, coords_);
  MUNDY_METHOD_HAS_OBJECT(ExampleMetaMethod, bool, enable_print_ptr_);

  void run {
    if (*enable_print_ptr_) {
      std:: cout << coords_->name() << std::endl;
    }
  }
};


The periphery poses a particular challenge in that it requires precomputation.
It also shows that we need documentation.

The issue of needing precomputation is equivilant to the issue of needing private member
variables with a longer lifetime then when the function is called. The solution is
trivial. So trivial that I'm supprised I didn't think of it before. Quite simply, instead
of construct, set, set, set, run, it's construct all methods up front and then set, set,
set, run repeatedly on the constructed methods. Keep a run counter and perform certain
actions on the first run if desired. That fits the design of FBP perfectly.

Things like BINDING_STATE_CHANGE can become enums within the
perform_crosslinker_state_change method. This enum is stored on the particles and doesn't
correspond to a means for selecting one method or another. For enums that simply
distinguish one method from another, they can be deleted. The method name is enough to
distinguish. This aids extensibility. If you want to add a new method, you don't have to
add a new enum. You just add a new method.

Inheritance can be done via a helper macro
MUNDY_METHOD(some::namespace, SomeClass)
  MUNDY_METHOD_INHERITS_FROM(public ParentClass, public SecondParentClass) {
};
*/

namespace mundy {

namespace alens {

enum class BINDING_STATE_CHANGE : unsigned {
  NONE = 0u,
  LEFT_TO_DOUBLY,
  RIGHT_TO_DOUBLY,
  DOUBLY_TO_LEFT,
  DOUBLY_TO_RIGHT
};

std::ostream &operator<<(std::ostream &os, const BINDING_STATE_CHANGE &state) {
  switch (state) {
    case BINDING_STATE_CHANGE::NONE:
      os << "NONE";
      break;
    case BINDING_STATE_CHANGE::LEFT_TO_DOUBLY:
      os << "LEFT_TO_DOUBLY";
      break;
    case BINDING_STATE_CHANGE::RIGHT_TO_DOUBLY:
      os << "RIGHT_TO_DOUBLY";
      break;
    case BINDING_STATE_CHANGE::DOUBLY_TO_LEFT:
      os << "DOUBLY_TO_LEFT";
      break;
    case BINDING_STATE_CHANGE::DOUBLY_TO_RIGHT:
      os << "DOUBLY_TO_RIGHT";
      break;
    default:
      os << "UNKNOWN";
      break;
  }
  return os;
}

/*
  We can use a macro to hide CRTP! This way we can remove some boiler plate code.
  The MUNDY_METHOD macro will not give the bracket after the class name and public
  inheritance declaration, meaning that users can have methods inherit from their own
  classes as desired. This should feel like GTEST's TEST macro.
*/

MUNDY_METHOD(mundy::alens, GhostLinkedEntities) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_FIELD(double, constraint_linked_entities_field);
  MUNDY_METHOD_HAS_FIELD(int, constraint_linked_entity_owners_field);

 public:
  void run() override {
    bulk_data_ptr_->modification_begin();
    mundy::linkers::fixup_linker_entity_ghosting(*bulk_data_ptr_, *constraint_linked_entities_field_ptr_,
                                                 *constraint_linked_entity_owners_field_ptr_, *selector_ptr_);
    bulk_data_ptr_->modification_end();
  }
};

MUNDY_METHOD(mundy::alens, DeclareChromatinAndHP1) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromosomes);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromatin_repeats);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_euchromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_heterochromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(bool, enable_backbone_collision);
  MUNDY_METHOD_HAS_OBJECT(bool, enable_backbone_springs);
  MUNDY_METHOD_HAS_OBJECT(bool, enable_crosslinkers);

 public:
  void run() override {
    // Dereference everything
    const size_t num_chromosomes = *num_chromosomes_ptr_;
    const size_t num_chromatin_repeats = *num_chromatin_repeats_ptr_;
    const size_t num_euchromatin_per_repeat = *num_euchromatin_per_repeat_ptr_;
    const size_t num_heterochromatin_per_repeat = *num_heterochromatin_per_repeat_ptr_;
    const bool enable_backbone_collision = *enable_backbone_collision_ptr_;
    const bool enable_backbone_springs = *enable_backbone_springs_ptr_;
    const bool enable_crosslinkers = *enable_crosslinkers_ptr_;

    // Calculate some constants, like the total number of spheres or segments per
    // chromosome
    const size_t num_heterochromatin_spheres = num_chromatin_repeats / 2 * num_heterochromatin_per_repeat +
                                               num_chromatin_repeats % 2 * num_heterochromatin_per_repeat;
    const size_t num_euchromatin_spheres = num_chromatin_repeats / 2 * num_euchromatin_per_repeat;
    const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
    const size_t num_spheres_per_chromosome = num_nodes_per_chromosome;
    const size_t num_segments_per_chromosome = num_nodes_per_chromosome - 1;
    const size_t num_elements_created_per_chromosome =
        num_spheres_per_chromosome +
        (enable_backbone_springs || enable_backbone_collision) * num_segments_per_chromosome +
        enable_crosslinkers * num_heterochromatin_spheres;

    std::cout << "Per chromosome:\n";
    std::cout << "num_heterochromatin_spheres: " << num_heterochromatin_spheres << std::endl;
    std::cout << "num_euchromatin_spheres:     " << num_euchromatin_spheres << std::endl;
    std::cout << "num_nodes_per_chromosome:    " << num_nodes_per_chromosome << std::endl;
    std::cout << "num_spheres_per_chromosome:  " << num_spheres_per_chromosome << std::endl;
    std::cout << "num_segments_per_chromosome: " << num_segments_per_chromosome << std::endl;

    bulk_data_ptr_->modification_begin();

    // Rank 0: Declare N chromatin chains randomly in space
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes; j++) {
        std::cout << "Creating chromosome " << j << std::endl;

        // Some notes on what will and will not be created
        // There are three enables that matter: enble_crosslinkers,
        // enable_backbone_springs, and enable_backbone_collision
        //   The backbone spheres will always be created.
        //   The HP1 crosslinkers will be created if enable_crosslinkers_ is true.
        //   The backbone segments will be created if EITHER enable_backbone_springs_ or
        //   enable_backbone_collision_ is true, but the segment will only be placed in
        //   the corresponding part if it is enabled.

        // Figure out the starting indices of the nodes and elements
        const size_t start_node_id = num_nodes_per_chromosome * j + 1u;
        const size_t start_element_id = num_elements_created_per_chromosome * j + 1u;

        // Helper functions for getting the IDs of various objects
        auto get_node_id = [start_node_id](const size_t &seq_node_index) { return start_node_id + seq_node_index; };

        auto get_sphere_id = [start_element_id](const size_t &seq_sphere_index) {
          return start_element_id + seq_sphere_index;
        };

        auto get_segment_id = [start_element_id, num_spheres_per_chromosome](const size_t &seq_segment_index) {
          return start_element_id + num_spheres_per_chromosome + seq_segment_index;
        };

        auto get_crosslinker_id = [start_element_id, num_spheres_per_chromosome, num_segments_per_chromosome,
                                   enable_backbone_collision,
                                   enable_backbone_springs](const size_t &seq_crosslinker_index) {
          return start_element_id + num_spheres_per_chromosome +
                 (enable_backbone_springs || enable_backbone_collision) * num_segments_per_chromosome +
                 seq_crosslinker_index;
        };

        // Try to use modulo math to determine region
        auto get_region_by_id = [num_heterochromatin_per_repeat,
                                 num_euchromatin_per_repeat](const size_t &seq_sphere_id) {
          auto local_idx = seq_sphere_id % (num_heterochromatin_per_repeat + num_euchromatin_per_repeat);
          return local_idx < num_heterochromatin_per_repeat ? std::string("H") : std::string("E");
        };

        // Show what a single chromatin chain would be in terms of membership
        std::cout << "Regional map:" << std::endl;
        for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
          std::cout << get_region_by_id(i);
        }
        std::cout << std::endl;

        // Temporary/scratch variables
        stk::mesh::PartVector empty;

        // Logically, it makes the most sense to march down the segments in a single
        // chromosome, adjusting their part membership as we go. Do this across the
        // elements of the chromatin backbone. Initialize the backbone such that we have
        // different sphere types.
        //  E : euchromatin spheres
        //  H : heterochromatin spheres
        // ---: backbone springs (EE, EH, or HH depending on attached spheres)
        //
        //  H---H---E---E---E---E---E---E---H---H
        //
        for (size_t segment_local_idx = 0; segment_local_idx < num_segments_per_chromosome; segment_local_idx++) {
          // Keep track of the vertex IDs for part membership (local index into array)
          const size_t vertex_left_idx = segment_local_idx;
          const size_t vertex_right_idx = segment_local_idx + 1;
          // Process the nodes for this segment
          stk::mesh::EntityId left_node_id = get_node_id(segment_local_idx);
          stk::mesh::EntityId right_node_id = get_node_id(segment_local_idx + 1);

          stk::mesh::Entity left_node = bulk_data_ptr_->get_entity(node_rank_, left_node_id);
          stk::mesh::Entity right_node = bulk_data_ptr_->get_entity(node_rank_, right_node_id);
          if (!bulk_data_ptr_->is_valid(left_node)) {
            left_node = bulk_data_ptr_->declare_node(left_node_id, empty);
          }
          if (!bulk_data_ptr_->is_valid(right_node)) {
            right_node = bulk_data_ptr_->declare_node(right_node_id, empty);
          }

          // Each node is attached to a sphere that is (H)eterochromatin, (E)uchromatin,
          // or (BS)BindingSite
          stk::mesh::EntityId left_sphere_id = get_sphere_id(segment_local_idx);
          stk::mesh::EntityId right_sphere_id = get_sphere_id(segment_local_idx + 1);
          stk::mesh::Entity left_sphere = bulk_data_ptr_->get_entity(element_rank_, left_sphere_id);
          stk::mesh::Entity right_sphere = bulk_data_ptr_->get_entity(element_rank_, right_sphere_id);
          if (!bulk_data_ptr_->is_valid(left_sphere)) {
            // Figure out the part we belong to
            stk::mesh::PartVector pvector;
            if (get_region_by_id(vertex_left_idx) == "H") {
              pvector.push_back(h_part_ptr_);
            } else if (get_region_by_id(vertex_left_idx) == "E") {
              pvector.push_back(e_part_ptr_);
            }
            // Declare the sphere and connect to it's node
            left_sphere = bulk_data_ptr_->declare_element(left_sphere_id, pvector);
            bulk_data_ptr_->declare_relation(left_sphere, left_node, 0);
          }
          if (!bulk_data_ptr_->is_valid(right_sphere)) {
            // Figure out the part we belong to
            stk::mesh::PartVector pvector;
            if (get_region_by_id(vertex_right_idx) == "H") {
              pvector.push_back(h_part_ptr_);
            } else if (get_region_by_id(vertex_right_idx) == "E") {
              pvector.push_back(e_part_ptr_);
            }
            // Declare the sphere and connect to it's node
            right_sphere = bulk_data_ptr_->declare_element(right_sphere_id, pvector);
            bulk_data_ptr_->declare_relation(right_sphere, right_node, 0);
          }

          // Figure out how to do the spherocylinder segments along the edges now
          if (enable_backbone_springs || enable_backbone_collision) {
            stk::mesh::Entity segment = bulk_data_ptr_->get_entity(element_rank_, get_segment_id(segment_local_idx));
            if (!bulk_data_ptr_->is_valid(segment)) {
              stk::mesh::PartVector pvector;
              pvector.push_back(backbone_segments_part_ptr_);
              if (enable_backbone_springs) {
                if (get_region_by_id(vertex_left_idx) == "E" && get_region_by_id(vertex_right_idx) == "E") {
                  pvector.push_back(ee_springs_part_ptr_);
                } else if (get_region_by_id(vertex_left_idx) == "E" && get_region_by_id(vertex_right_idx) == "H") {
                  pvector.push_back(eh_springs_part_ptr_);
                } else if (get_region_by_id(vertex_left_idx) == "H" && get_region_by_id(vertex_right_idx) == "E") {
                  pvector.push_back(eh_springs_part_ptr_);
                } else if (get_region_by_id(vertex_left_idx) == "H" && get_region_by_id(vertex_right_idx) == "H") {
                  pvector.push_back(hh_springs_part_ptr_);
                }
              }
              segment = bulk_data_ptr_->declare_element(get_segment_id(segment_local_idx), pvector);
              bulk_data_ptr_->declare_relation(segment, left_node, 0);
              bulk_data_ptr_->declare_relation(segment, right_node, 1);
            }
          }
        }

        // Declare the crosslinkers along the backbone
        // Every sphere gets a left bound crosslinker
        //  E : euchromatin spheres
        //  H : heterochromatin spheres
        //  | : crosslinkers
        // ---: backbone springs
        //
        //  |   |                           |   |
        //  H---H---E---E---E---E---E---E---H---H

        // March down the chain of spheres, adding crosslinkers as we go. We just want to
        // add to the heterochromatin spheres, and so keep track of a running
        // hp1_sphere_index.
        if (enable_crosslinkers_) {
          size_t hp1_sphere_index = 0;
          for (size_t sphere_local_idx = 0; sphere_local_idx < num_spheres_per_chromosome; sphere_local_idx++) {
            stk::mesh::Entity sphere_node = bulk_data_ptr_->get_entity(node_rank_, get_node_id(sphere_local_idx));
            MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(sphere_node), "Node " << sphere_local_idx << " is not valid.");
            // Check if we are a heterochromatin sphere
            if (get_region_by_id(sphere_local_idx) == "H") {
              // Bind left and right nodes to the same node to start simulation (everybody
              // is left bound) Create the HP1 crosslinker
              auto left_bound_hp1_part_vector = stk::mesh::PartVector{left_hp1_part_ptr_};
              stk::mesh::EntityId hp1_crosslinker_id = get_crosslinker_id(hp1_sphere_index);
              stk::mesh::Entity hp1_crosslinker =
                  bulk_data_ptr_->declare_element(hp1_crosslinker_id, left_bound_hp1_part_vector);
              stk::mesh::Permutation invalid_perm = stk::mesh::Permutation::INVALID_PERMUTATION;
              bulk_data_ptr_->declare_relation(hp1_crosslinker, sphere_node, 0, invalid_perm);
              bulk_data_ptr_->declare_relation(hp1_crosslinker, sphere_node, 1, invalid_perm);
              MUNDY_THROW_ASSERT(bulk_data_ptr_->bucket(hp1_crosslinker).topology() != stk::topology::INVALID_TOPOLOGY,
                                 "The crosslinker with id " << hp1_crosslinker_id << " has an invalid topology.");

              hp1_sphere_index++;
            }
          }
        }
      }
    }
    bulk_data_ptr_->modification_end();
  }
};

MUNDY_METHOD(mundy::alens, InitializeChromosomePositionsGrid) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromosomes);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromatin_repeats);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_euchromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_heterochromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(double, initial_chromosome_separation);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);

 public:
  void run() override {
    // Dereference everything
    const size_t num_chromosomes = *num_chromosomes_ptr_;
    const size_t num_chromatin_repeats = *num_chromatin_repeats_ptr_;
    const size_t num_euchromatin_per_repeat = *num_euchromatin_per_repeat_ptr_;
    const size_t num_heterochromatin_per_repeat = *num_heterochromatin_per_repeat_ptr_;
    const double initial_chromosome_separation = *initial_chromosome_separation_ptr_;
    auto &node_coord_field = *node_coord_field_ptr_;

    // We need to get which chromosome this rank is responsible for initializing, luckily,
    // should follow what was done for the creation step. Do this inside a modification
    // loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes; j++) {
        openrand::Philox rng(j, 0);
        double jdouble = static_cast<double>(j);
        mundy::math::Vector3<double> r_start(2.0 * j, 0.0, 0.0);
        // Add a tiny random change in X to make sure we don't wind up in perfectly
        // parallel pathological states
        mundy::math::Vector3<double> u_hat(rng.uniform<double>(0.0, 0.001), 0.0, 1.0);
        u_hat = u_hat / mundy::math::two_norm(u_hat);

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats / 2 * num_heterochromatin_per_repeat +
                                                   num_chromatin_repeats % 2 * num_heterochromatin_per_repeat;
        const size_t num_euchromatin_spheres = num_chromatin_repeats / 2 * num_euchromatin_per_repeat;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), "Node " << i << " is not valid.");

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_chromosome_separation * u_hat;
          stk::mesh::field_data(node_coord_field, node)[0] = r[0];
          stk::mesh::field_data(node_coord_field, node)[1] = r[1];
          stk::mesh::field_data(node_coord_field, node)[2] = r[2];
        }
      }
    }
  }
};  // class InitializeChromosomePositionsGrid

MUNDY_METHOD(mundy::alens, InitializeChromosomePositionsRandomUnitCell) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromosomes);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromatin_repeats);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_euchromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_heterochromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(double, initial_chromosome_separation);
  MUNDY_METHOD_HAS_OBJECT(mundy::math::Vector3<double>, unit_cell_size);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);

 public:
  void run() override {
    // Dereference everything
    const size_t num_chromosomes = *num_chromosomes_ptr_;
    const size_t num_chromatin_repeats = *num_chromatin_repeats_ptr_;
    const size_t num_euchromatin_per_repeat = *num_euchromatin_per_repeat_ptr_;
    const size_t num_heterochromatin_per_repeat = *num_heterochromatin_per_repeat_ptr_;
    const double initial_chromosome_separation = *initial_chromosome_separation_ptr_;
    const mundy::math::Vector3<double> unit_cell_size = *unit_cell_size_ptr_;
    auto &node_coord_field = *node_coord_field_ptr_;

    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t j = 0; j < num_chromosomes_; j++) {
        // Find a random place within the unit cell with a random orientation for the
        // chain.
        openrand::Philox rng(j, 0);
        mundy::math::Vector3<double> r_start(rng.uniform<double>(-0.5 * unit_cell_size[0], 0.5 * unit_cell_size[0]),
                                             rng.uniform<double>(-0.5 * unit_cell_size[1], 0.5 * unit_cell_size[1]),
                                             rng.uniform<double>(-0.5 * unit_cell_size[2], 0.5 * unit_cell_size[2]));
        // Find a random unit vector direction
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats / 2 * num_heterochromatin_per_repeat +
                                                   num_chromatin_repeats_ % 2 * num_heterochromatin_per_repeat;
        const size_t num_euchromatin_spheres = num_chromatin_repeats / 2 * num_euchromatin_per_repeat;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * j + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (j + 1) + 1u;
        for (size_t i = start_node_index; i < end_node_index; ++i) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), "Node " << i << " is not valid.");

          // Assign the node coordinates
          mundy::math::Vector3<double> r =
              r_start + static_cast<double>(i - start_node_index) * initial_chromosome_separation * u_hat;
          stk::mesh::field_data(node_coord_field, node)[0] = r[0];
          stk::mesh::field_data(node_coord_field, node)[1] = r[1];
          stk::mesh::field_data(node_coord_field, node)[2] = r[2];
        }
      }
    }
  }
};  // class InitializeChromosomePositionsRandomUnitCell

MUNDY_METHOD(mundy::alens, InitializeChromosomePositionsHilbertRandomUnitCell) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromosomes);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_chromatin_repeats);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_euchromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_heterochromatin_per_repeat);
  MUNDY_METHOD_HAS_OBJECT(double, initial_chromosome_separation);
  MUNDY_METHOD_HAS_OBJECT(mundy::math::Vector3<double>, unit_cell_size);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);

 public:
  void run() override {
    // Dereference everything
    const size_t num_chromosomes = *num_chromosomes_ptr_;
    const size_t num_chromatin_repeats = *num_chromatin_repeats_ptr_;
    const size_t num_euchromatin_per_repeat = *num_euchromatin_per_repeat_ptr_;
    const size_t num_heterochromatin_per_repeat = *num_heterochromatin_per_repeat_ptr_;
    const double initial_chromosome_separation = *initial_chromosome_separation_ptr_;
    const mundy::math::Vector3<double> unit_cell_size = *unit_cell_size_ptr_;
    auto &node_coord_field = *node_coord_field_ptr_;

    // We need to get which chromosome this rank is responsible for initializing, luckily,
    // should follow what was done for the creation step. Do this inside a modification
    // loop so we can go by node index, rather than ID.
    if (bulk_data_ptr_->parallel_rank() == 0) {
      // Initialize the chromosomes randomly in the unit cell
      //
      // If we want to initialize uniformly inside a sphere packing, here are the
      // coordinates for a given number of spheres within a bigger sphere.
      // http://hydra.nat.uni-magdeburg.de/packing/ssp/ssp.html
      std::vector<mundy::math::Vector3<double>> chromosome_centers_array;
      std::vector<double> chromosome_radii_array;
      for (size_t ichromosome = 0; ichromosome < num_chromosomes_; ichromosome++) {
        // Figure out which nodes we are doing
        const size_t num_heterochromatin_spheres = num_chromatin_repeats / 2 * num_heterochromatin_per_repeat +
                                                   num_chromatin_repeats % 2 * num_heterochromatin_per_repeat;
        const size_t num_euchromatin_spheres = num_chromatin_repeats / 2 * num_euchromatin_per_repeat;
        const size_t num_nodes_per_chromosome = num_heterochromatin_spheres + num_euchromatin_spheres;
        size_t start_node_index = num_nodes_per_chromosome * ichromosome + 1u;
        size_t end_node_index = num_nodes_per_chromosome * (ichromosome + 1) + 1u;

        // Generate a random unit vector (will be used for creating the location of the
        // nodes, the random position in the unit cell will be handled later).
        openrand::Philox rng(ichromosome, 0);
        const double zrand = rng.rand<double>() - 1.0;
        const double wrand = std::sqrt(1.0 - zrand * zrand);
        const double trand = 2.0 * M_PI * rng.rand<double>();
        mundy::math::Vector3<double> u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

        // Once we have the number of chromosome spheres we can get the hilbert curve set
        // up. This will be at some orientation and then have sides with a length of
        // initial_chromosome_separation.
        auto [hilbert_position_array, hilbert_directors] = mundy::math::create_hilbert_positions_and_directors(
            num_nodes_per_chromosome, u_hat, initial_chromosome_separation);

        // Create the local positions of the spheres
        std::vector<mundy::math::Vector3<double>> sphere_position_array;
        for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
          sphere_position_array.push_back(hilbert_position_array[isphere]);
        }

        // Figure out where the center of the chromosome is, and its radius, in its own
        // local space
        mundy::math::Vector3<double> r_chromosome_center_local(0.0, 0.0, 0.0);
        double r_max = 0.0;
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          r_chromosome_center_local += sphere_position_array[i];
        }
        r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
        }

        // Do max_trials number of insertion attempts to get a random position and
        // orientation within the unit cell that doesn't overlap with exiting chromosomes.
        const size_t max_trials = 1000;
        size_t itrial = 0;
        bool chromosome_inserted = false;
        while (itrial <= max_trials) {
          // Generate a random position within the unit cell.
          mundy::math::Vector3<double> r_start(rng.uniform<double>(-0.5 * unit_cell_size[0], 0.5 * unit_cell_size[0]),
                                               rng.uniform<double>(-0.5 * unit_cell_size[1], 0.5 * unit_cell_size[1]),
                                               rng.uniform<double>(-0.5 * unit_cell_size[2], 0.5 * unit_cell_size[2]));

          // Check for overlaps with existing chromosomes
          bool found_overlap = false;
          for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
            double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
            if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
              found_overlap = true;
              break;
            }
          }
          if (found_overlap) {
            itrial++;
          } else {
            chromosome_inserted = true;
            chromosome_centers_array.push_back(r_start);
            chromosome_radii_array.push_back(r_max);
            break;
          }
        }
        MUNDY_THROW_ASSERT(chromosome_inserted, "Failed to insert chromosome after " << max_trials << " trials.");

        // Generate all the positions along the curve due to the placement in the global
        // space
        std::vector<mundy::math::Vector3<double>> new_position_array;
        for (size_t i = 0; i < sphere_position_array.size(); i++) {
          new_position_array.push_back(chromosome_centers_array.back() + r_chromosome_center_local -
                                       sphere_position_array[i]);
        }

        // Update the coordinates for this chromosome
        for (size_t i = start_node_index, idx = 0; i < end_node_index; ++i, ++idx) {
          stk::mesh::Entity node = bulk_data_ptr_->get_entity(node_rank_, i);
          MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), "Node " << i << " is not valid.");

          // Assign the node coordinates
          stk::mesh::field_data(node_coord_field, node)[0] = new_position_array[idx][0];
          stk::mesh::field_data(node_coord_field, node)[1] = new_position_array[idx][1];
          stk::mesh::field_data(node_coord_field, node)[2] = new_position_array[idx][2];
        }
      }
    }
  }
};  // class InitializeChromosomePositionsHilbert

MUNDY_METHOD(mundy::alens, HydroPeripheryGuassLagendre) {
  MUNDY_METHOD_HAS_OBJECT(double, viscosity);
  MUNDY_METHOD_HAS_OBJECT(double, radius);
  MUNDY_METHOD_HAS_OBJECT(size_t, spectral_order);

 public:
  void run() override {
    if (run_counter_ == 0) {
      // Generate the quadrature points and weights for the sphere
      std::vector<double> points_vec;
      std::vector<double> weights_vec;
      std::vector<double> normals_vec;
      const bool invert = true;
      const bool include_poles = false;
      mundy::alens::periphery::gen_sphere_quadrature(*spectral_order_ptr_, *radius_ptr_, &points_vec, &weights_vec,
                                                     &normals_vec, include_poles, invert);

      // Create the periphery object
      const size_t num_surface_nodes = weights_vec.size();
      periphery_ptr_ =
          std::make_shared<mundy::alens::periphery::FastDirectPeriphery>(num_surface_nodes, *viscosity_ptr_);
      periphery_ptr_->set_surface_positions(points_vec.data())
          .set_quadrature_weights(weights_vec.data())
          .set_surface_normals(normals_vec.data());

      // Run the precomputation for the inverse self-interaction matrix
      const bool write_to_file = false;
      periphery_ptr_->build_inverse_self_interaction_matrix(write_to_file);
    }

    // Compute the surface forces that would induce the given surface velocities
    periphery_ptr_->compute_surface_forces(surface_velocities, surface_forces);
    run_counter_++;
  }

 private:
  int run_counter_ = 0;
  std::shared_ptr<mundy::alens::periphery::FastDirectPeriphery> periphery_ptr_;
};  // class InitializeHydrodynamicPeripheryGuassLagendre

MUNDY_METHOD(mundy::alens, HydroPeripheryFromFile) {
  MUNDY_METHOD_HAS_OBJECT(double, viscosity);
  MUNDY_METHOD_HAS_OBJECT(size_t, num_quadrature_points);
  MUNDY_METHOD_HAS_OBJECT(std::string, quadrature_points_filename);
  MUNDY_METHOD_HAS_OBJECT(std::string, quadrature_weights_filename);
  MUNDY_METHOD_HAS_OBJECT(std::string, quadrature_normals_filename);

 public:
  void run() override {
    if (run_counter_ == 0) {
      periphery_ptr_ =
          std::make_shared<mundy::alens::periphery::FastDirectPeriphery>(*num_quadrature_points_ptr_, *viscosity_ptr_);
      periphery_ptr_->set_surface_positions(*periphery_hydro_quadrature_points_filename_ptr_)
          .set_quadrature_weights(*periphery_hydro_quadrature_weights_filename_ptr_)
          .set_surface_normals(*periphery_hydro_quadrature_normals_filename_ptr_);

      // Run the precomputation for the inverse self-interaction matrix
      const bool write_to_file = false;
      periphery_ptr_->build_inverse_self_interaction_matrix(write_to_file);
    }

    // Compute the surface forces that would induce the given surface velocities
    periphery_ptr_->compute_surface_forces(surface_velocities, surface_forces);
  }

 private:
  int run_counter_ = 0;
  std::shared_ptr<mundy::alens::periphery::FastDirectPeriphery> periphery_ptr_;
};  // class InitializeHydrodynamicPeriphery

MUNDY_METHOD(mundy::alens, DeclareAndInitializeRandomSphericalPeripheryBindSites) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_METADATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_bind_sites);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_radius);
  MUNDY_METHOD_HAS_SELECTOR(binding_site_selector);
  MUNDY_METHOD_HAS_OBJECT(stk::mesh::Field<double>, node_coord_field);

 public:
  void run() override {
    // Dereference everything
    const size_t num_bind_sites = *num_bind_sites_ptr_;
    const double periphery_radius = *periphery_radius_ptr_;
    auto &binding_site_selector = *binding_site_selector_ptr_;
    auto &node_coord_field = *node_coord_field_ptr_;

    // Fetch the binding site parts from the selector
    stk::mesh::PartVector bs_parts;
    binding_site_selector.get_parts(bs_parts);

    // Declare the binding sites
    bulk_data_ptr_->modification_begin();
    std::vector<std::size_t> requests(meta_data_ptr_->entity_rank_count(), 0);
    if (bulk_data_ptr_->parallel_rank() == 0) {
      requests[stk::topology::NODE_RANK] = num_bind_sites;
      requests[stk::topology::ELEMENT_RANK] = num_bind_sites;
    }
    std::vector<stk::mesh::Entity> requested_entities;
    bulk_data_ptr_->generate_new_entities(requests, requested_entities);
    bulk_data_ptr_->change_entity_parts(requested_entities, stk::mesh::PartVector{bs_part_ptr_},
                                        stk::mesh::PartVector{});
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t i = 0; i < num_bind_sites; i++) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[num_bind_sites + i];
        bulk_data_ptr_->declare_relation(sphere_i, node_i, 0);
      }
    }
    bulk_data_ptr_->modification_end();

    // Initialize the binding site positions
    if (bulk_data_ptr_->parallel_rank() == 0) {
      openrand::Philox rng(1234, 0);
      for (size_t i = 0; i < num_bind_sites; i++) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[num_bind_sites + i];
        double *node_coords = stk::mesh::field_data(node_coord_field, node_i);

        const double u1 = rng.rand<double>();
        const double u2 = rng.rand<double>();
        const double theta = 2.0 * M_PI * u1;
        const double phi = std::acos(2.0 * u2 - 1.0);
        node_coords[0] = periphery_radius * std::sin(phi) * std::cos(theta);
        node_coords[1] = periphery_radius * std::sin(phi) * std::sin(theta);
        node_coords[2] = periphery_radius * std::cos(phi);
      }
    }
  }
};

MUNDY_METHOD(mundy::alens, DeclareAndInitializeRandomEllipsoidalPeripheryBindSites) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_METADATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_bind_sites);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r1);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r2);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r3);
  MUNDY_METHOD_HAS_SELECTOR(binding_site_selector);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);

 public:
  void run() override {
    // Dereference everything
    const size_t num_bind_sites = *num_bind_sites_ptr_;
    const double periphery_r1 = *periphery_r1_ptr_;
    const double periphery_r2 = *periphery_r2_ptr_;
    const double periphery_r3 = *periphery_r3_ptr_;
    auto &binding_site_selector = *binding_site_selector_ptr_;
    auto &node_coord_field = *node_coord_field_ptr_;

    // Fetch the binding site parts from the selector
    stk::mesh::PartVector bs_parts;
    binding_site_selector.get_parts(bs_parts);

    // Declare the binding sites
    bulk_data_ptr_->modification_begin();
    std::vector<std::size_t> requests(meta_data_ptr_->entity_rank_count(), 0);
    if (bulk_data_ptr_->parallel_rank() == 0) {
      requests[stk::topology::NODE_RANK] = num_bind_sites;
      requests[stk::topology::ELEMENT_RANK] = num_bind_sites;
    }
    std::vector<stk::mesh::Entity> requested_entities;
    bulk_data_ptr_->generate_new_entities(requests, requested_entities);
    bulk_data_ptr_->change_entity_parts(requested_entities, stk::mesh::PartVector{bs_part_ptr_},
                                        stk::mesh::PartVector{});
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t i = 0; i < num_bind_sites; i++) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[num_bind_sites + i];
        bulk_data_ptr_->declare_relation(sphere_i, node_i, 0);
      }
    }
    bulk_data_ptr_->modification_end();

    // Initialize the binding site positions
    if (bulk_data_ptr_->parallel_rank() == 0) {
      const double a = periphery_r1;
      const double b = periphery_r2;
      const double c = periphery_r3;
      const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
      auto keep = [&a, &b, &c, &inv_mu_max, &rng](double x, double y, double z) {
        const double mu_xyz =
            std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
        return inv_mu_max * mu_xyz > rng.rand<double>();
      };

      for (size_t i = 0; i < num_bind_sites; i++) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[num_bind_sites + i];
        double *node_coords = stk::mesh::field_data(node_coord_field, node_i);

        while (true) {
          // Generate a random point on the unit sphere
          const double u1 = rng.rand<double>();
          const double u2 = rng.rand<double>();
          const double theta = 2.0 * M_PI * u1;
          const double phi = std::acos(2.0 * u2 - 1.0);
          node_coords[0] = std::sin(phi) * std::cos(theta);
          node_coords[1] = std::sin(phi) * std::sin(theta);
          node_coords[2] = std::cos(phi);

          // Keep this point with probability proportional to the surface area element
          if (keep(node_coords[0], node_coords[1], node_coords[2])) {
            // Pushforward the point to the ellipsoid
            node_coords[0] *= a;
            node_coords[1] *= b;
            node_coords[2] *= c;
            break;
          }
        }
      }
    }
  }
};

MUNDY_METHOD(mundy::alens, DeclareAndInitializePeripheryBindSitesFromFile) {
  MUNDY_METHOD_HAS_BULKDATA();
  MUNDY_METHOD_HAS_METADATA();
  MUNDY_METHOD_HAS_OBJECT(size_t, num_bind_sites);
  MUNDY_METHOD_HAS_SELECTOR(binding_site_selector);
  MUNDY_METHOD_HAS_OBJECT(stk::mesh::Field<double>, node_coord_field);
  MUNDY_METHOD_HAS_OBJECT(std::string, bind_site_coords_filename);

 public:
  void run() override {
    // Dereference everything
    const size_t num_bind_sites = *num_bind_sites_ptr_;
    const std::string bind_site_coords_filename = *bind_site_coords_filename_ptr_;
    auto &binding_site_selector = *binding_site_selector_ptr_;
    auto &node_coord_field = *node_coord_field_ptr_;

    // Fetch the binding site parts from the selector
    stk::mesh::PartVector bs_parts;
    binding_site_selector.get_parts(bs_parts);

    // Declare the binding sites
    bulk_data_ptr_->modification_begin();
    std::vector<std::size_t> requests(meta_data_ptr_->entity_rank_count(), 0);
    if (bulk_data_ptr_->parallel_rank() == 0) {
      requests[stk::topology::NODE_RANK] = num_bind_sites;
      requests[stk::topology::ELEMENT_RANK] = num_bind_sites;
    }
    std::vector<stk::mesh::Entity> requested_entities;
    bulk_data_ptr_->generate_new_entities(requests, requested_entities);
    bulk_data_ptr_->change_entity_parts(requested_entities, stk::mesh::PartVector{bs_part_ptr_},
                                        stk::mesh::PartVector{});
    if (bulk_data_ptr_->parallel_rank() == 0) {
      for (size_t i = 0; i < num_bind_sites; i++) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[num_bind_sites + i];
        bulk_data_ptr_->declare_relation(sphere_i, node_i, 0);
      }
    }
    bulk_data_ptr_->modification_end();

    // Initialize the binding site positions
    if (bulk_data_ptr_->parallel_rank() == 0) {
      std::ifstream infile(bind_site_coords_filename, std::ios::binary);
      if (!infile) {
        std::cerr << "Failed to open file: " << bind_site_coords_filename << std::endl;
        return;
      }

      // Parse the input
      size_t num_elements;
      infile.read(reinterpret_cast<char *>(&num_elements), sizeof(size_t));
      MUNDY_THROW_ASSERT(num_elements == 3 * num_bind_sites, "Num bind sites mismatch: expected " << num_bind_sites << ", got " << num_elements / 3);
      for (size_t i = 0; i < num_bind_sites; ++i) {
        const stk::mesh::Entity &node_i = requested_entities[i];
        const stk::mesh::Entity &sphere_i = requested_entities[num_bind_sites + i];
        double *node_coords = stk::mesh::field_data(node_coord_field, node_i);
        for (size_t j = 0; j < 3; ++j) {
          infile.read(reinterpret_cast<char *>(&node_coords[3 * i + j]), sizeof(double));
        }
      }

      // Close the file
      infile.close();
    }
  }
};

MUNDY_METHOD(mundy::alens, ComputeAABBDisplacements) {
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_FIELD(double, old_element_aabb_field);
  MUNDY_METHOD_HAS_FIELD(double, new_element_aabb_field);
  MUNDY_METHOD_HAS_OBJECT(double, alpha);
  MUNDY_METHOD_HAS_OBJECT(double, beta);
  MUNDY_METHOD_HAS_FIELD(double, element_aabb_displacement_field);

 public:
  void run() override {
    // Dereference everything
    auto &selector = *selector_ptr_;
    auto &old_element_aabb_field = *old_element_aabb_field_ptr_;
    auto &new_element_aabb_field = *new_element_aabb_field_ptr_;
    const double alpha = *alpha_ptr_;
    const double beta = *beta_ptr_;
    auto &element_aabb_displacement_field = *element_aabb_displacement_field_ptr_;

    // Update the accumulators based on the difference to the previous state
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, *selector_ptr_,
        [&old_element_aabb_field, &new_element_aabb_field, &element_aabb_displacement_field, alpha, beta](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the
          // difference) into the accumulator.
          const double *element_aabb = stk::mesh::field_data(new_element_aabb_field, aabb_entity);
          const double *element_aabb_old = stk::mesh::field_data(old_element_aabb_field, aabb_entity);
          double *element_aabb_displacement = stk::mesh::field_data(element_aabb_displacement_field, aabb_entity);

          // Add the (new_aabb - old_aabb) to the corner displacement
          element_aabb_displacement[0] =
              beta * element_aabb_displacement[0] + alpha * (element_aabb[0] - element_aabb_old[0]);
          element_aabb_displacement[1] =
              beta * element_aabb_displacement[1] + alpha * (element_aabb[1] - element_aabb_old[1]);
          element_aabb_displacement[2] =
              beta * element_aabb_displacement[2] + alpha * (element_aabb[2] - element_aabb_old[2]);
          element_aabb_displacement[3] =
              beta * element_aabb_displacement[3] + alpha * (element_aabb[3] - element_aabb_old[3]);
          element_aabb_displacement[4] =
              beta * element_aabb_displacement[4] + alpha * (element_aabb[4] - element_aabb_old[4]);
          element_aabb_displacement[5] =
              beta * element_aabb_displacement[5] + alpha * (element_aabb[5] - element_aabb_old[5]);
        });
  }
};

MUNDY_METHOD(mundy::alens, ComputeMaxAABBDisplacement) {
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_FIELD(double, aabb_displacement_field);
  MUNDY_METHOD_HAS_OBJECT(double, max_aabb_displacement);

 public:
  void run() override {
    // Dereference everything
    auto &selector = *selector_ptr_;
    auto &aabb_displacement_field = *aabb_displacement_field_ptr_;
    double &max_aabb_displacement = *max_aabb_displacement_ptr_;

    double local_max_aabb_displacement = 0.0;

    stk::mesh::for_each_entity_run(
        aabb_displacement_field.get_mesh(), aabb_displacement_field.entity_rank(), selector,
        [&element_aabb_displacement_field, &local_max_aabb_displacement](const stk::mesh::BulkData &bulk_data,
                                                                         const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the
          // difference) into the accumulator.
          const double *aabb_displacement = stk::mesh::field_data(element_aabb_displacement_field, aabb_entity);

          // Compute dr2 for each corner
          double dr2_corner0 = aabb_displacement[0] * aabb_displacement[0] +
                               aabb_displacement[1] * aabb_displacement[1] +
                               aabb_displacement[2] * aabb_displacement[2];
          double dr2_corner1 = aabb_displacement[3] * aabb_displacement[3] +
                               aabb_displacement[4] * aabb_displacement[4] +
                               aabb_displacement[5] * aabb_displacement[5];

      // Update the max displacement
      // TODO(palmerb4): This should be replaced with the NGP reduction.
#pragma omp critical
          local_max_aabb_displacement = std::max(local_max_aabb_displacement, std::max(dr2_corner0, dr2_corner1));
        });

    stk::all_reduce_max(*bulk_data_ptr_, &local_max_aabb_displacement, &max_aabb_displacement, 1);
    max_aabb_displacement = std::sqrt(max_aabb_displacement);
  }

  // stk::mesh::for_each_entity_run(
  //     *bulk_data_ptr_, stk::topology::ELEMENT_RANK, combined_selector,
  //     [&local_update_neighbor_list_int, &skin_distance2_over4,
  //     &element_corner_displacement_field](
  //         [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const
  //         stk::mesh::Entity &aabb_entity) {
  //       // Get the dr for each element (should be able to just do an addition of the
  //       difference) into the accumulator. double *element_corner_displacement =
  //       stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

  //       // Compute dr2 for each corner
  //       double dr2_corner0 = element_corner_displacement[0] *
  //       element_corner_displacement[0] +
  //                            element_corner_displacement[1] *
  //                            element_corner_displacement[1] +
  //                            element_corner_displacement[2] *
  //                            element_corner_displacement[2];
  //       double dr2_corner1 = element_corner_displacement[3] *
  //       element_corner_displacement[3] +
  //                            element_corner_displacement[4] *
  //                            element_corner_displacement[4] +
  //                            element_corner_displacement[5] *
  //                            element_corner_displacement[5];

  //       if (dr2_corner0 >= skin_distance2_over4 || dr2_corner1 >= skin_distance2_over4)
  //       {
  //         local_update_neighbor_list_int = 1;
  //       }
  //     });

  // Communicate local_update_neighbor_list to all ranks. Convert to an integer first (MPI
  // doesn't handle booleans well).
  int global_update_neighbor_list_int = 0;
  MPI_Allreduce(&local_update_neighbor_list_int, &global_update_neighbor_list_int, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  // Convert back to the boolean for the global version and or it with the original value
  // (in case somebody else set the neighbor list update 'signal').
  update_neighbor_list_ = update_neighbor_list_ || (global_update_neighbor_list_int == 1);
}

MUNDY_METHOD(mundy::alens, ComputeBindLeftBoundHarmonicToSphereZPartition) {
  MUNDY_METHOD_HAS_OBJECT(double, kt);
  MUNDY_METHOD_HAS_OBJECT(double, right_binding_rate);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_hookean_spring_constant_field);
  MUNDY_METHOD_HAS_FIELD(double, element_hookean_spring_rest_length_field);
  MUNDY_METHOD_HAS_FIELD(double, constraint_state_change_rate_field);
  MUNDY_METHOD_HAS_PART(left_bound_spring_part);
  MUNDY_METHOD_HAS_PART(spring_sphere_neighbor_genx_part);

 public:
  void run() override {
    // Dereference everything
    const double kt = *kt_ptr_;
    const double right_binding_rate = *right_binding_rate_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &constraint_state_change_rate = *constraint_state_change_rate_field_ptr_;
    const auto &crosslinker_spring_constant = *element_hookean_spring_constant_field_ptr_;
    const auto &crosslinker_spring_rest_length = *element_hookean_spring_rest_length_field_ptr_;
    const auto &constraint_linked_entities_field = *constraint_linked_entities_field_ptr_;
    const auto &left_bound_spring_part = *left_bound_spring_part_ptr_;
    const auto &spring_sphere_neighbor_genx_part = *spring_sphere_neighbor_genx_part_ptr_;

    const double inv_kt = 1.0 / kt;
    stk::mesh::for_each_entity_run(
        node_coord_field.get_mesh(), stk::topology::CONSTRAINT_RANK, spring_sphere_neighbor_genx_part,
        [&node_coord_field, &constraint_linked_entities_field, &constraint_state_change_rate,
         &crosslinker_spring_constant, &crosslinker_spring_rest_length, &left_bound_spring_part, &inv_kt,
         &crosslinker_right_binding_rate]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                          const stk::mesh::Entity &neighbor_genx) {
          // Get the sphere and crosslinker attached to the linker.
          const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
              stk::mesh::field_data(constraint_linked_entities_field, neighbor_genx));
          const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
          const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);

          MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker), "Encountered invalid crosslinker entity in "
                             "compute_z_partition_left_bound_harmonic.");
          MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere), "Encountered invalid sphere entity in "
                             "compute_z_partition_left_bound_harmonic.");

          // Only act on the left-bound crosslinkers that don't self-interact
          if (bulk_data.bucket(crosslinker).member(left_bound_spring_part)) {
            const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
            is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
            if (!is_self_interaction) {
              const auto dr = mundy::mesh::vector3_field_data(node_coord_field, sphere_node) -
                              mundy::mesh::vector3_field_data(node_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
              const double dr_mag = mundy::math::norm(dr);

              // Compute the Z-partition score
              // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
              // A = crosslinker_binding_rates
              // k = crosslinker_spring_constant
              // r0 = crosslinker_spring_rest_length
              const double A = crosslinker_right_binding_rate;
              const double k = stk::mesh::field_data(crosslinker_spring_constant, crosslinker)[0];
              const double r0 = stk::mesh::field_data(crosslinker_spring_rest_length, crosslinker)[0];
              double z = A * std::exp(-0.5 * inv_kt * k * (dr_mag - r0) * (dr_mag - r0));
              stk::mesh::field_data(constraint_state_change_rate, neighbor_genx)[0] = z;
            }
          }
        });
  }
};

MUNDY_METHOD(mundy::alens, KMCSpringLeftToDoubly) {
  MUNDY_METHOD_HAS_SELECTOR(neighbor_linker_selector);
  MUNDY_METHOD_HAS_SELECTOR(left_bound_spring_selector);
  MUNDY_METHOD_HAS_OBJECT(double, timestep_size);
  MUNDY_METHOD_HAS_FIELD(unsigned, element_rng_field);
  MUNDY_METHOD_HAS_FIELD(unsigned, element_perform_state_change_field);
  MUNDY_METHOD_HAS_FIELD(unsigned, constraint_perform_state_change_field);
  MUNDY_METHOD_HAS_FIELD(double, constraint_state_change_rate_field);

 public:
  void run() override {
    // Dereference everything
    const double timestep_size = *timestep_size_ptr_;
    const auto &element_rng_field = *element_rng_field_ptr_;
    const auto &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const auto &constraint_perform_state_change_field = *constraint_perform_state_change_field_ptr_;
    const auto &constraint_state_change_rate_field = *constraint_state_change_rate_field_ptr_;
    const auto &constraint_linked_entities_field = *constraint_linked_entities_field_ptr_;
    const auto &neighbor_linker_selector = *neighbor_linker_selector_ptr_;
    const auto &left_bound_spring_selector = *left_bound_spring_selector_ptr_;

    // Loop over left-bound spring and decide if they bind or not
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, left_bound_spring_selector,
        [&neighbor_linker_selector, &element_rng_field, &constraint_perform_state_change_field,
         &element_perform_state_change_field, &constraint_state_change_rate_field, &constraint_linked_entities_field,
         &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spring) {
          // Get all of my associated linkers
          const stk::mesh::Entity &any_arbitrary_spring_node = bulk_data.begin_nodes(spring)[0];
          const stk::mesh::Entity *neighbor_genx_linkers =
              bulk_data.begin(any_arbitrary_spring_node, stk::topology::CONSTRAINT_RANK);
          const unsigned num_neighbor_genx_linkers =
              bulk_data.num_connectivity(any_arbitrary_spring_node, stk::topology::CONSTRAINT_RANK);

          // Loop over the attached linker and bind if the rng falls in their range.
          double z_tot = 0.0;
          for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
            const auto &constraint_rank_entity = neighbor_genx_linkers[j];
            const bool is_selected_neighbor_genx = neighbor_linker_selector(bulk_data.bucket(constraint_rank_entity));
            if (is_selected_neighbor_genx) {
              const double z_i =
                  timestep_size * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
              z_tot += z_i;
            }
          }

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, spring);
          const stk::mesh::EntityId spring_gid = bulk_data.identifier(spring);
          openrand::Philox rng(spring_gid, element_rng_counter[0]);
          const double randu01 = rng.rand<double>();
          element_rng_counter[0]++;

          // Notice that the sum of all probabilities is 1.
          // The probability of nothing happening is
          //   std::exp(-z_tot)
          // The probability of an individual event happening is
          //   z_i / z_tot * (1 - std::exp(-z_tot))
          //
          // This is (by construction) true since
          //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
          //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
          //
          // This means that binding only happens if randu01 < (1 - std::exp(-z_tot))
          const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
          if (randu01 < probability_of_no_state_change) {
            // Binding occurs.
            // Loop back over the neighbor linkers to see if one of them binds in the
            // running sum

            const double scale_factor = probability_of_no_state_change * timestep_size / z_tot;
            double cumsum = 0.0;
            for (unsigned j = 0; j < num_neighbor_genx_linkers; j++) {
              auto &constraint_rank_entity = neighbor_genx_linkers[j];
              const bool is_selected_neighbor_genx = neighbor_linker_selector(bulk_data.bucket(constraint_rank_entity));
              if (is_selected_neighbor_genx) {
                const double binding_probability =
                    scale_factor * stk::mesh::field_data(constraint_state_change_rate_field, constraint_rank_entity)[0];
                cumsum += binding_probability;
                if (randu01 < cumsum) {
                  // We have a binding event, set this, then bail on the for loop
                  // Store the state change on both the genx and the spring
                  stk::mesh::field_data(constraint_perform_state_change_field, constraint_rank_entity)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  stk::mesh::field_data(element_perform_state_change_field, spring)[0] =
                      static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                  break;
                }
              }
            }
          }
        });

    // At this point, constraint_state_change_rate_field is only up-to-date for
    // locally-owned entities. We need to communicate this information to all other
    // processors.
    stk::mesh::communicate_field_data(
        *bulk_data_ptr_, {element_perform_state_change_field_ptr_, constraint_perform_state_change_field_ptr_});
  }
};

MUNDY_METHOD(mundy::alens, KMCSpringDoublyToRight) {
  MUNDY_METHOD_HAS_SELECTOR(doubly_bound_spring_selector);
  MUNDY_METHOD_HAS_OBJECT(double, timestep_size);
  MUNDY_METHOD_HAS_FIELD(unsigned, element_rng_field);
  MUNDY_METHOD_HAS_FIELD(double, element_unbinding_rates_field);
  MUNDY_METHOD_HAS_FIELD(unsigned, element_perform_state_change_field);

  void run() override {
    // Dereference everything
    const double timestep_size = *timestep_size_ptr_;
    const auto &element_rng_field = *element_rng_field_ptr_;
    const auto &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const auto &element_unbinding_rates_field = *element_unbinding_rates_field_ptr_;
    const auto &doubly_bound_spring_selector = *doubly_bound_spring_selector_ptr_;

    // Note, this assumes that we only have DOUBLY->DOUBLY and DOUBLY->RIGHT transitions
    // and not DOUBLY->LEFT.
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_bound_spring_selector,
        [&element_rng_field, &element_perform_state_change_field, &element_unbinding_rates_field, &timestep_size](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spring) {
          // Only DOUBLY->RIGHT transitions are allowed (for now)
          const double right_unbinding_probability =
              timestep_size * stk::mesh::field_data(element_unbinding_rates_field, spring)[1];
          double z_tot = right_unbinding_probability;

          // Fetch the RNG state, get a random number out of it, and increment
          unsigned *element_rng_counter = stk::mesh::field_data(element_rng_field, spring);
          const stk::mesh::EntityId spring_gid = bulk_data.identifier(spring);
          openrand::Philox rng(spring_gid, element_rng_counter[0]);
          const double randu01 = rng.rand<double>();
          element_rng_counter[0]++;

          // Notice that the sum of all probabilities is 1.
          // The probability of nothing happening is
          //   std::exp(-z_tot)
          // The probability of an individual event happening is
          //   z_i / z_tot * (1 - std::exp(-z_tot))
          //
          // This is (by construction) true since
          //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
          //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
          //
          // This means that unbinding only happens if randu01 < (1 - std::exp(-z_tot))
          // For now, its either transition to right bound or nothing
          const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
          if (randu01 < probability_of_no_state_change) {
            stk::mesh::field_data(element_perform_state_change_field, spring)[0] =
                static_cast<unsigned>(BINDING_STATE_CHANGE::DOUBLY_TO_LEFT);
          }
        });

    // At this point, state change field is only up-to-date for locally-owned entities. We
    // need to communicate this information to all other processors.
    stk::mesh::communicate_field_data(*bulk_data_ptr_, {*element_perform_state_change_field});
  }
};

MUNDY_METHOD(mundy::alens, PerformStateChangeDynamicSprings) {
  MUNDY_METHOD_HAS_SELECTOR(left_bound_spring_selector);
  MUNDY_METHOD_HAS_SELECTOR(doubly_bound_spring_selector);
  MUNDY_METHOD_HAS_SELECTOR(spring_sphere_genx_selector);
  MUNDY_METHOD_HAS_FIELD(unsigned, element_perform_state_change_field);
  MUNDY_METHOD_HAS_FIELD(unsigned, constraint_perform_state_change_field);

 public:
  void run() override {
    // Dereference everything
    const auto &left_bound_spring_selector = *left_bound_spring_selector_ptr_;
    const auto &doubly_bound_spring_selector = *doubly_bound_spring_selector_ptr_;
    const auto &spring_sphere_genx_selector = *spring_sphere_genx_selector_ptr_;
    const auto &element_perform_state_change_field = *element_perform_state_change_field_ptr_;
    const auto &constraint_perform_state_change_field = *constraint_perform_state_change_field_ptr_;
    const auto &bulk_data = element_perform_state_change_field.get_mesh();

    // Get the vector of left/right bound parts in the selector
    stk::mesh::PartVector left_bound_spring_parts;
    stk::mesh::PartVector doubly_bound_spring_parts;
    left_bound_spring_selector.get_parts(left_bound_spring_parts);
    doubly_bound_spring_selector.get_parts(doubly_bound_spring_parts);

    // Get the vector of entities to modify
    stk::mesh::EntityVector spring_sphere_linkers;
    stk::mesh::EntityVector doubly_bound_springs;
    stk::mesh::get_selected_entities(spring_sphere_genx_selector, bulk_data.buckets(stk::mesh::CONSTRAINT_RANK),
                                     spring_sphere_linkers);
    stk::mesh::get_selected_entities(doubly_bound_spring_selector, bulk_data.buckets(stk::mesh::ELEMENT_RANK),
                                     doubly_bound_springs);

    bulk_data.modification_begin();

    // Perform L->D
    for (const stk::mesh::Entity &spring_sphere_genx : spring_sphere_linkers) {
      // Decode the binding type enum for this entity
      auto state_change_action = static_cast<BINDING_STATE_CHANGE>(
          stk::mesh::field_data(constraint_perform_state_change_field, spring_sphere_genx)[0]);
      const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
      if (perform_state_change) {
        // Get our connections
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(constraint_linked_entities_field, spring_sphere_genx));
        const stk::mesh::Entity &spring = bulk_data.get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &target_sphere = bulk_data.get_entity(key_t_ptr[1]);
        MUNDY_THROW_ASSERT(bulk_data.is_valid(spring), "Encountered invalid crosslinker entity in state_change_crosslinkers.");
        MUNDY_THROW_ASSERT(bulk_data.is_valid(target_sphere), "Encountered invalid sphere entity in state_change_crosslinkers.");

        // Call the binding function
        if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
          // Unbind the right side of the crosslinker from the left node and bind it to
          // the target node
          const stk::mesh::Entity &target_sphere_node = bulk_data.begin_nodes(target_sphere)[0];
          const bool bind_worked =
              bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, spring, target_sphere_node, 1);
          MUNDY_THROW_ASSERT(bind_worked, "Failed to bind crosslinker to node.");

          std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Binding crosslinker "
                    << bulk_data.identifier(spring) << " to node " << bulk_data.identifier(target_sphere_node)
                    << std::endl;

          // Now change the part from left to doubly bound. Add to doubly bound, remove
          // from left bound
          const bool is_spring_locally_owned = bulk_data.parallel_owner_rank(spring) == bulk_data.parallel_rank();
          if (is_spring_locally_owned) {
            bulk_data.change_entity_parts(spring, doubly_bound_spring_parts, left_bound_spring_parts);
          }
        }
      }
    }

    // Perform D->L
    for (const stk::mesh::Entity &spring : doubly_bound_springs) {
      // Decode the binding type enum for this entity
      auto state_change_action =
          static_cast<BINDING_STATE_CHANGE>(stk::mesh::field_data(*element_perform_state_change_field_ptr_, spring)[0]);
      if (state_change_action == BINDING_STATE_CHANGE::DOUBLY_TO_LEFT) {
        // Unbind the right side of the crosslinker from the current node and bind it to
        // the left crosslinker node
        const stk::mesh::Entity &left_node = bulk_data.begin_nodes(spring)[0];
        const bool unbind_worked = bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, spring, left_node, 1);
        MUNDY_THROW_ASSERT(unbind_worked, "Failed to unbind crosslinker from node.");

        std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Unbinding crosslinker "
                  << bulk_data.identifier(spring) << " from node "
                  << bulk_data.identifier(bulk_data.begin_nodes(spring)[1]) << std::endl;

        // Now change the part from doubly to left bound. Add to left bound, remove from
        // doubly bound
        const bool is_spring_locally_owned = bulk_data.parallel_owner_rank(spring) == bulk_data.parallel_rank();
        if (is_spring_locally_owned) {
          bulk_data.change_entity_parts(crosslinker_hp1, left_bound_spring_parts, doubly_bound_spring_parts);
        }
      }
    }

    bulk_data.modification_end();
  }
};

MUNDY_METHOD(mundy::alens, CheckMaxOverlapWithSphericalPeriphery) {
  MUNDY_METHOD_HAS_OBJECT(double, target_sphere_radius);
  MUNDY_METHOD_HAS_OBJECT(std::array<double, 3>, target_sphere_center);
  MUNDY_METHOD_HAS_OBJECT(bool, invert_target_sphere);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_radius_field);
  MUNDY_METHOD_HAS_SELECTOR(spheres_selector);
  MUNDY_METHOD_HAS_OBJECT(double, max_allowable_overlap);

 public:
  void run() override {
    // Dereference everything
    const double target_sphere_radius = *target_sphere_radius_ptr_;
    const std::array<double, 3> &target_sphere_center = *target_sphere_center_ptr_;
    const bool invert_target_sphere = *invert_target_sphere_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &element_radius_field = *element_radius_field_ptr_;
    const auto &spheres_selector = *spheres_selector_ptr_;
    const double max_allowable_overlap = *max_allowable_overlap_ptr_;

    double local_max_overlap = 0.0;
    const double sign = invert_target_sphere ? -1.0 : 1.0;
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_selector,
        [&node_coord_field, &element_radius_field, &target_sphere_center, &target_sphere_radius, &sign,
         &local_max_overlap](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
          const double ssd =
              sign * mundy::math::norm(node_coords - target_sphere_center) - sphere_radius - target_sphere_radius;

#pragma omp critical
          local_min_ssd = std::min(local_min_ssd, ssd);
        });

    double min_ssd = std::numeric_limits<double>::max();
    stk::all_reduce_min(*bulk_data_ptr_, &local_min_ssd, &min_ssd, 1);
    if (min_ssd < -max_allowable_overlap) {
      MUNDY_THROW_ASSERT(false, "Sphere overlaps with hydrodynamic periphery more than the allowabe amount.");
    }
  }
};

MUNDY_METHOD(mundy::alens, CheckMaxOverlapWithEllipsoidalPeripheryFastApprox) {
  MUNDY_METHOD_HAS_OBJECT(double, target_ellipsoid_r1);
  MUNDY_METHOD_HAS_OBJECT(double, target_ellipsoid_r2);
  MUNDY_METHOD_HAS_OBJECT(double, target_ellipsoid_r3);
  MUNDY_METHOD_HAS_OBJECT(std::array<double, 3>, target_ellipsoid_center);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_radius_field);
  MUNDY_METHOD_HAS_SELECTOR(spheres_selector);
  MUNDY_METHOD_HAS_OBJECT(double, max_allowable_overlap);

 public:
  void run() override {
    std::cout << "###### WARNING #######" << std::endl;
    std::cout << "The fast approximation for ellipsoidal periphery ssd " << std::endl;
    std::cout << "  is UNFIT for use in scientific publications." << std::endl;
    std::cout << "###### WARNING #######" << std::endl;

    // Dereference everything
    const double target_ellipsoid_r1 = *target_ellipsoid_r1_ptr_;
    const double target_ellipsoid_r2 = *target_ellipsoid_r2_ptr_;
    const double target_ellipsoid_r3 = *target_ellipsoid_r3_ptr_;
    const std::array<double, 3> &target_ellipsoid_center = *target_ellipsoid_center_ptr_;
    const bool invert_target_ellipsoid = *invert_target_ellipsoid_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &element_radius_field = *element_radius_field_ptr_;
    const auto &spheres_selector = *spheres_selector_ptr_;
    const double max_allowable_overlap = *max_allowable_overlap_ptr_;

    const double shifted_target_ellipsoid_r1 = target_ellipsoid_r1 + max_allowable_overlap;
    const double shifted_target_ellipsoid_r2 = target_ellipsoid_r2 + max_allowable_overlap;
    const double shifted_target_ellipsoid_r3 = target_ellipsoid_r3 + max_allowable_overlap;

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres_selector,
        [&node_coord_field, &element_hydro_radius_field, &shifted_target_ellipsoid_r1, &shifted_target_ellipsoid_r2,
         &shifted_target_ellipsoid_r3](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double sphere_radius = stk::mesh::field_data(element_hydro_radius_field, sphere_element)[0];

          // The following is an in-exact but cheap check.
          // If shrinks the ellipsoid by the maximum allowed overlap and the sphere radius
          // and then checks if the sphere is inside the shrunk ellipsoid. Level sets
          // don't follow the same rules as Euclidean geometry, so this is a rough check.
          const double x = node_coords[0];
          const double y = node_coords[1];
          const double z = node_coords[2];
          const double x2 = x * x;
          const double y2 = y * y;
          const double z2 = z * z;
          const double a2 =
              (shifted_target_ellipsoid_r1 - sphere_radius) * (shifted_target_ellipsoid_r1 - sphere_radius);
          const double b2 =
              (shifted_target_ellipsoid_r2 - sphere_radius) * (shifted_target_ellipsoid_r2 - sphere_radius);
          const double c2 =
              (shifted_target_ellipsoid_r3 - sphere_radius) * (shifted_target_ellipsoid_r3 - sphere_radius);
          const double value = x2 / a2 + y2 / b2 + z2 / c2;
          if (value > 1.0) {
#pragma omp critical
            {
              std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                        << " overlaps with the periphery more than the allowable threshold." << std::endl;
              std::cout << "  node_coords: " << node_coords << std::endl;
              std::cout << "  value: " << value << std::endl;
            }
            MUNDY_THROW_ASSERT(false, "Sphere node outside hydrodynamic periphery.");
          }
        });
  }
};

MUNDY_METHOD(ComputeRPYHydro) {
  MUNDY_METHOD_HAS_SELECTOR(spheres_selector);
  MUNDY_METHOD_HAS_OBJECT(double, viscosity);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_radius_field);
  MUNDY_METHOD_HAS_FIELD(double, node_force_field);
  MUNDY_METHOD_HAS_FIELD(double, node_velocity_field);
  MUNDY_METHOD_HAS_OBJECT(mundy::alens::Periphery,
                          periphery);  // Optional. If given, will apply no-slip boundary
  MUNDY_METHOD_HAS_OBJECT(bool,
                          validate_noslip_boundary);  // Optional. If true, will check if
                                                      // the no-slip boundary is satisfied

 public:
  void run() override {
    // Dereference everything
    const double viscosity = *viscosity_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &element_radius_field = *element_radius_field_ptr_;
    const auto &node_force_field = *node_force_field_ptr_;
    const auto &node_velocity_field = *node_velocity_field_ptr_;
    const auto &spheres_selector = *spheres_selector_ptr_;
    const bool enable_periphery = periphery_ptr_ != nullptr;
    const bool validate_noslip_boundary = (validate_noslip_boundary_ptr_ != nullptr) && *validate_noslip_boundary_ptr_;

    // Fetch the bucket of spheres to act on.
    stk::mesh::EntityVector sphere_elements;
    stk::mesh::get_selected_entities(spheres_selector, bulk_data_ptr_->buckets(stk::topology::ELEMENT_RANK),
                                     sphere_elements);
    const size_t num_spheres = sphere_elements.size();

    // Copy the sphere positions, radii, forces, and velocities to Kokkos views
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions("sphere_positions", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii("sphere_radii", num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces("sphere_forces", num_spheres * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities("sphere_velocities",
                                                                                    num_spheres * 3);

#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      const double *sphere_position = stk::mesh::field_data(*node_coord_field_ptr_, sphere_node);
      const double *sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);
      const double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      const double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_positions(i * 3 + j) = sphere_position[j];
        sphere_forces(i * 3 + j) = sphere_force[j];
        sphere_velocities(i * 3 + j) = sphere_velocity[j];
      }
      sphere_radii(i) = *sphere_radius;
    }

    // Apply the RPY kernel from spheres to spheres
    mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                              sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

    // If enabled, apply the correction for the no-slip boundary condition
    if (enable_periphery) {
      const auto &periphery = *periphery_ptr_;
      const size_t num_surface_nodes = periphery.get_num_nodes();
      auto surface_positions = periphery.get_surface_positions();
      auto surface_weights = periphery.get_quadrature_weights();
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_radii("surface_radii", num_surface_nodes);
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_velocities("surface_velocities",
                                                                                       3 * num_surface_nodes);
      Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_forces("surface_forces",
                                                                                   3 * num_surface_nodes);
      Kokkos::deep_copy(surface_radii, 0.0);

      // Apply the RPY kernel from spheres to periphery
      mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, surface_positions,
                                                sphere_radii, surface_radii, sphere_forces, surface_velocities);

      // Apply no-slip boundary conditions
      // This is done in two steps: first, we compute the forces on the periphery
      // necessary to enforce no-slip Then we evaluate the flow these forces induce on the
      // spheres.
      periphery.compute_surface_forces(surface_velocities, surface_forces);

      if (validate_noslip_boundary) {
        // If we evaluate the flow these forces induce on the periphery, do they actually
        // satisfy no-slip?
        Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> M("Mnew", 3 * num_surface_nodes,
                                                                         3 * num_surface_nodes);
        fill_skfie_matrix(DeviceExecutionSpace(), viscosity, num_surface_nodes, num_surface_nodes, surface_positions,
                          surface_positions, surface_normals, surface_weights, M);
        KokkosBlas::gemv(DeviceExecutionSpace(), "N", 1.0, M, surface_forces, 1.0, surface_velocities);
        MUNDY_THROW_ASSERT(max_speed(surface_velocities) < 1.0e-10, "No-slip boundary condition not satisfied.");
      }

      mundy::alens::periphery::apply_weighted_stokes_kernel(DeviceExecutionSpace(), viscosity, surface_positions,
                                                            sphere_positions, surface_forces, surface_weights,
                                                            sphere_velocities);

      // The RPY kernel is only long-range, it doesn't add on self-interaction for the
      // spheres
      mundy::alens::periphery::apply_local_drag(DeviceExecutionSpace(), viscosity, sphere_velocities, sphere_forces,
                                                sphere_radii);
    }

    // Copy the sphere forces and velocities back to STK fields
#pragma omp parallel for
    for (size_t i = 0; i < num_spheres; i++) {
      stk::mesh::Entity sphere_element = sphere_elements[i];
      stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
      double *sphere_force = stk::mesh::field_data(*node_force_field_ptr_, sphere_node);
      double *sphere_velocity = stk::mesh::field_data(*node_velocity_field_ptr_, sphere_node);

      for (size_t j = 0; j < 3; j++) {
        sphere_force[j] = sphere_forces(i * 3 + j);
        sphere_velocity[j] = sphere_velocities(i * 3 + j);
      }
    }
  }
};

MUNDY_METHOD(mundy::alens, ComputeEllipsoidalPeripheryCollisionForcesWithSpheres) {
  MUNDY_METHOD_HAS_SELECTOR(spheres_selector);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r1);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r2);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r3);
  MUNDY_METHOD_HAS_OBJECT(std::array<double, 3>, periphery_center);
  MUNDY_METHOD_HAS_OBJECT(mundy::math::Quaternion<double>, periphery_orientation);
  MUNDY_METHOD_HAS_OBJECT(double, collision_spring_constant);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_radius_field);
  MUNDY_METHOD_HAS_FIELD(double, node_force_field);

 public:
  void run() override {
    // Dereference everything
    const double collision_spring_constant = *collision_spring_constant_ptr_;
    const double periphery_r1 = *periphery_r1_ptr_;
    const double periphery_r2 = *periphery_r2_ptr_;
    const double periphery_r3 = *periphery_r3_ptr_;
    const auto &periphery_center = *periphery_center_ptr_;
    const auto &periphery_orientation = *periphery_orientation_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &element_radius_field = *element_radius_field_ptr_;
    const auto &node_force_field = *node_force_field_ptr_;
    const auto &spheres_selector = *spheres_selector_ptr_;

    // Setup the level set function for the ellipsoidal periphery
    const double inv_periphery_r1_sq = 1.0 / (periphery_r1 * periphery_r1);
    const double inv_periphery_r2_sq = 1.0 / (periphery_r2 * periphery_r2);
    const double inv_periphery_r3_sq = 1.0 / (periphery_r3 * periphery_r3);
    auto level_set = [&periphery_center, &periphery_orientation, &inv_periphery_r1_sq, &inv_periphery_r2_sq,
                      &inv_periphery_r3_sq](const mundy::math::Vector3<double> &point) -> double {
      const auto body_frame_point = conjugate(periphery_orientation) * (point - periphery_center);
      return (body_frame_point[0] * body_frame_point[0] * inv_periphery_r1_sq +
              body_frame_point[1] * body_frame_point[1] * inv_periphery_r2_sq +
              body_frame_point[2] * body_frame_point[2] * inv_periphery_r3_sq) -
             1;
    };

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_selector,
        [&node_coord_field, &node_force_field, &element_aabb_field, &element_radius_field, &level_set, &center,
         &orientation, &periphery_r1, &periphery_r2, &periphery_r3,
         &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // For our coarse search, we check if the coners of the sphere's aabb lie inside
          // the ellipsoidal periphery This can be done via the (body frame) inside
          // outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2 + z^2/c^2) This is
          // possible due to the convexity of the ellipsoid
          const double *sphere_aabb = stk::mesh::field_data(element_aabb_field, sphere_element);
          const double x0 = sphere_aabb[0];
          const double y0 = sphere_aabb[1];
          const double z0 = sphere_aabb[2];
          const double x1 = sphere_aabb[3];
          const double y1 = sphere_aabb[4];
          const double z1 = sphere_aabb[5];

          // Compute all 8 corners of the AABB
          const auto bottom_left_front = mundy::math::Vector3<double>(x0, y0, z0);
          const auto bottom_right_front = mundy::math::Vector3<double>(x1, y0, z0);
          const auto top_left_front = mundy::math::Vector3<double>(x0, y1, z0);
          const auto top_right_front = mundy::math::Vector3<double>(x1, y1, z0);
          const auto bottom_left_back = mundy::math::Vector3<double>(x0, y0, z1);
          const auto bottom_right_back = mundy::math::Vector3<double>(x1, y0, z1);
          const auto top_left_back = mundy::math::Vector3<double>(x0, y1, z1);
          const auto top_right_back = mundy::math::Vector3<double>(x1, y1, z1);
          const double all_points_inside_periphery =
              level_set(bottom_left_front) < 0.0 && level_set(bottom_right_front) < 0.0 &&
              level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 &&
              level_set(bottom_left_back) < 0.0 && level_set(bottom_right_back) < 0.0 &&
              level_set(top_left_back) < 0.0 && level_set(top_right_back) < 0.0;

          if (!all_points_inside_periphery) {
            // We might have a collision, perform the more expensive check
            const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
            const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
            const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

            // Note, the ellipsoid for the ssd calc has outward normal, whereas the
            // periphery has inward normal. Hence, the sign flip.
            mundy::math::Vector3<double> contact_point;
            mundy::math::Vector3<double> ellipsoid_nhat;
            const double shared_normal_ssd = -mundy::math::distance::shared_normal_ssd_between_ellipsoid_and_point(
                                                 periphery_center, periphery_orientation, periphery_r1, periphery_r2,
                                                 periphery_r3, node_coords, &contact_point, &ellipsoid_nhat) -
                                             sphere_radius;

            if (shared_normal_ssd < 0.0) {
              // We have a collision, compute the force
              auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);
              auto periphery_nhat = -ellipsoid_nhat;
              node_force[0] -= spring_constant * periphery_nhat[0] * shared_normal_ssd;
              node_force[1] -= spring_constant * periphery_nhat[1] * shared_normal_ssd;
              node_force[2] -= spring_constant * periphery_nhat[2] * shared_normal_ssd;
            }
          }
        });
  }
};

MUNDY_METHOD(mundy::alens, ComputeEllipsoidalPeripheryCollisionForcesWithSpheresFastApprox) {
  MUNDY_METHOD_HAS_SELECTOR(spheres_selector);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r1);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r2);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_r3);
  MUNDY_METHOD_HAS_OBJECT(std::array<double, 3>, periphery_center);
  MUNDY_METHOD_HAS_OBJECT(mundy::math::Quaternion<double>, periphery_orientation);
  MUNDY_METHOD_HAS_OBJECT(double, collision_spring_constant);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_radius_field);
  MUNDY_METHOD_HAS_FIELD(double, node_force_field);

 public:
  void run() override {
    std::cout << "###### WARNING #######" << std::endl;
    std::cout << "The fast approximation for ellipsoidal periphery collision " << std::endl;
    std::cout << "  is UNFIT for use in scientific publications." << std::endl;
    std::cout << "###### WARNING #######" << std::endl;

    // Dereference everything
    const double collision_spring_constant = *collision_spring_constant_ptr_;
    const double periphery_r1 = *periphery_r1_ptr_;
    const double periphery_r2 = *periphery_r2_ptr_;
    const double periphery_r3 = *periphery_r3_ptr_;
    const std::array<double, 3> &periphery_center = *periphery_center_ptr_;
    const auto &periphery_orientation = *periphery_orientation_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &element_radius_field = *element_radius_field_ptr_;
    const auto &node_force_field = *node_force_field_ptr_;
    const auto &spheres_selector = *spheres_selector_ptr_;

    // Setup the level set function for the ellipsoidal periphery shifted inward by the
    // sphere radius
    auto level_set = [&periphery_r1, &periphery_r2, &periphery_r3, &periphery_center, &periphery_orientation](
                         const double &radius, const mundy::math::Vector3<double> &point) -> double {
      const auto body_frame_point = conjugate(periphery_orientation) * (point - periphery_center);
      const double inv_a2 = 1.0 / ((periphery_r1 - radius) * (periphery_r1 - radius));
      const double inv_b2 = 1.0 / ((periphery_r2 - radius) * (periphery_r2 - radius));
      const double inv_c2 = 1.0 / ((periphery_r3 - radius) * (periphery_r3 - radius));
      return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
              body_frame_point[2] * body_frame_point[2] * inv_c2) -
             1;
    };

    // Setup the outward normal function for the ellipsoidal periphery
    auto outward_normal = [&periphery_r1, &periphery_r2, &periphery_r3, &periphery_center, &periphery_orientation](
                              const double &radius,
                              const mundy::math::Vector3<double> &point) -> mundy::math::Vector3<double> {
      const auto body_frame_point = conjugate(periphery_orientation) * (point - periphery_center);
      const double inv_a2 = 1.0 / ((periphery_r1 - radius) * (periphery_r1 - radius));
      const double inv_b2 = 1.0 / ((periphery_r2 - radius) * (periphery_r2 - radius));
      const double inv_c2 = 1.0 / ((periphery_r3 - radius) * (periphery_r3 - radius));
      return periphery_orientation * mundy::math::Vector3<double>(2.0 * body_frame_point[0] * inv_a2,
                                                                  2.0 * body_frame_point[1] * inv_b2,
                                                                  2.0 * body_frame_point[2] * inv_c2);
    };

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_selector,
        [&node_coord_field, &node_force_field, &element_radius_field, &level_set, &outward_normal,
         &collision_spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          // Do a fast loop over all of the spheres we are checking, e.g., brute-force the
          // calc.
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

          // Simply check if we are outside the sphere via the level-set function
          if (level_set(sphere_radius, node_coords) > 0.0) {
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);

            // Compute the outward normal
            auto out_normal = outward_normal(sphere_radius, node_coords);
            node_force[0] -= collision_spring_constant * out_normal[0];
            node_force[1] -= collision_spring_constant * out_normal[1];
            node_force[2] -= collision_spring_constant * out_normal[2];
          }
        });
  }
};

MUNDY_METHOD(mundy::alens, ComputeSphericalPeripheryCollisionForcesWithSpheres) {
  MUNDY_METHOD_HAS_SELECTOR(spheres_selector);
  MUNDY_METHOD_HAS_OBJECT(double, periphery_radius);
  MUNDY_METHOD_HAS_OBJECT(std::array<double, 3>, periphery_center);
  MUNDY_METHOD_HAS_OBJECT(double, collision_spring_constant);
  MUNDY_METHOD_HAS_FIELD(double, node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, element_radius_field);

 public:
  void run() override {
    // Dereference everything
    const double collision_spring_constant = *collision_spring_constant_ptr_;
    const double periphery_radius = *periphery_radius_ptr_;
    const std::array<double, 3> &periphery_center = *periphery_center_ptr_;
    const auto &node_coord_field = *node_coord_field_ptr_;
    const auto &element_radius_field = *element_radius_field_ptr_;
    const auto &spheres_selector = *spheres_selector_ptr_;

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres_selector,
        [&node_coord_field, &element_radius_field, &periphery_center, &periphery_radius, &collision_spring_constant](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(node_coord_field, sphere_node);
          const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
          const double shared_normal_ssd =
              periphery_radius - mundy::math::norm(node_coords - periphery_center) - sphere_radius;
          const bool sphere_collides_with_periphery = shared_normal_ssd < 0.0;
          if (sphere_collides_with_periphery) {
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, sphere_node);
            auto inward_normal = (node_coords - periphery_center) / mundy::math::norm(node_coords - periphery_center);
            node_force[0] -= collision_spring_constant * inward_normal[0] * shared_normal_ssd;
            node_force[1] -= collision_spring_constant * inward_normal[1] * shared_normal_ssd;
            node_force[2] -= collision_spring_constant * inward_normal[2] * shared_normal_ssd;
          }
        });
  }
};

MUNDY_METHOD(mundy::alens, ComputeBrownianVelocitySpheres) {
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_OBJECT(double, timestep_size);
  MUNDY_METHOD_HAS_OBJECT(double, viscosity);
  MUNDY_METHOD_HAS_OBJECT(double, brownian_kt);
  MUNDY_METHOD_HAS_OBJECT(double, sphere_radius);
  MUNDY_METHOD_HAS_FIELD(unsigned, node_rng_field);
  MUNDY_METHOD_HAS_FIELD(double, node_velocity_field);
  MUNDY_METHOD_HAS_FIELD(double, node_force_field);

 public:
  void run() override {
    // Dereference everything
    const auto &selector = *selector_ptr_;
    const double &viscosity = *viscosity_ptr_;
    const double &brownian_kt = *brownian_kt_ptr_;
    const double &sphere_radius = *sphere_radius_ptr_;
    const auto &node_rng_field = *node_rng_field_ptr_;
    const auto &node_velocity_field = *node_velocity_field_ptr_;
    const auto &node_force_field = *node_force_field_ptr_;
    const double &timestep_size = *timestep_size_ptr_;

    const double sphere_drag_coeff = 6.0 * M_PI * viscosity * sphere_radius;
    const double inv_drag_coeff = 1.0 / sphere_drag_coeff;
    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, selector,
        [&node_velocity_field, &node_force_field, &node_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff,
         &brownian_kt](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
          // Get the specific values for each sphere
          double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
          const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
          unsigned *node_rng_counter = stk::mesh::field_data(node_rng_field, sphere_node);

          // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
          openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
          const double coeff = std::sqrt(2.0 * brownian_kt * sphere_drag_coeff / timestep_size) * inv_drag_coeff;
          node_velocity[0] += coeff * rng.randn<double>();
          node_velocity[1] += coeff * rng.randn<double>();
          node_velocity[2] += coeff * rng.randn<double>();
          node_rng_counter[0]++;
        });
  }
};

MUNDY_METHOD(mundy::alens, ComputeDryLocalDragSpheres) {
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_OBJECT(double, viscosity);
  MUNDY_METHOD_HAS_OBJECT(double, sphere_radius);
  MUNDY_METHOD_HAS_FIELD(double, node_force_field);
  MUNDY_METHOD_HAS_FIELD(double, node_velocity_field);

 public:
  void run() override {
    // Dereference everything
    const auto &selector = *selector_ptr_;
    const double &viscosity = *viscosity_ptr_;
    const double &sphere_radius = *sphere_radius_ptr_;
    const auto &node_force_field = *node_force_field_ptr_;
    const auto &node_velocity_field = *node_velocity_field_ptr_;

    const double inv_drag_coeff = 1.0 / (6.0 * M_PI * viscosity * sphere_radius);
    stk::mesh::for_each_entity_run(*bulk_data_ptr_, stk::topology::NODE_RANK, selector,
                                   [&node_force_field, &node_velocity_field, &inv_drag_coeff](
                                       const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
                                     double *node_velocity = stk::mesh::field_data(node_velocity_field, sphere_node);
                                     double *node_force = stk::mesh::field_data(node_force_field, sphere_node);

                                     // Uext = Fext * inv_drag_coeff
                                     node_velocity[0] += node_force[0] * inv_drag_coeff;
                                     node_velocity[1] += node_force[1] * inv_drag_coeff;
                                     node_velocity[2] += node_force[2] * inv_drag_coeff;
                                   });
  }
};

MUNDY_METHOD(mundy::alens, NodeEulerPositionIntegrator) {
  MUNDY_METHOD_HAS_SELECTOR(selector);
  MUNDY_METHOD_HAS_OBJECT(double, timestep_size);
  MUNDY_METHOD_HAS_FIELD(double, old_node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, new_node_coord_field);
  MUNDY_METHOD_HAS_FIELD(double, node_velocity_field);

 public:
  void run() override {
    // Dereference everything
    const auto &selector = *selector_ptr_;
    const double &timestep_size = *timestep_size_ptr_;
    const auto &old_node_coord_field = *old_node_coord_field_ptr_;
    const auto &new_node_coord_field = *new_node_coord_field_ptr_;
    const auto &node_velocity_field = *node_velocity_field_ptr_;

    stk::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::NODE_RANK, selector,
        [&old_node_coord_field, &new_node_coord_field, &node_velocity_field, &timestep_size](
            const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
          const auto old_node_coords = mundy::mesh::vector3_field_data(old_node_coord_field, node);
          const auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node);
          auto new_node_coords = mundy::mesh::vector3_field_data(new_node_coord_field, node);

          // x(t+dt) = x(t) + dt * v(t)
          new_node_coords[0] = old_node_coords[0] + timestep_size * node_velocity[0];
          new_node_coords[1] = old_node_coords[1] + timestep_size * node_velocity[1];
          new_node_coords[2] = old_node_coords[2] + timestep_size * node_velocity[2];
        });
  }
};

class RcbSettings : public stk::balance::BalanceSettings {
 public:
  RcbSettings() {
  }
  virtual ~RcbSettings() {
  }

  virtual bool isIncrementalRebalance() const {
    return false;
  }
  virtual std::string getDecompMethod() const {
    return std::string("rcb");
  }
  virtual std::string getCoordinateFieldName() const {
    return std::string("NODE_COORDS");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings

Teuchos::ParameterList get_valid_hp1_params() {
  // Create a paramater entity validator for our large integers to allow for both int and
  // long long.
  auto prefer_size_t = []() {
    if (std::is_same_v<size_t, unsigned short>) {
      return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_SHORT;
    } else if (std::is_same_v<size_t, unsigned int>) {
      return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
    } else if (std::is_same_v<size_t, unsigned long>) {
      return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG;
    } else if (std::is_same_v<size_t, unsigned long long>) {
      return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG_LONG;
    } else {
      throw std::runtime_error("Unknown size_t type.");
      return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
    }
  }();
  const bool allow_all_types_by_default = false;
  mundy::core::OurAnyNumberParameterEntryValidator::AcceptedTypes accept_int(allow_all_types_by_default);
  accept_int.allow_all_integer_types(true);
  auto make_new_validator = [](const auto &preferred_type, const auto &accepted_types) {
    return Teuchos::rcp(new mundy::core::OurAnyNumberParameterEntryValidator(preferred_type, accepted_types));
  };

  // Default values are hard-coded. Trust me, this is the clearest way.
  static Teuchos::ParameterList valid_parameter_list;
  valid_parameter_list.sublist("simulation")
      .set("num_time_steps", 100, "Number of time steps.", make_new_validator(prefer_size_t, accept_int))
      .set("timestep_size", 0.001, "Time step size.")
      .set("viscosity", 1.0, "Viscosity.")
      .set("num_chromosomes", 1, "Number of chromosomes.", make_new_validator(prefer_size_t, accept_int))
      .set("num_chromatin_repeats", 2, "Number of chromatin repeats per chain.",
           make_new_validator(prefer_size_t, accept_int))
      .set("num_euchromatin_per_repeat", 1, "Number of euchromatin beads per repeat.",
           make_new_validator(prefer_size_t, accept_int))
      .set("num_heterochromatin_per_repeat", 1, "Number of heterochromatin beads per repeat.",
           make_new_validator(prefer_size_t, accept_int))
      .set("backbone_sphere_hydrodynamic_radius", 0.05,
           "Backbone sphere hydrodynamic radius. Even if n-body hydrodynamics is "
           "disabled, we still have "
           "self-interaction.")
      .set("initial_chromosome_separation", 1.0, "Initial chromosome separation.")
      .set("initialization_type", std::string("GRID"), "Initialization_type.")
      .set<Teuchos::Array<double>>("unit_cell_size", Teuchos::tuple<double>(10.0, 10.0, 10.0),
                                   "Unit cell size in each dimension. (Only used if "
                                   "initialization_type involves a 'UNIT_CELL').")
      .set("check_maximum_speed_pre_position_update", false, "Check maximum speed before updating positions.")
      .set("max_allowable_speed", std::numeric_limits<double>::max(),
           "Maximum allowable speed (only used if "
           "check_maximum_speed_pre_position_update is true).")
      // IO
      .set("loadbalance_post_initialization", false, "If we should load balance post-initialization or not.")
      .set("io_frequency", 10, "Number of timesteps between writing output.",
           make_new_validator(prefer_size_t, accept_int))
      .set("log_frequency", 10, "Number of timesteps between logging.", make_new_validator(prefer_size_t, accept_int))
      .set("output_filename", std::string("HP1"), "Output filename.")
      .set("enable_continuation_if_available", true,
           "Enable continuing a previous simulation if an output file already exists.")
      // Control flags
      .set("enable_chromatin_brownian_motion", true, "Enable chromatin Brownian motion.")
      .set("enable_backbone_springs", true, "Enable backbone springs.")
      .set("enable_backbone_collision", true, "Enable backbone collision.")
      .set("enable_backbone_n_body_hydrodynamics", true, "Enable backbone N-body hydrodynamics.")
      .set("enable_crosslinkers", true, "Enable crosslinkers.")
      .set("enable_periphery_collision", true, "Enable periphery collision.")
      .set("enable_periphery_hydrodynamics", true, "Enable periphery hydrodynamics.")
      .set("enable_periphery_binding", true, "Enable periphery binding.");

  valid_parameter_list.sublist("brownian_motion").set("kt", 1.0, "Temperature kT for Brownian Motion.");

  valid_parameter_list.sublist("backbone_springs")
      .set("spring_type", std::string("HARMONIC"), "Chromatin spring type.")
      .set("spring_constant", 100.0, "Chromatin spring constant.")
      .set("spring_rest_length", 1.0, "Chromatin rest length.");

  valid_parameter_list.sublist("backbone_collision")
      .set("excluded_volume_radius", 0.5, "Backbone excluded volume radius.")
      .set("youngs_modulus", 1000.0, "Backbone Young's modulus.")
      .set("poissons_ratio", 0.3, "Backbone Poisson's ratio.");

  valid_parameter_list.sublist("crosslinker")
      .set("spring_type", std::string("HARMONIC"), "Crosslinker spring type.")
      .set("kt", 1.0, "Temperature kT for crosslinkers.")
      .set("spring_constant", 10.0, "Crosslinker spring constant.")
      .set("rest_length", 2.5, "Crosslinker rest length.")
      .set("left_binding_rate", 1.0, "Crosslinker left binding rate.")
      .set("right_binding_rate", 1.0, "Crosslinker right binding rate.")
      .set("left_unbinding_rate", 1.0, "Crosslinker left unbinding rate.")
      .set("right_unbinding_rate", 1.0, "Crosslinker right unbinding rate.");

  valid_parameter_list.sublist("periphery_hydro")
      .set("check_maximum_periphery_overlap", false, "Check maximum periphery overlap.")
      .set("maximum_allowed_periphery_overlap", 1e-6, "Maximum allowed periphery overlap.")
      .set("shape", std::string("SPHERE"), "Periphery hydrodynamic shape.")
      .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
      .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
      .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
      .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
      .set("quadrature", std::string("GAUSS_LEGENDRE"), "Periphery quadrature.")
      .set("spectral_order", 32,
           "Periphery spectral order (only used if periphery is spherical is "
           "Gauss-Legendre quadrature).",
           make_new_validator(prefer_size_t, accept_int))
      .set("num_quadrature_points", 1000,
           "Periphery number of quadrature points (only used if quadrature type is "
           "FROM_FILE). Number of points in "
           "the files must match this quantity.",
           make_new_validator(prefer_size_t, accept_int))
      .set("quadrature_points_filename", std::string("hp1_periphery_hydro_quadrature_points.dat"),
           "Periphery quadrature points filename (only used if quadrature type is "
           "FROM_FILE).")
      .set("quadrature_weights_filename", std::string("hp1_periphery_hydro_quadrature_weights.dat"),
           "Periphery quadrature weights filename (only used if quadrature type is "
           "FROM_FILE).")
      .set("quadrature_normals_filename", std::string("hp1_periphery_hydro_quadrature_normals.dat"),
           "Periphery quadrature normals filename (only used if quadrature type is "
           "FROM_FILE).");

  valid_parameter_list.sublist("periphery_collision")
      .set("shape", std::string("SPHERE"), "Periphery collision shape.")
      .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
      .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
      .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
      .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
      .set("use_fast_approx", false, "Use fast periphery collision.")
      .set("shrink_periphery_over_time", false, "Shrink periphery over time.")
      .sublist("shrinkage")
      .set("num_shrinkage_steps", 1000,
           "Number of steps over which to perform the shrinking process (should not "
           "exceed num_time_steps).",
           make_new_validator(prefer_size_t, accept_int))
      .set("scale_factor_before_shrinking", 1.0, "Scale factor before shrinking.");

  valid_parameter_list.sublist("periphery_binding")
      .set("binding_rate", 1.0, "Periphery binding rate.")
      .set("unbinding_rate", 1.0, "Periphery unbinding rate.")
      .set("spring_constant", 1000.0, "Periphery spring constant.")
      .set("rest_length", 1.0, "Periphery spring rest length.")
      .set("bind_sites_type", std::string("RANDOM"), "Periphery bind sites type.")
      .set("num_bind_sites", 1000,
           "Periphery number of binding sites (only used if periphery_binding_sites_type is RANDOM and periphery has "
           "spherical or ellipsoidal shape).",
           make_new_validator(prefer_size_t, accept_int))
      .set("bind_site_locations_filename", std::string("periphery_bind_sites.dat"),
           "Periphery binding sites filename (only used if periphery_binding_sites_type "
           "is FROM_FILE).");

  valid_parameter_list.sublist("neighbor_list")
      .set("skin_distance", 1.0, "Neighbor list skin distance.")
      .set("force_neighborlist_update", false, "Force update of the neighbor list.")
      .set("force_neighborlist_update_nsteps", 10, "Number of timesteps between force update of the neighbor list.",
           make_new_validator(prefer_size_t, accept_int))
      .set("print_neighborlist_statistics", false, "Print neighbor list statistics.");

  return valid_parameter_list;
}

}  // namespace alens

}  // namespace mundy

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    //////////////////////////////////////
    // Read in and validate user params //
    //////////////////////////////////////

    // Parse the command line options to find the input filename
    Teuchos::CommandLineProcessor cmdp(false, true);
    std::string input_parameter_filename = "hp1.yaml";
    cmdp.setOption("params", input_parameter_filename, "The name of the input file. Defaults to hp1.yaml.");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_result = cmdp.parse(argc, argv);
    if (parse_result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      std::cout << "#############################################################################################"
                << std::endl;
      std::cout << "To run this code, please pass in --params=<input.yaml> as a command line argument." << std::endl;
      std::cout << std::endl;
      std::cout << "Note, all parameters and sublists in input.yaml must be contained in a single top-level list."
                << std::endl;
      std::cout << "Such as:" << std::endl;
      std::cout << std::endl;
      std::cout << "HP1:" << std::endl;
      std::cout << "  num_time_steps: 1000" << std::endl;
      std::cout << "  timestep_size: 1e-6" << std::endl;
      std::cout << "#############################################################################################"
                << std::endl;
      std::cout << "The valid parameters that can be set in the input file are:" << std::endl;
      Teuchos::ParameterList valid_params = get_valid_hp1_params();

      auto print_options = Teuchos::ParameterList::PrintOptions()
                               .showTypes(false)
                               .showDoc(true)
                               .showDefault(true)
                               .showFlags(false)
                               .indent(1);
      valid_params.print(std::cout, print_options);
      std::cout << "#############################################################################################"
                << std::endl;

      // Safely exit the program. If we print the help message, we don't need to do
      // anything else.
      exit(0);
    } else if (parse_result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      throw std::invalid_argument("Failed to parse the command line arguments.");
    }

    // Validate and set the default params from the parameter list.
    std::cout << "#############################################################################################"
              << std::endl;
    Teuchos::ParameterList params = *Teuchos::getParametersFromYamlFile(input_parameter_filename);
    params.validateParametersAndSetDefaults(get_valid_hp1_params());
    auto print_options = Teuchos::ParameterList::PrintOptions()
                             .showTypes(false)
                             .showDoc(false)
                             .showDefault(false)
                             .showFlags(false)
                             .indent(1);
    params.print(std::cout, print_options);
    std::cout << "#############################################################################################"
              << std::endl;

    //////////////////////////////////////
    // Declare and instantiate the mesh //
    //////////////////////////////////////
    mundy::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD)
        .set_spatial_dimension(3)
        .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
    meta_data_ptr->use_simple_fields();

    // Parts and their subsets
    stk::mesh::Part &e_part =                                             //
        meta_data_ptr->declare_part_with_topology("EUCHROMATIN_SPHERES",  //
                                                  stk::topology::PARTICLE);
    stk::mesh::Part &h_part =                                                 //
        meta_data_ptr->declare_part_with_topology("HETEROCHROMATIN_SPHERES",  //
                                                  stk::topology::PARTICLE);
    stk::mesh::Part &bs_part =                                   //
        meta_data_ptr->declare_part_with_topology("BIND_SITES",  //
                                                  stk::topology::NODE);
    stk::io::put_io_part_attribute(e_part);
    stk::io::put_io_part_attribute(h_part);
    stk::io::put_io_part_attribute(bs_part);

    stk::mesh::Part &hp1_part =                           //
        meta_data_ptr->declare_part_with_topology("HP1",  //
                                                  stk::topology::BEAM_2);
    stk::mesh::Part &left_hp1_part =                           //
        meta_data_ptr->declare_part_with_topology("LEFT_HP1",  //
                                                  stk::topology::BEAM_2);
    stk::mesh::Part &doubly_hp1_h_part =                           //
        meta_data_ptr->declare_part_with_topology("DOUBLY_HP1_H",  //
                                                  stk::topology::BEAM_2);
    stk::mesh::Part &doubly_hp1_bs_part =                           //
        meta_data_ptr->declare_part_with_topology("DOUBLY_HP1_BS",  //
                                                  stk::topology::BEAM_2);
    meta_data_ptr->declare_part_subset(hp1_part, left_hp1_part);
    meta_data_ptr->declare_part_subset(hp1_part, doubly_hp1_h_part);
    meta_data_ptr->declare_part_subset(hp1_part, doubly_hp1_bs_part);
    stk::io::put_io_part_attribute(hp1_part);
    stk::io::put_io_part_attribute(left_hp1_part);
    stk::io::put_io_part_attribute(doubly_hp1_h_part);
    stk::io::put_io_part_attribute(doubly_hp1_bs_part);

    stk::mesh::Part &backbone_segments_part =                           //
        meta_data_ptr->declare_part_with_topology("BACKBONE_SEGMENTS",  //
                                                  stk::topology::BEAM_2);
    stk::mesh::Part &ee_segments_part                               //
        = meta_data_ptr->declare_part_with_topology("EE_SEGMENTS",  //
                                                    stk::topology::BEAM_2);
    stk::mesh::Part &eh_segments_part                               //
        = meta_data_ptr->declare_part_with_topology("EH_SEGMENTS",  //
                                                    stk::topology::BEAM_2);
    stk::mesh::Part &hh_segments_part                               //
        = meta_data_ptr->declare_part_with_topology("HH_SEGMENTS",  //
                                                    stk::topology::BEAM_2);
    meta_data_ptr->declare_part_subset(backbone_segments_part, ee_segments_part);
    meta_data_ptr->declare_part_subset(backbone_segments_part, eh_segments_part);
    meta_data_ptr->declare_part_subset(backbone_segments_part, hh_segments_part);
    stk::io::put_io_part_attribute(backbone_segments_part);
    stk::io::put_io_part_attribute(ee_segments_part);
    stk::io::put_io_part_attribute(eh_segments_part);
    stk::io::put_io_part_attribute(hh_segments_part);

    stk::mesh::Part &backbone_backbone_neighbor_genx_part =
        meta_data_ptr->declare_part("BACKBONE_BACKBONE_NEIGHBOR_GENX", stk::topology::CONSTRAINT_RANK);
    stk::mesh::Part &hp1_h_neighbor_genx_part =
        meta_data_ptr->declare_part("HP1_H_NEIGHBOR_GENX", stk::topology::CONSTRAINT_RANK);
    stk::mesh::Part &hp1_bs_neighbor_genx_part =
        meta_data_ptr->declare_part("HP1_BS_NEIGHBOR_GENX", stk::topology::CONSTRAINT_RANK);

    // Fields
    auto &node_coord_field =                                            //
        meta_data_ptr->declare_field<double>(stk::topology::NODE_RANK,  //
                                             "NODE_COORDS");
    auto &node_velocity_field =                                         //
        meta_data_ptr->declare_field<double>(stk::topology::NODE_RANK,  //
                                             "NODE_VELOCITY");
    auto &node_force_field =                                            //
        meta_data_ptr->declare_field<double>(stk::topology::NODE_RANK,  //
                                             "NODE_FORCE");
    auto &node_rng_field =                                                //
        meta_data_ptr->declare_field<unsigned>(stk::topology::NODE_RANK,  //
                                               "NODE_RNG_COUNTER");
    stk::io::set_field_output_type(node_coord_field, "vector_3d");
    stk::io::set_field_output_type(node_velocity_field, "vector_3d");
    stk::io::set_field_output_type(node_force_field, "vector_3d");

    auto &element_hydrodynamic_radius_field =                              //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_HYDRODYNAMIC_RADIUS");
    auto &element_binding_radius_field =                                   //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_BINDING_RADIUS");
    auto &element_collision_radius_field =                                 //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_COLLISION_RADIUS");
    auto &element_hookean_spring_constant_field =                          //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_HOOKEAN_SPRING_CONSTANT");
    auto &element_hookean_spring_rest_length_field =                       //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_HOOKEAN_SPRING_REST_LENGTH");
    auto &element_youngs_modulus_field =                                   //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_YOUNGS_MODULUS");
    auto &element_poissons_ratio_field =                                   //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_POISSONS_RATIO");
    auto &element_aabb_field =                                             //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_AABB");
    auto &element_aabb_displacement_field =                                //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_AABB_DISPLACEMENT");
    auto &element_binding_rates_field =                                    //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_BINDING_RATES");
    auto &element_unbinding_rates_field =                                  //
        meta_data_ptr->declare_field<double>(stk::topology::ELEMENT_RANK,  //
                                             "ELEMENT_UNBINDING_RATES");
    auto &element_perform_state_change_field =                               //
        meta_data_ptr->declare_field<unsigned>(stk::topology::ELEMENT_RANK,  //
                                               "ELEMENT_PERFORM_STATE_CHANGE");
    auto &element_rng_field =                                                //
        meta_data_ptr->declare_field<unsigned>(stk::topology::ELEMENT_RANK,  //
                                               "ELEMENT_RNG_COUNTER");
    stk::io::set_field_output_type(element_hydrodynamic_radius_field, "scalar");
    stk::io::set_field_output_type(element_binding_radius_field, "scalar");
    stk::io::set_field_output_type(element_collision_radius_field, "scalar");
    stk::io::set_field_output_type(element_hookean_spring_constant_field, "scalar");
    stk::io::set_field_output_type(element_hookean_spring_rest_length_field, "scalar");
    stk::io::set_field_output_type(element_youngs_modulus_field, "scalar");
    stk::io::set_field_output_type(element_poissons_ratio_field, "scalar");
    stk::io::set_field_output_type(element_binding_rates_field, "vector_2d");
    stk::io::set_field_output_type(element_unbinding_rates_field, "vector_2d");

    auto &constraint_potential_force_field =                                  //
        meta_data_ptr->declare_field<double>(stk::topology::CONSTRAINT_RANK,  //
                                             "CONSTRAINT_POTENTIAL_FORCE");
    auto &constraint_state_change_rate_field =                                //
        meta_data_ptr->declare_field<double>(stk::topology::CONSTRAINT_RANK,  //
                                             "CONSTRAINT_STATE_CHANGE_RATE");
    auto &constraint_perform_state_change_field =                               //
        meta_data_ptr->declare_field<unsigned>(stk::topology::CONSTRAINT_RANK,  //
                                               "CONSTRAINT_PERFORM_STATE_CHANGE");

    // Sew it all together
    // Any field with a constant initial value should have that value set here. If it is
    // to be uninitialized, use nullptr.
    const double zero_vector3d[3] = {0.0, 0.0, 0.0};
    const double zero_scalar[1] = 0.0;
    const unsigned zero_unsigned[1] = 0;
    stk::mesh::put_field_on_entire_mesh(node_coord_field, nullptr);

    // Heterochromatin and euchromatin spheres are used for hydrodynamics. They move and
    // have forces applied to them. If brownian motion is enabled, they will have a
    // stocastic velocity. Heterochromatin spheres are considered for hp1 binding and
    // require an AABB for neighbor detection.
    stk::mesh::put_field_on_mesh(node_velocity_field,  //
                                 e_part | h_part,      //
                                 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(node_force_field,  //
                                 e_part | h_part,   //
                                 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(node_rng_field,   //
                                 e_part | h_part,  //
                                 1, zero_unsigned);
    stk::mesh::put_field_on_mesh(element_hydrodynamic_radius_field,  //
                                 e_part | h_part,                    //
                                 1, chromatin_hydrodynamic_radius);
    stk::mesh::put_field_on_mesh(element_aabb_field,  //
                                 h_part,              //
                                 6, nullptr);
    stk::mesh::put_field_on_mesh(element_aabb_displacement_field,  //
                                 h_part,                           //
                                 6, nullptr);

    // Backbone segments apply spring forces and act as spherocylinders for the sake of
    // collision. They apply forces to their nodes and have a collision radius. The
    // difference between ee, eh, and hh segments is that ee segments can exert an active
    // dipole.
    stk::mesh::put_field_on_mesh(node_force_field,        //
                                 backbone_segments_part,  //
                                 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(element_collision_radius_field,  //
                                 backbone_segments_part,          //
                                 1, backbone_collision_radius);
    stk::mesh::put_field_on_mesh(element_hookean_spring_constant_field,  //
                                 backbone_segments_part,                 //
                                 1, backbone_spring_constant);
    stk::mesh::put_field_on_mesh(element_hookean_spring_rest_length_field,  //
                                 backbone_segments_part,                    //
                                 1, backbone_rest_length);
    stk::mesh::put_field_on_mesh(element_youngs_modulus_field,  //
                                 backbone_segments_part,        //
                                 1, collision_youngs_modulus);
    stk::mesh::put_field_on_mesh(element_poissons_ratio_field,  //
                                 backbone_segments_part,        //
                                 1, collision_poissons_ratio);
    stk::mesh::put_field_on_mesh(element_aabb_field,      //
                                 backbone_segments_part,  //
                                 6, nullptr);
    stk::mesh::put_field_on_mesh(element_aabb_displacement_field,  //
                                 backbone_segments_part,           //
                                 6, nullptr);

    // HP1 crosslinkers are used for binding/unbinding and apply forces to their nodes.
    const double left_and_right_binding_rates[2] = {left_binding_rate, right_binding_rate};
    const double left_and_right_unbinding_rates[2] = {left_unbinding_rate, right_unbinding_rate};
    stk::mesh::put_field_on_mesh(node_force_field,  //
                                 hp1_part,          //
                                 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(element_binding_rates_field,  //
                                 hp1_part,                     //
                                 2, left_and_right_binding_rates);
    stk::mesh::put_field_on_mesh(element_unbinding_rates_field,  //
                                 hp1_part,                       //
                                 2, left_and_right_unbinding_rates);
    stk::mesh::put_field_on_mesh(element_perform_state_change_field,  //
                                 hp1_part,                            //
                                 1, zero_unsigned);
    stk::mesh::put_field_on_mesh(element_binding_radius_field,  //
                                 hp1_part,                      //
                                 1, crosslinker_binding_radius);
    stk::mesh::put_field_on_mesh(element_rng_field,  //
                                 hp1_part,           //
                                 1, zero_unsigned);

    // That's it for the mesh. Commit it's structure and create the bulk data.
    meta_data_ptr->commit();
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);

    if (!restart_performed_) {
      // TODO(palmerb4): The fact that some of these inputs are repeated so many times
      // Can be mitigated by using a ChainOfSprings-like class and passing it to
      // initialize.
      DeclareChromatinAndHP1()
          .set_num_chromosomes(num_chromosomes)                                //
          .set_num_chromatin_repeats(num_chromatin_repeats)                    //
          .set_num_euchromatin_per_repeat(num_euchromatin_per_repeat)          //
          .set_num_heterochromatin_per_repeat(num_heterochromatin_per_repeat)  //
          .set_enable_backbone_collision(enable_backbone_collision)            //
          .set_enable_backbone_springs(enable_backbone_springs)                //
          .set_enable_crosslinkers(enable_crosslinkers)                        //
          .run();

      // Initialize node positions for each chromosome
      if (initialization_type_ == INITIALIZATION_TYPE::GRID) {
        InitializeChromosomePositionsGrid()
            .set_num_chromosomes(num_chromosomes)                                //
            .set_num_chromatin_repeats(num_chromatin_repeats)                    //
            .set_num_euchromatin_per_repeat(num_euchromatin_per_repeat)          //
            .set_num_heterochromatin_per_repeat(num_heterochromatin_per_repeat)  //
            .set_initial_chromosome_separation(initial_chromosome_separation)    //
            .set_node_coord_field(node_coord_field)                              //
            .run();
      } else if (initialization_type_ == INITIALIZATION_TYPE::RANDOM_UNIT_CELL) {
        InitializeChromosomePositionsRandomUnitCell()
            .set_num_chromosomes(num_chromosomes)                                //
            .set_num_chromatin_repeats(num_chromatin_repeats)                    //
            .set_num_euchromatin_per_repeat(num_euchromatin_per_repeat)          //
            .set_num_heterochromatin_per_repeat(num_heterochromatin_per_repeat)  //
            .set_initial_chromosome_separation(initial_chromosome_separation)    //
            .set_unit_cell_size(unit_cell_size)                                  //
            .set_node_coord_field(node_coord_field)                              //
            .run();
      } else if (initialization_type_ == INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL) {
        InitializeChromosomePositionsHilbertRandomUnitCell()
            .set_num_chromosomes(num_chromosomes)                                //
            .set_num_chromatin_repeats(num_chromatin_repeats)                    //
            .set_num_euchromatin_per_repeat(num_euchromatin_per_repeat)          //
            .set_num_heterochromatin_per_repeat(num_heterochromatin_per_repeat)  //
            .set_initial_chromosome_separation(initial_chromosome_separation)    //
            .set_unit_cell_size(unit_cell_size)                                  //
            .set_node_coord_field(node_coord_field)                              //
            .run();
      } else {
        MUNDY_THROW_ASSERT(false, "Unknown initialization type: " << initialization_type_);
      }
    }

    if (enable_periphery_binding_ && !restart_performed_) {
      if (periphery_bind_sites_type_ == BIND_SITES_TYPE::RANDOM) {
        if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
          DeclareAndInitializeRandomSphericalPeripheryBindSites()
              .set_num_bind_sites(num_periphery_bind_sites)      //
              .set_periphery_radius(periphery_collision_radius)  //
              .set_bind_site_selector(bs_part)                   //
              .set_node_coord_field(node_coord_field)            //
              .run();
        } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          DeclareAndInitializeRandomEllipsoidalPeripheryBindSites()
              .set_num_bind_sites(num_periphery_bind_sites)  //
              .set_periphery_r1(periphery_collision_r1)      //
              .set_periphery_r2(periphery_collision_r2)      //
              .set_periphery_r3(periphery_collision_r3)      //
              .set_bind_site_selector(bs_part)               //
              .set_node_coord_field(node_coord_field)        //
              .run();
        } else {
          MUNDY_THROW_ASSERT(false, "Unknown periphery collision shape: " << periphery_collision_shape_);
        }
      } else if (periphery_bind_sites_type_ == BIND_SITES_TYPE::FROM_FILE) {
        DeclareAndInitializePeripheryBindSitesFromFile()
            .set_num_bind_sites(num_periphery_bind_sites)              //
            .set_bind_site_selector(bs_part)                           //
            .set_node_coord_field(node_coord_field)                    //
            .set_bind_site_coords_filename(bind_site_coords_filename)  //
            .run();
      } else {
        MUNDY_THROW_ASSERT(false, "Unknown periphery bind sites type: " << periphery_bind_sites_type_);
      }
    }

    // Post setup but pre-run
    if (loadbalance_post_initialization_) {
      LoadBalance()
          .set_bulk_data(bulk_data_ptr_)        //
          .set_balance_settings(RcbSettings())  //
          .run();
    }

    // Reset simulation control variables
    timestep_index = 0;

    // Check to see if we need to do anything for compressing the system.
    if (enable_periphery_collision && shrink_periphery_over_time) {
      if (periphery_collision_shape == PERIPHERY_SHAPE::SPHERE) {
        periphery_collision_radius *= periphery_collision_scale_factor_before_shrinking;
      } else if (periphery_collision_shape == PERIPHERY_SHAPE::ELLIPSOID) {
        periphery_collision_r1 *= periphery_collision_scale_factor_before_shrinking;
        periphery_collision_r2 *= periphery_collision_scale_factor_before_shrinking;
        periphery_collision_r3 *= periphery_collision_scale_factor_before_shrinking;
      }
    }

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer overall_timer;
    Kokkos::Timer timer;
    for (timestep_index_ = 0; timestep_index_ < num_time_steps_; timestep_index_++) {
      // Prepare the current configuration.
      FieldFill()
          .set_const_value(std::array<double, 3>{0.0, 0.0, 0.0})  //
          .set_field(node_velocity_field)                         //
          .run();
      FieldFill()
          .set_const_value(std::array<double, 3>{0.0, 0.0, 0.0})  //
          .set_field(node_force_field)                            //
          .run();
      FieldFill()
          .set_const_value(std::array<double, 2>{0.0, 0.0})  //
          .set_field(element_binding_rates_field)            //
          .run();
      FieldFill()
          .set_const_value(std::array<double, 2>{0.0, 0.0})  //
          .set_field(element_unbinding_rates_field)          //
          .run();
      FieldFill()
          .set_const_value(std::array<unsigned, 1>{0u})   //
          .set_field(element_perform_state_change_field)  //
          .run();
      FieldFill()
          .set_const_value(std::array<unsigned, 1>{0u})      //
          .set_field(constraint_perform_state_change_field)  //
          .run();
      FieldFill()
          .set_const_value(std::array<double, 1>{0.0})    //
          .set_field(constraint_state_change_rate_field)  //
          .run();
      FieldFill()
          .set_const_value(std::array<double, 3>{0.0, 0.0, 0.0})  //
          .set_field(constraint_potential_force_field)            //
          .run();

      // If we are doing a compression run, shrink the periphery
      if (enable_periphery_collision_ && shrink_periphery_over_time_ &&
          (timestep_index_ < periphery_collision_shrinkage_num_steps_)) {
        const double shrink_factor = std::pow(1.0 / periphery_collision_scale_factor_before_shrinking_,
                                              1.0 / periphery_collision_shrinkage_num_steps_);
        if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
          periphery_collision_radius_ *= shrink_factor;
        } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          periphery_collision_r1_ *= shrink_factor;
          periphery_collision_r2_ *= shrink_factor;
          periphery_collision_r3_ *= shrink_factor;
        }
      }

      //////////////////////
      // Detect neighbors //
      //////////////////////
      update_neighbor_list_ = false;

      // ComputeAABB for everybody at each time step. The accumulator uses this updated
      // information to calculate if we need to update the entire neighbor list.
      ComputeAABBSegs()
          .set_const_buffer_distance(skin_distance)                             //
          .set_const_seg_selector(backbone_segments_selector)                   //
          .set_const_node_coord_field(node_coord_field)                         //
          .set_const_element_radius_field(element_collision_radius_field_ptr_)  //
          .set_aabb_field(element_aabb_field)                                   //
          .run();
      ComputeAABBSpheres()
          .set_const_buffer_distance(skin_distance)                                //
          .set_const_sphere_selector(hp1_selector | h_selector | bs_selector)      //
          .set_const_node_coord_field(node_coord_field_ptr_)                       //
          .set_const_element_radius_field(element_hydrodynamic_radius_field_ptr_)  //
          .set_aabb_field(element_aabb_field_ptr_)                                 //
          .run();
      ComputeAABBDisplacements()  // displacement = beta * displacement + alpha (new - old)
          .set_const_old_aabb_field(element_old_aabb_field_ptr_)  //
          .set_const_aabb_field(element_aabb_field_ptr_)          //
          .set_const_alpha(1.0)                                   //
          .set_const_beta(1.0)                                    //  accumulate the displacement rather than reset it
          .set_aabb_displacement_field(element_aabb_displacement_field_ptr_)  //
          .run();

      // Check if we need to update the neighbor list. Eventually this will be replaced
      // with a mesh attribute to synchronize across multiple tasks. For now, make sure
      // that the default is to not update neighbor lists.
      double max_aabb_displacement = 0.0;
      ComputeMaxAABBDisplacement()
          .set_seletor(backbone_segments_selector | hp1_selector | h_selector | bs_selector)  //
          .set_const_aabb_displacement_field(element_aabb_displacement_field)                 //
          .set_max_aabb_displacement(max_aabb_displacement)                                   //
          .run();

      // Now do a check to see if we need to update the neighbor list.
      if (max_aabb_displacement > skin_distance) {
        // Reset the accumulated AABB displacements
        FieldFill()
            .set_const_value(std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0})  //
            .set_field(element_aabb_displacement_field)                            //
            .run();

        // Update the neighbor list
        if (enable_backbone_collision_ || enable_crosslinkers_ || enable_periphery_binding_) {
          DestroyDistantNeighbors()
              .set_const_max_distance(0)                 // Only destroy non-overlapping neighbors
              .set_const_aabb_field(element_aabb_field)  //
              .set_const_neighbor_selector(backbone_backbone_neighbor_genx_selector | hp1_h_neighbor_genx_selector |
                                           hp1_bs_neighbor_genx_selector)  //
              .run();
        }

        // Generate the GENX neighbor linkers
        if (enable_backbone_collision_) {
          GenerateNeighborLinkers()
              .set_const_source_selector(backbone_segments_selector)        //
              .set_const_target_selector(backbone_segments_selector)        //
              .set_const_source_aabb_field(element_aabb_field)              //
              .set_const_target_aabb_field(element_aabb_field)              //
              .set_const_output_part(backbone_backbone_neighbor_genx_part)  //
              .run();
          DestroyBoundNeighbors()
              .set_const_neighbor_selector(backbone_backbone_neighbor_genx_selector)  //
              .run();
        }
        if (enable_crosslinkers_) {
          GenerateNeighborLinkers()
              .set_const_source_selector(hp1_selector)          //
              .set_const_target_selector(h_selector)            //
              .set_const_source_aabb_field(element_aabb_field)  //
              .set_const_target_aabb_field(element_aabb_field)  //
              .set_const_output_part(hp1_h_neighbor_genx_part)  //
              .run();
        }
        if (enable_periphery_binding_) {
          GenerateNeighborLinkers()
              .set_const_source_selector(hp1_selector)           //
              .set_const_target_selector(bs_selector)            //
              .set_const_source_aabb_field(element_aabb_field)   //
              .set_const_target_aabb_field(element_aabb_field)   //
              .set_const_output_part(hp1_bs_neighbor_genx_part)  //
              .run();
        }
      }

      if (enable_crosslinkers_) {
        ComputeBindLeftBoundHarmonicToSphereZPartition()
            .set_kt(binding_kt)                                                                            //
            .set_const_node_coord_field(node_coord_field)                                                  //
            .set_const_element_hookean_spring_constant_field(element_hookean_spring_constant_field)        //
            .set_const_element_hookean_spring_rest_length_field(element_hookean_spring_rest_length_field)  //
            .set_const_left_bound_spring_part(left_bound_spring_part)                                      //
            .set_const_constraint_state_change_rate_field(constraint_state_change_rate_field)              //
            .set_const_spring_sphere_neighbor_genx_part(spring_sphere_neighbor_genx_part)                  //
            .run();

        KMCSpringLeftToDoubly()
            .set_time_step(timestep_size)                                                      //
            .set_element_rng_field(element_rng_field)                                          //
            .set_element_perform_state_change_field(element_perform_state_change_field)        //
            .set_constraint_perform_state_change_field(constraint_perform_state_change_field)  //
            .set_constraint_state_change_rate_field(constraint_state_change_rate_field)        //
            .set_neighbor_linker_selector(neighbor_linker_selector)                            //
            .set_left_bound_spring_selector(left_bound_spring_selector)                        //
            .run();

        KMCSpringDoublyToRight()
            .set_time_step(timestep_size)                                                      //
            .set_element_rng_field(element_rng_field)                                          //
            .set_element_perform_state_change_field(element_perform_state_change_field)        //
            .set_constraint_perform_state_change_field(constraint_perform_state_change_field)  //
            .set_constraint_state_change_rate_field(constraint_state_change_rate_field)        //
            .set_element_unbinding_rates_field(element_unbinding_rates_field)                  //
            .set_double_bound_spring_selector(double_bound_spring_selector)                    //
            .run();

        PerformStateChangeDynamicSprings()
            .set_left_bound_spring_selector(left_bound_spring_selector)                        //
            .set_double_bound_spring_selector(double_bound_spring_selector)                    //
            .set_spring_sphere_genx_selector(spring_sphere_genx_selector)                      //
            .set_element_perform_state_change_field(element_perform_state_change_field)        //
            .set_constraint_perform_state_change_field(constraint_perform_state_change_field)  //
            .run();
      }

      // Evaluate forces f(x(t)).
      if (enable_backbone_collision_) {
        // Potential evaluation (Hertzian contact)
        ComputeSignedSepDistContactNormalAndContactPointsSegSeg()
            .set_selector(backbone_backbone_neighbor_genx_selector)                    //
            .set_node_coord_field(node_coord_field)                                    //
            .set_element_radius_field(element_collision_radius_field)                  //
            .set_constraint_signed_sep_dist_field(constraint_signed_sep_dist_field)    //
            .set_constraint_contact_normal_field(constraint_contact_normal_field)      //
            .set_constraint_contact_location_field(constraint_contact_location_field)  //
            .run();
        ComputeHertzianContactForcesSegSeg()
            .set_selector(backbone_backbone_neighbor_genx_selector)                    //
            .set_node_coord_field(node_coord_field)                                    //
            .set_element_radius_field(element_collision_radius_field)                  //
            .set_constraint_signed_sep_dist_field(constraint_signed_sep_dist_field)    //
            .set_constraint_contact_normal_field(constraint_contact_normal_field)      //
            .set_constraint_contact_location_field(constraint_contact_location_field)  //
            .set_constraint_potential_force_field(constraint_potential_force_field)    //
            .run();
        LinkerPotentialForceReductionSegs()
            .set_selector(backbone_backbone_neighbor_genx_selector)                  //
            .set_node_coord_field(node_coord_field)                                  //
            .set_element_radius_field(element_collision_radius_field)                //
            .set_constraint_potential_force_field(constraint_potential_force_field)  //
            .set_node_force_field(node_force_field)                                  //
            .run();
      }

      if (enable_backbone_springs_) {
        ComputeHookeanSpringForces()
            .set_selector(backbone_segments_selector)                                                //
            .set_node_coord_field(node_coord_field)                                                  //
            .set_element_hookean_spring_constant_field(element_hookean_spring_constant_field)        //
            .set_element_hookean_spring_rest_length_field(element_hookean_spring_rest_length_field)  //
            .set_node_force_field(node_force_field)                                                  //
            .run();
      }

      if (enable_crosslinkers_) {
        // Select only active springs in the system. Aka, not left bound.
        ComputeHookeanSpringForces()
            .set_selector(hp1_selector - left_hp1_selector)                                          //
            .set_node_coord_field(node_coord_field)                                                  //
            .set_element_hookean_spring_constant_field(element_hookean_spring_constant_field)        //
            .set_element_hookean_spring_rest_length_field(element_hookean_spring_rest_length_field)  //
            .set_node_force_field(node_force_field)                                                  //
            .run();
      }

      if (enable_periphery_collision_) {
        if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
          ComputeSphericalPeripheryCollisionForcesWithSpheres()
              .set_selector(h_selector | e_selector)                               //
              .set_periphery_radius(periphery_collision_radius)                    //
              .set_periphery_center({0.0, 0.0, 0.0})                               //
              .set_collision_spring_constant(periphery_collision_spring_constant)  //
              .set_node_coord_field(node_coord_field)                              //
              .set_element_radius_field(element_radius_field)                      //
              .run()
        } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
          if (periphery_collision_use_fast_approx_) {
            ComputeEllipsoidalPeripheryCollisionForcesWithSpheresFastApprox()
                .set_spheres_selector(h_selector | e_selector)                       //
                .set_periphery_r1(periphery_collision_r1)                            //
                .set_periphery_r2(periphery_collision_r2)                            //
                .set_periphery_r3(periphery_collision_r3)                            //
                .set_periphery_center({0.0, 0.0, 0.0})                               //
                .set_periphery_orientation({1.0, 0.0, 0.0, 0.0})                     //
                .set_collision_spring_constant(periphery_collision_spring_constant)  //
                .set_node_coord_field(node_coord_field)                              //
                .set_element_radius_field(element_radius_field)                      //
                .set_node_force_field(node_force_field)                              //
                .run();
          } else {
            ComputeEllipsoidalPeripheryCollisionForcesWithSpheres()
                .set_spheres_selector(h_selector | e_selector)                       //
                .set_periphery_r1(periphery_collision_r1)                            //
                .set_periphery_r2(periphery_collision_r2)                            //
                .set_periphery_r3(periphery_collision_r3)                            //
                .set_periphery_center({0.0, 0.0, 0.0})                               //
                .set_periphery_orientation({1.0, 0.0, 0.0, 0.0})                     //
                .set_collision_spring_constant(periphery_collision_spring_constant)  //
                .set_node_coord_field(node_coord_field)                              //
                .set_element_radius_field(element_radius_field)                      //
                .set_node_force_field(node_force_field)                              //
                .run();
          }
        } else {
          MUNDY_THROW_ASSERT(false, "Invalid periphery type.");
        }
      }

      // Compute velocities.
      if (enable_chromatin_brownian_motion_) {
        ComputeBrownianVelocitySpheres()
            .set_selector(h_selector | e_selector)             //
            .set_time_step(timestep_size)                      //
            .set_viscosity(viscosity)                          //
            .set_brownian_kt(brownian_kt)                      //
            .set_sphere_radius(chromatin_hydrodynamic_radius)  //
            .set_node_rng_field(node_rng_field)                //
            .set_node_force_field(node_force_field)            //
            .set_node_velocity_field(node_velocity_field)      //
            .run();
      }

      if (enable_backbone_n_body_hydrodynamics_) {
        ComputeRPYHydro()
            .set_spheres_selector(backbone_segments_selector)             //
            .set_periphery(periphery_ptr_)                                //
            .set_viscosity(viscosity)                                     //
            .set_node_coord_field(node_coord_field)                       //
            .set_element_radius_field(element_hydrodynamic_radius_field)  //
            .set_node_force_field(node_force_field)                       //
            .set_node_velocity_field(node_velocity_field)                 //
            .set_validate_noslip_boundary(true)                           //
            .run();
      } else {
        ComputeDryLocalDragSpheres()
            .set_selector(h_selector | e_selector)             //
            .set_viscosity(viscosity)                          //
            .set_sphere_radius(chromatin_hydrodynamic_radius)  //
            .set_node_force_field(node_force_field)            //
            .set_node_velocity_field(node_velocity_field)      //
            .run();
      }

      // Logging, if desired, write to console
      if (timestep_index_ % log_frequency_ == 0) {
        if (bulk_data_ptr_->parallel_rank() == 0) {
          double tps = static_cast<double>(log_frequency_) / static_cast<double>(timer.seconds());
          std::cout << "Step: " << std::setw(15) << timestep_index_ << ", tps: " << std::setprecision(15) << tps;
          if (enable_periphery_collision_ && shrink_periphery_over_time_ &&
              timestep_index_ < periphery_collision_shrinkage_num_steps_) {
            if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
              std::cout << ", periphery_collision_radius: " << periphery_collision_radius_;
            } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
              std::cout << ", periphery_collision_r1: " << periphery_collision_r1_
                        << ", periphery_collision_r2: " << periphery_collision_r2_
                        << ", periphery_collision_r3: " << periphery_collision_r3_;
            }
          }
          std::cout << std::endl;
          timer.reset();
        }
      }

      // IO. If desired, write out the data for time t (STK or mundy)
      if (timestep_index_ % io_frequency_ == 0) {
        io_broker_ptr_->write_io_broker_timestep(static_cast<int>(timestep_index_),
                                                 static_cast<double>(timestep_index_));
      }

      // Update positions. x(t + dt) = x(t) + dt * v(t).
      NodeEulerPositionIntegrator()
          .set_selector(h_selector | e_selector)         //
          .set_time_step(timestep_size)                  //
          .set_old_node_coord_field(node_coord_field)    //
          .set_new_node_coord_field(node_coord_field)    //
          .set_node_velocity_field(node_velocity_field)  //
          .run();
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep =
          static_cast<double>(overall_timer.seconds()) / static_cast<double>(num_time_steps_);
      double tps = 1.0 / avg_time_per_timestep;
      std::cout << "******************Final statistics (Rank 0)**************\n";
      if (print_neighborlist_statistics_) {
        std::cout << "****************\n";
        std::cout << "Neighbor list statistics\n";
        for (auto &neighborlist_entry : neighborlist_update_steps_times_) {
          auto [timestep, elasped_step, elapsed_time] = neighborlist_entry;
          auto tps_nl = static_cast<double>(elasped_step) / elapsed_time;
          std::cout << "  Rebuild timestep: " << timestep << ", elapsed_steps: " << elasped_step
                    << ", elapsed_time: " << elapsed_time << ", tps: " << tps_nl << std::endl;
        }
      }
      std::cout << "****************\n";
      std::cout << "Simulation statistics\n";
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
      std::cout << "Timesteps per second: " << std::setprecision(15) << tps << std::endl;
    }
  }

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
