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

namespace mundy {

namespace alens {

// The Spheres class merely contains a selector, a node coordinates field, and an element radius field.
// It is NOT meant to contain methods. It is meant to be a data container used to reduce the inputs to functions.
// This also allows for function overloading and can perform check like if a field has the right rank or not.
//
// For now all, primitive shapes are assumed heterogeneous until we do performance testing on the usefulness of
// homogenous shapes.
struct SphereFields {
  stk::mesh::Field<double> &n_coords;
  stk::mesh::Field<double> &el_radius;
};

struct MotileSphereFields {
  stk::mesh::Field<double> &n_force;
  stk::mesh::Field<double> &n_velocity;
};

struct HookeanSpringFields {
  stk::mesh::Field<double> &n_coords;
  stk::mesh::Field<double> &el_spring_constant;
  stk::mesh::Field<double> &el_rest_length;
};

struct FeneSpringFields {
  stk::mesh::Field<double> &n_coords;
  stk::mesh::Field<double> &el_spring_constant;
  stk::mesh::Field<double> &el_rest_length;
};

struct DynSpringFields {
  stk::mesh::Field<double> &el_binding_rates;
  stk::mesh::Field<double> &el_unbinding_rates;
  stk::mesh::Field<size_t> &el_rng_counter;
  stk::mesh::Field<unsigned> &el_perform_state_change;
};

struct GenXBindDynSpringToNodeFields {
  stk::mesh::Field<double> &el_state_change_rate;
  stk::mesh::Field<unsigned> &el_perform_state_change;
};

struct Sphere {
  double radius;
  Kokkos::Array<double, 3> center;
};

struct Ellipsoid {
  Kokkos::Array<double, 3> center;
  Kokkos::Array<double, 3> radii;
};


namespace dynamic_springs {

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

void compute_state_change_rate_left_to_doubly(const double kt, const stk::mesh::Selector &left_bound_springs,
                                              const stk::mesh::Selector &spring_node_linkers,
                                              const HookeanSpringFields &hookean_spring_fields,
                                              const DynamicSpringFields &dynamic_spring_fields,
                                              GenXBindDynSpringToNodeFields *const genx_bind_dyn_spring_to_node_fields) {
  const double inv_kt = 1.0 / kt;
  stk::mesh::for_each_entity_run(
      n_coord_field.get_mesh(), stk::topology::CONSTRAINT_RANK, spring_sphere_linkers,
      [&n_coord_field, &c_linked_entities_field, &constraint_state_change_rate, &crosslinker_spring_constant,
       &crosslinker_spring_rest_length, &left_bound_spring_part, &inv_kt, &crosslinker_right_binding_rate](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &neighbor_genx) {
        // Get the sphere and crosslinker attached to the linker.
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(c_linked_entities_field, neighbor_genx));
        const stk::mesh::Entity &crosslinker = bulk_data.get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &sphere = bulk_data.get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data.is_valid(crosslinker),  std::runtime_error, "Encountered invalid crosslinker entity in "
                           "compute_z_partition_left_bound_harmonic.");
        MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere),  std::runtime_error, "Encountered invalid sphere entity in "
                           "compute_z_partition_left_bound_harmonic.");

        // Only act on the left-bound crosslinkers that don't self-interact
        if (bulk_data.bucket(crosslinker).member(left_bound_spring_part)) {
          const stk::mesh::Entity &sphere_node = bulk_data.begin_nodes(sphere)[0];
          is_self_interaction = bulk_data.begin_nodes(crosslinker)[0] == sphere_node;
          if (!is_self_interaction) {
            const auto dr = mundy::mesh::vector3_field_data(n_coord_field, sphere_node) -
                            mundy::mesh::vector3_field_data(n_coord_field, bulk_data.begin_nodes(crosslinker)[0]);
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

void kmc_choose_state_left_bound(const double timestep_size, const stk::mesh::Selector &left_bound_springs,
                                 DynamicSpringFields *const dynamic_spring_fields,
                                 const stk::mesh::Selector &spring_node_linkers,
                                 GenXBindDynSpringToNodeFields *const genx_bind_dyn_spring_to_node_fields) {
  // Loop over left-bound spring and decide if they bind or not
  stk::mesh::for_each_entity_run(
      c_perform_state_change_field.get_mesh(), stk::topology::ELEMENT_RANK, left_bound_springs,
      [&spring_sphere_linkers, &el_rng_field, &c_perform_state_change_field, &el_perform_state_change_field,
       &c_state_change_rate_field,
       &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spring) {
        // Get all of my associated linkers
        const stk::mesh::Entity &any_arbitrary_spring_node = bulk_data.begin_nodes(spring)[0];
        const stk::mesh::Entity *linkers = bulk_data.begin(any_arbitrary_spring_node, stk::topology::CONSTRAINT_RANK);
        const unsigned num_linkers =
            bulk_data.num_connectivity(any_arbitrary_spring_node, stk::topology::CONSTRAINT_RANK);

        // Loop over the attached linker and bind if the rng falls in their range.
        double z_tot = 0.0;
        for (unsigned j = 0; j < num_linkers; j++) {
          const auto &linker = linkers[j];
          const bool desired_linker_type = spring_sphere_linkers(bulk_data.bucket(linker));
          if (desired_linker_type) {
            const double z_i = timestep_size * stk::mesh::field_data(c_state_change_rate_field, linker)[0];
            z_tot += z_i;
          }
        }

        // Fetch the RNG state, get a random number out of it, and increment
        unsigned *element_rng_counter = stk::mesh::field_data(el_rng_field, spring);
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
          for (unsigned j = 0; j < num_linker_linkers; j++) {
            auto &linker = linkers[j];
            const bool desired_linker_type = spring_sphere_linkers(bulk_data.bucket(linker));
            if (desired_linker_type) {
              const double binding_probability =
                  scale_factor * stk::mesh::field_data(c_state_change_rate_field, linker)[0];
              cumsum += binding_probability;
              if (randu01 < cumsum) {
                // We have a binding event, set this, then bail on the for loop
                // Store the state change on both the genx and the spring
                stk::mesh::field_data(c_perform_state_change_field, linker)[0] =
                    static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                stk::mesh::field_data(el_perform_state_change_field, spring)[0] =
                    static_cast<unsigned>(BINDING_STATE_CHANGE::LEFT_TO_DOUBLY);
                break;
              }
            }
          }
        }
      });

  // At this point, c_state_change_rate_field is only up-to-date for
  // locally-owned entities. We need to communicate this information to all other
  // processors.
  stk::mesh::communicate_field_data(c_perform_state_change_field.get_mesh(),
                                    {&el_perform_state_change_field, &c_perform_state_change_field});
}

void kmc_choose_state_doubly_bound(const stk::mesh::Selector &doubly_bound_springs,
                                   DynamicSpringFields *const dynamic_spring_fields) {
  // Note, this assumes that we only have DOUBLY->DOUBLY and DOUBLY->RIGHT transitions
  // and not DOUBLY->LEFT.
  stk::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, doubly_bound_springs,
      [&el_rng_field, &el_perform_state_change_field, &el_unbinding_rates_field, &timestep_size](
          [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spring) {
        // Only DOUBLY->RIGHT transitions are allowed (for now)
        const double right_unbinding_probability =
            timestep_size * stk::mesh::field_data(el_unbinding_rates_field, spring)[1];
        double z_tot = right_unbinding_probability;

        // Fetch the RNG state, get a random number out of it, and increment
        unsigned *element_rng_counter = stk::mesh::field_data(el_rng_field, spring);
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
          stk::mesh::field_data(el_perform_state_change_field, spring)[0] =
              static_cast<unsigned>(BINDING_STATE_CHANGE::DOUBLY_TO_LEFT);
        }
      });

  // At this point, state change field is only up-to-date for locally-owned entities. We
  // need to communicate this information to all other processors.
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {*el_perform_state_change_field});
}

void perform_state_change(const stk::mesh::Selector &left_bound_springs,
                          const stk::mesh::Selector &doubly_bound_springs,
                          const stk::mesh::Selector &spring_node_linkers,
                          const DynamicSpringFields &dynamic_spring_fields, 
                          GenXBindDynSpringToNodeFields *const genx_bind_dyn_spring_to_node_fields) {
  // Get the vector of left/right bound parts in the selector
  stk::mesh::PartVector left_bound_spring_parts;
  stk::mesh::PartVector doubly_bound_spring_parts;
  left_bound_springs.get_parts(left_bound_spring_parts);
  doubly_bound_springs.get_parts(doubly_bound_spring_parts);

  // Get the vector of entities to modify
  stk::mesh::EntityVector spring_sphere_linkers;
  stk::mesh::EntityVector doubly_bound_springs;
  stk::mesh::get_selected_entities(spring_sphere_genxs, bulk_data.buckets(stk::mesh::CONSTRAINT_RANK),
                                   spring_sphere_linkers);
  stk::mesh::get_selected_entities(doubly_bound_springs, bulk_data.buckets(stk::mesh::ELEMENT_RANK),
                                   doubly_bound_springs);

  bulk_data.modification_begin();

  // Perform L->D
  for (const stk::mesh::Entity &spring_sphere_genx : spring_sphere_linkers) {
    // Decode the binding type enum for this entity
    auto state_change_action =
        static_cast<BINDING_STATE_CHANGE>(stk::mesh::field_data(c_perform_state_change_field, spring_sphere_genx)[0]);
    const bool perform_state_change = state_change_action != BINDING_STATE_CHANGE::NONE;
    if (perform_state_change) {
      // Get our connections
      const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
          stk::mesh::field_data(c_linked_entities_field, spring_sphere_genx));
      const stk::mesh::Entity &spring = bulk_data.get_entity(key_t_ptr[0]);
      const stk::mesh::Entity &target_sphere = bulk_data.get_entity(key_t_ptr[1]);
      MUNDY_THROW_ASSERT(bulk_data.is_valid(spring), std::runtime_error,  "Encountered invalid crosslinker entity in state_change_crosslinkers.");
      MUNDY_THROW_ASSERT(bulk_data.is_valid(target_sphere),  std::runtime_error, "Encountered invalid sphere entity in state_change_crosslinkers.");

      // Call the binding function
      if (state_change_action == BINDING_STATE_CHANGE::LEFT_TO_DOUBLY) {
        // Unbind the right side of the crosslinker from the left node and bind it to
        // the target node
        const stk::mesh::Entity &target_sphere_node = bulk_data.begin_nodes(target_sphere)[0];
        const bool bind_worked =
            bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, spring, target_sphere_node, 1);
        MUNDY_THROW_ASSERT(bind_worked,  std::runtime_error, "Failed to bind crosslinker to node.");

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
        static_cast<BINDING_STATE_CHANGE>(stk::mesh::field_data(*el_perform_state_change_field_ptr_, spring)[0]);
    if (state_change_action == BINDING_STATE_CHANGE::DOUBLY_TO_LEFT) {
      // Unbind the right side of the crosslinker from the current node and bind it to
      // the left crosslinker node
      const stk::mesh::Entity &left_node = bulk_data.begin_nodes(spring)[0];
      const bool unbind_worked = bind_crosslinker_to_node_unbind_existing(*bulk_data_ptr_, spring, left_node, 1);
      MUNDY_THROW_ASSERT(unbind_worked,  std::runtime_error, "Failed to unbind crosslinker from node.");

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

}  // namespace dynamic_springs

void check_max_overlap_with_periphery(const Sphere &target_sphere, const bool invert_target_sphere,
                                      const stk::mesh::Selector &spheres, const SphereFields &sphere_fields,
                                      const double max_allowable_overlap) {
  double local_max_overlap = 0.0;
  const double sign = invert_target_sphere ? -1.0 : 1.0;
  stk::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres,
      [&n_coord_field, &element_radius_field, &target_sphere_center, &target_sphere_radius, &sign, &local_max_overlap](
          const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
        const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
        const auto node_coords = mundy::mesh::vector3_field_data(n_coord_field, sphere_node);
        const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
        const double ssd =
            sign * mundy::math::norm(node_coords - target_sphere_center) - sphere_radius - target_sphere_radius;

#pragma omp critical
        local_min_ssd = std::min(local_min_ssd, ssd);
      });

  double min_ssd = std::numeric_limits<double>::max();
  stk::all_reduce_min(*bulk_data_ptr_, &local_min_ssd, &min_ssd, 1);
  if (min_ssd < -max_allowable_overlap) {
    MUNDY_THROW_ASSERT(false,  std::runtime_error, "Sphere overlaps with hydrodynamic periphery more than the allowabe amount.");
  }
}

void check_max_overlap_with_periphery_fast_approx(const Ellipsoid &target_ellipsoid, const bool invert_target_ellipsoid,
                                                  const stk::mesh::Selector &spheres, const SphereFields &sphere_fields,
                                                  const double max_allowable_overlap) {
  std::cout << "###### WARNING #######" << std::endl;
  std::cout << "The fast approximation for ellipsoidal periphery ssd " << std::endl;
  std::cout << "  is UNFIT for use in scientific publications." << std::endl;
  std::cout << "###### WARNING #######" << std::endl;

  const double shifted_target_ellipsoid_r1 = target_ellipsoid_r1 + max_allowable_overlap;
  const double shifted_target_ellipsoid_r2 = target_ellipsoid_r2 + max_allowable_overlap;
  const double shifted_target_ellipsoid_r3 = target_ellipsoid_r3 + max_allowable_overlap;

  stk::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, chromatin_spheres,
      [&n_coord_field, &element_hydro_radius_field, &shifted_target_ellipsoid_r1, &shifted_target_ellipsoid_r2,
       &shifted_target_ellipsoid_r3](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
        const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
        const auto node_coords = mundy::mesh::vector3_field_data(n_coord_field, sphere_node);
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
        const double a2 = (shifted_target_ellipsoid_r1 - sphere_radius) * (shifted_target_ellipsoid_r1 - sphere_radius);
        const double b2 = (shifted_target_ellipsoid_r2 - sphere_radius) * (shifted_target_ellipsoid_r2 - sphere_radius);
        const double c2 = (shifted_target_ellipsoid_r3 - sphere_radius) * (shifted_target_ellipsoid_r3 - sphere_radius);
        const double value = x2 / a2 + y2 / b2 + z2 / c2;
        if (value > 1.0) {
#pragma omp critical
          {
            std::cout << "Sphere node " << bulk_data.identifier(sphere_node)
                      << " overlaps with the periphery more than the allowable threshold." << std::endl;
            std::cout << "  node_coords: " << node_coords << std::endl;
            std::cout << "  value: " << value << std::endl;
          }
          MUNDY_THROW_ASSERT(false,  std::runtime_error, "Sphere node outside hydrodynamic periphery.");
        }
      });
}

void compute_rpy_hydro(const double viscosity, const stk::mesh::Selector &spheres, const SphereFields &sphere_fields,
                       const MotileSphereFields &motile_sphere_fields, const bool validate_noslip_boundary = false,
                       const alens::Periphery *periphery = nullptr) {
  // Fetch the bucket of spheres to act on.
  stk::mesh::EntityVector sphere_elements;
  stk::mesh::get_selected_entities(spheres, bulk_data_ptr_->buckets(stk::topology::ELEMENT_RANK), sphere_elements);
  const size_t num_spheres = sphere_elements.size();

  // Copy the sphere positions, radii, forces, and velocities to Kokkos views
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions("sphere_positions", num_spheres * 3);
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii("sphere_radii", num_spheres);
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces("sphere_forces", num_spheres * 3);
  Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities("sphere_velocities", num_spheres * 3);

#pragma omp parallel for
  for (size_t i = 0; i < num_spheres; i++) {
    stk::mesh::Entity sphere_element = sphere_elements[i];
    stk::mesh::Entity sphere_node = bulk_data_ptr_->begin_nodes(sphere_element)[0];
    const double *sphere_position = stk::mesh::field_data(*n_coord_field_ptr_, sphere_node);
    const double *sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere_element);
    const double *sphere_force = stk::mesh::field_data(*n_force_field_ptr_, sphere_node);
    const double *sphere_velocity = stk::mesh::field_data(*n_velocity_field_ptr_, sphere_node);

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
      MUNDY_THROW_ASSERT(max_speed(surface_velocities) < 1.0e-10,  std::runtime_error, "No-slip boundary condition not satisfied.");
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
    double *sphere_force = stk::mesh::field_data(*n_force_field_ptr_, sphere_node);
    double *sphere_velocity = stk::mesh::field_data(*n_velocity_field_ptr_, sphere_node);

    for (size_t j = 0; j < 3; j++) {
      sphere_force[j] = sphere_forces(i * 3 + j);
      sphere_velocity[j] = sphere_velocities(i * 3 + j);
    }
  }
}

void compute_periphery_collision_forces(const Ellipsoid &periphery, const stk::mesh::Selector &spheres,
                                        const SphereFields &sphere_fields,
                                        const MotileSphereFields &motile_sphere_fields,
                                        const double collision_spring_constant) {
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
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres,
      [&n_coord_field, &n_force_field, &el_aabb_field, &element_radius_field, &level_set, &center, &orientation,
       &periphery_r1, &periphery_r2, &periphery_r3,
       &spring_constant](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
        // For our coarse search, we check if the coners of the sphere's aabb lie inside
        // the ellipsoidal periphery This can be done via the (body frame) inside
        // outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2 + z^2/c^2) This is
        // possible due to the convexity of the ellipsoid
        const double *sphere_aabb = stk::mesh::field_data(el_aabb_field, sphere_element);
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
            level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 && level_set(bottom_left_back) < 0.0 &&
            level_set(bottom_right_back) < 0.0 && level_set(top_left_back) < 0.0 && level_set(top_right_back) < 0.0;

        if (!all_points_inside_periphery) {
          // We might have a collision, perform the more expensive check
          const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
          const auto node_coords = mundy::mesh::vector3_field_data(n_coord_field, sphere_node);
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
            auto node_force = mundy::mesh::vector3_field_data(n_force_field, sphere_node);
            auto periphery_nhat = -ellipsoid_nhat;
            node_force[0] -= spring_constant * periphery_nhat[0] * shared_normal_ssd;
            node_force[1] -= spring_constant * periphery_nhat[1] * shared_normal_ssd;
            node_force[2] -= spring_constant * periphery_nhat[2] * shared_normal_ssd;
          }
        }
      });
}

void compute_periphery_collision_forces_fast_approx(const Ellipsoid &periphery, const stk::mesh::Selector &spheres,
                                                    const SphereFields &sphere_fields,
                                                    const MotileSphereFields &motile_sphere_fields,
                                                    const double collision_spring_constant) {
  std::cout << "###### WARNING #######" << std::endl;
  std::cout << "The fast approximation for ellipsoidal periphery collision " << std::endl;
  std::cout << "  is UNFIT for use in scientific publications." << std::endl;
  std::cout << "###### WARNING #######" << std::endl;

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
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres,
      [&n_coord_field, &n_force_field, &element_radius_field, &level_set, &outward_normal, &collision_spring_constant](
          const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
        // Do a fast loop over all of the spheres we are checking, e.g., brute-force the
        // calc.
        const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
        const auto node_coords = mundy::mesh::vector3_field_data(n_coord_field, sphere_node);
        const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];

        // Simply check if we are outside the sphere via the level-set function
        if (level_set(sphere_radius, node_coords) > 0.0) {
          auto node_force = mundy::mesh::vector3_field_data(n_force_field, sphere_node);

          // Compute the outward normal
          auto out_normal = outward_normal(sphere_radius, node_coords);
          node_force[0] -= collision_spring_constant * out_normal[0];
          node_force[1] -= collision_spring_constant * out_normal[1];
          node_force[2] -= collision_spring_constant * out_normal[2];
        }
      });
}

void compute_periphery_collision_forces(const Sphere &periphery, const stk::mesh::Selector &spheres,
                                        const SphereFields &sphere_fields,
                                        const MotileSphereFields &motile_sphere_fields,
                                        const double collision_spring_constant) {
  stk::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, spheres,
      [&n_coord_field, &element_radius_field, &periphery_center, &periphery_radius, &collision_spring_constant](
          const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_element) {
        const stk::mesh::Entity sphere_node = bulk_data.begin_nodes(sphere_element)[0];
        const auto node_coords = mundy::mesh::vector3_field_data(n_coord_field, sphere_node);
        const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
        const double shared_normal_ssd =
            periphery_radius - mundy::math::norm(node_coords - periphery_center) - sphere_radius;
        const bool sphere_collides_with_periphery = shared_normal_ssd < 0.0;
        if (sphere_collides_with_periphery) {
          auto node_force = mundy::mesh::vector3_field_data(n_force_field, sphere_node);
          auto inward_normal = (node_coords - periphery_center) / mundy::math::norm(node_coords - periphery_center);
          node_force[0] -= collision_spring_constant * inward_normal[0] * shared_normal_ssd;
          node_force[1] -= collision_spring_constant * inward_normal[1] * shared_normal_ssd;
          node_force[2] -= collision_spring_constant * inward_normal[2] * shared_normal_ssd;
        }
      });
}

void compute_brownian_motion(const double &timestep_size, const double &viscosity, const double &brownian_kt,
                             const stk::mesh::Selector &spheres, SphereFields &sphere_fields,
                             stk::mesh::Field<double> &n_rng_field) {
  const double sphere_drag_coeff = 6.0 * M_PI * viscosity * sphere_radius;
  const double inv_drag_coeff = 1.0 / sphere_drag_coeff;
  stk::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::NODE_RANK, selector,
      [&n_velocity_field, &n_force_field, &n_rng_field, &timestep_size, &sphere_drag_coeff, &inv_drag_coeff,
       &brownian_kt](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
        // Get the specific values for each sphere
        double *node_velocity = stk::mesh::field_data(n_velocity_field, sphere_node);
        const stk::mesh::EntityId sphere_node_gid = bulk_data.identifier(sphere_node);
        unsigned *node_rng_counter = stk::mesh::field_data(n_rng_field, sphere_node);

        // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
        openrand::Philox rng(sphere_node_gid, node_rng_counter[0]);
        const double coeff = std::sqrt(2.0 * brownian_kt * sphere_drag_coeff / timestep_size) * inv_drag_coeff;
        node_velocity[0] += coeff * rng.randn<double>();
        node_velocity[1] += coeff * rng.randn<double>();
        node_velocity[2] += coeff * rng.randn<double>();
        node_rng_counter[0]++;
      });
}

void compute_dry_local_drag(const double &viscosity, const stk::mesh::Selector &spheres,
                            const SphereFields &sphere_fields, MotileSphereFields &motile_sphere_fields) {
  const double inv_drag_coeff = 1.0 / (6.0 * M_PI * viscosity * sphere_radius);
  stk::mesh::for_each_entity_run(bulk_data, stk::topology::NODE_RANK, selector,
                                 [&n_force_field, &n_velocity_field, &inv_drag_coeff](
                                     const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_node) {
                                   double *node_velocity = stk::mesh::field_data(n_velocity_field, sphere_node);
                                   double *node_force = stk::mesh::field_data(n_force_field, sphere_node);

                                   // Uext = Fext * inv_drag_coeff
                                   node_velocity[0] += node_force[0] * inv_drag_coeff;
                                   node_velocity[1] += node_force[1] * inv_drag_coeff;
                                   node_velocity[2] += node_force[2] * inv_drag_coeff;
                                 });
}

void node_euler_position_integrator(const double &timestep_size, const stk::mesh::Selector &selector,
                                    const stk::mesh::Field<double> &old_n_coord_field,
                                    const stk::mesh::Field<double> &new_n_coord_field,
                                    const stk::mesh::Field<double> &n_velocity_field) {
  stk::mesh::for_each_entity_run(bulk_data, stk::topology::NODE_RANK, selector,
                                 [&old_n_coord_field, &new_n_coord_field, &n_velocity_field, &timestep_size](
                                     const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &node) {
                                   const auto old_node_coords =
                                       mundy::mesh::vector3_field_data(old_n_coord_field, node);
                                   const auto node_velocity = mundy::mesh::vector3_field_data(n_velocity_field, node);
                                   auto new_node_coords = mundy::mesh::vector3_field_data(new_n_coord_field, node);

                                   // x(t+dt) = x(t) + dt * v(t)
                                   new_node_coords[0] = old_node_coords[0] + timestep_size * node_velocity[0];
                                   new_node_coords[1] = old_node_coords[1] + timestep_size * node_velocity[1];
                                   new_node_coords[2] = old_node_coords[2] + timestep_size * node_velocity[2];
                                 });
}

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
    return std::string("COORDS");
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
  // Un-namespace anything that is clear from the context
  using stk::topology::CONSTRAINT_RANK;
  using stk::topology::EDGE_RANK;
  using stk::topology::ELEMENT_RANK;
  using stk::topology::FACE_RANK;
  using stk::topology::NODE_RANK;

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
    meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition to
                                         // all fields are simple.
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
    stk::mesh::MetaData &meta_data = *meta_data_ptr;
    stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

    // Parts and their subsets
    auto &e_part = mundy::declare_io_part(meta_data, "EUCHROMATIN_SPHERES", stk::topology::PARTICLE);
    auto &h_part = mundy::declare_io_part(meta_data, "HETEROCHROMATIN_SPHERES", stk::topology::PARTICLE);
    auto &bs_part = mundy::declare_io_part(meta_data, "BIND_SITES", stk::topology::NODE);

    auto &hp1_part = mundy::declare_io_part(meta_data, "HP1", stk::topology::BEAM_2);
    auto &left_hp1_part = mundy::declare_io_part(meta_data, "LEFT_HP1", stk::topology::BEAM_2);
    auto &doubly_hp1_h_part = mundy::declare_io_part(meta_data, "DOUBLY_HP1_H", stk::topology::BEAM_2);
    auto &doubly_hp1_bs_part = mundy::declare_io_part(meta_data, "DOUBLY_HP1_BS", stk::topology::BEAM_2);
    meta_data.declare_part_subset(hp1_part, left_hp1_part);
    meta_data.declare_part_subset(hp1_part, doubly_hp1_h_part);
    meta_data.declare_part_subset(hp1_part, doubly_hp1_bs_part);

    auto &backbone_segs_part = mundy::declare_io_part(meta_data, "BACKBONE_SEGMENTS", stk::topology::BEAM_2);
    auto &ee_segs_part = mundy::declare_io_part(meta_data, "EE_SEGMENTS", stk::topology::BEAM_2);
    auto &eh_segs_part = mundy::declare_io_part(meta_data, "EH_SEGMENTS", stk::topology::BEAM_2);
    auto &hh_segs_part = mundy::declare_io_part(meta_data, "HH_SEGMENTS", stk::topology::BEAM_2);
    meta_data.declare_part_subset(backbone_segs_part, ee_segs_part);
    meta_data.declare_part_subset(backbone_segs_part, eh_segs_part);
    meta_data.declare_part_subset(backbone_segs_part, hh_segs_part);

    // Fields
    auto &n_coord_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
    auto &n_velocity_field = meta_data.declare_field<double>(NODE_RANK, "VELOCITY");
    auto &n_force_field = meta_data.declare_field<double>(NODE_RANK, "FORCE");
    auto &n_rng_field = meta_data.declare_field<unsigned>(NODE_RANK, "RNG_COUNTER");

    auto &el_hydrodynamic_radius_field = meta_data.declare_field<double>(ELEMENT_RANK, "HYDRODYNAMIC_RADIUS");
    auto &el_binding_radius_field = meta_data.declare_field<double>(ELEMENT_RANK, "BINDING_RADIUS");
    auto &el_collision_radius_field = meta_data.declare_field<double>(ELEMENT_RANK, "COLLISION_RADIUS");
    auto &el_hookean_spring_constant_field = meta_data.declare_field<double>(ELEMENT_RANK, "SPRING_CONSTANT");
    auto &el_hookean_spring_rest_length_field = meta_data.declare_field<double>(ELEMENT_RANK, "SPRING_REST_LENGTH");
    auto &el_youngs_modulus_field = meta_data.declare_field<double>(ELEMENT_RANK, "YOUNGS_MODULUS");
    auto &el_poissons_ratio_field = meta_data.declare_field<double>(ELEMENT_RANK, "POISSONS_RATIO");
    auto &el_aabb_field = meta_data.declare_field<double>(ELEMENT_RANK, "AABB");
    auto &el_aabb_displacement_field = meta_data.declare_field<double>(ELEMENT_RANK, "AABB_DISPLACEMENT");
    auto &el_binding_rates_field = meta_data.declare_field<double>(ELEMENT_RANK, "BINDING_RATES");
    auto &el_unbinding_rates_field = meta_data.declare_field<double>(ELEMENT_RANK, "UNBINDING_RATES");
    auto &el_perform_state_change_field = meta_data.declare_field<unsigned>(ELEMENT_RANK, "PERFORM_STATE_CHANGE");
    auto &el_rng_field = meta_data.declare_field<unsigned>(ELEMENT_RANK, "RNG_COUNTER");

    // Sew it all together
    // Any field with a constant initial value should have that value set here. If it is
    // to be uninitialized, use nullptr.
    const double zero_vector3d[3] = {0.0, 0.0, 0.0};
    const double zero_scalar[1] = 0.0;
    const unsigned zero_unsigned[1] = 0;
    stk::mesh::put_field_on_entire_mesh(n_coord_field, nullptr);

    // Heterochromatin and euchromatin spheres are used for hydrodynamics. They move and
    // have forces applied to them. If brownian motion is enabled, they will have a
    // stocastic velocity. Heterochromatin spheres are considered for hp1 binding and
    // require an AABB for neighbor detection.
    stk::mesh::put_field_on_mesh(n_velocity_field, e_part | h_part, 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(n_force_field, e_part | h_part, 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(n_rng_field, e_part | h_part, 1, zero_unsigned);
    stk::mesh::put_field_on_mesh(el_hydrodynamic_radius_field, e_part | h_part, 1, chromatin_hydrodynamic_radius);
    stk::mesh::put_field_on_mesh(el_aabb_field, h_part, 6, nullptr);
    stk::mesh::put_field_on_mesh(el_aabb_displacement_field, h_part, 6, nullptr);

    // Backbone segs apply spring forces and act as spherocylinders for the sake of
    // collision. They apply forces to their nodes and have a collision radius. The
    // difference between ee, eh, and hh segs is that ee segs can exert an active
    // dipole.
    stk::mesh::put_field_on_mesh(n_force_field, backbone_segs_part, 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(el_collision_radius_field, backbone_segs_part, 1, backbone_collision_radius);
    stk::mesh::put_field_on_mesh(el_hookean_spring_constant_field, backbone_segs_part, 1, backbone_spring_constant);
    stk::mesh::put_field_on_mesh(el_hookean_spring_rest_length_field, backbone_segs_part, 1, backbone_rest_length);
    stk::mesh::put_field_on_mesh(el_youngs_modulus_field, backbone_segs_part, 1, collision_youngs_modulus);
    stk::mesh::put_field_on_mesh(el_poissons_ratio_field, backbone_segs_part, 1, collision_poissons_ratio);
    stk::mesh::put_field_on_mesh(el_aabb_field, backbone_segs_part, 6, nullptr);
    stk::mesh::put_field_on_mesh(el_aabb_displacement_field, backbone_segs_part, 6, nullptr);

    // HP1 crosslinkers are used for binding/unbinding and apply forces to their nodes.
    const double left_and_right_binding_rates[2] = {left_binding_rate, right_binding_rate};
    const double left_and_right_unbinding_rates[2] = {left_unbinding_rate, right_unbinding_rate};
    stk::mesh::put_field_on_mesh(n_force_field, hp1_part, 3, zero_vector3d);
    stk::mesh::put_field_on_mesh(el_binding_rates_field, hp1_part, 2, left_and_right_binding_rates);
    stk::mesh::put_field_on_mesh(el_unbinding_rates_field, hp1_part, 2, left_and_right_unbinding_rates);
    stk::mesh::put_field_on_mesh(el_perform_state_change_field, hp1_part, 1, zero_unsigned);
    stk::mesh::put_field_on_mesh(el_binding_radius_field, hp1_part, 1, crosslinker_binding_radius);
    stk::mesh::put_field_on_mesh(el_rng_field, hp1_part, 1, zero_unsigned);

    // That's it for the mesh. Commit it's structure and create the bulk data.
    meta_data.commit();
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
    stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

    if (!restart_performed_) {
      std::vector<mundy::NodeInfo> node_info;
      std::vector<mundy::ElementInfo> element_info;

      /* Declare the chromatin and HP1
      //  E : euchromatin spheres
      //  H : heterochromatin spheres
      //  | : crosslinkers
      // ---: backbone springs/backbone segments
      //
      //  |   |                           |   |
      //  H---H---E---E---E---E---E---E---H---H
      //
      // The actual connectivity looks like this:
      //  n : node, s : segment and or spring, c : crosslinker
      //
      // c1_      c3_       c5_       c7_
      // | /      | /       | /       | /
      // n1       n3        n5        n7
      //  \      /  \      /  \      /
      //   s1   s2   s3   s4   s5   s6
      //    \  /      \  /      \  /
      //     n2        n4        n6
      //     | \       | \       | \
      //     c2⎻       c4⎻       c6⎻
      //
      // If you look at this long enough, the pattern is clear.
      //  - One less segegment than nodes.
      //  - Same number of crosslinkers as heterochromatin nodes.
      //  - Segement i connects to nodes i and i+1.
      //  - Crosslinker i connects to nodes i and i.
      //
      // We need to use this information to populate the node and element info vectors.
      // Mundy will handle passing off this information to the bulk data. Just make sure that all
      // MPI ranks contain the same node and element info. This way, we can determine which nodes
      // should become shared.
      //
      // Rules (non-exhaustive):
      //  - Neither nodes nor elements need to have parts or fields.
      //  - The rank and type of the fields must be consistant. You can't pass an element field to a node,
      //    nor can you set the value of a field to a different type or size than it was declared as.
      //  - The owner of a node must be the same as one of the elements that connects to it.
      //  - A node connected to an element not on the same rank as the node will be shared with the owner of the
      element.
      //  - Field/Part names are case-sensitive but don't attempt to declare "field_1" and "Field_1" as if
      //    that will give two different fields since STKIO will not be able to distinguish between them.
      //  - A negative node id in the element connection list can be used to indicate that a node should be left
      unassigned.
      //  - All parts need to be able to contain an element of the given topology.
      //
      // std::vector<NodeInfo> nodes = {
      //     {0, 1, {"N_PART", "OTHER_PART"}, {}}, // No fields for this node
      //     {0, 2, {"N_PART"}, {{"COORDS", std::vector<double>{2.0, 3.0, 4.0}}}}, // Node with fields
      // };
      //
      // std::vector<ElementInfo> elements = {
      //     {0, 1, BEAM_2, {1, 2}, {}, {}},                             // No fields or parts for this element
      //     {0, 2, BEAM_2, {2, 3}, {"BEAM_PART"}, {{"RADIUS", 0.75}}},  // Element with fields
      // };
      */
      int owning_proc = 0;
      int64_t node_id = 1;
      int64_t element_id = 1;

      // Helpers to reduce code duplication
      auto add_node = [&node_info, &n_coord_field](int owning_proc, int64_t node_id,
                                                   const std::array<double, 3> &coords) {
        node_info.push_back({owning_proc, node_id, {}, {{&n_coord_field, coords}}});
      };
      auto add_element = [&element_info](int owning_proc, int64_t element_id, const std::vector<int64_t> &node_ids,
                                         stk::topology topology, const std::vector<std::string> &parts) {
        element_info.push_back({owning_proc, element_id, topology, node_ids, parts, {}});
      };

      // Compute the node coordinates for each chromosom
      std::vector<std::vector<std::array<double, 3>>> chromosome_positions;
      if (initialization_type == INITIALIZATION_TYPE::GRID) {
        get_chromosome_positions_grid(chromosome_positions, num_chromosomes, num_chromatin_repeats,
                                      num_euchromatin_per_repeat, num_heterochromatin_per_repeat,
                                      initial_chromosome_separation, n_coord_field);
      } else if (initialization_type_ == INITIALIZATION_TYPE::RANDOM_UNIT_CELL) {
        get_chromosome_positions_random_unit_cell(chromosome_positions, num_chromosomes, num_chromatin_repeats,
                                                  num_euchromatin_per_repeat, num_heterochromatin_per_repeat,
                                                  initial_chromosome_separation, unit_cell_size, n_coord_field);
      } else if (initialization_type_ == INITIALIZATION_TYPE::HILBERT_RANDOM_UNIT_CELL) {
        get_chromosome_positions_hilbert_random_unit_cell(chromosome_positions, num_chromosomes, num_chromatin_repeats,
                                                          num_euchromatin_per_repeat, num_heterochromatin_per_repeat,
                                                          initial_chromosome_separation, unit_cell_size, n_coord_field);
      } else {
        MUNDY_THROW_ASSERT(false,  std::runtime_error, "Unknown initialization type: " << initialization_type_);
      }

      // Loop over chromosomes, chromatin repeats, and chromatin beads
      for (size_t i = 0; i < num_chromosomes; ++i) {
        for (size_t j = 0; j < num_chromatin_repeats; ++j) {
          // Heterochromatin
          for (size_t k = 0; k < num_heterochromatin_per_repeat; ++k) {
            const auto &coords =
                chromosome_positions[i][j * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat) + k];
            add_node(owning_proc, node_id, coords);
            add_element(owning_proc, element_id, {node_id - 1, node_id}, stk::topology::PARTICLE, {&h_part});
            ++node_id;
            ++element_id;

            // Backbone segment/spring
            if ((k != num_heterochromatin_per_repeat - 1) &&
                (enable_backbone_collision || enable_backbone_springs || enable_heterochromatin_activity)) {
              stk::mesh::PartVector parts = {};
              if (enable_backbone_collision || enable_backbone_springs) {
                parts.push_back(&backbone_segs_part);
              }
              if (enable_heterochromatin_activity) {
                // Logic-wise, we allow all k values except k=0 and k=num_heterochromatin_per_repeat-1 with the
                // exception of k=0 if j=0 and k=num_heterochromatin_per_repeat-1 if j=num_chromatin_repeats-1. The or
                // statements accounts for these exceptions.
                const bool left_and_right_node_in_heterochromatin =
                    (k != 0 || j == 0) && (k != num_heterochromatin_per_repeat - 1 || j == num_chromatin_repeats - 1);
                if (left_and_right_node_in_heterochromatin) {
                  parts.push_back(&hh_segs_part);
                }
              }

              add_element(owning_proc, element_id, {node_id, node_id + 1}, stk::topology::BEAM_2, parts);
              ++element_id;
            }

            // HP1 (if enabled)
            if (enable_hp1) {
              add_element(owning_proc, element_id, stk::topology::BEAM_2, {node_id, node_id}, {&hp1_part});
              ++element_id;
            }
          }

          // Euchromatin
          for (size_t k = 0; k < num_euchromatin_per_repeat; ++k) {
            const auto &coords =
                chromosome_positions[i][j * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat) +
                                        num_heterochromatin_per_repeat + k];
            add_node(owning_proc, node_id, coords);
            add_element(owning_proc, element_id, {node_id, node_id + 1}, stk::topology::PARTICLE, {&e_part});
            ++node_id;
            ++element_id;

            // Backbone segment/spring
            if ((k != num_heterochromatin_per_repeat - 1) && (enable_backbone_collision || enable_backbone_springs)) {
              stk::mesh::PartVector parts = {&e_part};
              if (enable_backbone_collision || enable_backbone_springs) {
                parts.push_back(&backbone_segs_part);
              }
              add_element(owning_proc, element_id, {node_id, node_id + 1}, stk::topology::BEAM_2, parts);
              ++element_id;
            }
          }
        }
      }

      // Loop over the bindng sites
      if (enable_periphery_binding_) {
        // Compute the node coordinates for each binding site
        std::vector<std::array<double, 3>> binding_site_positions;
        if (periphery_bind_sites_type_ == BIND_SITES_TYPE::RANDOM) {
          get_binding_site_positions_random(binding_site_positions, num_periphery_bind_sites, periphery, n_coord_field);
        } else if (periphery_bind_sites_type_ == BIND_SITES_TYPE::FROM_FILE) {
          get_binding_site_positions_from_file(binding_site_positions, bind_site_coords_filename, n_coord_field);
        } else {
          MUNDY_THROW_ASSERT(false,  std::runtime_error, "Unknown periphery bind sites type: " << periphery_bind_sites_type_);
        }

        for (size_t i = 0; i < num_periphery_bind_sites; ++i) {
          const auto &coords = binding_site_positions[i];
          add_node(owning_proc, node_id, coords);
          add_element(owning_proc, element_id, {node_id}, stk::topology::NODE,
                      {&bs_part});  // TODO(palmerb4): How to handle NODE topology?
          ++node_id;
          ++element_id;
        }
      }

      mundy::declare_entities(bulk_data, node_info, element_info);
    }

    // Post setup but pre-run
    if (loadbalance_post_initialization_) {
      mundy::loadbalance(bulk_data, RcbSettings());
    }

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
    for (size_t timestep_idx = 0; timestep_idx < num_time_steps_; timestep_idx++) {
      // Prepare the current configuration.
      mundy::deep_copy(n_velocity_field, std::array<double, 3>{0.0, 0.0, 0.0});
      mundy::deep_copy(n_force_field, std::array<double, 3>{0.0, 0.0, 0.0});
      mundy::deep_copy(el_binding_rates_field, std::array<double, 2>{0.0, 0.0});
      mundy::deep_copy(el_unbinding_rates_field, std::array<double, 2>{0.0, 0.0});
      mundy::deep_copy(el_perform_state_change_field, std::array<unsigned, 1>{0u});
      mundy::deep_copy(c_perform_state_change_field, std::array<unsigned, 1>{0u});
      mundy::deep_copy(c_state_change_rate_field, std::array<double, 1>{0.0});
      mundy::deep_copy(c_potential_force_field, std::array<double, 3>{0.0, 0.0, 0.0});

      // If we are doing a compression run, shrink the periphery
      if (enable_periphery_collision_ && shrink_periphery_over_time_ &&
          (timestep_idx < periphery_collision_shrinkage_num_steps_)) {
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
      mundy::compute_aabb(segs, skin_distance, el_aabb_field);
      mundy::compute_aabb(spheres, skin_distance, el_aabb_field);
      const double alpha = 1.0;
      const double beta = 1.0;
      mundy::compute_aabb_displacements(element_old_aabb_field, el_aabb_field, el_aabb_displacement_field, alpha, beta);

      // Check if we need to update the neighbor list. Eventually this will be replaced
      // with a mesh attribute to synchronize across multiple tasks. For now, make sure
      // that the default is to not update neighbor lists.
      double max_aabb_displacement = 0.0;
      mundy::compute_max_aabb_displacement(el_aabb_displacement_field, max_aabb_displacement);

      // Now do a check to see if we need to update the neighbor list.
      if (max_aabb_displacement > skin_distance) {
        // Reset the accumulated AABB displacements
        mundy::deep_copy(el_aabb_displacement_field, std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        // Update the neighbor list
        if (enable_backbone_collision_ || enable_crosslinkers_ || enable_periphery_binding_) {
          const double max_allowed_distance = 0.0;  // Only destroy non-overlapping neighbors
          const auto neighbor_selector = backbone_backbone_linker_part | spring_sphere_linker_part;
          mundy::destroy_distant_neighbors(neighbor_selector, max_allowed_distance, el_aabb_field);
        }

        // Generate the GENX neighbor linkers
        if (enable_backbone_collision_) {
          // Inputs are: Source, target, source_aabb, target_aabb, output_part
          mundy::generate_neighbor_linkers(backbone_segs_selector, backbone_segs_selector, el_aabb_field, el_aabb_field,
                                           backbone_backbone_linker_part);
          mundy::destroy_bound_neighbors(backbone_backbone_linker_selector);
        }
        if (enable_crosslinkers_) {
          mundy::generate_neighbor_linkers(hp1_selector, h_selector, el_aabb_field, el_aabb_field, hp1_h_linker_part);
        }
        if (enable_periphery_binding_) {
          mundy::generate_neighbor_linkers(hp1_selector, bs_selector, el_aabb_field, el_aabb_field, hp1_bs_linker_part);
        }
      }

      if (enable_crosslinkers_) {
        mundy::compute_state_change_rate_attach_left_bound_attach_to_node(
            binding_kt, dynamic_springs, dynamic_springs_to_node_linkers, c_state_change_rate_field);
        mundy::kmc_decide_state_change_left_bound_attach_to_node(
            timestep_size, dynamic_springs, dynamic_springs_to_node_linkers, c_state_change_rate_field,
            c_perform_state_change_field);
        mundy::kmc_decide_state_change_detach_doubly_bound_from_node(timestep_size, dynamic_springs);
        mundy::perform_state_change(dynamic_springs, dynamic_springs_to_node_linkers, c_perform_state_change_field);
      }

      // Evaluate forces f(x(t)).
      if (enable_backbone_collision_) {
        // Potential evaluation (Hertzian contact)
        mundy::compute_signed_sep_dist_contact_normal_and_contact_points(backbone_backbone_neighbors);
        mundy::compute_hertzian_contact_forces(backbone_backbone_neighbors, constraint_contact_force_field);
        mundy::sum_contact_forces(backbone_segs, backbone_backbone_neighbors, constraint_contact_force_field,
                                  n_force_field);
      }

      if (enable_backbone_springs_) {
        mundy::compute_hookean_spring_forces(backbone_springs, n_force_field);
      }

      if (enable_crosslinkers_) {
        // Select only active springs in the system. Aka, not left bound.
        mundy::compute_hookean_spring_forces(doubly_bound_dynamic_springs, n_force_field);
      }

      if (enable_periphery_collision_) {
        mundy::compute_periphery_collision_forces(periphery, spheres, periphery_collision_spring_constant);
      }

      // Compute velocities.
      if (enable_chromatin_brownian_motion_) {
        mundy::compute_brownian_velocity(timestep_size, viscosity, brownian_kt, spheres, node_rng_counter_field,
                                         n_force_field, n_velocity_field);
      }

      if (enable_backbone_n_body_hydrodynamics_) {
        const bool validate_noslip_boundary = true;
        mundy::compute_rpy_hydrodynamics(viscosity, hydrodynamic_spheres, no_slip_periphery_evaluator, n_force_field,
                                         n_velocity_field, validate_noslip_boundary);
      } else {
        mundy::compute_dry_local_drag(viscosity, spheres, n_force_field, n_velocity_field);
      }

      // Logging, if desired, write to console
      if (timestep_idx % log_frequency_ == 0) {
        if (bulk_data.parallel_rank() == 0) {
          double tps = static_cast<double>(log_frequency_) / static_cast<double>(timer.seconds());
          std::cout << "Step: " << std::setw(15) << timestep_idx << ", tps: " << std::setprecision(15) << tps;
          if (enable_periphery_collision_ && shrink_periphery_over_time_ &&
              timestep_idx < periphery_collision_shrinkage_num_steps_) {
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
      if (timestep_idx % io_frequency_ == 0) {
        io_broker_ptr_->write_io_broker_timestep(static_cast<int>(timestep_idx), static_cast<double>(timestep_idx));
      }

      // Update positions. x(t + dt) = x(t) + dt * v(t).
      mundy::integrate_positions_node_euler(timestep_size, hydrodynamic_spheres, n_velocity_field);
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data.parallel());
    if (bulk_data.parallel_rank() == 0) {
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
