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
#include <openrand/philox.h>

// C++ core
#include <algorithm>  // for std::copy_if, std::back_inserter
#include <cassert>    // for assert
#include <cmath>      // for std::sqrt
#include <iostream>   // for std::cout, std::endl
#include <span>       // for std::span
#include <vector>     // for std::vector

// Trilinos libs
#include <Kokkos_Core.hpp>                       // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer
#include <Kokkos_UnorderedMap.hpp>               // for Kokkos::UnorderedMap
#include <Teuchos_CommandLineProcessor.hpp>      // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>             // for Teuchos::ParameterList
#include <Teuchos_YamlParameterListHelpers.hpp>  // for Teuchos::getParametersFromYamlFile
#include <stk_balance/balance.hpp>               // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings
#include <stk_io/StkMeshIoBroker.hpp>            // for stk::io::StkMeshIoBroker
#include <stk_mesh/base/Comm.hpp>                // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>        // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>              // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>               // for stk::mesh::Field, stk::mesh::field_data
#include <stk_mesh/base/ForEachEntity.hpp>       // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>                // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>            // for stk::mesh::Selector
#include <stk_topology/topology.hpp>             // for stk::topology
#include <stk_util/parallel/Parallel.hpp>        // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy libs
#include <mundy_core/MakeStringArray.hpp>                                     // for mundy::core::make_string_array
#include <mundy_core/throw_assert.hpp>                                        // for MUNDY_THROW_ASSERT
#include <mundy_io/IOBroker.hpp>                                              // for mundy::io::IOBroker
#include <mundy_linkers/ComputeSignedSeparationDistanceAndContactNormal.hpp>  // for mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal
#include <mundy_linkers/DestroyNeighborLinkers.hpp>         // for mundy::linkers::DestroyNeighborLinkers
#include <mundy_linkers/EvaluateLinkerPotentials.hpp>       // for mundy::linkers::EvaluateLinkerPotentials
#include <mundy_linkers/GenerateNeighborLinkers.hpp>        // for mundy::linkers::GenerateNeighborLinkers
#include <mundy_linkers/LinkerPotentialForceReduction.hpp>  // for mundy::linkers::LinkerPotentialForceReduction
#include <mundy_linkers/NeighborLinkers.hpp>                // for mundy::linkers::NeighborLinkers
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>
#include <mundy_math/Matrix3.hpp>     // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion, mundy::math::quat_from_parallel_transport
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>    // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MetaData.hpp>    // for mundy::mesh::MetaData
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value
#include <mundy_meta/FieldReqs.hpp>                 // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>                  // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>                  // for mundy::meta::PartReqs
#include <mundy_shapes/ComputeAABB.hpp>             // for mundy::shapes::ComputeAABB

namespace impl {

template <size_t... Is, typename FieldToCopyType>
void copy_entity_field_impl(std::index_sequence<Is...>, const stk::mesh::Entity &parent, const stk::mesh::Entity &child,
                            const FieldToCopyType &field_to_copy) {
  // The size of the index sequence should be the size of the field to copy
  ((stk::mesh::field_data(field_to_copy, child)[Is] = stk::mesh::field_data(field_to_copy, parent)[Is]), ...);
}

/// @brief Copy fields from the parent entity to the child entity.
/// How to use:
/// copy_entity_fields(std::make_index_sequence<sizeof...(FieldToCopyType)>(), bulk_data, parent, child,
///   Kokkos::Array{1, 3, 1}, scalar_field1, vector3_field, scalar_field2);
/// @param bulk_data
/// @param parent
/// @param child
/// @param size_of_fields_to_copy
/// @param ...fields_to_copy
template <size_t N, Kokkos::Array<unsigned, N> size_of_fields_to_copy, size_t... Is, typename... FieldToCopyType>
  requires(sizeof...(FieldToCopyType) == N)
void copy_entity_fields_impl(std::index_sequence<Is...>, const stk::mesh::Entity &parent,
                             const stk::mesh::Entity &child, const std::tuple<FieldToCopyType...> &fields_to_copy) {
  // Unpack and copy each field.
  (copy_entity_field_impl(std::make_index_sequence<size_of_fields_to_copy[Is]>(), parent, child,
                          std::get<Is>(fields_to_copy)),
   ...);
}

template <size_t... Is, typename... FieldTypes>
void append_fields_to_field_pointer_vector_impl(std::index_sequence<Is...>,
                                                const std::tuple<FieldTypes...> &fields_to_copy,
                                                std::vector<const stk::mesh::FieldBase *> &field_pointers) {
  (field_pointers.push_back(&std::get<Is>(fields_to_copy)), ...);
}

}  // namespace impl

/// @brief Copy field from the parent entity to the child entity.
///
/// How to use:
/// copy_entity_field<3>(bulk_data, parent, child, vector3_field);
template <unsigned size_of_field_to_copy, typename FieldToCopyType>
void copy_entity_field(const stk::mesh::Entity &parent, const stk::mesh::Entity &child,
                       const FieldToCopyType &field_to_copy) {
  impl::copy_entity_field_impl(std::make_index_sequence<size_of_field_to_copy>(), parent, child, field_to_copy);
}

/// @brief Copy fields from the parent entity to the child entity.
///
/// How to use:
/// constexpr size_t num_fields = 4;
/// constexpr Kokkos::Array<unsigned, num_fields> size_of_fields_to_copy = {1, 3, 1, 1};
/// copy_entity_fields<num_fields, size_of_fields_to_copy>(parent, child,
///    scalar_field1, vector3_field, scalar_field2, scalar_field3);
template <size_t N, Kokkos::Array<unsigned, N> size_of_fields_to_copy, typename... FieldToCopyType>
  requires(sizeof...(FieldToCopyType) == N)
void copy_entity_fields(const stk::mesh::Entity &parent, const stk::mesh::Entity &child,
                        const std::tuple<FieldToCopyType...> &fields_to_copy) {
  constexpr size_t num_fields = sizeof...(FieldToCopyType);
  impl::copy_entity_fields_impl<N, size_of_fields_to_copy>(std::make_index_sequence<num_fields>(), parent, child,
                                                           fields_to_copy);
}

void get_parent_entity_parts(stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &parent,
                             stk::mesh::ConstPartVector &child_parts) {
  // Adding/removing entities from internally managed parts is not allowed.
  // These must be excluded in order to use `bulk_data.change_entity_part()`
  std::set<const stk::mesh::Part *> internal_parts = {
      &bulk_data.mesh_meta_data().universal_part(), &bulk_data.mesh_meta_data().locally_owned_part(),
      &bulk_data.mesh_meta_data().globally_shared_part(), &bulk_data.mesh_meta_data().aura_part()};

  // Get parent parts including internal parts
  const stk::mesh::PartVector &parent_parts = bulk_data.bucket(parent).supersets();

  // Copy parts, excluding internal parts, into `child_parts`
  std::copy_if(
      parent_parts.begin(), parent_parts.end(), std::back_inserter(child_parts),
      [&internal_parts](const stk::mesh::Part *part) { return internal_parts.find(part) == internal_parts.end(); });
}

template <size_t num_node_fields, size_t num_element_fields,
          Kokkos::Array<unsigned, num_node_fields> size_of_node_fields_to_copy,
          Kokkos::Array<unsigned, num_element_fields> size_of_element_fields_to_copy, typename... NodeFieldToCopyType,
          typename... ElementFieldToCopyType>
void subdivide_spherocylinder_segments(
    stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
    const stk::mesh::Field<double> &coordinate_field, const stk::mesh::Field<double> &length_field,
    const double &initial_length, const stk::mesh::Field<double> &radius_field,
    const Kokkos::UnorderedMap<stk::mesh::Entity, stk::mesh::Entity> &child_to_parent_map,
    const std::tuple<NodeFieldToCopyType...> &node_fields_to_copy,
    const std::tuple<ElementFieldToCopyType...> &element_fields_to_copy) {
  // Update the length and coordinates of the children
  // This is only done for locally owned entities.
  mundy::mesh::for_each_entity_run(
      bulk_data, stk::topology::ELEMENT_RANK, selector & bulk_data.mesh_meta_data().locally_owned_part(),
      [&coordinate_field, &radius_field, &length_field, &initial_length, &child_to_parent_map, &node_fields_to_copy,
       &element_fields_to_copy](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &child) {
        const size_t child_map_index = child_to_parent_map.find(child);
        const bool valid_child_index = child_to_parent_map.valid_at(child_map_index);
        if (valid_child_index) {
          // The parent becomes the left spherocylinder segment and the child becomes the right.
          // Fetch the downward connections.
          const stk::mesh::Entity &parent = child_to_parent_map.value_at(child_map_index);
          const stk::mesh::Entity &parent_left_node = bulk_data.begin_nodes(parent)[0];
          const stk::mesh::Entity &parent_right_node = bulk_data.begin_nodes(parent)[1];
          const stk::mesh::Entity &child_left_node = bulk_data.begin_nodes(child)[0];
          const stk::mesh::Entity &child_right_node = bulk_data.begin_nodes(child)[1];

          // Copy the parent's fields to the children
          copy_entity_fields<num_node_fields, size_of_node_fields_to_copy>(parent_left_node, child_left_node,
                                                                           node_fields_to_copy);
          copy_entity_fields<num_node_fields, size_of_node_fields_to_copy>(parent_right_node, child_right_node,
                                                                           node_fields_to_copy);
          copy_entity_fields<num_element_fields, size_of_element_fields_to_copy>(parent, child, element_fields_to_copy);

          // Compute the position of the new node
          // n----n -> n----N----n
          // Parents and children share a common node (N).  The child spherocylinder segment is aligned with the parent,
          // so it's position will be 2 lengths (parent_right_node_coords - parent_left_node_coords) added to the
          // left parent node with an orientation given by parent_right_node_coords - parent_left_node_coords.

          auto parent_left_node_coords = mundy::mesh::vector3_field_data(coordinate_field, parent_left_node);
          auto parent_right_node_coords = mundy::mesh::vector3_field_data(coordinate_field, parent_right_node);
          auto child_right_node_coords = mundy::mesh::vector3_field_data(coordinate_field, child_right_node);

          const double parent_radius = stk::mesh::field_data(radius_field, parent)[0];
          stk::mesh::field_data(radius_field, child)[0] = parent_radius;

          child_right_node_coords = parent_left_node_coords + (parent_right_node_coords - parent_left_node_coords) * 2;
        }
      });

  // At this point, all locally owned parents and their children are up-to-date.
  // Communicate the changes to the ghosted entities.
  std::vector<const stk::mesh::FieldBase *> fields_to_communicate = {&coordinate_field, &length_field, &radius_field};
  impl::append_fields_to_field_pointer_vector_impl(std::make_index_sequence<num_node_fields>(), node_fields_to_copy,
                                                   fields_to_communicate);
  impl::append_fields_to_field_pointer_vector_impl(std::make_index_sequence<num_element_fields>(),
                                                   element_fields_to_copy, fields_to_communicate);
  stk::mesh::communicate_field_data(bulk_data, fields_to_communicate);
}

template <size_t num_node_fields, size_t num_element_fields,
          Kokkos::Array<unsigned, num_node_fields> size_of_node_fields_to_copy,
          Kokkos::Array<unsigned, num_element_fields> size_of_element_fields_to_copy, typename... NodeFieldToCopyType,
          typename... ElementFieldToCopyType>
void subdivide_flagged_spherocylinder_segments(stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
                                               const stk::mesh::Field<double> &coordinate_field,
                                               const stk::mesh::Field<double> &length_field,
                                               const double &initial_length,
                                               const stk::mesh::Field<double> &radius_field,
                                               const stk::mesh::Field<int> &flag_field, const int &divide_flag_value,
                                               const std::tuple<NodeFieldToCopyType...> &node_fields_to_copy,
                                               const std::tuple<ElementFieldToCopyType...> &element_fields_to_copy) {
  // Fetch the entities to maybe divide.
  // To avoid double counting, we only select locally owned entities.
  std::vector<stk::mesh::Entity> entities_to_maybe_divide;
  stk::mesh::get_selected_entities(selector & bulk_data.mesh_meta_data().locally_owned_part(),
                                   bulk_data.buckets(stk::topology::ELEMENT_RANK), entities_to_maybe_divide);

  // Store if the entities should be divided or not.
  const size_t num_entities_to_maybe_divide = entities_to_maybe_divide.size();
  std::vector<int> should_divide(num_entities_to_maybe_divide);
#pragma omp parallel for
  for (size_t i = 0; i < num_entities_to_maybe_divide; i++) {
    const stk::mesh::Entity &entity = entities_to_maybe_divide[i];
    const int should_entity_divide = *stk::mesh::field_data(flag_field, entity) == divide_flag_value;
    should_divide[i] = should_entity_divide;
  }

  // The cumulative sum of the division flags minus 1 gives the mapping from parent lid (the index of the entity in the
  // entities to maybe divide vector) to the child lid (the index of the entity in the new entities vector).
  std::vector<size_t> parent_lid_to_child_lid(num_entities_to_maybe_divide, 0);
  std::partial_sum(should_divide.begin(), should_divide.end(), parent_lid_to_child_lid.begin(),
                   [](const int a, const int b) { return a + b; });
  const size_t new_element_count = num_entities_to_maybe_divide > 0 ? parent_lid_to_child_lid.back() : 0;
  const size_t new_node_count = new_element_count;

#pragma omp parallel for
  for (size_t i = 0; i < num_entities_to_maybe_divide; i++) {
    parent_lid_to_child_lid[i] -= (parent_lid_to_child_lid[i] > 0) * 1;
  }

  // Create the new entities.
  bulk_data.modification_begin();
  std::vector<std::size_t> requests(bulk_data.mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = new_node_count;
  requests[stk::topology::ELEMENT_RANK] = new_element_count;
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data.generate_new_entities(requests, requested_entities);

  // Connect the child to its nodes
  std::span<stk::mesh::Entity> new_nodes = {requested_entities.begin(), new_node_count};
  std::span<stk::mesh::Entity> new_elements = {requested_entities.begin() + new_node_count, new_element_count};
  Kokkos::UnorderedMap<stk::mesh::Entity, stk::mesh::Entity> child_to_parent_map(new_element_count);
  for (size_t i = 0; i < num_entities_to_maybe_divide; i++) {
    if (should_divide[i]) {
      // Fetch the parent and child and their downward connections
      const stk::mesh::Entity &parent = entities_to_maybe_divide[i];
      const stk::mesh::Entity &parent_node = bulk_data.begin_nodes(parent)[1];
      const size_t child_lid = parent_lid_to_child_lid[i];
      const stk::mesh::Entity &child = new_elements[child_lid];
      const stk::mesh::Entity &child_node = new_nodes[child_lid];

      MUNDY_THROW_ASSERT(bulk_data.is_valid(parent), std::invalid_argument, "Parent entity is not valid.");
      MUNDY_THROW_ASSERT(bulk_data.is_valid(parent_node), std::invalid_argument, "Parent node is not valid.");
      MUNDY_THROW_ASSERT(bulk_data.is_valid(child), std::invalid_argument, "Child entity is not valid.");
      MUNDY_THROW_ASSERT(bulk_data.is_valid(child_node), std::invalid_argument, "Child node is not valid.");

      // Connect the child to its node
      bulk_data.declare_relation(child, parent_node, 0);
      bulk_data.declare_relation(child, child_node, 1);

      // Add the child to the parent's parts (non-internal)
      stk::mesh::ConstPartVector elem_parent_parts_without_internal;
      get_parent_entity_parts(bulk_data, parent, elem_parent_parts_without_internal);
      bulk_data.change_entity_parts(child, elem_parent_parts_without_internal);

      // Remove the tip part membership from the parent's part
      stk::mesh::PartVector tip_part;
      selector.get_parts(tip_part);
      bulk_data.change_entity_parts(parent, stk::mesh::PartVector{}, tip_part);

      // Store the child to parent map
      bool kv_insert_success = child_to_parent_map.insert(child, parent).success();
      assert(kv_insert_success == true);
    }
  }

  bulk_data.modification_end();

  // Subdivide the entities.
  subdivide_spherocylinder_segments<num_node_fields, num_element_fields, size_of_node_fields_to_copy,
                                    size_of_element_fields_to_copy>(bulk_data, selector, coordinate_field, length_field,
                                                                    initial_length, radius_field, child_to_parent_map,
                                                                    node_fields_to_copy, element_fields_to_copy);
}

void subdivide_flagged_spherocylinder_segments(stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
                                               const stk::mesh::Field<double> &coordinate_field,
                                               const stk::mesh::Field<double> &length_field,
                                               const double &initial_length,
                                               const stk::mesh::Field<double> &radius_field,
                                               const stk::mesh::Field<int> &flag_field, const int &divide_flag_value) {
  constexpr size_t num_node_fields = 0;
  constexpr size_t num_element_fields = 0;
  constexpr Kokkos::Array<unsigned, num_node_fields> size_of_node_fields_to_copy = {};
  constexpr Kokkos::Array<unsigned, num_element_fields> size_of_element_fields_to_copy = {};
  return subdivide_flagged_spherocylinder_segments<num_node_fields, num_element_fields, size_of_node_fields_to_copy,
                                                   size_of_element_fields_to_copy>(
      bulk_data, selector, coordinate_field, length_field, initial_length, radius_field, flag_field, divide_flag_value,
      std::make_tuple(), std::make_tuple());
}

class FilamentSim {
 public:
  FilamentSim() = default;

  void print_rank0(auto thing_to_print, int indent_level = 0) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::string indent(indent_level * 2, ' ');
      std::cout << indent << thing_to_print << std::endl;
    }
  }

  void debug_print([[maybe_unused]] auto thing_to_print, [[maybe_unused]] int indent_level = 0) {
#ifdef DEBUG
    // print_rank0(thing_to_print, indent_level);
    std::string indent(indent_level * 2, ' ');
    std::cout << indent << " Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " " << thing_to_print
              << std::endl;
#endif
  }

  void parse_user_inputs(int argc, char **argv) {
    debug_print("Parsing user inputs.");

    // Parse the command line options.
    // a.out --params=input.yaml
    const bool throw_exception_if_not_found = true;
    const bool recognise_all_options = true;
    Teuchos::CommandLineProcessor cmdp(throw_exception_if_not_found, recognise_all_options);
    cmdp.setOption("params", &input_file_name_, "The name of the input file.");
    bool was_parse_successful = cmdp.parse(argc, argv) == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
    MUNDY_THROW_REQUIRE(was_parse_successful, std::invalid_argument, "Failed to parse the command line arguments.");

    // Read in the parameters from the parameter list.
    Teuchos::ParameterList param_list_ = *Teuchos::getParametersFromYamlFile(input_file_name_);

    filament_radius_ = param_list_.get<double>("filament_radius");
    filament_initial_length_ = param_list_.get<double>("filament_initial_length");
    filament_division_length_ = param_list_.get<double>("filament_division_length");
    filament_growth_rate_ = param_list_.get<double>("filament_growth_rate");

    filament_density_ = param_list_.get<double>("filament_density");
    filament_youngs_modulus_ = param_list_.get<double>("filament_youngs_modulus");
    filament_poissons_ratio_ = param_list_.get<double>("filament_poissons_ratio");

    num_time_steps_ = param_list_.get<int>("num_time_steps");
    timestep_size_ = param_list_.get<double>("timestep_size");
    io_frequency_ = param_list_.get<int>("io_frequency");
    load_balance_frequency_ = param_list_.get<int>("load_balance_frequency");
    boundary_radius_ = param_list_.get<double>("boundary_radius");

    check_input_parameters();
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_REQUIRE(filament_radius_ > 0, std::invalid_argument, "filament_radius_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(filament_initial_length_ > -1e-12, std::invalid_argument,
                        "filament_initial_length_ must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(filament_division_length_ > 0, std::invalid_argument,
                        "filament_division_length_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(filament_density_ > 0, std::invalid_argument, "filament_density_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(filament_youngs_modulus_ > 0, std::invalid_argument,
                        "filament_youngs_modulus_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(filament_poissons_ratio_ > 0, std::invalid_argument,
                        "filament_poissons_ratio_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(io_frequency_ > 0, std::invalid_argument, "io_frequency_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(load_balance_frequency_ > 0, std::invalid_argument,
                        "load_balance_frequency_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(boundary_radius_ > 0, std::invalid_argument, "boundary_radius_ must be greater than 0.");
  }

  void dump_user_inputs() {
    debug_print("Dumping user inputs.");
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "  filament_radius_: " << filament_radius_ << std::endl;
      std::cout << "  filament_initial_length_: " << filament_initial_length_ << std::endl;
      std::cout << "  filament_division_length_: " << filament_division_length_ << std::endl;
      std::cout << "  filament_growth_rate_: " << filament_growth_rate_ << std::endl;
      std::cout << "  filament_youngs_modulus_: " << filament_youngs_modulus_ << std::endl;
      std::cout << "  filament_poissons_ratio_: " << filament_poissons_ratio_ << std::endl;
      std::cout << "  filament_density_: " << filament_density_ << std::endl;
      std::cout << "  num_time_steps_: " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size_: " << timestep_size_ << std::endl;
      std::cout << "  io_frequency_: " << io_frequency_ << std::endl;
      std::cout << "  load_balance_frequency_: " << load_balance_frequency_ << std::endl;
      std::cout << "  boundary_radius_: " << boundary_radius_ << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

  void build_our_mesh_and_method_instances() {
    debug_print("Building our mesh and method instances.");

    // Setup the mesh requirements.
    // First, we need to fetch the mesh requirements for each method, then we can create the class instances.
    // In the future, all of this will be done via the Configurator.
    mesh_reqs_ptr_ = std::make_shared<mundy::meta::MeshReqs>(MPI_COMM_WORLD);
    mesh_reqs_ptr_->set_spatial_dimension(3);
    mesh_reqs_ptr_->set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    // Add custom requirements for this example. These are requirements that exceed those of the enabled methods and
    // allow us to extend the functionality offered natively by Mundy.
    //
    // We require that the filament are sphereocylinder segments, so they need to have BEAM_2 topology
    auto filament_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    filament_part_reqs->set_part_name("FILAMENT")
        .set_part_topology(stk::topology::BEAM_2)

        // Add the node fields
        .add_field_reqs<double>("NODE_COORDS", node_rank_, 3, 2)
        .add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 2)
        .add_field_reqs<double>("NODE_OMEGA", node_rank_, 3, 2)
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_TORQUE", node_rank_, 3, 1)
        .add_field_reqs<size_t>("NODE_RNG_COUNTER", node_rank_, 1, 1)

        // Add the element fields
        .add_field_reqs<double>("ELEMENT_RADIUS", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_ORIENTATION", element_rank_, 4, 2)
        .add_field_reqs<double>("ELEMENT_TANGENT", element_rank_, 3, 1)
        .add_field_reqs<double>("ELEMENT_LENGTH", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_YOUNGS_MODULUS", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_POISSONS_RATIO", element_rank_, 1, 1)
        .add_field_reqs<int>("ELEMENT_MARKED_FOR_DIVISION", element_rank_, 1, 1);

    mesh_reqs_ptr_->add_and_sync_part_reqs(filament_part_reqs);
    mundy::shapes::SpherocylinderSegments::add_and_sync_subpart_reqs(filament_part_reqs);

    auto tip_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    tip_part_reqs->set_part_name("TIP")
        .set_part_topology(stk::topology::BEAM_2)

        // Add the node fields
        .add_field_reqs<double>("NODE_COORDS", node_rank_, 3, 2)
        .add_field_reqs<double>("TRANSIENT_NODE_COORDINATES", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_VELOCITY", node_rank_, 3, 2)
        .add_field_reqs<double>("NODE_OMEGA", node_rank_, 3, 2)
        .add_field_reqs<double>("NODE_FORCE", node_rank_, 3, 1)
        .add_field_reqs<double>("NODE_TORQUE", node_rank_, 3, 1)
        .add_field_reqs<size_t>("NODE_RNG_COUNTER", node_rank_, 1, 1)

        // Add the element fields
        .add_field_reqs<double>("ELEMENT_RADIUS", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_ORIENTATION", element_rank_, 4, 2)
        .add_field_reqs<double>("ELEMENT_TANGENT", element_rank_, 3, 1)
        .add_field_reqs<double>("ELEMENT_LENGTH", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_YOUNGS_MODULUS", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_POISSONS_RATIO", element_rank_, 1, 1)
        .add_field_reqs<int>("ELEMENT_MARKED_FOR_DIVISION", element_rank_, 1, 1);

    mesh_reqs_ptr_->add_and_sync_part_reqs(tip_part_reqs);
    mundy::shapes::SpherocylinderSegments::add_and_sync_subpart_reqs(tip_part_reqs);

#ifdef DEBUG
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      mesh_reqs_ptr_->print();
    }
#endif

    // Add the requirements for our initialized methods to the mesh
    // When we eventually switch to the configurator, these individual fixed params will become sublists within a single
    // master parameter list. Note, sublist will return a reference to the sublist with the given name.
    auto compute_ssd_and_cn_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKER"));
    auto compute_aabb_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));
    compute_aabb_fixed_params.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("FILAMENT"));
    auto generate_neighbor_linkers_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS"));
    generate_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("FILAMENT"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("FILAMENT"));
    auto evaluate_linker_potentials_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names",
        mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_HERTZIAN_CONTACT"));
    auto linker_potential_force_reduction_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SEGMENT"));
    linker_potential_force_reduction_fixed_params.sublist("SPHEROCYLINDER_SEGMENT")
        .set("valid_entity_part_names", mundy::core::make_string_array("FILAMENT"));
    auto destroy_distant_neighbor_linkers_fixed_params =
        Teuchos::ParameterList().set("enabled_technique_name", "DESTROY_DISTANT_NEIGHBORS");

    // Synchronize (merge and rectify differences) the requirements for each method based on the fixed parameters.
    // For now, we will directly use the types that each method corresponds to. The configurator will
    // fetch the static members of these methods using the configurable method factory.
    mesh_reqs_ptr_->sync(mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::get_mesh_requirements(
        compute_ssd_and_cn_fixed_params));
    mesh_reqs_ptr_->sync(mundy::shapes::ComputeAABB::get_mesh_requirements(compute_aabb_fixed_params));
    mesh_reqs_ptr_->sync(
        mundy::linkers::GenerateNeighborLinkers::get_mesh_requirements(generate_neighbor_linkers_fixed_params));
    mesh_reqs_ptr_->sync(
        mundy::linkers::EvaluateLinkerPotentials::get_mesh_requirements(evaluate_linker_potentials_fixed_params));
    mesh_reqs_ptr_->sync(mundy::linkers::LinkerPotentialForceReduction::get_mesh_requirements(
        linker_potential_force_reduction_fixed_params));
    mesh_reqs_ptr_->sync(
        mundy::linkers::DestroyNeighborLinkers::get_mesh_requirements(destroy_distant_neighbor_linkers_fixed_params));

#ifdef DEBUG
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      mesh_reqs_ptr_->print();
    }
#endif

    // The mesh requirements are now set up, so we solidify the mesh structure.
    bulk_data_ptr_ = mesh_reqs_ptr_->declare_mesh();
    meta_data_ptr_ = bulk_data_ptr_->mesh_meta_data_ptr();
    meta_data_ptr_->set_coordinate_field_name("NODE_COORDS");
    meta_data_ptr_->use_simple_fields();
    meta_data_ptr_->commit();

    // Now that the mesh is set up, we can create our method instances.
    // For now, we will directly use the types that each method corresponds to. The configurator will
    // fetch the static members of these methods using the configurable method factory.
    compute_ssd_and_cn_ptr_ = mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::create_new_instance(
        bulk_data_ptr_.get(), compute_ssd_and_cn_fixed_params);
    compute_aabb_ptr_ =
        mundy::shapes::ComputeAABB::create_new_instance(bulk_data_ptr_.get(), compute_aabb_fixed_params);
    generate_neighbor_linkers_ptr_ = mundy::linkers::GenerateNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), generate_neighbor_linkers_fixed_params);
    evaluate_linker_potentials_ptr_ = mundy::linkers::EvaluateLinkerPotentials::create_new_instance(
        bulk_data_ptr_.get(), evaluate_linker_potentials_fixed_params);
    linker_potential_force_reduction_ptr_ = mundy::linkers::LinkerPotentialForceReduction::create_new_instance(
        bulk_data_ptr_.get(), linker_potential_force_reduction_fixed_params);
    destroy_distant_neighbor_linkers_ptr_ = mundy::linkers::DestroyNeighborLinkers::create_new_instance(
        bulk_data_ptr_.get(), destroy_distant_neighbor_linkers_fixed_params);

    // Set up the mutable parameters for the classes
    // If a class doesn't have mutable parameters, we can skip setting them.

    // ComputeAABB mutable parameters
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", 0.0);
    compute_aabb_ptr_->set_mutable_params(compute_aabb_mutable_params);
  }

  template <typename FieldType>
  stk::mesh::Field<FieldType> *fetch_field(const std::string &field_name, stk::topology::rank_t rank) {
    auto field_ptr = meta_data_ptr_->get_field<FieldType>(rank, field_name);
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                        std::string("Field ") + field_name + " not found in the mesh meta data.");
    return field_ptr;
  }

  stk::mesh::Part *fetch_part(const std::string &part_name) {
    auto part_ptr = meta_data_ptr_->get_part(part_name);
    MUNDY_THROW_REQUIRE(part_ptr != nullptr, std::invalid_argument,
                        std::string("Part ") + part_name + " not found in the mesh meta data.");
    return part_ptr;
  }

  void fetch_fields_and_parts() {
    debug_print("Fetching fields and parts.");

    // Fetch the fields
    node_coord_field_ptr_ = fetch_field<double>("NODE_COORDS", node_rank_);
    node_velocity_field_ptr_ = fetch_field<double>("NODE_VELOCITY", node_rank_);
    node_omega_field_ptr_ = fetch_field<double>("NODE_OMEGA", node_rank_);
    node_force_field_ptr_ = fetch_field<double>("NODE_FORCE", node_rank_);
    node_torque_field_ptr_ = fetch_field<double>("NODE_TORQUE", node_rank_);
    node_rng_counter_field_ptr_ = fetch_field<size_t>("NODE_RNG_COUNTER", node_rank_);

    element_radius_field_ptr_ = fetch_field<double>("ELEMENT_RADIUS", element_rank_);
    element_length_field_ptr_ = fetch_field<double>("ELEMENT_LENGTH", element_rank_);
    element_orientation_field_ptr_ = fetch_field<double>("ELEMENT_ORIENTATION", element_rank_);
    element_tangent_field_ptr_ = fetch_field<double>("ELEMENT_TANGENT", element_rank_);
    element_youngs_modulus_field_ptr_ = fetch_field<double>("ELEMENT_YOUNGS_MODULUS", element_rank_);
    element_poissons_ratio_field_ptr_ = fetch_field<double>("ELEMENT_POISSONS_RATIO", element_rank_);
    element_marked_for_division_field_ptr_ = fetch_field<int>("ELEMENT_MARKED_FOR_DIVISION", element_rank_);
    element_aabb_field_ptr_ = fetch_field<double>("ELEMENT_AABB", element_rank_);

    linker_potential_force_field_ptr_ = fetch_field<double>("LINKER_POTENTIAL_FORCE", constraint_rank_);

    // Fetch the parts
    filament_part_ptr_ = fetch_part("FILAMENT");
    tip_part_ptr_ = fetch_part("TIP");
    spherocylinder_segment_spherocylinder_segment_linkers_part_ptr_ =
        fetch_part("SPHEROCYLINDER_SEGMENT_SPHEROCYLINDER_SEGMENT_LINKERS");
    MUNDY_THROW_REQUIRE(filament_part_ptr_->topology() == stk::topology::BEAM_2, std::logic_error,
                        "FILAMENT part must have BEAM_2 topology.");
  }

  void setup_io() {
    debug_print("Setting up IO.");

    // Create a mundy io broker via it's fixed parameters
    auto fixed_params_iobroker =
        Teuchos::ParameterList()
            .set("enabled_io_parts", mundy::core::make_string_array("FILAMENT"))
            .set("enabled_io_fields_node_rank",
                 mundy::core::make_string_array("NODE_VELOCITY", "NODE_OMEGA", "NODE_FORCE", "NODE_TORQUE",
                                                "NODE_RNG_COUNTER"))
            .set("enabled_io_fields_element_rank",
                 mundy::core::make_string_array("ELEMENT_RADIUS", "ELEMENT_LENGTH", "ELEMENT_ORIENTATION",
                                                "ELEMENT_TANGENT", "ELEMENT_AABB", "ELEMENT_YOUNGS_MODULUS",
                                                "ELEMENT_POISSONS_RATIO", "ELEMENT_MARKED_FOR_DIVISION"))
            .set("coordinate_field_name", "NODE_COORDS")
            .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
            .set("exodus_database_output_filename", "Filament.exo")
            .set("parallel_io_mode", "hdf5")
            .set("database_purpose", "results");
    // Create the IO broker
    io_broker_ptr_ = mundy::io::IOBroker::create_new_instance(bulk_data_ptr_.get(), fixed_params_iobroker);
  }

  void declare_and_initialize_filament() {
    debug_print("Declaring and initializing the filament.");

    // Declare the filament on rank 0
    bulk_data_ptr_->modification_begin();
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      stk::mesh::Entity filament_node1 = bulk_data_ptr_->declare_entity(
          stk::topology::NODE_RANK, 1, stk::mesh::PartVector{filament_part_ptr_, tip_part_ptr_});
      stk::mesh::Entity filament_node2 = bulk_data_ptr_->declare_entity(
          stk::topology::NODE_RANK, 2, stk::mesh::PartVector{filament_part_ptr_, tip_part_ptr_});
      stk::mesh::Entity filament = bulk_data_ptr_->declare_entity(
          stk::topology::ELEMENT_RANK, 1, stk::mesh::PartVector{filament_part_ptr_, tip_part_ptr_});
      bulk_data_ptr_->declare_relation(filament, filament_node1, 0);
      bulk_data_ptr_->declare_relation(filament, filament_node2, 1);
    }
    bulk_data_ptr_->modification_end();

    // Initialize it
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      stk::mesh::Entity filament_node1 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 1);
      stk::mesh::Entity filament_node2 = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, 2);
      stk::mesh::Entity filament = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, 1);

      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(filament_node1), std::invalid_argument,
                         "Filament node 1 is not valid.");
      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(filament_node2), std::invalid_argument,
                         "Filament node 2 is not valid.");
      MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(filament), std::invalid_argument, "Filament element is not valid.");

      stk::mesh::field_data(*node_rng_counter_field_ptr_, filament_node1)[0] = 0;
      mundy::mesh::vector3_field_data(*node_coord_field_ptr_, filament_node1).set(1.0, 1.0, 1.0);
      std::cout << "node1 coords after init: "
                << mundy::mesh::vector3_field_data(*node_coord_field_ptr_, filament_node1) << "\n";
      mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, filament_node1).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_omega_field_ptr_, filament_node1).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_force_field_ptr_, filament_node1).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_torque_field_ptr_, filament_node1).set(0.0, 0.0, 0.0);

      stk::mesh::field_data(*node_rng_counter_field_ptr_, filament_node2)[0] = 0;
      mundy::mesh::vector3_field_data(*node_coord_field_ptr_, filament_node2)
          .set(1.0 + filament_initial_length_, 1.0, 1.0);
      mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, filament_node2).set(filament_growth_rate_, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_omega_field_ptr_, filament_node2).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_force_field_ptr_, filament_node2).set(0.0, 0.0, 0.0);
      mundy::mesh::vector3_field_data(*node_torque_field_ptr_, filament_node2).set(0.0, 0.0, 0.0);

      stk::mesh::field_data(*element_radius_field_ptr_, filament)[0] = filament_radius_;
      stk::mesh::field_data(*element_length_field_ptr_, filament)[0] = filament_initial_length_;
      stk::mesh::field_data(*element_youngs_modulus_field_ptr_, filament)[0] = filament_youngs_modulus_;
      stk::mesh::field_data(*element_poissons_ratio_field_ptr_, filament)[0] = filament_poissons_ratio_;

      mundy::math::Vector3d current_tangent(1.0, 0.0, 0.0);
      mundy::math::Vector3d x_axis(1.0, 0.0, 0.0);
      mundy::mesh::quaternion_field_data(*element_orientation_field_ptr_, filament) =
          mundy::math::quat_from_parallel_transport(x_axis, current_tangent);
      mundy::mesh::vector3_field_data(*element_tangent_field_ptr_, filament) = current_tangent;
    }
  }

  void load_balance() {
    debug_print("Load balancing the mesh.");
    stk::balance::balanceStkMesh(balance_settings_, *bulk_data_ptr_);
  }

  void rotate_field_states() {
    debug_print("Rotating the field states.");
    bulk_data_ptr_->update_field_data_states();
  }

  void zero_out_transient_node_fields() {
    debug_print("Zeroing out the transient node fields.");
    mundy::mesh::utils::fill_field_with_value<double>(*node_velocity_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_omega_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_force_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*node_torque_field_ptr_, std::array<double, 3>{0.0, 0.0, 0.0});
    mundy::mesh::utils::fill_field_with_value<double>(*linker_potential_force_field_ptr_,
                                                      std::array<double, 3>{0.0, 0.0, 0.0});
  }

  // this is also node used right now in the simulation.  there are no force calculations
  void compute_hertzian_contact_force_and_torque() {
    debug_print("Computing the Hertzian contact force and torque.");

    // Check if the rod-rod neighbor list needs updated or not
    bool rod_rod_neighbor_list_needs_updated = true;
    if (rod_rod_neighbor_list_needs_updated) {
      // Compute the AABBs for the rods
      debug_print("Computing the AABBs for the rods.");
      compute_aabb_ptr_->execute(*filament_part_ptr_);

      // Delete rod-rod neighbor linkers that are too far apart
      debug_print("Deleting rod-rod neighbor linkers that are too far apart.");
      Kokkos::Timer timer0;
      destroy_distant_neighbor_linkers_ptr_->execute(*spherocylinder_segment_spherocylinder_segment_linkers_part_ptr_);
      debug_print("Time to destroy distant neighbor linkers: " + std::to_string(timer0.seconds()));

      // Generate neighbor linkers between nearby rods
      debug_print("Generating neighbor linkers between nearby rods.");
      Kokkos::Timer timer1;
      generate_neighbor_linkers_ptr_->execute(*filament_part_ptr_, *filament_part_ptr_);
      debug_print("Time to generate neighbor linkers: " + std::to_string(timer1.seconds()));
    }

    // Hertzian contact force evaluation
    // Compute the signed separation distance and contact normal between neighboring rods
    debug_print("Computing the signed separation distance and contact normal between neighboring rods.");
    compute_ssd_and_cn_ptr_->execute(*spherocylinder_segment_spherocylinder_segment_linkers_part_ptr_);

    // Evaluate the Hertzian contact potential between neighboring rods
    debug_print("Evaluating the Hertzian contact potential between neighboring rods.");
    evaluate_linker_potentials_ptr_->execute(*spherocylinder_segment_spherocylinder_segment_linkers_part_ptr_);

    // Sum the linker potential force to get the induced node force on each rod
    debug_print("Summing the linker potential force to get the induced node force on each rod.");
    linker_potential_force_reduction_ptr_->execute(*filament_part_ptr_);
  }

  void compute_generalized_velocity() {
    debug_print("Computing the generalized velocity using the mobility problem.");
    // the velocity is constant, with magnitude given by filament_growth_rate_.
    // the direction is along the tangent unless a boundary is hit, and then it is
    // the inward normal of the spherical boundary at the hit point
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<size_t> &node_rng_counter_field = *node_rng_counter_field_ptr_;
    const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &element_tangent_field = *element_tangent_field_ptr_;
    const double filament_growth_rate = filament_growth_rate_;
    const double boundary_radius = boundary_radius_;

    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *tip_part_ptr_,
        [&node_coord_field, &node_velocity_field, &node_rng_counter_field, &element_radius_field,
         &element_tangent_field, &filament_growth_rate,
         &boundary_radius]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Fetch the connected node. Only the right node moves
          const stk::mesh::Entity node1 = bulk_data.begin_nodes(element)[0];
          const stk::mesh::Entity node2 = bulk_data.begin_nodes(element)[1];

          // Get node coordinates and velocities
          auto node2_coords = mundy::mesh::vector3_field_data(node_coord_field, node2);
          auto node2_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node2);
          // Get element tangent to serve as the direction of velocity
          auto element_tangent = mundy::mesh::vector3_field_data(element_tangent_field, element);

          size_t *node_rng_counter = stk::mesh::field_data(node_rng_counter_field, node2);
          const size_t node_gid = bulk_data.identifier(node2);
          openrand::Philox rng(node_gid, node_rng_counter[0]);

          // Apply a very small random orientational kick to keep filament from growing in a circle
          const double max_kick = 0.8;
          element_tangent[0] += max_kick * (rng.rand<double>());
          element_tangent[1] -= max_kick * (rng.rand<double>());
          element_tangent[2] += max_kick * (rng.rand<double>());
          node_rng_counter[0] += 1;

          node2_velocity = filament_growth_rate * element_tangent;

          // Assume the filament is growing inside a sphere, so check if the filament has crossed the boundary.
          const double distance_from_origin = sqrt(mundy::math::dot(node2_coords, node2_coords));
          const double element_radius = stk::mesh::field_data(element_radius_field, element)[0];
          if (distance_from_origin + element_radius > boundary_radius) {
            // then adjust the positions and velocities of the node
            auto normal = node2_coords / distance_from_origin;
            node2_velocity -= 2 * mundy::math::dot(node2_velocity, normal) * normal;
          }
        });
  }

  void update_generalized_position() {
    debug_print("Updating the generalized position using Euler's method.");

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    const stk::mesh::Field<double> &node_coord_field_old = node_coord_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    const stk::mesh::Field<double> &node_velocity_field_old =
        node_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    stk::mesh::Field<double> &element_tangent_field = *element_tangent_field_ptr_;
    stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
    const double timestep_size = timestep_size_;
    const double boundary_radius = boundary_radius_;

    // Update the generalized position using Euler's method
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *tip_part_ptr_,
        [&node_coord_field, &node_coord_field_old, &node_velocity_field, &node_velocity_field_old,
         &element_radius_field, &element_tangent_field, &element_length_field, &timestep_size,
         &boundary_radius]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Fetch the connected node
          const stk::mesh::Entity node1 = bulk_data.begin_nodes(element)[0];
          const stk::mesh::Entity node2 = bulk_data.begin_nodes(element)[1];

          // Get node coordinates and velocities
          auto node1_coords = mundy::mesh::vector3_field_data(node_coord_field, node1);
          auto node1_coords_old = mundy::mesh::vector3_field_data(node_coord_field_old, node1);
          auto node2_coords = mundy::mesh::vector3_field_data(node_coord_field, node2);
          auto node2_coords_old = mundy::mesh::vector3_field_data(node_coord_field_old, node2);
          auto node2_velocity_old = mundy::mesh::vector3_field_data(node_velocity_field_old, node2);

          // Update the position
          node1_coords = node1_coords_old;
          node2_coords = node2_coords_old + timestep_size * node2_velocity_old;

          // Assume the filament is growing inside a sphere, so check if the filament has crossed the boundary.
          const double distance_from_origin = sqrt(mundy::math::dot(node2_coords, node2_coords));
          const double element_radius = stk::mesh::field_data(element_radius_field, element)[0];
          if (distance_from_origin + element_radius > boundary_radius) {
            // then adjust the positions of the node
            auto normal = node2_coords / distance_from_origin;
            double overlap = distance_from_origin + element_radius - boundary_radius;
            node2_coords -= normal * overlap;
          }
          // Get the output fields
          auto element_tangent = mundy::mesh::vector3_field_data(element_tangent_field, element);
          auto element_length = mundy::mesh::vector3_field_data(element_length_field, element)[0];

          // Compute the tangent and length
          const auto separation_vector = node2_coords - node1_coords;
          const double separation_distance = sqrt(mundy::math::dot(separation_vector, separation_vector));
          const auto unit_tangent = separation_vector / separation_distance;
          element_tangent = unit_tangent;
          element_length = separation_distance;
        });
  }

  void grow_filament() {
    debug_print("Growing the filament.");

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
    const double filament_growth_rate = filament_growth_rate_;
    const double timestep_size = timestep_size_;

    // Grow the filament
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *tip_part_ptr_,
        [&element_length_field, &filament_growth_rate, &timestep_size](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &filament) {
          stk::mesh::field_data(element_length_field, filament)[0] += timestep_size * filament_growth_rate;
        });
  }

  auto make_ref_tuple(auto &...args) {
    return std::make_tuple(std::ref(args)...);
  }

  void divide_filament() {
    debug_print("Dividing the filament.");

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
    stk::mesh::Field<int> &element_marked_for_division_field = *element_marked_for_division_field_ptr_;
    stk::mesh::Field<size_t> &node_rng_counter_field = *node_rng_counter_field_ptr_;
    stk::mesh::Field<double> &node_omega_field = *node_omega_field_ptr_;
    const double filament_division_length = filament_division_length_;

    // Loop over all particles and stash if they divide or not
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *filament_part_ptr_,
        [&element_length_field, &element_marked_for_division_field, &filament_division_length](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &filament) {
          stk::mesh::field_data(element_marked_for_division_field, filament)[0] =
              (stk::mesh::field_data(element_length_field, filament)[0] > filament_division_length);
        });

    // Subdivide. Create copies of each field from the parent to the child. Update the length and coords.
    // The coords, orientation, length, and radius are copied by default.
    constexpr size_t num_node_fields = 5;
    constexpr size_t num_element_fields = 3;
    constexpr Kokkos::Array<unsigned, num_node_fields> size_of_node_fields_to_copy = {3, 3, 3, 3, 1};
    constexpr Kokkos::Array<unsigned, num_element_fields> size_of_element_fields_to_copy = {1, 1, 1};
    auto extra_node_fields_to_copy =
        make_ref_tuple(*node_velocity_field_ptr_, *node_omega_field_ptr_, *node_force_field_ptr_,
                       *node_torque_field_ptr_, *node_rng_counter_field_ptr_);
    auto extra_element_fields_to_copy =
        make_ref_tuple(*element_youngs_modulus_field_ptr_, *element_poissons_ratio_field_ptr_,
                       *element_marked_for_division_field_ptr_);
    const int divide_flag_value = 1;
    subdivide_flagged_spherocylinder_segments<num_node_fields, num_element_fields, size_of_node_fields_to_copy,
                                              size_of_element_fields_to_copy>(
        *bulk_data_ptr_, *tip_part_ptr_, *node_coord_field_ptr_, *element_length_field_ptr_, filament_initial_length_,
        *element_radius_field_ptr_, *element_marked_for_division_field_ptr_, divide_flag_value,
        extra_node_fields_to_copy, extra_element_fields_to_copy);
  }

  // not really needed since tangent is updated in each node coordinate update
  void update_element_tangent() {
    // Get references to internal members so we aren't passing around *this
    const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &element_tangent_field = *element_tangent_field_ptr_;

    // Update the tangent
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *tip_part_ptr_,
        [&node_coord_field, &element_tangent_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                    const stk::mesh::Entity &polymer_segment) {
          // Fetch the connected node
          const stk::mesh::Entity node1 = bulk_data.begin_nodes(polymer_segment)[0];
          const stk::mesh::Entity node2 = bulk_data.begin_nodes(polymer_segment)[1];

          // Get node coordinates
          const auto node1_coords = mundy::mesh::vector3_field_data(node_coord_field, node1);
          const auto node2_coords = mundy::mesh::vector3_field_data(node_coord_field, node2);

          // Get the output fields
          auto element_tangent = mundy::mesh::vector3_field_data(element_tangent_field, polymer_segment);

          // Compute the tangent
          const auto separation_vector = node2_coords - node1_coords;
          const double separation_distance = sqrt(mundy::math::dot(separation_vector, separation_vector));
          const auto unit_tangent = separation_vector / separation_distance;
          element_tangent = unit_tangent;
        });
  }

  void run(int argc, char **argv) {
    debug_print("Running the simulation.");

    // Preprocess
    parse_user_inputs(argc, argv);
    dump_user_inputs();

    // Setup
    timestep_index_ = 0;
    build_our_mesh_and_method_instances();
    fetch_fields_and_parts();
    declare_and_initialize_filament();
    setup_io();

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    Kokkos::Timer timer;
    for (; timestep_index_ < num_time_steps_; timestep_index_++) {
      debug_print(std::string("Time step ") + std::to_string(timestep_index_) + " of " +
                  std::to_string(num_time_steps_));

      // Prepare the current configuration.
      {
        // Rotate the field states.
        rotate_field_states();

        // Move the nodes from t -> t + dt.
        //   x(t + dt) = x(t) + dt v(t)
        update_generalized_position();

        // Reset the fields in the current timestep.
        zero_out_transient_node_fields();
      }

      // Growth then division
      {
        divide_filament();
        grow_filament();
      }

      // Evaluate forces f(x(t + dt)).
      {
        // Hertzian contact force
        // compute_hertzian_contact_force_and_torque();
      }

      // Compute velocity v(x(t+dt))
      {
        // Compute the current velocity from the current forces.
        compute_generalized_velocity();
      }

      // Load balance
      if (timestep_index_ % load_balance_frequency_ == 0) {
        std::cout << "load balancing on proc " << stk::parallel_machine_rank(MPI_COMM_WORLD) << "\n";
        load_balance();
      }

      // IO. If desired, write out the data for time t.
      if (timestep_index_ % io_frequency_ == 0) {
        std::cout << "Time step " << timestep_index_ << " of " << num_time_steps_ << std::endl;
        // update_element_tangent();
        io_broker_ptr_->write_io_broker_timestep(timestep_index_, static_cast<double>(timestep_index_));
      }
    }

    // Do a synchronize to force everybody to stop here, then write the time
    stk::parallel_machine_barrier(bulk_data_ptr_->parallel());
    if (bulk_data_ptr_->parallel_rank() == 0) {
      double avg_time_per_timestep = static_cast<double>(timer.seconds()) / static_cast<double>(num_time_steps_);
      std::cout << "Time per timestep: " << std::setprecision(15) << avg_time_per_timestep << std::endl;
    }
  }

 private:
  //! \name Useful aliases
  //@{

  static constexpr auto node_rank_ = stk::topology::NODE_RANK;
  static constexpr auto edge_rank_ = stk::topology::EDGE_RANK;
  static constexpr auto element_rank_ = stk::topology::ELEMENT_RANK;
  static constexpr auto constraint_rank_ = stk::topology::CONSTRAINT_RANK;
  //@}

  //! \name Internal state
  //@{

  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr_;
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<mundy::meta::MeshReqs> mesh_reqs_ptr_;
  std::shared_ptr<mundy::io::IOBroker> io_broker_ptr_;
  size_t output_file_index_;
  size_t timestep_index_ = 0;
  //@}

  //! \name Class instances
  //@{

  // In the future, these will all become shared pointers to MetaMethods.
  std::shared_ptr<mundy::linkers::ComputeSignedSeparationDistanceAndContactNormal::PolymorphicBaseType>
      compute_ssd_and_cn_ptr_;
  std::shared_ptr<mundy::shapes::ComputeAABB::PolymorphicBaseType> compute_aabb_ptr_;
  std::shared_ptr<mundy::linkers::GenerateNeighborLinkers::PolymorphicBaseType> generate_neighbor_linkers_ptr_;
  std::shared_ptr<mundy::linkers::EvaluateLinkerPotentials::PolymorphicBaseType> evaluate_linker_potentials_ptr_;
  std::shared_ptr<mundy::linkers::LinkerPotentialForceReduction::PolymorphicBaseType>
      linker_potential_force_reduction_ptr_;
  std::shared_ptr<mundy::linkers::DestroyNeighborLinkers::PolymorphicBaseType> destroy_distant_neighbor_linkers_ptr_;
  //@}

  //! \name Fields
  //@{

  stk::mesh::Field<double> *node_coord_field_ptr_;
  stk::mesh::Field<double> *node_velocity_field_ptr_;
  stk::mesh::Field<double> *node_omega_field_ptr_;
  stk::mesh::Field<double> *node_force_field_ptr_;
  stk::mesh::Field<double> *node_torque_field_ptr_;
  stk::mesh::Field<double> *node_radius_field_ptr_;
  stk::mesh::Field<double> *node_length_field_ptr_;
  stk::mesh::Field<size_t> *node_rng_counter_field_ptr_;

  stk::mesh::Field<double> *element_radius_field_ptr_;
  stk::mesh::Field<double> *element_length_field_ptr_;
  stk::mesh::Field<double> *element_orientation_field_ptr_;
  stk::mesh::Field<double> *element_tangent_field_ptr_;
  stk::mesh::Field<double> *element_aabb_field_ptr_;
  stk::mesh::Field<double> *element_youngs_modulus_field_ptr_;
  stk::mesh::Field<double> *element_poissons_ratio_field_ptr_;
  stk::mesh::Field<int> *element_marked_for_division_field_ptr_;

  stk::mesh::Field<double> *linker_potential_force_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *filament_part_ptr_;
  stk::mesh::Part *tip_part_ptr_;
  stk::mesh::Part *spherocylinder_segment_spherocylinder_segment_linkers_part_ptr_;
  //@}

  //! \name Partitioning settings
  //@{

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

  RcbSettings balance_settings_;
  //@}

  //! \name User parameters
  //@{
  std::string input_file_name_ = "filament_params.yaml";

  double filament_radius_ = 0.5;
  double filament_division_length_ = 4.0 * filament_radius_;
  double filament_initial_length_ = 2.0 * filament_radius_;
  double filament_growth_rate_ = 0.1;

  double filament_youngs_modulus_ = 1000.0;
  double filament_poissons_ratio_ = 0.3;
  double filament_density_ = 1.0;

  double viscosity_ = 1;

  double timestep_size_ = 1e-3;
  size_t num_time_steps_ = 1000;
  size_t io_frequency_ = 10;
  size_t load_balance_frequency_ = 1000;
  double boundary_radius_ = 10;
  //@}
};  // FilamentSim

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  // Run the simulation using the given parameters
  FilamentSim().run(argc, argv);

  MPI_Barrier(MPI_COMM_WORLD);
  double end = MPI_Wtime();

  if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
    std::cout << "(nproc = " << stk::parallel_machine_size(MPI_COMM_WORLD);
    std::cout << ") time: " << end - start << "\n";
  }

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
