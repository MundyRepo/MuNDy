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
#include <fstream>
#include <iostream>  // for std::cout, std::endl
#include <iterator>
#include <span>    // for std::span
#include <vector>  // for std::vector

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
#include <mundy_linkers/neighbor_linkers/SpherocylinderSpherocylinderLinkers.hpp>  // for mundy::...::SpherocylinderSpherocylinderLinkers
#include <mundy_math/Matrix3.hpp>                                                  // for mundy::math::Matrix3
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
void subdivide_spherocylinders(stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
                               const stk::mesh::Field<double> &coordinate_field,
                               const stk::mesh::Field<double> &orientation_field,
                               const stk::mesh::Field<double> &length_field,
                               const stk::mesh::Field<double> &radius_field,
                               const Kokkos::UnorderedMap<stk::mesh::Entity, stk::mesh::Entity> &parent_to_child_map,
                               const std::tuple<NodeFieldToCopyType...> &node_fields_to_copy,
                               const std::tuple<ElementFieldToCopyType...> &element_fields_to_copy) {
  // Update the length and coordinates of the children
  // This is only done for locally owned entities.
  mundy::mesh::for_each_entity_run(
      bulk_data, stk::topology::ELEMENT_RANK, selector & bulk_data.mesh_meta_data().locally_owned_part(),
      [&coordinate_field, &orientation_field, &radius_field, &length_field, &parent_to_child_map, &node_fields_to_copy,
       &element_fields_to_copy](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &parent) {
        const size_t parent_map_index = parent_to_child_map.find(parent);
        const bool valid_parent_index = parent_to_child_map.valid_at(parent_map_index);
        if (valid_parent_index) {
          // The parent becomes the left spherocylinder and the child becomes the right.
          // Fetch the downward connections.
          const stk::mesh::Entity &child = parent_to_child_map.value_at(parent_map_index);
          const stk::mesh::Entity &parent_node = bulk_data.begin_nodes(parent)[0];
          const stk::mesh::Entity &child_node = bulk_data.begin_nodes(child)[0];

          // Copy the parent's fields to the children
          copy_entity_fields<num_node_fields, size_of_node_fields_to_copy>(parent_node, child_node,
                                                                           node_fields_to_copy);
          copy_entity_fields<num_element_fields, size_of_element_fields_to_copy>(parent, child, element_fields_to_copy);

          // Compute the position of the new node
          // o-----n-----o -> o--n--o o--n--o
          // To prevent overlap, the position of the center nodes are offset by the element radius.
          //
          // At this point, the children have the same fields as the parent. The only think that needs updated is
          // their length and coordinates. Length of children is parent length / 2 - parent radius Center of
          // children is parent center +/- parent tangent * (parent radius - child length / 2)
          const auto parent_orientation = mundy::mesh::quaternion_field_data(orientation_field, parent);
          const auto parent_tangent = parent_orientation * mundy::math::Vector3<double>(1.0, 0.0, 0.0);
          auto parent_node_coords = mundy::mesh::vector3_field_data(coordinate_field, parent_node);
          auto child_node_coords = mundy::mesh::vector3_field_data(coordinate_field, child_node);

          const double parent_length = stk::mesh::field_data(length_field, parent)[0];
          const double parent_radius = stk::mesh::field_data(radius_field, parent)[0];
          const double child_length = 0.5 * parent_length - parent_radius;
          stk::mesh::field_data(radius_field, child)[0] = parent_radius;
          stk::mesh::field_data(length_field, child)[0] = child_length;
          stk::mesh::field_data(length_field, parent)[0] = child_length;
          mundy::mesh::quaternion_field_data(orientation_field, child) = parent_orientation;

          const auto center_offset = parent_tangent * (parent_radius + 0.5 * child_length);
          child_node_coords = parent_node_coords + center_offset;
          parent_node_coords = parent_node_coords - center_offset;
        }
      });

  // At this point, all locally owned parents and their children are up-to-date.
  // Communicate the changes to the ghosted entities.
  std::vector<const stk::mesh::FieldBase *> fields_to_communicate = {&coordinate_field, &orientation_field,
                                                                     &length_field, &radius_field};
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
bool subdivide_flagged_spherocylinders(stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
                                       const stk::mesh::Field<double> &coordinate_field,
                                       const stk::mesh::Field<double> &orientation_field,
                                       const stk::mesh::Field<double> &length_field,
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
  Kokkos::UnorderedMap<stk::mesh::Entity, stk::mesh::Entity> parent_to_child_map(new_element_count);
  for (size_t i = 0; i < num_entities_to_maybe_divide; i++) {
    if (should_divide[i]) {
      const stk::mesh::Entity &parent = entities_to_maybe_divide[i];
      const size_t child_lid = parent_lid_to_child_lid[i];
      const stk::mesh::Entity &child = new_elements[child_lid];
      const stk::mesh::Entity &child_node = new_nodes[child_lid];

      MUNDY_THROW_ASSERT(bulk_data.is_valid(parent), std::invalid_argument, "Parent entity is not valid.");
      MUNDY_THROW_ASSERT(bulk_data.is_valid(child), std::invalid_argument, "Child entity is not valid.");
      MUNDY_THROW_ASSERT(bulk_data.is_valid(child_node), std::invalid_argument, "Child node is not valid.");

      // Store the parent to child map
      bool kv_insert_success = parent_to_child_map.insert(parent, child).success();
      assert(kv_insert_success == true);

      // Add the child to the parent's parts (non-internal)
      stk::mesh::ConstPartVector elem_parent_parts_without_internal;
      get_parent_entity_parts(bulk_data, parent, elem_parent_parts_without_internal);
      bulk_data.change_entity_parts(child, elem_parent_parts_without_internal);

      // Connect the child to its node
      bulk_data.declare_relation(child, child_node, 0);
    }
  }

  bulk_data.modification_end();

  // Subdivide the entities.
  subdivide_spherocylinders<num_node_fields, num_element_fields, size_of_node_fields_to_copy,
                            size_of_element_fields_to_copy>(bulk_data, selector, coordinate_field, orientation_field,
                                                            length_field, radius_field, parent_to_child_map,
                                                            node_fields_to_copy, element_fields_to_copy);

  return new_element_count > 0;
}

bool subdivide_flagged_spherocylinders(stk::mesh::BulkData &bulk_data, const stk::mesh::Selector &selector,
                                       const stk::mesh::Field<double> &coordinate_field,
                                       const stk::mesh::Field<double> &orientation_field,
                                       const stk::mesh::Field<double> &length_field,
                                       const stk::mesh::Field<double> &radius_field,
                                       const stk::mesh::Field<int> &flag_field, const int &divide_flag_value) {
  constexpr size_t num_node_fields = 0;
  constexpr size_t num_element_fields = 0;
  constexpr Kokkos::Array<unsigned, num_node_fields> size_of_node_fields_to_copy = {};
  constexpr Kokkos::Array<unsigned, num_element_fields> size_of_element_fields_to_copy = {};
  return subdivide_flagged_spherocylinders<num_node_fields, num_element_fields, size_of_node_fields_to_copy,
                                           size_of_element_fields_to_copy>(
      bulk_data, selector, coordinate_field, orientation_field, length_field, radius_field, flag_field,
      divide_flag_value, std::make_tuple(), std::make_tuple());
}

class BacteriaSim {
 public:
  BacteriaSim() = default;

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

    bacteria_radius_ = param_list_.get<double>("bacteria_radius");
    bacteria_initial_length_ = param_list_.get<double>("bacteria_initial_length");
    bacteria_division_length_ = param_list_.get<double>("bacteria_division_length");
    bacteria_growth_rate_ = param_list_.get<double>("bacteria_growth_rate");
    number_of_bacteria_ = param_list_.get<int>("number_of_bacteria");

    bacteria_density_ = param_list_.get<double>("bacteria_density");
    bacteria_youngs_modulus_ = param_list_.get<double>("bacteria_youngs_modulus");
    bacteria_poissons_ratio_ = param_list_.get<double>("bacteria_poissons_ratio");

    buffer_distance_ = param_list_.get<double>("buffer_distance");

    num_time_steps_ = param_list_.get<long long>("num_time_steps");
    timestep_size_ = param_list_.get<double>("timestep_size");
    io_frequency_ = param_list_.get<int>("io_frequency");
    load_balance_frequency_ = param_list_.get<int>("load_balance_frequency");

    check_input_parameters();
  }

  void check_input_parameters() {
    debug_print("Checking input parameters.");
    MUNDY_THROW_REQUIRE(bacteria_radius_ > 0, std::invalid_argument, "bacteria_radius_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(bacteria_initial_length_ > -1e-12, std::invalid_argument,
                        "bacteria_initial_length_ must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(bacteria_division_length_ > 0, std::invalid_argument,
                        "bacteria_division_length_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(bacteria_density_ > 0, std::invalid_argument, "bacteria_density_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(bacteria_youngs_modulus_ > 0, std::invalid_argument,
                        "bacteria_youngs_modulus_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(bacteria_poissons_ratio_ > 0, std::invalid_argument,
                        "bacteria_poissons_ratio_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(number_of_bacteria_ > 0, std::invalid_argument, "number_of_bacteria_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(buffer_distance_ > 0, std::invalid_argument, "buffer_distance_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(num_time_steps_ > 0, std::invalid_argument, "num_time_steps_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(timestep_size_ > 0, std::invalid_argument, "timestep_size_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(io_frequency_ > 0, std::invalid_argument, "io_frequency_ must be greater than 0.");
    MUNDY_THROW_REQUIRE(load_balance_frequency_ > 0, std::invalid_argument,
                        "load_balance_frequency_ must be greater than 0.");
  }

  void dump_user_inputs() {
    debug_print("Dumping user inputs.");
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;
      std::cout << "  bacteria_radius_: " << bacteria_radius_ << std::endl;
      std::cout << "  bacteria_initial_length_: " << bacteria_initial_length_ << std::endl;
      std::cout << "  bacteria_division_length_: " << bacteria_division_length_ << std::endl;
      std::cout << "  bacteria_growth_rate_: " << bacteria_growth_rate_ << std::endl;
      std::cout << "  bacteria_youngs_modulus_: " << bacteria_youngs_modulus_ << std::endl;
      std::cout << "  bacteria_poissons_ratio_: " << bacteria_poissons_ratio_ << std::endl;
      std::cout << "  bacteria_density_: " << bacteria_density_ << std::endl;
      std::cout << "  number_of_bacteria_: " << number_of_bacteria_ << std::endl;
      std::cout << "  num_time_steps_: " << num_time_steps_ << std::endl;
      std::cout << "  timestep_size_: " << timestep_size_ << std::endl;
      std::cout << "  io_frequency_: " << io_frequency_ << std::endl;
      std::cout << "  load_balance_frequency_: " << load_balance_frequency_ << std::endl;
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
    // We require that the bacteria are sphereocylinders, so they need to have PARTICLE topology
    auto bacteria_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    bacteria_part_reqs->set_part_name("BACTERIA")
        .set_part_topology(stk::topology::PARTICLE)

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
        .add_field_reqs<int>("ELEMENT_MARKED_FOR_DIVISION", element_rank_, 1, 1)
        .add_field_reqs<double>("ELEMENT_AABB_DISPLACEMENT", element_rank_, 6, 1);
    mesh_reqs_ptr_->add_field_reqs<double>("ELEMENT_AABB", element_rank_, 6, 2);
    mesh_reqs_ptr_->add_and_sync_part_reqs(bacteria_part_reqs);
    mundy::shapes::Spherocylinders::add_and_sync_subpart_reqs(bacteria_part_reqs);

#ifdef DEBUG
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      mesh_reqs_ptr_->print();
    }
#endif

    // Add the requirements for our initialized methods to the mesh
    // When we eventually switch to the configurator, these individual fixed params will become sublists within a single
    // master parameter list. Note, sublist will return a reference to the sublist with the given name.
    auto compute_ssd_and_cn_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SPHEROCYLINDER_LINKER"));
    auto compute_aabb_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER"));
    compute_aabb_fixed_params.sublist("SPHEROCYLINDER")
        .set("valid_entity_part_names", mundy::core::make_string_array("BACTERIA"));
    auto generate_neighbor_linkers_fixed_params =
        Teuchos::ParameterList()
            .set("enabled_technique_name", "STK_SEARCH")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array("SPHEROCYLINDER_SPHEROCYLINDER_LINKERS"));
    generate_neighbor_linkers_fixed_params.sublist("STK_SEARCH")
        .set("valid_source_entity_part_names", mundy::core::make_string_array("BACTERIA"))
        .set("valid_target_entity_part_names", mundy::core::make_string_array("BACTERIA"));
    auto evaluate_linker_potentials_fixed_params = Teuchos::ParameterList().set(
        "enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER_SPHEROCYLINDER_HERTZIAN_CONTACT"));
    auto linker_potential_force_reduction_fixed_params =
        Teuchos::ParameterList().set("enabled_kernel_names", mundy::core::make_string_array("SPHEROCYLINDER"));
    linker_potential_force_reduction_fixed_params.sublist("SPHEROCYLINDER")
        .set("valid_entity_part_names", mundy::core::make_string_array("BACTERIA"));
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
    auto compute_aabb_mutable_params = Teuchos::ParameterList().set("buffer_distance", buffer_distance_);
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
    element_aabb_displacement_field_ptr_ = fetch_field<double>("ELEMENT_AABB_DISPLACEMENT", element_rank_);

    linker_potential_force_field_ptr_ = fetch_field<double>("LINKER_POTENTIAL_FORCE", constraint_rank_);

    // Fetch the parts
    bacteria_part_ptr_ = fetch_part("BACTERIA");
    spherocylinder_spherocylinder_linkers_part_ptr_ = fetch_part("SPHEROCYLINDER_SPHEROCYLINDER_LINKERS");
    MUNDY_THROW_REQUIRE(bacteria_part_ptr_->topology() == stk::topology::PARTICLE, std::logic_error,
                        "BACTERIA part must have PARTICLE topology.");
  }

  void setup_io() {
    debug_print("Setting up IO.");

    // Create a mundy io broker via it's fixed parameters
    auto fixed_params_iobroker =
        Teuchos::ParameterList()
            .set("enabled_io_parts", mundy::core::make_string_array("BACTERIA"))
            .set("enabled_io_fields_node_rank",
                 mundy::core::make_string_array("NODE_VELOCITY", "NODE_OMEGA", "NODE_FORCE", "NODE_TORQUE",
                                                "NODE_RNG_COUNTER"))
            .set("enabled_io_fields_element_rank",
                 mundy::core::make_string_array("ELEMENT_RADIUS", "ELEMENT_LENGTH", "ELEMENT_ORIENTATION",
                                                "ELEMENT_TANGENT", "ELEMENT_AABB", "ELEMENT_YOUNGS_MODULUS",
                                                "ELEMENT_POISSONS_RATIO", "ELEMENT_MARKED_FOR_DIVISION"))
            .set("coordinate_field_name", "NODE_COORDS")
            .set("transient_coordinate_field_name", "TRANSIENT_NODE_COORDINATES")
            .set("exodus_database_output_filename", "Bacteria.exo")
            .set("parallel_io_mode", "hdf5")
            .set("database_purpose", "results");
    // Create the IO broker
    io_broker_ptr_ = mundy::io::IOBroker::create_new_instance(bulk_data_ptr_.get(), fixed_params_iobroker);
  }

  void declare_and_initialize_bacteria() {
    debug_print("Declaring and initializing the bacteria.");

    // Declare the bacteria on rank 0
    bulk_data_ptr_->modification_begin();
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      for (int i = 1; i < number_of_bacteria_ + 1; i++) {
        stk::mesh::Entity bacteria_node =
            bulk_data_ptr_->declare_entity(stk::topology::NODE_RANK, i, stk::mesh::PartVector{bacteria_part_ptr_});
        stk::mesh::Entity bacteria =
            bulk_data_ptr_->declare_entity(stk::topology::ELEMENT_RANK, i, stk::mesh::PartVector{bacteria_part_ptr_});
        bulk_data_ptr_->declare_relation(bacteria, bacteria_node, 0);
      }
    }
    bulk_data_ptr_->modification_end();

    // Initialize it
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      for (int i = 1; i < number_of_bacteria_ + 1; i++) {
        stk::mesh::Entity bacteria_node = bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, i);
        stk::mesh::Entity bacteria = bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, i);

        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(bacteria_node), std::invalid_argument,
                           "Bacteria node is not valid.");
        MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(bacteria), std::invalid_argument, "Bacteria element is not valid.");

        stk::mesh::field_data(*node_rng_counter_field_ptr_, bacteria_node)[0] = 0;
        mundy::mesh::vector3_field_data(*node_coord_field_ptr_, bacteria_node).set(i * 1000.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_velocity_field_ptr_, bacteria_node).set(0.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_omega_field_ptr_, bacteria_node).set(0.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_force_field_ptr_, bacteria_node).set(0.0, 0.0, 0.0);
        mundy::mesh::vector3_field_data(*node_torque_field_ptr_, bacteria_node).set(0.0, 0.0, 0.0);

        stk::mesh::field_data(*element_radius_field_ptr_, bacteria)[0] = bacteria_radius_;
        stk::mesh::field_data(*element_length_field_ptr_, bacteria)[0] = bacteria_initial_length_;
        stk::mesh::field_data(*element_youngs_modulus_field_ptr_, bacteria)[0] = bacteria_youngs_modulus_;
        stk::mesh::field_data(*element_poissons_ratio_field_ptr_, bacteria)[0] = bacteria_poissons_ratio_;

        mundy::math::Vector3<double> current_tangent(1.0, 0.0, 0.0);
        mundy::math::Vector3<double> x_axis(1.0, 0.0, 0.0);
        mundy::mesh::quaternion_field_data(*element_orientation_field_ptr_, bacteria) =
            mundy::math::quat_from_parallel_transport(x_axis, current_tangent);
        mundy::mesh::vector3_field_data(*element_tangent_field_ptr_, bacteria) = current_tangent;
      }
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

  void update_accumulators() {
    stk::mesh::Field<double> &element_aabb_field = *element_aabb_field_ptr_;
    stk::mesh::Field<double> &element_aabb_field_old = element_aabb_field.field_of_state(stk::mesh::StateN);
    stk::mesh::Field<double> &element_corner_displacement_field = *element_aabb_displacement_field_ptr_;

    // Update the accumulators based on the difference to the previous state
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, *bacteria_part_ptr_,
        [&element_aabb_field, &element_aabb_field_old, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_aabb = stk::mesh::field_data(element_aabb_field, aabb_entity);
          double *element_aabb_old = stk::mesh::field_data(element_aabb_field_old, aabb_entity);
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Add the (new_aabb - old_aabb) to the corner displacement
          element_corner_displacement[0] += element_aabb[0] - element_aabb_old[0];
          element_corner_displacement[1] += element_aabb[1] - element_aabb_old[1];
          element_corner_displacement[2] += element_aabb[2] - element_aabb_old[2];
          element_corner_displacement[3] += element_aabb[3] - element_aabb_old[3];
          element_corner_displacement[4] += element_aabb[4] - element_aabb_old[4];
          element_corner_displacement[5] += element_aabb[5] - element_aabb_old[5];
        });
  }

  void check_update_neighbor_list() {
    // Local variable for if we should update the neighbor list (do as an integer for now because MPI doesn't like
    // bools)
    int local_update_neighbor_list_int = 0;

    stk::mesh::Field<double> &element_corner_displacement_field = *element_aabb_displacement_field_ptr_;
    const double buffer_distance_sqr = buffer_distance_ * buffer_distance_;

    // Check if each corner has moved skin_distance/2. Or, if dr_mag2 >= skin_distance^2/4
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, stk::topology::ELEMENT_RANK, *bacteria_part_ptr_,
        [&local_update_neighbor_list_int, &buffer_distance_sqr, &element_corner_displacement_field](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &aabb_entity) {
          // Get the dr for each element (should be able to just do an addition of the difference) into the accumulator.
          double *element_corner_displacement = stk::mesh::field_data(element_corner_displacement_field, aabb_entity);

          // Compute dr2 for each corner
          double dr2_corner0 = element_corner_displacement[0] * element_corner_displacement[0] +
                               element_corner_displacement[1] * element_corner_displacement[1] +
                               element_corner_displacement[2] * element_corner_displacement[2];
          double dr2_corner1 = element_corner_displacement[3] * element_corner_displacement[3] +
                               element_corner_displacement[4] * element_corner_displacement[4] +
                               element_corner_displacement[5] * element_corner_displacement[5];

          if (dr2_corner0 >= buffer_distance_sqr || dr2_corner1 >= buffer_distance_sqr) {
#pragma omp write
            local_update_neighbor_list_int = 1;
          }
        });

    // Communicate local_update_neighbor_list to all ranks. Convert to an integer first (MPI doesn't handle booleans
    // well).
    int global_update_neighbor_list_int = 0;
    MPI_Allreduce(&local_update_neighbor_list_int, &global_update_neighbor_list_int, 1, MPI_INT, MPI_LOR,
                  MPI_COMM_WORLD);
    // Convert back to the boolean for the global version and or it with the original value (in case somebody else set
    // the neighbor list update 'signal').
    update_neighbor_list_ = update_neighbor_list_ || (global_update_neighbor_list_int == 1);
  }

  void zero_out_accumulator_fields() {
    mundy::mesh::utils::fill_field_with_value<double>(*element_aabb_displacement_field_ptr_,
                                                      std::array<double, 6>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  }

  void compute_hertzian_contact_force_and_torque() {
    debug_print("Computing the Hertzian contact force and torque.");

    // Compute the AABBs for the rods
    debug_print("Computing the AABBs for the rods.");
    compute_aabb_ptr_->execute(*bacteria_part_ptr_);
    check_update_neighbor_list();

    double nl_start = 0.0, nl_end = 0.0;
    double force_start = 0.0, force_end = 0.0;

    // Check if the rod-rod neighbor list needs updated or not
    if (update_neighbor_list_) {
      nl_start = MPI_Wtime();

      zero_out_accumulator_fields();

      // Delete rod-rod neighbor linkers that are too far apart
      debug_print("Deleting rod-rod neighbor linkers that are too far apart.");
      Kokkos::Timer timer0;
      destroy_distant_neighbor_linkers_ptr_->execute(*spherocylinder_spherocylinder_linkers_part_ptr_);
      debug_print("Time to destroy distant neighbor linkers: " + std::to_string(timer0.seconds()));

      // Generate neighbor linkers between nearby rods
      debug_print("Generating neighbor linkers between nearby rods.");
      Kokkos::Timer timer1;
      generate_neighbor_linkers_ptr_->execute(*bacteria_part_ptr_, *bacteria_part_ptr_);
      debug_print("Time to generate neighbor linkers: " + std::to_string(timer1.seconds()));

      update_neighbor_list_ = false;

      nl_end = MPI_Wtime();
    }

    force_start = MPI_Wtime();

    // Hertzian contact force evaluation
    // Compute the signed separation distance and contact normal between neighboring rods
    debug_print("Computing the signed separation distance and contact normal between neighboring rods.");
    compute_ssd_and_cn_ptr_->execute(*spherocylinder_spherocylinder_linkers_part_ptr_);

    // Evaluate the Hertzian contact potential between neighboring rods
    debug_print("Evaluating the Hertzian contact potential between neighboring rods.");
    evaluate_linker_potentials_ptr_->execute(*spherocylinder_spherocylinder_linkers_part_ptr_);

    // Sum the linker potential force to get the induced node force on each rod
    debug_print("Summing the linker potential force to get the induced node force on each rod.");
    linker_potential_force_reduction_ptr_->execute(*bacteria_part_ptr_);

    force_end = MPI_Wtime();
  }

  void compute_generalized_velocity() {
    debug_print("Computing the generalized velocity using the mobility problem.");

    // For us, we consider dry local drag with mass lumping at the nodes. This diagonalized the mobility problem and
    // makes each node independent, coupled only through the internal and constraint forces. The mobility problem is
    //
    // \dot{x}(t) = f(t) / (6 pi viscosity r)
    // omega(t) = torque(t) / (8 pi viscosity r^3)

    // Get references to internal members so we aren't passing around *this
    const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
    const stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
    const stk::mesh::Field<double> &node_torque_field = *node_torque_field_ptr_;
    stk::mesh::Field<double> &node_velocity_field = *node_velocity_field_ptr_;
    stk::mesh::Field<double> &node_omega_field = *node_omega_field_ptr_;
    const double viscosity = viscosity_;

    // Solve the mobility problem for the nodes
    const double one_over_6_pi_viscosity = 1.0 / (6.0 * M_PI * viscosity);
    const double one_over_8_pi_viscosity = 1.0 / (8.0 * M_PI * viscosity);
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *bacteria_part_ptr_,
        [&node_force_field, &node_velocity_field, &element_radius_field, &node_torque_field, &node_omega_field,
         &one_over_6_pi_viscosity, &one_over_8_pi_viscosity]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                             const stk::mesh::Entity &element) {
          // Fetch the connected node
          const stk::mesh::Entity node = bulk_data.begin_nodes(element)[0];

          // Get the required input fields
          const auto node_force = mundy::mesh::vector3_field_data(node_force_field, node);
          const auto node_torque = mundy::mesh::vector3_field_data(node_torque_field, node);
          const double element_radius = stk::mesh::field_data(element_radius_field, element)[0];

          // Get the output fields
          auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node);
          auto node_omega = mundy::mesh::vector3_field_data(node_omega_field, node);

          // Compute the generalized velocity
          const double inv_radius = 1.0 / element_radius;
          const double inv_radius3 = inv_radius * inv_radius * inv_radius;
          node_velocity += (one_over_6_pi_viscosity * inv_radius) * node_force;
          node_omega += (one_over_8_pi_viscosity * inv_radius3) * node_torque;
        });
  }

  void update_generalized_position() {
    debug_print("Updating the generalized position using Euler's method.");

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
    stk::mesh::Field<double> &element_orientation_field = *element_orientation_field_ptr_;
    const stk::mesh::Field<double> &node_coord_field_old = node_coord_field.field_of_state(stk::mesh::StateN);
    const stk::mesh::Field<double> &node_velocity_field_old =
        node_velocity_field_ptr_->field_of_state(stk::mesh::StateN);
    const stk::mesh::Field<double> &element_orientation_field_old =
        element_orientation_field.field_of_state(stk::mesh::StateN);
    const stk::mesh::Field<double> &node_omega_field_old = node_omega_field_ptr_->field_of_state(stk::mesh::StateN);
    const double timestep_size = timestep_size_;

    // Update the generalized position using Euler's method
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *bacteria_part_ptr_,
        [&node_coord_field, &node_coord_field_old, &node_velocity_field_old, &element_orientation_field,
         &element_orientation_field_old, &node_omega_field_old,
         &timestep_size]([[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &element) {
          // Fetch the connected node
          const stk::mesh::Entity node = bulk_data.begin_nodes(element)[0];

          // Update the position
          mundy::mesh::vector3_field_data(node_coord_field, node) =
              mundy::mesh::vector3_field_data(node_coord_field_old, node) +
              timestep_size * mundy::mesh::vector3_field_data(node_velocity_field_old, node);

          // Update the quaternion using Delong, JCP, 2015, Appendix A eq1, not linearized
          const auto omega = mundy::mesh::vector3_field_data(node_omega_field_old, node);
          const auto old_orientation = mundy::mesh::quaternion_field_data(element_orientation_field_old, element);
          const double w = mundy::math::norm(omega);
          if (w > std::numeric_limits<double>::epsilon()) {
            const double winv = 1 / w;
            const double sw = sin(w * timestep_size / 2);
            const double cw = cos(w * timestep_size / 2);
            const double s = old_orientation.w();
            const mundy::math::Vector3<double> p(old_orientation.x(), old_orientation.y(), old_orientation.z());
            const mundy::math::Vector3<double> xyz =
                s * sw * omega * winv + cw * p + sw * winv * (mundy::math::cross(omega, p));
            mundy::mesh::quaternion_field_data(element_orientation_field, element).w() =
                s * cw - (mundy::math::dot(p, omega))*sw * winv;
            mundy::mesh::quaternion_field_data(element_orientation_field, element).x() = xyz[0];
            mundy::mesh::quaternion_field_data(element_orientation_field, element).y() = xyz[1];
            mundy::mesh::quaternion_field_data(element_orientation_field, element).z() = xyz[2];
            mundy::mesh::quaternion_field_data(element_orientation_field, element).normalize();
          } else {
            mundy::mesh::quaternion_field_data(element_orientation_field, element) =
                mundy::mesh::quaternion_field_data(element_orientation_field_old, element);
          }
        });
  }

  void grow_bacteria() {
    debug_print("Growing the bacteria.");

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
    const double bacteria_growth_rate = bacteria_growth_rate_;
    const double timestep_size = timestep_size_;

    // Grow the bacteria
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *bacteria_part_ptr_,
        [&element_length_field, &bacteria_growth_rate, &timestep_size](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &bacteria) {
          stk::mesh::field_data(element_length_field, bacteria)[0] += timestep_size * bacteria_growth_rate;
        });
  }

  auto make_ref_tuple(auto &...args) {
    return std::make_tuple(std::ref(args)...);
  }

  void divide_bacteria() {
    debug_print("Dividing the bacteria.");

    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
    stk::mesh::Field<int> &element_marked_for_division_field = *element_marked_for_division_field_ptr_;
    stk::mesh::Field<size_t> &node_rng_counter_field = *node_rng_counter_field_ptr_;
    stk::mesh::Field<double> &node_omega_field = *node_omega_field_ptr_;
    const double bacteria_division_length = bacteria_division_length_;

    // Loop over all particles and stash if they divide or not
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *bacteria_part_ptr_,
        [&element_length_field, &element_marked_for_division_field, &bacteria_division_length](
            [[maybe_unused]] const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &bacteria) {
          stk::mesh::field_data(element_marked_for_division_field, bacteria)[0] =
              (stk::mesh::field_data(element_length_field, bacteria)[0] > bacteria_division_length);
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
    const bool division_occurred =
        subdivide_flagged_spherocylinders<num_node_fields, num_element_fields, size_of_node_fields_to_copy,
                                          size_of_element_fields_to_copy>(
            *bulk_data_ptr_, *bacteria_part_ptr_, *node_coord_field_ptr_, *element_orientation_field_ptr_,
            *element_length_field_ptr_, *element_radius_field_ptr_, *element_marked_for_division_field_ptr_,
            divide_flag_value, extra_node_fields_to_copy, extra_element_fields_to_copy);

    update_neighbor_list_ = division_occurred;

    // // Loop over all particles and apply a small random orientational kick to any that divided
    // // This is independent of the parent-child relationship.
    // mundy::mesh::for_each_entity_run(*bulk_data_ptr_, element_rank_, *bacteria_part_ptr_,
    //                                [&node_rng_counter_field, &node_omega_field, &element_marked_for_division_field](
    //                                    const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &bacteria) {
    //                                  if (stk::mesh::field_data(element_marked_for_division_field, bacteria)[0] == 1)
    //                                  {
    //                                    // Fetch the connected node
    //                                    const stk::mesh::Entity node = bulk_data.begin_nodes(bacteria)[0];

    //                                    // Get the output fields
    //                                    auto node_omega = mundy::mesh::vector3_field_data(node_omega_field, node);
    //                                    size_t *node_rng_counter = stk::mesh::field_data(node_rng_counter_field,
    //                                    node); const size_t node_gid = bulk_data.identifier(node); openrand::Philox
    //                                    rng(node_gid, node_rng_counter[0]);

    //                                    // Apply a very small random orientational kick to keep particles from being
    //                                    // perfectly aligned
    //                                    const double max_kick = 1.0;
    //                                    node_omega[0] += max_kick * (2.0 * rng.rand<double>() - 1.0);
    //                                    node_omega[1] += max_kick * (2.0 * rng.rand<double>() - 1.0);
    //                                    node_omega[2] += max_kick * (2.0 * rng.rand<double>() - 1.0);
    //                                    node_rng_counter[0] += 1;
    //                                  }
    //                                });
  }

  void update_element_tangent() {
    // Get references to internal members so we aren't passing around *this
    stk::mesh::Field<double> &element_orientation_field = *element_orientation_field_ptr_;
    stk::mesh::Field<double> &element_tangent_field = *element_tangent_field_ptr_;

    // Update the tangent
    mundy::mesh::for_each_entity_run(
        *bulk_data_ptr_, element_rank_, *bacteria_part_ptr_,
        [&element_orientation_field, &element_tangent_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                                                             const stk::mesh::Entity &bacteria) {
          // Get the output fields
          auto element_tangent = mundy::mesh::vector3_field_data(element_tangent_field, bacteria);
          const auto element_orientation = mundy::mesh::quaternion_field_data(element_orientation_field, bacteria);

          // Compute the tangent
          element_tangent = element_orientation * mundy::math::Vector3<double>(1.0, 0.0, 0.0);
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
    declare_and_initialize_bacteria();
    setup_io();

    // Time loop
    print_rank0(std::string("Running the simulation for ") + std::to_string(num_time_steps_) + " time steps.");

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
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

      // time division
      // Growth then division
      {
        // MPI_Barrier(MPI_COMM_WORLD);
        double growth_start = MPI_Wtime();

        divide_bacteria();
        grow_bacteria();

        // MPI_Barrier(MPI_COMM_WORLD);
        double growth_end = MPI_Wtime();
      }

      // time neighbor list and force computations
      // Evaluate forces f(x(t + dt)).
      {
        // Hertzian contact force
        compute_hertzian_contact_force_and_torque();
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

        // Some fields are only needed for I/O, so we only compute them when we need to write out the data.
        // So far, this is just the element tangent
        update_element_tangent();

        io_broker_ptr_->write_io_broker_timestep(timestep_index_, static_cast<double>(timestep_index_));
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "(nproc = " << stk::parallel_machine_size(MPI_COMM_WORLD);
      std::cout << ") time: " << end - start << "\n";
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
  size_t number_of_bacteria_;
  bool update_neighbor_list_;
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
  stk::mesh::Field<double> *element_aabb_displacement_field_ptr_;

  stk::mesh::Field<double> *linker_potential_force_field_ptr_;
  //@}

  //! \name Parts
  //@{

  stk::mesh::Part *bacteria_part_ptr_;
  stk::mesh::Part *spherocylinder_spherocylinder_linkers_part_ptr_;
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
  std::string input_file_name_ = "bacteria_params.yaml";

  double bacteria_radius_ = 0.5;
  double bacteria_division_length_ = 4.0 * bacteria_radius_;
  double bacteria_initial_length_ = 2.0 * bacteria_radius_;
  double bacteria_growth_rate_ = 0.1;

  double bacteria_youngs_modulus_ = 1000.0;
  double bacteria_poissons_ratio_ = 0.3;
  double bacteria_density_ = 1.0;

  double buffer_distance_ = bacteria_radius_;

  double viscosity_ = 1;

  double timestep_size_ = 1e-3;
  size_t num_time_steps_ = 1000;
  size_t io_frequency_ = 10;
  size_t load_balance_frequency_ = 10;
  //@}
};  // BacteriaSim

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation using the given parameters
  BacteriaSim().run(argc, argv);

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
