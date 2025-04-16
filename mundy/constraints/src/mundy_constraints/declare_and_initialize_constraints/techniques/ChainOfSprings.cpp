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

/// \file ChainOfSprings.cpp
/// \brief Definition of DeclareAndInitConstraints' ChainOfSprings technique.

// External libs
#include <fmt/format.h>  // for fmt::format
#include <openrand/philox.h>

// C++ core libs
#include <iostream>  // for std::cout, std::endl
#include <memory>    // for std::shared_ptr, std::unique_ptr
#include <string>    // for std::string
#include <vector>    // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_constraints/HookeanSprings.hpp>  // for mundy::constraints::HookeanSprings
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ArchlengthCoordinateMapping.hpp>  // for mundy::constraints::...::ArchlengthCoordinateMapping
#include <mundy_constraints/declare_and_initialize_constraints/techniques/ChainOfSprings.hpp>  // for mundy::constraints::...::ChainOfSprings
#include <mundy_core/throw_assert.hpp>                                                         // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                  // for mundy::mesh::BulkData
#include <mundy_mesh/fmt_stk_types.hpp>             // adds fmt::format for stk types
#include <mundy_shapes/Spheres.hpp>                 // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace constraints {

namespace declare_and_initialize_constraints {

namespace techniques {

// \name Constructors and destructor
//{

ChainOfSprings::ChainOfSprings(mundy::mesh::BulkData *const bulk_data_ptr_, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr_), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "ChainOfSprings: bulk_data_ptr_ cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(ChainOfSprings::get_valid_fixed_params());

  // Get the control parameters.
  generate_hookean_springs_ = valid_fixed_params.get<bool>("generate_hookean_springs");
  generate_angular_springs_ = valid_fixed_params.get<bool>("generate_angular_springs");
  generate_spheres_at_nodes_ = valid_fixed_params.get<bool>("generate_spheres_at_nodes");
  generate_spherocylinder_segments_along_edges_ =
      valid_fixed_params.get<bool>("generate_spherocylinder_segments_along_edges");

  MUNDY_THROW_REQUIRE(
      generate_hookean_springs_ || generate_angular_springs_ || generate_spheres_at_nodes_ ||
          generate_spherocylinder_segments_along_edges_,
      std::invalid_argument,
      "ChainOfSprings: At least one of the following must be true: generate_hookean_springs, "
      "generate_angular_springs, generate_spheres_at_nodes, or generate_spherocylinder_segments_along_edges.");

  // Get the field and pointers.
  auto validate_field_ptr = [](const stk::mesh::FieldBase *const field_ptr, const std::string &field_name) {
    MUNDY_THROW_REQUIRE(
        field_ptr != nullptr, std::invalid_argument,
        std::string("ChainOfSprings: Expected a field with name '") + field_name + "' but field does not exist.");
  };
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(
          part != nullptr, std::invalid_argument,
          std::string("ChainOfSprings: Expected a part with name '") + part_name + "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  const std::string node_coord_field_name = mundy::constraints::HookeanSprings::get_node_coord_field_name();
  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  validate_field_ptr(node_coord_field_ptr_, node_coord_field_name);

  if (generate_hookean_springs_) {
    const std::string element_hookean_spring_constant_field_name =
        mundy::constraints::HookeanSprings::get_element_spring_constant_field_name();
    const std::string element_hookean_spring_rest_length_field_name =
        mundy::constraints::HookeanSprings::get_element_rest_length_field_name();

    element_hookean_spring_constant_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_hookean_spring_constant_field_name);
    element_hookean_spring_rest_length_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_hookean_spring_rest_length_field_name);

    validate_field_ptr(element_hookean_spring_constant_field_ptr_, element_hookean_spring_constant_field_name);
    validate_field_ptr(element_hookean_spring_rest_length_field_ptr_, element_hookean_spring_rest_length_field_name);

    const Teuchos::Array<std::string> hookean_springs_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("hookean_springs_part_names");
    hookean_spring_part_ptrs_ = parts_from_names(*meta_data_ptr_, hookean_springs_part_names);
  }

  if (generate_angular_springs_) {
    const std::string element_angular_spring_constant_field_name =
        mundy::constraints::AngularSprings::get_element_spring_constant_field_name();
    const std::string element_angular_spring_rest_angle_field_name =
        mundy::constraints::AngularSprings::get_element_rest_angle_field_name();

    element_angular_spring_constant_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_angular_spring_constant_field_name);
    element_angular_spring_rest_angle_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_angular_spring_rest_angle_field_name);

    validate_field_ptr(element_angular_spring_constant_field_ptr_, element_angular_spring_constant_field_name);
    validate_field_ptr(element_angular_spring_rest_angle_field_ptr_, element_angular_spring_rest_angle_field_name);

    const Teuchos::Array<std::string> angular_springs_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("angular_springs_part_names");
    angular_spring_part_ptrs_ = parts_from_names(*meta_data_ptr_, angular_springs_part_names);
  }

  if (generate_spheres_at_nodes_) {
    const std::string element_sphere_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();
    element_sphere_radius_field_ptr_ =
        meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_sphere_radius_field_name);
    validate_field_ptr(element_sphere_radius_field_ptr_, element_sphere_radius_field_name);

    const Teuchos::Array<std::string> sphere_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("sphere_part_names");
    sphere_part_ptrs_ = parts_from_names(*meta_data_ptr_, sphere_part_names);
  }

  if (generate_spherocylinder_segments_along_edges_) {
    const std::string element_spherocylinder_segment_radius_field_name =
        mundy::shapes::SpherocylinderSegments::get_element_radius_field_name();
    element_spherocylinder_segment_radius_field_ptr_ = meta_data_ptr_->get_field<double>(
        stk::topology::ELEMENT_RANK, element_spherocylinder_segment_radius_field_name);
    validate_field_ptr(element_spherocylinder_segment_radius_field_ptr_,
                       element_spherocylinder_segment_radius_field_name);

    const Teuchos::Array<std::string> spherocylinder_segment_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("spherocylinder_segment_part_names");
    spherocylinder_segment_part_ptrs_ = parts_from_names(*meta_data_ptr_, spherocylinder_segment_part_names);
  }
}
//}

// \name Setters
//{

void ChainOfSprings::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(ChainOfSprings::get_valid_mutable_params());

  num_nodes_ = valid_mutable_params.get<size_t>("num_nodes");
  num_hookean_springs_ = generate_hookean_springs_ ? num_nodes_ - 1 : 0;
  num_angular_springs_ = generate_angular_springs_ ? num_nodes_ - 2 : 0;
  num_spheres_ = generate_spheres_at_nodes_ ? num_nodes_ : 0;
  num_spherocylinder_segments_ = generate_spherocylinder_segments_along_edges_ ? num_nodes_ - 1 : 0;

  element_id_start_ = valid_mutable_params.get<size_t>("element_id_start");
  node_id_start_ = valid_mutable_params.get<size_t>("node_id_start");
  coordinate_map_ptr_ = valid_mutable_params.get<std::shared_ptr<ArchlengthCoordinateMapping>>("coordinate_mapping");
  hookean_spring_constant_ = valid_mutable_params.get<double>("hookean_spring_constant");
  hookean_spring_rest_length_ = valid_mutable_params.get<double>("hookean_spring_rest_length");
  angular_spring_constant_ = valid_mutable_params.get<double>("angular_spring_constant");
  angular_spring_rest_angle_ = valid_mutable_params.get<double>("angular_spring_rest_angle");
  sphere_radius_ = valid_mutable_params.get<double>("sphere_radius");
  spherocylinder_segment_radius_ = valid_mutable_params.get<double>("spherocylinder_segment_radius");
}
//}

// \name Getters
//{

stk::mesh::EntityId ChainOfSprings::get_node_id(const size_t &sequential_node_index) const {
  return node_id_start_ + sequential_node_index;
}

stk::mesh::EntityId ChainOfSprings::get_hookean_spring_id(const size_t &sequential_hookean_spring_index) const {
  return element_id_start_ + sequential_hookean_spring_index;
}

stk::mesh::EntityId ChainOfSprings::get_angular_spring_id(const size_t &sequential_angular_spring_index) const {
  return element_id_start_ + num_hookean_springs_ + sequential_angular_spring_index;
}

stk::mesh::EntityId ChainOfSprings::get_sphere_id(const size_t &sequential_sphere_index) const {
  return element_id_start_ + num_hookean_springs_ + num_angular_springs_ + sequential_sphere_index;
}

stk::mesh::EntityId ChainOfSprings::get_spherocylinder_segment_id(
    const size_t &sequential_spherocylinder_segment_index) const {
  return element_id_start_ + num_hookean_springs_ + num_angular_springs_ + num_spheres_ +
         sequential_spherocylinder_segment_index;
}

stk::mesh::Entity ChainOfSprings::get_node(const size_t &sequential_node_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::NODE_RANK, get_node_id(sequential_node_index));
}

stk::mesh::Entity ChainOfSprings::get_hookean_spring(const size_t &sequential_hookean_spring_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK,
                                    get_hookean_spring_id(sequential_hookean_spring_index));
}

stk::mesh::Entity ChainOfSprings::get_angular_spring(const size_t &sequential_angular_spring_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK,
                                    get_angular_spring_id(sequential_angular_spring_index));
}

stk::mesh::Entity ChainOfSprings::get_sphere(const size_t &sequential_sphere_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK, get_sphere_id(sequential_sphere_index));
}

stk::mesh::Entity ChainOfSprings::get_spherocylinder_segment(
    const size_t &sequential_spherocylinder_segment_index) const {
  return bulk_data_ptr_->get_entity(stk::topology::ELEMENT_RANK,
                                    get_spherocylinder_segment_id(sequential_spherocylinder_segment_index));
}

stk::mesh::Entity ChainOfSprings::get_valid_node(const size_t &sequential_node_index) const {
  stk::mesh::Entity node = get_node(sequential_node_index);
  MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(node), std::logic_error,
                     fmt::format("The node with id {} is not valid.", get_node_id(sequential_node_index)));
  return node;
}

stk::mesh::Entity ChainOfSprings::get_valid_hookean_spring(const size_t &sequential_hookean_spring_index) const {
  stk::mesh::Entity spring = get_hookean_spring(sequential_hookean_spring_index);
  MUNDY_THROW_ASSERT(
      bulk_data_ptr_->is_valid(spring), std::logic_error,
      fmt::format("The spring with id {} is not valid.", get_hookean_spring_id(sequential_hookean_spring_index)));
  return spring;
}

stk::mesh::Entity ChainOfSprings::get_valid_angular_spring(const size_t &sequential_angular_spring_index) const {
  stk::mesh::Entity spring = get_angular_spring(sequential_angular_spring_index);
  MUNDY_THROW_ASSERT(
      bulk_data_ptr_->is_valid(spring), std::logic_error,
      fmt::format("The spring with id {} is not valid.", get_angular_spring_id(sequential_angular_spring_index)));
  return spring;
}

stk::mesh::Entity ChainOfSprings::get_valid_sphere(const size_t &sequential_sphere_index) const {
  stk::mesh::Entity sphere = get_sphere(sequential_sphere_index);
  MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(sphere), std::logic_error,
                     fmt::format("The sphere with id {} is not valid.", get_sphere_id(sequential_sphere_index)));
  return sphere;
}

stk::mesh::Entity ChainOfSprings::get_valid_spherocylinder_segment(
    const size_t &sequential_spherocylinder_segment_index) const {
  stk::mesh::Entity segment = get_spherocylinder_segment(sequential_spherocylinder_segment_index);
  MUNDY_THROW_ASSERT(bulk_data_ptr_->is_valid(segment), std::logic_error,
                     fmt::format("The segment with id {} is not valid.",
                                 get_spherocylinder_segment_id(sequential_spherocylinder_segment_index)));
  return segment;
}
//}

// \name Actions
//{

void ChainOfSprings::execute() {
  // Create the springs and their connected nodes, distributing the work across the ranks.
  const int rank = bulk_data_ptr_->parallel_rank();
  const size_t nodes_per_rank = num_nodes_ / bulk_data_ptr_->parallel_size();
  const size_t remainder = num_nodes_ % bulk_data_ptr_->parallel_size();
  const size_t start_node_index = rank * nodes_per_rank + Kokkos::min(static_cast<size_t>(rank), remainder);
  const size_t end_node_index = start_node_index + nodes_per_rank + (static_cast<size_t>(rank) < remainder ? 1 : 0);

  // Concatenate the spring part pointers.
  std::vector<stk::mesh::Part *> element_part_ptrs_;
  if (generate_hookean_springs_) {
    element_part_ptrs_.insert(element_part_ptrs_.end(), hookean_spring_part_ptrs_.begin(),
                              hookean_spring_part_ptrs_.end());
  }
  if (generate_angular_springs_) {
    element_part_ptrs_.insert(element_part_ptrs_.end(), angular_spring_part_ptrs_.begin(),
                              angular_spring_part_ptrs_.end());
  }
  if (generate_spheres_at_nodes_) {
    element_part_ptrs_.insert(element_part_ptrs_.end(), sphere_part_ptrs_.begin(), sphere_part_ptrs_.end());
  }
  if (generate_spherocylinder_segments_along_edges_) {
    element_part_ptrs_.insert(element_part_ptrs_.end(), spherocylinder_segment_part_ptrs_.begin(),
                              spherocylinder_segment_part_ptrs_.end());
  }
  MUNDY_THROW_REQUIRE(!element_part_ptrs_.empty(), std::invalid_argument,
                      "ChainOfSprings: No parts were added to the elements.");

  bulk_data_ptr_->modification_begin();
  openrand::Philox rng(1, 0);
  for (size_t i = start_node_index; i < end_node_index + 1; ++i) {
    // Create the node.
    stk::mesh::EntityId our_node_id = get_node_id(i);
    stk::mesh::Entity node = bulk_data_ptr_->declare_node(our_node_id);
    bulk_data_ptr_->change_entity_parts(node, element_part_ptrs_);

    // Set the node's coordinates using the given coordinate map.
    double *const node_coords = stk::mesh::field_data(*node_coord_field_ptr_, node);
    const auto [coord_x, coord_y, coord_z] = coordinate_map_ptr_->get_grid_coordinate({i});
    node_coords[0] = coord_x;
    node_coords[1] = coord_y;
    node_coords[2] = coord_z;
  }

  // Share the nodes with the neighboring ranks.
  // Note, node sharing is symmetric. If we don't own the node that we intend to share, we need to declare it before
  // marking it as shared. If we are rank 0, we share our final node with rank 1 and receive their first node. If we are
  // rank N, we share our first node with rank N - 1 and receive their final node. Otherwise, we share our first and
  // last nodes with the corresponding neighboring ranks and receive their corresponding nodes.
  if (bulk_data_ptr_->parallel_size() > 1) {
    if (rank == 0) {
      // Share the last node with rank 1. Should already be declared.
      stk::mesh::Entity node = get_valid_node(end_node_index - 1);
      bulk_data_ptr_->add_node_sharing(node, rank + 1);

      // Receive the first node from rank 1. Should already be declared.
      stk::mesh::Entity received_node = get_valid_node(end_node_index);
      bulk_data_ptr_->add_node_sharing(received_node, rank + 1);

      // Populate the data for the received nodes
      bulk_data_ptr_->change_entity_parts(received_node, element_part_ptrs_);
      double *const node_coords = stk::mesh::field_data(*node_coord_field_ptr_, received_node);
      const auto [coord_x, coord_y, coord_z] = coordinate_map_ptr_->get_grid_coordinate({end_node_index});
      node_coords[0] = coord_x;
      node_coords[1] = coord_y;
      node_coords[2] = coord_z;
    } else if (rank == static_cast<int>(bulk_data_ptr_->parallel_size() - 1)) {
      // Share the first node with rank N - 1. Should already be declared.
      stk::mesh::Entity node = get_valid_node(start_node_index);
      bulk_data_ptr_->add_node_sharing(node, rank - 1);

      // Receive the last node from rank N - 1. Shouldn't be declared yet.
      stk::mesh::EntityId received_node_id = get_node_id(start_node_index - 1);
      stk::mesh::Entity received_node = bulk_data_ptr_->declare_node(received_node_id);
      bulk_data_ptr_->add_node_sharing(received_node, rank - 1);

      // Populate the data for the received nodes
      bulk_data_ptr_->change_entity_parts(received_node, element_part_ptrs_);
      double *const node_coords = stk::mesh::field_data(*node_coord_field_ptr_, received_node);
      const auto [coord_x, coord_y, coord_z] = coordinate_map_ptr_->get_grid_coordinate({start_node_index - 1});
      node_coords[0] = coord_x;
      node_coords[1] = coord_y;
      node_coords[2] = coord_z;
    } else {
      // Share the first and last nodes with the corresponding neighboring ranks.
      stk::mesh::Entity first_node = get_valid_node(start_node_index);
      stk::mesh::Entity last_node = get_valid_node(end_node_index - 1);
      bulk_data_ptr_->add_node_sharing(first_node, rank - 1);
      bulk_data_ptr_->add_node_sharing(last_node, rank + 1);

      // Receive the corresponding nodes from the neighboring ranks.
      stk::mesh::EntityId received_first_node_id = get_node_id(start_node_index - 1);
      stk::mesh::Entity received_first_node = bulk_data_ptr_->declare_node(received_first_node_id);

      stk::mesh::Entity received_last_node = get_valid_node(end_node_index);
      bulk_data_ptr_->add_node_sharing(received_first_node, rank - 1);
      bulk_data_ptr_->add_node_sharing(received_last_node, rank + 1);

      // Populate the data for the received nodes
      bulk_data_ptr_->change_entity_parts(received_first_node, element_part_ptrs_);
      bulk_data_ptr_->change_entity_parts(received_last_node, element_part_ptrs_);
      double *const first_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, received_first_node);
      const auto [first_coord_x, first_coord_y, first_coord_z] =
          coordinate_map_ptr_->get_grid_coordinate({start_node_index - 1});
      first_node_coords[0] = first_coord_x;
      first_node_coords[1] = first_coord_y;
      first_node_coords[2] = first_coord_z;
      double *const last_node_coords = stk::mesh::field_data(*node_coord_field_ptr_, received_last_node);
      const auto [last_coord_x, last_coord_y, last_coord_z] =
          coordinate_map_ptr_->get_grid_coordinate({end_node_index});
      last_node_coords[0] = last_coord_x;
      last_node_coords[1] = last_coord_y;
      last_node_coords[2] = last_coord_z;
    }
  }

  if (generate_hookean_springs_) {
    // Linear springs connect nodes i and i + 1. We need to start at node 0 and end at node N - 1.
    const size_t start_element_chain_ordinal = start_node_index;
    const size_t end_start_element_chain_ordinal =
        (rank == bulk_data_ptr_->parallel_size() - 1) ? end_node_index - 1 : end_node_index;

    for (size_t i = start_element_chain_ordinal; i < end_start_element_chain_ordinal; ++i) {
      // Create the hookean spring.
      stk::mesh::EntityId spring_id = get_hookean_spring_id(i);
      stk::mesh::Entity spring = bulk_data_ptr_->declare_element(spring_id);
      bulk_data_ptr_->change_entity_parts(spring, hookean_spring_part_ptrs_);

      // Create the node and connect it to the spring.
      // To map our sequential index to the node sequential index, we connect to node i and i + 1.
      // node i --- spring i --- node i + 1
      stk::mesh::Entity node0 = get_valid_node(i);
      stk::mesh::Entity node1 = get_valid_node(i + 1);

      bulk_data_ptr_->declare_relation(spring, node0, 0);
      bulk_data_ptr_->declare_relation(spring, node1, 1);

      // Populate the spring constants and rest lengths. For the time being, we use a single user defined value.
      stk::mesh::field_data(*element_hookean_spring_constant_field_ptr_, spring)[0] = hookean_spring_constant_;
      stk::mesh::field_data(*element_hookean_spring_rest_length_field_ptr_, spring)[0] = hookean_spring_rest_length_;
    }
  }

  if (generate_angular_springs_) {
    // Angular springs connect connect nodes i, i+1, and i+2. We need to start at node i=0 and end at node N - 2.
    const size_t start_element_chain_ordinal = start_node_index;
    const size_t end_start_element_chain_ordinal =
        (rank == bulk_data_ptr_->parallel_size() - 1) ? end_node_index - 2 : end_node_index - 1;

    for (size_t i = start_element_chain_ordinal; i < end_start_element_chain_ordinal; ++i) {
      // Create the angular spring.
      stk::mesh::EntityId spring_id = get_angular_spring_id(i);
      stk::mesh::Entity spring = bulk_data_ptr_->declare_element(spring_id);
      bulk_data_ptr_->change_entity_parts(spring, angular_spring_part_ptrs_);

      // Create the nodes and connect them to the spring.
      // To map our sequential index to the node sequential index, we connect to node i, i + 1, and i + 2.
      // Our center node is node i. Note, the node ordinals for BEAM_3 are
      /* n1      n2
      //   \    /
      //    \  /
      //     n3
      */
      stk::mesh::Entity left_node = get_valid_node(i);
      stk::mesh::Entity center_node = get_valid_node(i + 1);
      stk::mesh::Entity right_node = get_valid_node(i + 2);

      bulk_data_ptr_->declare_relation(spring, left_node, 0);
      bulk_data_ptr_->declare_relation(spring, right_node, 1);
      bulk_data_ptr_->declare_relation(spring, center_node, 2);

      // Populate the spring constants and rest angles. For the time being, we use a single user defined value.
      stk::mesh::field_data(*element_angular_spring_constant_field_ptr_, spring)[0] = angular_spring_constant_;
      stk::mesh::field_data(*element_angular_spring_rest_angle_field_ptr_, spring)[0] = angular_spring_rest_angle_;
    }
  }

  if (generate_spheres_at_nodes_) {
    // Springs connect to node i. We need to loop over all nodes.
    const size_t start_element_chain_ordinal = start_node_index;
    const size_t end_start_element_chain_ordinal = end_node_index;
    for (size_t i = start_element_chain_ordinal; i < end_start_element_chain_ordinal; ++i) {
      // Create the sphere.
      stk::mesh::Entity sphere = bulk_data_ptr_->declare_element(get_sphere_id(i));
      std::cout << "RANK: " << rank << " sphere: " << sphere << std::endl;
      bulk_data_ptr_->change_entity_parts(sphere, sphere_part_ptrs_);

      // Populate the sphere radius. For the time being, we use a single user defined value.
      stk::mesh::field_data(*element_sphere_radius_field_ptr_, sphere)[0] = sphere_radius_;

      // Create the node and connect it to the sphere.
      stk::mesh::Entity node = get_valid_node(i);
      bulk_data_ptr_->declare_relation(sphere, node, 0);
    }
  }

  if (generate_spherocylinder_segments_along_edges_) {
    // Segments connect nodes i and i + 1. We need to start at node 0 and end at node N - 1. Just like the hookean
    // springs.
    const size_t start_element_chain_ordinal = start_node_index;
    const size_t end_start_element_chain_ordinal =
        (rank == bulk_data_ptr_->parallel_size() - 1) ? end_node_index - 1 : end_node_index;

    for (size_t i = start_element_chain_ordinal; i < end_start_element_chain_ordinal; ++i) {
      // Create the spherocylinder segment.
      stk::mesh::Entity segment = bulk_data_ptr_->declare_element(get_spherocylinder_segment_id(i));
      bulk_data_ptr_->change_entity_parts(segment, spherocylinder_segment_part_ptrs_);

      // Populate the segment radius. For the time being, we use a single user defined value.
      stk::mesh::field_data(*element_spherocylinder_segment_radius_field_ptr_, segment)[0] =
          spherocylinder_segment_radius_;

      // Create the nodes and connect them to the segment.
      stk::mesh::Entity node0 = get_valid_node(i);
      stk::mesh::Entity node1 = get_valid_node(i + 1);

      bulk_data_ptr_->declare_relation(segment, node0, 0);
      bulk_data_ptr_->declare_relation(segment, node1, 1);
    }
  }

  bulk_data_ptr_->modification_end();
}
//}

}  // namespace techniques

}  // namespace declare_and_initialize_constraints

}  // namespace constraints

}  // namespace mundy
