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

/// \file AttachSpheresToNodes.cpp
/// \brief Definition of DeclareAndInitShapes's AttachSpheresToNodes technique.

// C++ core libs
#include <iostream>  // for std::cout, std::endl
#include <memory>    // for std::shared_ptr, std::unique_ptr
#include <string>    // for std::string
#include <vector>    // for std::vector

// External libs
#include <openrand/philox.h>

// Trilinos libs
#include <Teuchos_ParameterList.hpp>      // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>        // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/GetEntities.hpp>  // for stk::mesh::count_entities
#include <stk_search/BoundingBox.hpp>     // for stk::search::Box
#include <stk_search/CoarseSearch.hpp>    // for stk::search::coarse_search
#include <stk_search/SearchMethod.hpp>    // for stk::search::KDTREE

// Mundy libs
#include <mundy_core/throw_assert.hpp>                                                     // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                                                         // for mundy::mesh::BulkData
#include <mundy_shapes/Spheres.hpp>                                                        // for mundy::shapes::Spheres
#include <mundy_shapes/declare_and_initialize_shapes/techniques/AttachSpheresToNodes.hpp>  // for mundy::shapes::...::AttachSpheresToNodes

namespace mundy {

namespace shapes {

namespace declare_and_initialize_shapes {

namespace techniques {

// \name Constructors and destructor
//{

AttachSpheresToNodes::AttachSpheresToNodes(mundy::mesh::BulkData *const bulk_data_ptr,
                                           const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                     "AttachSpheresToNodes: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(AttachSpheresToNodes::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  MUNDY_THROW_REQUIRE(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                     "AttachSpheresToNodes: radius_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  const Teuchos::Array<std::string> sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("sphere_part_names");
  auto parts_from_names = [](mundy::mesh::MetaData &meta_data,
                             const Teuchos::Array<std::string> &part_names) -> std::vector<stk::mesh::Part *> {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(
          part != nullptr, std::invalid_argument,
          std::string("AttachSpheresToNodes: Expected a part with name '") + part_name + "' but part does not exist.");
      parts.push_back(part);
    }
    return parts;
  };

  sphere_part_ptrs_ = parts_from_names(*meta_data_ptr_, sphere_part_names);
}
//}

// \name Setters
//{

void AttachSpheresToNodes::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(AttachSpheresToNodes::get_valid_mutable_params());

  sphere_radius_lower_bound_ = valid_mutable_params.get<double>("sphere_radius_lower_bound");
  sphere_radius_upper_bound_ = valid_mutable_params.get<double>("sphere_radius_upper_bound");
}
//}

// \name Actions
//{

void AttachSpheresToNodes::execute(const stk::mesh::Selector &node_selector) {
  // Methodology:
  //  We are given a node selector containing an unknown number of locally owned nodes on each process.
  //  We can count the number of locally owned nodes to determine how many elements to create.
  //  Generating the sphere entities based on these local counts is perfect for STK's generate_new_entities routine.
  //  The only difficulty is how to efficiently assign one of the generated spheres to each of the nodes.
  //
  //  Because we are within a modification cycle, we cannot rely on the bucket structures. We can, instead, employ a
  //  EntityVector of spheres. This way, the index of the entity vector would exactly correspond to the index of the
  //  corresponding sphere in the requests vector.
  bulk_data_ptr_->modification_begin();

  // Get the locally owned nodes.
  stk::mesh::EntityVector locally_owned_nodes;
  stk::mesh::get_selected_entities(node_selector, bulk_data_ptr_->buckets(stk::topology::NODE_RANK),
                                   locally_owned_nodes);
  const size_t num_nodes_local = locally_owned_nodes.size();

  // Generate the spheres.
  std::vector<size_t> requests(meta_data_ptr_->entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_nodes_local;
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data_ptr_->generate_new_entities(requests, requested_entities);

  // Connect the spheres to the nodes.
  for (size_t i = 0; i < num_nodes_local; ++i) {
    stk::mesh::Entity node = locally_owned_nodes[i];
    stk::mesh::Entity sphere = requested_entities[i];
    bulk_data_ptr_->declare_relation(sphere, node, 0);
  }
  bulk_data_ptr_->modification_end();

  // Set the sphere radii.
  // TODO(palmerb4): Need to think about how to consistently handle RNG counters for particles.
  openrand::Philox rng(1, 0);
  for (size_t i = 0; i < num_nodes_local; ++i) {
    stk::mesh::Entity sphere = requested_entities[i];
    double *const sphere_radius = stk::mesh::field_data(*element_radius_field_ptr_, sphere);
    sphere_radius[0] =
        rng.rand<double>() * (sphere_radius_upper_bound_ - sphere_radius_lower_bound_) + sphere_radius_lower_bound_;
  }
}
//}

}  // namespace techniques

}  // namespace declare_and_initialize_shapes

}  // namespace shapes

}  // namespace mundy