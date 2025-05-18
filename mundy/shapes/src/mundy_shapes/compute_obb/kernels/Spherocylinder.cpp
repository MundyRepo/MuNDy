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

/// \file Spherocylinder.cpp
/// \brief Definition of the ComputeOBB's Spherocylinder kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>          // for stk::mesh::Field, stl::mesh::field_data
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run

// Mundy libs
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>     // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>    // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_shapes/Spherocylinders.hpp>                     // for mundy::shapes::Spherocylinders
#include <mundy_shapes/compute_obb/kernels/Spherocylinder.hpp>  // for mundy::shapes::compute_obb::kernels::Spherocylinder

namespace mundy {

namespace shapes {

namespace compute_obb {

namespace kernels {

// \name Constructors and destructor
//{

Spherocylinder::Spherocylinder(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "Spherocylinder: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(Spherocylinder::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::Spherocylinders::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::Spherocylinders::get_element_radius_field_name();
  const std::string element_length_field_name = mundy::shapes::Spherocylinders::get_element_length_field_name();
  const std::string element_orientation_field_name =
      mundy::shapes::Spherocylinders::get_element_orientation_field_name();
  const std::string element_obb_field_name = valid_fixed_params.get<std::string>("element_obb_field_name");

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_orientation_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_orientation_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  element_length_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_length_field_name);
  element_obb_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_obb_field_name);

  MUNDY_THROW_REQUIRE(node_coord_field_ptr_ != nullptr, std::invalid_argument,
                      "Spherocylinder: node_coord_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(
      element_orientation_field_ptr_ != nullptr, std::invalid_argument,
      "Spherocylinder: element_orientation_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(element_radius_field_ptr_ != nullptr, std::invalid_argument,
                      "Spherocylinder: element_radius_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(element_length_field_ptr_ != nullptr, std::invalid_argument,
                      "Spherocylinder: element_length_field_ptr cannot be a nullptr. Check that the field exists.");
  MUNDY_THROW_REQUIRE(element_obb_field_ptr_ != nullptr, std::invalid_argument,
                      "Spherocylinder: element_obb_field_ptr cannot be a nullptr. Check that the field exists.");

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(
          part != nullptr, std::invalid_argument,
          std::string("Spherocylinder: Part ") + part_name + " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> Spherocylinder::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void Spherocylinder::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(Spherocylinder::get_valid_mutable_params());

  // Fill the internal members using the given parameter list.
  buffer_distance_ = valid_mutable_params.get<double>("buffer_distance");
}
//}

// \name Actions
//{

void Spherocylinder::execute(const stk::mesh::Selector &spherocylinder_selector) {
  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {node_coord_field_ptr_, element_orientation_field_ptr_,
                                                      element_radius_field_ptr_, element_length_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  stk::mesh::Field<double> &element_orientation_field = *element_orientation_field_ptr_;
  stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  stk::mesh::Field<double> &element_length_field = *element_length_field_ptr_;
  const stk::mesh::Field<double> &element_obb_field = *element_obb_field_ptr_;
  const double buffer_distance = buffer_distance_;

  // At the end of this loop, all locally owned and ghosted entities will be up-to-date.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & spherocylinder_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK, intersection_with_valid_entity_parts,
      [&node_coord_field, &element_orientation_field, &element_radius_field, &element_length_field, &element_obb_field,
       &buffer_distance]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                         const stk::mesh::Entity &spherocylinder_element) {
        // Element data
        const double radius = stk::mesh::field_data(element_radius_field, spherocylinder_element)[0];
        const double length = stk::mesh::field_data(element_length_field, spherocylinder_element)[0];
        const auto element_orientation = mundy::math::get_quaternion_view<double>(
            stk::mesh::field_data(element_orientation_field, spherocylinder_element));

        // Node data
        const stk::mesh::Entity &node = bulk_data.begin_nodes(spherocylinder_element)[0];
        const auto node_coord = mundy::mesh::vector3_field_data(node_coord_field, node);

        // Find the endpoints.
        // Note, the orientation maps the reference configuration to the current configuration and in the reference
        // configuration the spherocylinder is aligned with the x-axis.
        const auto tangent_vector = element_orientation * mundy::math::Vector3d(1.0, 0.0, 0.0);
        const auto left_endpoint = node_coord - 0.5 * tangent_vector * length;
        const auto right_endpoint = node_coord + 0.5 * tangent_vector * length;

        // Populate the OBB.
        // Note, the OBB is just the AABB in the reference configuration rotated by the orientation and then shifted by
        // the center of the spherocylinder.
        double *obb = stk::mesh::field_data(element_obb_field, spherocylinder_element);
        auto bottom_left = mundy::math::get_vector3_view<double>(obb);
        auto top_right = mundy::math::get_vector3_view<double>(obb + 3);

        const mundy::math::Vector3d bottom_left_ref_config(-radius - buffer_distance, -radius - buffer_distance,
                                                           -0.5 * length - buffer_distance);
        const mundy::math::Vector3d top_right_ref_config(radius + buffer_distance, radius + buffer_distance,
                                                         0.5 * length + buffer_distance);
        bottom_left = element_orientation * bottom_left_ref_config + node_coord;
        top_right = element_orientation * top_right_ref_config + node_coord;
      });
}
//}

}  // namespace kernels

}  // namespace compute_obb

}  // namespace shapes

}  // namespace mundy
