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

/// \file SphereSphereLinker.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SphereSphereLinker kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/compute_signed_separation_distance_contact_normal_and_contact_points/kernels/SphereSphereLinker.hpp>  // for mundy::linkers::...::kernels::SphereSphereLinker
#include <mundy_math/Vector3.hpp>   // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_mesh/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <mundy_shapes/Spheres.hpp>      // for mundy::shapes::Spheres

namespace mundy {

namespace linkers {

namespace compute_signed_separation_distance_contact_normal_and_contact_points {

namespace kernels {

// \name Constructors and destructor
//{

SphereSphereLinker::SphereSphereLinker(mundy::mesh::BulkData *const bulk_data_ptr,
                                       const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "SphereSphereLinker: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SphereSphereLinker::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::Spheres::get_node_coord_field_name();
  const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
  const std::string linker_contact_points_field_name =
      valid_fixed_params.get<std::string>("linker_contact_points_field_name");
  const std::string linked_entities_field_name = NeighborLinkers::get_linked_entities_field_name();

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);
  linker_contact_points_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_points_field_name);
  linked_entities_field_ptr_ = meta_data_ptr_->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, linked_entities_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_REQUIRE(
        field_ptr != nullptr, std::invalid_argument,
        std::string("SphereSphereLinker: Field ") + field_name + " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(node_coord_field_ptr_, node_coord_field_name);
  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);
  field_exists(linker_contact_points_field_ptr_, linker_contact_points_field_name);
  field_exists(linked_entities_field_ptr_, linked_entities_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(
          part != nullptr, std::invalid_argument,
          std::string("SphereSphereLinker: Part ") + part_name + " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sphere_parts_ = parts_from_names(*meta_data_ptr_, valid_sphere_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SphereSphereLinker::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SphereSphereLinker::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SphereSphereLinker::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SphereSphereLinker::execute(const stk::mesh::Selector &sphere_sphere_linker_selector) {
  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(*bulk_data_ptr_, {node_coord_field_ptr_, element_radius_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  const LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  stk::mesh::Field<double> &linker_contact_points_field = *linker_contact_points_field_ptr_;
  stk::mesh::Field<double> &linker_signed_separation_distance_field = *linker_signed_separation_distance_field_ptr_;

  // At the end of this loop, all locally owned and ghosted linkers will be up-to-date.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & sphere_sphere_linker_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, intersection_with_valid_entity_parts,
      [&node_coord_field, &element_radius_field, &linker_contact_normal_field, &linker_contact_points_field,
       &linker_signed_separation_distance_field,
       &linked_entities_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &sphere_sphere_linker) {
        // Use references to avoid copying entities
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(linked_entities_field, sphere_sphere_linker));
        const stk::mesh::Entity &left_sphere_element = bulk_data.get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &right_sphere_element = bulk_data.get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data.is_valid(left_sphere_element), std::invalid_argument,
                           "SphereSphereLinker: left_sphere_element entity is not valid.");
        MUNDY_THROW_ASSERT(bulk_data.is_valid(right_sphere_element), std::invalid_argument,
                           "SphereSphereLinker: right_sphere_element entity is not valid.");

        const stk::mesh::Entity &left_sphere_node = bulk_data.begin_nodes(left_sphere_element)[0];
        const stk::mesh::Entity &right_sphere_node = bulk_data.begin_nodes(right_sphere_element)[0];

        // Get the sphere data
        const auto left_coords = mundy::mesh::vector3_field_data(node_coord_field, left_sphere_node);
        const auto right_coords = mundy::mesh::vector3_field_data(node_coord_field, right_sphere_node);
        const double left_radius = stk::mesh::field_data(element_radius_field, left_sphere_element)[0];
        const double right_radius = stk::mesh::field_data(element_radius_field, right_sphere_element)[0];

        auto contact_normal = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_normal_field, sphere_sphere_linker));
        auto left_contact_point = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_points_field, sphere_sphere_linker));
        auto right_contact_point = mundy::math::get_vector3_view<double>(
            stk::mesh::field_data(linker_contact_points_field, sphere_sphere_linker) + 3);
        double *signed_separation_distance =
            stk::mesh::field_data(linker_signed_separation_distance_field, sphere_sphere_linker);

        // Compute the separation distance and contact normal
        const auto left_to_right_vector = right_coords - left_coords;
        const double distance = mundy::math::norm(left_to_right_vector);
        const double radius_sum = left_radius + right_radius;
        const double separation_distance = distance - radius_sum;
        const double inv_distance = 1.0 / distance;

        // Set the separation distance and contact normal
        // Notice that the contact normal points from the left sphere to the right sphere
        // It is the normal to the left sphere and negative the normal of the right sphere.
        signed_separation_distance[0] = separation_distance;
        contact_normal = left_to_right_vector * inv_distance;
        // left_contact_point = left_coords + left_radius * contact_normal;
        // right_contact_point = right_coords - right_radius * contact_normal;
        left_contact_point = left_coords;
        right_contact_point = right_coords;
      });
}
//}

}  // namespace kernels

}  // namespace compute_signed_separation_distance_contact_normal_and_contact_points

}  // namespace linkers

}  // namespace mundy
