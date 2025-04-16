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

/// \file SpherocylinderSegment.cpp
/// \brief Definition of the ComputeSignedSeparationDistanceAndContactNormal's SpherocylinderSegment kernel.

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
#include <mundy_core/throw_assert.hpp>                                                       // for MUNDY_THROW_ASSERT
#include <mundy_linkers/linker_potential_force_reduction/kernels/SpherocylinderSegment.hpp>  // for mundy::linkers::...::kernels::SpherocylinderSegment
#include <mundy_math/Vector3.hpp>                                                            // for mundy::math::Vector3
#include <mundy_mesh/BulkData.hpp>  // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data, mundy::mesh::matrix3_field_data
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace linker_potential_force_reduction {

namespace kernels {

// \name Constructors and destructor
//{

SpherocylinderSegment::SpherocylinderSegment(mundy::mesh::BulkData *const bulk_data_ptr,
                                             const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "SpherocylinderSegment: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(SpherocylinderSegment::get_valid_fixed_params());

  // Get the field pointers.
  const std::string node_coord_field_name = mundy::shapes::SpherocylinderSegments::get_node_coord_field_name();
  const std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
  const std::string linker_contact_points_field_name =
      valid_fixed_params.get<std::string>("linker_contact_points_field_name");
  const std::string linker_potential_force_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_field_name");
  const std::string linked_entities_field_name = NeighborLinkers::get_linked_entities_field_name();

  node_coord_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_coord_field_name);
  node_force_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::NODE_RANK, node_force_field_name);
  linker_contact_points_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_points_field_name);
  linker_potential_force_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_field_name);
  linked_entities_field_ptr_ = meta_data_ptr_->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, linked_entities_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                        std::string("SpherocylinderSegment: Field ") + field_name +
                            " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(linker_contact_points_field_ptr_, linker_contact_points_field_name);
  field_exists(linker_potential_force_field_ptr_, linker_potential_force_field_name);
  field_exists(node_coord_field_ptr_, node_coord_field_name);
  field_exists(node_force_field_ptr_, node_force_field_name);
  field_exists(linked_entities_field_ptr_, linked_entities_field_name);

  // Get the part pointers.
  const std::string name_of_linker_part_to_reduce_over =
      valid_fixed_params.get<std::string>("name_of_linker_part_to_reduce_over");
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(part != nullptr, std::invalid_argument,
                          std::string("SpherocylinderSegment: Part ") + part_name +
                              " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  linkers_part_to_reduce_over_ = meta_data_ptr_->get_part(name_of_linker_part_to_reduce_over);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SpherocylinderSegment::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SpherocylinderSegment::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(SpherocylinderSegment::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SpherocylinderSegment::execute(const stk::mesh::Selector &spherocylinder_segment_selector) {
  // Communicate the linker fields.
  stk::mesh::communicate_field_data(*bulk_data_ptr_,
                                    {linker_contact_points_field_ptr_, linker_potential_force_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &linker_contact_points_field = *linker_contact_points_field_ptr_;
  const stk::mesh::Field<double> &linker_potential_force_field = *linker_potential_force_field_ptr_;
  const stk::mesh::Field<double> &node_coord_field = *node_coord_field_ptr_;
  stk::mesh::Field<double> &node_force_field = *node_force_field_ptr_;
  const LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Part &linkers_part_to_reduce_over = *linkers_part_to_reduce_over_;

  // At the end of this loop, all locally owned and shared spheres will be up-to-date.
  stk::mesh::Selector locally_owned_or_globally_shared_intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & spherocylinder_segment_selector &
      (meta_data_ptr_->locally_owned_part() | meta_data_ptr_->globally_shared_part());
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::ELEMENT_RANK,
      locally_owned_or_globally_shared_intersection_with_valid_entity_parts,
      [&linker_contact_points_field, &linker_potential_force_field, &node_coord_field, &node_force_field,
       &linkers_part_to_reduce_over,
       &linked_entities_field](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &spherocylinder_segment) {
        // Get our nodes and their force

        // The force experienced by each node is determine by the traspose of the map from endpoint velocity to contact
        // point velocity.
        const stk::mesh::Entity &node0 = bulk_data.begin_nodes(spherocylinder_segment)[0];
        const stk::mesh::Entity &node1 = bulk_data.begin_nodes(spherocylinder_segment)[1];

        auto pos0 = mundy::mesh::vector3_field_data(node_coord_field, node0);
        auto pos1 = mundy::mesh::vector3_field_data(node_coord_field, node1);

        auto force0 = mundy::mesh::vector3_field_data(node_force_field, node0);
        auto force1 = mundy::mesh::vector3_field_data(node_force_field, node1);

        // Loop over the connected constraint rank entities
        const unsigned num_constraint_rank_conn = bulk_data.num_connectivity(node0, stk::topology::CONSTRAINT_RANK);
        const stk::mesh::Entity *connected_linkers = bulk_data.begin(node0, stk::topology::CONSTRAINT_RANK);

        for (unsigned i = 0; i < num_constraint_rank_conn; i++) {
          const stk::mesh::Entity &connected_linker = connected_linkers[i];
          MUNDY_THROW_ASSERT(bulk_data.is_valid(connected_linker), std::logic_error,
                             "SpherocylinderSegment: connected_linker is not valid.");

          const bool is_reduction_linker = bulk_data.bucket(connected_linker).member(linkers_part_to_reduce_over);
          if (is_reduction_linker) {
            // The linker force is the force on the left element. The force on the right element is equal and opposite.
            // This is important, as it means we should multiply by -1 if we are the right element.
            const stk::mesh::EntityKey::entity_key_t *key_t_ptr =
                reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
                    stk::mesh::field_data(linked_entities_field, connected_linker));

            const bool is_left_seg = (key_t_ptr[0] == bulk_data.entity_key(spherocylinder_segment));
            const bool is_right_seg = (key_t_ptr[1] == bulk_data.entity_key(spherocylinder_segment));
            const double sign = is_left_seg ? 1.0 : (is_right_seg ? -1.0 : 0.0);
            const auto potential_force =
                sign * mundy::mesh::vector3_field_data(linker_potential_force_field, connected_linker);

            auto contact_point = mundy::math::get_vector3_view<double>(
                stk::mesh::field_data(linker_contact_points_field, connected_linker) + 3 * !is_left_seg);

            // For now, we ignore the contribution to twist torque.
            const auto left_to_cp = contact_point - pos0;
            const auto left_to_right = pos1 - pos0;
            const double length = mundy::math::norm(left_to_right);
            const double inv_length = 1.0 / length;
            const auto tangent = left_to_right * inv_length;

            const auto term1 = mundy::math::dot(tangent, potential_force) * left_to_cp * inv_length;
            const auto term2 = mundy::math::dot(left_to_cp, tangent) *
                               (potential_force + mundy::math::dot(tangent, potential_force) * tangent) * inv_length;
            const auto sum = term2 - term1;
#pragma omp atomic
            force0[0] += potential_force[0] - sum[0];
#pragma omp atomic
            force0[1] += potential_force[1] - sum[1];
#pragma omp atomic
            force0[2] += potential_force[2] - sum[2];
#pragma omp atomic
            force1[0] += sum[0];
#pragma omp atomic
            force1[1] += sum[1];
#pragma omp atomic
            force1[2] += sum[2];
          }
        }
      });
}
//}

}  // namespace kernels

}  // namespace linker_potential_force_reduction

}  // namespace linkers

}  // namespace mundy
