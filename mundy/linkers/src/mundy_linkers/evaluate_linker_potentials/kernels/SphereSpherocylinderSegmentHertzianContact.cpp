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

/// \file SphereSpherocylinderSegmentHertzianContact.cpp
/// \brief Definition of the EvaluateLinkerPotentials' SphereSpherocylinderSegmentHertzianContact kernel.

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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_linkers/evaluate_linker_potentials/kernels/SphereSpherocylinderSegmentHertzianContact.hpp>  // for mundy::linkers::...::kernels::SphereSpherocylinderSegmentHertzianContact
#include <mundy_mesh/BulkData.hpp>    // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>  // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_shapes/Spheres.hpp>   // for mundy::shapes::Spheres
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace evaluate_linker_potentials {

namespace kernels {

// \name Constructors and destructor
//{

SphereSpherocylinderSegmentHertzianContact::SphereSpherocylinderSegmentHertzianContact(
    mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params)
    : bulk_data_ptr_(bulk_data_ptr), meta_data_ptr_(&bulk_data_ptr_->mesh_meta_data()) {
  // The bulk data pointer must not be null.
  MUNDY_THROW_REQUIRE(bulk_data_ptr_ != nullptr, std::invalid_argument,
                      "SphereSpherocylinderSegmentHertzianContact: bulk_data_ptr cannot be a nullptr.");

  // Validate the input params. Use default values for any parameter not given.
  Teuchos::ParameterList valid_fixed_params = fixed_params;
  valid_fixed_params.validateParametersAndSetDefaults(
      SphereSpherocylinderSegmentHertzianContact::get_valid_fixed_params());

  // Get the field pointers.
  const std::string element_radius_field_name = mundy::shapes::Spheres::get_element_radius_field_name();
  const std::string element_youngs_modulus_field_name =
      valid_fixed_params.get<std::string>("element_youngs_modulus_field_name");
  const std::string element_poissons_ratio_field_name =
      valid_fixed_params.get<std::string>("element_poissons_ratio_field_name");
  const std::string linker_potential_force_field_name =
      valid_fixed_params.get<std::string>("linker_potential_force_field_name");
  const std::string linker_signed_separation_distance_field_name =
      valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
  const std::string linker_contact_normal_field_name =
      valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
  const std::string linked_entities_field_name = NeighborLinkers::get_linked_entities_field_name();

  element_radius_field_ptr_ = meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_radius_field_name);
  element_youngs_modulus_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_youngs_modulus_field_name);
  element_poissons_ratio_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::ELEMENT_RANK, element_poissons_ratio_field_name);
  linker_potential_force_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_potential_force_field_name);
  linker_signed_separation_distance_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_signed_separation_distance_field_name);
  linker_contact_normal_field_ptr_ =
      meta_data_ptr_->get_field<double>(stk::topology::CONSTRAINT_RANK, linker_contact_normal_field_name);
  linked_entities_field_ptr_ = meta_data_ptr_->get_field<LinkedEntitiesFieldType::value_type>(
      stk::topology::CONSTRAINT_RANK, linked_entities_field_name);

  auto field_exists = [](const stk::mesh::FieldBase *field_ptr, const std::string &field_name) {
    MUNDY_THROW_REQUIRE(field_ptr != nullptr, std::invalid_argument,
                        std::string("SphereSpherocylinderSegmentHertzianContact: Field ") + field_name +
                            " cannot be a nullptr. Check that the field exists.");
  };  // field_exists

  field_exists(element_radius_field_ptr_, element_radius_field_name);
  field_exists(element_youngs_modulus_field_ptr_, element_youngs_modulus_field_name);
  field_exists(element_poissons_ratio_field_ptr_, element_poissons_ratio_field_name);
  field_exists(linker_potential_force_field_ptr_, linker_potential_force_field_name);
  field_exists(linker_signed_separation_distance_field_ptr_, linker_signed_separation_distance_field_name);
  field_exists(linker_contact_normal_field_ptr_, linker_contact_normal_field_name);
  field_exists(linked_entities_field_ptr_, linked_entities_field_name);

  // Get the part pointers.
  Teuchos::Array<std::string> valid_entity_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
  Teuchos::Array<std::string> valid_sphere_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");
  Teuchos::Array<std::string> valid_spherocylinder_segment_part_names =
      valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_segment_part_names");

  auto parts_from_names = [](mundy::mesh::MetaData &meta_data, const Teuchos::Array<std::string> &part_names) {
    std::vector<stk::mesh::Part *> parts;
    for (const std::string &part_name : part_names) {
      stk::mesh::Part *part = meta_data.get_part(part_name);
      MUNDY_THROW_REQUIRE(part != nullptr, std::invalid_argument,
                          std::string("SphereSpherocylinderSegmentHertzianContact: Part ") + part_name +
                              " cannot be a nullptr. Check that the part exists.");
      parts.push_back(part);
    }
    return parts;
  };  // parts_from_names

  valid_entity_parts_ = parts_from_names(*meta_data_ptr_, valid_entity_part_names);
  valid_sphere_parts_ = parts_from_names(*meta_data_ptr_, valid_sphere_part_names);
  valid_spherocylinder_segment_parts_ = parts_from_names(*meta_data_ptr_, valid_spherocylinder_segment_part_names);
}
//}

// \name MetaKernel interface implementation
//{

std::vector<stk::mesh::Part *> SphereSpherocylinderSegmentHertzianContact::get_valid_entity_parts() const {
  return valid_entity_parts_;
}

void SphereSpherocylinderSegmentHertzianContact::set_mutable_params(const Teuchos::ParameterList &mutable_params) {
  // Validate the input params. Use default values for any parameter not given.
  // We don't have any valid mutable params, so this seems pointless but it's useful in that it will throw if the user
  // gives us parameters. This is useful for catching user errors.
  Teuchos::ParameterList valid_mutable_params = mutable_params;
  valid_mutable_params.validateParametersAndSetDefaults(
      SphereSpherocylinderSegmentHertzianContact::get_valid_mutable_params());
}
//}

// \name Actions
//{

void SphereSpherocylinderSegmentHertzianContact::execute(
    const stk::mesh::Selector &sphere_spherocylinder_segment_linker_selector) {
  // Communicate ghosted fields.
  stk::mesh::communicate_field_data(
      *bulk_data_ptr_, {element_radius_field_ptr_, element_youngs_modulus_field_ptr_, element_poissons_ratio_field_ptr_,
                        linker_signed_separation_distance_field_ptr_, linker_contact_normal_field_ptr_,
                        linker_potential_force_field_ptr_});

  // Get references to internal members so we aren't passing around *this
  const stk::mesh::Field<double> &element_radius_field = *element_radius_field_ptr_;
  const stk::mesh::Field<double> &element_youngs_modulus_field = *element_youngs_modulus_field_ptr_;
  const stk::mesh::Field<double> &element_poissons_ratio_field = *element_poissons_ratio_field_ptr_;
  const stk::mesh::Field<double> &linker_signed_separation_distance_field =
      *linker_signed_separation_distance_field_ptr_;
  const stk::mesh::Field<double> &linker_contact_normal_field = *linker_contact_normal_field_ptr_;
  const LinkedEntitiesFieldType &linked_entities_field = *linked_entities_field_ptr_;
  stk::mesh::Field<double> &linker_potential_force_field = *linker_potential_force_field_ptr_;

  // At the end of this loop, all locally owned and ghosted linkers will be up-to-date.
  stk::mesh::Selector intersection_with_valid_entity_parts =
      stk::mesh::selectUnion(valid_entity_parts_) & sphere_spherocylinder_segment_linker_selector;
  mundy::mesh::for_each_entity_run(
      *bulk_data_ptr_, stk::topology::CONSTRAINT_RANK, intersection_with_valid_entity_parts,
      [&element_radius_field, &element_youngs_modulus_field, &element_poissons_ratio_field,
       &linker_potential_force_field, &linker_signed_separation_distance_field, &linker_contact_normal_field,
       &linked_entities_field]([[maybe_unused]] const stk::mesh::BulkData &bulk_data,
                               const stk::mesh::Entity &sphere_spherocylinder_segment_linker) {
        // Use references to avoid copying entities
        const stk::mesh::EntityKey::entity_key_t *key_t_ptr = reinterpret_cast<stk::mesh::EntityKey::entity_key_t *>(
            stk::mesh::field_data(linked_entities_field, sphere_spherocylinder_segment_linker));
        const stk::mesh::Entity &sphere_element = bulk_data.get_entity(key_t_ptr[0]);
        const stk::mesh::Entity &spherocylinder_segment_element = bulk_data.get_entity(key_t_ptr[1]);

        MUNDY_THROW_ASSERT(bulk_data.is_valid(sphere_element), std::invalid_argument,
                           "SphereSpherocylinderSegmentHertzianContact: sphere_element entity is not valid.");
        MUNDY_THROW_ASSERT(
            bulk_data.is_valid(spherocylinder_segment_element), std::invalid_argument,
            "SphereSpherocylinderSegmentHertzianContact: spherocylinder_segment_element entity is not valid.");

        const double sphere_radius = stk::mesh::field_data(element_radius_field, sphere_element)[0];
        const double spherocylinder_segment_radius =
            stk::mesh::field_data(element_radius_field, spherocylinder_segment_element)[0];
        const double sphere_youngs_modulus = stk::mesh::field_data(element_youngs_modulus_field, sphere_element)[0];
        const double spherocylinder_segment_youngs_modulus =
            stk::mesh::field_data(element_youngs_modulus_field, spherocylinder_segment_element)[0];
        const double sphere_poissons_ratio = stk::mesh::field_data(element_poissons_ratio_field, sphere_element)[0];
        const double spherocylinder_segment_poissons_ratio =
            stk::mesh::field_data(element_poissons_ratio_field, spherocylinder_segment_element)[0];
        const double linker_signed_separation_distance =
            stk::mesh::field_data(linker_signed_separation_distance_field, sphere_spherocylinder_segment_linker)[0];
        const auto left_contact_normal =
            mundy::mesh::vector3_field_data(linker_contact_normal_field, sphere_spherocylinder_segment_linker);

        const double effective_radius =
            (sphere_radius * spherocylinder_segment_radius) / (sphere_radius + spherocylinder_segment_radius);
        const double effective_youngs_modulus =
            (sphere_youngs_modulus * spherocylinder_segment_youngs_modulus) /
            (spherocylinder_segment_youngs_modulus -
             spherocylinder_segment_youngs_modulus * sphere_poissons_ratio * sphere_poissons_ratio +
             sphere_youngs_modulus -
             sphere_youngs_modulus * spherocylinder_segment_poissons_ratio * spherocylinder_segment_poissons_ratio);

        // Compute the force for overlapping particles
        // Note, signed separation distance is negative when particles overlap, so delta = -signed_separation_distance.
        const bool do_particles_overlap = linker_signed_separation_distance < 0;
        const double normal_force_magnitude =
            do_particles_overlap ? (4.0 / 3.0) * effective_youngs_modulus * std::sqrt(effective_radius) *
                                       std::pow(-linker_signed_separation_distance, 1.5)
                                 : 0.0;

        // Save the contact force (Forces are equal and opposite, so we only save the left force)
        mundy::mesh::vector3_field_data(linker_potential_force_field, sphere_spherocylinder_segment_linker) =
            -left_contact_normal * normal_force_magnitude;
      });
}

//}

}  // namespace kernels

}  // namespace evaluate_linker_potentials

}  // namespace linkers

}  // namespace mundy
