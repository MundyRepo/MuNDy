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

#ifndef MUNDY_LINKERS_COMPUTE_SIGNED_SEPARATION_DISTANCE_CONTACT_NORMAL_AND_CONTACT_POINTS_SPHERESPHEROCYLINDERLINKER_HPP_
#define MUNDY_LINKERS_COMPUTE_SIGNED_SEPARATION_DISTANCE_CONTACT_NORMAL_AND_CONTACT_POINTS_SPHERESPHEROCYLINDERLINKER_HPP_

/// \file SphereSpherocylinderLinker.hpp
/// \brief Declaration of the ComputeSignedSeparationDistanceAndContactNormal's SphereSpherocylinderLinker kernel.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/MakeStringArray.hpp>                                  // for mundy::core::make_string_array
#include <mundy_linkers/neighbor_linkers/SphereSpherocylinderLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SphereSpherocylinderLinkers
#include <mundy_mesh/BulkData.hpp>                                         // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                                         // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>                                        // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>                                      // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                                       // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>                                     // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>           // for mundy::meta::PartReqs
#include <mundy_shapes/Spheres.hpp>          // for mundy::shapes::Spheres
#include <mundy_shapes/Spherocylinders.hpp>  // for mundy::shapes::Spherocylinders

namespace mundy {

namespace linkers {

namespace compute_signed_separation_distance_contact_normal_and_contact_points {

namespace kernels {

/// \class SphereSpherocylinderLinker
/// \brief Concrete implementation of \c MetaKernel for computing the signed separation distance and contact normal
/// between a sphere and a spherocylinder.
class SphereSpherocylinderLinker : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit SphereSpherocylinderLinker(mundy::mesh::BulkData *const bulk_data_ptr,
                                      const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaKernel interface implementation
  //@{

  /// \brief Get the requirements that this method imposes upon each particle and/or constraint.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  ///
  /// \note This method does not cache its return value, so every time you call this method, a new \c MeshReqs
  /// will be created. You can save the result yourself if you wish to reuse it.
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(SphereSpherocylinderLinker::get_valid_fixed_params());

    // Add the requirements for the linker.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    std::string linker_signed_separation_distance_field_name =
        valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
    std::string linker_contact_normal_field_name =
        valid_fixed_params.get<std::string>("linker_contact_normal_field_name");
    std::string linker_contact_points_field_name =
        valid_fixed_params.get<std::string>("linker_contact_points_field_name");

    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_linker_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_linker_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name).set_part_rank(stk::topology::CONSTRAINT_RANK);
      part_reqs->add_field_reqs<double>(linker_signed_separation_distance_field_name, stk::topology::CONSTRAINT_RANK, 1,
                                        1);
      part_reqs->add_field_reqs<double>(linker_contact_normal_field_name, stk::topology::CONSTRAINT_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(linker_contact_points_field_name, stk::topology::CONSTRAINT_RANK, 6, 1);

      if (part_name == neighbor_linkers::SphereSpherocylinderLinkers::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        neighbor_linkers::SphereSpherocylinderLinkers::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        neighbor_linkers::SphereSpherocylinderLinkers::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(neighbor_linkers::SphereSpherocylinderLinkers::get_mesh_requirements());

    // Add the requirements for the connected spheres.
    // We don't have any requirements for the connected spheres not already specified by the sphere agent
    Teuchos::Array<std::string> valid_sphere_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_sphere_part_names");
    const int num_sphere_parts = static_cast<int>(valid_sphere_part_names.size());
    for (int i = 0; i < num_sphere_parts; i++) {
      const std::string part_name = valid_sphere_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(mundy::shapes::Spheres::get_mesh_requirements());

    // Add the requirements for the connected spherocylinders.
    // We don't have any requirements for the connected spheres not already specified by the spherocylinders agent
    Teuchos::Array<std::string> valid_spherocylinder_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_part_names");
    const int num_spherocylinder_parts = static_cast<int>(valid_spherocylinder_part_names.size());
    for (int i = 0; i < num_spherocylinder_parts; i++) {
      const std::string part_name = valid_spherocylinder_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->set_part_topology(stk::topology::PARTICLE);

      if (part_name == mundy::shapes::Spherocylinders::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spherocylinders::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spherocylinders::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(mundy::shapes::Spherocylinders::get_mesh_requirements());

    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    const static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_entity_part_names",
                 mundy::core::make_string_array(neighbor_linkers::SphereSpherocylinderLinkers::get_name()),
                 "List of valid entity part names for the kernel.")
            .set("valid_sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()),
                 "List of valid sphere part names for the kernel.")
            .set("valid_spherocylinder_part_names",
                 mundy::core::make_string_array(mundy::shapes::Spherocylinders::get_name()),
                 "List of valid spherocylinder part names for the kernel.")
            .set("linker_signed_separation_distance_field_name",
                 std::string(default_linker_signed_separation_distance_field_name_),
                 "Name of the constraint-rank field within which the signed separation distance will be written.")
            .set("linker_contact_normal_field_name", std::string(default_linker_contact_normal_field_name_),
                 "Name of the constraint-rank field within which the contact normal (pointing from left "
                 "entity to right entity) will be written.")
            .set("linker_contact_points_field_name", std::string(default_linker_contact_points_field_name_),
                 "Name of the constraint-rank field within which the contact points will be written. The "
                 "first three entries are the left contact point and the next three the right.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override;

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<SphereSpherocylinderLinker>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_sphere_linker [in] The linker acted on by this kernel.
  void execute(const stk::mesh::Selector &sphere_spherocylinder_linker_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_linker_signed_separation_distance_field_name_ =
      "LINKER_SIGNED_SEPARATION_DISTANCE";
  static constexpr std::string_view default_linker_contact_normal_field_name_ = "LINKER_CONTACT_NORMAL";
  static constexpr std::string_view default_linker_contact_points_field_name_ = "LINKER_CONTACT_POINTS";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The valid sphere parts.
  std::vector<stk::mesh::Part *> valid_sphere_parts_;

  /// \brief The valid spherocylinder parts.
  std::vector<stk::mesh::Part *> valid_spherocylinder_parts_;

  /// \brief Node coordinate field.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Element radius field.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;

  /// \brief Element length field.
  stk::mesh::Field<double> *element_length_field_ptr_ = nullptr;

  /// \brief Element orientation field.
  stk::mesh::Field<double> *element_orientation_field_ptr_ = nullptr;

  /// \brief Linker signed separation distance field.
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr_ = nullptr;

  /// \brief Linker contact normal field.
  stk::mesh::Field<double> *linker_contact_normal_field_ptr_ = nullptr;

  /// \brief Linker contact points field.
  stk::mesh::Field<double> *linker_contact_points_field_ptr_ = nullptr;

  /// \brief The linked entities field pointer.
  LinkedEntitiesFieldType *linked_entities_field_ptr_ = nullptr;
  //@}
};  // SphereSpherocylinderLinker

}  // namespace kernels

}  // namespace compute_signed_separation_distance_contact_normal_and_contact_points

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_COMPUTE_SIGNED_SEPARATION_DISTANCE_CONTACT_NORMAL_AND_CONTACT_POINTS_SPHERESPHEROCYLINDERLINKER_HPP_
