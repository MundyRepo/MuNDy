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

#ifndef MUNDY_LINKERS_EVALUATE_LINKER_POTENTIALS_SPHEROCYLINDERSEGMENTSPHEROCYLINDERSEGMENTFRICTIONALHERTZIANCONTACT_HPP_
#define MUNDY_LINKERS_EVALUATE_LINKER_POTENTIALS_SPHEROCYLINDERSEGMENTSPHEROCYLINDERSEGMENTFRICTIONALHERTZIANCONTACT_HPP_

/// \file SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact.hpp
/// \brief Declaration of the EvaluateLinkerPotentials's
/// SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact kernel.

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
#include <mundy_core/MakeStringArray.hpp>  // for mundy::core::make_string_array
#include <mundy_linkers/neighbor_linkers/SpherocylinderSegmentSpherocylinderSegmentLinkers.hpp>  // for mundy::linkers::neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>     // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>    // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>  // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>                  // for mundy::meta::PartReqs
#include <mundy_shapes/SpherocylinderSegments.hpp>  // for mundy::shapes::SpherocylinderSegments

namespace mundy {

namespace linkers {

namespace evaluate_linker_potentials {

namespace kernels {

/// \class SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact
/// \brief Concrete implementation of \c MetaKernel for computing the Hertzian contact force between two
/// spherocylinder_segments.
///
/// By definition the Hertzian contact force between two spherocylinder_segments can be calculated using the formula:
/// \f[
/// F = \frac{4}{3} E \sqrt{R} \delta^{3/2}
/// \f]
/// where:
/// - \f$F\f$ is the contact force,
/// - \f$E\f$ is the effective modulus of elasticity, calculated as
///   \f$E = \left( \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2} \right)^{-1}\f$,
/// - \f$R\f$ is the effective radius of contact, defined as
///   \f$\frac{1}{R} = \frac{1}{R_1} + \frac{1}{R_2}\f$,
/// - \f$\delta\f$ is the deformation at the contact point,
/// - \f$R_1\f$ and \f$R_2\f$ are the radii of the two spherocylinder_segments,
/// - \f$E_1\f$ and \f$E_2\f$ are the Young's moduli of the materials,
/// - \f$\nu_1\f$ and \f$\nu_2\f$ are the Poisson's ratios of the materials.
///
/// F = \frac{4}{3} E \sqrt{R} \delta^{3/2}
/// where:
/// - F is the contact force,
/// - E is the effective modulus of elasticity, calculated as
///   E = \left( \frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2} \right)^{-1},
/// - R is the effective radius of contact, defined as
///   \frac{1}{R} = \frac{1}{R_1} + \frac{1}{R_2},
/// - \delta is the deformation at the contact point,
/// - R_1 and R_2 are the radii of the two spherocylinder_segments,
/// - E_1 and E_2 are the Young's moduli of the materials,
/// - \nu_1 and \nu_2 are the Poisson's ratios of the materials.
///
/// The formula assumes isotropic and linearly elastic materials and small deformations. There are more complex Hertzian
/// contact models, so make sure this model is appropriate for your use case.
///
/// In terms of fixed parameters, this class requires
/// - "valid_entity_part_names" (Teuchos::Array<std::string>): The list of valid linker entity part names for the
/// kernel.
/// - "valid_spherocylinder_segment_part_names" (Teuchos::Array<std::string>): The list of valid spherocylinder_segment
/// part names for the kernel.
/// - "linker_potential_force_field_name" (std::string): The name of the field in which to write the linker's
/// potential force.
/// - "linker_signed_separation_distance_field_name" (std::string): The name of the field in which to write the signed
/// separation distance.
/// - "element_youngs_modulus_field_name" (std::string): The name of the field in which to read the Young's modulus.
/// - "element_poissons_ratio_field_name" (std::string): The name of the field in which to read the Poisson's ratio.
class SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    valid_fixed_params.validateParametersAndSetDefaults(
        SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact::get_valid_fixed_params());

    // Add the requirements for the linker.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    std::string linker_potential_force_field_name =
        valid_fixed_params.get<std::string>("linker_potential_force_field_name");
    std::string linker_signed_separation_distance_field_name =
        valid_fixed_params.get<std::string>("linker_signed_separation_distance_field_name");
    std::string linker_tangential_displacement_field_name =
        valid_fixed_params.get<std::string>("linker_tangential_displacement_field_name");
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
      part_reqs->add_field_reqs<double>(linker_potential_force_field_name, stk::topology::CONSTRAINT_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(linker_signed_separation_distance_field_name, stk::topology::CONSTRAINT_RANK, 1,
                                        1);
      part_reqs->add_field_reqs<double>(linker_tangential_displacement_field_name, stk::topology::CONSTRAINT_RANK, 3,
                                        1);
      part_reqs->add_field_reqs<double>(linker_contact_normal_field_name, stk::topology::CONSTRAINT_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(linker_contact_points_field_name, stk::topology::CONSTRAINT_RANK, 6, 2);

      if (part_name == neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::get_name()) {
        // Add the requirements directly to spherocylinder_segment spherocylinder_segment linkers agent.
        neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the spherocylinder_segment spherocylinder_segment linkers agent.
        neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::get_mesh_requirements());

    // Add the requirements for the connected spherocylinder_segments.
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    Teuchos::Array<std::string> valid_sy_seg_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_spherocylinder_segment_part_names");
    const int num_sy_seg_parts = static_cast<int>(valid_sy_seg_part_names.size());
    for (int i = 0; i < num_sy_seg_parts; i++) {
      const std::string part_name = valid_sy_seg_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(
          node_velocity_field_name, stk::topology::NODE_RANK, 3,
          2);  // TODO(palmerb4): This is a temporary workaround while we wait for a fix to MeshReqs.

      if (part_name == mundy::shapes::SpherocylinderSegments::get_name()) {
        // Add the requirements directly to spherocylinder_segment spherocylinder_segment linkers agent.
        mundy::shapes::SpherocylinderSegments::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the spherocylinder_segment spherocylinder_segment linkers agent.
        mundy::shapes::SpherocylinderSegments::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(mundy::shapes::SpherocylinderSegments::get_mesh_requirements());

    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    const static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_entity_part_names",
                 mundy::core::make_string_array(
                     neighbor_linkers::SpherocylinderSegmentSpherocylinderSegmentLinkers::get_name()),
                 "List of valid entity part names for the kernel.")
            .set("valid_spherocylinder_segment_part_names",
                 mundy::core::make_string_array(mundy::shapes::SpherocylinderSegments::get_name()),
                 "List of valid spherocylinder_segment part names for the kernel.")
            .set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                 "Name of the field containing the node velocity.")
            .set("linker_potential_force_field_name", std::string(default_linker_potential_force_field_name_),
                 "Name of the constraint-rank field within which the linker's potential force will be written.")
            .set("linker_signed_separation_distance_field_name",
                 std::string(default_linker_signed_separation_distance_field_name_),
                 "Name of the constraint-rank field within which the signed separation distance will be written.")
            .set("linker_tangential_displacement_field_name",
                 std::string(default_linker_tangential_displacement_field_name_),
                 "Name of the constraint-rank field within which the tangential displacement will be written.")
            .set("linker_contact_normal_field_name", std::string(default_linker_contact_normal_field_name_),
                 "Name of the constraint-rank field containing the left contact normal.")
            .set("linker_contact_points_field_name", std::string(default_linker_contact_points_field_name_),
                 "Name of the constraint-rank field containing the contact points (both left and right).");
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
    return std::make_shared<SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact>(bulk_data_ptr,
                                                                                                 fixed_params);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param spherocylinder_segment_spherocylinder_segment_linker_selector [in] The linker selector acted on by this
  /// kernel.
  void execute(const stk::mesh::Selector &sy_seg_sy_seg_linker_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_linker_potential_force_field_name_ = "LINKER_POTENTIAL_FORCE";
  static constexpr std::string_view default_linker_signed_separation_distance_field_name_ =
      "LINKER_SIGNED_SEPARATION_DISTANCE";
  static constexpr std::string_view default_linker_tangential_displacement_field_name_ =
      "LINKER_TANGENTIAL_DISPLACEMENT";
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

  /// \brief The valid spherocylinder_segment parts.
  std::vector<stk::mesh::Part *> valid_sy_seg_parts_;

  /// \brief Node coords field.
  stk::mesh::Field<double> *node_coords_field_ptr_ = nullptr;

  /// \brief Node velocity field
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Element radius field.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;

  /// \brief Linker potential force field.
  stk::mesh::Field<double> *linker_potential_force_field_ptr_ = nullptr;

  /// \brief Linker signed separation distance field.
  stk::mesh::Field<double> *linker_signed_separation_distance_field_ptr_ = nullptr;

  /// \brief Linker tangential displacement field (history field for the current contact duration).
  stk::mesh::Field<double> *linker_tangential_displacement_field_ptr_ = nullptr;

  /// \brief Linker contact normal field.
  stk::mesh::Field<double> *linker_contact_normal_field_ptr_ = nullptr;

  /// \brief Linker contact points field.
  stk::mesh::Field<double> *linker_contact_points_field_ptr_ = nullptr;

  /// \brief The linked entities field pointer.
  LinkedEntitiesFieldType *linked_entities_field_ptr_ = nullptr;
  //@}
};  // SpherocylinderSegmentSpherocylinderSegmentFrictionalHertzianContact

}  // namespace kernels

}  // namespace evaluate_linker_potentials

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_EVALUATE_LINKER_POTENTIALS_SPHEROCYLINDERSEGMENTSPHEROCYLINDERSEGMENTFRICTIONALHERTZIANCONTACT_HPP_
