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

#ifndef MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCAL_DRAG_KERNELS_SPHERE_HPP_
#define MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCAL_DRAG_KERNELS_SPHERE_HPP_

/// \file Sphere.hpp
/// \brief Declaration of the LocalDrag's Sphere kernel.

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
#include <mundy_mesh/BulkData.hpp>      // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>     // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>   // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>    // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>  // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/PartReqs.hpp>      // for mundy::meta::PartReqs

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

namespace map_rigid_body_force_to_rigid_body_velocity {

namespace techniques {

namespace local_drag {

namespace kernels {

/// \class Sphere
/// \brief Concrete implementation of \c MetaKernel for computing the axis aligned boundary box of spheres.
class Sphere : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Sphere(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string node_torque_field_name = valid_fixed_params.get<std::string>("node_torque_field_name");
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    std::string node_omega_field_name = valid_fixed_params.get<std::string>("node_omega_field_name");
    std::string element_radius_field_name = valid_fixed_params.get<std::string>("element_radius_field_name");
    std::string associated_part_name = valid_fixed_params.get<std::string>("part_name");

    // Create the requirements.
    auto sphere_part_reqs = std::make_shared<mundy::meta::PartReqs>();
    sphere_part_reqs->set_part_name(associated_part_name);
    sphere_part_reqs->set_part_topology(stk::topology::PARTICLE);
    sphere_part_reqs->add_field_reqs<double>(node_force_field_name, stk::topology::NODE_RANK, 3, 1);
    sphere_part_reqs->add_field_reqs<double>(node_torque_field_name, stk::topology::NODE_RANK, 3, 1);
    sphere_part_reqs->add_field_reqs<double>(node_velocity_field_name, stk::topology::NODE_RANK, 3, 1);
    sphere_part_reqs->add_field_reqs<double>(node_omega_field_name, stk::topology::NODE_RANK, 3, 1);
    sphere_part_reqs->add_field_reqs<double>(element_radius_field_name, stk::topology::ELEMENT_RANK, 1, 1);

    auto mesh_reqs = std::make_shared<mundy::meta::MeshReqs>();
    mesh_reqs->add_and_sync_part_reqs(sphere_part_reqs);

    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_force_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_force_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "Sphere: Type error. Given a parameter with name 'element_aabb_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_force_field_name", std::string(default_node_force_field_name_),
                            "Name of the node field containing the force on the sphere's center.");
    }

    if (fixed_params_ptr->isParameter("node_torque_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_torque_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "Sphere: Type error. Given a parameter with name 'node_torque_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_torque_field_name", std::string(default_node_torque_field_name_),
                            "Name of the node field containing the torque on the sphere's center.");
    }

    if (fixed_params_ptr->isParameter("node_velocity_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_velocity_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "Sphere: Type error. Given a parameter with name 'node_velocity_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                            "Name of the node field containing the translational velocity of the sphere's center.");
    }

    if (fixed_params_ptr->isParameter("node_omega_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_omega_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "Sphere: Type error. Given a parameter with name 'node_omega_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_omega_field_name", std::string(default_node_omega_field_name_),
                            "Name of the node field containing the coordinate of the sphere's center.");
    }

    if (fixed_params_ptr->isParameter("element_radius_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("element_radius_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "Sphere: Type error. Given a parameter with name 'element_radius_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_coord_field_name", std::string(default_element_radius_field_name_),
                            "Name of the element field containing the sphere's radius.");
    }

    if (fixed_params_ptr->isParameter("part_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("part_name");
      MUNDY_THROW_REQUIRE(
          valid_type, std::invalid_argument,
          "Sphere: Type error. Given a parameter with name 'part_name' but with a type other than std::string");
    } else {
      fixed_params_ptr->set("part_name", std::string(default_part_name_),
                            "Name of the part associated with this kernel.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isParameter("viscosity")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("viscosity");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'viscosity' but "
                             "with a type other than unsigned double");
      const bool is_viscocity_positive = mutable_params_ptr->get<double>("viscosity") > 0;
      MUNDY_THROW_REQUIRE(is_viscocity_positive, std::invalid_argument,
                         "NodeEuler: Invalid parameter. Given a parameter with name 'viscosity' but "
                             "with a value less than or equal to zero.");
    } else {
      mutable_params_ptr->set("viscosity", default_viscosity_, "The viscosity of the suspending fluid.");
    }
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<Sphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param sphere_element [in] The sphere element acted on by the kernel.
  KOKKOS_INLINE_FUNCTION void execute(const stk::mesh::Entity &sphere_element) const override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_viscosity_ = 1;
  static constexpr std::string_view default_part_name_ = "SPHERES";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_node_torque_field_name_ = "NODE_TORQUE";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_ = "NODE_OMEGA";
  static constexpr std::string_view default_element_radius_field_name_ = "ELEMENT_RADIUS";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "SPHERES";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The numerical timestep size.
  double viscosity_;

  /// \brief Name of the node field containing the force on the sphere's center.
  std::string node_force_field_name_;

  /// \brief Name of the node field containing the torque on the sphere's center.
  std::string node_torque_field_name_;

  /// \brief Name of the node field containing the translational velocity of the sphere's center.
  std::string node_velocity_field_name_;

  /// \brief Name of the node field containing the rotational velocity of the sphere's center.
  std::string node_omega_field_name_;

  /// \brief Name of the element field containing the sphere's radius.
  std::string element_radius_field_name_;

  /// \brief Node field containing the force on the sphere's center.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Node field containing the torque on the sphere's center..
  stk::mesh::Field<double> *node_torque_field_ptr_ = nullptr;

  /// \brief Node field containing the translational velocity of the sphere's center.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the rotational velocity of the sphere's center.
  stk::mesh::Field<double> *node_omega_field_ptr_ = nullptr;

  /// \brief Element field containing the sphere's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // Sphere

}  // namespace kernels

}  // namespace local_drag

}  // namespace techniques

}  // namespace map_rigid_body_force_to_rigid_body_velocity

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy

#endif  // MUNDY_MOTION_COMPUTE_MOBILITY_TECHNIQUES_RIGID_BODY_MOTION_MAP_RIGID_BODY_FORCE_TO_RIGID_BODY_VELOCITY_TECHNIQUES_LOCAL_DRAG_KERNELS_SPHERE_HPP_
