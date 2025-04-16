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

#ifndef MUNDY_CONSTRAINTS_GENERATE_COLLISION_CONSTRAINTS_KERNELS_COLLISIONSPHERESPHERE_HPP_
#define MUNDY_CONSTRAINTS_GENERATE_COLLISION_CONSTRAINTS_KERNELS_COLLISIONSPHERESPHERE_HPP_

/// \file CollisionSphereSphere.hpp
/// \brief Declaration of the GenerateCollisionConstraints' CollisionSphereSphere kernel.

// C++ core libs
#include <memory>   // for std::shared_ptr, std::unique_ptr
#include <string>   // for std::string
#include <utility>  // for std::make_pair
#include <vector>   // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>   // for stk::mesh::Entity
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field, stl::mesh::field_data
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_constraints/GenerateCollisionConstraints.hpp>  // for mundy::constraint::GenerateCollisionConstraints
#include <mundy_mesh/BulkData.hpp>                             // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                             // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>                    // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>                          // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                           // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>                         // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs

namespace mundy {

namespace constraints {

namespace generate_collision_constraints {

namespace kernels {

/// \class CollisionSphereSphere
/// \brief Concrete implementation of a \c MetaKWayKernel for generating a collision constraint between two
/// spheres.
class CollisionSphereSphere : public mundy::meta::MetaKWayKernel<3, void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKWayKernel<3, void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit CollisionSphereSphere(mundy::mesh::BulkData *const bulk_data_ptr,
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
    validate_fixed_parameters_and_set_defaults(&valid_fixed_params);

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    std::string node_coord_field_name = valid_fixed_params.get<std::string>("node_coord_field_name");
    std::string node_normal_field_name = valid_fixed_params.get<std::string>("node_normal_field_name");
    std::string element_radius_field_name = valid_fixed_params.get<std::string>("element_radius_field_name");
    std::string element_signed_separation_dist_field_name =
        valid_fixed_params.get<std::string>("element_signed_separation_dist_field_name");
    auto valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->set_part_topology(stk::topology::PARTICLE);
      part_reqs->add_field_reqs<double>(
          node_coord_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(
          node_normal_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(
          element_radius_field_name, stk::topology::ELEMENT_RANK, 1, 1);
      part_reqs->add_field_reqs<double>(
          element_signed_separation_dist_field_name, stk::topology::ELEMENT_RANK, 1, 1);
      mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs);
    }

    return mesh_reqs_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.

  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr, mundy::meta::ParamConfig<Teuchos::Array<std::string>>{
                              .name = "valid_entity_part_names",
                              .default_value = mundy::core::make_string_array(default_part_name_),
                              .doc_string = "Name of the parts associated with this kernel."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr,
        mundy::meta::ParamConfig<std::string>{
            .name = "node_coord_field_name",
            .default_value = std::string(default_node_coord_field_name_),
            .doc_string = "Name of the node field containing the coordinate of each sphere's center."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr,
        mundy::meta::ParamConfig<std::string>{
            .name = "node_normal_field_name",
            .default_value = std::string(default_node_normal_field_name_),
            .doc_string =
                "Name of the node field containing the normal at the attachment point to each sphere's surface."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr, mundy::meta::ParamConfig<std::string>{
                              .name = "element_radius_field_name",
                              .default_value = std::string(default_element_radius_field_name_),
                              .doc_string = "Name of the element field containing the radius of each sphere."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr,
        mundy::meta::ParamConfig<std::string>{
            .name = "element_signed_separation_dist_field_name",
            .default_value = std::string(default_element_signed_separation_dist_field_name_),
            .doc_string = "Name of the element field containing the signed separation distance between each sphere."});
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaKWayKernel<3, void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<CollisionSphereSphere>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// For example, calculate the force on an entity.
  /// \param entity_array The array of entities to act on. These are
  /// [collision_element, left_sphere_element, right_sphere_element]
  void execute(const std::array<stk::mesh::Entity, 3> &entity_array) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORDS";
  static constexpr std::string_view default_node_normal_field_name_ = "NODE_NORMAL";
  static constexpr std::string_view default_element_radius_field_name_ = "ELEMENT_RADIUS";
  static constexpr std::string_view default_element_signed_separation_dist_field_name_ =
      "ELEMENT_SIGNED_SEPARATION_DIST";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our \c MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "COLLISIONSPHERESPHERE";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Name of the node field containing the coordinate of each Sphere's center and the contact points.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the normal at the point of contact.
  std::string node_normal_field_name_;

  /// \brief Name of the element field containing the radius of the spheres.
  std::string element_radius_field_name_;

  /// \brief Name of the element field containing the signed separation distance between the spheres.
  std::string element_signed_separation_dist_field_name_;

  /// \brief Node field containing the coordinate of each Sphere's center and the contact points.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the normal at the point of contact.
  stk::mesh::Field<double> *node_normal_field_ptr_ = nullptr;

  /// \brief Element field containing the signed separation distance between the spheres.
  stk::mesh::Field<double> *element_signed_separation_dist_field_ptr_ = nullptr;

  /// \brief Element field containing the radius of the spheres.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // CollisionSphereSphere

}  // namespace kernels

}  // namespace generate_collision_constraints

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_GENERATE_COLLISION_CONSTRAINTS_KERNELS_COLLISIONSPHERESPHERE_HPP_
