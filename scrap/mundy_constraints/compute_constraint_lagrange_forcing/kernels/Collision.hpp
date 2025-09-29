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

#ifndef MUNDY_CONSTRAINTS_COMPUTE_CONSTRAINT_LAGRANGE_FORCING_KERNELS_COLLISION_HPP_
#define MUNDY_CONSTRAINTS_COMPUTE_CONSTRAINT_LAGRANGE_FORCING_KERNELS_COLLISION_HPP_

/// \file Collision.hpp
/// \brief Declaration of the ComputeConstraintLagrangeForcing's Collision kernel.

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
#include <mundy_mesh/BulkData.hpp>           // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>           // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>  // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>        // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>         // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>       // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>  // for mundy::meta::PartReqs

namespace mundy {

namespace constraints {

namespace compute_constraint_forcing {

namespace kernels {

/// \class Collision
/// \brief Concrete implementation of \c MetaKernel for computing the force exerted by a collision constraint onto its nodes given the constraint's Lagrange multiplier.
class Collision : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Collision(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    std::string node_normal_field_name = valid_fixed_params.get<std::string>("node_normal_field_name");
    std::string node_force_field_name = valid_fixed_params.get<std::string>("node_force_field_name");
    std::string element_lagrange_multiplier_field_name =
        valid_fixed_params.get<std::string>("element_lagrange_multiplier_field_name");
    auto valid_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->set_part_topology(stk::topology::BEAM_2);
      part_reqs->add_field_reqs<double>(
          node_normal_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(
          node_force_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<double>(
          element_lagrange_multiplier_field_name, stk::topology::ELEMENT_RANK, 1, 1);
      mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs);
    }
    return mesh_reqs_ptr;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr,
        mundy::meta::ParamConfig<std::string>{.name = "node_normal_field_name",
                                              .default_value = std::string(default_node_normal_field_name_),
                                              .doc_string = "Name of the node field containing the node's normal."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr, mundy::meta::ParamConfig<std::string>{
                              .name = "node_force_field_name",
                              .default_value = std::string(default_node_force_field_name_),
                              .doc_string = "Name of the node field containing force on the constraint's endpoints."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr,
        mundy::meta::ParamConfig<std::string>{
            .name = "element_lagrange_multiplier_field_name",
            .default_value = std::string(default_element_lagrange_multiplier_field_name_),
            .doc_string = "Name of the element field containing the constraint's Lagrange multiplier."});

    mundy::meta::check_parameter_and_set_default(
        fixed_params_ptr, mundy::meta::ParamConfig<Teuchos::Array<std::string>>{
                              .name = "valid_entity_part_names",
                              .default_value = mundy::core::make_string_array(default_part_name_),
                              .doc_string = "Name of the parts associated with this kernel."});
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<Collision>(bulk_data_ptr, fixed_params);
  }

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the kernel.
  /// By "valid entity parts," we mean the parts whose entities the kernel can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param collision_node [in] A node connected to a collision element.
  KOKKOS_INLINE_FUNCTION void execute(const stk::mesh::Entity &collision_node) const override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr std::string_view default_part_name_ = "COLLISION";
  static constexpr std::string_view default_node_normal_field_name_ = "NODE_NORMAL";
  static constexpr std::string_view default_node_force_field_name_ = "NODE_FORCE";
  static constexpr std::string_view default_element_lagrange_multiplier_field_name_ = "ELEMENT_LAGRANGE_MULTIPLIER";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other kernels in our MetaKernelRegistry.
  static constexpr std::string_view registration_id_ = "COLLISION";

  /// \brief The name of the part associated with this kernel.
  std::string associated_part_name_;

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Mesh ordinal for the part containing all collision constraints.
  size_t collision_part_ordinal_;

  /// \brief Minimum allowable signed separation distance between colliding bodies.
  double minimum_allowable_separation_;

  /// \brief Name of the node field containing the normal of each node.
  std::string node_normal_field_name_;

  /// \brief Name of the node field containing the force on each node.
  std::string node_force_field_name_;

  /// \brief Name of the element field containing the constraint's Lagrange multiplier.
  std::string element_lagrange_multiplier_field_name_;

  /// \brief Node field containing the surface normal at the attachment points.
  stk::mesh::Field<double> *node_normal_field_ptr_ = nullptr;

  /// \brief Node field containing the force at the attachment points.
  stk::mesh::Field<double> *node_force_field_ptr_ = nullptr;

  /// \brief Element field containing the lagrange multiplier associated with the collision constraint.
  stk::mesh::Field<double> *element_lagrange_multiplier_field_ptr_ = nullptr;
  //@}
};  // Collision

}  // namespace kernels

}  // namespace compute_constraint_forcing

}  // namespace constraints

}  // namespace mundy

#endif  // MUNDY_CONSTRAINTS_COMPUTE_CONSTRAINT_LAGRANGE_FORCING_KERNELS_COLLISION_HPP_
