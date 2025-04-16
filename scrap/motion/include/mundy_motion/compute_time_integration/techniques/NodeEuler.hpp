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

#ifndef MUNDY_MOTION_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODEEULER_HPP_
#define MUNDY_MOTION_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODEEULER_HPP_

/// \file NodeEuler.hpp
/// \brief Declaration of ComputeTimeIntegration's NodeEuler technique.

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>   // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>    // for stk::mesh::Entity
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector
#include <stk_topology/topology.hpp>   // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for mundy::meta::MetaMethodRegistry
#include <mundy_meta/PartReqs.hpp>                            // for mundy::meta::PartReqs

namespace mundy {

namespace motion {

namespace compute_time_integration {

namespace techniques {

/// \class NodeEuler
/// \brief Method for computing the euler time integration of a node's position and velocity.
class NodeEuler : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  NodeEuler() = delete;

  /// \brief Constructor
  NodeEuler(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
  //@}

  //! \name MetaFactory static interface implementation
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
    // For now, we allow this method to assign these fields to all bodies.
    // TODO(palmerb4): We should allow these fields to differ from multibody type to multibody type.
    std::string node_coord_field_name = valid_fixed_params.get<std::string>("node_coord_field_name");
    std::string node_velocity_field_name = valid_fixed_params.get<std::string>("node_velocity_field_name");
    std::string node_omega_field_name_name = valid_fixed_params.get<std::string>("node_omega_field_name_name");

    auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
    part_reqs->set_part_name("BODY");
    part_reqs->set_part_rank(stk::topology::ELEMENT_RANK);
    part_reqs->add_field_reqs<double>(node_coord_field_name, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(node_velocity_field_name, stk::topology::NODE_RANK, 3, 1);
    part_reqs->add_field_reqs<double>(node_omega_field_name_name, stk::topology::NODE_RANK, 3, 1);

    auto mesh_reqs = std::make_shared<mundy::meta::MeshReqs>();
    mesh_reqs->add_and_sync_part_reqs(part_reqs);

    return mesh_reqs;
  }

  /// \brief Validate the fixed parameters and use defaults for unset parameters.
  static void validate_fixed_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const fixed_params_ptr) {
    if (fixed_params_ptr->isParameter("node_coord_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_coord_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'node_coord_field_name' but "
                             "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_coord_field_name", std::string(default_node_coord_field_name_),
                            "Name of the node field containing the node's spatial coordinate.");
    }

    if (fixed_params_ptr->isParameter("node_velocity_field_name")) {
      const bool valid_type =
          fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_velocity_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'node_velocity_field_name' but "
                             "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_velocity_field_name", std::string(default_node_velocity_field_name_),
                            "Name of the node field containing the node's translational velocity.");
    }

    if (fixed_params_ptr->isParameter("node_omega_field_name")) {
      const bool valid_type = fixed_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<std::string>("node_omega_field_name");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'node_omega_field_name' but "
                         "with a type other than std::string");
    } else {
      fixed_params_ptr->set("node_omega_field_name", std::string(default_node_omega_field_name_),
                            "Name of the node field containing the node's rotational velocity.");
    }
  }

  /// \brief Validate the mutable parameters and use defaults for unset parameters.
  static void validate_mutable_parameters_and_set_defaults(
      [[maybe_unused]] Teuchos::ParameterList *const mutable_params_ptr) {
    if (mutable_params_ptr->isParameter("time_step_size")) {
      const bool valid_type = mutable_params_ptr->INVALID_TEMPLATE_QUALIFIER isType<double>("time_step_size");
      MUNDY_THROW_REQUIRE(valid_type, std::invalid_argument,
                         "NodeEuler: Type error. Given a parameter with name 'time_step_size' but "
                             "with a type other than double");
      const bool is_time_step_size_positive = mutable_params_ptr->get<double>("time_step_size") > 0;
      MUNDY_THROW_REQUIRE(is_time_step_size_positive, std::invalid_argument,
                         "NodeEuler: Invalid parameter. Given a parameter with name 'time_step_size' but "
                             "with a value less than or equal to zero.");
    } else {
      mutable_params_ptr->set("time_step_size", default_time_step_size_, "The numerical timestep size.");
    }
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<mundy::meta::MetaMethodSubsetExecutionInterface<void>> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<NodeEuler>(bulk_data_ptr, fixed_params);
  }

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &input_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_time_step_size_ = 1.0;
  static constexpr std::string_view default_node_coord_field_name_ = "NODE_COORDS";
  static constexpr std::string_view default_node_velocity_field_name_ = "NODE_VELOCITY";
  static constexpr std::string_view default_node_omega_field_name_ = "NODE_OMEGA";
  //@}

  //! \name Internal members
  //@{

  /// \brief The unique string identifier for this class.
  /// By unique, we mean with respect to other methods in our MetaMethodRegistry.
  static constexpr std::string_view registration_id_ = "NODE_EULER";

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief Number of parts that this method acts on.
  size_t num_parts_ = 0;

  /// \brief Vector of pointers to the parts that this class will act upon.
  std::vector<stk::mesh::Part *> part_ptr_vector_;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief Name of the node field containing the node's spatial coordinate.
  std::string node_coord_field_name_;

  /// \brief Name of the node field containing the node's translational velocity.
  std::string node_velocity_field_name_;

  /// \brief Name of the node field containing the node's rotational velocity.
  std::string node_omega_field_name_;

  /// \brief Node field containing the node's spatial coordinate.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Node field containing the node's translational velocity.
  stk::mesh::Field<double> *node_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's rotational velocity.
  stk::mesh::Field<double> *node_omega_field_ptr_ = nullptr;
  //@}
};  // NodeEuler

}  // namespace techniques

}  // namespace compute_time_integration

}  // namespace motion

}  // namespace mundy

#endif  // MUNDY_MOTION_COMPUTE_TIME_INTEGRATION_TECHNIQUES_NODEEULER_HPP_
