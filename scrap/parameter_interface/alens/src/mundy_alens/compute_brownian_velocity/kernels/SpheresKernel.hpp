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

#ifndef MUNDY_ALENS_COMPUTE_BROWNIAN_VELOCITY_KERNELS_SPHERESKERNEL_HPP_
#define MUNDY_ALENS_COMPUTE_BROWNIAN_VELOCITY_KERNELS_SPHERESKERNEL_HPP_

/// \file SpheresKernel.hpp
/// \brief Declaration of the ComputeBrownianVelocity's Spheres kernel.

// External libs
#include <openrand/philox.h>

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
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>        // for mundy::meta::FieldReqs
#include <mundy_meta/MetaFactory.hpp>      // for mundy::meta::MetaKernelFactory
#include <mundy_meta/MetaKernel.hpp>       // for mundy::meta::MetaKernel
#include <mundy_meta/MetaRegistry.hpp>     // for mundy::meta::MetaKernelRegistry
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_meta/PartReqs.hpp>   // for mundy::meta::PartReqs
#include <mundy_shapes/Spheres.hpp>  // for mundy::shapes::Spheres

namespace mundy {

namespace alens {

namespace compute_brownian_velocity {

namespace kernels {

/// \class SpheresKernel
/// \brief Concrete implementation of \c MetaKernel for computing the brownian velocity of spheres.
class SpheresKernel : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit SpheresKernel(mundy::mesh::BulkData *const bulk_data_ptr,
                         const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList());
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
      [[maybe_unused]] const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(SpheresKernel::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string node_brownian_velocity_field_name =
        valid_fixed_params.get<std::string>("node_brownian_velocity_field_name");
    std::string node_rng_counter_field_name = valid_fixed_params.get<std::string>("node_rng_counter_field_name");
    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->add_field_reqs<double>(node_brownian_velocity_field_name, stk::topology::NODE_RANK, 3, 1);
      part_reqs->add_field_reqs<unsigned>(node_rng_counter_field_name, stk::topology::NODE_RANK, 1, 1);

      if (part_name == mundy::shapes::Spheres::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("valid_entity_part_names",
                               mundy::core::make_string_array(mundy::shapes::Spheres::get_name()),
                               "Name of the parts associated with this kernel.");
    default_parameter_list.set("node_brownian_velocity_field_name",
                               std::string(default_node_brownian_velocity_field_name_),
                               "Name of the node velocity field to sum the node's translational velocity into.");
    default_parameter_list.set("node_rng_counter_field_name", std::string(default_node_rng_counter_field_name_),
                               "Name of the node field counter used to generate random streams.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("time_step_size", default_time_step_size_, "The timestep size.");
    default_parameter_list.set("diffusion_coeff", default_diffusion_coeff_, "The diffusion coefficient.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(
      mundy::mesh::BulkData *const bulk_data_ptr,
      const Teuchos::ParameterList &fixed_params = Teuchos::ParameterList()) {
    return std::make_shared<SpheresKernel>(bulk_data_ptr, fixed_params);
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
  /// \param sphere_node [in] The sphere's node acted on by the kernel.
  void execute(const stk::mesh::Selector &sphere_selector);
  //@}

 private:
  //! \name Default parameters
  //@{

  static inline double default_time_step_size_ = 0.0;
  static inline double default_diffusion_coeff_ = 0.0;
  static constexpr std::string_view default_node_brownian_velocity_field_name_ = "NODE_BROWNIAN_VELOCITY";
  static constexpr std::string_view default_node_rng_counter_field_name_ = "NODE_RNG_COUNTER";
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

  /// \brief The valid entity parts for the kernel.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief The numerical timestep size.
  double time_step_size_;

  /// \brief The diffusion coefficient.
  double diffusion_coeff_;

  /// \brief Node field to sum the node's translational velocity into.
  stk::mesh::Field<double> *node_brownian_velocity_field_ptr_ = nullptr;

  /// \brief Node field containing the node's random number generator counter.
  stk::mesh::Field<unsigned> *node_rng_counter_field_ptr_ = nullptr;
  //@}
};  // SpheresKernel

}  // namespace kernels

}  // namespace compute_brownian_velocity

}  // namespace alens

}  // namespace mundy

#endif  // MUNDY_ALENS_COMPUTE_BROWNIAN_VELOCITY_KERNELS_SPHERESKERNEL_HPP_
