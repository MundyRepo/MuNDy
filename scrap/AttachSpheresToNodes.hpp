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

#ifndef MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_ATTACHSPHERESTONODES_HPP_
#define MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_ATTACHSPHERESTONODES_HPP_

/// \file AttachSpheresToNodes.hpp
/// \brief Declaration of the AttachSpheresToNodes class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>        // for Teuchos::ParameterList
#include <stk_mesh/base/Entity.hpp>         // for stk::mesh::Entity
#include <stk_mesh/base/ForEachEntity.hpp>  // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/Part.hpp>           // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>       // for stk::mesh::Selector
#include <stk_topology/topology.hpp>        // for stk::topology

// Mundy libs
#include <mundy_core/StringLiteral.hpp>                       // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>                    // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter
#include <mundy_shapes/Spheres.hpp>  // for mundy::shapes::Spheres
#include <mundy_core/MakeStringArray.hpp>    // for mundy::core::make_string_array

namespace mundy {

namespace shapes {

namespace declare_and_initialize_shapes {

namespace techniques {

/* Our goal is to declare and attach spheres to a given set of nodes. Each locally owned node will receive a sphere.
 */
class AttachSpheresToNodes : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  AttachSpheresToNodes() = delete;

  /// \brief Constructor
  AttachSpheresToNodes(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
      const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(AttachSpheresToNodes::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    Teuchos::Array<std::string> sphere_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("sphere_part_names");

    for (int i = 0; i < sphere_part_names.size(); i++) {
      const std::string part_name = sphere_part_names[i];
      if (part_name == mundy::shapes::Spheres::get_name()) {
        // No specialization is required.
      } else {
        // The specialized part must be a subset of the spheres part.
        auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
        part_reqs->set_part_name(part_name);
        mundy::shapes::Spheres::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spheres::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set(
        "sphere_part_names", mundy::core::make_string_array(mundy::shapes::Spheres::get_name()),
        "The names of the parts to which we will add the generated spheres.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    default_parameter_list.set("sphere_radius_lower_bound", default_sphere_radius_lower_bound_,
                               "The lower bound on the sphere radius.");
    default_parameter_list.set("sphere_radius_upper_bound", default_sphere_radius_upper_bound_,
                               "The upper bound on the sphere radius.");

    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<AttachSpheresToNodes>(bulk_data_ptr, fixed_params);
  }
  //@}


  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override {
    return sphere_part_ptrs_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  void execute(const stk::mesh::Selector &node_selector) override;

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_sphere_radius_lower_bound_ = 0.0;
  static constexpr double default_sphere_radius_upper_bound_ = 1.0;
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The lower bound on the sphere radius.
  double sphere_radius_lower_bound_;

  /// \brief The upper bound on the sphere radius.
  double sphere_radius_upper_bound_;

  /// \brief Sphere element id start.
  size_t sphere_element_id_start_ = 1;

  /// \brief The sphere parts.
  std::vector<stk::mesh::Part *> sphere_part_ptrs_;

  /// \brief The sphere radius field pointer.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;
  //@}
};  // AttachSpheresToNodes

}  // namespace techniques

}  // namespace declare_and_initialize_shapes

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_DECLARE_AND_INITIALIZE_SHAPES_TECHNIQUES_ATTACHSPHERESTONODES_HPP_
