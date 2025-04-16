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

#ifndef MUNDY_SHAPES_COMPUTE_OBB_KERNELS_SPHEROCYLINDER_HPP_
#define MUNDY_SHAPES_COMPUTE_OBB_KERNELS_SPHEROCYLINDER_HPP_

/// \file Spherocylinder.hpp
/// \brief Declaration of the ComputeOBB's Spherocylinder kernel.

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
#include <mundy_meta/PartReqs.hpp>           // for mundy::meta::PartReqs
#include <mundy_shapes/Spherocylinders.hpp>  // for mundy::shapes::Spherocylinders

namespace mundy {

namespace shapes {

namespace compute_obb {

namespace kernels {

/// \class Spherocylinder
/// \brief Concrete implementation of \c MetaKernel for computing the object aligned boundary box of Spherocylinders.
class Spherocylinder : public mundy::meta::MetaKernel<> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaKernel<>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Constructor
  explicit Spherocylinder(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    valid_fixed_params.validateParametersAndSetDefaults(Spherocylinder::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    std::string element_obb_field_name = valid_fixed_params.get<std::string>("element_obb_field_name");

    auto valid_entity_part_names = valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");
    const int num_parts = static_cast<int>(valid_entity_part_names.size());
    for (int i = 0; i < num_parts; i++) {
      const std::string part_name = valid_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      part_reqs->set_part_topology(stk::topology::PARTICLE);
      part_reqs->add_field_reqs<double>(element_obb_field_name, stk::topology::ELEMENT_RANK, 6, 1);

      if (part_name == mundy::shapes::Spherocylinders::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        mundy::shapes::Spherocylinders::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        mundy::shapes::Spherocylinders::add_and_sync_subpart_reqs(part_reqs);
      }
    }

    return mundy::shapes::Spherocylinders::get_mesh_requirements();
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_entity_part_names", mundy::core::make_string_array(default_part_name_),
                 "Name of the parts associated with this kernel.")
            .set("element_obb_field_name", std::string(default_element_obb_field_name_),
                 "Name of the element field within which the output object-aligned boundary "
                 "boxes will be written.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list = Teuchos::ParameterList().set(
        "buffer_distance", default_buffer_distance_, "Buffer distance to be added to the object-aligned boundary box.");
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
    return std::make_shared<Spherocylinder>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Run the kernel's core calculation.
  /// \param Spherocylinder_element [in] The Spherocylinder element acted on by the kernel.
  void execute(const stk::mesh::Selector &spherocylinder_selector) override;
  //@}

 private:
  //! \name Default parameters
  //@{

  static constexpr double default_buffer_distance_ = 0.0;
  static constexpr std::string_view default_part_name_ = "SPHEROCYLINDERS";
  static constexpr std::string_view default_element_obb_field_name_ = "ELEMENT_OBB";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid entity parts.
  std::vector<stk::mesh::Part *> valid_entity_parts_;

  /// \brief Buffer distance to be added to the object-aligned boundary box.
  double buffer_distance_ = default_buffer_distance_;

  /// \brief Node field containing the coordinate of the Spherocylinder's center.
  stk::mesh::Field<double> *node_coord_field_ptr_ = nullptr;

  /// \brief Element field containing the Spherocylinder's radius.
  stk::mesh::Field<double> *element_radius_field_ptr_ = nullptr;

  /// \brief Element field containing the Spherocylinder's length.
  stk::mesh::Field<double> *element_length_field_ptr_ = nullptr;

  /// \brief Element field containing the Spherocylinder's orientation (as a quaternion).
  stk::mesh::Field<double> *element_orientation_field_ptr_ = nullptr;

  /// \brief Element field within which the output object-aligned boundary boxes will be written.
  stk::mesh::Field<double> *element_obb_field_ptr_ = nullptr;
  //@}
};  // Spherocylinder

}  // namespace kernels

}  // namespace compute_obb

}  // namespace shapes

}  // namespace mundy

#endif  // MUNDY_SHAPES_COMPUTE_OBB_KERNELS_SPHEROCYLINDER_HPP_
