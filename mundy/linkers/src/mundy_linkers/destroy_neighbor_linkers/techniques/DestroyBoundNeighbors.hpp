// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
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

#ifndef MUNDY_LINKERS_DESTROY_NEIGHBOR_LINKERS_TECHNIQUES_DESTROYBOUNDNEIGHBORS_HPP_
#define MUNDY_LINKERS_DESTROY_NEIGHBOR_LINKERS_TECHNIQUES_DESTROYBOUNDNEIGHBORS_HPP_

/// \file DestroyBoundNeighbors.hpp
/// \brief Declaration of the DestroyBoundNeighbors class

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
#include <mundy_core/MakeStringArray.hpp>                     // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>                       // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_linkers/NeighborLinkers.hpp>                  // for mundy::linkers::NeighborLinkers
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                            // for mundy::mesh::MetaData
#include <mundy_mesh/utils/DestroyFlaggedEntities.hpp>        // for mundy::mesh::utils::destroy_flagged_entities
#include <mundy_meta/MeshReqs.hpp>                            // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                        // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace linkers {

namespace destroy_neighbor_linkers {

namespace techniques {

/* This class is responsible for destroying neighbor linkers between elements that are bound to one another via a shared
low-rank entity.

For now, we force the rank of the source and target entities connected to the linkers to be ELEMENT_RANK. There are
undoubtedly use cases for allowing the source and target ranks to be arbitrary and different. However, we will not
support that for now.

We loop over each neighbor linker in the given selector, fetch the connected source and target entities, and check if
they share low-rank entities. If they do, we destroy the neighbor linker.
*/
class DestroyBoundNeighbors : public mundy::meta::MetaMethodSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodSubsetExecutionInterface<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  DestroyBoundNeighbors() = delete;

  /// \brief Constructor
  DestroyBoundNeighbors(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
  static std::shared_ptr<mundy::meta::MeshReqs> get_mesh_requirements(const Teuchos::ParameterList &fixed_params) {
    Teuchos::ParameterList valid_fixed_params = fixed_params;
    valid_fixed_params.validateParametersAndSetDefaults(DestroyBoundNeighbors::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();

    // Add the neighbor linkers part requirements.
    Teuchos::Array<std::string> valid_linker_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_entity_part_names");

    const std::string linker_destroy_flag_field_name =
        valid_fixed_params.get<std::string>("linker_destroy_flag_field_name");
    for (int i = 0; i < valid_linker_entity_part_names.size(); i++) {
      const std::string part_name = valid_linker_entity_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name).set_part_rank(stk::topology::CONSTRAINT_RANK);
      part_reqs->add_field_reqs<int>(linker_destroy_flag_field_name, stk::topology::CONSTRAINT_RANK, 1, 1);

      if (part_name == NeighborLinkers::get_name()) {
        // Add the requirements directly to sphere sphere linkers agent.
        NeighborLinkers::add_and_sync_part_reqs(part_reqs);
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        NeighborLinkers::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(NeighborLinkers::get_mesh_requirements());

    // Add our source/target element part requirements.
    Teuchos::Array<std::string> valid_connected_source_and_target_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_connected_source_and_target_part_names");
    for (int i = 0; i < valid_connected_source_and_target_part_names.size(); i++) {
      const std::string part_name = valid_connected_source_and_target_part_names[i];
      auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
      part_reqs->set_part_name(part_name);
      mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs);
    }

    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_entity_part_names",
                 mundy::core::make_string_array(std::string(default_neighbor_linkers_part_name_)),
                 "Name of the linker entity parts potentially acted on by this meta method.")
            .set("valid_connected_source_and_target_part_names",
                 mundy::core::make_string_array(std::string(universal_part_name_)),
                 "Name of the source and target parts that linker may connect to.")
            .set<std::string>("linker_destroy_flag_field_name", std::string(default_linker_destroy_flag_field_name_),
                              "Name of the field used to flag linkers for destruction.");

    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list;
    return default_parameter_list;
  }

  /// \brief destroy a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<DestroyBoundNeighbors>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid entity parts for the method.
  /// By "valid entity parts," we mean the parts whose entities this method can act on.
  std::vector<stk::mesh::Part *> get_valid_entity_parts() const override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  /// \param input_selector The selector that defines the linker entities to act on.
  void execute(const stk::mesh::Selector &input_selector) override;

 private:
  //! \name Default parameters
  //@{

  /// \brief The default universal part name. This is an implementation detail hidden by STK and is subject to change.
  static constexpr std::string_view universal_part_name_ = "{UNIVERSAL}";
  static constexpr std::string_view default_linker_destroy_flag_field_name_ = "LINKER_DESTROY_FLAG";
  static constexpr std::string_view default_neighbor_linkers_part_name_ = "NEIGHBOR_LINKERS";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid source entity parts.
  std::vector<stk::mesh::Part *> valid_source_target_entity_part_ptrs_;

  /// \brief The valid linker entity parts.
  std::vector<stk::mesh::Part *> valid_linker_entity_part_ptrs_;

  /// \brief The linker destroy flag field pointer.
  stk::mesh::Field<int> *linker_destroy_flag_field_ptr_ = nullptr;

  /// \brief The linked entities field pointer.
  LinkedEntitiesFieldType *linked_entities_field_ptr_ = nullptr;
  //@}
};  // DestroyBoundNeighbors

}  // namespace techniques

}  // namespace destroy_neighbor_linkers

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_DESTROY_NEIGHBOR_LINKERS_TECHNIQUES_DESTROYBOUNDNEIGHBORS_HPP_
