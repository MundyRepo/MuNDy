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

#ifndef MUNDY_LINKERS_GENERATE_NEIGHBOR_LINKERS_TECHNIQUES_STKSEARCH_HPP_
#define MUNDY_LINKERS_GENERATE_NEIGHBOR_LINKERS_TECHNIQUES_STKSEARCH_HPP_

/// \file STKSearch.hpp
/// \brief Declaration of the STKSearch class

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
#include <mundy_core/MakeStringArray.hpp>                             // for mundy::core::make_string_array
#include <mundy_core/StringLiteral.hpp>                               // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                                // for MUNDY_THROW_ASSERT
#include <mundy_linkers/NeighborLinkers.hpp>                          // for mundy::linkers::NeighborLinkers
#include <mundy_mesh/BulkData.hpp>                                    // for mundy::mesh::BulkData
#include <mundy_mesh/MetaData.hpp>                                    // for mundy::mesh::MetaData
#include <mundy_meta/MeshReqs.hpp>                                    // for mundy::meta::MeshReqs
#include <mundy_meta/MetaFactory.hpp>                                 // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                                  // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodPairwiseSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodPairwiseSubsetExecutionInterface
#include <mundy_meta/MetaRegistry.hpp>                                // for MUNDY_REGISTER_METACLASS
#include <mundy_meta/ParameterValidationHelpers.hpp>  // for mundy::meta::check_parameter_and_set_default and mundy::meta::check_required_parameter

namespace mundy {

namespace linkers {

namespace generate_neighbor_linkers {

namespace techniques {

/* This class is responsible for generating neighbor linkers between elements of the source whose AABBs intersect those
of elements in the target using the STK's internal stk_search.

For now, we force the rank of the source and target entities to be ELEMENT_RANK. There are undoubtedly use
cases for allowing the source and target ranks to be arbitrary and different. However, we will not support that for now.

Performing the neighbor linker generation with stk_search required a couple steps:
  First, we copy the AABBs, owning procs, and entity keys into a separate contiguous vector for the sources and targets.
  Second, we perform the coarse search using stk_search to get a collection of potential source-target pairs.
  Third, filter out or ignore self-intersections and non-locally owned sources. The latter is necessary to prevent the
    generation of duplicate linkers.
  Fourth, loop over each neighbor pair and add a linker between the source and target.

Our only fixed parameter is the element_aabb_field_name (which should direct us to an already computed aabb field for
each element in the source and target selectors).

The choice of having the user compute the aabb field gives them the liberty to apply buffer/skin distances to the aabbs.

Users will often want to generate specialized neighbor linkers for a specific source-target part pair, such as a
SphereRodNeighborLinker meant to connect spheres and rods. If this is the case, they should set the neighbor linker
part name to the specialization's part name. When doing so, we will not create duplicate linkers between source and
target pairs. Instead, if a linker already exists between a pair, it will be added to the specified neighbor linker
part.

We emphasize, we will NOT allow duplicate linkers between pairs. If, for example, someone wants linkers for a short
and long range potential. They would need to create two different linker parts and call GenerateNeighborLinkers twice.
Consider the following, three level deep hierarchy:

                      NeighborLinkers
                            |
                    SphereSphereLinkers
              |                             |
ShortRangeSphereSphereLinkers  LongRangeSphereSphereLinkers

If someone calls GenerateNeighborLinkers once for ShortRangeSphereSphereLinkers and again for
LongRangeSphereSphereLinkers, a pair of particles that falls within both the short and long ranged potential will have
their linker reside within both the ShortRangeSphereSphereLinkers and LongRangeSphereSphereLinkers parts. Care should
be taken to ensure that linker technique are compatable with this behavior. For example, if a method that acts on
ShortRangeSphereSphereLinkers and another that acts on LongRangeSphereSphereLinkers both write to a force field, then
the correct behavior is likely to sum into the existing field rather than overwrite it.

For each entity of the source part, loop over their connected constraint-rank entities. If they are connected to a
linker, then check if the second entity of that linker is the target entity. If it is, we'll add that linker to the
specified neighbor linker part instead of creating a new linker. For each pair, store a bool for if the pair gets a
linker or not. The sum of this vector of bools is the number of linkers to be created. Then, in the following loop, we
would only generate a linker if the bool for the pair was true. Notice that this requires an up-then-down connection
tree traversal. Typically this is invalid, but we already ghost our neighbors, so it is valid.

Add a flag that deletes linkers that no longer link neighbors with intersection AABBs... No. Bad idea. Separation of
concerns is important. We should have a separate class that does this.
*/
class STKSearch : public mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void> {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = mundy::meta::MetaMethodPairwiseSubsetExecutionInterface<void>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  STKSearch() = delete;

  /// \brief Constructor
  STKSearch(mundy::mesh::BulkData *const bulk_data_ptr, const Teuchos::ParameterList &fixed_params);
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
    valid_fixed_params.validateParametersAndSetDefaults(STKSearch::get_valid_fixed_params());

    // Fill the requirements using the given parameter list.
    auto mesh_reqs_ptr = std::make_shared<mundy::meta::MeshReqs>();
    std::string element_aabb_field_name = valid_fixed_params.get<std::string>("element_aabb_field_name");

    Teuchos::Array<std::string> valid_source_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_source_entity_part_names");
    Teuchos::Array<std::string> valid_target_entity_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("valid_target_entity_part_names");

    auto fetch_and_add_part_reqs =
        [&mesh_reqs_ptr, &element_aabb_field_name](const Teuchos::Array<std::string> &valid_entity_part_names) {
          const int num_parts = static_cast<int>(valid_entity_part_names.size());
          for (int i = 0; i < num_parts; i++) {
            const std::string part_name = valid_entity_part_names[i];
            auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
            part_reqs->set_part_name(part_name);
            part_reqs->add_field_reqs<double>(element_aabb_field_name, stk::topology::ELEMENT_RANK, 6, 1);
            mesh_reqs_ptr->add_and_sync_part_reqs(part_reqs);
          }
        };  // fetch_and_add_part_reqs

    fetch_and_add_part_reqs(valid_source_entity_part_names);
    fetch_and_add_part_reqs(valid_target_entity_part_names);

    // Add the specialized neighbor linkers part requirements.
    const auto specialized_neighbor_linkers_part_names =
        valid_fixed_params.get<Teuchos::Array<std::string>>("specialized_neighbor_linkers_part_names");
    for (int i = 0; i < specialized_neighbor_linkers_part_names.size(); i++) {
      const std::string part_name = specialized_neighbor_linkers_part_names[i];
      if (part_name == NeighborLinkers::get_name()) {
        // Nothing to add.
      } else {
        // Add the associated part as a subset of the sphere sphere linkers agent.
        auto part_reqs = std::make_shared<mundy::meta::PartReqs>();
        part_reqs->set_part_name(part_name).set_part_rank(stk::topology::CONSTRAINT_RANK);
        NeighborLinkers::add_and_sync_subpart_reqs(part_reqs);
      }
    }
    mesh_reqs_ptr->sync(NeighborLinkers::get_mesh_requirements());

    return mesh_reqs_ptr;
  }

  /// \brief Get the valid fixed parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_fixed_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList()
            .set("valid_source_entity_part_names", mundy::core::make_string_array(universal_part_name_),
                 "Name of the source parts associated with this pairwise meta method.")
            .set("valid_target_entity_part_names", mundy::core::make_string_array(universal_part_name_),
                 "Name of the target parts associated with this pairwise meta method.")
            .set("specialized_neighbor_linkers_part_names",
                 mundy::core::make_string_array(default_specialized_neighbor_linkers_part_name_),
                 "The part names to which we will add the generated neighbor linkers. This should be a specialization "
                 "of the "
                 "neighbor linkers part or the neighbor linkers part itself.")
            .set("element_aabb_field_name", std::string(default_element_aabb_field_name_),
                 "Name of the element field containing the output axis-aligned boundary boxes.");
    return default_parameter_list;
  }

  /// \brief Get the valid mutable parameters for this class and their defaults.
  static Teuchos::ParameterList get_valid_mutable_params() {
    static Teuchos::ParameterList default_parameter_list =
        Teuchos::ParameterList().set("enforce_symmetry", true, "Enforce symmetry of the neighbor linkers.");
    return default_parameter_list;
  }

  /// \brief Generate a new instance of this class.
  ///
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A
  /// default fixed parameter list is accessible via \c get_fixed_valid_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                  const Teuchos::ParameterList &fixed_params) {
    return std::make_shared<STKSearch>(bulk_data_ptr, fixed_params);
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the mutable parameters. If a parameter is not provided, we use the default value.
  void set_mutable_params(const Teuchos::ParameterList &mutable_params) override;
  //@}

  //! \name Getters
  //@{

  /// \brief Get valid source entity parts for the method.
  /// By "valid source entity parts," we mean the parts whose entities this method can act on as source entities.
  std::vector<stk::mesh::Part *> get_valid_source_entity_parts() const override;

  /// \brief Get valid target entity parts for the method.
  /// By "valid target entity parts," we mean the parts whose entities this method can act on as target entities.
  std::vector<stk::mesh::Part *> get_valid_target_entity_parts() const override;
  //@}

  //! \name Actions
  //@{

  /// \brief Run the method's core calculation.
  /// \param source_input_selector The selector that defines the source entities to act on.
  /// \param target_input_selector The selector that defines the target entities to act on.
  void execute(const stk::mesh::Selector &source_input_selector,
               const stk::mesh::Selector &target_input_selector) override;

 private:
  //! \name Default parameters
  //@{

  /// \brief The default universal part name. This is an implementation detail hidden by STK and is subject to change.
  static constexpr std::string_view universal_part_name_ = "{UNIVERSAL}";
  static constexpr std::string_view default_specialized_neighbor_linkers_part_name_ = "NEIGHBOR_LINKERS";
  static constexpr std::string_view default_element_aabb_field_name_ = "ELEMENT_AABB";
  //@}

  //! \name Internal members
  //@{

  /// \brief The BulkData object this class acts upon.
  mundy::mesh::BulkData *bulk_data_ptr_ = nullptr;

  /// \brief The MetaData object this class acts upon.
  mundy::mesh::MetaData *meta_data_ptr_ = nullptr;

  /// \brief The valid source entity parts.
  std::vector<stk::mesh::Part *> valid_source_entity_part_ptrs_;

  /// \brief The valid target entity parts.
  std::vector<stk::mesh::Part *> valid_target_entity_part_ptrs_;

  /// \brief The neighbor linkers part.
  stk::mesh::Part *neighbor_linkers_part_ptr_ = nullptr;

  /// \brief The specialized neighbor linkers parts.
  std::vector<stk::mesh::Part *> specialized_neighbor_linkers_part_ptrs_;

  /// \brief The linked entities field pointer.
  LinkedEntitiesFieldType *linked_entities_field_ptr_ = nullptr;

  /// \brief The linked entity owners field pointer.
  stk::mesh::Field<int> *linked_entity_owners_field_ptr_ = nullptr;

  /// \brief The element aabb field pointer.
  stk::mesh::Field<double> *element_aabb_field_ptr_ = nullptr;

  /// \brief The enforce symmetry flag.
  bool enforce_symmetry_ = true;
  //@}
};  // STKSearch

}  // namespace techniques

}  // namespace generate_neighbor_linkers

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_GENERATE_NEIGHBOR_LINKERS_TECHNIQUES_STKSEARCH_HPP_
