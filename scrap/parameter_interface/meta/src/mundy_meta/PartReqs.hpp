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

#ifndef MUNDY_META_PARTREQS_HPP_
#define MUNDY_META_PARTREQS_HPP_

/// \file PartReqs.hpp
/// \brief Declaration of the PartReqs class

// C++ core libs
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <sstream>      // for std::stringstream
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_mesh/MetaData.hpp>   // for mundy::mesh::MetaData
#include <mundy_meta/FieldReqs.hpp>  // for mundy::meta::FieldReqs, mundy::meta::FieldReqsBase

namespace mundy {

namespace meta {

/// \class PartReqs
/// \brief A set requirements imposed upon a Part and its fields.
class PartReqs {
 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements.
  PartReqs() = default;

  /// \brief Fully  Constructor with partial requirements. Version 1.
  ///
  /// \param part_name [in] Name of the part.
  explicit PartReqs(const std::string &part_name);

  /// \brief Constructor with partial requirements. Version 2.
  ///
  /// \param part_name [in] Name of the part.
  ///
  /// \param part_topology [in] Topology of entities within the part.
  PartReqs(const std::string &part_name, const stk::topology::topology_t &part_topology);

  /// \brief Constructor with partial requirements. Version 3.
  ///
  /// \param part_name [in] Name of the part. If the name already exists for the given topology, the two parameter
  /// sets must be valid.
  ///
  /// \param part_rank [in] Maximum rank of entities within the part. Can contain any element of lower rank, regardless
  /// of topology.
  ///
  /// \param part_fields [in] Vector of field parameters for the fields defined on this part.
  PartReqs(const std::string &part_name, const stk::topology::rank_t &part_rank);
  //@}

  //! \name Setters
  //@{

  /// \brief Set the required part name.
  /// \param part_name [in] Required name of the part.
  PartReqs &set_part_name(const std::string &part_name);

  /// \brief Set the required part topology.
  /// \param part_topology [in] Required topology of the part.
  PartReqs &set_part_topology(const stk::topology::topology_t &part_topology);

  /// \brief Set the required part rank.
  /// \param part_rank [in] Required rank of the part.
  PartReqs &set_part_rank(const stk::topology::rank_t &part_rank);

  /// \brief Disable entity induction for this part.
  PartReqs &disable_entity_induction();

  /// \brief Enable entity induction for this part.
  PartReqs &enable_entity_induction();

  /// \brief Delete the part name constraint (if it exists).
  PartReqs &delete_part_name();

  /// \brief Delete the part topology constraint (if it exists).
  PartReqs &delete_part_topology();

  /// \brief Delete the part rank constraint (if it exists).
  PartReqs &delete_part_rank();

  /// \brief Delete the part induction constraint (if it exists).
  PartReqs &delete_part_induction();

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  /// If the field already exists, we sync their requirements.
  ///
  /// \param field_reqs_ptr [in] Pointer to the field parameters to add to the part.
  PartReqs &add_and_sync_field_reqs(std::shared_ptr<FieldReqsBase> field_reqs_ptr);

  /// \brief Add the provided field to the part, given that it is valid and does not conflict with existing fields.
  ///
  /// \param field_name [in] Name of the field to add to the part.
  /// \param field_rank [in] Rank of the field to add to the part.
  /// \param field_dimension [in] Dimension of the field to add to the part.
  /// \param field_min_number_of_states [in] Minimum number of states for the field to add to the part.
  ///
  /// \tparam FieldType [in] The type of the field to add to the part.
  template <typename FieldType>
  PartReqs &add_field_reqs(const std::string &field_name, const stk::topology::rank_t &field_rank,
                           const unsigned field_dimension, const unsigned field_min_number_of_states) {
    return add_and_sync_field_reqs(
        std::make_shared<FieldReqs<FieldType>>(field_name, field_rank, field_dimension, field_min_number_of_states));
  }

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  /// If the subpart already exists, we sync their requirements. We also add all of our requirements to the subpart, to
  /// mimic inheritance.
  ///
  /// TODO(palmerb4): Are there any restrictions on what can and cannot be a subpart? If so, encode them here.
  ///
  /// \param part_reqs_ptr [in] Pointer to the sub-part requirements to add to the part.
  PartReqs &add_and_sync_subpart_reqs(std::shared_ptr<PartReqs> part_reqs_ptr);

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  ///
  /// \param part_name [in] Name of the sub-part to add to the part.
  PartReqs &add_subpart_reqs(const std::string &part_name) {
    return add_and_sync_subpart_reqs(std::make_shared<PartReqs>(part_name));
  }

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  ///
  /// \param part_name [in] Name of the sub-part to add to the part.
  /// \param part_topology [in] Topology of entities within the sub-part.
  PartReqs &add_subpart_reqs(const std::string &part_name, const stk::topology::topology_t &part_topology) {
    return add_and_sync_subpart_reqs(std::make_shared<PartReqs>(part_name, part_topology));
  }

  /// \brief Add the provided part as a subpart of this part, given that it is valid.
  ///
  /// \param part_name [in] Name of the sub-part to add to the part.
  /// \param part_rank [in] Maximum rank of entities within the sub-part. Can contain any element of lower rank,
  /// regardless of topology.
  PartReqs &add_subpart_reqs(const std::string &part_name, const stk::topology::rank_t &part_rank) {
    return add_and_sync_subpart_reqs(std::make_shared<PartReqs>(part_name, part_rank));
  }

  /// \brief Require that an attribute with the given name be present on the part.
  ///
  /// \param attribute_name [in] The name of the attribute that must be present on the part.
  PartReqs &add_part_attribute(const std::string &attribute_name);
  //@}

  //! \name Getters
  //@{

  /// \brief Get if the part name is constrained or not.
  bool constrains_part_name() const;

  /// \brief Get if the part topology is constrained or not.
  bool constrains_part_topology() const;

  /// \brief Get if the part rank is constrained or not.
  bool constrains_part_rank() const;

  /// \brief Get if the part induction is constrained or not.
  bool constrains_part_induction() const;

  /// \brief Get if the part is fully specified.
  bool is_fully_specified() const;

  /// \brief Return the part name.
  /// Will throw an error if the part name is not constrained.
  std::string get_part_name() const;

  /// \brief Return the part topology.
  /// Will throw an error if the part topology is not constrained.
  stk::topology::topology_t get_part_topology() const;

  /// \brief Return the part rank.
  /// Will throw an error if the part rank is not constrained.
  stk::topology::rank_t get_part_rank() const;

  /// \brief Return if the part has entity induction.
  bool has_entity_induction() const;

  /// \brief Return the part field map.
  std::vector<std::map<std::string, std::shared_ptr<FieldReqsBase>>> &get_part_ranked_field_map();

  /// \brief Return the part subpart map.
  std::map<std::string, std::shared_ptr<PartReqs>> &get_part_subpart_map();

  /// \brief Return the required part attribute names.
  std::vector<std::string> &get_part_attribute_names();
  //@}

  //! \name Actions
  //@{

  /// \brief Declare the part that this class defines including any of its fields, its subparts, and their
  /// fields/subparts.
  ///
  /// This method can return three different types of parts based on the existing set of constraints.
  ///  - those with a predefined topology (if constrains_part_topology is true),
  ///  - those with a predefined rank that will (if constrains_part_topology is false but constrains_part_rank is true),
  ///  - those with no topology or rank (if neither constrains_part_topology nor constrains_part_rank are true).
  ///
  /// In each case, the part name must be set or an error will be thrown.
  ///
  /// \note Redeclaration of a previously declared part, will return the previous part.
  stk::mesh::Part &declare_part_on_mesh(mundy::mesh::MetaData *const meta_data_ptr) const;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means:
  ///   1. the rank of the fields does not exceed the rank of the part's topology.
  PartReqs &check_if_valid();

  /// \brief Synchronize (merge and rectify differences) the current requirements with another \c PartReqs.
  ///
  /// \param part_reqs_ptr [in] An \c PartReqs object to sync with the current object.
  PartReqs &sync(std::shared_ptr<PartReqs> part_reqs_ptr);

  /// \brief Dump the contents of \c PartReqs to the given stream (defaults to std::cout).
  void print(std::ostream &os = std::cout, int indent_level = 0) const;

  /// \brief Return a string representation of the current set of requirements.
  std::string get_reqs_as_a_string() const;
  //@}

 private:
  //! \name Private member functions
  //@{

  /// \brief Set the master field requirements for this class.
  PartReqs &set_master_part_reqs(std::shared_ptr<PartReqs> master_part_req_ptr);

  /// \brief Get the master part requirements for this class.
  std::shared_ptr<PartReqs> get_master_part_reqs();

  /// \brief Get if the current reqs have a master part reqs.
  bool has_master_part_reqs() const;
  //@}

  //! \name Private data
  //@{

  /// \brief Pointer to the master part requirements.
  std::shared_ptr<PartReqs> master_part_reqs_ptr_ = nullptr;

  /// \brief Name of the part.
  std::string part_name_;

  /// \brief Topology of entities in the part.
  stk::topology::topology_t part_topology_;

  /// \brief Rank of the part.
  stk::topology::rank_t part_rank_;

  /// \brief If we are driven by a master PartReqs object.
  bool has_master_part_reqs_ = false;

  /// \brief If the name of the part is set or not.
  bool part_name_is_set_ = false;

  /// \brief If the topology of entities in the part is set or not.
  bool part_topology_is_set_ = false;

  /// \brief If the rank of the part is set or not.
  bool part_rank_is_set_ = false;

  /// \brief If the part is an io-compatible part or not.
  bool is_io_part_ = false;

  /// \brief If the part has entity induction or not.
  bool has_entity_induction_ = true;

  /// \brief If the part has entity induction is set or not.
  bool has_entity_induction_is_set_ = false;

  /// \brief A set of maps from field name to field params for each rank.
  std::vector<std::map<std::string, std::shared_ptr<FieldReqsBase>>> part_ranked_field_maps_{stk::topology::NUM_RANKS};

  /// \brief A map from subpart name to the part params of each sub-part.
  std::map<std::string, std::shared_ptr<PartReqs>> part_subpart_map_;

  /// \brief A vector of required part attribute names.
  std::vector<std::string> required_part_attribute_names_;
  //@}
};  // PartReqs

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_PARTREQS_HPP_
