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

#ifndef MUNDY_META_FIELDREQS_HPP_
#define MUNDY_META_FIELDREQS_HPP_

/// \file FieldReqs.hpp
/// \brief Declaration of the FieldReqs class

// External
#include <fmt/format.h>  // for fmt::format

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>    // for stk::mesh::Field
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>       // for mundy::mesh::MetaData
#include <mundy_mesh/fmt_stk_types.hpp>  // adds fmt::format for stk types
#include <mundy_meta/FieldReqsBase.hpp>  // for mundy::meta::FieldReqsBase

namespace mundy {

namespace meta {

/// \class FieldReqs
/// \brief A set of necessary parameters for declaring a new field.
///
/// \tparam FieldType_t Type for elements in the field.
template <typename FieldType_t>
class FieldReqs : public FieldReqsBase {
 public:
  //! \name Typedefs
  //@{

  /// \tparam FieldType Type for elements in the field. Set by the template parameter.
  using FieldType = FieldType_t;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default construction is allowed
  /// Default construction corresponds to having no requirements (expect for the field type).
  FieldReqs() = default;

  /// \brief Constructor with full fill.
  ///
  /// \param field_name [in] Name of the field.
  ///
  /// \param field_rank [in] Rank that the field will be assigned to.
  ///
  /// \param field_dimension [in] Dimension of the field. For example, a dimension of three would be a vector.
  ///
  /// \param field_min_number_of_states [in] Minimum number of rotating states that this field will have.
  FieldReqs(const std::string &field_name, const stk::topology::rank_t &field_rank, const unsigned field_dimension,
            const unsigned field_min_number_of_states);
  //@}

  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \param field_name [in] Required name of the field.
  FieldReqs<FieldType_t> &set_field_name(const std::string &field_name) override;

  /// \brief Set the required field rank.
  /// \param field_rank [in] Required rank of the field.
  FieldReqs<FieldType_t> &set_field_rank(const stk::topology::rank_t &field_rank) override;

  /// \brief Set the required field dimension.
  /// \param field_dimension [in] Required dimension of the field.
  FieldReqs<FieldType_t> &set_field_dimension(const unsigned field_dimension) override;

  /// \brief Set the minimum required number of field states.
  /// \param field_min_number_of_states [in] Minimum required number of states of the field.
  FieldReqs<FieldType_t> &set_field_min_number_of_states(const unsigned field_min_number_of_states) override;

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \param field_min_number_of_states [in] Minimum required number of states of the field.
  FieldReqs<FieldType_t> &set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) override;

  /// \brief Get if the field name is constrained or not.
  bool constrains_field_name() const override;

  /// \brief Get if the field rank is constrained or not.
  bool constrains_field_rank() const override;

  /// \brief Get if the field dimension is constrained or not.
  bool constrains_field_dimension() const override;

  /// \brief Get if the field minimum number of states is constrained or not.
  bool constrains_field_min_number_of_states() const override;

  /// \brief Get if the field is fully specified.
  bool is_fully_specified() const override;

  /// \brief Return the field name.
  /// Will throw an error if the field name is not constrained.
  std::string get_field_name() const override;

  /// \brief Return the field rank.
  /// Will throw an error if the field rank is not constrained.
  stk::topology::rank_t get_field_rank() const override;

  /// \brief Return the field dimension.
  /// Will throw an error if the field dimension is not constrained.
  unsigned get_field_dimension() const override;

  /// \brief Return the minimum number of field states.
  /// Will throw an error if the minimum number of field states.
  unsigned get_field_min_num_states() const override;

  /// \brief Return the typeinfo related to the field's type.
  const std::type_info &get_field_type_info() const override;

  /// \brief Return the required field attribute names.
  std::vector<std::string> &get_field_attribute_names() override;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines and assign it to a part.
  stk::mesh::Field<FieldType_t> &declare_field_on_part(mundy::mesh::MetaData *const meta_data_ptr,
                                                       const stk::mesh::Part &part) const override;

  /// \brief Declare/create the field that this class defines and assign it to the entire mesh.
  stk::mesh::Field<FieldType_t> &declare_field_on_entire_mesh(
      mundy::mesh::MetaData *const meta_data_ptr) const override;

  /// \brief Delete the field name constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_name() override;

  /// \brief Delete the field rank constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_rank() override;

  /// \brief Delete the field dimension constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_dimension() override;

  /// \brief Delete the field minimum number of states constraint (if it exists).
  FieldReqs<FieldType_t> &delete_field_min_number_of_states() override;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  FieldReqs<FieldType_t> &check_if_valid() override;

  /// \brief Require that an attribute with the given name be present on the field.
  ///
  /// \param attribute_name [in] The name of the attribute that must be present on the field.
  FieldReqs<FieldType_t> &add_field_attribute(const std::string &attribute_name) override;

  /// \brief Synchronize (merge and rectify differences) the current parameters with any number of other \c FieldReqs.
  ///
  /// Here, syncing two \c FieldReqs object amounts to setting their number of states to be the
  /// maximum of their set min num states. For this process to be valid, the given
  /// \c FieldReqs must have the same rank, type, and dimension. It also syncs their attributes.
  ///
  /// \param field_reqs_ptr [in] A \c FieldReqs objects to sync with the current object.
  FieldReqs<FieldType_t> &sync(std::shared_ptr<FieldReqsBase> field_reqs_ptr) override;

  /// \brief Dump the contents of \c FieldReqs to the given stream (defaults to std::cout).
  void print(std::ostream &os = std::cout, int indent_level = 0) const override;

  /// \brief Return a string representation of the current set of requirements.
  std::string get_reqs_as_a_string() const override;
  //@}

 private:
  //! \name Private member functions
  //@{

  /// \brief Set the master field requirements for this class.
  FieldReqs<FieldType_t> &set_master_field_reqs(std::shared_ptr<FieldReqsBase> master_field_req_ptr) override;

  /// \brief Get the master field requirements for this class.
  std::shared_ptr<FieldReqsBase> get_master_field_reqs() override;

  /// \brief Get if the current reqs have a master field reqs.
  bool has_master_field_reqs() const override;
  //@}

  //! \name Private data
  //@{

  /// \brief Pointer to the master FieldReqs object.
  ///
  /// FieldReqs need to be able to synchronize with other FieldReqs. In doing so, changing one of those field reqs
  /// should directly result in all other synchronized field reqs being updated. We choose to implement this by letting
  /// each FieldReqs class store a pointer to a master FieldReqs object.
  ///
  /// If we declare two FieldReqs are synced and neither of them currently has a master, then we will create a shared
  /// master FieldReqs object from our merged requirements. If one of the two FieldReqs has a master, then the one
  /// without a master will point to the existing master reqs. On the other hand, if both have master reqs, then we
  /// synchronize the masters (leading to a recursion).
  std::shared_ptr<FieldReqsBase> master_field_reqs_ptr_ = nullptr;

  /// \brief Name of the field.
  std::string field_name_;

  /// \brief Rank that the field will be assigned to.
  stk::topology::rank_t field_rank_;

  /// \brief Dimension of the field. For example, a dimension of three would be a vector.
  unsigned field_dimension_;

  /// \brief Minimum number of rotating states that this field will have.
  unsigned field_min_number_of_states_;

  /// \brief If we are driven by a master FieldReqs object.
  bool has_master_field_reqs_ = false;

  /// \brief If the name of the field is set or not.
  bool field_name_is_set_;

  /// \brief If the rank that the field will be assigned to is set or not.
  bool field_rank_is_set_;

  /// \brief If the dimension of the field. For example, a dimension of three would be a vector is set or not.
  bool field_dimension_is_set_;

  /// \brief If the minimum number of rotating states that this field will have is set or not.
  bool field_min_number_of_states_is_set_;

  /// \brief A vector of required field attribute names.
  std::vector<std::string> required_field_attribute_names_;
  //@}
};  // FieldReqs

//! \name template implementations
//@{

// \name Constructors and destructor
//{
template <typename FieldType>
FieldReqs<FieldType>::FieldReqs(const std::string &field_name, const stk::topology::rank_t &field_rank,
                                const unsigned field_dimension, const unsigned field_min_number_of_states) {
  this->set_field_name(field_name);
  this->set_field_rank(field_rank);
  this->set_field_dimension(field_dimension);
  this->set_field_min_number_of_states(field_min_number_of_states);
}
//}

// \name Setters and Getters
//{

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_name(const std::string &field_name) {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->set_field_name(field_name);
  } else {
    field_name_ = field_name;
    field_name_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_rank(const stk::topology::rank_t &field_rank) {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->set_field_rank(field_rank);
  } else {
    field_rank_ = field_rank;
    field_rank_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_dimension(const unsigned field_dimension) {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->set_field_dimension(field_dimension);
  } else {
    field_dimension_ = field_dimension;
    field_dimension_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_min_number_of_states(const unsigned field_min_number_of_states) {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->set_field_min_number_of_states(field_min_number_of_states);
  } else {
    field_min_number_of_states_ = field_min_number_of_states;
    field_min_number_of_states_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_field_min_number_of_states_if_larger(
    const unsigned field_min_number_of_states) {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->set_field_min_number_of_states_if_larger(field_min_number_of_states);
  } else {
    if (this->constrains_field_min_number_of_states()) {
      field_min_number_of_states_ = std::max(field_min_number_of_states, field_min_number_of_states_);
    } else {
      field_min_number_of_states_ = field_min_number_of_states;
    }
    field_min_number_of_states_is_set_ = true;
    this->check_if_valid();
  }
  return *this;
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_name() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->constrains_field_name();
  } else {
    return field_name_is_set_;
  }
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_rank() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->constrains_field_rank();
  } else {
    return field_rank_is_set_;
  }
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_dimension() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->constrains_field_dimension();
  } else {
    return field_dimension_is_set_;
  }
}

template <typename FieldType>
bool FieldReqs<FieldType>::constrains_field_min_number_of_states() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->constrains_field_min_number_of_states();
  } else {
    return field_min_number_of_states_is_set_;
  }
}

template <typename FieldType>
bool FieldReqs<FieldType>::is_fully_specified() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->is_fully_specified();
  } else {
    return this->constrains_field_name() && this->constrains_field_rank() && this->constrains_field_dimension() &&
           this->constrains_field_min_number_of_states();
  }
}

template <typename FieldType>
std::string FieldReqs<FieldType>::get_field_name() const {
  if (has_master_field_reqs_) {
    MUNDY_THROW_REQUIRE(master_field_reqs_ptr_ != nullptr, std::logic_error,
                        "FieldReqs: The master field requirements have not been set. Cannot return a null pointer.");
    return master_field_reqs_ptr_->get_field_name();
  } else {
    MUNDY_THROW_REQUIRE(
        this->constrains_field_name(), std::logic_error,
        "FieldReqs: Attempting to access the field name requirement even though field name is unconstrained.");

    return field_name_;
  }
}

template <typename FieldType>
stk::topology::rank_t FieldReqs<FieldType>::get_field_rank() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->get_field_rank();
  } else {
    MUNDY_THROW_REQUIRE(
        this->constrains_field_rank(), std::logic_error,
        "FieldReqs: Attempting to access the field rank requirement even though field rank is unconstrained.");

    return field_rank_;
  }
}

template <typename FieldType>
unsigned FieldReqs<FieldType>::get_field_dimension() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->get_field_dimension();
  } else {
    MUNDY_THROW_REQUIRE(this->constrains_field_dimension(), std::logic_error,
                        std::string("FieldReqs: Attempting to access the field dimension requirement even though ") +
                            "field dimension is unconstrained.");

    return field_dimension_;
  }
}

template <typename FieldType>
unsigned FieldReqs<FieldType>::get_field_min_num_states() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->get_field_min_num_states();
  } else {
    MUNDY_THROW_REQUIRE(
        this->constrains_field_min_number_of_states(), std::logic_error,
        std::string(
            "FieldReqs: Attempting to access the field minimum number of states requirement even though field ") +
            "min_number_of_states is unconstrained.");

    return field_min_number_of_states_;
  }
}

template <typename FieldType>
const std::type_info &FieldReqs<FieldType>::get_field_type_info() const {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->get_field_type_info();
  } else {
    return typeid(FieldType);
  }
}

template <typename FieldType>
std::vector<std::string> &FieldReqs<FieldType>::get_field_attribute_names() {
  if (has_master_field_reqs_) {
    return master_field_reqs_ptr_->get_field_attribute_names();
  } else {
    return required_field_attribute_names_;
  }
}
//}

// \name Private member functions
//{

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::set_master_field_reqs(std::shared_ptr<FieldReqsBase> master_field_req_ptr) {
  MUNDY_THROW_REQUIRE(
      !has_master_field_reqs_, std::logic_error,
      "The master field requirements have already been set. Overriding it could lead to undefined behavior.");
  master_field_reqs_ptr_ = std::move(master_field_req_ptr);
  has_master_field_reqs_ = true;
  return *this;
}

template <typename FieldType>
std::shared_ptr<FieldReqsBase> FieldReqs<FieldType>::get_master_field_reqs() {
  MUNDY_THROW_REQUIRE(has_master_field_reqs_, std::logic_error,
                      "The master field requirements have not been set. Cannot return a null pointer.");
  return master_field_reqs_ptr_;
}

template <typename FieldType>
bool FieldReqs<FieldType>::has_master_field_reqs() const {
  return has_master_field_reqs_;
}
//}

// \name Actions
//{

template <typename FieldType>
stk::mesh::Field<FieldType> &FieldReqs<FieldType>::declare_field_on_part(mundy::mesh::MetaData *const meta_data_ptr,
                                                                         const stk::mesh::Part &part) const {
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument, "FieldReqs: MetaData pointer cannot be null).");

  MUNDY_THROW_REQUIRE(this->constrains_field_name(), std::logic_error,
                      "FieldReqs: Field name must be set before calling declare_field.");
  MUNDY_THROW_REQUIRE(this->constrains_field_rank(), std::logic_error,
                      "FieldReqs: Field rank must be set before calling declare_field.");
  MUNDY_THROW_REQUIRE(this->constrains_field_dimension(), std::logic_error,
                      "FieldReqs: Field dimension must be set before calling declare_field.");
  MUNDY_THROW_REQUIRE(this->constrains_field_min_number_of_states(), std::logic_error,
                      "FieldReqs: Field minimum number of states must be set before calling declare_field.");

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field = meta_data_ptr->declare_field<FieldType>(
      this->get_field_rank(), this->get_field_name(), this->get_field_min_num_states());
  stk::mesh::put_field_on_mesh(field, part, this->get_field_dimension(), nullptr);

  // Set the field attributes.
  for (auto const &attribute_name : const_cast<FieldReqs<FieldType> *>(this)->get_field_attribute_names()) {
    std::any empty_attribute;
    meta_data_ptr->declare_attribute(field, attribute_name, empty_attribute);
  }
  return field;
}

template <typename FieldType>
stk::mesh::Field<FieldType> &FieldReqs<FieldType>::declare_field_on_entire_mesh(
    mundy::mesh::MetaData *const meta_data_ptr) const {
  MUNDY_THROW_REQUIRE(meta_data_ptr != nullptr, std::invalid_argument, "FieldReqs: MetaData pointer cannot be null).");

  MUNDY_THROW_REQUIRE(this->constrains_field_name(), std::logic_error,
                      "FieldReqs: Field name must be set before calling declare_field.");
  MUNDY_THROW_REQUIRE(this->constrains_field_rank(), std::logic_error,
                      "FieldReqs: Field rank must be set before calling declare_field.");
  MUNDY_THROW_REQUIRE(this->constrains_field_dimension(), std::logic_error,
                      "FieldReqs: Field dimension must be set before calling declare_field.");
  MUNDY_THROW_REQUIRE(this->constrains_field_min_number_of_states(), std::logic_error,
                      "FieldReqs: Field minimum number of states must be set before calling declare_field.");

  // Declare the field and assign it to the given part.
  stk::mesh::Field<FieldType> &field = meta_data_ptr->declare_field<FieldType>(
      this->get_field_rank(), this->get_field_name(), this->get_field_min_num_states());
  stk::mesh::put_field_on_entire_mesh(field, this->get_field_dimension());

  // Set the field attributes.
  for (auto const &attribute_name : const_cast<FieldReqs<FieldType> *>(this)->get_field_attribute_names()) {
    std::any empty_attribute;
    meta_data_ptr->declare_attribute(field, attribute_name, empty_attribute);
  }
  return field;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_name() {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->delete_field_name();
  } else {
    field_name_is_set_ = false;
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_rank() {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->delete_field_rank();
  } else {
    field_rank_is_set_ = false;
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_dimension() {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->delete_field_dimension();
  } else {
    field_dimension_is_set_ = false;
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::delete_field_min_number_of_states() {
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->delete_field_min_number_of_states();
  } else {
    field_min_number_of_states_is_set_ = false;
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::check_if_valid() {
  // The only invalid state is if we have a master field reqs object but master_field_reqs_ptr_ is null.
  if (has_master_field_reqs_) {
    MUNDY_THROW_REQUIRE(master_field_reqs_ptr_ != nullptr, std::logic_error,
                        "FieldReqs: We have a master field reqs object but master_field_reqs_ptr_ is null.");
    master_field_reqs_ptr_->check_if_valid();
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::add_field_attribute(const std::string &attribute_name) {
  // Adding an existing attribute is perfectly fine. It's a no-op. This merely adds more responsibility to
  // the user to ensure that an they don't unintentionally edit an attribute that is used by another method.
  if (has_master_field_reqs_) {
    master_field_reqs_ptr_->add_field_attribute(attribute_name);
  } else {
    const bool attribute_exists =
        std::count(required_field_attribute_names_.begin(), required_field_attribute_names_.end(), attribute_name) > 0;
    if (!attribute_exists) {
      required_field_attribute_names_.push_back(attribute_name);
    }
  }
  return *this;
}

template <typename FieldType>
FieldReqs<FieldType> &FieldReqs<FieldType>::sync(std::shared_ptr<FieldReqsBase> field_reqs_ptr) {
  // TODO(palmerb4): Move this to a friend non-member function.
  // TODO(palmerb4): Optimize this function for perfect forwarding.

  // Check if the provided pointer is valid. Throw an error if it is not. Originally, we had this as a no-op, but now
  // that synchronizing sets the passed in FieldReqs object to be the master, we need to ensure that the passed in
  // object is valid.
  MUNDY_THROW_REQUIRE(field_reqs_ptr != nullptr, std::invalid_argument,
                      "FieldReqs: The given FieldReqs pointer cannot be null.");

  auto merge = [&](FieldReqsBase *us_ptr, FieldReqsBase *them_ptr, FieldReqsBase *merged_ptr) {
    // Check if the provided parameters are valid.
    us_ptr->check_if_valid();
    them_ptr->check_if_valid();

    // Check for compatibility if both classes define a requirement, otherwise store the new requirement.
    MUNDY_THROW_REQUIRE(us_ptr->get_field_type_info() == them_ptr->get_field_type_info(), std::invalid_argument,
                        "FieldReqs: Field type mismatch between our field type and the given requirements.");

    const bool we_constrain_field_name = us_ptr->constrains_field_name();
    const bool they_constrain_field_name = them_ptr->constrains_field_name();
    if (we_constrain_field_name && they_constrain_field_name) {
      MUNDY_THROW_REQUIRE(
          us_ptr->get_field_name() == them_ptr->get_field_name(), std::invalid_argument,
          std::string("FieldReqs: One of the inputs has incompatible name (") + them_ptr->get_field_name() + ").\n");
      merged_ptr->set_field_name(us_ptr->get_field_name());
    } else if (we_constrain_field_name) {
      merged_ptr->set_field_name(us_ptr->get_field_name());
    } else if (they_constrain_field_name) {
      merged_ptr->set_field_name(them_ptr->get_field_name());
    }

    const bool we_constrain_field_rank = us_ptr->constrains_field_rank();
    const bool they_constrain_field_rank = them_ptr->constrains_field_rank();
    if (we_constrain_field_rank && they_constrain_field_rank) {
      MUNDY_THROW_REQUIRE(
          us_ptr->get_field_rank() == them_ptr->get_field_rank(), std::invalid_argument,
          fmt::format("FieldReqs: One of the inputs has incompatible rank ({}).", them_ptr->get_field_rank()));
      merged_ptr->set_field_rank(us_ptr->get_field_rank());
    } else if (we_constrain_field_rank) {
      merged_ptr->set_field_rank(us_ptr->get_field_rank());
    } else if (they_constrain_field_rank) {
      merged_ptr->set_field_rank(them_ptr->get_field_rank());
    }

    const bool we_constrain_field_dimension = us_ptr->constrains_field_dimension();
    const bool they_constrain_field_dimension = them_ptr->constrains_field_dimension();
    if (we_constrain_field_dimension && they_constrain_field_dimension) {
      MUNDY_THROW_REQUIRE(
          us_ptr->get_field_dimension() == them_ptr->get_field_dimension(), std::invalid_argument,
          fmt::format("FieldReqs: One of the inputs has incompatible dimension ({})", them_ptr->get_field_dimension()));
      merged_ptr->set_field_dimension(us_ptr->get_field_dimension());
    } else if (we_constrain_field_dimension) {
      merged_ptr->set_field_dimension(us_ptr->get_field_dimension());
    } else if (they_constrain_field_dimension) {
      merged_ptr->set_field_dimension(them_ptr->get_field_dimension());
    }

    const bool we_constrain_field_min_number_of_states = us_ptr->constrains_field_min_number_of_states();
    const bool they_constrain_field_min_number_of_states = them_ptr->constrains_field_min_number_of_states();
    if (we_constrain_field_min_number_of_states && they_constrain_field_min_number_of_states) {
      // Min num states is special. We take the max of the two.
      const unsigned max_num_states =
          std::max(us_ptr->get_field_min_num_states(), them_ptr->get_field_min_num_states());
      merged_ptr->set_field_min_number_of_states(max_num_states);
    } else if (we_constrain_field_min_number_of_states) {
      merged_ptr->set_field_min_number_of_states(us_ptr->get_field_min_num_states());
    } else if (they_constrain_field_min_number_of_states) {
      merged_ptr->set_field_min_number_of_states(them_ptr->get_field_min_num_states());
    }

    // Loop over our attribute map and add our attributes to the given FieldReqs object.
    for (const std::string &attribute_name : us_ptr->get_field_attribute_names()) {
      merged_ptr->add_field_attribute(attribute_name);
    }
  };  // merge

  // To prevent circular dependencies, we will check if the given FieldReqs pointer points to us. If it does, then
  // their's nothing to do.
  bool does_field_req_ptr_point_to_us = field_reqs_ptr.get() == this;
  if (!does_field_req_ptr_point_to_us) {
    const bool we_have_master_field_reqs = this->has_master_field_reqs();
    const bool they_have_master_field_reqs = field_reqs_ptr->has_master_field_reqs();

    if (we_have_master_field_reqs && they_have_master_field_reqs) {
      // If both have master reqs, then we synchronize the masters (potentially leading to an upward tree traversal).
      this->get_master_field_reqs()->sync(field_reqs_ptr->get_master_field_reqs());
    } else if (we_have_master_field_reqs && !they_have_master_field_reqs) {
      // If we have a master and they don't, then we merge their requirements with our master and then set their master
      // to be our master.
      merge(this, field_reqs_ptr.get(), this->get_master_field_reqs().get());
      field_reqs_ptr->set_master_field_reqs(this->get_master_field_reqs());
    } else if (!we_have_master_field_reqs && they_have_master_field_reqs) {
      // If they have a master and we don't, then we merge our requirements with their master and then set our master to
      // be their master.
      merge(this, field_reqs_ptr.get(), field_reqs_ptr->get_master_field_reqs().get());
      this->set_master_field_reqs(field_reqs_ptr->get_master_field_reqs());
    } else {
      // If neither has a master, then we will create a shared master FieldReqs object from our merged requirements.
      auto shared_master_field_reqs_ptr = std::make_shared<FieldReqs<FieldType>>();
      merge(this, field_reqs_ptr.get(), shared_master_field_reqs_ptr.get());
      this->set_master_field_reqs(shared_master_field_reqs_ptr);
      field_reqs_ptr->set_master_field_reqs(shared_master_field_reqs_ptr);
    }
  }
  return *this;
}

template <typename FieldType>
void FieldReqs<FieldType>::print(std::ostream &os, int indent_level) const {
  std::cout << "printing FieldReqs" << std::endl;
  std::string indent(indent_level * 2, ' ');

  os << indent << "FieldReqs: " << std::endl;

  if (this->constrains_field_name()) {
    os << indent << "  name: " << this->get_field_name() << std::endl;
  } else {
    os << indent << "  name is not set." << std::endl;
  }

  if (this->constrains_field_rank()) {
    os << indent << "  rank: " << this->get_field_rank() << std::endl;
  } else {
    os << indent << "  rank is not set." << std::endl;
  }

  if (this->constrains_field_dimension()) {
    os << indent << "  dimension: " << this->get_field_dimension() << std::endl;
  } else {
    os << indent << "  dimension is not set." << std::endl;
  }

  if (this->constrains_field_min_number_of_states()) {
    os << indent << "  min number of states: " << this->get_field_min_num_states() << std::endl;
  } else {
    os << indent << "  min number of states is not set." << std::endl;
  }

  os << indent << "  type info: " << this->get_field_type_info().name() << std::endl;

  os << indent << "  attributes: " << std::endl;
  int attribute_count = 0;
  for (const std::string &attribute_name : const_cast<FieldReqs<FieldType> *>(this)->get_field_attribute_names()) {
    os << indent << "  Attribute " << attribute_count << " has name (" << attribute_name << ")" << std::endl;
    attribute_count++;
  }

  os << indent << "End of FieldReqs" << std::endl;

  std::cout << "done printing FieldReqs" << std::endl;
}

template <typename FieldType>
std::string FieldReqs<FieldType>::get_reqs_as_a_string() const {
  std::stringstream ss;
  this->print(ss);
  return ss.str();
}
//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQS_HPP_
