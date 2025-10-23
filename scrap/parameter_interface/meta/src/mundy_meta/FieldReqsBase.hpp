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

#ifndef MUNDY_META_FIELDREQSBASE_HPP_
#define MUNDY_META_FIELDREQSBASE_HPP_

/// \file FieldReqsBase.hpp
/// \brief Declaration of the FieldReqsBase class

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/Field.hpp>      // for stk::mesh::Field
#include <stk_mesh/base/FieldBase.hpp>  // for stk::mesh::FieldBase
#include <stk_mesh/base/Part.hpp>       // for stk::mesh::Part
#include <stk_topology/topology.hpp>    // for stk::topology

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_mesh/MetaData.hpp>      // for mundy::mesh::MetaData

namespace mundy {

namespace meta {

/// \class FieldReqsBase
/// \brief A consistent interface for all \c FieldReqsBase.
class FieldReqsBase {
 public:
  //! \name Setters and Getters
  //@{

  /// \brief Set the required field name.
  /// \brief field_name [in] Required name of the field.
  virtual FieldReqsBase& set_field_name(const std::string& field_name) = 0;

  /// \brief Set the required field rank.
  /// \brief field_rank [in] Required rank of the field.
  virtual FieldReqsBase& set_field_rank(const stk::topology::rank_t& field_rank) = 0;

  /// \brief Set the required field dimension.
  /// \brief field_dimension [in] Required dimension of the field.
  virtual FieldReqsBase& set_field_dimension(const unsigned field_dimension) = 0;

  /// \brief Set the minimum required number of field states to the given value.
  /// \brief field_min_number_of_states [in] Minimum required number of states of the field.
  virtual FieldReqsBase& set_field_min_number_of_states(const unsigned field_min_number_of_states) = 0;

  /// \brief Set the minimum required number of field states UNLESS the current minimum number of states is larger.
  /// \brief field_min_number_of_states [in] Minimum required number of states of the field.
  virtual FieldReqsBase& set_field_min_number_of_states_if_larger(const unsigned field_min_number_of_states) = 0;

  /// \brief Get if the field name is constrained or not.
  virtual bool constrains_field_name() const = 0;

  /// \brief Get if the field rank is constrained or not.
  virtual bool constrains_field_rank() const = 0;

  /// \brief Get if the field dimension is constrained or not.
  virtual bool constrains_field_dimension() const = 0;

  /// \brief Get if the field minimum number of states is constrained or not.
  virtual bool constrains_field_min_number_of_states() const = 0;

  /// \brief Get if the field is fully specified.
  virtual bool is_fully_specified() const = 0;

  /// \brief Return the field name.
  /// Will throw an error if the field name is not constrained.
  virtual std::string get_field_name() const = 0;

  /// \brief Return the field rank.
  /// Will throw an error if the field rank is not constrained.
  virtual stk::topology::rank_t get_field_rank() const = 0;

  /// \brief Return the field dimension.
  /// Will throw an error if the field dimension is not constrained.
  virtual unsigned get_field_dimension() const = 0;

  /// \brief Return the minimum number of field states.
  /// Will throw an error if the minimum number of field states.
  virtual unsigned get_field_min_num_states() const = 0;

  /// \brief Return the typeinfo related to the field's type.
  virtual const std::type_info& get_field_type_info() const = 0;

  /// \brief Return the required field attribute names.
  virtual std::vector<std::string>& get_field_attribute_names() = 0;
  //@}

  //! \name Actions
  //@{

  /// \brief Declare/create the field that this class defines.
  virtual stk::mesh::FieldBase& declare_field_on_part(mundy::mesh::MetaData* const meta_data_ptr,
                                                      const stk::mesh::Part& part) const = 0;

  /// \brief Declare/create the field that this class defines and assign it to the entire mesh.
  virtual stk::mesh::FieldBase& declare_field_on_entire_mesh(mundy::mesh::MetaData* const meta_data_ptr) const = 0;

  /// \brief Delete the field name constraint (if it exists).
  virtual FieldReqsBase& delete_field_name() = 0;

  /// \brief Delete the field rank constraint (if it exists).
  virtual FieldReqsBase& delete_field_rank() = 0;

  /// \brief Delete the field dimension constraint (if it exists).
  virtual FieldReqsBase& delete_field_dimension() = 0;

  /// \brief Delete the field minimum number of states constraint (if it exists).
  virtual FieldReqsBase& delete_field_min_number_of_states() = 0;

  /// \brief Ensure that the current set of parameters is valid.
  ///
  /// Here, valid means that the rank, dimension, and number of states are > 0, but as unsigned ints, this is always the
  /// case. We will however, leave this checker incase the class grows and the set of requirements is no longer
  /// automatically satisfied.
  virtual FieldReqsBase& check_if_valid() = 0;

  /// \brief Require that an attribute with the given name be present on the field.
  ///
  /// \param attribute_name [in] The name of the attribute that must be present on the field.
  virtual FieldReqsBase& add_field_attribute(const std::string& attribute_name) = 0;

  /// \brief Synchronize (merge and rectify differences) the current parameters with any number of other \c FieldReqs.
  ///
  /// Here, syncing two \c FieldReqs object amounts to setting their number of states to be the
  /// maximum of their set min num states. For this process to be valid, the given
  /// \c FieldReqs must have the same rank, type, and dimension. It also syncs their attributes.
  ///
  /// \param field_reqs_ptr [in] A \c FieldReqs objects to sync with the current object.
  virtual FieldReqsBase& sync(std::shared_ptr<FieldReqsBase> field_reqs_ptr) = 0;

  /// \brief Dump the contents of \c FieldReqs to the given stream (defaults to std::cout).
  virtual void print(std::ostream& os = std::cout, int indent_level = 0) const = 0;

  /// \brief Return a string representation of the current set of requirements.
  virtual std::string get_reqs_as_a_string() const = 0;
  //@}

 protected:
  //! \name Protected member functions
  //@{

  /// \brief Set the master field requirements for this class.
  virtual FieldReqsBase& set_master_field_reqs(std::shared_ptr<FieldReqsBase> master_field_req_ptr) = 0;

  /// \brief Get the master field requirements for this class.
  virtual std::shared_ptr<FieldReqsBase> get_master_field_reqs() = 0;

  /// \brief Get if the current reqs have a master field reqs.
  virtual bool has_master_field_reqs() const = 0;
  //@}

 private:
  //! \name Friends <3
  //@{

  /// \brief Give FieldReqs<T> access to the protected member functions.
  template <typename T>
  friend class FieldReqs;
};  // FieldReqsBase

//}

//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQSBASE_HPP_
