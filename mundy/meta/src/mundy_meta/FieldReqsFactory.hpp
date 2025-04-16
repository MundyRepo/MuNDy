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

#ifndef MUNDY_META_FIELDREQSFACTORY_HPP_
#define MUNDY_META_FIELDREQSFACTORY_HPP_

/// \file FieldReqsFactory.hpp
/// \brief Declaration of the FieldReqsFactory class

// C++ core libs
#include <functional>   // for std::function
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_meta/FieldReqs.hpp>     // for mundy::meta::FieldReqs

namespace mundy {

namespace meta {

/// \class FieldReqsFactory
/// \brief A factory containing generation routines for all of Mundy's \c FieldReqs.
///
/// The goal of \c FieldReqsFactory, as with most factories, is to provide an abstraction for case switches
/// between different methods. This factory is a bit different in that it always users to register different field types
/// such that a \c FieldReqs<FieldType> can then be generated based on the registered FieldType and its
/// corresponding string. This allows us to generate field requirements with custom types. Most importantly, it enables
/// users to add their own trivially copyable field types without modifying Mundy's source code.
///
/// The current set of registered field types and their string corresponding identifier is:
///  - SHORT              -> short
///  - UNSIGNED_SHORT     -> unsigned short
///  - INT                -> int
///  - UNSIGNED_INT       -> unsigned int
///  - LONG               -> long
///  - UNSIGNED_LONG      -> unsigned long
///  - LONG_LONG          -> long long
///  - UNSIGNED_LONG_LONG -> unsigned long long
///  - FLOAT              -> float
///  - DOUBLE             -> double
///  - LONG_DOUBLE        -> long double
///  - COMPLES_FLOAT      -> std::complex<float> // TODO(stk): Probably not right
///  - COMPLEX_DOUBLE     -> std::complex<double> // TODO(stk): Probably not right
///
/// \note This factory does not store an instance of \c FieldReqs; rather, it stores maps from a string to some
/// of \c FieldReqs's static member functions.
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design (albeit with heavy modifications).
/// https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
class FieldReqsFactory {
 public:
  //! \name Typedefs
  //@{

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// \c FieldReqsBase.
  using NewFieldReqsGenerator = std::function<std::shared_ptr<FieldReqsBase>(const Teuchos::ParameterList&)>;

  /// \brief A function type that produces a Teuchos::ParameterList instance.
  using NewDefaultParamsGenerator = std::function<Teuchos::ParameterList()>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of field types this factory recognizes.
  static size_t get_number_of_valid_field_types() {
    return get_instance_generator_map().size();
  }

  /// \brief Get if the provided field type string is valid or not
  /// \param field_type_string [in] A field type string that may or may not correspond to a registered field type.
  static bool is_valid_field_type_string(const std::string& field_type_string) {
    return get_instance_generator_map().count(field_type_string) != 0;
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Register a new field type with the given string. This type must be trivially copyable.
  /// \param field_type_string [in] The field type string to associate with \c FieldTypeToRegister.
  template <typename FieldTypeToRegister,
            std::enable_if_t<std::is_trivially_copyable<FieldTypeToRegister>::value, bool> = true>
  void register_new_field_type(const std::string& field_type_string) {
    MUNDY_THROW_REQUIRE(
        is_valid_field_type_string(field_type_string), std::invalid_argument,
        std::string("FieldReqsFactory: The provided field type string ") + field_type_string + " already exists.");
    get_instance_generator_map().insert(
        std::make_pair(field_type_string, FieldReqs<FieldTypeToRegister>::create_new_instance));
  }

  /// \brief Generate a new instance of a registered \c FieldReqs.
  ///
  /// The registered \c FieldReqs accessed by this function is fetched based on the provided field type string.
  /// This field type string must be valid; that is, \c is_valid_field_type_string(field_type_string) must return true.
  /// To register a \c FieldReqs with this factory, use the provided \c register_new_field_type function.
  ///
  /// \param field_type_string [in] A field type string correspond to a registered field type.
  ///
  /// \param parameter_list [in] Optional list of parameters for setting up this class. A default parameter list is
  /// accessible via \c get_valid_params.
  static std::shared_ptr<FieldReqsBase> create_new_instance(const std::string& field_type_string,
                                                            const Teuchos::ParameterList& parameter_list) {
    return get_instance_generator_map()[field_type_string](parameter_list);
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief A map from a string to a function for generating a new \c FieldReqs.
  using InstanceGeneratorMap = std::map<std::string, NewFieldReqsGenerator>;
  //@}

  //! \name Attributes
  //@{
  static InstanceGeneratorMap& get_instance_generator_map() {
    // Static: One and the same instance for all function calls.
    static InstanceGeneratorMap instance_generator_map;
    return instance_generator_map;
  }
  //@}

  //! \name Friends
  //@{

  /// \brief Registratrion of new types is done through \c FieldReqsRegistry.
  /// This process requires friendship <3.
  template <typename AnyFieldType, std::enable_if_t<std::is_trivially_copyable<AnyFieldType>::value, bool> EnableIfType>
  friend class FieldReqsRegistry;
  //@}
};  // FieldReqsFactory

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQSFACTORY_HPP_
