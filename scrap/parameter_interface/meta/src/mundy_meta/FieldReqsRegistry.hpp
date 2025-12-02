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

#ifndef MUNDY_META_FIELDREQSREGISTRY_HPP_
#define MUNDY_META_FIELDREQSREGISTRY_HPP_

/// \file FieldReqsRegistry.hpp
/// \brief Declaration of the FieldReqsRegistry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_trivially_copyable

// Mundy libs
#include <mundy_meta/FieldReqsFactory.hpp>  // for mundy::meta::FieldReqsFactory

namespace mundy {

namespace meta {

/// \class FieldReqsRegistry
/// \brief A class for registering \c FieldReqss within \c FieldReqsFactory.
///
/// All valid field types that can be passed to \c FieldReqs should be registered within the
/// \c FieldReqsFactory. This registry aids in the registration process.
///
/// \tparam FieldTypeToRegister A trivially copyable type to be registered with the \c FieldReqsFactory.
template <class FieldTypeToRegister,
          std::enable_if_t<std::is_trivially_copyable<FieldTypeToRegister>::value, bool> = true>
struct FieldReqsRegistry {
  //! \name Actions
  //@{

  /// \brief Register \c FieldTypeToRegister with the \c FieldReqsFactory.
  ///
  /// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
  /// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
  ///
  /// \param field_type_string [in] The field type string to associate with \c FieldTypeToRegister.
  static inline bool register_type(const std::string& field_type_string) {
    FieldReqsFactory::register_new_field_type<FieldTypeToRegister>(field_type_string);
    return true;
  }
  //@}

  //! \name Member variables
  //@{

  /// \brief A flag for if the given type has been registered with the \c FieldReqsFactory or not.
  static const bool is_registered;
  //@}
};  // FieldReqsRegistry

/// \brief Perform the static registration of the desired FieldReqs FieldTypes.
///
/// \note When the program is started, one of the first steps is to initialize static objects. Even if is_registered
/// appears to be unused, static storage duration guarantees that this variable won’t be optimized away.
// clang-format off
const bool FieldReqsRegistry<short>::is_registered = FieldReqsRegistry<short>::register_type("SHORT");
const bool FieldReqsRegistry<unsigned short>::is_registered = FieldReqsRegistry<unsigned short>::register_type("UNSIGNED_SHORT");
const bool FieldReqsRegistry<int>::is_registered = FieldReqsRegistry<int>::register_type("INT");
const bool FieldReqsRegistry<unsigned int>::is_registered = FieldReqsRegistry<unsigned int>::register_type("UNSIGNED_INT");
const bool FieldReqsRegistry<long>::is_registered = FieldReqsRegistry<long>::register_type("LONG");
const bool FieldReqsRegistry<unsigned long>::is_registered = FieldReqsRegistry<unsigned long>::register_type("UNSIGNED_LONG");
const bool FieldReqsRegistry<long long>::is_registered = FieldReqsRegistry<long long>::register_type("LONG_LONG");
const bool FieldReqsRegistry<unsigned long long>::is_registered = FieldReqsRegistry<unsigned long long>::register_type("UNSIGNED_LONG_LONG");
const bool FieldReqsRegistry<float>::is_registered = FieldReqsRegistry<float>::register_type("FLOAT");
const bool FieldReqsRegistry<double>::is_registered = FieldReqsRegistry<double>::register_type("DOUBLE");
const bool FieldReqsRegistry<long double>::is_registered = FieldReqsRegistry<long double>::register_type("LONG_DOUBLE");
const bool FieldReqsRegistry<std::complex<float>>::is_registered = FieldReqsRegistry<std::complex<float>>::register_type("COMPLES_FLOAT");
const bool FieldReqsRegistry<std::complex<double>>::is_registered = FieldReqsRegistry<std::complex<double>>::register_type("COMPLEX_DOUBLE");
// clang-format on

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_FIELDREQSREGISTRY_HPP_
