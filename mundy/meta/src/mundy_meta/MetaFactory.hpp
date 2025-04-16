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

#ifndef MUNDY_META_METAFACTORY_HPP_
#define MUNDY_META_METAFACTORY_HPP_

/// \file MetaFactory.hpp
/// \brief Declaration of the MetaFactory class

// External
#include <fmt/format.h>  // for fmt::format

// C++ core libs
#include <functional>  // for std::function
#include <iostream>
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <set>          // for std::set
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList
#include <stk_mesh/base/Part.hpp>     // for stk::mesh::Part

// Mundy libs
#include <mundy_core/StringLiteral.hpp>                       // for mundy::core::StringLiteral
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_mesh/BulkData.hpp>                            // for mundy::mesh::BulkData
#include <mundy_mesh/fmt_stk_types.hpp>                       // adds fmt::format for stk types
#include <mundy_meta/HasMeshReqsAndIsRegisterable.hpp>        // for mundy::meta::HasMeshReqsAndIsRegisterable
#include <mundy_meta/MeshReqs.hpp>                            // for mundy::meta::MeshReqs
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface

namespace mundy {

namespace meta {

/// \brief Concepts for checking if a class has the desired registration value wrapper interface: defines a type alias
/// \c T::Type and a function \c registration_value_wrapper.value(). \c T::Type must be the return type of \c
/// registration_value_wrapper.value().
template <typename T>
concept IsValidRegistrationValueWrapper = requires(T registration_value_wrapper) {
  typename T::Type;
  { registration_value_wrapper.value() } -> std::same_as<typename T::Type>;
};  // IsValidRegistrationValueWrapper

//! \name Type specializations for a MetaFactory with string registration types
//@{

/// \brief A class that providers a non-template type-compatable wrapper for strings.
///
/// Designed to satisfy the requirements of \c MetaFactory's IsValidRegistrationValueWrapper concept.
/// @tparam StrSize
template <size_t StrSize>
struct RegistrationStringValueWrapper {
  using Type = std::string;

  /// \brief Default constructor that initializes the string literal to an empty string.
  KOKKOS_INLINE_FUNCTION
  constexpr RegistrationStringValueWrapper() : value_{} {
  }

  /// \brief Constructor that copies the string literal into the struct.
  /// \param str The string literal to copy
  KOKKOS_INLINE_FUNCTION
  constexpr explicit RegistrationStringValueWrapper(const char (&str)[StrSize]) {
    copy_char_arrays(std::make_index_sequence<StrSize>(), str, value_);
  }

  /// \brief Convert the string literal to a std::string.
  std::string to_string() const {
    return std::string(value_);
  }

  std::string value() const {
    return to_string();
  }

  /// \brief The string literal's content.
  char value_[StrSize];

  /// \brief The string literal's size.
  static constexpr size_t size = StrSize;

 private:
  /// \brief Deep copy the first (Is...) characters of the source array into the destination array.
  template <size_t... Is, size_t SourceSize, size_t DestSize>
  KOKKOS_INLINE_FUNCTION static constexpr void copy_char_arrays(std::index_sequence<Is...>,
                                                                const char (&source)[SourceSize],
                                                                char (&dest)[DestSize]) {
    ((dest[Is] = source[Is]), ...);
  }
};  // RegistrationStringValueWrapper

/// \brief Non-member equality operator for comparing two RegistrationStringValueWrappers.
/// \param lhs The left-hand side of the comparison
/// \param rhs The right-hand side of the comparison
template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const RegistrationStringValueWrapper<N>& lhs,
                                                 const RegistrationStringValueWrapper<N>& rhs) {
  for (size_t i = 0; i < N; ++i) {
    if (lhs.value_[i] != rhs.value_[i]) {
      return false;
    }
  }
  return true;
}

/// \brief Non-member << operator for printing a RegistrationStringValueWrapper.
/// \param os The output stream to print to
/// \param str The RegistrationStringValueWrapper to print
template <size_t N>
std::ostream& operator<<(std::ostream& os, const RegistrationStringValueWrapper<N>& str) {
  os << str.to_string();
  return os;
}

/// \brief A helper function for generating a \c RegistrationStringValueWrapper from a string.
/// @tparam StrSize
/// \param str The string to wrap.
///
/// Usage example (also works inside of a template):
/// \code{.cpp}
/// auto registration_string = make_registration_string("MY_REGISTRATION_STRING");
/// \endcode
template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr RegistrationStringValueWrapper<N> make_registration_string(const char (&str)[N]) {
  return RegistrationStringValueWrapper<N>(str);
}

/// \class MetaFactory
/// \brief A factory containing generation routines for classes that have mesh requirements and are registerable.
///
/// The goal of \c MetaFactory, as with most factories, is to provide an abstraction for case switches between
/// different methods. This factory is a bit different in that it always users to register new classes (that match the
/// desired interface) and associate them with corresponding keys. These classes can then be fetched based using their
/// registration id. Most importantly, this enables users to register their own derived classes without modifying
/// Mundy's source code.
///
/// It's important to note that the static members of this factory will be shared between any \c MetaFactory
/// with the same set of template parameters. As a result, we can create a new factory with its own set of registered
/// classes by simply changing the \c RegistryIdentifier. For methods that should be globally accessible, we offer a
/// GlobalMetaFactory type specialization.
///
/// Any class that wishes to be registered with this factory must must implement the following static interface. Don't
/// worry, if your class fails to meet one of these requirements, register_new_class will throw a human-readable compile
/// time error telling you which functions you need to implement/modify. The specific signatures of these functions are
/// given below.
///
/// \code{.cpp}
/// // The type of this class's polymorphic base type (e.g. MetaKernel).
/// using PolymorphicBaseType = ...;
///
/// // Get the requirements generator for this class. May be nullptr.
/// static std::shared_ptr<MeshReqs> get_mesh_requirements();
///
/// // Get the valid fixed parameters for this class and their defaults.
/// static Teuchos::ParameterList get_valid_fixed_params();
///
/// // Get the valid mutable parameters for this class and their defaults.
/// static Teuchos::ParameterList get_valid_mutable_params();
///
/// // Get the new class generator for this class. May be nullptr.
/// static std::shared_ptr<PolymorphicBaseType> create_new_instance(mundy::mesh::BulkData* const bulk_data_ptr, const
///                                                                 Teuchos::ParameterList& fixed_params);
/// \endcode
///
/// \note Credit where credit is due: The design for this class originates from Andreas Zimmerer and his
/// self-registering types design (albeit with heavy modifications).
/// https://www.jibbow.com/posts/cpp-header-only-self-registering-types/
///
/// \tparam PolymorphicBaseType_t A polymorphic base type shared by each registered class.
/// \tparam RegistrationValueWrapperType_t A wrapper type for the registration value.
/// \tparam registration_value_wrapper A wrapper for the registration value. \c registration_value_wrapper::Type must be
/// the return type of \c registration_value_wrapper.value().
template <typename PolymorphicBaseType_t, RegistrationStringValueWrapper registration_value_wrapper>
class MetaFactory {
 public:
  //! \name Typedefs
  //@{

  using PolymorphicBaseType = PolymorphicBaseType_t;
  using RegistrationType = typename std::decay_t<decltype(registration_value_wrapper)>::Type;

  /// \brief A function type that takes a parameter list and produces a shared pointer to an object derived from
  /// class.
  using NewClassGenerator =
      std::function<std::shared_ptr<PolymorphicBaseType>(mundy::mesh::BulkData* const, const Teuchos::ParameterList&)>;

  /// \brief A function type that takes a parameter list and produces a vector of shared pointers to PartReqs
  /// instances.
  using NewRequirementsGenerator = std::function<std::shared_ptr<MeshReqs>(const Teuchos::ParameterList&)>;

  /// \brief A function type that returns a Teuchos::ParameterList.
  using NewValidParamsGenerator = std::function<Teuchos::ParameterList()>;
  //@}

  //! \name Getters
  //@{

  /// \brief Get the registration id for this factory. If two factories have the same registration id, they will share
  /// the same set of registered classes.
  static RegistrationType get_registration_id() {
    return registration_value_wrapper.value();
  }

  /// \brief Get the number of classes this factory recognizes.
  static int num_registered_classes() {
    return static_cast<int>(get_internal_keys().size());
  }

  /// \brief Get the set of all registered keys.
  static std::vector<RegistrationType> get_keys() {
    return get_internal_keys();
  }

  /// \brief Get the set of all registered keys as a string.
  static std::string get_keys_as_string() {
    std::string keys_as_string;
    for (const auto& key : get_internal_keys()) {
      keys_as_string += std::string(key);
      keys_as_string += ", ";
    }
    return keys_as_string;
  }

  /// \brief Get if the provided key is valid or not.
  /// \param key [in] A key that may or may not correspond to a registered class.
  static bool is_valid_key(const RegistrationType& key) {
    return get_instance_generator_map().count(key) != 0;
  }

  /// \brief Get the requirements that this a registered class imposes upon each part.
  ///
  /// The set of part requirements returned by this function are meant to encode the assumptions made by this class
  /// with respect to the structure, topology, and fields of the STK mesh. These assumptions may vary
  /// based on parameters in the \c fixed_params but not the \c mutable_params.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  /// \param fixed_params [in] Optional list of fixed parameters for setting up this class. A default fixed
  /// parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<MeshReqs> get_mesh_requirements(const RegistrationType& key,
                                                         const Teuchos::ParameterList& fixed_params) {
    return get_requirement_generator_map()[key](fixed_params);
  }

  /// \brief Get the valid fixed parameters for the given key.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static Teuchos::ParameterList get_valid_fixed_params(const RegistrationType& key) {
    MUNDY_THROW_REQUIRE(is_valid_key(key), std::invalid_argument,
                        fmt::format("MetaFactory: The provided key {} is not valid.", key));
    return get_valid_fixed_params_generator_map()[key]();
  }

  /// \brief Get the valid mutable parameters for the given key.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  static Teuchos::ParameterList get_valid_mutable_params(const RegistrationType& key) {
    MUNDY_THROW_REQUIRE(is_valid_key(key), std::invalid_argument,
                        fmt::format("MetaFactory: The provided key {} is not valid.", key));
    return get_valid_mutable_params_generator_map()[key]();
  }
  //@}

  //! \name Actions
  //@{

  /// \brief Default constructor.
  MetaFactory() = default;

  /// \brief Reset the factory to its initial state.
  ///
  /// This function removes all registered classes and clears all internal data structures.
  static void reset() {
    get_internal_keys().clear();
    get_instance_generator_map().clear();
    get_requirement_generator_map().clear();
    get_valid_fixed_params_generator_map().clear();
    get_valid_mutable_params_generator_map().clear();
  }

  /// \brief Register a new class. The key for the class is determined by its class identifier.
  template <typename ClassToRegister>
  static inline bool register_new_class(const RegistrationType& key) {
    // Check that the ClassToRegister has the desired interface.
    using Checker = HasMeshReqsAndIsRegisterable<ClassToRegister>;
    static_assert(Checker::has_get_mesh_requirements,
                  "MetaFactory: The class to register doesn't have the correct get_mesh_requirements function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_get_valid_fixed_params,
                  "MetaFactory: The class to register doesn't have the correct "
                  "get_valid_fixed_params function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_get_valid_mutable_params,
                  "MetaFactory: The class to register doesn't have the correct "
                  "get_valid_mutable_params function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_polymorphic_base_type,
                  "MetaFactory: The class to register doesn't have a PolymorphicBaseType type alias .\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");
    static_assert(Checker::has_create_new_instance,
                  "MetaFactory: The class to register doesn't have the correct create_new_instance function.\n"
                  "See the documentation of MetaFactory for more information about the expected interface.");

    // Register the class.
    MUNDY_THROW_REQUIRE(!is_valid_key(key), std::invalid_argument,
                        fmt::format("MetaFactory: The provided key {} already exists.", key));
    get_internal_keys().push_back(key);
    get_instance_generator_map().insert(std::make_pair(key, ClassToRegister::create_new_instance));
    get_requirement_generator_map().insert(std::make_pair(key, ClassToRegister::get_mesh_requirements));
    get_valid_fixed_params_generator_map().insert(std::make_pair(key, ClassToRegister::get_valid_fixed_params));
    get_valid_mutable_params_generator_map().insert(std::make_pair(key, ClassToRegister::get_valid_mutable_params));

    return true;
  }

  /// \brief Generate a new instance of a registered class.
  ///
  /// The registered class accessed by this function is fetched based on the provided key. This key must be
  /// valid; that is, is_valid_key(key) must return true. To register a class with this factory, use the
  /// provided \c register_new_class function.
  ///
  /// \param key [in] A key corresponding to a registered class.
  ///
  /// \param fixed_params [in] Optional list of parameters for setting up this class. A
  /// default parameter list is accessible via \c get_valid_fixed_params.
  static std::shared_ptr<PolymorphicBaseType> create_new_instance(const RegistrationType& key,
                                                                  mundy::mesh::BulkData* const bulk_data_ptr,
                                                                  const Teuchos::ParameterList& fixed_params) {
    return get_instance_generator_map()[key](bulk_data_ptr, fixed_params);
  }
  //@}

 private:
  //! \name Typedefs
  //@{

  /// \brief A set of keys.
  using SetOfKeys = std::vector<RegistrationType>;

  /// \brief A map from key to a function for generating a new class.
  using InstanceGeneratorMap = std::map<RegistrationType, NewClassGenerator>;

  /// \brief A map from key to a function for generating a class's mesh requirements.
  using RequirementGeneratorMap = std::map<RegistrationType, NewRequirementsGenerator>;

  /// \brief A map from key to a function for generating a class's part default requirements.
  using ValidParamsGeneratorMap = std::map<RegistrationType, NewValidParamsGenerator>;
  //@}

  //! \name Internal getters
  //@{
  static SetOfKeys& get_internal_keys() {
    // Static: One and the same instance for all function calls.
    static SetOfKeys keys;
    return keys;
  }

  static InstanceGeneratorMap& get_instance_generator_map() {
    // Static: One and the same instance for all function calls.
    static InstanceGeneratorMap instance_generator_map;
    return instance_generator_map;
  }

  static RequirementGeneratorMap& get_requirement_generator_map() {
    // Static: One and the same instance for all function calls.
    static RequirementGeneratorMap requirement_generator_map;
    return requirement_generator_map;
  }

  static ValidParamsGeneratorMap& get_valid_fixed_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static ValidParamsGeneratorMap valid_fixed_params_generator_map;
    return valid_fixed_params_generator_map;
  }

  static ValidParamsGeneratorMap& get_valid_mutable_params_generator_map() {
    // Static: One and the same instance for all function calls.
    static ValidParamsGeneratorMap valid_mutable_params_generator_map;
    return valid_mutable_params_generator_map;
  }
  //@}
};  // MetaFactory

/// \brief A type specialization of \c MetaFactory that uses a string as the registration identifier. See \c MetaFactory
/// for details.
/// @tparam PolymorphicBaseType
/// @tparam registration_string_value_wrapper
///
/// To generate a \c MetaFactory that uses a string as the registration identifier, use the following syntax:
/// \code{.cpp}
/// using MyMetaFactory = StringBasedMetaFactory<MyPolymorphicBaseType,
/// make_registration_string("MY_REGISTRATION_STRING")>;
/// \endcode
template <typename PolymorphicBaseType, RegistrationStringValueWrapper registration_string_value_wrapper>
using StringBasedMetaFactory = MetaFactory<PolymorphicBaseType, registration_string_value_wrapper>;
//@}

}  // namespace meta

}  // namespace mundy

#endif  // MUNDY_META_METAFACTORY_HPP_
