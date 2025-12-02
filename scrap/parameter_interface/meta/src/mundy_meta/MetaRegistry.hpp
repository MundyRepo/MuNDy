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

#ifndef MUNDY_META_METAREGISTRY_HPP_
#define MUNDY_META_METAREGISTRY_HPP_

/// \file MetaRegistry.hpp
/// \brief Declaration of the MetaRegistry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::pair

// Mundy libs
#include <mundy_core/attribute_unused.hpp>                    // for MUNDY_ATTRIBUTE_UNUSED
#include <mundy_core/throw_assert.hpp>                        // for MUNDY_THROW_ASSERT
#include <mundy_meta/MetaFactory.hpp>                         // for mundy::meta::MetaMethodFactory
#include <mundy_meta/MetaKernel.hpp>                          // for mundy::meta::MetaKernel
#include <mundy_meta/MetaMethodSubsetExecutionInterface.hpp>  // for mundy::meta::MetaMethodSubsetExecutionInterface

namespace mundy {

namespace meta {

/// \class MetaRegistry
/// \brief A class for registering \c MetaMethods within \c MetaMethodFactory.
///
/// Most users shouldn't directly interface with this registry; instead, registration is performed using the provided
/// MUNDY_REGISTER_METACLASS macro. See the documentation for that macro for more information.
///
/// \tparam ClassToRegister A class derived from \c MetaMethodSubsetExecutionInterface that we wish to register.
/// \param FactoryToRegisterWith The \c MetaMethodFactory to register the class with.
template <typename ClassToRegister, typename FactoryToRegisterWith>
struct MetaRegistry {
  //! \name Member variable definitions
  //@{

  /// \brief A flag for if the given type has been registered with the \c MetaMethodFactory or not.
  static inline volatile const bool is_registered MUNDY_ATTRIBUTE_UNUSED = false;
  //@}
};  // MetaRegistry

}  // namespace meta

}  // namespace mundy

/// \brief A helper macro for checking if a \c MetaMethodSubsetExecutionInterface has been registered with the \c
/// MetaMethodFactory.
///
/// The macro should be used in the following way:
///
/// \code{.cpp}
/// MUNDY_IS_REGISTERED(ClassToCheck, FactoryToCheckWith)
/// \endcode
///
/// \note The second argument to this macro is supposed to be the \c MetaMethodFactory that the class should be
/// registered with. The reason we use the weird "... /* FactoryToCheckWith */" syntax is because we want to allow
/// FactoryToCheckWith to potentially be a templated class with multiple template arguments. In this case, the C++
/// macro system will interpret the comma in the template arguments as a macro argument separator, which is not what we
/// want. As a result, we need to use the "..." syntax to collect those additional arguments and sync them
/// together into the desired \c FactoryToCheckWith using \c __VA_ARGS__.
///
/// \note This macro used a lambda function to check if the class has been registered. This ensures that each use of
/// \c MUNDY_IS_REGISTERED does not create a new definition of \c is_registered, thereby avoiding multiple definition
/// errors.
///
/// \param ClassToCheck A class derived from \c MetaMethodSubsetExecutionInterface that we wish to check if it has been
/// registered. \param FactoryToCheckWith The \c MetaMethodFactory to check if the class has been registered with.
#define MUNDY_IS_REGISTERED(ClassToCheck, ... /* FactoryToCheckWith */) \
  ([]() -> bool { return mundy::meta::MetaRegistry<ClassToCheck, __VA_ARGS__>::is_registered; }())

/// \brief A helper macro for registering a \c MetaMethodSubsetExecutionInterface with the \c MetaMethodFactory.
///
/// This macro is used to register a \c MetaMethodSubsetExecutionInterface with the \c MetaMethodFactory. The macro
/// should be used in the following way:
///
/// \code{.cpp}
/// MUNDY_REGISTER_METACLASS(Key, ClassToRegister, FactoryToRegisterWith)
/// \endcode
///
/// There are some important notes about proper use of this macro:
///
/// - Registration in Global Scope: The registration should typically be done in the global scope, not inside any
/// function (including main()).
///
/// - Registration in Source Files: Best practice is to perform registration in a source file, not a header file. This
/// is because if the registration is done in a header file, it will be registered in every translation unit that
/// includes that header file. This can lead to multiple registrations of the same class, causing an error.
/// Using header guards will not prevent this issue.
///
/// - No Dependency on Other Static Variables in Registration: Since C++ doesn't guarantee an order of initialization
/// for static variables across different translation units, you should make sure that the registration of a MetaClass
/// does not depend on the initialization of other static variables. If it does, those variables might not be
/// initialized at the time the MetaClass is registered.
///
/// - No Duplicated Names: Each MetaClass must be registered (to a certain MetaFactory) with a unique
/// identifier. If two classes are registered with the same identifier, only the first one will actually be registered,
/// and subsequent registrations will throw an error. While allowing overwriting of existing registrations is possible,
/// we chose to not implement such functionality because it could lead to unexpected behavior. Specifically, when using
/// static initialization across multiple translation units, there is no guaranteed order of initialization. This means
/// that if two classes are registered with the same identifier, it is not guaranteed which one will be registered
/// first.
///
/// - No commas within the key: The key should not contain any commas, as this will cause the macro to interpret the
/// key as multiple arguments. The workaround is to just perform the template specialization manually.
///
/// \note The third argument to this macro is supposed to be the \c MetaMethodFactory that the class should be
/// registered with. The reason we use the weird "... /* FactoryToRegisterWith */" syntax is because we want to allow
/// FactoryToRegisterWith to potentially be a templated class with multiple template arguments. In this case, the C++
/// macro system will interpret the comma in the template arguments as a macro argument separator, which is not what we
/// want. As a result, we need to use the "..." syntax to collect those additional arguments and sync them
/// together into the desired \c FactoryToRegisterWith using \c __VA_ARGS__.
///
/// \param Key The key to register the class with. This key should be unique within the \c MetaMethodFactory.
/// \param ClassToRegister A class derived from \c MetaMethodSubsetExecutionInterface that we wish to register.
/// \param FactoryToRegisterWith The \c MetaMethodFactory to register the class with.
#define MUNDY_REGISTER_METACLASS(Key, ClassToRegister, ... /* FactoryToRegisterWith */)                      \
  template <>                                                                                                \
  struct mundy::meta::MetaRegistry<ClassToRegister, __VA_ARGS__> {                                           \
    static inline volatile const bool is_registered = __VA_ARGS__::register_new_class<ClassToRegister>(Key); \
  };

#endif  // MUNDY_META_METAREGISTRY_HPP_
