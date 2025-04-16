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

#ifndef MUNDY_AGENTS_AGENTREGISTRY_HPP_
#define MUNDY_AGENTS_AGENTREGISTRY_HPP_

/// \file Registry.hpp
/// \brief Declaration of the Registry class

// C++ core libs
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of
#include <utility>      // for std::pair

// Mundy libs
#include <mundy_agents/HierarchyOfAgents.hpp>   // for mundy::agents::HierarchyOfAgents
#include <mundy_core/attribute_unused.hpp>  // for MUNDY_ATTRIBUTE_UNUSED

/// \brief A helper macro for registering a \c Agent with \c HierarchyOfAgents.
///
/// This macro is used to register a \c Agent with \c HierarchyOfAgents. The macro should be
/// used in the following way:
///
/// \code{.cpp}
/// MUNDY_REGISTER_AGENT(ClassToRegister)
/// \endcode
///
/// There are some important notes about proper use of this macro:
///
/// - Registration in Global Scope: The registration should typically be done in the global scope, not inside any
/// function (including main()). This is because if the registration is done inside a function, it will not happen until
/// that function is called, which could be after main() starts.
///
/// - Registration in Source Files: Best practice is to perform registration in a source file, not a header file. This
/// is because if the registration is done in a header file, it will be registered in every translation unit that
/// includes that header file. This can lead to multiple registrations of the same class, which will cause an error.
/// Using header guards will not prevent this issue.
///
/// - No Dependency on Other Static Variables in Registration: Since C++ doesn't guarantee an order of initialization
/// for static variables across different translation units, you should make sure that the registration of a agent
/// type does not depend on the initialization of other static variables. If it does, those variables might not be
/// initialized at the time the agent type is registered.
///
/// - No Duplicated Names: Each agent type must be registered (to a certain MetaFactory) with a unique
/// identifier. If two classes are registered with the same identifier, only the first one will actually be registered,
/// and subsequent registrations will throw an error. While allowing overwriting of existing registrations is possible,
/// we chose to not implement such functionality because it could lead to unexpected behavior. Specifically, when using
/// static initialization across multiple translation units, there is no guaranteed order of initialization. This means
/// that if two classes are registered with the same identifier, it is not guaranteed which one will be registered
/// first.
///
/// As long as these points are followed, the registration of agent type subclasses should occur before main()
/// starts.
///
/// \param ClassToRegister A class derived from \c Agent that we wish to register.
#define TOKENPASTE(x, y) x##y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)
#define MUNDY_REGISTER_AGENT(... /* ClassToRegister */)                              \
  static volatile bool TOKENPASTE2(__registered__, __LINE__) MUNDY_ATTRIBUTE_UNUSED = \
      mundy::agents::HierarchyOfAgents::register_new_class<__VA_ARGS__>();

#endif  // MUNDY_AGENTS_AGENTREGISTRY_HPP_
