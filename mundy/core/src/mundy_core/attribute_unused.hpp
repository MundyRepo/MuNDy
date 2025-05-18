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

#ifndef MUNDY_CORE_ATTRIBUTE_UNUSED_HPP_
#define MUNDY_CORE_ATTRIBUTE_UNUSED_HPP_

/// \file attribute_unused.cpp
/// \brief Declaration of our attribute_unused macro

/// \def MUNDY_HAVE_ATTRIBUTE
/// \brief A helper macro for checking if a compiler supports a given attribute.
///
/// A function-like feature checking macro that is a wrapper around
/// `__has_attribute`, which is defined by GCC 5+ and Clang and evaluates to a
/// nonzero constant integer if the attribute is supported or 0 if not.
///
/// It evaluates to zero if `__has_attribute` is not defined by the compiler.
///
/// GCC: https://gcc.gnu.org/gcc-5/changes.html
/// Clang: https://clang.llvm.org/docs/LanguageExtensions.html
#ifdef __has_attribute
#define MUNDY_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define MUNDY_HAVE_ATTRIBUTE(x) 0
#endif

/// \def MUNDY_ATTRIBUTE_UNUSED
/// \brief A helper macro for forcing the compiler to compile an unused variable.
///
/// This macro is repurposed from GTEST_ATTRIBUTE_UNUSED_ in gtest-port.h. It is used to force the compiler to compile a
/// variable that is otherwise unused. Use this annotation after a variable or parameter declaration to tell the
/// compiler the variable/parameter does not have to be used but should be compiled anyway. Example usage:
/// \code{.cpp}
//    MUNDY_ATTRIBUTE_UNUSED int foo = bar();
/// \endcode
#if MUNDY_HAVE_ATTRIBUTE(unused)
#define MUNDY_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define MUNDY_ATTRIBUTE_UNUSED
#endif

#endif  // MUNDY_CORE_ATTRIBUTE_UNUSED_HPP_
