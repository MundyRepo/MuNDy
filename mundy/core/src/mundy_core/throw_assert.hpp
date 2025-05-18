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

#ifndef MUNDY_CORE_THROW_ASSERT_HPP_
#define MUNDY_CORE_THROW_ASSERT_HPP_

/// \file MundyAssert.cpp
/// \brief Declaration of our assertion macros

// C++ core
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::is_string_literal and mundy::core::StringLiteral

#define MUNDY_STRINGIFY(x) MUNDY_STRINGIFY2(x)
#define MUNDY_STRINGIFY2(x) #x
#define MUNDY_LINE_STRING MUNDY_STRINGIFY(__LINE__)

#ifdef __PRETTY_FUNCTION__
#define MUNDY_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#define MUNDY_FUNCTION_NAME __FUNCSIG__
#else
#define MUNDY_FUNCTION_NAME __func__
#endif

namespace mundy {

template <size_t AssertionStringSize, size_t FileStringSize, size_t LineStringSize>
std::string get_throw_require_host_string(
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const std::string& message_to_print,                                                           //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  std::ostringstream message_to_print_ostring_stream;
  message_to_print_ostring_stream << "Assertion (" << assertion_string << ") failed."
                                  << "\nFile: " << file_string << "\nLine: " << line_string
                                  << "\nMessage: " << message_to_print;
  return message_to_print_ostring_stream.str();
}

template <size_t AssertionStringSize, size_t MessageStringSize, size_t FileStringSize, size_t LineStringSize>
std::string get_throw_require_host_string(
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const char (&message_to_print)[MessageStringSize],                                             //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  return get_throw_require_host_string(assertion_string, std::string(message_to_print), file_string, line_string);
}

template <size_t AssertionStringSize, size_t MessageStringSize, size_t FileStringSize, size_t LineStringSize>
std::string get_throw_require_host_string(
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const core::StringLiteral<MessageStringSize>& message_to_print,                                //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  return get_throw_require_host_string(assertion_string, message_to_print.value, file_string, line_string);
}

template <size_t AssertionStringSize, size_t MessageStringSize, size_t FileStringSize, size_t LineStringSize>
KOKKOS_INLINE_FUNCTION constexpr auto get_throw_require_device_string(
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const char (&message_to_print)[MessageStringSize],                                             //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  return core::make_string_literal("Assertion (") + assertion_string + ") failed.\nFile: " + file_string +
         "\nLine: " + line_string + "\nMessage: " + core::make_string_literal(message_to_print);
}

template <size_t AssertionStringSize, size_t MessageStringSize, size_t FileStringSize, size_t LineStringSize>
KOKKOS_INLINE_FUNCTION constexpr auto get_throw_require_device_string(
    const core::StringLiteral<AssertionStringSize>& assertion_string,  //
    const core::StringLiteral<MessageStringSize>& message_to_print,    //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  return core::make_string_literal("Assertion (") + assertion_string + ") failed.\nFile: " + file_string +
         "\nLine: " + line_string + "\nMessage: " + message_to_print;
}

template <size_t AssertionStringSize, size_t MessageStringSize, size_t FileStringSize, size_t LineStringSize>
std::string get_throw_require_device_string(
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const std::string& message_to_print,                                                           //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  constexpr auto prefix = core::make_string_literal("Assertion (") + assertion_string +
                          ") failed.\nFile: " + file_string + "\nLine: " + line_string + "\nMessage: ";
  return prefix.to_string() + message_to_print;
}

template <typename ExceptionType, typename MessageStringType, size_t AssertionStringSize, size_t FileStringSize,
          size_t LineStringSize>
  requires std::is_base_of_v<std::exception, ExceptionType>
void throw_require(
    const bool assertion_value,                                                                    //
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const MessageStringType& message_to_print,                                                     //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  if (!assertion_value) {
    throw ExceptionType(get_throw_require_host_string(assertion_string, message_to_print, file_string, line_string));
  }
}

template <typename MessageStringType, size_t AssertionStringSize, size_t FileStringSize, size_t LineStringSize>
KOKKOS_INLINE_FUNCTION void abort_require(
    const bool assertion_value,                                                                    //
    const core::StringLiteral<AssertionStringSize>& assertion_string,                              //
    const MessageStringType& message_to_print,                                                     //
    const core::StringLiteral<FileStringSize>& file_string = core::make_string_literal(__FILE__),  //
    const core::StringLiteral<LineStringSize>& line_string = core::make_string_literal(MUNDY_LINE_STRING)) {
  if (!assertion_value) {
    if constexpr (std::is_same_v<std::remove_const<decltype(message_to_print)>, std::string>) {
      Kokkos::abort(
          get_throw_require_device_string(assertion_string, message_to_print, file_string, line_string).c_str());
    } else if constexpr (MUNDY_IS_CHAR_ARRAY(message_to_print) || MUNDY_IS_OUR_STRING_LITERAL(message_to_print)) {
      Kokkos::abort(
          get_throw_require_device_string(assertion_string, message_to_print, file_string, line_string).value);
    } else {
      // The message to print is not a string literal or a mundy::core::StringLiteral, but we can still print the
      // file and line information, which they can use to go read the message in the source code.
      Kokkos::abort(get_throw_require_device_string(assertion_string, "\nUnable to print user-specified message.",
                                                    file_string, line_string)
                        .value);
    }
  }
}

}  // namespace mundy

#define MUNDY_THROW_REQUIRE_HOST(assertion_to_test, exception_to_throw, message_to_print)                             \
  do {                                                                                                                \
    const bool __mundy_assertion_value = static_cast<bool>(assertion_to_test);                                        \
    constexpr auto __mundy_assertion_string = ::mundy::core::make_string_literal(MUNDY_STRINGIFY(assertion_to_test)); \
    ::mundy::throw_require<exception_to_throw>(__mundy_assertion_value, __mundy_assertion_string, message_to_print,   \
                                               ::mundy::core::make_string_literal(__FILE__),                          \
                                               ::mundy::core::make_string_literal(MUNDY_LINE_STRING));                \
  } while (false);

#define MUNDY_THROW_REQUIRE_DEVICE(assertion_to_test, exception_to_throw, message_to_print)                           \
  do {                                                                                                                \
    const bool __mundy_assertion_value = static_cast<bool>(assertion_to_test);                                        \
    constexpr auto __mundy_assertion_string = ::mundy::core::make_string_literal(MUNDY_STRINGIFY(assertion_to_test)); \
    ::mundy::abort_require(__mundy_assertion_value, __mundy_assertion_string, message_to_print,                       \
                           ::mundy::core::make_string_literal(__FILE__),                                              \
                           ::mundy::core::make_string_literal(MUNDY_LINE_STRING));                                    \
  } while (false);

/// \def MUNDY_THROW_REQUIRE
/// \brief Abort the code if the given assertion is false.
/// \note This macro will always test the assertion, regardless of whether NDEBUG is defined. Use it to enforce critical
/// requirements (hence the name).
///
/// Host and device compatible. The two differ heavily in that host can print the function name, while device cannot.
/// The host code will throw the given exception, while the device code will abort. The abort is unavoidable, as
/// device code cannot throw exceptions.
///
/// A comment about types: The message to print (on the host) can be any object that can be printed to an ostream. On
/// the device, the message to print must be a string literal or a mundy::core::StringLiteral. This is because the
/// device code must be able to print the message at compile time. If the message is not a string literal, the code will
/// throw a static_assert error.
///
/// \param assertion_to_test The assertion to test
/// \param exception_to_throw The exception to throw if the assertion is false (will only be thrown on the host)
/// \param message_to_print The message to print if the assertion is false
#define MUNDY_THROW_REQUIRE(assertion_to_test, exception_to_throw, message_to_print)                          \
  do {                                                                                                        \
    KOKKOS_IF_ON_HOST(MUNDY_THROW_REQUIRE_HOST(assertion_to_test, exception_to_throw, message_to_print);)     \
    KOKKOS_IF_ON_DEVICE(MUNDY_THROW_REQUIRE_DEVICE(assertion_to_test, exception_to_throw, message_to_print);) \
  } while (false);

/// \def MUNDY_THROW_ASSERT
/// \brief Throw an exception if the given assertion is false.
/// \note This macro is only compiled if NDEBUG is not defined. Well, that's a lie. Technically, we still use the
/// assertion_to_test but only its type and not its value. This avoids unused variable warnings and shouldn't have any
/// performance impact. TODO(palmerb4): Confirm this claim.
///
/// \param assertion_to_test The assertion to test
/// \param exception_to_throw The exception to throw if the assertion is false (will only be thrown on the host)
/// \param message_to_print The message to print if the assertion is false
#ifndef NDEBUG
#define MUNDY_THROW_ASSERT(assertion_to_test, exception_to_throw, message_to_print) \
  MUNDY_THROW_REQUIRE(assertion_to_test, exception_to_throw, message_to_print)
#else
#define MUNDY_THROW_ASSERT(assertion_to_test, exception_to_throw, message_to_print) \
  static_cast<void>(sizeof(assertion_to_test));
#endif

#endif  // MUNDY_CORE_THROW_ASSERT_HPP_
