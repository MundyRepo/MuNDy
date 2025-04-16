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

#ifndef MUNDY_CORE_STRINGLITERAL_HPP_
#define MUNDY_CORE_STRINGLITERAL_HPP_

/// \file StringLiteral.cpp
/// \brief Declaration of our StringLiteral class type.

// C++ core
#include <algorithm>
#include <concepts>
#include <iostream>
#include <type_traits>

// Kokkos
#include <Kokkos_Core.hpp>

static_assert(__cplusplus >= 202002L, "This code requires C++20 or later");

namespace mundy {

namespace core {

/// \brief Literal class type that wraps a constant expression string.
///
/// The StringLiteral struct allows templates to accept constant strings,
/// effectively turning them into compile-time constants. This can result in
/// performance benefits as computations using these strings can be performed at
/// compile time.
///
/// How It Works:
/// The `constexpr` keyword is used to ensure that all computations involving
/// StringLiteral are done at compile time. The StringLiteral struct takes in the
/// string and its size as template parameters and stores them. The string can
/// then be accessed with the `value` member, and the size with the `size` member.
///
/// Example Usage:
/// \code{.cpp}
/// template <StringLiteral StrLit>
/// void Print() {
///   // The size of the string is available as a constant expression.
///   constexpr auto size = StrLit.size;
///
///   // and so is the string's content.
///   constexpr auto contents = StrLit.value;
///
///   std::cout << "Size: " << size << ", Contents: " << contents << std::endl;
/// }
///
/// int main() {
///   Print<make_string_literal("literal string")>();  // Prints "Size: 15, Contents: literal string"
/// }
/// \endcode
///
/// Credit where credit is due: This design entirely originates from Kevin Hartman's Passing String Literals as Template
/// Parameters in C++20 blog post:
/// https://ctrpeach.io/posts/cpp20-string-literal-template-parameters/#:~:text=This%20would%20work%20by%20wrapping,e.g.%20char%5BN%5D%20).
///
/// \tparam StrSize The size of the string literal
template <size_t StrSize>
struct StringLiteral {
  /// \brief Default constructor that initializes the string literal to an empty string.
  KOKKOS_INLINE_FUNCTION
  constexpr StringLiteral() : value{} {
  }

  /// \brief Constructor that copies the string literal into the struct.
  /// \param str The string literal to copy
  KOKKOS_INLINE_FUNCTION
  constexpr explicit StringLiteral(const char (&str)[StrSize]) {
    copy_char_arrays(std::make_index_sequence<StrSize>(), str, value);
  }

  /// \brief Convert the string literal to a std::string.
  std::string to_string() const {
    return std::string(value);
  }

  /// \brief The string literal's content.
  char value[StrSize];

  /// \brief The string literal's size.
  static constexpr size_t size = StrSize;

  /// \brief Compile-time concatenation with another StringLiteral.
  /// \param rhs The right-hand side of the concatenation
  /// \return A new StringLiteral with the combined content of the two input StringLiterals
  template <size_t OtherStrSize>
  KOKKOS_INLINE_FUNCTION constexpr StringLiteral<StrSize + OtherStrSize - 1> operator+(
      const StringLiteral<OtherStrSize>& rhs) const {
    StringLiteral<StrSize + OtherStrSize - 1> result;
    copy_char_arrays(std::make_index_sequence<StrSize - 1>(), value, result.value);  // Don't copy the null terminator
    copy_char_arrays_shift(std::make_index_sequence<OtherStrSize>(), rhs.value, result.value, 0, StrSize - 1);
    return result;
  }

  /// \brief Compile-time concatenation with a char array.
  /// \param rhs The right-hand side of the concatenation
  /// \return A new StringLiteral with the combined content of the input StringLiteral and char array
  template <size_t OtherStrSize>
  KOKKOS_INLINE_FUNCTION constexpr StringLiteral<StrSize + OtherStrSize - 1> operator+(
      const char (&rhs)[OtherStrSize]) const {
    StringLiteral<StrSize + OtherStrSize - 1> result;
    copy_char_arrays(std::make_index_sequence<StrSize - 1>(), value, result.value);
    copy_char_arrays_shift(std::make_index_sequence<OtherStrSize>(), rhs, result.value, 0, StrSize - 1);
    return result;
  }

 private:
  /// \brief Deep copy the first (Is...) characters of the source array into the destination array.
  template <size_t... Is, size_t SourceSize, size_t DestSize>
  KOKKOS_INLINE_FUNCTION static constexpr void copy_char_arrays(std::index_sequence<Is...>,
                                                                const char (&source)[SourceSize],
                                                                char (&dest)[DestSize]) {
    ((dest[Is] = source[Is]), ...);
  }

  /// \brief Deep copy with shift the first (Is... + source_shift) from the source into the destination array +
  /// destination shift
  template <size_t... Is, size_t SourceSize, size_t DestSize>
  KOKKOS_INLINE_FUNCTION static constexpr void copy_char_arrays_shift(std::index_sequence<Is...>,
                                                                      const char (&source)[SourceSize],
                                                                      char (&dest)[DestSize], size_t source_shift,
                                                                      size_t dest_shift) {
    ((dest[Is + dest_shift] = source[Is + source_shift]), ...);
  }
};  // StringLiteral

/// \brief Non-member equality operator for comparing two StringLiterals.
/// \param lhs The left-hand side of the comparison
/// \param rhs The right-hand side of the comparison
template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const StringLiteral<N>& lhs, const StringLiteral<N>& rhs) {
  for (size_t i = 0; i < N; ++i) {
    if (lhs.value[i] != rhs.value[i]) {
      return false;
    }
  }
  return true;
}

/// \brief Non-member << operator for printing a StringLiteral.
/// \param os The output stream to print to
/// \param str The StringLiteral to print
template <size_t N>
std::ostream& operator<<(std::ostream& os, const StringLiteral<N>& str) {
  os << str.to_string();
  return os;
}

/// \brief Helper function for creating a StringLiteral.
///
/// This function is used to create a StringLiteral without having to specify the template parameters.
///
/// \param str The string literal to copy
/// \return A StringLiteral with the same content as the input string literal
template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr StringLiteral<N> make_string_literal(const char (&str)[N]) {
  return StringLiteral<N>(str);
}

}  // namespace core

}  // namespace mundy

//! \name Helpers for determining if an object is a string literal (type traits fails for string literals)
//@{

#define MUNDY_IS_CHAR_ARRAY(x) ([&]<class __T = char>() constexpr {       \
    return std::is_same_v<__T const (&)[sizeof(x)], decltype(x)>;    \
}())

#define MUNDY_IS_STRING_LITERAL(x)                                          \
  ([&]<class __mundy_T = char>() constexpr {                                \
    return std::is_same_v<__mundy_T const(&)[sizeof(x)], decltype(x)> &&    \
           requires { std::type_identity_t<__mundy_T[sizeof(x) + 1]>{x}; }; \
  }())

#define MUNDY_IS_OUR_STRING_LITERAL(x) \
  (std::is_same_v<mundy::core::StringLiteral<sizeof(x)>, std::decay_t<decltype(x)>>)
//@}

#endif  // MUNDY_CORE_STRINGLITERAL_HPP_
