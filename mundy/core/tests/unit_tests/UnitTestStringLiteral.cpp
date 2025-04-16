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

// External libs
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <iostream>
#include <stdexcept>  // for logic_error, invalid_argument, etc

// Mundy libs
#include <mundy_core/StringLiteral.hpp>  // for mundy::core::StringLiteral and mundy::core::make_string_literal

namespace mundy {

namespace core {

namespace {

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

  Type value() const {
    return to_string();
  }

  /// \brief The string literal's content.
  char value_[StrSize];

  /// \brief The string literal's size.
  static constexpr size_t size = StrSize;

  /// \brief Compile-time concatenation with another RegistrationStringValueWrapper.
  /// \param rhs The right-hand side of the concatenation
  /// \return A new RegistrationStringValueWrapper with the combined content of the two input
  /// RegistrationStringValueWrappers
  template <size_t OtherStrSize>
  KOKKOS_INLINE_FUNCTION constexpr RegistrationStringValueWrapper<StrSize + OtherStrSize> operator+(
      const RegistrationStringValueWrapper<OtherStrSize>& rhs) const {
    RegistrationStringValueWrapper<StrSize + OtherStrSize> result;
    copy_char_arrays(std::make_index_sequence<StrSize>(), value_, result.value_);
    copy_char_arrays_shift(std::make_index_sequence<OtherStrSize>(), rhs.value_, result.value_, 0, StrSize);
    return result;
  }

  /// \brief Compile-time concatenation with a char array.
  /// \param rhs The right-hand side of the concatenation
  /// \return A new RegistrationStringValueWrapper with the combined content of the input RegistrationStringValueWrapper
  /// and char array
  template <size_t OtherStrSize>
  KOKKOS_INLINE_FUNCTION constexpr RegistrationStringValueWrapper<StrSize + OtherStrSize> operator+(
      const char (&rhs)[OtherStrSize]) const {
    RegistrationStringValueWrapper<StrSize + OtherStrSize> result;
    copy_char_arrays(std::make_index_sequence<StrSize>(), value_, result.value_);
    copy_char_arrays_shift(std::make_index_sequence<OtherStrSize>(), rhs, result.value_, 0, StrSize);
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

/// \brief Helper function for creating a RegistrationStringValueWrapper.
///
/// This function is used to create a RegistrationStringValueWrapper without having to specify the template parameters.
///
/// \param str The string literal to copy
/// \return A RegistrationStringValueWrapper with the same content as the input string literal
template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr RegistrationStringValueWrapper<N> make_registration_string(const char (&str)[N]) {
  return RegistrationStringValueWrapper<N>(str);
}

/// \brief Use a string literal as a non-type template parameter
template <StringLiteral StrLit>
std::string lit_to_string() {
  std::cout << StrLit << std::endl;
  return StrLit.to_string();
}

/// \brief Use a string literal as a non-type template parameter
template <RegistrationStringValueWrapper StrLit>
std::string wrapper_to_string() {
  std::cout << StrLit << std::endl;
  return StrLit.to_string();
}

TEST(StringLiteral, StringLiteralWorks) {
  // Test that we can use a string literal as a non-type template parameter
  EXPECT_EQ(lit_to_string<make_string_literal("Hello, world!")>(), std::string("Hello, world!"));
  EXPECT_EQ(wrapper_to_string<make_registration_string("Hello, world!")>(), std::string("Hello, world!"));
}

}  // namespace

}  // namespace core

}  // namespace mundy
