// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
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
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace core {

namespace {

//! \name Throw assert tests
//@{

TEST(ThrowAssert, Predicates) {
  // These are all the language features we need to be true for MUNDY_THROW_REQUIRE to operate as expected.

  // Check that is_string_literal works as expected
  static_assert(MUNDY_IS_STRING_LITERAL("string literal"));
  static_assert(!MUNDY_IS_STRING_LITERAL(42));
  constexpr auto a = "a";
  static_assert(!MUNDY_IS_STRING_LITERAL(a));

  // Check that is_mundy_string_literal works as expected
  static_assert(!MUNDY_IS_OUR_STRING_LITERAL("string literal"));
  constexpr auto b = make_string_literal("b");
  static_assert(MUNDY_IS_OUR_STRING_LITERAL(b));

  // Check that host code is host code and device code is device code
  std::string space;
  KOKKOS_IF_ON_HOST(space = "  on host"; std::cout << space << std::endl;)
  KOKKOS_IF_ON_DEVICE(space = "  on device"; std::cout << space << std::endl;)
  EXPECT_THAT(space, ::testing::HasSubstr("on host"));
}

TEST(ThrowAssert, DoesNotThrowWhenTrue) {
  // Check that throw assert does not throw when the condition is true

  // Check that throw assert does not throw a logic error
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::logic_error, "Logic error"));

  // Check that throw assert does not throw an invalid argument
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::invalid_argument, "Invalid argument"));

  // Check that throw assert does not throw a runtime error
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::runtime_error, "Runtime error"));

  // Check that throw assert does not throw a domain error
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::domain_error, "Domain error"));

  // Check that throw assert does not throw a length error
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::length_error, "Length error"));

  // Check that throw assert does not throw an out of range error
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::out_of_range, "Out of range error"));

  // Check that throw assert does not throw a range error
  ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::range_error, "Range error"));
}

TEST(ThrowAssert, ThrowsCorrectErrorType) {
  // Check that throw assert throws the correct error type

  // Check that throw assert throws a logic error
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::logic_error, "Logic error"), std::logic_error);

  // Check that throw assert throws an invalid argument
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Invalid argument"), std::invalid_argument);

  // Check that throw assert throws a runtime error
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::runtime_error, "Runtime error"), std::runtime_error);

  // Check that throw assert throws a domain error
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::domain_error, "Domain error"), std::domain_error);

  // Check that throw assert throws a length error
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::length_error, "Length error"), std::length_error);

  // Check that throw assert throws an out of range error
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::out_of_range, "Out of range error"), std::out_of_range);

  // Check that throw assert throws a range error
  ASSERT_THROW(MUNDY_THROW_REQUIRE(false, std::range_error, "Range error"), std::range_error);
}

TEST(ThrowAssert, ThrowsCorrectMessage) {
  using some_exception = std::logic_error;

  // Throws correctly for regular message
  {
    std::string expected_error_message = "Some error message";
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, "Some error message"), some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, "Some error message");
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(expected_error_message));
    }
  }

  // Throws correctly for string literal message
  {
    constexpr auto some_literal_error_message = make_string_literal("Some error message");
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, some_literal_error_message), some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, some_literal_error_message);
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(some_literal_error_message.value));
    }
  }

  // Throws correct for string message (given that we are on host)
  {
    std::string some_string_error_message = "Some error message";
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, some_string_error_message), some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, some_string_error_message);
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(some_string_error_message));
    }
  }

  // Throws correct for message with pipe (on host)
  {
    std::string expected_message = "Some error message with pipe";
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, std::string("Some error message ") + "with pipe"),
                 some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, std::string("Some error message ") + "with pipe");
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(expected_message));
    }
  }
}
//@}

}  // namespace

}  // namespace core

}  // namespace mundy
