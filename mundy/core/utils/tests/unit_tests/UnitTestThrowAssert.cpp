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
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

namespace mundy {

namespace {

//! \name Throw assert tests
//@{

TEST(ThrowAssert, Predicates) {
  // These are all the language features we need to be true for MUNDY_THROW_REQUIRE to operate as expected.

  // Check type-trait interfaces.
  static_assert(is_char_array_v<const char[3]>);
  static_assert(!is_char_array_v<const char*>);
  static_assert(is_string_literal_v<const char[3]>);
  static_assert(!is_string_literal_v<int>);
  static_assert(!is_our_string_literal_v<const char[3]>);
  static_assert(is_our_string_literal_v<decltype(make_string_literal("b"))>);

  // Check that is_string_literal works as expected
  static_assert(MUNDY_IS_STRING_LITERAL("string literal"));
  static_assert(!MUNDY_IS_STRING_LITERAL(42));
  constexpr auto a = "a";
  static_assert(!MUNDY_IS_STRING_LITERAL(a));

  // Check that is_mundy_string_literal works as expected
  static_assert(!MUNDY_IS_OUR_STRING_LITERAL("string literal"));
  constexpr auto b = make_string_literal("b");
  static_assert(MUNDY_IS_OUR_STRING_LITERAL(b));

  // Check that all compile-time sink messages are device printable.
  constexpr auto literal_sink = sink() << "literal " << "sink";
  static_assert(DevicePrintableThrowMessage<decltype(literal_sink)>);
  constexpr StringSink<StringLiteral<6>, StringLiteral<5>> compile_time_sink(
      ::mundy::make_tuple(make_string_literal("left "), make_string_literal("side")));
  static_assert(DevicePrintableThrowMessage<decltype(compile_time_sink)>);
  static_assert(get_throw_require_device_string(make_string_literal("false"), compile_time_sink,
                                                make_string_literal("file.cpp"), make_string_literal("7")) ==
                make_string_literal("Assertion (false) failed.\nFile: file.cpp\nLine: 7\nMessage: left side"));
  const auto runtime_sink = sink() << "value = " << 2;
  static_assert(!DevicePrintableThrowMessage<decltype(runtime_sink)>);

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

  // Check that sink messages are not evaluated when the assertion is true.
  {
    int num_message_constructions = 0;
    auto make_message = [&]() {
      ++num_message_constructions;
      return sink() << "Failure count " << num_message_constructions;
    };
    ASSERT_NO_THROW(MUNDY_THROW_REQUIRE(true, std::logic_error, make_message()));
    EXPECT_EQ(num_message_constructions, 0);
  }
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

  // Throws correct for message with addition (given that we are on host)
  {
    std::string expected_message = "Some error message with addition";
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, std::string("Some error message ") + "with addition"),
                 some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, std::string("Some error message ") + "with addition");
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(expected_message));
    }
  }

  // Throws correctly for literal sink message.
  {
    constexpr auto sink_message = sink() << "Some sink " << "error message";
    std::string expected_message = "Some sink error message";
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, sink_message), some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, sink_message);
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(expected_message));
    }
  }

  // Throws correctly for runtime sink message (given that we are on host).
  {
    int a = 2;
    std::string expected_message = "Some error message with streaming 2";
    ASSERT_THROW(MUNDY_THROW_REQUIRE(false, some_exception, sink() << "Some error message with streaming " << a),
                 some_exception);
    try {
      MUNDY_THROW_REQUIRE(false, some_exception, sink() << "Some error message with streaming " << a);
    } catch (const some_exception& e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr(expected_message));
    }
  }
}
//@}

}  // namespace

}  // namespace mundy
