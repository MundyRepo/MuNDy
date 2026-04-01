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
#include <gtest/gtest.h>  // for TEST, EXPECT_EQ, etc

// C++ core libs
#include <sstream>
#include <string>

// Mundy libs
#include <mundy_utils/StringSink.hpp>

namespace mundy {

namespace {

TEST(StringSink, ConceptsAndTraits) {
  constexpr auto literal_sink = sink() << "abc" << "def";
  static_assert(LiteralStringSink<decltype(literal_sink)>);
  static_assert(is_string_literal_sink_v<decltype(literal_sink)>);
  static_assert(literal_sink.value == make_string_literal("abcdef"));

  constexpr auto literal_sink_from_string_literal = sink() << make_string_literal("ghi") << "jkl";
  static_assert(LiteralStringSink<decltype(literal_sink_from_string_literal)>);
  static_assert(literal_sink_from_string_literal.value == make_string_literal("ghijkl"));

  const auto runtime_sink = sink() << "value = " << 2;
  static_assert(RuntimeStringSink<decltype(runtime_sink)>);
  static_assert(is_string_sink_v<decltype(runtime_sink)>);
}

TEST(StringSink, MaterializesExpectedStrings) {
  constexpr auto literal_sink = sink() << "Some " << "error message";
  EXPECT_EQ(literal_sink.to_string(), "Some error message");

  const auto runtime_sink = sink() << "Failure for a = " << 2;
  EXPECT_EQ(runtime_sink.to_string(), "Failure for a = 2");

  const auto string_started_sink = sink() << std::string("prefix ") << 2 << " suffix";
  EXPECT_EQ(string_started_sink.to_string(), "prefix 2 suffix");
}

TEST(StringSink, StreamsToOstream) {
  std::ostringstream os;
  os << (sink() << "Entity " << 7 << " missing");
  EXPECT_EQ(os.str(), "Entity 7 missing");
}

}  // namespace

}  // namespace mundy
