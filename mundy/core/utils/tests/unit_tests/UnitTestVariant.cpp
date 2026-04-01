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
#include <string>
#include <type_traits>

// Mundy libs
#include <mundy_utils/variant.hpp>  // for mundy::variant, mundy::get, mundy::make_variant, etc

namespace mundy {

namespace {

// Even if we are in the mundy namespace and do not have the std:: prefix, the compiler may still attempt to use
// std::variant if the object stored in it is in the std.

TEST(VariantTest, DefaultConstructor) {
  ::mundy::variant<int, double> var;

  EXPECT_EQ(var.index(), 0u);
  EXPECT_TRUE(var.holds_alternative<int>());
  EXPECT_FALSE(var.holds_alternative<double>());
  EXPECT_EQ(::mundy::get<int>(var), 0);
}

TEST(VariantTest, ConstructWithSpecificType) {
  ::mundy::variant<int, double, std::string> var(2.5);

  EXPECT_EQ(var.index(), 1u);
  EXPECT_TRUE(::mundy::holds_alternative<double>(var));
  EXPECT_EQ(::mundy::get<double>(var), 2.5);
}

TEST(VariantTest, SizeAndIndexOf) {
  using V = ::mundy::variant<int, double, std::string>;
  static_assert(V::size() == 3, "variant size should match number of alternatives");
  static_assert(V::template index_of<int>() == 0, "int should be index 0");
  static_assert(V::template index_of<double>() == 1, "double should be index 1");
  static_assert(V::template index_of<std::string>() == 2, "string should be index 2");
  static_assert(::mundy::index_of<std::string, int, double, std::string>() == 2, "free function index_of should match");

  EXPECT_EQ((::mundy::variant_size_v<V>), 3u);
}

TEST(VariantTest, AlternativeTypeAlias) {
  using V = ::mundy::variant<int, double, std::string>;
  static_assert(std::is_same_v<typename V::template alternative_t<0>, int>, "alternative_t<0> should be int");
  static_assert(std::is_same_v<typename V::template alternative_t<1>, double>, "alternative_t<1> should be double");
  static_assert(std::is_same_v<::mundy::variant_alternative_t<2, V>, std::string>,
                "variant_alternative_t<2, V> should be std::string");
}

TEST(VariantTest, GetByTypeMutable) {
  ::mundy::variant<int, double> var(11);
  auto& value = ::mundy::get<int>(var);
  value = 42;

  EXPECT_EQ(::mundy::get<int>(var), 42);
}

TEST(VariantTest, GetByIndexMutable) {
  ::mundy::variant<int, double, std::string> var(std::string("abc"));
  auto& value = ::mundy::get<2>(var);
  value += "def";

  EXPECT_EQ(::mundy::get<std::string>(var), "abcdef");
}

TEST(VariantTest, ConstGet) {
  const ::mundy::variant<int, double> var(7);
  static_assert(std::is_same_v<decltype(::mundy::get<int>(var)), const int&>, "const get should return const ref");
  EXPECT_EQ(::mundy::get<int>(var), 7);
}

TEST(VariantTest, AssignmentChangesActiveType) {
  ::mundy::variant<int, double, std::string> var(1);
  EXPECT_TRUE(::mundy::holds_alternative<int>(var));

  var = 4.25;
  EXPECT_TRUE(::mundy::holds_alternative<double>(var));
  EXPECT_EQ(::mundy::get<double>(var), 4.25);

  var = std::string("done");
  EXPECT_TRUE(::mundy::holds_alternative<std::string>(var));
  EXPECT_EQ(::mundy::get<std::string>(var), "done");
}

TEST(VariantTest, VisitReturnsValue) {
  const ::mundy::variant<int, double, std::string> var(std::string("abcd"));

  const int result = ::mundy::visit(
      [](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, int>) {
          return value;
        } else if constexpr (std::is_same_v<T, double>) {
          return static_cast<int>(value);
        } else {
          return static_cast<int>(value.size());
        }
      },
      var);

  EXPECT_EQ(result, 4);
}

TEST(VariantTest, VisitCanMutateActiveAlternative) {
  ::mundy::variant<int, double, std::string> var(2);

  ::mundy::visit(
      [](auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, int>) {
          value += 10;
        } else if constexpr (std::is_same_v<T, double>) {
          value *= 2.0;
        } else {
          value += "_x";
        }
      },
      var);

  EXPECT_EQ(::mundy::get<int>(var), 12);
}

TEST(VariantTest, VisitVoidReturn) {
  ::mundy::variant<int, double, std::string> var(3.0);
  bool visited = false;

  ::mundy::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        visited = true;
        if constexpr (std::is_same_v<T, double>) {
          EXPECT_DOUBLE_EQ(value, 3.0);
        }
      },
      var);

  EXPECT_TRUE(visited);
}

#ifndef NDEBUG
TEST(VariantTest, WrongTypeGetThrows) {
  ::mundy::variant<int, double> var(5);
  EXPECT_THROW((void)::mundy::get<double>(var), std::runtime_error);
}

TEST(VariantTest, WrongIndexGetThrows) {
  ::mundy::variant<int, double> var(5);
  EXPECT_THROW((void)::mundy::get<1>(var), std::runtime_error);
}
#endif

}  // namespace

}  // namespace mundy
