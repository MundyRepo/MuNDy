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
#include <mundy_core/tuple.hpp>  // for mundytuple, mundymake_tuple, etc

namespace mundy {

namespace core {

namespace {

// Even if we are in the core namespace and do not have the std:: prefix, the compiler may still attempt to use
// std::tuple if the object stored in it is in the std. As such, we need to explicitly alias our tuple, get, make_tuple,
// and tuple_cat functions at all times, even in the core namespace.

TEST(TupleTest, MakeTuple) {
  auto t = ::mundy::core::make_tuple(1, 2.5, "test");
  static_assert(std::is_same_v<decltype(t), ::mundy::core::tuple<int, double, const char*>>,
                "Tuple should have types int, double, const char*");
  EXPECT_EQ(::mundy::core::get<0>(t), 1);
  EXPECT_EQ(::mundy::core::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::core::get<2>(t), "test");
}

TEST(TupleTest, TupleCat) {
  auto t1 = ::mundy::core::make_tuple(1, 2.5);
  auto t2 = ::mundy::core::make_tuple("test", 'a');
  auto t = ::mundy::core::tuple_cat(t1, t2);
  EXPECT_EQ(::mundy::core::get<0>(t), 1);
  EXPECT_EQ(::mundy::core::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::core::get<2>(t), "test");
  EXPECT_EQ(::mundy::core::get<3>(t), 'a');
}

TEST(TupleTest, Get) {
  auto t = make_tuple(1, 2.5, "test");
  EXPECT_EQ(::mundy::core::get<0>(t), 1);
  EXPECT_EQ(::mundy::core::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::core::get<2>(t), "test");
}

TEST(TupleTest, ConstTuple) {
  const auto t = ::mundy::core::make_tuple(1, 2.5, "test");
  EXPECT_EQ(::mundy::core::get<0>(t), 1);
  EXPECT_EQ(::mundy::core::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::core::get<2>(t), "test");
}

TEST(TupleTest, EmptyTuple) {
  auto t = ::mundy::core::make_tuple();
  EXPECT_EQ(sizeof(t), 1);  // Empty tuple should have size 1 due to empty base optimization
}

TEST(TupleTest, TupleDefaultConstructable) {
  ::mundy::core::tuple<> t;
  EXPECT_EQ(sizeof(t), 1);  // Empty tuple should have size 1 due to empty base optimization
}
TEST(TupleTest, ConstexprTuple) {
  constexpr auto t = ::mundy::core::make_tuple(1, 2.5, 42);

  // Correct values:
  static_assert(::mundy::core::get<0>(t) == 1, "First element should be 1");
  static_assert(::mundy::core::get<1>(t) == 2.5, "Second element should be 2.5");
  static_assert(::mundy::core::get<2>(t) == 42, "Third element should be 42");

  // Correct types:
  static_assert(std::is_same_v<decltype(::mundy::core::get<0>(t)), const int&>, "First element should be const int&");
  static_assert(std::is_same_v<decltype(::mundy::core::get<1>(t)), const double&>,
                "Second element should be const double&");
  static_assert(std::is_same_v<decltype(::mundy::core::get<2>(t)), const int&>, "Third element should be const int&");
}

TEST(TupleTest, CopyTuple) {
  auto t1 = ::mundy::core::make_tuple(1, 2.5, "test");
  auto t2 = t1;  // Copy
  EXPECT_EQ(::mundy::core::get<0>(t2), 1);
  EXPECT_EQ(::mundy::core::get<1>(t2), 2.5);
  EXPECT_STREQ(::mundy::core::get<2>(t2), "test");
}

TEST(TupleTest, MoveTuple) {
  auto t1 = ::mundy::core::make_tuple(1, 2.5, std::string("test"));
  static_assert(std::is_same_v<decltype(t1), ::mundy::core::tuple<int, double, std::string>>,
                "Tuple should have types int, double, std::string");
  auto t2 = std::move(t1);  // Move
  EXPECT_EQ(::mundy::core::get<0>(t2), 1);
  EXPECT_EQ(::mundy::core::get<1>(t2), 2.5);
  EXPECT_EQ(::mundy::core::get<2>(t2), "test");
}

}  // namespace

}  // namespace core

}  // namespace mundy
