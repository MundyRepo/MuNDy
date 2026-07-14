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
#include <mundy_utils/tuple.hpp>  // for mundy::tuple, mundy::make_tuple, etc

namespace mundy {

namespace {

// Even if we are in the core namespace and do not have the std:: prefix, the compiler may still attempt to use
// std::tuple if the object stored in it is in the std. As such, we need to explicitly alias our tuple, get, make_tuple,
// and tuple_cat functions at all times, even in the core namespace.

TEST(TupleTest, MakeTuple) {
  auto t = ::mundy::make_tuple(1, 2.5, "test");
  static_assert(std::is_same_v<decltype(t), ::mundy::tuple<int, double, const char*>>,
                "Tuple should have types int, double, const char*");
  EXPECT_EQ(::mundy::get<0>(t), 1);
  EXPECT_EQ(::mundy::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::get<2>(t), "test");
}

TEST(TupleTest, TupleCat) {
  auto t1 = ::mundy::make_tuple(1, 2.5);
  auto t2 = ::mundy::make_tuple("test", 'a');
  auto t = ::mundy::tuple_cat(t1, t2);
  EXPECT_EQ(::mundy::get<0>(t), 1);
  EXPECT_EQ(::mundy::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::get<2>(t), "test");
  EXPECT_EQ(::mundy::get<3>(t), 'a');
}

TEST(TupleTest, Get) {
  auto t = make_tuple(1, 2.5, "test");
  EXPECT_EQ(::mundy::get<0>(t), 1);
  EXPECT_EQ(::mundy::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::get<2>(t), "test");
}

TEST(TupleTest, ConstTuple) {
  const auto t = ::mundy::make_tuple(1, 2.5, "test");
  EXPECT_EQ(::mundy::get<0>(t), 1);
  EXPECT_EQ(::mundy::get<1>(t), 2.5);
  EXPECT_STREQ(::mundy::get<2>(t), "test");
}

TEST(TupleTest, EmptyTuple) {
  [[maybe_unused]] auto t = ::mundy::make_tuple();
  EXPECT_EQ(sizeof(t), 1);  // Empty tuple should have size 1 due to empty base optimization
}

TEST(TupleTest, TupleDefaultConstructible) {
  ::mundy::tuple<> t;
  EXPECT_EQ(sizeof(t), 1);  // Empty tuple should have size 1 due to empty base optimization
}
TEST(TupleTest, ConstexprTuple) {
  constexpr auto t = ::mundy::make_tuple(1, 2.5, 42);

  // Correct values:
  static_assert(::mundy::get<0>(t) == 1, "First element should be 1");
  static_assert(::mundy::get<1>(t) == 2.5, "Second element should be 2.5");
  static_assert(::mundy::get<2>(t) == 42, "Third element should be 42");

  // Correct types:
  static_assert(std::is_same_v<decltype(::mundy::get<0>(t)), const int&>, "First element should be const int&");
  static_assert(std::is_same_v<decltype(::mundy::get<1>(t)), const double&>, "Second element should be const double&");
  static_assert(std::is_same_v<decltype(::mundy::get<2>(t)), const int&>, "Third element should be const int&");
}

TEST(TupleTest, CopyTuple) {
  auto t1 = ::mundy::make_tuple(1, 2.5, "test");
  auto t2 = t1;  // Copy
  EXPECT_EQ(::mundy::get<0>(t2), 1);
  EXPECT_EQ(::mundy::get<1>(t2), 2.5);
  EXPECT_STREQ(::mundy::get<2>(t2), "test");
}

TEST(TupleTest, MoveTuple) {
  auto t1 = ::mundy::make_tuple(1, 2.5, std::string("test"));
  static_assert(std::is_same_v<decltype(t1), ::mundy::tuple<int, double, std::string>>,
                "Tuple should have types int, double, std::string");
  auto t2 = std::move(t1);  // Move
  EXPECT_EQ(::mundy::get<0>(t2), 1);
  EXPECT_EQ(::mundy::get<1>(t2), 2.5);
  EXPECT_EQ(::mundy::get<2>(t2), "test");
}

struct NonCopyConstructibleType {
  int value = 0;
  explicit NonCopyConstructibleType(int value_in) : value(value_in) {
  }
  NonCopyConstructibleType(const NonCopyConstructibleType&) = delete;
  NonCopyConstructibleType(NonCopyConstructibleType&&) = default;
};

struct NonMoveConstructibleType {
  int value = 0;
  explicit NonMoveConstructibleType(int value_in) : value(value_in) {
  }
  NonMoveConstructibleType(const NonMoveConstructibleType&) = default;
  NonMoveConstructibleType(NonMoveConstructibleType&&) = delete;
};

TEST(TupleTest, MoveOnlyElement) {
  auto t = ::mundy::tuple<NonCopyConstructibleType>(NonCopyConstructibleType(3));
  EXPECT_EQ(::mundy::get<0>(t).value, 3);

  auto t2 = std::move(t);
  EXPECT_EQ(::mundy::get<0>(t2).value, 3);
}

TEST(TupleTest, CopyOnlyElement) {
  NonMoveConstructibleType source(5);
  auto t = ::mundy::tuple<NonMoveConstructibleType>(source);
  EXPECT_EQ(::mundy::get<0>(t).value, 5);

  auto t2 = t;
  EXPECT_EQ(::mundy::get<0>(t2).value, 5);
}

TEST(TupleTest, TupleCatSupportsMoveOnlyElements) {
  auto t1 = ::mundy::tuple<NonCopyConstructibleType>(NonCopyConstructibleType(1));
  auto t2 = ::mundy::tuple<NonCopyConstructibleType>(NonCopyConstructibleType(2));
  auto t = ::mundy::tuple_cat(std::move(t1), std::move(t2));
  EXPECT_EQ(::mundy::get<0>(t).value, 1);
  EXPECT_EQ(::mundy::get<1>(t).value, 2);
}

TEST(TupleTest, ForEach) {
  auto t = ::mundy::make_tuple(1, 2.5, 3);
  double sum = 0;
  ::mundy::for_each(t, [&sum](const auto& value) { sum += static_cast<double>(value); });
  EXPECT_EQ(sum, 6.5);

  const auto const_t = ::mundy::make_tuple(1, 2.5, 3);
  double const_sum = 0;
  ::mundy::for_each(const_t, [&const_sum](const auto& value) { const_sum += static_cast<double>(value); });
  EXPECT_EQ(const_sum, 6.5);

  ::mundy::tuple<> empty_t;
  int calls = 0;
  ::mundy::for_each(empty_t, [&calls](const auto&) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(TupleTest, AllOf) {
  auto t = ::mundy::make_tuple(1, 2, 3);
  EXPECT_TRUE(::mundy::all_of(t, [](const auto& value) { return value > 0; }));
  EXPECT_FALSE(::mundy::all_of(t, [](const auto& value) { return value > 1; }));

  ::mundy::tuple<> empty_t;
  EXPECT_TRUE(::mundy::all_of(empty_t, [](const auto&) { return false; }));
}

TEST(TupleTest, AnyOf) {
  auto t = ::mundy::make_tuple(1, 2, 3);
  EXPECT_TRUE(::mundy::any_of(t, [](const auto& value) { return value > 2; }));
  EXPECT_FALSE(::mundy::any_of(t, [](const auto& value) { return value > 3; }));

  ::mundy::tuple<> empty_t;
  EXPECT_FALSE(::mundy::any_of(empty_t, [](const auto&) { return true; }));
}

TEST(TupleTest, Apply) {
  auto t = ::mundy::make_tuple(1, 2.5, 3);
  auto sum = ::mundy::apply([](const auto& a, const auto& b, const auto& c) { return a + b + c; }, t);
  EXPECT_EQ(sum, 6.5);

  const auto const_t = ::mundy::make_tuple(1, 2.5, 3);
  auto const_sum = ::mundy::apply([](const auto& a, const auto& b, const auto& c) { return a + b + c; }, const_t);
  EXPECT_EQ(const_sum, 6.5);
}

}  // namespace

}  // namespace mundy
