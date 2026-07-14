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

struct CopyMoveTracked {
  int copies = 0;
  int moves = 0;

  CopyMoveTracked() = default;
  CopyMoveTracked(const CopyMoveTracked& other) : copies(other.copies + 1), moves(other.moves) {
  }
  CopyMoveTracked(CopyMoveTracked&& other) noexcept : copies(other.copies), moves(other.moves + 1) {
  }
};

TEST(TupleTest, ConstructingFromRvaluesMovesRatherThanCopies) {
  // tuple's constructor forwards through tuple_impl to tuple_member (2 internal hops), so a single rvalue
  // argument is moved 3 times total (once per hop) and copied zero times.
  CopyMoveTracked tracked;
  auto t = ::mundy::tuple<CopyMoveTracked>(std::move(tracked));
  EXPECT_EQ(::mundy::get<0>(t).copies, 0);
  EXPECT_EQ(::mundy::get<0>(t).moves, 3);
}

TEST(TupleTest, ConstructingFromLvaluesCopiesOnceThenMoves) {
  // An lvalue argument forces exactly one copy at the outermost layer (into tuple's own by-value parameter);
  // every hop after that moves the already-local copy, rather than copying again at each layer.
  CopyMoveTracked tracked;
  auto t = ::mundy::tuple<CopyMoveTracked>(tracked);
  EXPECT_EQ(::mundy::get<0>(t).copies, 1);
  EXPECT_EQ(::mundy::get<0>(t).moves, 2);
}

TEST(TupleTest, MakeTupleFromRvaluesMovesRatherThanCopies) {
  // A prvalue argument (CopyMoveTracked{}) materializes directly into make_tuple's by-value parameter (mandatory
  // copy elision), then gets moved through make_tuple -> tuple -> tuple_impl -> tuple_member (3 hops) -- zero
  // copies either way.
  auto t = ::mundy::make_tuple(CopyMoveTracked{});
  EXPECT_EQ(::mundy::get<0>(t).copies, 0);
}

TEST(TupleTest, ForEach) {
  auto t = ::mundy::make_tuple(1, 2.5, 3);
  double sum = 0;
  ::mundy::for_each(t, [&sum](const auto& value) { sum += static_cast<double>(value); });
  EXPECT_EQ(sum, 6.5);
}

TEST(TupleTest, ForEachConstTuple) {
  const auto t = ::mundy::make_tuple(1, 2.5, 3);
  double sum = 0;
  ::mundy::for_each(t, [&sum](const auto& value) { sum += static_cast<double>(value); });
  EXPECT_EQ(sum, 6.5);
}

TEST(TupleTest, ForEachEmptyTuple) {
  ::mundy::tuple<> t;
  int calls = 0;
  ::mundy::for_each(t, [&calls](const auto&) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(TupleTest, AllOf) {
  auto t = ::mundy::make_tuple(1, 2, 3);
  EXPECT_TRUE(::mundy::all_of(t, [](const auto& value) { return value > 0; }));
  EXPECT_FALSE(::mundy::all_of(t, [](const auto& value) { return value > 1; }));
}

TEST(TupleTest, AllOfEmptyTuple) {
  ::mundy::tuple<> t;
  EXPECT_TRUE(::mundy::all_of(t, [](const auto&) { return false; }));
}

TEST(TupleTest, AnyOf) {
  auto t = ::mundy::make_tuple(1, 2, 3);
  EXPECT_TRUE(::mundy::any_of(t, [](const auto& value) { return value > 2; }));
  EXPECT_FALSE(::mundy::any_of(t, [](const auto& value) { return value > 3; }));
}

TEST(TupleTest, AnyOfEmptyTuple) {
  ::mundy::tuple<> t;
  EXPECT_FALSE(::mundy::any_of(t, [](const auto&) { return true; }));
}

TEST(TupleTest, Apply) {
  auto t = ::mundy::make_tuple(1, 2.5, 3);
  auto sum = ::mundy::apply([](const auto& a, const auto& b, const auto& c) { return a + b + c; }, t);
  EXPECT_EQ(sum, 6.5);
}

TEST(TupleTest, ApplyConstTuple) {
  const auto t = ::mundy::make_tuple(1, 2.5, 3);
  auto sum = ::mundy::apply([](const auto& a, const auto& b, const auto& c) { return a + b + c; }, t);
  EXPECT_EQ(sum, 6.5);
}

}  // namespace

}  // namespace mundy
