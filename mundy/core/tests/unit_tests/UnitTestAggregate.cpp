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
#include <mundy_core/aggregate.hpp>  // for mundy::core::aggregate
#include <mundy_core/tuple.hpp>      // for mundy::core::tuple

namespace mundy {

namespace core {

namespace {

// Define some tags as incomplete structs
struct DT;
struct MAX_ITERS;
struct CENTER;
struct RADIUS;
struct SOLVER;
struct PRECONDITIONER;
struct SORT;
struct FILTER;
struct POS;
struct VEL;

// Test case 1: Compile-time extensible tuple
TEST(AggregateTest, CompileTimeExtensibleTuple) {
  auto cfg = make_aggregate()       //
                 .append<DT>(0.01)  //
                 .append<MAX_ITERS>(1000);

  EXPECT_EQ(cfg.get<DT>(), 0.01);
  EXPECT_EQ(cfg.get<MAX_ITERS>(), 1000);

  static_assert(aggregate_has_v<DT, decltype(cfg)>);
  static_assert(aggregate_has_v<MAX_ITERS, decltype(cfg)>);
}

// Test case 2: Aggregation of accessors
TEST(AggregateTest, AggregationOfAccessors) {
  auto center_accessor = [](int i) { return i * 2; };
  auto radius_accessor = [](int i) { return i + 1; };

  auto spheres = make_aggregate()                      //
                     .append<CENTER>(center_accessor)  //
                     .append<RADIUS>(radius_accessor);

  EXPECT_EQ(spheres.get<CENTER>()(10), 20);
  EXPECT_EQ(spheres.get<RADIUS>()(3), 4);

  auto stored_center_accessor = spheres.get<CENTER>();
  EXPECT_EQ(stored_center_accessor(5), 10);
}

// Test case 3: Aggregation of policies/strategies
TEST(AggregateTest, AggregationOfPolicies) {
  struct SolverPolicy {
    int solve(int a, int b) const {
      return a + b;
    }
  };
  struct PreconditionerPolicy {
    int operator()(int x) const {
      return x * 2;
    }
  };

  auto solver_policies = make_aggregate()                     //
                             .append<SOLVER>(SolverPolicy{})  //
                             .append<PRECONDITIONER>(PreconditionerPolicy{});

  EXPECT_EQ(solver_policies.get<SOLVER>().solve(3, 4), 7);
  EXPECT_EQ(solver_policies.get<PRECONDITIONER>()(5), 10);
}

// Test case 4: Aggregation of algorithms/functors
TEST(AggregateTest, AggregationOfAlgorithms) {
  struct SortAlgorithm {
    void operator()(std::vector<int>& data) const {
      std::sort(data.begin(), data.end());
    }
  };
  struct FilterAlgorithm {
    std::vector<int> operator()(const std::vector<int>& data) const {
      std::vector<int> result;
      std::copy_if(data.begin(), data.end(), std::back_inserter(result), [](int x) { return x % 2 == 0; });
      return result;
    }
  };

  auto algs = make_aggregate()                    //
                  .append<SORT>(SortAlgorithm{})  //
                  .append<FILTER>(FilterAlgorithm{});

  std::vector<int> data = {5, 3, 8, 1};
  algs.get<SORT>()(data);
  EXPECT_EQ(data, (std::vector<int>{1, 3, 5, 8}));

  auto filtered = algs.get<FILTER>()(data);
  EXPECT_EQ(filtered, (std::vector<int>{8}));
}

// Test case 5: Mixed usage
TEST(AggregateTest, MixedUsage) {
  auto pos_accessor = [](int i) { return i * 10; };
  auto vel_accessor = [](int i) { return i + 2; };

  auto agg = make_aggregate()                //
                 .append<POS>(pos_accessor)  //
                 .append<VEL>(vel_accessor)  //
                 .append<DT>(0.01);

  int i = 3;
  double new_pos = agg.get<POS>()(i) + agg.get<VEL>()(i) * agg.get<DT>();
  EXPECT_DOUBLE_EQ(new_pos, (i * 10) + (i + 2) * 0.01);
}

// Test case 6: Check if a tag exists
TEST(AggregateTest, HasTag) {
  auto agg = make_aggregate()       //
                 .append<DT>(0.01)  //
                 .append<MAX_ITERS>(1000);

  EXPECT_TRUE(has<DT>(agg));
  EXPECT_TRUE(has<MAX_ITERS>(agg));
  EXPECT_FALSE(has<CENTER>(agg));
}

}  // namespace

}  // namespace core

}  // namespace mundy
