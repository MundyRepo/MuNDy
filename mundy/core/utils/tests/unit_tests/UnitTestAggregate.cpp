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
#include <type_traits>

// Mundy libs
#include <mundy_utils/aggregate.hpp>  // for mundy::aggregate
#include <mundy_utils/tuple.hpp>      // for mundy::tuple

namespace mundy {

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
struct COUNT;

TEST(AggregateTest, CompileTimeExtensibleTuple) {
  auto cfg = make_aggregate()       //
                 .append<DT>(0.01)  //
                 .append<MAX_ITERS>(1000);

  EXPECT_EQ(cfg.get<DT>(), 0.01);
  EXPECT_EQ(cfg.get<MAX_ITERS>(), 1000);

  static_assert(aggregate_has_v<DT, decltype(cfg)>);
  static_assert(aggregate_has_v<MAX_ITERS, decltype(cfg)>);
}

TEST(AggregateTest, ConstexprUsage) {
  constexpr auto cfg = make_aggregate()       //
                           .append<DT>(0.02)  //
                           .append<MAX_ITERS>(500);

  static_assert(cfg.get<DT>() == 0.02);
  static_assert(cfg.get<MAX_ITERS>() == 500);
}

TEST(AggregateTest, AggregationOfAccessors) {
  auto center_accessor = [](int i) { return i * 2; };
  auto radius_accessor = [](int i) { return i + 1; };

  auto spheres = make_aggregate()                      //
                     .append<CENTER>(center_accessor)  //
                     .append<RADIUS>(radius_accessor);

  EXPECT_EQ(spheres.get<CENTER>()(10), 20);
  EXPECT_EQ(spheres.get<RADIUS>()(3), 4);
  EXPECT_EQ(spheres.get<CENTER>(10), 20);
  EXPECT_EQ(spheres.get<RADIUS>(3), 4);

  auto stored_center_accessor = spheres.get<CENTER>();
  EXPECT_EQ(stored_center_accessor(5), 10);
}

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
  EXPECT_EQ(solver_policies.get<PRECONDITIONER>(5), 10);
}

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
  algs.get<SORT>(data);
  EXPECT_EQ(data, (std::vector<int>{1, 3, 5, 8}));

  auto filtered = algs.get<FILTER>(data);
  EXPECT_EQ(filtered, (std::vector<int>{8}));
}

TEST(AggregateTest, MixedUsage) {
  auto pos_accessor = [](int i) { return i * 10; };
  auto vel_accessor = [](int i) { return i + 2; };

  auto agg = make_aggregate()                //
                 .append<POS>(pos_accessor)  //
                 .append<VEL>(vel_accessor)  //
                 .append<DT>(0.01);

  int i = 3;
  double new_pos = agg.get<POS>(i) + agg.get<VEL>(i) * agg.get<DT>();
  EXPECT_DOUBLE_EQ(new_pos, (i * 10) + (i + 2) * 0.01);
}

TEST(AggregateTest, HasTag) {
  auto agg = make_aggregate()       //
                 .append<DT>(0.01)  //
                 .append<MAX_ITERS>(1000);

  EXPECT_TRUE(has<DT>(agg));
  EXPECT_TRUE(has<MAX_ITERS>(agg));
  EXPECT_FALSE(has<CENTER>(agg));
}

TEST(AggregateTest, ProjectSelectsRequestedTagsAndPreservesOrder) {
  auto agg = make_aggregate()       //
                 .append<DT>(0.01)  //
                 .append<MAX_ITERS>(1000)
                 .append<CENTER>(7);

  auto projected = project<MAX_ITERS, DT>(agg);

  static_assert(aggregate_has_v<MAX_ITERS, decltype(projected)>);
  static_assert(aggregate_has_v<DT, decltype(projected)>);
  static_assert(!aggregate_has_v<CENTER, decltype(projected)>);
  static_assert(std::is_same_v<aggregate_tag_t<0, decltype(projected)>, MAX_ITERS>);
  static_assert(std::is_same_v<aggregate_tag_t<1, decltype(projected)>, DT>);

  EXPECT_EQ(projected.get<MAX_ITERS>(), 1000);
  EXPECT_DOUBLE_EQ(projected.get<DT>(), 0.01);
}

TEST(AggregateTest, ProjectWorksForConstAggregate) {
  auto agg = make_aggregate()       //
                 .append<DT>(0.02)  //
                 .append<MAX_ITERS>(500)
                 .append<CENTER>(42);
  const auto& cagg = agg;

  auto projected = project<DT, CENTER>(cagg);

  EXPECT_DOUBLE_EQ(projected.get<DT>(), 0.02);
  EXPECT_EQ(projected.get<CENTER>(), 42);
}

TEST(AggregateTest, ProjectCreatesValueCopy) {
  auto agg = make_aggregate()        //
                 .append<COUNT>(10)  //
                 .append<DT>(0.1);

  auto projected = project<COUNT>(agg);
  agg.get<COUNT>() = 25;

  EXPECT_EQ(projected.get<COUNT>(), 10);
  EXPECT_EQ(agg.get<COUNT>(), 25);
}

TEST(VariantAggregateTest, CompileTimeExtensibleTuple) {
  using VariantType = variant<int, double>;

  auto vagg = make_variant_aggregate<VariantType>()  //
                  .append<DT>(VariantType(0.01))     //
                  .append<MAX_ITERS>(VariantType(1000));

  EXPECT_EQ(vagg.size(), 2u);
  EXPECT_TRUE(vagg.get<DT>().template holds_alternative<double>());
  EXPECT_DOUBLE_EQ(vagg.get<DT>().template get<double>(), 0.01);
  EXPECT_TRUE(vagg.get<MAX_ITERS>().template holds_alternative<int>());
  EXPECT_EQ(vagg.get<MAX_ITERS>().template get<int>(), 1000);
  EXPECT_DOUBLE_EQ(vagg.get<0>().template get<double>(), 0.01);
  EXPECT_EQ(vagg.get(1).template get<int>(), 1000);

  EXPECT_TRUE(has<DT>(vagg));
  EXPECT_TRUE(has<MAX_ITERS>(vagg));
  EXPECT_FALSE(has<CENTER>(vagg));

  static_assert(variant_aggregate_has_v<DT, decltype(vagg)>);
  static_assert(variant_aggregate_has_v<MAX_ITERS, decltype(vagg)>);
  static_assert(!variant_aggregate_has_v<CENTER, decltype(vagg)>);
  static_assert(std::is_same_v<variant_aggregate_tag_t<0, decltype(vagg)>, DT>);
  static_assert(std::is_same_v<variant_aggregate_tag_t<1, decltype(vagg)>, MAX_ITERS>);
}

TEST(VariantAggregateTest, NonMemberHelpersForwardToMemberApi) {
  using VariantType = variant<int, double>;
  auto vagg = make_variant_aggregate<VariantType>();

  auto with_dt = append<DT>(vagg, VariantType(7));
  EXPECT_EQ(get<DT>(with_dt).template get<int>(), 7);
  EXPECT_EQ(get<0>(with_dt).template get<int>(), 7);
}

TEST(VariantAggregateTest, ProjectSelectsRequestedTagsAndPreservesOrder) {
  using VariantType = variant<int, double>;
  auto vagg = make_variant_aggregate<VariantType>()  //
                  .append<DT>(VariantType(0.01))     //
                  .append<MAX_ITERS>(VariantType(1000))
                  .append<CENTER>(VariantType(7));

  auto projected = project<MAX_ITERS, DT>(vagg);

  static_assert(variant_aggregate_has_v<MAX_ITERS, decltype(projected)>);
  static_assert(variant_aggregate_has_v<DT, decltype(projected)>);
  static_assert(!variant_aggregate_has_v<CENTER, decltype(projected)>);
  static_assert(std::is_same_v<variant_aggregate_tag_t<0, decltype(projected)>, MAX_ITERS>);
  static_assert(std::is_same_v<variant_aggregate_tag_t<1, decltype(projected)>, DT>);

  EXPECT_EQ(projected.get<MAX_ITERS>().template get<int>(), 1000);
  EXPECT_DOUBLE_EQ(projected.get<DT>().template get<double>(), 0.01);
}

TEST(VariantAggregateTest, ProjectWorksForConstVariantAggregate) {
  using VariantType = variant<int, double>;
  auto vagg = make_variant_aggregate<VariantType>()  //
                  .append<DT>(VariantType(0.02))     //
                  .append<MAX_ITERS>(VariantType(500))
                  .append<CENTER>(VariantType(42));
  const auto& cvagg = vagg;

  auto projected = project<DT, CENTER>(cvagg);

  EXPECT_DOUBLE_EQ(projected.get<DT>().template get<double>(), 0.02);
  EXPECT_EQ(projected.get<CENTER>().template get<int>(), 42);
}

TEST(VariantAggregateTest, ProjectCreatesValueCopy) {
  using VariantType = variant<int, double>;
  auto vagg = make_variant_aggregate<VariantType>()  //
                  .append<COUNT>(VariantType(10))    //
                  .append<DT>(VariantType(0.1));

  auto projected = project<COUNT>(vagg);
  vagg.get<COUNT>() = 25;

  EXPECT_EQ(projected.get<COUNT>().template get<int>(), 10);
  EXPECT_EQ(vagg.get<COUNT>().template get<int>(), 25);
}

}  // namespace

}  // namespace mundy
