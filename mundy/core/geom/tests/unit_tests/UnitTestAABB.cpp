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
#include <gtest/gtest.h>  // for TEST, EXPECT_*

// C++ core
#include <type_traits>  // for std::is_assignable_v
#include <utility>      // for std::declval, std::move

// Mundy
#include <mundy_geom/primitives.hpp>  // for mundy::AABB, mundy::Point
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3, mundy::get_vector3

namespace mundy {

namespace {

// A view-backed AABB: its corners alias external storage rather than owning it.
using Vec3View = AVector3<double, double*>;
using ViewAABB = mundy::AABB<double, Vec3View, Vec3View>;

// Compile-time guards for the cross-type operations AABB offers.
static_assert(std::is_assignable_v<mundy::AABB<double>&, const mundy::AABB<float>&>,
              "AABB must offer cross-scalar assignment.");
static_assert(std::is_assignable_v<mundy::AABB<double>&, const ViewAABB&>,
              "AABB must offer cross-type assignment from a view-backed AABB.");
static_assert(std::is_constructible_v<mundy::AABB<double>, const mundy::AABB<float>&>,
              "AABB must offer cross-scalar construction.");
static_assert(std::is_constructible_v<mundy::AABB<double>, const ViewAABB&>,
              "AABB must offer cross-type construction from a view-backed AABB.");

// The default AABB is empty (inside-out): min > max on every axis.
TEST(AABBTest, DefaultConstructionIsEmpty) {
  mundy::AABB<double> a;
  EXPECT_GT(a.x_min(), a.x_max());
  EXPECT_GT(a.y_min(), a.y_max());
  EXPECT_GT(a.z_min(), a.z_max());
}

TEST(AABBTest, ComponentAndPointConstruction) {
  mundy::AABB<double> a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  EXPECT_DOUBLE_EQ(a.x_min(), 1.0);
  EXPECT_DOUBLE_EQ(a.y_min(), 2.0);
  EXPECT_DOUBLE_EQ(a.z_min(), 3.0);
  EXPECT_DOUBLE_EQ(a.x_max(), 4.0);
  EXPECT_DOUBLE_EQ(a.y_max(), 5.0);
  EXPECT_DOUBLE_EQ(a.z_max(), 6.0);

  mundy::AABB<double> b(mundy::Point<double>{1.0, 2.0, 3.0}, mundy::Point<double>{4.0, 5.0, 6.0});
  EXPECT_TRUE(is_close(a, b));
}

TEST(AABBTest, SameTypeCopyAndMoveAssignment) {
  const mundy::AABB<double> src(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

  mundy::AABB<double> copy_dst;
  copy_dst = src;
  EXPECT_TRUE(is_close(copy_dst, src));

  mundy::AABB<double> move_src(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  mundy::AABB<double> move_dst;
  move_dst = std::move(move_src);
  EXPECT_TRUE(is_close(move_dst, src));
}

// float -> double is exact for these integer-valued corners.
TEST(AABBTest, CrossScalarConstructionAndAssignment) {
  const mundy::AABB<float> src(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);

  const mundy::AABB<double> constructed(src);
  EXPECT_DOUBLE_EQ(constructed.x_min(), 1.0);
  EXPECT_DOUBLE_EQ(constructed.z_max(), 6.0);

  mundy::AABB<double> assigned;
  assigned = src;
  EXPECT_DOUBLE_EQ(assigned.x_min(), 1.0);
  EXPECT_DOUBLE_EQ(assigned.z_max(), 6.0);
}

// View-backed -> owning assignment deep-copies: mutating the view's storage must not change the owning copy.
TEST(AABBTest, ViewToOwningAssignmentDeepCopies) {
  std::array<double, 3> min_store = {1.0, 2.0, 3.0};
  std::array<double, 3> max_store = {4.0, 5.0, 6.0};

  Vec3View min_vec = mundy::get_vector3<double>(min_store.data());
  Vec3View max_vec = mundy::get_vector3<double>(max_store.data());

  const ViewAABB view_aabb(min_vec, max_vec);
  ASSERT_DOUBLE_EQ(view_aabb.x_min(), 1.0);
  ASSERT_DOUBLE_EQ(view_aabb.z_max(), 6.0);

  mundy::AABB<double> owning;
  owning = view_aabb;
  EXPECT_DOUBLE_EQ(owning.x_min(), 1.0);
  EXPECT_DOUBLE_EQ(owning.z_max(), 6.0);

  // Mutating the view's backing storage must not touch the owning copy.
  min_store[0] = 999.0;
  max_store[2] = -999.0;
  EXPECT_DOUBLE_EQ(owning.x_min(), 1.0);
  EXPECT_DOUBLE_EQ(owning.z_max(), 6.0);
}

TEST(AABBTest, CastToOtherScalar) {
  const mundy::AABB<double> a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  const auto af = a.template cast<float>();
  static_assert(std::is_same_v<decltype(af)::value_type, float>, "cast<float>() must yield a float-valued AABB.");
  EXPECT_FLOAT_EQ(af.x_min(), 1.0f);
  EXPECT_FLOAT_EQ(af.z_max(), 6.0f);
}

TEST(AABBTest, IsClose) {
  const mundy::AABB<double> a(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  const mundy::AABB<double> same(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
  const mundy::AABB<double> different(1.0, 2.0, 3.0, 4.0, 5.0, 6.5);
  EXPECT_TRUE(is_close(a, same));
  EXPECT_FALSE(is_close(a, different));
}

}  // namespace

}  // namespace mundy
