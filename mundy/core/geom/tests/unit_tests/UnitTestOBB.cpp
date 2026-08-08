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
#include <mundy_geom/primitives.hpp>  // for mundy::OBB, mundy::Point
#include <mundy_math/Quaternion.hpp>  // for mundy::Quaternion, mundy::get_quaternion_view
#include <mundy_math/Vector3.hpp>     // for mundy::Vector3, mundy::get_vector3_view

namespace mundy {

namespace {

// A view-backed OBB: center, orientation, and half-extents alias external storage rather than owning it.
using ViewVec = decltype(mundy::get_vector3_view<double>(std::declval<double*>()));
using ViewQuat = decltype(mundy::get_quaternion_view<double>(std::declval<double*>()));
using ViewOBB = mundy::OBB<double, ViewVec, ViewQuat, ViewVec>;

// Compile-time guards for the cross-type operations OBB offers.
static_assert(std::is_assignable_v<mundy::OBB<double>&, const mundy::OBB<float>&>,
              "OBB must offer cross-scalar assignment.");
static_assert(std::is_assignable_v<mundy::OBB<double>&, const ViewOBB&>,
              "OBB must offer cross-type assignment from a view-backed OBB.");
static_assert(std::is_constructible_v<mundy::OBB<double>, const mundy::OBB<float>&>,
              "OBB must offer cross-scalar construction.");
static_assert(std::is_constructible_v<mundy::OBB<double>, const ViewOBB&>,
              "OBB must offer cross-type construction from a view-backed OBB.");

TEST(OBBTest, DefaultConstruction) {
  mundy::OBB<double> a;
  EXPECT_DOUBLE_EQ(a.center()[0], 0.0);
  EXPECT_DOUBLE_EQ(a.center()[1], 0.0);
  EXPECT_DOUBLE_EQ(a.center()[2], 0.0);
  EXPECT_DOUBLE_EQ(a.orientation().w(), 1.0);
  EXPECT_DOUBLE_EQ(a.orientation().x(), 0.0);
  EXPECT_DOUBLE_EQ(a.orientation().y(), 0.0);
  EXPECT_DOUBLE_EQ(a.orientation().z(), 0.0);
  // Half-extents default to the invalid sentinel -1.
  EXPECT_DOUBLE_EQ(a.half_extent(0), -1.0);
  EXPECT_DOUBLE_EQ(a.half_extent(1), -1.0);
  EXPECT_DOUBLE_EQ(a.half_extent(2), -1.0);
}

TEST(OBBTest, ComponentConstruction) {
  const mundy::Point<double> center{1.0, 2.0, 3.0};
  const mundy::Quaternion<double> orientation{0.5, 0.5, 0.5, 0.5};

  const mundy::OBB<double> per_axis(center, orientation, 4.0, 5.0, 6.0);
  EXPECT_DOUBLE_EQ(per_axis.center()[0], 1.0);
  EXPECT_DOUBLE_EQ(per_axis.half_extent(0), 4.0);
  EXPECT_DOUBLE_EQ(per_axis.half_extent(2), 6.0);

  const mundy::OBB<double> from_vec(center, orientation, mundy::Vector3<double>{4.0, 5.0, 6.0});
  EXPECT_TRUE(is_close(per_axis, from_vec));

  // Center + uniform half-extent, identity orientation.
  const mundy::OBB<double> uniform(center, 2.0);
  EXPECT_DOUBLE_EQ(uniform.half_extent(0), 2.0);
  EXPECT_DOUBLE_EQ(uniform.half_extent(1), 2.0);
  EXPECT_DOUBLE_EQ(uniform.half_extent(2), 2.0);
  EXPECT_DOUBLE_EQ(uniform.orientation().w(), 1.0);
}

TEST(OBBTest, SameTypeCopyAndMoveAssignment) {
  const mundy::OBB<double> src(mundy::Point<double>{1.0, 2.0, 3.0},
                               mundy::Quaternion<double>{0.5, 0.5, 0.5, 0.5}, 4.0, 5.0, 6.0);

  mundy::OBB<double> copy_dst;
  copy_dst = src;
  EXPECT_TRUE(is_close(copy_dst, src));

  mundy::OBB<double> move_src = src;
  mundy::OBB<double> move_dst;
  move_dst = std::move(move_src);
  EXPECT_TRUE(is_close(move_dst, src));
}

// float -> double is exact for these values (0.5 and small integers are representable in both).
TEST(OBBTest, CrossScalarConstructionAndAssignment) {
  const mundy::OBB<float> src(mundy::Point<float>{1.0f, 2.0f, 3.0f},
                              mundy::Quaternion<float>{0.5f, 0.5f, 0.5f, 0.5f}, 4.0f, 5.0f, 6.0f);

  const mundy::OBB<double> constructed(src);
  EXPECT_DOUBLE_EQ(constructed.center()[0], 1.0);
  EXPECT_DOUBLE_EQ(constructed.orientation().w(), 0.5);
  EXPECT_DOUBLE_EQ(constructed.half_extent(2), 6.0);

  mundy::OBB<double> assigned;
  assigned = src;
  EXPECT_DOUBLE_EQ(assigned.center()[2], 3.0);
  EXPECT_DOUBLE_EQ(assigned.half_extent(0), 4.0);
}

// View-backed -> owning assignment deep-copies: mutating the view's storage must not change the owning copy.
TEST(OBBTest, ViewToOwningAssignmentDeepCopies) {
  double center_store[3] = {1.0, 2.0, 3.0};
  double orient_store[4] = {0.5, 0.5, 0.5, 0.5};
  double he_store[3] = {4.0, 5.0, 6.0};
  const ViewOBB view_obb(mundy::get_vector3_view<double>(&center_store[0]),
                         mundy::get_quaternion_view<double>(&orient_store[0]),
                         mundy::get_vector3_view<double>(&he_store[0]));

  mundy::OBB<double> owning;
  owning = view_obb;
  EXPECT_DOUBLE_EQ(owning.center()[1], 2.0);
  EXPECT_DOUBLE_EQ(owning.orientation().x(), 0.5);
  EXPECT_DOUBLE_EQ(owning.half_extent(2), 6.0);

  // Mutating the view's backing storage must not touch the owning copy.
  center_store[1] = 999.0;
  orient_store[1] = -999.0;
  he_store[2] = -999.0;
  EXPECT_DOUBLE_EQ(owning.center()[1], 2.0);
  EXPECT_DOUBLE_EQ(owning.orientation().x(), 0.5);
  EXPECT_DOUBLE_EQ(owning.half_extent(2), 6.0);
}

// Cross-type move construction deep-copies: moving a view-backed OBB yields an independent owning OBB.
TEST(OBBTest, CrossTypeMoveConstruction) {
  // From an rvalue OBB<float> (cross-scalar).
  const mundy::OBB<double> from_scalar(mundy::OBB<float>{mundy::Point<float>{1.0f, 2.0f, 3.0f},
                                                         mundy::Quaternion<float>{0.5f, 0.5f, 0.5f, 0.5f},
                                                         4.0f, 5.0f, 6.0f});
  EXPECT_DOUBLE_EQ(from_scalar.center()[0], 1.0);
  EXPECT_DOUBLE_EQ(from_scalar.orientation().w(), 0.5);
  EXPECT_DOUBLE_EQ(from_scalar.half_extent(2), 6.0);

  // From an rvalue view-backed OBB: mutating the backing storage afterward must not change the result.
  double center_store[3] = {1.0, 2.0, 3.0};
  double orient_store[4] = {0.5, 0.5, 0.5, 0.5};
  double he_store[3] = {4.0, 5.0, 6.0};
  const mundy::OBB<double> owning(ViewOBB(mundy::get_vector3_view<double>(&center_store[0]),
                                          mundy::get_quaternion_view<double>(&orient_store[0]),
                                          mundy::get_vector3_view<double>(&he_store[0])));
  center_store[1] = 999.0;
  orient_store[1] = -999.0;
  he_store[2] = -999.0;
  EXPECT_DOUBLE_EQ(owning.center()[1], 2.0);
  EXPECT_DOUBLE_EQ(owning.orientation().x(), 0.5);
  EXPECT_DOUBLE_EQ(owning.half_extent(2), 6.0);
}

TEST(OBBTest, IsClose) {
  const mundy::OBB<double> a(mundy::Point<double>{1.0, 2.0, 3.0},
                             mundy::Quaternion<double>{0.5, 0.5, 0.5, 0.5}, 4.0, 5.0, 6.0);
  const mundy::OBB<double> same = a;
  const mundy::OBB<double> different(mundy::Point<double>{1.0, 2.0, 3.5},
                                     mundy::Quaternion<double>{0.5, 0.5, 0.5, 0.5}, 4.0, 5.0, 6.0);
  EXPECT_TRUE(is_close(a, same));
  EXPECT_FALSE(is_close(a, different));
}

}  // namespace

}  // namespace mundy
