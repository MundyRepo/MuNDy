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
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Mundy libs
#include <mundy_math/Array.hpp>           // for mundy::math::Array
#include <mundy_math/MaskedView.hpp>      // for mundy::math::MaskedView
#include <mundy_math/ShiftedView.hpp>     // for mundy::math::ShiftedView
#include <mundy_math/StridedView.hpp>     // for mundy::math::StridedView
#include <mundy_math/Tolerance.hpp>       // for mundy::math::get_relaxed_tolerance
#include <mundy_math/TransposedView.hpp>  // for mundy::math::TransposedView

namespace mundy {

namespace math {

namespace {

TEST(Views, ShiftedView) {
  auto run_test_nonconst_size_3 = [](auto &shifted_data, const auto &original_data) {
    ASSERT_EQ(shifted_data[0], original_data[1]);
    ASSERT_EQ(shifted_data[1], original_data[2]);
    shifted_data[0] = 4.0;
    shifted_data[1] = 5.0;
    ASSERT_EQ(original_data[1], 4.0);
    ASSERT_EQ(original_data[2], 5.0);
  };

  auto run_test_const_size_3 = [](auto &shifted_data, const auto &original_data) {
    ASSERT_EQ(shifted_data[0], original_data[1]);
    ASSERT_EQ(shifted_data[1], original_data[2]);
    const bool is_data_const = std::is_const_v<std::remove_reference_t<decltype(shifted_data[0])>>;
    ASSERT_TRUE(is_data_const);
  };

  // Reference view
  mundy::math::Array<double, 3> data1{1.0, 2.0, 3.0};
  constexpr size_t shift = 1;
  auto shifted_data = mundy::math::get_shifted_view<double, shift>(data1);
  run_test_nonconst_size_3(shifted_data, data1);

  // Pointer view
  mundy::math::Array<double, 3> data2{1.0, 2.0, 3.0};
  auto shifted_data_ptr = mundy::math::get_shifted_view<double, shift>(data2.data());
  run_test_nonconst_size_3(shifted_data_ptr, data2);

  // Const reference view
  const mundy::math::Array<double, 3> data3{1.0, 2.0, 3.0};
  auto shifted_data_const = mundy::math::get_shifted_view<double, shift>(data3);
  run_test_const_size_3(shifted_data_const, data3.data());

  // Const pointer view
  const mundy::math::Array<double, 3> data4{1.0, 2.0, 3.0};
  auto shifted_data_const_ptr = mundy::math::get_shifted_view<double, shift>(data4.data());
  run_test_const_size_3(shifted_data_const_ptr, data4.data());
}

TEST(Views, OwningShiftedAccessor) {
  auto create_owning_shifted_accessor_delete_original_data = []() {
    mundy::math::Array<double, 3> data{1.0, 2.0, 3.0};
    constexpr size_t shift = 1;
    return mundy::math::get_owning_shifted_accessor<double, shift>(data);
  };

  auto create_owning_shifted_accessor_delete_original_data_const = []() {
    mundy::math::Array<double, 3> data{1.0, 2.0, 3.0};
    constexpr size_t shift = 1;
    return mundy::math::get_owning_shifted_accessor<double, shift>(data);
  };

  auto create_owning_shifted_accessor_of_a_shifted_accessor_delete_original = []() {
    static mundy::math::Array<double, 3> data{1.0, 2.0, 3.0};
    constexpr size_t shift = 1;
    auto shifted_view = mundy::math::get_shifted_view<double, shift>(data);
    auto shifted_shifted_accessor = mundy::math::get_owning_shifted_accessor<double, shift>(shifted_view);
    return shifted_shifted_accessor;
  };

  // Reference view
  auto shifted_data = create_owning_shifted_accessor_delete_original_data();
  EXPECT_EQ(shifted_data[0], 2.0);
  EXPECT_EQ(shifted_data[1], 3.0);

  // Const reference view
  auto shifted_data_const = create_owning_shifted_accessor_delete_original_data_const();
  EXPECT_EQ(shifted_data_const[0], 2.0);
  EXPECT_EQ(shifted_data_const[1], 3.0);

  // Shifted shifted
  auto shifted_shifted_data = create_owning_shifted_accessor_of_a_shifted_accessor_delete_original();
  EXPECT_EQ(shifted_shifted_data[0], 3.0);
}

TEST(Views, StridedView) {
  auto run_test_nonconst_size_6 = [](auto &strided_data, const auto &original_data) {
    ASSERT_EQ(strided_data[0], original_data[0]);
    ASSERT_EQ(strided_data[1], original_data[2]);
    ASSERT_EQ(strided_data[2], original_data[4]);
    strided_data[0] = 7.0;
    strided_data[1] = 8.0;
    strided_data[2] = 9.0;
    ASSERT_EQ(original_data[0], 7.0);
    ASSERT_EQ(original_data[2], 8.0);
    ASSERT_EQ(original_data[4], 9.0);
  };

  auto run_test_const_size_6 = [](auto &strided_data, const auto &original_data) {
    ASSERT_EQ(strided_data[0], original_data[0]);
    ASSERT_EQ(strided_data[1], original_data[2]);
    ASSERT_EQ(strided_data[2], original_data[4]);
    const bool is_data_const = std::is_const_v<std::remove_reference_t<decltype(strided_data[0])>>;
    ASSERT_TRUE(is_data_const);
  };

  // Reference view
  mundy::math::Array<double, 6> data1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  constexpr size_t stride = 2;
  auto strided_data = mundy::math::get_strided_view<double, stride>(data1);
  run_test_nonconst_size_6(strided_data, data1);

  // Pointer view
  mundy::math::Array<double, 6> data2{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto strided_data_ptr = mundy::math::get_strided_view<double, stride>(data2.data());
  run_test_nonconst_size_6(strided_data_ptr, data2);

  // Const reference view
  const mundy::math::Array<double, 6> data3{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto strided_data_const = mundy::math::get_strided_view<double, stride>(data3);
  run_test_const_size_6(strided_data_const, data3.data());

  // Const pointer view
  const mundy::math::Array<double, 6> data4{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto strided_data_const_ptr = mundy::math::get_strided_view<double, stride>(data4.data());
  run_test_const_size_6(strided_data_const_ptr, data4.data());
}

TEST(Views, OwningStridedAccessor) {
  auto create_owning_strided_accessor_delete_original_data = []() {
    mundy::math::Array<double, 6> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    constexpr size_t stride = 2;
    return mundy::math::get_owning_strided_accessor<double, stride>(data);
  };

  auto create_owning_strided_accessor_delete_original_data_const = []() {
    mundy::math::Array<double, 6> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    constexpr size_t stride = 2;
    return mundy::math::get_owning_strided_accessor<double, stride>(data);
  };

  // Reference view
  auto strided_data = create_owning_strided_accessor_delete_original_data();
  EXPECT_EQ(strided_data[0], 1.0);
  EXPECT_EQ(strided_data[1], 3.0);
  EXPECT_EQ(strided_data[2], 5.0);

  // Const reference view
  auto strided_data_const = create_owning_strided_accessor_delete_original_data_const();
  EXPECT_EQ(strided_data_const[0], 1.0);
  EXPECT_EQ(strided_data_const[1], 3.0);
  EXPECT_EQ(strided_data_const[2], 5.0);
}

TEST(Views, TransposedView) {
  auto run_test_nonconst_size_3x3 = [](auto &transposed_data, const auto &original_data) {
    ASSERT_EQ(transposed_data[0], original_data[0]);
    ASSERT_EQ(transposed_data[1], original_data[3]);
    ASSERT_EQ(transposed_data[2], original_data[6]);
    ASSERT_EQ(transposed_data[3], original_data[1]);
    ASSERT_EQ(transposed_data[4], original_data[4]);
    ASSERT_EQ(transposed_data[5], original_data[7]);
    ASSERT_EQ(transposed_data[6], original_data[2]);
    ASSERT_EQ(transposed_data[7], original_data[5]);
    ASSERT_EQ(transposed_data[8], original_data[8]);

    transposed_data[0] = 2.0;
    transposed_data[1] = 5.0;
    transposed_data[2] = 8.0;
    transposed_data[3] = 3.0;
    transposed_data[4] = 6.0;
    transposed_data[5] = 9.0;
    transposed_data[6] = 4.0;
    transposed_data[7] = 7.0;
    transposed_data[8] = 10.0;
    ASSERT_EQ(original_data[0], 2.0);
    ASSERT_EQ(original_data[1], 3.0);
    ASSERT_EQ(original_data[2], 4.0);
    ASSERT_EQ(original_data[3], 5.0);
    ASSERT_EQ(original_data[4], 6.0);
    ASSERT_EQ(original_data[5], 7.0);
    ASSERT_EQ(original_data[6], 8.0);
    ASSERT_EQ(original_data[7], 9.0);
    ASSERT_EQ(original_data[8], 10.0);
  };

  auto run_test_const_size_3x3 = [](auto &transposed_data, const auto &original_data) {
    ASSERT_EQ(transposed_data[0], original_data[0]);
    ASSERT_EQ(transposed_data[1], original_data[3]);
    ASSERT_EQ(transposed_data[2], original_data[6]);
    ASSERT_EQ(transposed_data[3], original_data[1]);
    ASSERT_EQ(transposed_data[4], original_data[4]);
    ASSERT_EQ(transposed_data[5], original_data[7]);
    ASSERT_EQ(transposed_data[6], original_data[2]);
    ASSERT_EQ(transposed_data[7], original_data[5]);
    ASSERT_EQ(transposed_data[8], original_data[8]);
    const bool is_data_const = std::is_const_v<std::remove_reference_t<decltype(transposed_data[0])>>;
    ASSERT_TRUE(is_data_const);
  };

  // Reference view
  mundy::math::Array<double, 9> data1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  auto transposed_data = mundy::math::get_transposed_view<double, 3, 3>(data1);
  run_test_nonconst_size_3x3(transposed_data, data1);

  // Pointer view
  mundy::math::Array<double, 9> data2{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  auto transposed_data_ptr = mundy::math::get_transposed_view<double, 3, 3>(data2.data());
  run_test_nonconst_size_3x3(transposed_data_ptr, data2);

  // Const reference view
  const mundy::math::Array<double, 9> data3{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  auto transposed_data_const = mundy::math::get_transposed_view<double, 3, 3>(data3);
  run_test_const_size_3x3(transposed_data_const, data3.data());

  // Const pointer view
  const mundy::math::Array<double, 9> data4{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  auto transposed_data_const_ptr = mundy::math::get_transposed_view<double, 3, 3>(data4.data());
  run_test_const_size_3x3(transposed_data_const_ptr, data4.data());
}

TEST(Views, OwningTransposedAccessor) {
  auto create_owning_transposed_accessor_delete_original_data = []() {
    mundy::math::Array<double, 9> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    return mundy::math::get_owning_transposed_accessor<double, 3, 3>(data);
  };

  auto create_owning_transposed_accessor_delete_original_data_const = []() {
    mundy::math::Array<double, 9> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    return mundy::math::get_owning_transposed_accessor<double, 3, 3>(data);
  };

  // Reference view
  auto transposed_data = create_owning_transposed_accessor_delete_original_data();
  ASSERT_EQ(transposed_data[0], 1.0);
  ASSERT_EQ(transposed_data[1], 4.0);
  ASSERT_EQ(transposed_data[2], 7.0);
  ASSERT_EQ(transposed_data[3], 2.0);
  ASSERT_EQ(transposed_data[4], 5.0);
  ASSERT_EQ(transposed_data[5], 8.0);
  ASSERT_EQ(transposed_data[6], 3.0);
  ASSERT_EQ(transposed_data[7], 6.0);
  ASSERT_EQ(transposed_data[8], 9.0);

  // Const reference view
  auto transposed_data_const = create_owning_transposed_accessor_delete_original_data_const();
  ASSERT_EQ(transposed_data_const[0], 1.0);
  ASSERT_EQ(transposed_data_const[1], 4.0);
  ASSERT_EQ(transposed_data_const[2], 7.0);
  ASSERT_EQ(transposed_data_const[3], 2.0);
  ASSERT_EQ(transposed_data_const[4], 5.0);
  ASSERT_EQ(transposed_data_const[5], 8.0);
  ASSERT_EQ(transposed_data_const[6], 3.0);
  ASSERT_EQ(transposed_data_const[7], 6.0);
  ASSERT_EQ(transposed_data_const[8], 9.0);
}

TEST(Views, MixedViews) {
  auto create_owning_shifted_accessor_of_a_shifted_accessor_delete_original = []() {
    static mundy::math::Array<double, 3> data{1.0, 2.0, 3.0};
    constexpr size_t shift = 1;
    auto shifted_view = mundy::math::get_shifted_view<double, shift>(data);
    auto shifted_shifted_accessor = mundy::math::get_owning_shifted_accessor<double, shift>(shifted_view);
    return shifted_shifted_accessor;
  };

  auto create_owning_stided_accessor_of_a_shifted_accessor_delete_original = []() {
    static mundy::math::Array<double, 9> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    constexpr size_t shift = 1;
    auto shifted_view = mundy::math::get_shifted_view<double, shift>(data);
    constexpr size_t stride = 3;
    auto strided_shifted_accessor = mundy::math::get_owning_strided_accessor<double, stride>(shifted_view);
    return strided_shifted_accessor;
  };

  // Shifted shifted
  auto shifted_shifted_data = create_owning_shifted_accessor_of_a_shifted_accessor_delete_original();
  EXPECT_EQ(shifted_shifted_data[0], 3.0);

  // Strided shifted
  auto strided_shifted_data = create_owning_stided_accessor_of_a_shifted_accessor_delete_original();
  EXPECT_EQ(strided_shifted_data[0], 2.0);
  EXPECT_EQ(strided_shifted_data[1], 5.0);
  EXPECT_EQ(strided_shifted_data[2], 8.0);
}

}  // namespace

}  // namespace math

}  // namespace mundy
