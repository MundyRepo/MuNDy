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

//! \file MatrixVectorQuaternion.cpp
/// \brief Performance test the use of matrices, vectors, and quaternions.
#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <fstream>    // for std::ofstream
#include <iostream>   // for std::cout, std::endl
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <vector>     // for std::vector

// External
#include "nanobench.h"

// Trilinos
#include <Kokkos_Core.hpp>                 // for Kokkos::initialize, Kokkos::finalize
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_math/Array.hpp>       // for mundy::math::Array
#include <mundy_math/Matrix.hpp>      // for mundy::math::Matrix
#include <mundy_math/Quaternion.hpp>  // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>   // for mundy::math::get_relaxed_tolerance
#include <mundy_math/Vector.hpp>      // for mundy::math::Vector

/* Test design.
Compare the performance of performing different operations between matrices, vectors, and quaternions against
the inline implementation of the operations.

To start, we will create a three flattened std::vector meant to represent a set of 3x3 matrices, 3-vectors, and
quaternions. We will then perform the following operations on each set of data:
  BLAS (vec, mat, and quat): y = alpha x + beta y
  Mat-vec: v2 = m1 * v1
  Complex vector ops: v3 = v2.cross(v1.dot(v2) * v1)
  Quaternion rotation: q3 = q1 * q2
*/

using View1D = Kokkos::View<double *, Kokkos::DefaultExecutionSpace>;

void randomize(View1D &x) {
  // Host view to fill with random values
  auto host_view = Kokkos::create_mirror_view(x);
  for (size_t i = 0; i < x.extent(0); ++i) {
    host_view(i) = static_cast<double>(rand()) / RAND_MAX;
  }
  // Copy back to device
  Kokkos::deep_copy(x, host_view);
}

void test_vector3_blas(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 3;
  Kokkos::parallel_for(
      "test_vector3_blas", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const auto x_view = mundy::math::get_vector_view<double, 3>(x.data() + 3 * i);
        auto y_view = mundy::math::get_vector_view<double, 3>(y.data() + 3 * i);
        y_view = alpha * x_view + beta * y_view;
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_vector3_blas_no_views(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 3;
  Kokkos::parallel_for(
      "test_vector3_blas_no_views", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy into vectors
        const mundy::math::Vector<double, 3> x_vec(x(3 * i + 0), x(3 * i + 1), x(3 * i + 2));
        mundy::math::Vector<double, 3> y_vec(y(3 * i + 0), y(3 * i + 1), y(3 * i + 2));
        y_vec = alpha * x_vec + beta * y_vec;

        // Copy back into the result
        y(3 * i + 0) = y_vec[0];
        y(3 * i + 1) = y_vec[1];
        y(3 * i + 2) = y_vec[2];
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_vector3_blas_direct(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 3;
  Kokkos::parallel_for(
      "test_vector3_blas_direct", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy y/x into an intermediary to make CUDA happy
        double y0 = y(3 * i + 0);
        double y1 = y(3 * i + 1);
        double y2 = y(3 * i + 2);
        double x0 = x(3 * i + 0);
        double x1 = x(3 * i + 1);
        double x2 = x(3 * i + 2);

        // Perform the operation
        y(3 * i + 0) = alpha * x0 + beta * y0;
        y(3 * i + 1) = alpha * x1 + beta * y1;
        y(3 * i + 2) = alpha * x2 + beta * y2;
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_matrix3_blas(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 9;
  Kokkos::parallel_for(
      "test_matrix3_blas", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const auto x_view = mundy::math::get_matrix_view<double, 3, 3>(x.data() + 9 * i);
        auto y_view = mundy::math::get_matrix_view<double, 3, 3>(y.data() + 9 * i);
        y_view = alpha * x_view + beta * y_view;
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_matrix3_blas_no_views(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 9;
  Kokkos::parallel_for(
      "test_matrix3_blas_no_views", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy into matrices
        const mundy::math::Matrix<double, 3, 3> x_mat(x(9 * i + 0), x(9 * i + 1), x(9 * i + 2), x(9 * i + 3),
                                                      x(9 * i + 4), x(9 * i + 5), x(9 * i + 6), x(9 * i + 7),
                                                      x(9 * i + 8));
        mundy::math::Matrix<double, 3, 3> y_mat(y(9 * i + 0), y(9 * i + 1), y(9 * i + 2), y(9 * i + 3), y(9 * i + 4),
                                                y(9 * i + 5), y(9 * i + 6), y(9 * i + 7), y(9 * i + 8));
        y_mat = alpha * x_mat + beta * y_mat;

        // Copy back into the result
        y(9 * i + 0) = y_mat(0, 0);
        y(9 * i + 1) = y_mat(0, 1);
        y(9 * i + 2) = y_mat(0, 2);
        y(9 * i + 3) = y_mat(1, 0);
        y(9 * i + 4) = y_mat(1, 1);
        y(9 * i + 5) = y_mat(1, 2);
        y(9 * i + 6) = y_mat(2, 0);
        y(9 * i + 7) = y_mat(2, 1);
        y(9 * i + 8) = y_mat(2, 2);
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_matrix3_blas_direct(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 9;
  Kokkos::parallel_for(
      "test_matrix3_blas_direct", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy y/x into an intermediary to make CUDA happy
        double y00 = y(9 * i + 0);
        double y01 = y(9 * i + 1);
        double y02 = y(9 * i + 2);
        double y10 = y(9 * i + 3);
        double y11 = y(9 * i + 4);
        double y12 = y(9 * i + 5);
        double y20 = y(9 * i + 6);
        double y21 = y(9 * i + 7);
        double y22 = y(9 * i + 8);
        double x00 = x(9 * i + 0);
        double x01 = x(9 * i + 1);
        double x02 = x(9 * i + 2);
        double x10 = x(9 * i + 3);
        double x11 = x(9 * i + 4);
        double x12 = x(9 * i + 5);
        double x20 = x(9 * i + 6);
        double x21 = x(9 * i + 7);
        double x22 = x(9 * i + 8);

        // Perform the operation
        y(9 * i + 0) = alpha * x00 + beta * y00;
        y(9 * i + 1) = alpha * x01 + beta * y01;
        y(9 * i + 2) = alpha * x02 + beta * y02;
        y(9 * i + 3) = alpha * x10 + beta * y10;
        y(9 * i + 4) = alpha * x11 + beta * y11;
        y(9 * i + 5) = alpha * x12 + beta * y12;
        y(9 * i + 6) = alpha * x20 + beta * y20;
        y(9 * i + 7) = alpha * x21 + beta * y21;
        y(9 * i + 8) = alpha * x22 + beta * y22;
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_quaternion_blas(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 4;
  Kokkos::parallel_for(
      "test_quaternion_blas", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const auto x_view = mundy::math::get_quaternion_view<double>(x.data() + 4 * i);
        auto y_view = mundy::math::get_quaternion_view<double>(y.data() + 4 * i);
        y_view = alpha * x_view + beta * y_view;
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_quaternion_blas_no_views(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 4;
  Kokkos::parallel_for(
      "test_quaternion_blas_no_views", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy into quaternions
        const mundy::math::Quaterniond x_quat(x(4 * i + 0), x(4 * i + 1), x(4 * i + 2), x(4 * i + 3));
        mundy::math::Quaterniond y_quat(y(4 * i + 0), y(4 * i + 1), y(4 * i + 2), y(4 * i + 3));
        y_quat = alpha * x_quat + beta * y_quat;

        // Copy back into the result
        y(4 * i + 0) = y_quat.w();
        y(4 * i + 1) = y_quat.x();
        y(4 * i + 2) = y_quat.y();
        y(4 * i + 3) = y_quat.z();
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_quaternion_blas_direct(const double alpha, const View1D &x, const double beta, View1D &y) {
  const size_t num_entities = x.extent(0) / 4;
  Kokkos::parallel_for(
      "test_quaternion_blas_direct", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy y/x into an intermediary to make CUDA happy
        double y0 = y(4 * i + 0);
        double y1 = y(4 * i + 1);
        double y2 = y(4 * i + 2);
        double y3 = y(4 * i + 3);
        double x0 = x(4 * i + 0);
        double x1 = x(4 * i + 1);
        double x2 = x(4 * i + 2);
        double x3 = x(4 * i + 3);

        // Perform the operation
        y(4 * i + 0) = alpha * x0 + beta * y0;
        y(4 * i + 1) = alpha * x1 + beta * y1;
        y(4 * i + 2) = alpha * x2 + beta * y2;
        y(4 * i + 3) = alpha * x3 + beta * y3;
      });
  Kokkos::fence();                          // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(y);  // Prevent optimization of the result
}

void test_mat_vec(const View1D &m, const View1D &v, View1D &result) {
  const size_t num_entities = m.extent(0) / 9;
  Kokkos::parallel_for(
      "test_mat_vec", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const auto m_view = mundy::math::get_matrix_view<double, 3, 3>(m.data() + 9 * i);
        const auto v_view = mundy::math::get_vector_view<double, 3>(v.data() + 3 * i);
        auto result_view = mundy::math::get_vector_view<double, 3>(result.data() + 3 * i);
        result_view = m_view * v_view;
      });
  Kokkos::fence();                               // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(result);  // Prevent optimization of the result
}

void test_mat_vec_no_views(const View1D &m, const View1D &v, View1D &result) {
  const size_t num_entities = m.extent(0) / 9;
  Kokkos::parallel_for(
      "test_mat_vec_no_views", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy into matrices and vectors
        const mundy::math::Matrix<double, 3, 3> m_mat(m(9 * i + 0), m(9 * i + 1), m(9 * i + 2), m(9 * i + 3),
                                                      m(9 * i + 4), m(9 * i + 5), m(9 * i + 6), m(9 * i + 7),
                                                      m(9 * i + 8));
        const mundy::math::Vector<double, 3> v_vec(v(3 * i + 0), v(3 * i + 1), v(3 * i + 2));
        mundy::math::Vector<double, 3> result_vec = m_mat * v_vec;

        // Copy back into the result
        result(3 * i + 0) = result_vec[0];
        result(3 * i + 1) = result_vec[1];
        result(3 * i + 2) = result_vec[2];
      });
  Kokkos::fence();                               // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(result);  // Prevent optimization of the result
}

void test_mat_vec_direct(const View1D &m, const View1D &v, View1D &result) {
  const size_t num_entities = m.extent(0) / 9;
  Kokkos::parallel_for(
      "test_mat_vec_direct", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy m/v into an intermediary to make CUDA happy
        double m00 = m(9 * i + 0);
        double m01 = m(9 * i + 1);
        double m02 = m(9 * i + 2);
        double m10 = m(9 * i + 3);
        double m11 = m(9 * i + 4);
        double m12 = m(9 * i + 5);
        double m20 = m(9 * i + 6);
        double m21 = m(9 * i + 7);
        double m22 = m(9 * i + 8);
        double v0 = v(3 * i + 0);
        double v1 = v(3 * i + 1);
        double v2 = v(3 * i + 2);
        // Perform the operation
        result(3 * i + 0) = m00 * v0 + m01 * v1 + m02 * v2;
        result(3 * i + 1) = m10 * v0 + m11 * v1 + m12 * v2;
        result(3 * i + 2) = m20 * v0 + m21 * v1 + m22 * v2;
      });
  Kokkos::fence();                               // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(result);  // Prevent optimization of the result
}

void test_complex_vector_ops(const View1D &v1, const View1D &v2, View1D &v3) {
  const size_t num_entities = v1.extent(0) / 3;
  Kokkos::parallel_for(
      "test_complex_vector_ops", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const auto v1_view = mundy::math::get_vector_view<double, 3>(v1.data() + 3 * i);
        const auto v2_view = mundy::math::get_vector_view<double, 3>(v2.data() + 3 * i);
        auto v3_view = mundy::math::get_vector_view<double, 3>(v3.data() + 3 * i);
        v3_view = mundy::math::cross(v2_view, mundy::math::dot(v1_view, v2_view) * v1_view);
      });
  Kokkos::fence();                           // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(v3);  // Prevent optimization of the result
}

void test_complex_vector_ops_no_views(const View1D &v1, const View1D &v2, View1D &v3) {
  const size_t num_entities = v1.extent(0) / 3;
  Kokkos::parallel_for(
      "test_complex_vector_ops_no_views", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy into vectors
        const mundy::math::Vector<double, 3> v1_vec(v1(3 * i + 0), v1(3 * i + 1), v1(3 * i + 2));
        const mundy::math::Vector<double, 3> v2_vec(v2(3 * i + 0), v2(3 * i + 1), v2(3 * i + 2));
        const mundy::math::Vector<double, 3> v3_vec =
            mundy::math::cross(v2_vec, mundy::math::dot(v1_vec, v2_vec) * v1_vec);

        // Copy back into the result
        v3(3 * i + 0) = v3_vec[0];
        v3(3 * i + 1) = v3_vec[1];
        v3(3 * i + 2) = v3_vec[2];
      });
  Kokkos::fence();                           // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(v3);  // Prevent optimization of the result
}

void test_complex_vector_ops_direct(const View1D &v1, const View1D &v2, View1D &v3) {
  const size_t num_entities = v1.extent(0) / 3;
  Kokkos::parallel_for(
      "test_complex_vector_ops_direct", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        //   Complex vector ops: v3 = v2.cross(v1.dot(v2) * v1)
        //
        // Copy v1/v2 into an intermediary to make CUDA happy
        double v1_0 = v1(3 * i + 0);
        double v1_1 = v1(3 * i + 1);
        double v1_2 = v1(3 * i + 2);
        double v2_0 = v2(3 * i + 0);
        double v2_1 = v2(3 * i + 1);
        double v2_2 = v2(3 * i + 2);

        // Perform the operation
        const double v1_dot_v2 = v1_0 * v2_0 + v1_1 * v2_1 + v1_2 * v2_2;
        const double v1_dot_v2_times_v1[3] = {v1_dot_v2 * v1_0, v1_dot_v2 * v1_1, v1_dot_v2 * v1_2};
        v3(3 * i + 0) = v2_1 * v1_dot_v2_times_v1[2] - v2_2 * v1_dot_v2_times_v1[1];
        v3(3 * i + 1) = v2_2 * v1_dot_v2_times_v1[0] - v2_0 * v1_dot_v2_times_v1[2];
        v3(3 * i + 2) = v2_0 * v1_dot_v2_times_v1[1] - v2_1 * v1_dot_v2_times_v1[0];
      });
  Kokkos::fence();                           // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(v3);  // Prevent optimization of the result
}

void test_quaternion_rotation(const View1D &q1, const View1D &q2, View1D &q3) {
  const size_t num_entities = q1.extent(0) / 4;
  Kokkos::parallel_for(
      "test_quaternion_rotation", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const auto q1_view = mundy::math::get_quaternion_view<double>(q1.data() + 4 * i);
        const auto q2_view = mundy::math::get_quaternion_view<double>(q2.data() + 4 * i);
        auto q3_view = mundy::math::get_quaternion_view<double>(q3.data() + 4 * i);
        q3_view = q1_view * q2_view;
      });
  Kokkos::fence();                           // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(q3);  // Prevent optimization of the result
}

void test_quaternion_rotation_no_views(const View1D &q1, const View1D &q2, View1D &q3) {
  const size_t num_entities = q1.extent(0) / 4;
  Kokkos::parallel_for(
      "test_quaternion_rotation_no_views", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        // Copy into a quaternion
        const mundy::math::Quaterniond q1_quat(q1(4 * i + 0), q1(4 * i + 1), q1(4 * i + 2), q1(4 * i + 3));
        const mundy::math::Quaterniond q2_quat(q2(4 * i + 0), q2(4 * i + 1), q2(4 * i + 2), q2(4 * i + 3));
        const mundy::math::Quaterniond q3_quat = q1_quat * q2_quat;

        // Copy back into the result
        q3(4 * i + 0) = q3_quat.w();
        q3(4 * i + 1) = q3_quat.x();
        q3(4 * i + 2) = q3_quat.y();
        q3(4 * i + 3) = q3_quat.z();
      });
  Kokkos::fence();                           // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(q3);  // Prevent optimization of the result
}

void test_quaternion_rotation_direct(const View1D &q1, const View1D &q2, View1D &q3) {
  const size_t num_entities = q1.extent(0) / 4;
  Kokkos::parallel_for(
      "test_quaternion_rotation_direct", Kokkos::RangePolicy<>(0, num_entities), KOKKOS_LAMBDA(const size_t i) {
        const double q1_w = q1(4 * i + 0);
        const double q1_x = q1(4 * i + 1);
        const double q1_y = q1(4 * i + 2);
        const double q1_z = q1(4 * i + 3);
        const double q2_w = q2(4 * i + 0);
        const double q2_x = q2(4 * i + 1);
        const double q2_y = q2(4 * i + 2);
        const double q2_z = q2(4 * i + 3);
        q3(4 * i + 0) = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z;
        q3(4 * i + 1) = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y;
        q3(4 * i + 2) = q1_w * q2_y - q1_x * q2_z + q1_y * q2_w + q1_z * q2_x;
        q3(4 * i + 3) = q1_w * q2_z + q1_x * q2_y - q1_y * q2_x + q1_z * q2_w;
      });
  Kokkos::fence();                           // Ensure all operations are complete before returning
  ankerl::nanobench::doNotOptimizeAway(q3);  // Prevent optimization of the result
}

template <typename OurViewFunc, typename OurNoViewFunc, typename DirectFunc>
void time_test(const std::string &test_name, const OurViewFunc &our_view_func, const OurNoViewFunc &our_no_view_func,
               const DirectFunc &direct_func) {
  ankerl::nanobench::Bench bench;
  bench.relative(true).title(test_name).unit("op").performanceCounters(true).minEpochIterations(1000);

  bench.run("with views", [&] {
    our_view_func();
    Kokkos::fence();
  });

  bench.run("no views", [&] {
    our_no_view_func();
    Kokkos::fence();
  });

  bench.run("direct", [&] {
    direct_func();
    Kokkos::fence();
  });
}

int main(int argc, char **argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    const size_t num_entities = 1000000;
    const double alpha = 2.0;
    const double beta = 3.0;
    View1D v1("v1", 3 * num_entities);
    View1D v2("v2", 3 * num_entities);
    View1D m1("m1", 9 * num_entities);
    View1D m2("m2", 9 * num_entities);
    View1D q1("q1", 4 * num_entities);
    View1D q2("q2", 4 * num_entities);
    View1D q3("q3", 4 * num_entities);

    randomize(v1);
    randomize(v2);
    randomize(m1);
    randomize(m2);
    randomize(q1);
    randomize(q2);
    randomize(q3);

    time_test(
        "Vector3 BLAS",                                              //
        [&]() { test_vector3_blas(alpha, v1, beta, v2); },           //
        [&]() { test_vector3_blas_no_views(alpha, v1, beta, v2); },  //
        [&]() { test_vector3_blas_direct(alpha, v1, beta, v2); });

    time_test(
        "Matrix3 BLAS",                                              //
        [&]() { test_matrix3_blas(alpha, m1, beta, m2); },           //
        [&]() { test_matrix3_blas_no_views(alpha, m1, beta, m2); },  //
        [&]() { test_matrix3_blas_direct(alpha, m1, beta, m2); });

    time_test(
        "Quaternion BLAS",                                              //
        [&]() { test_quaternion_blas(alpha, q1, beta, q2); },           //
        [&]() { test_quaternion_blas_no_views(alpha, q1, beta, q2); },  //
        [&]() { test_quaternion_blas_direct(alpha, q1, beta, q2); });

    time_test(
        "Mat-vec",                                     //
        [&]() { test_mat_vec(m1, v1, v2); },           //
        [&]() { test_mat_vec_no_views(m1, v1, v2); },  //
        [&]() { test_mat_vec_direct(m1, v1, v2); });

    time_test(
        "Complex vector ops",                                     //
        [&]() { test_complex_vector_ops(v1, v2, v1); },           //
        [&]() { test_complex_vector_ops_no_views(v1, v2, v1); },  //
        [&]() { test_complex_vector_ops_direct(v1, v2, v1); });

    time_test(
        "Quaternion rotation",                                     //
        [&]() { test_quaternion_rotation(q1, q2, q3); },           //
        [&]() { test_quaternion_rotation_no_views(q1, q2, q3); },  //
        [&]() { test_quaternion_rotation_direct(q1, q2, q3); });
    std::cout << "All tests completed successfully." << std::endl;
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
