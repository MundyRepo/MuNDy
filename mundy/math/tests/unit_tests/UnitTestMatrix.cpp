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
#include <algorithm>  // for std::max
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <future>
#include <map>     // for std::map
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <sstream>
#include <stdexcept>  // for std::logic_error, std::invalid_argument
#include <string>     // for std::string
#include <thread>
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move
#include <vector>       // for std::vector

// Mundy libs
#include <mundy_math/Matrix.hpp>     // for mundy::math::Matrix
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_relaxed_zero_tolerance
#include <mundy_math/Vector.hpp>     // for mundy::math::Vector

namespace mundy {

namespace math {

namespace {

//! \name Helper functions
//@{

template <typename U, typename T>
void is_close_debug(const U& a, const T& b, const std::string& message_if_fail = "")
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
{
  EXPECT_TRUE(is_approx_close(a, b)) << message_if_fail;
}

template <typename U, size_t R1, size_t C1, ValidAccessor<U> A1, typename T, size_t R2, size_t C2,
          ValidAccessor<T> A2>
void is_close_debug(const AMatrix<U, R1, C1, A1>& m1, const AMatrix<T, R2, C2, A2>& m2,
                    const std::string& message_if_fail = "") {
  EXPECT_TRUE(is_approx_close(m1, m2)) << message_if_fail;
}

template <typename U, size_t N1, ValidAccessor<U> A1, typename T, size_t N2, ValidAccessor<T> A2>
void is_close_debug(const AVector<U, N1, A1>& v1, const AVector<T, N2, A2>& v2,
                    const std::string& message_if_fail = "") {
  EXPECT_TRUE(is_approx_close(v1, v2)) << message_if_fail;
}

template <typename T, size_t N, size_t M, ValidAccessor<T> A>
void expect_matrix_close(const AMatrix<T, N, M, A>& m, const std::initializer_list<double>& expected) {
  ASSERT_EQ(expected.size(), N * M);
  size_t i = 0;
  for (double value : expected) {
    EXPECT_TRUE(is_approx_close(m[i], value));
    ++i;
  }
}

template <size_t R, size_t C>
Matrix<double, R, C> make_matrix_linear(double start = 1.0, double step = 1.0) {
  Matrix<double, R, C> m;
  double value = start;
  for (size_t i = 0; i < R * C; ++i) {
    m(i) = value;
    value += step;
  }
  return m;
}

template <size_t N>
Vector<double, N> make_vector_linear(double start = 1.0, double step = 1.0) {
  Vector<double, N> v;
  double value = start;
  for (size_t i = 0; i < N; ++i) {
    v[i] = value;
    value += step;
  }
  return v;
}

template <size_t R, size_t C>
double matmul_entry(const Matrix<double, R, C>& lhs, const Matrix<double, C, R>& rhs, size_t i, size_t j) {
  double accum = 0.0;
  for (size_t k = 0; k < C; ++k) {
    accum += lhs(i, k) * rhs(k, j);
  }
  return accum;
}

template <typename T, size_t R, size_t C>
bool check_matrix_atomic_op_load_store_false_positive() {
  Matrix<T, R, C> finished(T(0));
  auto func_no_atomic = [&finished]() {
    bool hit_max_loops = false;
    size_t max_loops = 1'000'000'000;
    size_t i = 0;
    while (!(finished[0] == T(1))) {
      if (i > max_loops) {
        hit_max_loops = true;
        break;
      }
      ++i;
    }
    return hit_max_loops;
  };

  auto result = std::async(std::launch::async, func_no_atomic);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  finished[0] = T(1);
  bool false_positive = !result.get();
  return false_positive;
}

template <typename T, size_t R, size_t C>
bool check_matrix_atomic_op_add_sub_false_positive() {
  int num_threads = 8;
  int num_iterations = 2001;

  Matrix<T, R, C> m_add_neg(T(0));
  Matrix<T, R, C> m_sub_neg(T(0));

  auto thread_func = [&]() {
    for (int i = 0; i < num_iterations; ++i) {
      m_add_neg += T(1);
      m_sub_neg -= T(1);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  bool add_false_positive = true;
  bool sub_false_positive = true;
  const T expected_add = static_cast<T>(num_threads * num_iterations);
  const T expected_sub = static_cast<T>(-num_threads * num_iterations);
  for (size_t i = 0; i < R * C; ++i) {
    add_false_positive = add_false_positive && is_approx_close(m_add_neg(i), expected_add);
    sub_false_positive = sub_false_positive && is_approx_close(m_sub_neg(i), expected_sub);
  }
  return add_false_positive || sub_false_positive;
}

//@}

//! \name GTEST typed test fixtures
//@{

template <typename U1, typename U2>
struct TypePair {
  using T1 = U1;
  using T2 = U2;
};

template <typename Pair>
class MatrixPairwiseTypeTest : public ::testing::Test {};

using MyTypePairs = ::testing::Types<TypePair<int, float>, TypePair<int, double>, TypePair<float, double>,
                                     TypePair<int, int>, TypePair<float, float>, TypePair<double, double>>;
TYPED_TEST_SUITE(MatrixPairwiseTypeTest, MyTypePairs);

template <typename U, size_t R, size_t C>
struct MatrixCase {
  using scalar_t = U;
  static constexpr size_t rows = R;
  static constexpr size_t cols = C;
};

template <typename Shape>
class NonSquareMatrixShapeTest : public ::testing::Test {};

using NonSquareCases = ::testing::Types<MatrixCase<int, 2, 3>, MatrixCase<float, 2, 3>, MatrixCase<double, 2, 3>,
                                        MatrixCase<int, 3, 2>, MatrixCase<float, 3, 2>, MatrixCase<double, 3, 2>>;
TYPED_TEST_SUITE(NonSquareMatrixShapeTest, NonSquareCases);

//@}

//! \name Matrix non-square construction and access
//@{

TYPED_TEST(NonSquareMatrixShapeTest, ConstructionAccessAndTranspose) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  auto m = make_matrix_linear<R, C>(T(1), T(1));

  for (size_t i = 0; i < R * C; ++i) {
    EXPECT_TRUE(is_approx_close(m(i), static_cast<T>(i + 1)));
  }

  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      EXPECT_TRUE(is_approx_close(m(i, j), static_cast<T>(i * C + j + 1)));
    }
  }

  auto m_t = transpose(m);
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      EXPECT_TRUE(is_approx_close(m_t(j, i), m(i, j)));
    }
  }
}

//@}

//! \name Matrix non-square setters and views
//@{

TYPED_TEST(NonSquareMatrixShapeTest, SettersViewsMinorAndHelperOwnership) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  auto m = make_matrix_linear<R, C>(T(1), T(1));

  auto row0_copy = m.copy_row(0);
  row0_copy[0] = T(999);
  EXPECT_TRUE(is_approx_close(m(0, 0), T(1)));

  auto col0_copy = m.copy_column(0);
  col0_copy[0] = T(-999);
  EXPECT_TRUE(is_approx_close(m(0, 0), T(1)));

  auto new_row = make_vector_linear<C>(T(100), T(1));
  m.set_row(0, new_row);
  for (size_t j = 0; j < C; ++j) {
    EXPECT_TRUE(is_approx_close(m(0, j), new_row[j]));
  }

  auto new_col = make_vector_linear<R>(T(200), T(1));
  m.set_column(0, new_col);
  for (size_t i = 0; i < R; ++i) {
    EXPECT_TRUE(is_approx_close(m(i, 0), new_col[i]));
  }

  auto row1_view = m.template view_row<1>();
  for (size_t j = 0; j < C; ++j) {
    row1_view[j] = T(300 + static_cast<int>(j));
    EXPECT_TRUE(is_approx_close(m(1, j), T(300 + static_cast<int>(j))));
  }

  auto col1_view = m.template view_column<1>();
  for (size_t i = 0; i < R; ++i) {
    col1_view[i] = T(-50 - static_cast<int>(i));
    EXPECT_TRUE(is_approx_close(m(i, 1), T(-50 - static_cast<int>(i))));
  }

  auto diag_view = m.view_diagonal();
  for (size_t i = 0; i < std::min(R, C); ++i) {
    diag_view[i] = T(500 + static_cast<int>(i));
    EXPECT_TRUE(is_approx_close(m(i, i), T(500 + static_cast<int>(i))));
  }

  auto transposed_view = m.view_transpose();
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      EXPECT_TRUE(is_approx_close(transposed_view(j, i), m(i, j)));
    }
  }

  const auto minor00 = m.template view_minor<0, 0>();
  EXPECT_EQ(minor00.num_rows, R - 1);
  EXPECT_EQ(minor00.num_cols, C - 1);
  for (size_t i = 0; i < minor00.num_rows; ++i) {
    for (size_t j = 0; j < minor00.num_cols; ++j) {
      EXPECT_TRUE(is_approx_close(minor00(i, j), m(i + 1, j + 1)));
    }
  }

  std::array<T, R * C> data{};
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<T>(i + 1);
  }
  auto view = get_matrix_view<T, R, C>(data.data());
  auto owning = Matrix<T, R, C>(view);
  data[0] = T(777);
  EXPECT_TRUE(is_approx_close(view(0, 0), T(777)));
  EXPECT_TRUE(is_approx_close(owning(0, 0), T(1)));
}

//@}

//! \name Matrix non-square arithmetic
//@{

TYPED_TEST(NonSquareMatrixShapeTest, Arithmetic) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  auto a = make_matrix_linear<R, C>(T(1), T(1));
  auto b = make_matrix_linear<R, C>(T(2), T(2));

  auto plus = a + b;
  auto minus = b - a;
  auto scaled = a * T(2);
  auto shifted = T(3) + a;
  auto emul = elementwise_mul(a, b);
  auto ediv = elementwise_div(b, make_matrix_linear<R, C>(T(1), T(1)));

  for (size_t i = 0; i < R * C; ++i) {
    is_close_debug(plus(i), a(i) + b(i), "Matrix addition failed.");
    is_close_debug(minus(i), b(i) - a(i), "Matrix subtraction failed.");
    const T a_i = static_cast<T>(a(i));
    is_close_debug(scaled(i), static_cast<T>(T(2) * a_i), "Matrix scalar multiplication failed.");
    is_close_debug(shifted(i), static_cast<T>(T(3) + a_i), "Scalar-matrix addition failed.");
    is_close_debug(emul(i), a(i) * b(i), "Elementwise multiplication failed.");
    is_close_debug(ediv(i), T(2), "Elementwise division failed.");
  }

  auto v_right = make_vector_linear<C>(T(1), T(1));
  auto mv = a * v_right;
  for (size_t i = 0; i < R; ++i) {
    T expected = T(0);
    for (size_t j = 0; j < C; ++j) {
      expected = static_cast<T>(expected + static_cast<T>(a(i, j)) * static_cast<T>(v_right[j]));
    }
    is_close_debug(mv[i], expected, "Matrix-vector multiplication failed.");
  }

  auto v_left = make_vector_linear<R>(T(1), T(1));
  auto vm = v_left * a;
  for (size_t j = 0; j < C; ++j) {
    T expected = T(0);
    for (size_t i = 0; i < R; ++i) {
      expected = static_cast<T>(expected + static_cast<T>(v_left[i]) * static_cast<T>(a(i, j)));
    }
    is_close_debug(vm[j], expected, "Vector-matrix multiplication failed.");
  }

  auto a_t = transpose(a);
  auto mm = a * a_t;
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < R; ++j) {
      is_close_debug(mm(i, j), matmul_entry(a, a_t, i, j), "Matrix-matrix multiplication failed.");
    }
  }
}

//@}

//! \name Matrix non-square apply operations
//@{

struct non_square_apply_external_functor {
  template <typename T>
  KOKKOS_FUNCTION T operator()(const T& x) const {
    return x + T(1);
  }
};

struct non_square_apply_vector_functor {
  template <typename V>
  KOKKOS_FUNCTION auto operator()(const V& x) const {
    return sum(x) * Vector<typename V::scalar_t, V::size>{1};
  }
};

TYPED_TEST(NonSquareMatrixShapeTest, Apply) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  auto a = make_matrix_linear<R, C>(T(1), T(1));

  auto applied = apply(non_square_apply_external_functor{}, a);
  auto row_applied = apply_row(non_square_apply_vector_functor{}, a);
  auto col_applied = apply_column(non_square_apply_vector_functor{}, a);

  for (size_t i = 0; i < R * C; ++i) {
    is_close_debug(applied(i), static_cast<T>(static_cast<T>(a(i)) + T(1)), "Apply to elements failed.");
  }

  for (size_t i = 0; i < R; ++i) {
    T row_sum = T(0);
    for (size_t j = 0; j < C; ++j) {
      row_sum = static_cast<T>(row_sum + static_cast<T>(a(i, j)));
    }
    for (size_t j = 0; j < C; ++j) {
      is_close_debug(row_applied(i, j), row_sum, "Apply to rows failed.");
    }
  }

  for (size_t j = 0; j < C; ++j) {
    T col_sum = T(0);
    for (size_t i = 0; i < R; ++i) {
      col_sum = static_cast<T>(col_sum + static_cast<T>(a(i, j)));
    }
    for (size_t i = 0; i < R; ++i) {
      is_close_debug(col_applied(i, j), col_sum, "Apply to columns failed.");
    }
  }
}

//@}

//! \name Matrix non-square static factories and formatting
//@{

TYPED_TEST(NonSquareMatrixShapeTest, StaticFactoriesAndFormatting) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  auto zeros = Matrix<T, R, C>::zeros();
  auto ones = Matrix<T, R, C>::ones();
  auto ident = Matrix<T, R, C>::identity();

  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      EXPECT_TRUE(is_approx_close(zeros(i, j), T(0)));
      EXPECT_TRUE(is_approx_close(ones(i, j), T(1)));
      EXPECT_TRUE(is_approx_close(ident(i, j), i == j ? T(1) : T(0)));
    }
  }

  std::ostringstream os;
  os << ones;
  EXPECT_FALSE(os.str().empty());
  EXPECT_EQ(os.str().front(), '[');
  EXPECT_EQ(os.str().back(), ']');
}

//@}

//! \name Matrix non-square atomic operations
//@{

TYPED_TEST(NonSquareMatrixShapeTest, AtomicOpTestLoadStore) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  if (check_matrix_atomic_op_load_store_false_positive<T, R, C>()) {
    GTEST_SKIP() << "Skipping atomic load/store test due to false positive in non-atomic test.\n"
                 << "This typically occurs with very low optimization levels.";
  }

  Matrix<T, R, C> finished(T(0));
  auto func_atomic = [&finished]() {
    bool hit_max_loops = false;
    size_t max_loops = 1'000'000'000;
    size_t i = 0;
    while (!(atomic_load(&finished)[0] == T(1))) {
      if (i > max_loops) {
        hit_max_loops = true;
        break;
      }
      ++i;
    }
    return hit_max_loops;
  };

  auto result = std::async(std::launch::async, func_atomic);
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  atomic_store(&finished, T(1));
  EXPECT_FALSE(result.get()) << "Atomic load/store test failed.";
}

TYPED_TEST(NonSquareMatrixShapeTest, AtomicOpTestAddSub) {
  using T = typename TypeParam::scalar_t;
  constexpr size_t R = TypeParam::rows;
  constexpr size_t C = TypeParam::cols;

  if (check_matrix_atomic_op_add_sub_false_positive<T, R, C>()) {
    GTEST_SKIP() << "Skipping atomic add/sub test due to false positive in non-atomic test.\n"
                 << "This typically occurs if too many tests share too few CPU cores.";
  }

  int num_threads = 8;
  int num_iterations = 2001;

  Matrix<T, R, C> m_scalar_add(T(0));
  Matrix<T, R, C> m_scalar_sub(T(0));
  Matrix<T, R, C> m_matrix_add(T(0));
  Matrix<T, R, C> m_matrix_sub(T(0));

  auto thread_func = [&]() {
    const auto ones = Matrix<T, R, C>::ones();
    for (int i = 0; i < num_iterations; ++i) {
      atomic_add(&m_scalar_add, T(1));
      atomic_sub(&m_scalar_sub, T(1));
      atomic_add(&m_matrix_add, ones);
      atomic_sub(&m_matrix_sub, ones);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(thread_func);
  }
  for (auto& thread : threads) {
    thread.join();
  }

  const T expected_add = static_cast<T>(num_threads * num_iterations);
  const T expected_sub = static_cast<T>(-num_threads * num_iterations);
  for (size_t i = 0; i < R * C; ++i) {
    is_close_debug(m_scalar_add(i), expected_add, "Atomic scalar add failed.");
    is_close_debug(m_scalar_sub(i), expected_sub, "Atomic scalar sub failed.");
    is_close_debug(m_matrix_add(i), expected_add, "Atomic matrix add failed.");
    is_close_debug(m_matrix_sub(i), expected_sub, "Atomic matrix sub failed.");
  }
}

//@}

//! \name Matrix non-square pairwise mixed-type operations
//@{

TYPED_TEST(MatrixPairwiseTypeTest, NonSquareAdditionAndSubtractionWithMatrix) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix<T1, 2, 3> m1(1, 2, 3, 4, 5, 6);
  Matrix<T2, 2, 3> m2(2, 4, 6, 8, 10, 12);
  auto m3 = m1 + m2;
  expect_matrix_close(m3, {3, 6, 9, 12, 15, 18});

  m1 += m2;
  expect_matrix_close(m1, {3, 6, 9, 12, 15, 18});

  m3 = m1 - m2;
  expect_matrix_close(m3, {1, 2, 3, 4, 5, 6});

  m1 -= m2;
  expect_matrix_close(m1, {1, 2, 3, 4, 5, 6});

  auto left = T2(1) + m1;
  expect_matrix_close(left, {2, 3, 4, 5, 6, 7});

  auto right = m1 - T2(1);
  expect_matrix_close(right, {0, 1, 2, 3, 4, 5});
}

TYPED_TEST(MatrixPairwiseTypeTest, NonSquareMatrixVectorAndMatrixMatrixMixedTypes) {
  using T1 = typename TypeParam::T1;
  using T2 = typename TypeParam::T2;

  Matrix<T1, 2, 3> m(1, 2, 3, 4, 5, 6);
  Vector<T2, 3> v(1, 0, -1);
  Vector<T2, 2> w(2, -1);

  auto mv = m * v;
  EXPECT_TRUE(is_approx_close(mv[0], T2(-2)));
  EXPECT_TRUE(is_approx_close(mv[1], T2(-2)));

  auto vm = w * m;
  EXPECT_TRUE(is_approx_close(vm[0], T2(-2)));
  EXPECT_TRUE(is_approx_close(vm[1], T2(-1)));
  EXPECT_TRUE(is_approx_close(vm[2], T2(0)));

  auto mm = m * transpose(m);
  EXPECT_TRUE(is_approx_close(mm(0, 0), 14.0));
  EXPECT_TRUE(is_approx_close(mm(0, 1), 32.0));
  EXPECT_TRUE(is_approx_close(mm(1, 0), 32.0));
  EXPECT_TRUE(is_approx_close(mm(1, 1), 77.0));
}

//@}

}  // namespace

}  // namespace math

}  // namespace mundy
