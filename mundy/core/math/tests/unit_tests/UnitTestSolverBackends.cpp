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

/// \file UnitTestSolverBackends.cpp
/// \brief Backend contracts, operator application, and vector expressions (solver_backends.hpp) on MundyMathBackend
/// and KokkosBackend.
///
/// Reference vectors:
///   - x = (1, -2, 4, 1/2) and y = (2, 4, -8, 1/4). Their entries are small dyadic numbers, so every expression,
///     assignment, and reduction of them below is exact and equals the same arithmetic done on the host.

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

// Mundy
#include <MundyMath_config.hpp>            // for HAVE_MUNDYMATH_KOKKOSKERNELS
#include <mundy_math/Matrix.hpp>           // for mundy::Matrix3d
#include <mundy_math/Vector.hpp>           // for mundy::Vector3d
#include <mundy_math/preconditioners.hpp>  // for mundy::Preconditioner
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix

namespace mundy {

namespace {

using mm_backend_t = MundyMathBackend;
using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using kokkos_vector_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
using sparse_matrix_t =
    KokkosSparse::CrsMatrix<double, int,
                            Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>,
                            void, size_t>;
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

// A mock "vector" type that does not satisfy VectorBackend against any real backend (it has no entries to read) -- used
// to prove the concept actually rejects bad input, not just accepts good input.
struct NotAVector {};

// A mock operator with no apply/domain_size/range_size member and no operator*, so it does not satisfy
// LinearOperator against any real backend.
struct NotAnOperator {};

// A mock operator that provides the fused scaled-apply member (apply(alpha, x, beta, y)).
struct MockScaledOp {
  void apply(double alpha, const Vector3d& x, double beta, Vector3d& y) const {
    y = alpha * x + beta * y;
  }
};

// A mock operator that only provides the plain apply(x, y) -- no scaled-apply member.
struct MockUnscaledOp {
  void apply(const Vector3d& x, Vector3d& y) const {
    y = x;
  }
};

//! \name Compile-time contracts
//@{

static_assert(VectorBackend<mm_backend_t, Vector3d>, "Vector3d must satisfy VectorBackend under MundyMathBackend");
static_assert(!VectorBackend<mm_backend_t, NotAVector>,
              "NotAVector must NOT satisfy VectorBackend under MundyMathBackend");
static_assert(VectorBackend<kokkos_backend_t, kokkos_vector_t>,
              "Kokkos::View<double*> must satisfy VectorBackend under KokkosBackend");

static_assert(LinearOperator<mm_backend_t, Matrix3d, Vector3d, Vector3d>,
              "Matrix3d must satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<mm_backend_t, NotAnOperator, Vector3d, Vector3d>,
              "NotAnOperator must NOT satisfy LinearOperator under MundyMathBackend");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
static_assert(LinearOperator<kokkos_backend_t, dense_matrix_t, kokkos_vector_t, kokkos_vector_t>,
              "A dense matrix must satisfy LinearOperator under KokkosBackend");
static_assert(LinearOperator<kokkos_backend_t, sparse_matrix_t, kokkos_vector_t, kokkos_vector_t>,
              "A sparse matrix must satisfy LinearOperator under KokkosBackend");
static_assert(!LinearOperator<mm_backend_t, dense_matrix_t, Vector3d, Vector3d>,
              "A dense matrix must NOT satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<mm_backend_t, sparse_matrix_t, Vector3d, Vector3d>,
              "A sparse matrix must NOT satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<kokkos_backend_t, Matrix3d, kokkos_vector_t, kokkos_vector_t>,
              "Matrix3d must NOT satisfy LinearOperator under KokkosBackend");
static_assert(Preconditioner<dense_matrix_t, kokkos_backend_t, kokkos_vector_t>,
              "A dense matrix must satisfy Preconditioner under KokkosBackend");
static_assert(Preconditioner<sparse_matrix_t, kokkos_backend_t, kokkos_vector_t>,
              "A sparse matrix must satisfy Preconditioner under KokkosBackend");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

static_assert(HasScaledApplyMember<MockScaledOp, double, Vector3d, Vector3d>,
              "MockScaledOp must satisfy HasScaledApplyMember");
static_assert(!HasScaledApplyMember<MockUnscaledOp, double, Vector3d, Vector3d>,
              "MockUnscaledOp must NOT satisfy HasScaledApplyMember");

static_assert(BackendVectorExpr<decltype(mm_backend_t::entries(std::declval<const Vector3d&>()))>,
              "A MundyMath vector's entries are a vector expression.");
static_assert(BackendVectorExpr<decltype(kokkos_backend_t::entries(std::declval<const kokkos_vector_t&>()))>,
              "A Kokkos view's entries are a vector expression.");
static_assert(BackendVectorExpr<decltype(2.0 * mm_backend_t::entries(std::declval<const Vector3d&>()) - 1.0)>,
              "Expressions combined with scalars are vector expressions.");
static_assert(!BackendVectorExpr<Vector3d> && !BackendVectorExpr<kokkos_vector_t>,
              "A vector is not itself an expression; its entries are.");
static_assert(std::is_same_v<decltype(2.0 * std::declval<Vector3d>()), Vector3d>,
              "MundyMath vectors keep their own, eager arithmetic.");
//@}

//! \name Reference vectors
//@{

constexpr std::array<double, 4> x_ref{1.0, -2.0, 4.0, 0.5};
constexpr std::array<double, 4> y_ref{2.0, 4.0, -8.0, 0.25};

/// \brief value at every entry of a vector the size of x_ref.
constexpr std::array<double, x_ref.size()> constant_entries(double value) {
  std::array<double, x_ref.size()> entries{};
  entries.fill(value);
  return entries;
}

/// \brief f(x_i, y_i) for each entry of the reference vectors, on the host.
template <class F>
std::vector<double> entrywise(const F& f) {
  std::vector<double> entries(x_ref.size());
  for (size_t i = 0; i < x_ref.size(); ++i) {
    entries[i] = f(x_ref[i], y_ref[i]);
  }
  return entries;
}

/// \brief The given entries as a Backend vector: a MundyMath vector, or a Kokkos::View on the default space.
template <class Backend, size_t N>
auto make_backend_vector(const std::array<double, N>& entries) {
  if constexpr (Backend::has_static_sizes) {
    Vector<double, N> v;
    for (size_t i = 0; i < N; ++i) {
      v[i] = entries[i];
    }
    return v;
  } else {
    kokkos_vector_t v("v", N);
    const auto v_host = Kokkos::create_mirror_view(v);
    for (size_t i = 0; i < N; ++i) {
      v_host(i) = entries[i];
    }
    Kokkos::deep_copy(v, v_host);
    return v;
  }
}

/// \brief Every entry of a MundyMath vector or a Kokkos::View, on the host.
template <class Vector>
std::vector<double> host_entries(const Vector& v) {
  if constexpr (Kokkos::is_view_v<Vector>) {
    const auto v_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    return std::vector<double>(v_host.data(), v_host.data() + v_host.extent(0));
  } else {
    std::vector<double> entries(Vector::size);
    for (size_t i = 0; i < Vector::size; ++i) {
      entries[i] = v[i];
    }
    return entries;
  }
}

double host_sum(const std::vector<double>& entries) {
  return std::accumulate(entries.begin(), entries.end(), 0.0);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
/// \brief A rows x cols dense matrix with the given row-major entries.
dense_matrix_t make_dense_matrix(size_t rows, size_t cols, std::initializer_list<double> row_major) {
  dense_matrix_t A("A", rows, cols);
  const auto A_host = Kokkos::create_mirror_view(A);
  auto entry = row_major.begin();
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      A_host(i, j) = *entry++;
    }
  }
  Kokkos::deep_copy(A, A_host);
  return A;
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Running and checking backend operations
//@{

/// \brief Each operator, with scalars on either side, and a scalar and a vector as sources, assigned on Backend.
template <class Backend>
void expect_every_operator_assigns_exactly() {
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto y = make_backend_vector<Backend>(y_ref);
  auto z = make_backend_vector<Backend>(constant_entries(0.0));
  const auto X = Backend::entries(x);
  const auto Y = Backend::entries(y);

  Backend::assign("z", z, 3.0 * X + Y / 2.0 - 1.0);
  EXPECT_EQ(host_entries(z), entrywise([](double xi, double yi) { return 3.0 * xi + yi / 2.0 - 1.0; }))
      << "z = 3 x + y / 2 - 1";

  Backend::assign("z", z, 2.0 - X * Y + 1.0 / Y);
  EXPECT_EQ(host_entries(z), entrywise([](double xi, double yi) { return 2.0 - xi * yi + 1.0 / yi; }))
      << "z = 2 - x y + 1 / y";

  Backend::assign("z", z, -abs(X - 2.0) / Y);
  EXPECT_EQ(host_entries(z), entrywise([](double xi, double yi) { return -std::abs(xi - 2.0) / yi; }))
      << "z = -|x - 2| / y";

  Backend::assign("z", z, 7.0);
  EXPECT_EQ(host_entries(z), entrywise([](double, double) { return 7.0; })) << "z = 7";

  Backend::assign("z", z, x);
  EXPECT_EQ(host_entries(z), entrywise([](double xi, double) { return xi; })) << "z = x";
}

/// \brief Two targets, each read by both sources, assigned in one pass on Backend.
template <class Backend>
void expect_assignment_reads_every_source_first() {
  auto x = make_backend_vector<Backend>(x_ref);
  auto y = make_backend_vector<Backend>(y_ref);
  const auto X = Backend::entries(x);
  const auto Y = Backend::entries(y);

  Backend::assign("x and y", x, X + Y, y, X - Y);
  EXPECT_EQ(host_entries(x), entrywise([](double xi, double yi) { return xi + yi; })) << "x = x + y";
  EXPECT_EQ(host_entries(y), entrywise([](double xi, double yi) { return xi - yi; })) << "y = x - y, from the old x";
}

/// \brief Three reducers over three sources in one pass, and one reducer over a vector, on Backend.
template <class Backend>
void expect_reduction_joins_every_reducer() {
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto y = make_backend_vector<Backend>(y_ref);
  const auto X = Backend::entries(x);
  const auto Y = Backend::entries(y);

  double x_dot_y = 0.0;
  double largest = 0.0;
  double least = 0.0;
  Backend::reduce("three reducers", Kokkos::Sum<double>(x_dot_y), X * Y, Kokkos::Max<double>(largest), abs(X),
                  Kokkos::Min<double>(least), X - Y);
  EXPECT_EQ(x_dot_y, host_sum(entrywise([](double xi, double yi) { return xi * yi; }))) << "x^T y";
  EXPECT_EQ(largest, std::ranges::max(entrywise([](double xi, double) { return std::abs(xi); }))) << "max |x|";
  EXPECT_EQ(least, std::ranges::min(entrywise([](double xi, double yi) { return xi - yi; }))) << "min (x - y)";

  double sum = 0.0;
  Backend::reduce("one reducer", Kokkos::Sum<double>(sum), x);
  EXPECT_EQ(sum, host_sum(entrywise([](double xi, double) { return xi; }))) << "sum x";
}

/// \brief dot, reduce_sum/min/max, elementwise_mul/div, and axpby on Backend.
template <class Backend>
void expect_vector_operations_match_the_host() {
  const Backend backend;
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto y = make_backend_vector<Backend>(y_ref);
  auto z = make_backend_vector<Backend>(constant_entries(0.0));

  EXPECT_EQ(dot(backend, x, y), host_sum(entrywise([](double xi, double yi) { return xi * yi; }))) << "dot";
  EXPECT_EQ(reduce_sum(backend, x), host_sum(entrywise([](double xi, double) { return xi; }))) << "reduce_sum";
  EXPECT_EQ(reduce_min(backend, x), std::ranges::min(x_ref)) << "reduce_min";
  EXPECT_EQ(reduce_max(backend, abs(Backend::entries(y))),
            std::ranges::max(entrywise([](double, double yi) { return std::abs(yi); })))
      << "reduce_max of |y|";

  elementwise_mul(backend, x, y, z);
  EXPECT_EQ(host_entries(z), entrywise([](double xi, double yi) { return xi * yi; })) << "elementwise_mul";
  elementwise_div(backend, x, y, z);
  EXPECT_EQ(host_entries(z), entrywise([](double xi, double yi) { return xi / yi; })) << "elementwise_div";

  auto w = make_backend_vector<Backend>(y_ref);
  axpby(backend, 2.0, x, 3.0, w);
  EXPECT_EQ(host_entries(w), entrywise([](double xi, double yi) { return 2.0 * xi + 3.0 * yi; })) << "axpby";
}

/// \brief axpby on Backend with coefficients that are exactly zero, whose vectors hold NaN, and with a coefficient of
/// 2^-60 against entries of 2^60, which contributes exactly 1.
template <class Backend>
void expect_axpby_reads_exactly_the_nonzero_terms() {
  const Backend backend;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto x = make_backend_vector<Backend>(x_ref);
  const auto x_nan = make_backend_vector<Backend>(constant_entries(nan));

  auto y_nan = make_backend_vector<Backend>(constant_entries(nan));
  axpby(backend, 2.0, x, 0.0, y_nan);
  EXPECT_EQ(host_entries(y_nan), entrywise([](double xi, double) { return 2.0 * xi; })) << "y = 2 x, y unread";

  auto y = make_backend_vector<Backend>(y_ref);
  axpby(backend, 0.0, x_nan, 3.0, y);
  EXPECT_EQ(host_entries(y), entrywise([](double, double yi) { return 3.0 * yi; })) << "y = 3 y, x unread";

  auto zeroed = make_backend_vector<Backend>(constant_entries(nan));
  axpby(backend, 0.0, x_nan, 0.0, zeroed);
  EXPECT_EQ(host_entries(zeroed), entrywise([](double, double) { return 0.0; })) << "y = 0, x and y unread";

  auto big = make_backend_vector<Backend>(constant_entries(std::ldexp(1.0, 60)));
  axpby(backend, 1.0, x, std::ldexp(1.0, -60), big);
  EXPECT_EQ(host_entries(big), entrywise([](double xi, double) { return xi + 1.0; })) << "y = x + 2^-60 2^60";
}

// Keep KOKKOS_LAMBDA outside GTest bodies for CUDA. The kernel holds x and y as MundyMath vectors and writes back what
// MundyMathBackend computes from them: z = 3 x + y / 2 - 1, axpby's 2 x + 3 y, and the reductions x^T y by reduce,
// max |x|, x^T y by dot, and min x.
void evaluate_mundy_math_in_kernel(const kokkos_vector_t& x_view, const kokkos_vector_t& y_view,
                                   const kokkos_vector_t& assigned, const kokkos_vector_t& combined,
                                   const kokkos_vector_t& reduced) {
  constexpr size_t n = x_ref.size();
  Kokkos::parallel_for(
      "evaluate_mundy_math_in_kernel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(const size_t) {
        const mm_backend_t backend;
        Vector<double, n> x;
        Vector<double, n> y;
        Vector<double, n> z;
        for (size_t i = 0; i < n; ++i) {
          x[i] = x_view(i);
          y[i] = y_view(i);
        }
        const auto X = mm_backend_t::entries(x);
        const auto Y = mm_backend_t::entries(y);

        mm_backend_t::assign("z", z, 3.0 * X + Y / 2.0 - 1.0);
        double x_dot_y = 0.0;
        double largest = 0.0;
        mm_backend_t::reduce("x and y", Kokkos::Sum<double>(x_dot_y), X * Y, Kokkos::Max<double>(largest), abs(X));
        reduced(0) = x_dot_y;
        reduced(1) = largest;
        reduced(2) = dot(backend, x, y);
        reduced(3) = reduce_min(backend, x);
        axpby(backend, 2.0, x, 3.0, y);
        for (size_t i = 0; i < n; ++i) {
          assigned(i) = z[i];
          combined(i) = y[i];
        }
      });
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
/// \brief The Kokkos allocations one sparse apply makes on ExecSpace.
///
/// On CUDA its team launch takes a team-scratch slot, an allocation even when the kernel asks for no scratch; host
/// launches make none.
template <class ExecSpace>
constexpr size_t allocations_per_sparse_apply() {
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
    return 1;
  }
#endif
  return 0;
}

/// \brief The number of Kokkos allocations made while f runs.
template <class F>
size_t count_allocations(F&& f) {
  static size_t count = 0;
  count = 0;
  Kokkos::Tools::Experimental::set_init_callback(
      [](const int, const uint64_t, const uint32_t, Kokkos_Profiling_KokkosPDeviceInfo*) {});
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](const Kokkos_Profiling_SpaceHandle, const char*, const void*, const uint64_t) { ++count; });
  f();
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_init_callback(nullptr);
  return count;
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Operator application
//@{

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
// A dense and a sparse matrix of the same integer entries; every product is exact, so each apply is checked exactly. A
// dense apply allocates nothing and a sparse apply allocates allocations_per_sparse_apply.
TEST(SolverBackends, KokkosBackendAppliesDenseAndSparseMatrices) {
  // A, 4 x 3, with zeros the sparse matrix does not store
  const dense_matrix_t dense = make_dense_matrix(4, 3,
                                                 {2.0, 0.0, -1.0,  //
                                                  0.0, 0.0, 3.0,   //
                                                  4.0, 5.0, 0.0,   //
                                                  0.0, -6.0, 0.0});
  const sparse_matrix_t sparse = make_sparse_matrix<sparse_matrix_t>(dense);
  EXPECT_EQ(sparse.nnz(), 6u);

  // A x = (2 - 3, 9, 4 - 10, 12) for x = (1, -2, 3)
  const kokkos_vector_t x = make_backend_vector<kokkos_backend_t>(std::array{1.0, -2.0, 3.0});
  const double ax[4] = {-1.0, 9.0, -6.0, 12.0};
  const auto check = [&](const auto& A, const char* kind, size_t allocations_per_apply) {
    EXPECT_EQ(kokkos_backend_t::domain_size(A), 3u) << kind;
    EXPECT_EQ(kokkos_backend_t::range_size(A), 4u) << kind;
    EXPECT_EQ(kokkos_backend_t::make_domain_vector(A).extent(0), 3u) << kind;
    EXPECT_EQ(kokkos_backend_t::make_range_vector(A).extent(0), 4u) << kind;

    // y = A x, with and without a workspace
    auto workspace = kokkos_backend_t::make_workspace(A);
    kokkos_vector_t y = make_backend_vector<kokkos_backend_t>(std::array{7.0, 7.0, 7.0, 7.0});
    const size_t y_allocations = count_allocations([&] { kokkos_backend_t::apply(A, x, y); });
    kokkos_vector_t y_workspace = make_backend_vector<kokkos_backend_t>(std::array{7.0, 7.0, 7.0, 7.0});
    const size_t y_workspace_allocations =
        count_allocations([&] { kokkos_backend_t::apply(A, x, y_workspace, workspace); });

    // y = 2 A x - 3 y0 for y0 = (1, 2, 3, 4), with and without a workspace
    kokkos_vector_t z = make_backend_vector<kokkos_backend_t>(std::array{1.0, 2.0, 3.0, 4.0});
    const size_t z_allocations = count_allocations([&] { kokkos_backend_t::apply(2.0, A, x, -3.0, z); });
    kokkos_vector_t z_workspace = make_backend_vector<kokkos_backend_t>(std::array{1.0, 2.0, 3.0, 4.0});
    const size_t z_workspace_allocations =
        count_allocations([&] { kokkos_backend_t::apply(2.0, A, x, -3.0, z_workspace, workspace); });
    EXPECT_EQ(y_allocations, allocations_per_apply) << kind;
    EXPECT_EQ(y_workspace_allocations, allocations_per_apply) << kind;
    EXPECT_EQ(z_allocations, allocations_per_apply) << kind;
    EXPECT_EQ(z_workspace_allocations, allocations_per_apply) << kind;

    const auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y);
    const auto y_workspace_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, y_workspace);
    const auto z_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, z);
    const auto z_workspace_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, z_workspace);
    for (size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(y_host(i), ax[i]) << kind << " entry " << i;
      EXPECT_EQ(y_workspace_host(i), ax[i]) << kind << " entry " << i;
      EXPECT_EQ(z_host(i), 2.0 * ax[i] - 3.0 * (i + 1.0)) << kind << " entry " << i;
      EXPECT_EQ(z_workspace_host(i), 2.0 * ax[i] - 3.0 * (i + 1.0)) << kind << " entry " << i;
    }
  };
  check(dense, "dense", 0);
  check(sparse, "sparse", allocations_per_sparse_apply<Kokkos::DefaultExecutionSpace>());
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name Vector expressions
//@{

TEST(SolverBackends, AssignmentEvaluatesEveryOperatorExactly) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_every_operator_assigns_exactly<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_every_operator_assigns_exactly<kokkos_backend_t>();
  }
}

TEST(SolverBackends, AssignmentReadsEverySourceBeforeWritingAnyTarget) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_assignment_reads_every_source_first<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_assignment_reads_every_source_first<kokkos_backend_t>();
  }
}

TEST(SolverBackends, ReductionJoinsEveryReducerInOnePass) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_reduction_joins_every_reducer<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_reduction_joins_every_reducer<kokkos_backend_t>();
  }
}

TEST(SolverBackends, ReductionOfAnEmptyVectorLeavesEachReducerIdentity) {
  const kokkos_vector_t empty("empty", 0);
  double sum = 1.0;
  double largest = 1.0;
  double least = 1.0;
  kokkos_backend_t::reduce("empty", Kokkos::Sum<double>(sum), empty, Kokkos::Max<double>(largest), empty,
                           Kokkos::Min<double>(least), empty);
  EXPECT_EQ(sum, 0.0);
  EXPECT_EQ(largest, std::numeric_limits<double>::lowest());
  EXPECT_EQ(least, std::numeric_limits<double>::max());
}
//@}

//! \name Vector operations
//@{

TEST(SolverBackends, VectorOperationsMatchTheHost) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_vector_operations_match_the_host<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_vector_operations_match_the_host<kokkos_backend_t>();
  }
}

TEST(SolverBackends, AxpbyReadsExactlyTheNonzeroTerms) {
  {
    SCOPED_TRACE("MundyMathBackend");
    expect_axpby_reads_exactly_the_nonzero_terms<mm_backend_t>();
  }
  {
    SCOPED_TRACE("KokkosBackend");
    expect_axpby_reads_exactly_the_nonzero_terms<kokkos_backend_t>();
  }
}
//@}

//! \name Device execution
//@{

TEST(SolverBackends, MundyMathExpressionsEvaluateInsideAKernel) {
  const auto x = make_backend_vector<kokkos_backend_t>(x_ref);
  const auto y = make_backend_vector<kokkos_backend_t>(y_ref);
  const kokkos_vector_t assigned("assigned", x_ref.size());
  const kokkos_vector_t combined("combined", x_ref.size());
  const kokkos_vector_t reduced("reduced", 4);
  evaluate_mundy_math_in_kernel(x, y, assigned, combined, reduced);

  const double x_dot_y = host_sum(entrywise([](double xi, double yi) { return xi * yi; }));
  const double largest = std::ranges::max(entrywise([](double xi, double) { return std::abs(xi); }));
  EXPECT_EQ(host_entries(assigned), entrywise([](double xi, double yi) { return 3.0 * xi + yi / 2.0 - 1.0; }));
  EXPECT_EQ(host_entries(combined), entrywise([](double xi, double yi) { return 2.0 * xi + 3.0 * yi; }));
  EXPECT_EQ(host_entries(reduced), (std::vector<double>{x_dot_y, largest, x_dot_y, std::ranges::min(x_ref)}));
}
//@}

}  // namespace

}  // namespace mundy
