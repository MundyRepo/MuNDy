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

// External
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*

// C++ core
#include <sstream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/Matrix.hpp>           // for mundy::Matrix3d
#include <mundy_math/Vector.hpp>           // for mundy::Vector3d
#include <mundy_math/preconditioners.hpp>  // for mundy::Preconditioner
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix

namespace mundy {

namespace {

// A mock "vector" type that does not satisfy VectorBackend against any real backend (no size/axpby/dot/deep_copy
// defined for it anywhere) -- used to prove the concept actually rejects bad input, not just accepts good input.
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

//! \name VectorBackend concept
//@{
static_assert(VectorBackend<MundyMathBackend, Vector3d>, "Vector3d must satisfy VectorBackend under MundyMathBackend");
static_assert(!VectorBackend<MundyMathBackend, NotAVector>,
              "NotAVector must NOT satisfy VectorBackend under MundyMathBackend");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
static_assert(VectorBackend<KokkosBackend<Kokkos::DefaultExecutionSpace>,
                            Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "Kokkos::View<double*> must satisfy VectorBackend under KokkosBackend");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name LinearOperator concept
//@{
static_assert(LinearOperator<MundyMathBackend, Matrix3d, Vector3d, Vector3d>,
              "Matrix3d must satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<MundyMathBackend, NotAnOperator, Vector3d, Vector3d>,
              "NotAnOperator must NOT satisfy LinearOperator under MundyMathBackend");
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
using kokkos_backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using kokkos_vector_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;
using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
using sparse_matrix_t =
    KokkosSparse::CrsMatrix<double, int,
                            Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>,
                            void, size_t>;
static_assert(LinearOperator<kokkos_backend_t, dense_matrix_t, kokkos_vector_t, kokkos_vector_t>,
              "A dense matrix must satisfy LinearOperator under KokkosBackend");
static_assert(LinearOperator<kokkos_backend_t, sparse_matrix_t, kokkos_vector_t, kokkos_vector_t>,
              "A sparse matrix must satisfy LinearOperator under KokkosBackend");
static_assert(!LinearOperator<MundyMathBackend, dense_matrix_t, Vector3d, Vector3d>,
              "A dense matrix must NOT satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<MundyMathBackend, sparse_matrix_t, Vector3d, Vector3d>,
              "A sparse matrix must NOT satisfy LinearOperator under MundyMathBackend");
static_assert(!LinearOperator<kokkos_backend_t, Matrix3d, kokkos_vector_t, kokkos_vector_t>,
              "Matrix3d must NOT satisfy LinearOperator under KokkosBackend");
static_assert(Preconditioner<dense_matrix_t, kokkos_backend_t, kokkos_vector_t>,
              "A dense matrix must satisfy Preconditioner under KokkosBackend");
static_assert(Preconditioner<sparse_matrix_t, kokkos_backend_t, kokkos_vector_t>,
              "A sparse matrix must satisfy Preconditioner under KokkosBackend");
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

//! \name HasScaledApplyMember concept
//@{
static_assert(HasScaledApplyMember<MockScaledOp, double, Vector3d, Vector3d>,
              "MockScaledOp must satisfy HasScaledApplyMember");
static_assert(!HasScaledApplyMember<MockUnscaledOp, double, Vector3d, Vector3d>,
              "MockUnscaledOp must NOT satisfy HasScaledApplyMember");
//@}

}  // namespace

TEST(SolverBackends, MundyMathBackendDotMatchesFreeDot) {
  const Vector3d a{1.0, 2.0, 3.0};
  const Vector3d b{4.0, -5.0, 6.0};
  const double expected = dot(a, b);
  const double actual = MundyMathBackend::dot<double>(a, b);
  EXPECT_DOUBLE_EQ(actual, expected);
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
TEST(SolverBackends, KokkosBackendDotMatchesHandComputed) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space = exec_space::memory_space;
  using view_t = Kokkos::View<double*, mem_space>;

  view_t x(Kokkos::view_alloc(Kokkos::WithoutInitializing, "x"), 3);
  view_t y(Kokkos::view_alloc(Kokkos::WithoutInitializing, "y"), 3);
  auto x_host = Kokkos::create_mirror_view(x);
  auto y_host = Kokkos::create_mirror_view(y);
  x_host(0) = 1.0;
  x_host(1) = 2.0;
  x_host(2) = 3.0;
  y_host(0) = 4.0;
  y_host(1) = -5.0;
  y_host(2) = 6.0;
  Kokkos::deep_copy(x, x_host);
  Kokkos::deep_copy(y, y_host);

  using backend_t = KokkosBackend<exec_space>;
  const double actual = backend_t::dot<double>(x, y);
  EXPECT_DOUBLE_EQ(actual, 1.0 * 4.0 + 2.0 * -5.0 + 3.0 * 6.0);
}

namespace {

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

/// \brief A vector with the given entries.
kokkos_vector_t make_vector(std::initializer_list<double> entries) {
  kokkos_vector_t v("v", entries.size());
  const auto v_host = Kokkos::create_mirror_view(v);
  size_t i = 0;
  for (const double entry : entries) {
    v_host(i++) = entry;
  }
  Kokkos::deep_copy(v, v_host);
  return v;
}

}  // namespace

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
  const kokkos_vector_t x = make_vector({1.0, -2.0, 3.0});
  const double ax[4] = {-1.0, 9.0, -6.0, 12.0};
  const auto check = [&](const auto& A, const char* kind, size_t allocations_per_apply) {
    EXPECT_EQ(kokkos_backend_t::domain_size(A), 3u) << kind;
    EXPECT_EQ(kokkos_backend_t::range_size(A), 4u) << kind;
    EXPECT_EQ(kokkos_backend_t::make_domain_vector(A).extent(0), 3u) << kind;
    EXPECT_EQ(kokkos_backend_t::make_range_vector(A).extent(0), 4u) << kind;

    // y = A x, with and without a workspace
    auto workspace = kokkos_backend_t::make_workspace(A);
    kokkos_vector_t y = make_vector({7.0, 7.0, 7.0, 7.0});
    const size_t y_allocations = count_allocations([&] { kokkos_backend_t::apply(A, x, y); });
    kokkos_vector_t y_workspace = make_vector({7.0, 7.0, 7.0, 7.0});
    const size_t y_workspace_allocations =
        count_allocations([&] { kokkos_backend_t::apply(A, x, y_workspace, workspace); });

    // y = 2 A x - 3 y0 for y0 = (1, 2, 3, 4), with and without a workspace
    kokkos_vector_t z = make_vector({1.0, 2.0, 3.0, 4.0});
    const size_t z_allocations = count_allocations([&] { kokkos_backend_t::apply(2.0, A, x, -3.0, z); });
    kokkos_vector_t z_workspace = make_vector({1.0, 2.0, 3.0, 4.0});
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

}  // namespace mundy
