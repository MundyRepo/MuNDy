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
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_KOKKOSKERNELS

// Mundy
#include <mundy_math/Matrix.hpp>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_math/linear_ops.hpp>
#include <mundy_math/solver_backends.hpp>
#include <mundy_math/sparse_matrix.hpp>  // for mundy::make_sparse_matrix

namespace mundy {

namespace {

using backend_t = KokkosBackend<Kokkos::DefaultExecutionSpace>;
using view_t = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>;

view_t make_view(std::initializer_list<double> values) {
  view_t v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "v"), values.size());
  auto v_host = Kokkos::create_mirror_view(v);
  size_t i = 0;
  for (double value : values) {
    v_host(i++) = value;
  }
  Kokkos::deep_copy(v, v_host);
  return v;
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

std::vector<double> to_host(const view_t& v) {
  auto v_host = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(v_host, v);
  std::vector<double> out(v.extent(0));
  for (size_t i = 0; i < v.extent(0); ++i) {
    out[i] = v_host(i);
  }
  return out;
}

// y := scale * x, elementwise -- a minimal View-backed LinearOperator with only a plain apply(x, y) member (no
// workspace overload, no scaled-apply member), used as a generic building block below.
struct ScaleOp {
  explicit ScaleOp(double scale) : scale_(scale) {
  }
  size_t domain_size() const {
    return n_;
  }
  size_t range_size() const {
    return n_;
  }
  void set_size(size_t n) {
    n_ = n;
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "scale_domain"), n_);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "scale_range"), n_);
  }
  void apply(const view_t& x, view_t& y) const {
    const double scale = scale_;
    Kokkos::parallel_for(
        "ScaleOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_),
        KOKKOS_LAMBDA(const int i) { y(i) = scale * x(i); });
  }

 private:
  double scale_;
  size_t n_{3};
};

// y := scale * x, but the ONLY apply overload takes a workspace (no plain apply(x, y) at all) -- used to prove
// that SumOp/ConcatDomainOp/ConcatRangeOp thread a real workspace through to their children via
// Backend::apply's workspace-aware dispatch, rather than calling a bare 2-arg apply() directly (which would
// simply fail to compile/dispatch for an operator shaped like this one).
struct WorkspaceOnlyScaleOp {
  explicit WorkspaceOnlyScaleOp(double scale, size_t n) : scale_(scale), n_(n) {
  }
  size_t domain_size() const {
    return n_;
  }
  size_t range_size() const {
    return n_;
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "wo_domain"), n_);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "wo_range"), n_);
  }
  template <class Workspace>
  void apply(const view_t& x, view_t& y, Workspace&) const {
    const double scale = scale_;
    Kokkos::parallel_for(
        "WorkspaceOnlyScaleOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_),
        KOKKOS_LAMBDA(const int i) { y(i) = scale * x(i); });
  }

 private:
  double scale_;
  size_t n_;
};

// Provides its own fused scaled-apply (HasScaledApplyMember), so ScaledOp can pick the fast path for it.
struct FusedScaleOp {
  size_t domain_size() const {
    return 3;
  }
  size_t range_size() const {
    return 3;
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "fused_domain"), 3);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "fused_range"), 3);
  }
  void apply(const view_t& x, view_t& y) const {
    apply(1.0, x, 0.0, y);
  }
  void apply(double alpha, const view_t& x, double beta, view_t& y) const {
    Kokkos::parallel_for(
        "FusedScaleOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 3),
        KOKKOS_LAMBDA(const int i) { y(i) = alpha * (2.0 * x(i)) + beta * y(i); });
  }
};

static_assert(LinearOperator<backend_t, ScaleOp, view_t, view_t>, "ScaleOp must satisfy LinearOperator");
static_assert(!HasScaledApplyMember<ScaleOp, double, view_t, view_t>, "ScaleOp must NOT satisfy HasScaledApplyMember");
static_assert(HasScaledApplyMember<FusedScaleOp, double, view_t, view_t>,
              "FusedScaleOp must satisfy HasScaledApplyMember");

TEST(LinearOperators, SumOpAddsBothOperatorsContributions) {
  ScaleOp op1(2.0);
  op1.set_size(3);
  ScaleOp op2(3.0);
  op2.set_size(3);
  auto sum = SumOp(backend_t{}, ScaleOp(op1), ScaleOp(op2));

  const view_t x = make_view({1.0, 2.0, 3.0});
  view_t y = sum.make_range_vector();
  sum.apply(x, y);

  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 5.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[1], 5.0 * 2.0);
  EXPECT_DOUBLE_EQ(result[2], 5.0 * 3.0);
}

TEST(LinearOperators, SumOpWorksWithAWorkspaceOnlyChild) {
  // op1 only exposes apply(x, y, workspace); before the workspace-threading fix, a composite that called a bare
  // apply(x, y) on its children could not compose with an operator shaped like this at all.
  WorkspaceOnlyScaleOp op1(2.0, 3);
  ScaleOp op2(3.0);
  op2.set_size(3);
  auto sum = SumOp(backend_t{}, WorkspaceOnlyScaleOp(op1), ScaleOp(op2));

  const view_t x = make_view({1.0, 1.0, 1.0});
  view_t y = sum.make_range_vector();
  sum.apply(x, y);

  const std::vector<double> result = to_host(y);
  for (double value : result) {
    EXPECT_DOUBLE_EQ(value, 5.0);
  }
}

TEST(LinearOperators, QuadraticFormOpWorksWithWorkspaceOnlyChildren) {
  // Every child only exposes apply(x, y, workspace), so the form can apply them only through its own workspace.
  auto form = make_quadratic_form<backend_t>(WorkspaceOnlyScaleOp(2.0, 3), WorkspaceOnlyScaleOp(3.0, 3),
                                             WorkspaceOnlyScaleOp(5.0, 3));

  const view_t x = make_view({1.0, 2.0, 3.0});
  view_t y = form.make_range_vector();
  form.apply(x, y);

  // D^T M D x with D^T = 2, M = 3, D = 5
  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 30.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[1], 30.0 * 2.0);
  EXPECT_DOUBLE_EQ(result[2], 30.0 * 3.0);
}

TEST(LinearOperators, ScaledOpUsesGenericFallbackForAnUnfusedOp) {
  ScaleOp op(2.0);
  op.set_size(3);
  auto scaled = ScaledOp(backend_t{}, /*alpha=*/4.0, ScaleOp(op));

  const view_t x = make_view({1.0, 2.0, 3.0});
  view_t y = scaled.make_range_vector();
  scaled.apply(x, y);

  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 4.0 * 2.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[1], 4.0 * 2.0 * 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0 * 2.0 * 3.0);
}

TEST(LinearOperators, ScaledOpUsesFusedFastPathWhenAvailable) {
  auto scaled = ScaledOp(backend_t{}, /*alpha=*/4.0, FusedScaleOp{});

  const view_t x = make_view({1.0, 2.0, 3.0});
  view_t y = scaled.make_range_vector();
  scaled.apply(x, y);

  // FusedScaleOp::apply(alpha, x, beta, y) computes alpha*(2*x) + beta*y; ScaledOp calls it with beta=0, so the
  // observable result is identical to the generic-fallback case above -- what differs is which code path
  // Backend::apply(alpha, op, x, beta, y) selects, not the answer.
  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 4.0 * 2.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[1], 4.0 * 2.0 * 2.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0 * 2.0 * 3.0);
}

// The child only exposes apply(x, y, workspace), so the scaled op can apply it only through its own workspace, which
// then serves every apply without allocating.
TEST(LinearOperators, ScaledOpWorksWithAWorkspaceOnlyChild) {
  const auto scaled = ScaledOp(backend_t{}, /*alpha=*/4.0, WorkspaceOnlyScaleOp(2.0, 3));
  const view_t x = make_view({1.0, 2.0, 3.0});
  view_t y = scaled.make_range_vector();
  auto workspace = scaled.make_workspace();

  // Apply twice through one workspace
  backend_t::apply(scaled, x, y, workspace);
  const size_t num_allocations = count_allocations([&] { backend_t::apply(scaled, x, y, workspace); });

  const std::vector<double> result = to_host(y);
  EXPECT_EQ(result[0], 4.0 * 2.0 * 1.0);
  EXPECT_EQ(result[1], 4.0 * 2.0 * 2.0);
  EXPECT_EQ(result[2], 4.0 * 2.0 * 3.0);
  EXPECT_EQ(num_allocations, 0u);
}

// A linear operator with independent domain/range sizes that scales its input into a chosen slice of the range
// (rows [out_offset, out_offset + domain)), zeroing the rest. Two of these with disjoint output slices make
// ConcatDomainOp's domain split observable: op1 fills the top rows from x1, op2 the bottom rows from x2.
struct SliceScaleOp {
  double scale;
  size_t domain;
  size_t range;
  size_t out_offset;
  size_t domain_size() const {
    return domain;
  }
  size_t range_size() const {
    return range;
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "slice_domain"), domain);
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "slice_range"), range);
  }
  void apply(const view_t& x, view_t& y) const {
    const double s = scale;
    const size_t d = domain;
    const size_t off = out_offset;
    Kokkos::deep_copy(y, 0.0);
    Kokkos::parallel_for(
        "SliceScaleOp::apply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, d),
        KOKKOS_LAMBDA(const int i) { y(off + i) = s * x(i); });
  }
};

TEST(LinearOperators, ShiftedOpWorksWithAWorkspaceOnlyChild) {
  auto shifted = ShiftedOp(backend_t{}, /*sigma=*/0.5, WorkspaceOnlyScaleOp(2.0, 3));

  const view_t x = make_view({1.0, 2.0, 3.0});
  view_t y = shifted.make_range_vector();
  shifted.apply(x, y);

  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], (2.0 - 0.5) * 1.0);
  EXPECT_DOUBLE_EQ(result[1], (2.0 - 0.5) * 2.0);
  EXPECT_DOUBLE_EQ(result[2], (2.0 - 0.5) * 3.0);
}

TEST(LinearOperators, ShiftedOpMatchesAMinusSigmaIdentityOnMundyMathBackend) {
  const Matrix3d A{2.0, -1.0, 0.5,   //
                   -1.0, 3.0, 0.25,  //
                   0.5, 0.25, -4.0};
  constexpr double sigma = 1.75;
  const auto shifted = make_shifted_op<MundyMathBackend>(sigma, A);

  // Column j of the shifted operator is its action on the j-th basis vector.
  for (size_t j = 0; j < 3; ++j) {
    Vector3d e{0.0, 0.0, 0.0};
    e[j] = 1.0;
    Vector3d column{0.0, 0.0, 0.0};
    MundyMathBackend::apply(shifted, e, column);
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(column[i], A(i, j) - (i == j ? sigma : 0.0)) << "entry (" << i << ", " << j << ")";
    }
  }
}

// MundyMathBackend Concat cases, each applied to x = (1, -2, 3) with integer entries so every sum is exact. A1 is 3x2
// and A2 is 3x1; their transposes are 2x3 and 1x3.
struct ConcatDomainCase {
  // [A1 | A2] = [[1, 2, 7], [3, 4, 8], [5, 6, 9]]
  KOKKOS_INLINE_FUNCTION Vector3d operator()() const {
    const Matrix<double, 3, 2> A1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const Matrix<double, 3, 1> A2{7.0, 8.0, 9.0};
    const auto op = make_concat_domain_op<MundyMathBackend>(A1, A2);
    static_assert(std::remove_cvref_t<decltype(op)>::static_domain_size() == 3);
    static_assert(std::remove_cvref_t<decltype(op)>::static_range_size() == 3);
    Vector3d y;
    MundyMathBackend::apply(op, Vector3d{1.0, -2.0, 3.0}, y);
    return y;
  }
};

struct ConcatRangeCase {
  // [A1T; A2T] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  KOKKOS_INLINE_FUNCTION Vector3d operator()() const {
    const Matrix<double, 2, 3> A1T{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const Matrix<double, 1, 3> A2T{7.0, 8.0, 9.0};
    const auto op = make_concat_range_op<MundyMathBackend>(A1T, A2T);
    static_assert(std::remove_cvref_t<decltype(op)>::static_domain_size() == 3);
    static_assert(std::remove_cvref_t<decltype(op)>::static_range_size() == 3);
    Vector3d y;
    MundyMathBackend::apply(op, Vector3d{1.0, -2.0, 3.0}, y);
    return y;
  }
};

struct NestedConcatDomainCase {
  // [A1 + A1 | A2] = [[2, 4, 7], [6, 8, 8], [10, 12, 9]]; the first child's sizes come from a SumOp.
  KOKKOS_INLINE_FUNCTION Vector3d operator()() const {
    const Matrix<double, 3, 2> A1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const Matrix<double, 3, 1> A2{7.0, 8.0, 9.0};
    const auto op = make_concat_domain_op<MundyMathBackend>(make_sum_op<MundyMathBackend>(A1, A1), A2);
    static_assert(std::remove_cvref_t<decltype(op)>::static_domain_size() == 3);
    Vector3d y;
    MundyMathBackend::apply(op, Vector3d{1.0, -2.0, 3.0}, y);
    return y;
  }
};

// Evaluates the case on the host and inside a kernel; both must equal the dense block product exactly.
template <class Case>
void expect_case_on_host_and_device(const Case& concat_case, const Vector3d& expected) {
  const Vector3d host = concat_case();
  Kokkos::View<double[3], Kokkos::DefaultExecutionSpace::memory_space> device("device");
  Kokkos::parallel_for(
      "concat_case", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(const int) {
        const Vector3d y = concat_case();
        for (size_t i = 0; i < 3; ++i) {
          device(i) = y[i];
        }
      });
  const auto device_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(host[i], expected[i]) << "host entry " << i;
    EXPECT_DOUBLE_EQ(device_host(i), expected[i]) << "device entry " << i;
  }
}

TEST(LinearOperators, ConcatOpsOnMundyMathBackend) {
  expect_case_on_host_and_device(ConcatDomainCase{}, Vector3d{18.0, 19.0, 20.0});
  expect_case_on_host_and_device(ConcatRangeCase{}, Vector3d{6.0, 12.0, 18.0});
  expect_case_on_host_and_device(NestedConcatDomainCase{}, Vector3d{15.0, 14.0, 13.0});
}

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
using dense_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space>;
using sparse_matrix_t =
    KokkosSparse::CrsMatrix<double, int,
                            Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>,
                            void, size_t>;

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

// The MundyMath backend's Concat cases through the Kokkos backend, applied to x = (1, -2, 3), with every block a dense
// and then a sparse matrix. Entries are integers and halves, so every sum is exact.
TEST(LinearOperators, MatrixViewsCompose) {
  const dense_matrix_t A1 = make_dense_matrix(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const dense_matrix_t A2 = make_dense_matrix(3, 1, {7.0, 8.0, 9.0});
  const dense_matrix_t A1T = make_dense_matrix(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const dense_matrix_t A2T = make_dense_matrix(1, 3, {7.0, 8.0, 9.0});
  const dense_matrix_t A = make_dense_matrix(3, 3, {1.0, 2.0, 7.0, 3.0, 4.0, 8.0, 5.0, 6.0, 9.0});  // [A1 | A2]
  const view_t x = make_view({1.0, -2.0, 3.0});

  const auto check = [&](const auto& a1, const auto& a2, const auto& a1t, const auto& a2t, const auto& a,
                         const char* kind) {
    const auto applied = [&](const auto& op) {
      view_t y = backend_t::make_range_vector(op);
      backend_t::apply(op, x, y);
      return to_host(y);
    };
    EXPECT_EQ(applied(make_concat_domain_op<backend_t>(a1, a2)), (std::vector<double>{18.0, 19.0, 20.0})) << kind;
    EXPECT_EQ(applied(make_concat_range_op<backend_t>(a1t, a2t)), (std::vector<double>{6.0, 12.0, 18.0})) << kind;
    EXPECT_EQ(applied(make_concat_domain_op<backend_t>(make_sum_op<backend_t>(a1, a1), a2)),
              (std::vector<double>{15.0, 14.0, 13.0}))
        << kind;
    EXPECT_EQ(applied(make_scaled_op<backend_t>(2.0, a)), (std::vector<double>{36.0, 38.0, 40.0})) << kind;
    EXPECT_EQ(applied(make_shifted_op<backend_t>(0.5, a)), (std::vector<double>{17.5, 20.0, 18.5})) << kind;
  };
  check(A1, A2, A1T, A2T, A, "dense");
  check(make_sparse_matrix<sparse_matrix_t>(A1), make_sparse_matrix<sparse_matrix_t>(A2),
        make_sparse_matrix<sparse_matrix_t>(A1T), make_sparse_matrix<sparse_matrix_t>(A2T),
        make_sparse_matrix<sparse_matrix_t>(A), "sparse");
}
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

TEST(LinearOperators, ConcatDomainOpSplitsInputAndSumsContributions) {
  // [op1 | op2]: op1 has domain 2 -> writes rows {0,1}; op2 has domain 1 -> writes row {2}; shared range 3.
  // apply([x1; x2]) = op1(x1) + op2(x2), with disjoint output rows so the domain split is unambiguous.
  SliceScaleOp op1{2.0, 2, 3, 0};
  SliceScaleOp op2{3.0, 1, 3, 2};
  auto concat = ConcatDomainOp(backend_t{}, SliceScaleOp(op1), SliceScaleOp(op2));

  EXPECT_EQ(concat.domain_size(), 3u);
  EXPECT_EQ(concat.range_size(), 3u);

  const view_t x = make_view({1.0, 2.0, 5.0});  // x1 = [1, 2] (-> op1), x2 = [5] (-> op2)
  view_t y = concat.make_range_vector();
  concat.apply(x, y);

  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 2.0 * 1.0);  // op1 row 0
  EXPECT_DOUBLE_EQ(result[1], 2.0 * 2.0);  // op1 row 1
  EXPECT_DOUBLE_EQ(result[2], 3.0 * 5.0);  // op2 row 2
}

TEST(LinearOperators, ConcatRangeOpStacksBothOperatorsOutputs) {
  ScaleOp op1(2.0);
  op1.set_size(2);
  ScaleOp op2(3.0);
  op2.set_size(2);
  auto concat = ConcatRangeOp(backend_t{}, ScaleOp(op1), ScaleOp(op2));

  EXPECT_EQ(concat.domain_size(), 2u);
  EXPECT_EQ(concat.range_size(), 4u);

  const view_t x = make_view({1.0, 2.0});
  view_t y = concat.make_range_vector();
  concat.apply(x, y);

  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 2.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0 * 2.0);
  EXPECT_DOUBLE_EQ(result[2], 3.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[3], 3.0 * 2.0);
}

TEST(LinearOperators, DiagonalOpMultipliesElementwise) {
  const view_t diag = make_view({2.0, -3.0, 0.5});
  auto diag_op = DiagonalOp(backend_t{}, view_t(diag));

  const view_t x = make_view({1.0, 2.0, 4.0});
  view_t y = diag_op.make_range_vector();
  diag_op.apply(x, y);

  const std::vector<double> result = to_host(y);
  EXPECT_DOUBLE_EQ(result[0], 2.0 * 1.0);
  EXPECT_DOUBLE_EQ(result[1], -3.0 * 2.0);
  EXPECT_DOUBLE_EQ(result[2], 0.5 * 4.0);
}

TEST(LinearOperators, CommitGroupPropagatesToChildren) {
  impl::CommitGroup<impl::NoWorkspace, impl::NoWorkspace> group{impl::NoWorkspace{}, impl::NoWorkspace{}};

  EXPECT_FALSE(group.is_committed());
  group.commit();
  EXPECT_TRUE(group.is_committed());
  group.invalidate();
  EXPECT_FALSE(group.is_committed());
}

}  // namespace

}  // namespace mundy
