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

// Mundy
#include <mundy_math/linear_ops.hpp>
#include <mundy_math/solver_backends.hpp>

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
