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

#ifndef MUNDY_MATH_LINEAR_OPS_HPP_
#define MUNDY_MATH_LINEAR_OPS_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/impl/linear_ops_impl.hpp>  // for mundy::impl::CommitGroup
#include <mundy_math/solver_backends.hpp>       // for mundy::{Backend concepts, KokkosBackend, MundyMathBackend, ...}
#include <mundy_utils/storage.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

// Composite linear operators built from other operators; not specific to any one problem class.

/// \brief Declare a workspace member `name`: its backing storage and mutable/const getters.
///
/// Emits a private ::mundy::storage<VectorType> name##_storage_ and public name()/name() const over it. The
/// constructor still initializes name##_storage_ itself.
#define MUNDY_OP_WORKSPACE(name, VectorType)        \
 public:                                            \
  KOKKOS_INLINE_FUNCTION auto& name() {             \
    return name##_storage_.get();                   \
  }                                                 \
  KOKKOS_INLINE_FUNCTION const auto& name() const { \
    return name##_storage_.get();                   \
  }                                                 \
                                                    \
 private:                                           \
  ::mundy::storage<VectorType> name##_storage_;

/// \brief Declare mutable/const getters over the I'th CommitGroup child workspace.
#define MUNDY_OP_WORKSPACE_CHILD(name, index)       \
 public:                                            \
  KOKKOS_INLINE_FUNCTION auto& name() {             \
    return this->template child<index>();           \
  }                                                 \
  KOKKOS_INLINE_FUNCTION const auto& name() const { \
    return this->template child<index>();           \
  }

/// \brief The quadratic form Op := D^T M D.
template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
class QuadraticFormOp {
 public:
  using backend_t = Backend;
  using linear_op_dt_storage_t = ::mundy::storage<LinearOpDT>;
  using linear_op_m_storage_t = ::mundy::storage<LinearOpM>;
  using linear_op_d_storage_t = ::mundy::storage<LinearOpD>;

  template <class FVector, class UVector>
  struct Workspace : impl::CommitGroup<> {
   private:
    using base_t = impl::CommitGroup<>;

   public:
    KOKKOS_INLINE_FUNCTION Workspace(FVector&& f, UVector&& u, bool committed = false)
        : base_t(committed), f_storage_(std::forward<FVector>(f)), u_storage_(std::forward<UVector>(u)) {
    }

    KOKKOS_INLINE_FUNCTION Backend backend() const {
      return Backend{};
    }
    MUNDY_OP_WORKSPACE(f, FVector)
    MUNDY_OP_WORKSPACE(u, UVector)
  };

  KOKKOS_INLINE_FUNCTION
  QuadraticFormOp(backend_t, LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D)
      : DT_storage_(std::forward<LinearOpDT>(DT)),
        M_storage_(std::forward<LinearOpM>(M)),
        D_storage_(std::forward<LinearOpD>(D)) {
  }

  // Accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return DT_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return M_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return D_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(D());
  }

  size_t range_size() const {
    return Backend::range_size(DT());
  }

  auto make_domain_vector() const {
    return Backend::make_domain_vector(D());
  }

  auto make_range_vector() const {
    return Backend::make_range_vector(DT());
  }

  auto make_workspace() const {
    return make_workspace(Backend::make_domain_vector(M_storage_.get()), Backend::make_range_vector(M_storage_.get()));
  }

  template <class FVector, class UVector>
  auto make_workspace(FVector&& f, UVector&& u, bool committed = false) const {
    return Workspace<FVector, UVector>(std::forward<FVector>(f), std::forward<UVector>(u), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    Backend::apply(D_storage_.get(), x, workspace.f());
    Backend::apply(M_storage_.get(), workspace.f(), workspace.u());
    Backend::apply(DT_storage_.get(), workspace.u(), y);
  }

  // A freshly made workspace already starts invalidated (make_workspace() defaults committed=false), and this
  // direct member call bypasses the Backend::apply dispatch layer, so no separate invalidate is needed here.
  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  linear_op_dt_storage_t DT_storage_;
  linear_op_m_storage_t M_storage_;
  linear_op_d_storage_t D_storage_;
};

/// \brief The reduced operator Op := A (I - L A).
template <class Backend, class LinearOpA, class LinearOpL>
class MixedReducedOp {
 public:
  using backend_t = Backend;
  using linear_op_a_storage_t = ::mundy::storage<LinearOpA>;
  using linear_op_l_storage_t = ::mundy::storage<LinearOpL>;

  template <class AxVector, class LAxVector, class AWorkspace, class LWorkspace>
  struct Workspace : impl::CommitGroup<AWorkspace, LWorkspace> {
   private:
    using base_t = impl::CommitGroup<AWorkspace, LWorkspace>;

   public:
    KOKKOS_INLINE_FUNCTION
    Workspace(AxVector&& ax, LAxVector&& lax, AWorkspace&& a_workspace, LWorkspace&& l_workspace,
              bool committed = false)
        : base_t(std::forward<AWorkspace>(a_workspace), std::forward<LWorkspace>(l_workspace), committed),
          ax_storage_(std::forward<AxVector>(ax)),
          lax_storage_(std::forward<LAxVector>(lax)) {
    }

    KOKKOS_INLINE_FUNCTION Backend backend() const {
      return Backend{};
    }
    MUNDY_OP_WORKSPACE(ax, AxVector)
    MUNDY_OP_WORKSPACE(lax, LAxVector)
    MUNDY_OP_WORKSPACE_CHILD(a_workspace, 0)
    MUNDY_OP_WORKSPACE_CHILD(l_workspace, 1)
  };

  KOKKOS_INLINE_FUNCTION
  MixedReducedOp(backend_t, LinearOpA&& A, LinearOpL&& L)
      : A_storage_(std::forward<LinearOpA>(A)), L_storage_(std::forward<LinearOpL>(L)) {
  }
  // Accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return L_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(A());
  }

  size_t range_size() const {
    return Backend::range_size(A());
  }

  auto make_domain_vector() const {
    return Backend::make_domain_vector(A());
  }

  auto make_range_vector() const {
    return Backend::make_range_vector(A());
  }

  auto make_workspace(bool committed = false) const {
    return make_workspace(Backend::make_range_vector(A()), Backend::make_range_vector(L()), impl::make_workspace(A()),
                          impl::make_workspace(L()), committed);
  }

  template <class AxVector, class LAxVector, class AWorkspace, class LWorkspace>
  auto make_workspace(AxVector&& ax, LAxVector&& lax, AWorkspace&& a_workspace, LWorkspace&& l_workspace,
                      bool committed = false) const {
    return Workspace<AxVector, LAxVector, AWorkspace, LWorkspace>(
        std::forward<AxVector>(ax), std::forward<LAxVector>(lax), std::forward<AWorkspace>(a_workspace),
        std::forward<LWorkspace>(l_workspace), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    constexpr auto one = static_cast<impl::vector_value_type<XVector>>(1);
    // workspace.ax = A x
    Backend::apply(A(), x, workspace.ax(), workspace.a_workspace());

    // workspace.lax = L (A x)
    Backend::apply(L_storage_.get(), workspace.ax(), workspace.lax(), workspace.l_workspace());

    // workspace.lax = x - L (A x) = (I - L A) x
    Backend::axpby(one, x, -one, workspace.lax());

    // y = A (I - L A) x
    Backend::apply(A_storage_.get(), workspace.lax(), y, workspace.a_workspace());
  }

  // A freshly made workspace already starts invalidated (make_workspace() defaults committed=false), and this
  // direct member call bypasses the Backend::apply dispatch layer, so no separate invalidate is needed here.
  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  linear_op_a_storage_t A_storage_;
  linear_op_l_storage_t L_storage_;
};

/// \brief The congruent reduced operator Op := D^T M (D - L M D).
template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD, class LinearOpL>
class CongruentMixedReducedOp {
 public:
  using backend_t = Backend;
  using linear_op_dt_storage_t = ::mundy::storage<LinearOpDT>;
  using linear_op_m_storage_t = ::mundy::storage<LinearOpM>;
  using linear_op_d_storage_t = ::mundy::storage<LinearOpD>;
  using linear_op_l_storage_t = ::mundy::storage<LinearOpL>;

  template <class DxVector, class MDxVector, class LMDxVector, class DTWorkspace, class MWorkspace, class DWorkspace,
            class LWorkspace>
  struct Workspace : impl::CommitGroup<DTWorkspace, MWorkspace, DWorkspace, LWorkspace> {
   private:
    using base_t = impl::CommitGroup<DTWorkspace, MWorkspace, DWorkspace, LWorkspace>;

   public:
    KOKKOS_INLINE_FUNCTION
    Workspace(DxVector&& dx, MDxVector&& mdx, LMDxVector&& lmdx, DTWorkspace&& dt_workspace, MWorkspace&& m_workspace,
              DWorkspace&& d_workspace, LWorkspace&& l_workspace, bool committed = false)
        : base_t(std::forward<DTWorkspace>(dt_workspace), std::forward<MWorkspace>(m_workspace),
                 std::forward<DWorkspace>(d_workspace), std::forward<LWorkspace>(l_workspace), committed),
          dx_storage_(std::forward<DxVector>(dx)),
          mdx_storage_(std::forward<MDxVector>(mdx)),
          lmdx_storage_(std::forward<LMDxVector>(lmdx)) {
    }

    KOKKOS_INLINE_FUNCTION Backend backend() const {
      return Backend{};
    }
    MUNDY_OP_WORKSPACE(dx, DxVector)
    MUNDY_OP_WORKSPACE(mdx, MDxVector)
    MUNDY_OP_WORKSPACE(lmdx, LMDxVector)
    MUNDY_OP_WORKSPACE_CHILD(dt_workspace, 0)
    MUNDY_OP_WORKSPACE_CHILD(m_workspace, 1)
    MUNDY_OP_WORKSPACE_CHILD(d_workspace, 2)
    MUNDY_OP_WORKSPACE_CHILD(l_workspace, 3)
  };

  KOKKOS_INLINE_FUNCTION
  CongruentMixedReducedOp(backend_t, LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, LinearOpL&& L)
      : DT_storage_(std::forward<LinearOpDT>(DT)),
        M_storage_(std::forward<LinearOpM>(M)),
        D_storage_(std::forward<LinearOpD>(D)),
        L_storage_(std::forward<LinearOpL>(L)) {
  }

  // Accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return DT_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return M_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return D_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return L_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(D());
  }

  size_t range_size() const {
    return Backend::range_size(DT());
  }

  auto make_domain_vector() const {
    return Backend::make_domain_vector(D());
  }

  auto make_range_vector() const {
    return Backend::make_range_vector(DT());
  }

  auto make_workspace(bool committed = false) const {
    return make_workspace(Backend::make_range_vector(D()), Backend::make_range_vector(M()),
                          Backend::make_range_vector(L()), impl::make_workspace(DT()), impl::make_workspace(M()),
                          impl::make_workspace(D()), impl::make_workspace(L()), committed);
  }

  template <class DxVector, class MDxVector, class LMDxVector, class DTWorkspace, class MWorkspace, class DWorkspace,
            class LWorkspace>
  auto make_workspace(DxVector&& dx, MDxVector&& mdx, LMDxVector&& lmdx, DTWorkspace&& dt_workspace,
                      MWorkspace&& m_workspace, DWorkspace&& d_workspace, LWorkspace&& l_workspace,
                      bool committed = false) const {
    return Workspace<DxVector, MDxVector, LMDxVector, DTWorkspace, MWorkspace, DWorkspace, LWorkspace>(
        std::forward<DxVector>(dx), std::forward<MDxVector>(mdx), std::forward<LMDxVector>(lmdx),
        std::forward<DTWorkspace>(dt_workspace), std::forward<MWorkspace>(m_workspace),
        std::forward<DWorkspace>(d_workspace), std::forward<LWorkspace>(l_workspace), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    constexpr auto one = static_cast<impl::vector_value_type<XVector>>(1);
    Backend::apply(D(), x, workspace.dx(), workspace.d_workspace());
    Backend::apply(M(), workspace.dx(), workspace.mdx(), workspace.m_workspace());
    Backend::apply(L(), workspace.mdx(), workspace.lmdx(), workspace.l_workspace());
    Backend::axpby(one, workspace.dx(), -one, workspace.lmdx());
    Backend::apply(M(), workspace.lmdx(), workspace.mdx(), workspace.m_workspace());
    Backend::apply(DT(), workspace.mdx(), y, workspace.dt_workspace());
  }

  // A freshly made workspace already starts invalidated (make_workspace() defaults committed=false), and this
  // direct member call bypasses the Backend::apply dispatch layer, so no separate invalidate is needed here.
  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  linear_op_dt_storage_t DT_storage_;
  linear_op_m_storage_t M_storage_;
  linear_op_d_storage_t D_storage_;
  linear_op_l_storage_t L_storage_;
};

/// \brief The sum Op := op1 + op2 of two operators sharing a domain and range.
template <class Backend, class Op1, class Op2>
class SumOp {
 public:
  using backend_t = Backend;
  using op1_storage_t = ::mundy::storage<Op1>;
  using op2_storage_t = ::mundy::storage<Op2>;

  template <class TmpVector, class Op1Workspace, class Op2Workspace>
  struct Workspace : impl::CommitGroup<Op1Workspace, Op2Workspace> {
   private:
    using base_t = impl::CommitGroup<Op1Workspace, Op2Workspace>;

   public:
    KOKKOS_INLINE_FUNCTION
    Workspace(TmpVector&& tmp, Op1Workspace&& op1_workspace, Op2Workspace&& op2_workspace, bool committed = false)
        : base_t(std::forward<Op1Workspace>(op1_workspace), std::forward<Op2Workspace>(op2_workspace), committed),
          tmp_storage_(std::forward<TmpVector>(tmp)) {
    }

    KOKKOS_INLINE_FUNCTION Backend backend() const {
      return Backend{};
    }
    MUNDY_OP_WORKSPACE(tmp, TmpVector)
    MUNDY_OP_WORKSPACE_CHILD(op1_workspace, 0)
    MUNDY_OP_WORKSPACE_CHILD(op2_workspace, 1)
  };

  KOKKOS_INLINE_FUNCTION
  SumOp(backend_t, Op1&& op1, Op2&& op2) : op1_storage_(std::forward<Op1>(op1)), op2_storage_(std::forward<Op2>(op2)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(op1_storage_.get()) == Backend::domain_size(op2_storage_.get()),
                       std::invalid_argument, "SumOp: domain size mismatch.");
    MUNDY_THROW_ASSERT(Backend::range_size(op1_storage_.get()) == Backend::range_size(op2_storage_.get()),
                       std::invalid_argument, "SumOp: range size mismatch.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& op1() const { return op1_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& op2() const { return op2_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(op1());
  }

  size_t range_size() const {
    return Backend::range_size(op1());
  }

  auto make_domain_vector() const {
    return Backend::make_domain_vector(op1());
  }

  auto make_range_vector() const {
    return Backend::make_range_vector(op1());
  }

  auto make_workspace(bool committed = false) const {
    return make_workspace(Backend::make_range_vector(op1()), impl::make_workspace(op1()), impl::make_workspace(op2()),
                          committed);
  }

  template <class TmpVector, class Op1Workspace, class Op2Workspace>
  auto make_workspace(TmpVector&& tmp, Op1Workspace&& op1_workspace, Op2Workspace&& op2_workspace,
                      bool committed = false) const {
    return Workspace<TmpVector, Op1Workspace, Op2Workspace>(std::forward<TmpVector>(tmp),
                                                            std::forward<Op1Workspace>(op1_workspace),
                                                            std::forward<Op2Workspace>(op2_workspace), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    constexpr auto one = static_cast<impl::vector_value_type<YVector>>(1);
    Backend::apply(op1(), x, y, workspace.op1_workspace());
    Backend::apply(op2(), x, workspace.tmp(), workspace.op2_workspace());
    Backend::axpby(one, workspace.tmp(), one, y);
  }

  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  op1_storage_t op1_storage_;
  op2_storage_t op2_storage_;
};

/// \brief The scalar multiple Op := alpha * op.
///
/// Applies via Backend::apply(alpha, op, x, beta, y): the fused path when op provides HasScaledApplyMember, else
/// an apply-then-axpby fallback. Holds no persistent workspace; op's own (if any) is rebuilt per apply.
template <class Backend, class Scalar, class Op>
class ScaledOp {
 public:
  using backend_t = Backend;
  using op_storage_t = ::mundy::storage<Op>;

  KOKKOS_INLINE_FUNCTION
  ScaledOp(backend_t, Scalar alpha, Op&& op) : alpha_(alpha), op_storage_(std::forward<Op>(op)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION Scalar alpha() const { return alpha_; }
  KOKKOS_INLINE_FUNCTION const auto& op() const { return op_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(op());
  }

  size_t range_size() const {
    return Backend::range_size(op());
  }

  auto make_domain_vector() const {
    return Backend::make_domain_vector(op());
  }

  auto make_range_vector() const {
    return Backend::make_range_vector(op());
  }

  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    constexpr auto zero = static_cast<impl::vector_value_type<YVector>>(0);
    Backend::apply(alpha_, op(), x, zero, y);
  }

 private:
  Scalar alpha_;
  op_storage_t op_storage_;
};

/// \brief The domain concatenation Op := [op1 | op2].
///
/// op1 and op2 share a range but have independent domains: apply([x1; x2]) = op1(x1) + op2(x2). x/y must be
/// Kokkos::View-backed (apply splits x via Kokkos::subview).
template <class Backend, class Op1, class Op2>
class ConcatDomainOp {
 public:
  using backend_t = Backend;
  using op1_storage_t = ::mundy::storage<Op1>;
  using op2_storage_t = ::mundy::storage<Op2>;

  template <class Y2Vector, class Op1Workspace, class Op2Workspace>
  struct Workspace : impl::CommitGroup<Op1Workspace, Op2Workspace> {
   private:
    using base_t = impl::CommitGroup<Op1Workspace, Op2Workspace>;

   public:
    KOKKOS_INLINE_FUNCTION
    Workspace(Y2Vector&& y2, Op1Workspace&& op1_workspace, Op2Workspace&& op2_workspace, bool committed = false)
        : base_t(std::forward<Op1Workspace>(op1_workspace), std::forward<Op2Workspace>(op2_workspace), committed),
          y2_storage_(std::forward<Y2Vector>(y2)) {
    }

    KOKKOS_INLINE_FUNCTION Backend backend() const {
      return Backend{};
    }
    MUNDY_OP_WORKSPACE(y2, Y2Vector)
    MUNDY_OP_WORKSPACE_CHILD(op1_workspace, 0)
    MUNDY_OP_WORKSPACE_CHILD(op2_workspace, 1)
  };

  KOKKOS_INLINE_FUNCTION
  ConcatDomainOp(backend_t, Op1&& op1, Op2&& op2)
      : op1_storage_(std::forward<Op1>(op1)), op2_storage_(std::forward<Op2>(op2)) {
    MUNDY_THROW_ASSERT(Backend::range_size(op1_storage_.get()) == Backend::range_size(op2_storage_.get()),
                       std::invalid_argument, "ConcatDomainOp: range size mismatch.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& op1() const { return op1_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& op2() const { return op2_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(op1()) + Backend::domain_size(op2());
  }

  size_t range_size() const {
    return Backend::range_size(op1());
  }

  auto make_domain_vector() const {
    using vector_t = decltype(Backend::make_domain_vector(op1()));
    return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ConcatDomainOp_domain"), domain_size());
  }

  auto make_range_vector() const {
    return Backend::make_range_vector(op1());
  }

  auto make_workspace(bool committed = false) const {
    return make_workspace(Backend::make_range_vector(op2()), impl::make_workspace(op1()), impl::make_workspace(op2()),
                          committed);
  }

  template <class Y2Vector, class Op1Workspace, class Op2Workspace>
  auto make_workspace(Y2Vector&& y2, Op1Workspace&& op1_workspace, Op2Workspace&& op2_workspace,
                      bool committed = false) const {
    return Workspace<Y2Vector, Op1Workspace, Op2Workspace>(std::forward<Y2Vector>(y2),
                                                           std::forward<Op1Workspace>(op1_workspace),
                                                           std::forward<Op2Workspace>(op2_workspace), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    constexpr auto one = static_cast<impl::vector_value_type<YVector>>(1);
    const size_t n1 = Backend::domain_size(op1());
    const size_t n2 = Backend::domain_size(op2());
    const XVector x1 = Kokkos::subview(x, Kokkos::pair<size_t, size_t>(0, n1));
    const XVector x2 = Kokkos::subview(x, Kokkos::pair<size_t, size_t>(n1, n1 + n2));
    Backend::apply(op1(), x1, y, workspace.op1_workspace());
    Backend::apply(op2(), x2, workspace.y2(), workspace.op2_workspace());
    Backend::axpby(one, workspace.y2(), one, y);
  }

  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  op1_storage_t op1_storage_;
  op2_storage_t op2_storage_;
};

/// \brief The range concatenation Op := [op1T; op2T], the transpose of ConcatDomainOp.
///
/// op1T and op2T share a domain but produce independent ranges: apply(v) = [op1T(v); op2T(v)]. x/y must be
/// Kokkos::View-backed (apply splits y via Kokkos::subview).
template <class Backend, class Op1T, class Op2T>
class ConcatRangeOp {
 public:
  using backend_t = Backend;
  using op1t_storage_t = ::mundy::storage<Op1T>;
  using op2t_storage_t = ::mundy::storage<Op2T>;

  template <class Op1TWorkspace, class Op2TWorkspace>
  struct Workspace : impl::CommitGroup<Op1TWorkspace, Op2TWorkspace> {
   private:
    using base_t = impl::CommitGroup<Op1TWorkspace, Op2TWorkspace>;

   public:
    KOKKOS_INLINE_FUNCTION
    Workspace(Op1TWorkspace&& op1t_workspace, Op2TWorkspace&& op2t_workspace, bool committed = false)
        : base_t(std::forward<Op1TWorkspace>(op1t_workspace), std::forward<Op2TWorkspace>(op2t_workspace), committed) {
    }

    KOKKOS_INLINE_FUNCTION Backend backend() const {
      return Backend{};
    }
    MUNDY_OP_WORKSPACE_CHILD(op1t_workspace, 0)
    MUNDY_OP_WORKSPACE_CHILD(op2t_workspace, 1)
  };

  KOKKOS_INLINE_FUNCTION
  ConcatRangeOp(backend_t, Op1T&& op1t, Op2T&& op2t)
      : op1t_storage_(std::forward<Op1T>(op1t)), op2t_storage_(std::forward<Op2T>(op2t)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(op1t_storage_.get()) == Backend::domain_size(op2t_storage_.get()),
                       std::invalid_argument, "ConcatRangeOp: domain size mismatch.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& op1t() const { return op1t_storage_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& op2t() const { return op2t_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return Backend::domain_size(op1t());
  }

  size_t range_size() const {
    return Backend::range_size(op1t()) + Backend::range_size(op2t());
  }

  auto make_domain_vector() const {
    return Backend::make_domain_vector(op1t());
  }

  auto make_range_vector() const {
    using vector_t = decltype(Backend::make_range_vector(op1t()));
    return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ConcatRangeOp_range"), range_size());
  }

  auto make_workspace(bool committed = false) const {
    return make_workspace(impl::make_workspace(op1t()), impl::make_workspace(op2t()), committed);
  }

  template <class Op1TWorkspace, class Op2TWorkspace>
  auto make_workspace(Op1TWorkspace&& op1t_workspace, Op2TWorkspace&& op2t_workspace, bool committed = false) const {
    return Workspace<Op1TWorkspace, Op2TWorkspace>(std::forward<Op1TWorkspace>(op1t_workspace),
                                                   std::forward<Op2TWorkspace>(op2t_workspace), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    const size_t n1 = Backend::range_size(op1t());
    const size_t n2 = Backend::range_size(op2t());
    YVector y1 = Kokkos::subview(y, Kokkos::pair<size_t, size_t>(0, n1));
    YVector y2 = Kokkos::subview(y, Kokkos::pair<size_t, size_t>(n1, n1 + n2));
    Backend::apply(op1t(), x, y1, workspace.op1t_workspace());
    Backend::apply(op2t(), x, y2, workspace.op2t_workspace());
  }

  template <class XVector, class YVector>
  KOKKOS_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  op1t_storage_t op1t_storage_;
  op2t_storage_t op2t_storage_;
};

/// \brief The diagonal operator y := diag .* x.
///
/// Requires a Kokkos::View-backed backend (uses Kokkos::parallel_for over Backend::exec_space).
template <class Backend, class DiagVector>
class DiagonalOp {
 public:
  using backend_t = Backend;

  KOKKOS_INLINE_FUNCTION
  explicit DiagonalOp(backend_t, DiagVector&& diag) : diag_storage_(std::forward<DiagVector>(diag)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& diag() const { return diag_storage_.get(); }
  // clang-format on

  size_t domain_size() const {
    return diag().extent(0);
  }

  size_t range_size() const {
    return diag().extent(0);
  }

  auto make_domain_vector() const {
    return Backend::make_vector_like(diag());
  }

  auto make_range_vector() const {
    return Backend::make_vector_like(diag());
  }

  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    MUNDY_THROW_ASSERT(x.extent(0) == diag().extent(0), std::invalid_argument, "DiagonalOp: size mismatch.");
    auto diag = diag_storage_.get();
    Kokkos::parallel_for(
        "DiagonalOp::apply", Kokkos::RangePolicy<typename Backend::exec_space>(0, diag.extent(0)),
        KOKKOS_LAMBDA(const int i) { y(i) = diag(i) * x(i); });
  }

 private:
  ::mundy::storage<DiagVector> diag_storage_;
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
QuadraticFormOp(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&)
    -> QuadraticFormOp<Backend, LinearOpDT, LinearOpM, LinearOpD>;

template <class Backend, class LinearOpA, class LinearOpL>
MixedReducedOp(Backend, LinearOpA&&, LinearOpL&&) -> MixedReducedOp<Backend, LinearOpA, LinearOpL>;

template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD, class LinearOpL>
CongruentMixedReducedOp(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&, LinearOpL&&)
    -> CongruentMixedReducedOp<Backend, LinearOpDT, LinearOpM, LinearOpD, LinearOpL>;

template <class Backend, class Op1, class Op2>
SumOp(Backend, Op1&&, Op2&&) -> SumOp<Backend, Op1, Op2>;

template <class Backend, class Scalar, class Op>
ScaledOp(Backend, Scalar, Op&&) -> ScaledOp<Backend, Scalar, Op>;

template <class Backend, class Op1, class Op2>
ConcatDomainOp(Backend, Op1&&, Op2&&) -> ConcatDomainOp<Backend, Op1, Op2>;

template <class Backend, class Op1T, class Op2T>
ConcatRangeOp(Backend, Op1T&&, Op2T&&) -> ConcatRangeOp<Backend, Op1T, Op2T>;

template <class Backend, class DiagVector>
DiagonalOp(Backend, DiagVector&&) -> DiagonalOp<Backend, DiagVector>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <typename Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
KOKKOS_INLINE_FUNCTION auto make_quadratic_form(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D) {
  return QuadraticFormOp(Backend{}, std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                         std::forward<LinearOpD>(D));
}

template <typename Backend, class Op1, class Op2>
KOKKOS_INLINE_FUNCTION auto make_sum_op(Op1&& op1, Op2&& op2) {
  return SumOp(Backend{}, std::forward<Op1>(op1), std::forward<Op2>(op2));
}

template <typename Backend, class Scalar, class Op>
KOKKOS_INLINE_FUNCTION auto make_scaled_op(Scalar alpha, Op&& op) {
  return ScaledOp(Backend{}, alpha, std::forward<Op>(op));
}

template <typename Backend, class Op1, class Op2>
KOKKOS_INLINE_FUNCTION auto make_concat_domain_op(Op1&& op1, Op2&& op2) {
  return ConcatDomainOp(Backend{}, std::forward<Op1>(op1), std::forward<Op2>(op2));
}

template <typename Backend, class Op1T, class Op2T>
KOKKOS_INLINE_FUNCTION auto make_concat_range_op(Op1T&& op1t, Op2T&& op2t) {
  return ConcatRangeOp(Backend{}, std::forward<Op1T>(op1t), std::forward<Op2T>(op2t));
}

template <typename Backend, class DiagVector>
KOKKOS_INLINE_FUNCTION auto make_diagonal_op(DiagVector&& diag) {
  return DiagonalOp(Backend{}, std::forward<DiagVector>(diag));
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_LINEAR_OPS_HPP_
