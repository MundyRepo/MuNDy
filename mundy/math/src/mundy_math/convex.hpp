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

#ifndef MUNDY_MATH_CONVEX_HPP_
#define MUNDY_MATH_CONVEX_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <functional>
#include <ostream>
#include <type_traits>
#include <utility>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosBlas.hpp>
#include <KokkosBlas_gesv.hpp>
#endif

// Mundy core:
#include <mundy_core/reference_wrapper.hpp>
#include <mundy_core/throw_assert.hpp>

// Mundy math:
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_zero_tolerance<T>
#include <mundy_math/Vector.hpp>     // for mundy::math::Vector
#include <mundy_core/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP

namespace mundy {

namespace math {

namespace convex {

namespace space {

// These are 1d convex spaces, which will be applied to each element of a vector assuming a separable convex space

/// \brief Proj(x) = x for all x in R
template <typename Scalar>
struct Unconstrained {
  using scalar_t = Scalar;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return x;
  }
};

/// \brief Proj(x) = max(x, lower_bound) for all x in R
template <typename Scalar>
struct LowerBound {
  using scalar_t = Scalar;

  scalar_t lower_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::max(x, lower_bound);
  }
};

/// \brief Proj(x) = min(x, upper_bound) for all x in R
template <typename Scalar>
struct UpperBound {
  using scalar_t = Scalar;

  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(x, upper_bound);
  }
};

/// \brief Proj(x) = min(max(x, lower_bound), upper_bound) for all x in R
template <typename Scalar>
struct Bounded {
  using scalar_t = Scalar;

  scalar_t lower_bound;
  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(Kokkos::max(x, lower_bound), upper_bound);
  }
};

/// \brief Concept for a valid space
template <class Space>
concept ValidConvexSpace = requires {
  typename std::remove_cvref_t<Space>::scalar_t;  // make the nested type check explicit
} && requires(const std::remove_cvref_t<Space>& s, typename std::remove_cvref_t<Space>::scalar_t x) {
  { s.project(x) } -> std::same_as<typename std::remove_cvref_t<Space>::scalar_t>;
  { s(x) } -> std::same_as<typename std::remove_cvref_t<Space>::scalar_t>;
};

/// \brief Assert that all of our spaces are valid convex spaces
static_assert(ValidConvexSpace<Unconstrained<double>>, "Unconstrained<double> does not satisfy ValidConvexSpace");
static_assert(ValidConvexSpace<LowerBound<double>>, "LowerBound<double> does not satisfy ValidConvexSpace");
static_assert(ValidConvexSpace<UpperBound<double>>, "UpperBound<double> does not satisfy ValidConvexSpace");
static_assert(ValidConvexSpace<Bounded<double>>, "Bounded<double> does not satisfy ValidConvexSpace");

}  // namespace space

//! \name Backends
//@{

namespace impl {

template <class Op>
concept DenseMatView = Kokkos::is_view_v<Op> && requires {
  // These are checked in constraint context:
  { std::remove_reference_t<Op>::rank } -> std::convertible_to<int>;
  typename std::remove_reference_t<Op>::non_const_value_type;
} && (std::remove_reference_t<Op>::rank == 2);

template <class T>
struct is_reference_wrapper : std::false_type {};

template <class U>
struct is_reference_wrapper<::mundy::core::reference_wrapper<U>> : std::true_type {};

template <class T>
inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<std::remove_cvref_t<T>>::value;

template <class Storage>
KOKKOS_INLINE_FUNCTION decltype(auto) unwrap(Storage& value) {
  if constexpr (is_reference_wrapper_v<Storage>) {
    return value.get();
  } else {
    return (value);
  }
}

template <class Storage>
KOKKOS_INLINE_FUNCTION decltype(auto) unwrap(const Storage& value) {
  if constexpr (is_reference_wrapper_v<Storage>) {
    return value.get();
  } else {
    return (value);
  }
}

template <class Storage>
using unwrapped_storage_t = std::remove_cvref_t<decltype(unwrap(std::declval<const Storage&>()))>;

template <class Op, typename X, typename Y>
concept HasApplyMember = requires(const Op& op, const X& x, Y& y) {
  { op.apply(x, y) } -> std::same_as<void>;
};

template <class Op, typename X, typename Y, typename Workspace>
concept HasApplyMemberWithWorkspace = requires(const Op& op, const X& x, Y& y, Workspace& workspace) {
  { op.apply(x, y, workspace) } -> std::same_as<void>;
};

template <class Op>
concept HasDomainSizeMember = requires(const Op& op) {
  { op.domain_size() } -> std::convertible_to<size_t>;
};

template <class Op>
concept HasRangeSizeMember = requires(const Op& op) {
  { op.range_size() } -> std::convertible_to<size_t>;
};

template <class Op>
concept HasMakeDomainVectorMember = requires(const Op& op) {
  { op.make_domain_vector() };
};

template <class Op>
concept HasMakeRangeVectorMember = requires(const Op& op) {
  { op.make_range_vector() };
};

template <class Workspace>
concept HasCommitMember = requires(Workspace& workspace) {
  { workspace.commit() } -> std::same_as<void>;
};

template <class Workspace>
concept HasInvalidateMember = requires(Workspace& workspace) {
  { workspace.invalidate() } -> std::same_as<void>;
};

template <class Workspace>
concept HasIsCommittedMember = requires(const Workspace& workspace) {
  { workspace.is_committed() } -> std::convertible_to<bool>;
};

template <class Workspace>
KOKKOS_INLINE_FUNCTION void workspace_commit(Workspace& workspace) {
  if constexpr (HasCommitMember<Workspace>) {
    workspace.commit();
  }
}

template <class Workspace>
KOKKOS_INLINE_FUNCTION void workspace_invalidate(Workspace& workspace) {
  if constexpr (HasInvalidateMember<Workspace>) {
    workspace.invalidate();
  }
}

template <class Workspace>
KOKKOS_INLINE_FUNCTION bool workspace_is_committed(const Workspace& workspace) {
  if constexpr (HasIsCommittedMember<Workspace>) {
    return workspace.is_committed();
  } else {
    return true;
  }
}

struct NoWorkspace {
  KOKKOS_INLINE_FUNCTION NoWorkspace() = default;

  // clang-format off
  KOKKOS_INLINE_FUNCTION void commit() { committed_ = true; }
  KOKKOS_INLINE_FUNCTION void invalidate() { committed_ = false; }
  KOKKOS_INLINE_FUNCTION bool is_committed() const { return committed_; }
  // clang-format on

 private:
  bool committed_{false};
};

template <class Op, class Vector>
concept HasMakeWorkspaceMember = requires(const Op& op, const Vector& q) {
  { op.make_workspace(impl::unwrap(q)) };
};

template <class Op>
concept HasMakeWorkspaceNoArgMember = requires(const Op& op) {
  { op.make_workspace() };
};

template <class...>
inline constexpr bool dependent_false_v = false;

template <class LinearOp>
auto make_workspace(LinearOp&& op) {
  if constexpr (HasMakeWorkspaceNoArgMember<LinearOp>) {
    return std::forward<LinearOp>(op).make_workspace();
  } else {
    return NoWorkspace();
  }
}

template <class LinearOp>
using workspace_for_t = decltype(make_workspace(std::declval<LinearOp>()));

template <class Vector>
using vector_scalar_t = std::remove_cvref_t<decltype(std::declval<const std::remove_reference_t<Vector>&>()(size_t{}))>;

/// \brief Convert a value to a storage that can either own or view the value.
///
/// Usage example:
/// \code{.cpp}
/// auto foo(const Vector3d& x) {
///   Vector3d tmp = 2 * x;  // tmp is an lvalue
///   Stash stash(to_storage(tmp));  // tmp here is an rvalue, so it will be moved into the stash's storage
///   return stash;
/// }
///
/// auto bar(const Vector3d& x) {
///   Stash stash(to_storage(x));  // x here is an lvalue reference, so it will be wrapped in a reference wrapper and
///   the stash will view x's data without copying return stash;
/// }
template <class T>
KOKKOS_INLINE_FUNCTION auto to_storage(T&& value) {
  if constexpr (is_reference_wrapper_v<T>) {  // value is already a reference wrapper, so just return it
    return std::forward<T>(value);
  } else if constexpr (std::is_lvalue_reference_v<T>) {  // value is an lvalue reference but not a reference wrapper, so
                                                         // wrap it in a reference wrapper
    return ::mundy::core::ref(value);
  } else {
    return std::forward<T>(value);  // value is an rvalue, so just return it as is (will be moved if possible)
  }
}

}  // namespace impl

template <class Backend, class LinearOpDTStorage, class LinearOpMStorage, class LinearOpDStorage>
class QuadraticFormOp {
 public:
  using backend_t = Backend;
  using linear_op_dt_storage_t = LinearOpDTStorage;
  using linear_op_m_storage_t = LinearOpMStorage;
  using linear_op_d_storage_t = LinearOpDStorage;

  template <class FVectorStorage, class UVectorStorage>
  struct Workspace {
    KOKKOS_INLINE_FUNCTION Workspace(FVectorStorage f_storage, UVectorStorage u_storage, bool committed = false)
        : f_storage_(std::move(f_storage)), u_storage_(std::move(u_storage)), committed_(committed) {
    }

    // clang-format off
    KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
    KOKKOS_INLINE_FUNCTION       auto& f()       { return impl::unwrap(f_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& f() const { return impl::unwrap(f_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& u()       { return impl::unwrap(u_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& u() const { return impl::unwrap(u_storage_); }
    KOKKOS_INLINE_FUNCTION void commit() { committed_ = true; }
    KOKKOS_INLINE_FUNCTION void invalidate() { committed_ = false; }
    KOKKOS_INLINE_FUNCTION bool is_committed() const { return committed_; }
    // clang-format on

   private:
    FVectorStorage f_storage_;
    UVectorStorage u_storage_;
    bool committed_{false};
  };

  KOKKOS_INLINE_FUNCTION
  QuadraticFormOp(backend_t, linear_op_dt_storage_t DT_storage, linear_op_m_storage_t M_storage,
                  linear_op_d_storage_t D_storage)
      : DT_storage_(std::move(DT_storage)), M_storage_(std::move(M_storage)), D_storage_(std::move(D_storage)) {
  }

  // Accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return impl::unwrap(DT_storage_); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return impl::unwrap(M_storage_); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return impl::unwrap(D_storage_); }
  KOKKOS_INLINE_FUNCTION
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
    auto f_storage = impl::to_storage(Backend::make_domain_vector(impl::unwrap(M_storage_)));
    auto u_storage = impl::to_storage(Backend::make_range_vector(impl::unwrap(M_storage_)));
    using f_storage_t = decltype(f_storage);
    using u_storage_t = decltype(u_storage);
    return Workspace<f_storage_t, u_storage_t>(std::move(f_storage), std::move(u_storage), false);
  }

  template <class FVector, class UVector>
  auto make_workspace(FVector&& f, UVector&& u, bool committed = false) const {
    auto f_storage = impl::to_storage(std::forward<FVector>(f));
    auto u_storage = impl::to_storage(std::forward<UVector>(u));
    using f_storage_t = decltype(f_storage);
    using u_storage_t = decltype(u_storage);
    return Workspace<f_storage_t, u_storage_t>(std::move(f_storage), std::move(u_storage), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    impl::workspace_invalidate(workspace);
    Backend::apply(impl::unwrap(D_storage_), x, workspace.f());
    Backend::apply(impl::unwrap(M_storage_), workspace.f(), workspace.u());
    Backend::apply(impl::unwrap(DT_storage_), workspace.u(), y);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  linear_op_dt_storage_t DT_storage_;
  linear_op_m_storage_t M_storage_;
  linear_op_d_storage_t D_storage_;
};

/// \brief The operator that perform Op x for Op := A (I - L A)
template <class Backend, class LinearOpAStorage, class LinearOpLStorage>
class MixedReducedOp {
 public:
  using backend_t = Backend;
  using linear_op_a_storage_t = LinearOpAStorage;
  using linear_op_l_storage_t = LinearOpLStorage;

  template <class AxVectorStorage, class LAxVectorStorage, class AWorkspaceStorage, class LWorkspaceStorage>
  struct Workspace {
    KOKKOS_INLINE_FUNCTION
    Workspace(AxVectorStorage ax_storage, LAxVectorStorage lax_storage, AWorkspaceStorage a_workspace_storage,
              LWorkspaceStorage l_workspace_storage, bool committed = false)
        : ax_storage_(std::move(ax_storage)),
          lax_storage_(std::move(lax_storage)),
          a_workspace_storage_(std::move(a_workspace_storage)),
          l_workspace_storage_(std::move(l_workspace_storage)),
          committed_(committed) {
    }

    // clang-format off
    KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
    KOKKOS_INLINE_FUNCTION       auto& ax()       { return impl::unwrap(ax_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& ax() const { return impl::unwrap(ax_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& lax()       { return impl::unwrap(lax_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& lax() const { return impl::unwrap(lax_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& a_workspace()       { return impl::unwrap(a_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& a_workspace() const { return impl::unwrap(a_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& l_workspace()       { return impl::unwrap(l_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& l_workspace() const { return impl::unwrap(l_workspace_storage_); }
    // clang-format on

    KOKKOS_INLINE_FUNCTION void commit() {
      committed_ = true;
      impl::workspace_commit(a_workspace());
      impl::workspace_commit(l_workspace());
    }

    KOKKOS_INLINE_FUNCTION void invalidate() {
      committed_ = false;
      impl::workspace_invalidate(a_workspace());
      impl::workspace_invalidate(l_workspace());
    }

    KOKKOS_INLINE_FUNCTION bool is_committed() const {
      return committed_ && impl::workspace_is_committed(a_workspace()) && impl::workspace_is_committed(l_workspace());
    }

   private:
    AxVectorStorage ax_storage_;
    LAxVectorStorage lax_storage_;
    AWorkspaceStorage a_workspace_storage_;
    LWorkspaceStorage l_workspace_storage_;
    bool committed_{false};
  };

  KOKKOS_INLINE_FUNCTION
  MixedReducedOp(backend_t, linear_op_a_storage_t A_storage, linear_op_l_storage_t L_storage)
      : A_storage_(std::move(A_storage)), L_storage_(std::move(L_storage)) {
  }
  // Accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_storage_); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return impl::unwrap(L_storage_); }
  KOKKOS_INLINE_FUNCTION
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
    auto ax_storage = impl::to_storage(Backend::make_range_vector(A()));
    auto lax_storage = impl::to_storage(Backend::make_range_vector(L()));
    auto a_workspace_storage = impl::make_workspace(A());
    auto l_workspace_storage = impl::make_workspace(L());

    using ax_storage_t = decltype(ax_storage);
    using lax_storage_t = decltype(lax_storage);
    using a_workspace_storage_t = decltype(a_workspace_storage);
    using l_workspace_storage_t = decltype(l_workspace_storage);

    return Workspace<ax_storage_t, lax_storage_t, a_workspace_storage_t, l_workspace_storage_t>(
        std::move(ax_storage), std::move(lax_storage), std::move(a_workspace_storage), std::move(l_workspace_storage),
        committed);
  }

  template <class AxVector, class LAxVector, class AWorkspace, class LWorkspace>
  auto make_workspace(AxVector&& ax, LAxVector&& lax, AWorkspace&& a_workspace, LWorkspace&& l_workspace,
                      bool committed = false) const {
    auto ax_storage = impl::to_storage(std::forward<AxVector>(ax));
    auto lax_storage = impl::to_storage(std::forward<LAxVector>(lax));
    auto a_workspace_storage = impl::to_storage(std::forward<AWorkspace>(a_workspace));
    auto l_workspace_storage = impl::to_storage(std::forward<LWorkspace>(l_workspace));
    using ax_storage_t = decltype(ax_storage);
    using lax_storage_t = decltype(lax_storage);
    using a_workspace_storage_t = decltype(a_workspace_storage);
    using l_workspace_storage_t = decltype(l_workspace_storage);
    return Workspace<ax_storage_t, lax_storage_t, a_workspace_storage_t, l_workspace_storage_t>(
        std::move(ax_storage), std::move(lax_storage), std::move(a_workspace_storage), std::move(l_workspace_storage),
        committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    impl::workspace_invalidate(workspace);
    constexpr auto one = static_cast<impl::vector_scalar_t<XVector>>(1);
    // workspace.ax = A x
    Backend::apply(A(), x, workspace.ax(), workspace.a_workspace());

    // workspace.lax = L (A x)
    Backend::apply(impl::unwrap(L_storage_), workspace.ax(), workspace.lax(), workspace.l_workspace());

    // workspace.lax = x - L (A x) = (I - L A) x
    Backend::axpby(one, x, -one, workspace.lax());

    // y = A (I - L A) x
    Backend::apply(impl::unwrap(A_storage_), workspace.lax(), y, workspace.a_workspace());
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  linear_op_a_storage_t A_storage_;
  linear_op_l_storage_t L_storage_;
};

/// \brief The operator that performs Op x for Op := D^T M (D - L M D)
template <class Backend, class LinearOpDTStorage, class LinearOpMStorage, class LinearOpDStorage,
          class LinearOpLStorage>
class CongruentMixedReducedOp {
 public:
  using backend_t = Backend;
  using linear_op_dt_storage_t = LinearOpDTStorage;
  using linear_op_m_storage_t = LinearOpMStorage;
  using linear_op_d_storage_t = LinearOpDStorage;
  using linear_op_l_storage_t = LinearOpLStorage;

  template <class DxVectorStorage, class MDxVectorStorage, class LMDxVectorStorage, class DTWorkspaceStorage,
            class MWorkspaceStorage, class DWorkspaceStorage, class LWorkspaceStorage>
  struct Workspace {
    KOKKOS_INLINE_FUNCTION
    Workspace(DxVectorStorage dx_storage, MDxVectorStorage mdx_storage, LMDxVectorStorage lmdx_storage,
              DTWorkspaceStorage dt_workspace_storage, MWorkspaceStorage m_workspace_storage,
              DWorkspaceStorage d_workspace_storage, LWorkspaceStorage l_workspace_storage, bool committed = false)
        : dx_storage_(std::move(dx_storage)),
          mdx_storage_(std::move(mdx_storage)),
          lmdx_storage_(std::move(lmdx_storage)),
          dt_workspace_storage_(std::move(dt_workspace_storage)),
          m_workspace_storage_(std::move(m_workspace_storage)),
          d_workspace_storage_(std::move(d_workspace_storage)),
          l_workspace_storage_(std::move(l_workspace_storage)),
          committed_(committed) {
    }

    // clang-format off
    KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
    KOKKOS_INLINE_FUNCTION       auto& dx()       { return impl::unwrap(dx_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& dx() const { return impl::unwrap(dx_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& mdx()       { return impl::unwrap(mdx_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& mdx() const { return impl::unwrap(mdx_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& lmdx()       { return impl::unwrap(lmdx_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& lmdx() const { return impl::unwrap(lmdx_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& dt_workspace()       { return impl::unwrap(dt_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& dt_workspace() const { return impl::unwrap(dt_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& m_workspace()       { return impl::unwrap(m_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& m_workspace() const { return impl::unwrap(m_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& d_workspace()       { return impl::unwrap(d_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& d_workspace() const { return impl::unwrap(d_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION       auto& l_workspace()       { return impl::unwrap(l_workspace_storage_); }
    KOKKOS_INLINE_FUNCTION const auto& l_workspace() const { return impl::unwrap(l_workspace_storage_); }
    // clang-format on

    KOKKOS_INLINE_FUNCTION void commit() {
      committed_ = true;
      impl::workspace_commit(dt_workspace());
      impl::workspace_commit(m_workspace());
      impl::workspace_commit(d_workspace());
      impl::workspace_commit(l_workspace());
    }

    KOKKOS_INLINE_FUNCTION void invalidate() {
      committed_ = false;
      impl::workspace_invalidate(dt_workspace());
      impl::workspace_invalidate(m_workspace());
      impl::workspace_invalidate(d_workspace());
      impl::workspace_invalidate(l_workspace());
    }

    KOKKOS_INLINE_FUNCTION bool is_committed() const {
      return committed_ && impl::workspace_is_committed(dt_workspace()) &&
             impl::workspace_is_committed(m_workspace()) && impl::workspace_is_committed(d_workspace()) &&
             impl::workspace_is_committed(l_workspace());
    }

   private:
    DxVectorStorage dx_storage_;
    MDxVectorStorage mdx_storage_;
    LMDxVectorStorage lmdx_storage_;
    DTWorkspaceStorage dt_workspace_storage_;
    MWorkspaceStorage m_workspace_storage_;
    DWorkspaceStorage d_workspace_storage_;
    LWorkspaceStorage l_workspace_storage_;
    bool committed_{false};
  };

  KOKKOS_INLINE_FUNCTION
  CongruentMixedReducedOp(backend_t, linear_op_dt_storage_t DT_storage, linear_op_m_storage_t M_storage,
                          linear_op_d_storage_t D_storage, linear_op_l_storage_t L_storage)
      : DT_storage_(std::move(DT_storage)),
        M_storage_(std::move(M_storage)),
        D_storage_(std::move(D_storage)),
        L_storage_(std::move(L_storage)) {
  }

  // Accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return impl::unwrap(DT_storage_); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return impl::unwrap(M_storage_); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return impl::unwrap(D_storage_); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return impl::unwrap(L_storage_); }
  KOKKOS_INLINE_FUNCTION
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
    auto dx_storage = impl::to_storage(Backend::make_range_vector(D()));
    auto mdx_storage = impl::to_storage(Backend::make_range_vector(M()));
    auto lmdx_storage = impl::to_storage(Backend::make_range_vector(L()));
    auto dt_workspace_storage = impl::make_workspace(DT());
    auto m_workspace_storage = impl::make_workspace(M());
    auto d_workspace_storage = impl::make_workspace(D());
    auto l_workspace_storage = impl::make_workspace(L());

    using dx_storage_t = decltype(dx_storage);
    using mdx_storage_t = decltype(mdx_storage);
    using lmdx_storage_t = decltype(lmdx_storage);
    using dt_workspace_storage_t = decltype(dt_workspace_storage);
    using m_workspace_storage_t = decltype(m_workspace_storage);
    using d_workspace_storage_t = decltype(d_workspace_storage);
    using l_workspace_storage_t = decltype(l_workspace_storage);

    return Workspace<dx_storage_t, mdx_storage_t, lmdx_storage_t, dt_workspace_storage_t, m_workspace_storage_t,
                     d_workspace_storage_t, l_workspace_storage_t>(
        std::move(dx_storage), std::move(mdx_storage), std::move(lmdx_storage), std::move(dt_workspace_storage),
        std::move(m_workspace_storage), std::move(d_workspace_storage), std::move(l_workspace_storage), committed);
  }

  template <class DxVector, class MDxVector, class LMDxVector, class DTWorkspace, class MWorkspace, class DWorkspace,
            class LWorkspace>
  auto make_workspace(DxVector&& dx, MDxVector&& mdx, LMDxVector&& lmdx, DTWorkspace&& dt_workspace,
                      MWorkspace&& m_workspace, DWorkspace&& d_workspace, LWorkspace&& l_workspace,
                      bool committed = false) const {
    auto dx_storage = impl::to_storage(std::forward<DxVector>(dx));
    auto mdx_storage = impl::to_storage(std::forward<MDxVector>(mdx));
    auto lmdx_storage = impl::to_storage(std::forward<LMDxVector>(lmdx));
    auto dt_workspace_storage = impl::to_storage(std::forward<DTWorkspace>(dt_workspace));
    auto m_workspace_storage = impl::to_storage(std::forward<MWorkspace>(m_workspace));
    auto d_workspace_storage = impl::to_storage(std::forward<DWorkspace>(d_workspace));
    auto l_workspace_storage = impl::to_storage(std::forward<LWorkspace>(l_workspace));

    using dx_storage_t = decltype(dx_storage);
    using mdx_storage_t = decltype(mdx_storage);
    using lmdx_storage_t = decltype(lmdx_storage);
    using dt_workspace_storage_t = decltype(dt_workspace_storage);
    using m_workspace_storage_t = decltype(m_workspace_storage);
    using d_workspace_storage_t = decltype(d_workspace_storage);
    using l_workspace_storage_t = decltype(l_workspace_storage);

    return Workspace<dx_storage_t, mdx_storage_t, lmdx_storage_t, dt_workspace_storage_t, m_workspace_storage_t,
                     d_workspace_storage_t, l_workspace_storage_t>(
        std::move(dx_storage), std::move(mdx_storage), std::move(lmdx_storage), std::move(dt_workspace_storage),
        std::move(m_workspace_storage), std::move(d_workspace_storage), std::move(l_workspace_storage), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    impl::workspace_invalidate(workspace);
    constexpr auto one = static_cast<impl::vector_scalar_t<XVector>>(1);
    Backend::apply(D(), x, workspace.dx(), workspace.d_workspace());
    Backend::apply(M(), workspace.dx(), workspace.mdx(), workspace.m_workspace());
    Backend::apply(L(), workspace.mdx(), workspace.lmdx(), workspace.l_workspace());
    Backend::axpby(one, workspace.dx(), -one, workspace.lmdx());
    Backend::apply(M(), workspace.lmdx(), workspace.mdx(), workspace.m_workspace());
    Backend::apply(DT(), workspace.mdx(), y, workspace.dt_workspace());
  }

  template <class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y) const {
    auto tmp_workspace = make_workspace();
    apply(x, y, tmp_workspace);
  }

 private:
  linear_op_dt_storage_t DT_storage_;
  linear_op_m_storage_t M_storage_;
  linear_op_d_storage_t D_storage_;
  linear_op_l_storage_t L_storage_;
};

/// \brief Backend for Kokkos single process execution
template <typename ExecSpace>
struct KokkosBackend {
 public:
  using exec_space = ExecSpace;

 public:
  template <class Vector>
  static auto make_vector_like(const Vector& q) {
    return Vector(
        q.extent(0));  // assumes Vector has a constructor that takes a size, and that the size is given by extent(0)
  }

  // make_domain/range_vector is host only, but may be called from KOKKOS_FUNCTION code being called on the host
  // This will cause warnings, but is otherwise perfectly valid, so we suppress the warnings for these functions
  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION
  static auto make_domain_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeDomainVectorMember<LinearOp>) {
      return op.make_domain_vector();
    } else if constexpr (impl::DenseMatView<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      using scalar_t = typename op_t::non_const_value_type;
      using mem_space = typename op_t::memory_space;
      using layout_t = typename Kokkos::View<scalar_t*, mem_space>::array_layout;
      using vector_t = Kokkos::View<scalar_t*, layout_t, mem_space>;
      return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "domain_vector"), domain_size(op));
    } else {
      static_assert(impl::dependent_false_v<LinearOp>,
                    "KokkosBackend::make_domain_vector requires DenseMatView or op.make_domain_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION
  static auto make_range_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeRangeVectorMember<LinearOp>) {
      return op.make_range_vector();
    } else if constexpr (impl::DenseMatView<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      using scalar_t = typename op_t::non_const_value_type;
      using mem_space = typename op_t::memory_space;
      using layout_t = typename Kokkos::View<scalar_t*, mem_space>::array_layout;
      using vector_t = Kokkos::View<scalar_t*, layout_t, mem_space>;
      return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "range_vector"), range_size(op));
    } else {
      static_assert(impl::dependent_false_v<LinearOp>,
                    "KokkosBackend::make_range_vector requires DenseMatView or op.make_range_vector().");
    }
  }

  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_POP

  template <class LinearOp>
  static auto make_workspace(const LinearOp& op) {
    return impl::make_workspace(op);
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static size_t size(Vector& x) {
    return x.extent(0);
  }

  template <class LinearOp>
    requires(impl::DenseMatView<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.extent(1);
  }
  //
  template <class LinearOp>
    requires(!impl::DenseMatView<LinearOp> && impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.domain_size();
  }
  //
  template <class LinearOp>
    requires(!impl::DenseMatView<LinearOp> && !impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp&) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::domain_size: op must be a rank-2 Kokkos::View or provide size_t domain_size().");
    return 0;
  }

  template <class LinearOp>
    requires(impl::DenseMatView<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.extent(0);
  }
  //
  template <class LinearOp>
    requires(!impl::DenseMatView<LinearOp> && impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.range_size();
  }
  //
  template <class LinearOp>
    requires(!impl::DenseMatView<LinearOp> && !impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp&) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::range_size: op must be a rank-2 Kokkos::View or provide size_t range_size().");
    return 0;
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(Vector& x, size_t i) {
    return x(i);
  }

  template <class DestVector, class SrcVector>
  static void deep_copy(DestVector& dest, const SrcVector& src) {
    Kokkos::deep_copy(dest, src);
  }

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // Path 1: If op is a dense 2D Kokkos::View, call BLAS gemv.
  // y = A*x
  template <class LinearOp, class XVector, class YVector>
    requires(impl::DenseMatView<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y) {
    using scalar_t = impl::vector_scalar_t<YVector>;
    MUNDY_THROW_ASSERT(A.extent(1) == x.extent(0), std::invalid_argument, "gemv: dimension mismatch A(:,1) vs x");
    MUNDY_THROW_ASSERT(A.extent(0) == y.extent(0), std::invalid_argument, "gemv: dimension mismatch A(0,:) vs y");
    KokkosBlas::gemv(exec_space{}, "N", scalar_t(1), A, x, scalar_t(0), y);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
    requires(impl::DenseMatView<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y, Workspace&) {
    apply(A, x, y);
  }

  // Path 1: If op is a dense 2D Kokkos::View, call BLAS gemv.
  // y = alpha * A * x + beta * y
  template <class Scalar, class LinearOp, class XVector, class YVector>
    requires(impl::DenseMatView<LinearOp>)
  static void apply(Scalar alpha, const LinearOp& A, const XVector& x, Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(A.extent(1) == x.extent(0), std::invalid_argument, "gemv: dimension mismatch A(:,1) vs x");
    MUNDY_THROW_ASSERT(A.extent(0) == y.extent(0), std::invalid_argument, "gemv: dimension mismatch A(0,:) vs y");
    KokkosBlas::gemv(exec_space{}, "N", alpha, A, x, beta, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  // Path 2: If op has member `apply(x,y)`
  template <class LinearOp, class XVector, class YVector>
    requires(!impl::DenseMatView<LinearOp> && impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    op.apply(x, y);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
    requires(!impl::DenseMatView<LinearOp> && impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    op.apply(x, y, workspace);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
    requires(!impl::DenseMatView<LinearOp> &&
             !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
             impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace&) {
    op.apply(x, y);
  }

  // Path 3: Otherwise, runtime error.
  template <typename LinearOp, class XVector, class YVector>
    requires(!impl::DenseMatView<LinearOp> && !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::apply: op must be a rank-2 Kokkos::View or provide void apply(x,y).");
  }

  template <typename LinearOp, class XVector, class YVector, class Workspace>
    requires(!impl::DenseMatView<LinearOp> &&
             !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
             !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp&, const XVector&, YVector&, Workspace&) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::apply: op must be a rank-2 Kokkos::View or provide void apply(x,y[,workspace]).");
  }

  template <class Scalar, class XVector, class YVector>
  static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    const bool alpha_is_zero = Kokkos::abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = Kokkos::abs(beta) < get_zero_tolerance<Scalar>();

    if (!alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "axpby", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { y(i) = alpha * x(i) + beta * y(i); });
    } else if (alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "axpby", Kokkos::RangePolicy<exec_space>(0, x.extent(0)), KOKKOS_LAMBDA(const int i) { y(i) *= beta; });
    } else if (!alpha_is_zero && beta_is_zero) {
      Kokkos::parallel_for(
          "axpby", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { y(i) = alpha * x(i); });
    } else {
      // Both alpha and beta are zero. Set y to zero.
      Kokkos::deep_copy(y, Scalar(0));
    }
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta, const YVector& y, ZVector& z,
                             const Wrapper& wrapper) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0) && x.extent(0) == z.extent(0), std::invalid_argument,
                       "x, y, and z must have the same size.");
    const bool alpha_is_zero = Kokkos::abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = Kokkos::abs(beta) < get_zero_tolerance<Scalar>();

    if (!alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(alpha * x(i) + beta * y(i)); });
    } else if (alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(beta * y(i)); });
    } else if (!alpha_is_zero && beta_is_zero) {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(alpha * x(i)); });
    } else {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(Scalar(0)); });
    }
  }

  template <class ReductionScalar, class XVector, class YVector>
  static ReductionScalar diff_dot(const XVector& x, const YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    ReductionScalar result = 0;
    Kokkos::parallel_reduce(
        "diff_dot", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i, ReductionScalar& sum) {
          ReductionScalar diff = x(i) - y(i);
          sum += diff * diff;
        },
        result);
    return result;
  }

  template <class ReductionScalar, class X1Vector, class X2Vector, class Y1Vector, class Y2Vector>
  static ReductionScalar diff_dot(const X1Vector& x1, const X2Vector& x2,  //
                                  const Y1Vector& y1, const Y2Vector& y2) {
    MUNDY_THROW_ASSERT(x1.extent(0) == x2.extent(0) && x1.extent(0) == y1.extent(0) && x1.extent(0) == y2.extent(0),
                       std::invalid_argument, "x1, x2, y1, and y2 must have the same size.");
    ReductionScalar result = 0;
    Kokkos::parallel_reduce(
        "diff_dot", Kokkos::RangePolicy<exec_space>(0, x1.extent(0)),
        KOKKOS_LAMBDA(const int i, ReductionScalar& sum) {
          ReductionScalar x_diff = x1(i) - x2(i);
          ReductionScalar y_diff = y1(i) - y2(i);
          sum += x_diff * y_diff;
        },
        result);
    return result;
  }

  template <typename ReductionScalar, typename Functor>
  static void reduce_max(size_t n, const Functor& func, ReductionScalar& result) {
    Kokkos::parallel_reduce(
        "reduce_max", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const int i, ReductionScalar& max_val) { func(i, max_val); },
        Kokkos::Max<ReductionScalar>(result));
  }

  template <typename ReductionScalar, typename Functor, class Vector>
  static void reduce_max(const Vector&, size_t n, const Functor& func, ReductionScalar& result) {
    reduce_max<ReductionScalar>(n, func, result);
  }
};  // KokkosBackend

/// \brief Backend for Mundy math within a kernel
struct MundyMathBackend {
  template <class Vector>
  KOKKOS_INLINE_FUNCTION
  static auto make_vector_like(const Vector& /*x*/) {
    return Vector();
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_domain_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeDomainVectorMember<LinearOp>) {
      return op.make_domain_vector();
    } else if constexpr (requires {
                           typename std::remove_reference_t<LinearOp>::scalar_t;
                           std::remove_reference_t<LinearOp>::num_cols;
                         }) {
      using op_t = std::remove_reference_t<LinearOp>;
      return Vector<typename op_t::scalar_t, op_t::num_cols>{};
    } else {
      static_assert(impl::dependent_false_v<LinearOp>,
                    "MundyMathBackend::make_domain_vector requires static matrix metadata or op.make_domain_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_range_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeRangeVectorMember<LinearOp>) {
      return op.make_range_vector();
    } else if constexpr (requires {
                           typename std::remove_reference_t<LinearOp>::scalar_t;
                           std::remove_reference_t<LinearOp>::num_rows;
                         }) {
      using op_t = std::remove_reference_t<LinearOp>;
      return Vector<typename op_t::scalar_t, op_t::num_rows>{};
    } else {
      static_assert(impl::dependent_false_v<LinearOp>,
                    "MundyMathBackend::make_range_vector requires static matrix metadata or op.make_range_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_workspace(const LinearOp& op) {
    return impl::make_workspace(op);
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static size_t size(const Vector& /*x*/) {
    return std::remove_reference_t<Vector>::size;
  }

  template <class LinearOp>
    requires(is_matrix_v<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& /*op*/) {
    return std::remove_reference_t<LinearOp>::num_cols;
  }
  //
  template <class LinearOp>
    requires(!is_matrix_v<LinearOp> && impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.domain_size();
  }
  //
  template <class LinearOp>
    requires(!is_matrix_v<LinearOp> && !impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& /*op*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::domain_size: op must be a rank-2 Kokkos::View or provide size_t domain_size().");
    return 0;
  }

  template <class LinearOp>
    requires(is_matrix_v<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& /*op*/) {
    return std::remove_reference_t<LinearOp>::num_rows;
  }
  //
  template <class LinearOp>
    requires(!is_matrix_v<LinearOp> && impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.range_size();
  }
  //
  template <class LinearOp>
    requires(!is_matrix_v<LinearOp> && !impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& /*op*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "MundyBackend::range_size: op must be a mundy::math::Matrix or provide size_t range_size().");
    return 0;
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(const Vector& x, size_t i) {
    return x[i];
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(Vector& x, size_t i) {
    return x[i];
  }

  template <class DestVector, class SrcVector>
  KOKKOS_INLINE_FUNCTION static void deep_copy(DestVector& dest, const SrcVector& src) {
    dest = src;
  }

  template <typename LinearOp, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    if constexpr (impl::HasApplyMember<LinearOp, XVector, YVector>) {
      op.apply(x, y);
    } else {
      y = op * x;
    }
  }

  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    if constexpr (impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>) {
      op.apply(x, y, workspace);
    } else if constexpr (impl::HasApplyMember<LinearOp, XVector, YVector>) {
      op.apply(x, y);
    } else {
      y = op * x;
    }
  }

  template <class Scalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    y = alpha * x + beta * y;
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  KOKKOS_INLINE_FUNCTION static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta,
                                                    const YVector& y, ZVector& z, const Wrapper& wrapper) {
    z = ::mundy::math::apply(wrapper, alpha * x + beta * y);
  }

  template <class ReductionScalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar diff_dot(const XVector& x, const YVector& y) {
    auto diff = x - y;
    return static_cast<ReductionScalar>(dot(diff, diff));
  }

  template <class ReductionScalar, class X1Vector, class X2Vector, class Y1Vector, class Y2Vector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar diff_dot(const X1Vector& x1, const X2Vector& x2,  //
                                                         const Y1Vector& y1, const Y2Vector& y2) {
    auto x_diff = x1 - x2;
    auto y_diff = y1 - y2;
    return static_cast<ReductionScalar>(dot(x_diff, y_diff));
  }

  template <class ReductionScalar, class Functor, class Vector>
  KOKKOS_INLINE_FUNCTION static void reduce_max(const Vector&, size_t n, const Functor& func, ReductionScalar& result) {
    constexpr size_t N = std::remove_reference_t<Vector>::size;
    MUNDY_THROW_ASSERT(n == N, std::invalid_argument, "reduce_max: n must match the size of the vector.");
    reduce_max_impl(std::make_index_sequence<N>{}, func, result);
  }

 private:
  template <size_t... Is, class Functor, class ReductionScalar>
  KOKKOS_INLINE_FUNCTION static void reduce_max_impl(std::index_sequence<Is...>, const Functor& func,
                                                     ReductionScalar& result) {
    ReductionScalar max_val = -Kokkos::Experimental::infinity_v<ReductionScalar>;
    ((func(Is, max_val)), ...);
    result = max_val;
  }
};  // MundyMathBackend
//@}

//! \name Problems + state
//@{

/// \brief Constrained quadratic programming problem (CQPP) formulation
///
/// This is for a constrained quadratic programming problem of the form:
///   x^* = argmin_{x in Omega} 0.5 x^T A x + q^T x
/// where A is a symmetric positive semi-definite matrix, q is a vector, and Omega is a convex space.
///
/// \tparam Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOpStorage, typename QVectorStorage, space::ValidConvexSpace ConvexSpace,
          typename Workspace = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpStorage>>>
class CQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = LinearOpStorage;
  using vector_storage_t = QVectorStorage;
  using linear_op_t = impl::unwrapped_storage_t<linear_op_storage_t>;
  using vector_t = impl::unwrapped_storage_t<vector_storage_t>;
  using space_t = ConvexSpace;
  using workspace_t = Workspace;
  using scalar_t = impl::vector_scalar_t<vector_t>;

  CQPPProblem(Backend, linear_op_storage_t A, vector_storage_t q, const space_t& space)
      : A_(std::move(A)), q_(std::move(q)), space_(space), workspace_(impl::make_workspace(impl::unwrap(A_))) {
  }

  CQPPProblem(Backend, linear_op_storage_t A, vector_storage_t q, const space_t& space, workspace_t workspace)
      : A_(std::move(A)), q_(std::move(q)), space_(space), workspace_(std::move(workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return impl::unwrap(q_); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  vector_storage_t q_;
  const space_t& space_;
  mutable workspace_t workspace_;
};

/// \brief Mixed constrained quadratic programming problem (MCQPP) formulation
///
/// This is for a mixed constrained convex quadratic programming problem such as:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// where M and K^{-1} are symmetric positive definite matrices, D and B are linear operators,
/// q and b are vectors, and Omega_x is a convex space.
///
/// This can be mapped onto a reduced CQPP in x alone via the Schur complement.
/// Define:
///   S^{-1} := B^T M B + K^{-1} (symmetric positive definite)
///   H := D^T M D - D^T M B S B^T M D
///   g := q - D^T M B S b
/// Then the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y^* = -S^{-1}(b + B^T M D x^*)
///
/// In more general terms, the mixed problem exists because of an affine operator mapping x -> f:
///   f = (I + L M) D x + f_b,
/// where f_b = f(x = 0) and L is an spsd linear operator
///
/// In the less-general problem above: L = B S B^T and f_b = B S b. In this form, the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := D^T M (D - L M D)
///   g := q - D^T M f_b
/// y^*, f^*, and u^* are all intermediates that may or may not be explicitly formed during the solution
/// process depending on the structure of L. We refer to these variables as your "workspace" variables. They are updated
/// by the linear operators during the solution process, and their final values are "committed" at the end of the solve
/// to indicate that they are the correct values corresponding to the optimal x^*.
///
/// We may also have the simpler problem where D = I, in which case the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := A (I - L A)
///   g := q - A f_b
///
/// Just like the CQPP, we will either accept the simple case of (A, q, L, f_b) or the more general case of (DT, M, D,
/// q, BT, S, B, b).
///
/// TODO(palmerb4): The workspaces here need to be of a storge type.
template <typename Backend, typename LinearOpAStorage, typename QVectorStorage, typename LinearOpLStorage,
          typename FVectorStorage, space::ValidConvexSpace ConvexSpace,
          typename WorkspaceA = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpAStorage>>,
          typename WorkspaceL = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpLStorage>>>
class MCQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_a_t = LinearOpAStorage;
  using linear_op_storage_l_t = LinearOpLStorage;
  using q_vector_storage_t = QVectorStorage;
  using f_vector_storage_t = FVectorStorage;
  using q_vector_t = impl::unwrapped_storage_t<q_vector_storage_t>;
  using f_vector_t = impl::unwrapped_storage_t<f_vector_storage_t>;
  using space_t = ConvexSpace;
  using a_workspace_t = WorkspaceA;
  using l_workspace_t = WorkspaceL;
  using scalar_t = impl::vector_scalar_t<q_vector_t>;

  MCQPPProblem(Backend, linear_op_storage_a_t A, q_vector_storage_t q, linear_op_storage_l_t L, f_vector_storage_t f_b,
               const space_t& space)
      : A_(std::move(A)),
        q_(std::move(q)),
        L_(std::move(L)),
        f_b_(std::move(f_b)),
        space_(space),
        a_workspace_(impl::make_workspace(impl::unwrap(A_))),
        l_workspace_(impl::make_workspace(impl::unwrap(L_))) {
  }

  MCQPPProblem(Backend, linear_op_storage_a_t A, q_vector_storage_t q, linear_op_storage_l_t L, f_vector_storage_t f_b,
               const space_t& space, a_workspace_t a_workspace, l_workspace_t l_workspace)
      : A_(std::move(A)),
        q_(std::move(q)),
        L_(std::move(L)),
        f_b_(std::move(f_b)),
        space_(space),
        a_workspace_(std::move(a_workspace)),
        l_workspace_(std::move(l_workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return impl::unwrap(q_); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return impl::unwrap(L_); }
  KOKKOS_INLINE_FUNCTION const auto& f_b() const { return impl::unwrap(f_b_); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  KOKKOS_INLINE_FUNCTION a_workspace_t& a_workspace() const { return a_workspace_; }
  KOKKOS_INLINE_FUNCTION l_workspace_t& l_workspace() const { return l_workspace_; }
  // clang-format on

 private:
  linear_op_storage_a_t A_;
  q_vector_storage_t q_;
  linear_op_storage_l_t L_;
  f_vector_storage_t f_b_;
  const space_t& space_;
  mutable a_workspace_t a_workspace_;
  mutable l_workspace_t l_workspace_;
};

template <typename Backend, typename LinearOpDTStorage, typename LinearOpMStorage, typename LinearOpDStorage,
          typename QVectorStorage, typename LinearOpLStorage, typename FVectorStorage,
          space::ValidConvexSpace ConvexSpace,
          typename WorkspaceDT = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpDTStorage>>,
          typename WorkspaceM = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpMStorage>>,
          typename WorkspaceD = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpDStorage>>,
          typename WorkspaceL = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpLStorage>>>
class CongruentMCQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_dt_t = LinearOpDTStorage;
  using linear_op_storage_m_t = LinearOpMStorage;
  using linear_op_storage_d_t = LinearOpDStorage;
  using q_vector_storage_t = QVectorStorage;
  using linear_op_storage_l_t = LinearOpLStorage;
  using f_vector_storage_t = FVectorStorage;
  using q_vector_t = impl::unwrapped_storage_t<q_vector_storage_t>;
  using space_t = ConvexSpace;
  using dt_workspace_t = WorkspaceDT;
  using m_workspace_t = WorkspaceM;
  using d_workspace_t = WorkspaceD;
  using l_workspace_t = WorkspaceL;
  using scalar_t = impl::vector_scalar_t<q_vector_t>;

  CongruentMCQPPProblem(Backend, linear_op_storage_dt_t DT, linear_op_storage_m_t M, linear_op_storage_d_t D,
                        q_vector_storage_t q, linear_op_storage_l_t L, f_vector_storage_t f_b, const space_t& space)
      : DT_(std::move(DT)),
        M_(std::move(M)),
        D_(std::move(D)),
        q_(std::move(q)),
        L_(std::move(L)),
        f_b_(std::move(f_b)),
        space_(space),
        dt_workspace_(impl::make_workspace(impl::unwrap(DT_))),
        m_workspace_(impl::make_workspace(impl::unwrap(M_))),
        d_workspace_(impl::make_workspace(impl::unwrap(D_))),
        l_workspace_(impl::make_workspace(impl::unwrap(L_))) {
  }

  CongruentMCQPPProblem(Backend, linear_op_storage_dt_t DT, linear_op_storage_m_t M, linear_op_storage_d_t D,
                        q_vector_storage_t q, linear_op_storage_l_t L, f_vector_storage_t f_b, const space_t& space,
                        dt_workspace_t dt_workspace, m_workspace_t m_workspace, d_workspace_t d_workspace,
                        l_workspace_t l_workspace)
      : DT_(std::move(DT)),
        M_(std::move(M)),
        D_(std::move(D)),
        q_(std::move(q)),
        L_(std::move(L)),
        f_b_(std::move(f_b)),
        space_(space),
        dt_workspace_(std::move(dt_workspace)),
        m_workspace_(std::move(m_workspace)),
        d_workspace_(std::move(d_workspace)),
        l_workspace_(std::move(l_workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return impl::unwrap(DT_); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return impl::unwrap(M_); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return impl::unwrap(D_); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return impl::unwrap(q_); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return impl::unwrap(L_); }
  KOKKOS_INLINE_FUNCTION const auto& f_b() const { return impl::unwrap(f_b_); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  KOKKOS_INLINE_FUNCTION dt_workspace_t& dt_workspace() const { return dt_workspace_; }
  KOKKOS_INLINE_FUNCTION m_workspace_t& m_workspace() const { return m_workspace_; }
  KOKKOS_INLINE_FUNCTION d_workspace_t& d_workspace() const { return d_workspace_; }
  KOKKOS_INLINE_FUNCTION l_workspace_t& l_workspace() const { return l_workspace_; }
  // clang-format on

 private:
  linear_op_storage_dt_t DT_;
  linear_op_storage_m_t M_;
  linear_op_storage_d_t D_;
  q_vector_storage_t q_;
  linear_op_storage_l_t L_;
  f_vector_storage_t f_b_;
  const space_t& space_;
  mutable dt_workspace_t dt_workspace_;
  mutable m_workspace_t m_workspace_;
  mutable d_workspace_t d_workspace_;
  mutable l_workspace_t l_workspace_;
};

/// \brief Linear complementarity problem (LCP) formulation
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
///
/// 	param Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOpStorage, typename QVectorStorage,
          typename Workspace = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpStorage>>>
class LCPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = LinearOpStorage;
  using q_vector_storage_t = QVectorStorage;
  using linear_op_t = impl::unwrapped_storage_t<linear_op_storage_t>;
  using q_vector_t = impl::unwrapped_storage_t<q_vector_storage_t>;
  using workspace_t = Workspace;
  using scalar_t = impl::vector_scalar_t<q_vector_t>;

  LCPProblem(Backend, linear_op_storage_t A, q_vector_storage_t q)
      : A_(std::move(A)), q_(std::move(q)), workspace_(impl::make_workspace(impl::unwrap(A_))) {
  }

  LCPProblem(Backend, linear_op_storage_t A, q_vector_storage_t q, workspace_t workspace)
      : A_(std::move(A)), q_(std::move(q)), workspace_(std::move(workspace)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return impl::unwrap(q_); }
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  q_vector_storage_t q_;
  mutable workspace_t workspace_;
};

template <class Backend, class LinearOp, class QVectorStorage>
KOKKOS_INLINE_FUNCTION auto to_cqpp(const LCPProblem<Backend, LinearOp, QVectorStorage>& P) {
  using scalar_t = typename LCPProblem<Backend, LinearOp, QVectorStorage>::scalar_t;
  static constexpr space::LowerBound Rn_plus{static_cast<scalar_t>(0)};
  return CQPPProblem(P.backend(), P.A(), P.q(), Rn_plus, P.workspace());
}

template <class Backend, class LinearOpAStorage, class QVectorStorage, class LinearOpLStorage, class FVectorStorage,
          class ConvexSpace, class AWorkspace, class LWorkspace>
KOKKOS_INLINE_FUNCTION auto to_cqpp(const MCQPPProblem<Backend, LinearOpAStorage, QVectorStorage, LinearOpLStorage,
                                                       FVectorStorage, ConvexSpace, AWorkspace, LWorkspace>& P) {
  // get the type of P no ref
  using scalar_t = std::remove_reference_t<decltype(P)>::scalar_t;

  auto backend = P.backend();
  using backend_t = decltype(backend);

  auto H = MixedReducedOp<backend_t, std::remove_cvref_t<decltype(P.A())>, std::remove_cvref_t<decltype(P.L())>>(
      backend_t{}, P.A(), P.L());

  auto g = backend_t::make_vector_like(P.q());
  auto a_workspace = P.a_workspace();
  auto l_workspace = P.l_workspace();

  impl::workspace_invalidate(a_workspace);
  backend_t::apply(P.A(), P.f_b(), g, a_workspace);                                 // g = A f_b
  backend_t::axpby(static_cast<scalar_t>(1), P.q(), static_cast<scalar_t>(-1), g);  // g = q - A f_b

  auto ax = backend_t::make_range_vector(P.A());
  auto lax = backend_t::make_range_vector(P.L());
  auto workspace = H.make_workspace(std::move(ax), std::move(lax), std::move(a_workspace), std::move(l_workspace));
  return CQPPProblem(backend_t{}, H, g, P.space(), workspace);
}

template <class Backend, class LinearOpDTStorage, class LinearOpMStorage, class LinearOpDStorage, class QVectorStorage,
          class LinearOpLStorage, class FVectorStorage, class ConvexSpace, class DTWorkspace, class MWorkspace,
          class DWorkspace, class LWorkspace>
KOKKOS_INLINE_FUNCTION auto to_cqpp(
    const CongruentMCQPPProblem<Backend, LinearOpDTStorage, LinearOpMStorage, LinearOpDStorage, QVectorStorage,
                                LinearOpLStorage, FVectorStorage, ConvexSpace, DTWorkspace, MWorkspace, DWorkspace,
                                LWorkspace>& P) {
  using scalar_t = std::remove_reference_t<decltype(P)>::scalar_t;

  auto backend = P.backend();
  using backend_t = decltype(backend);

  auto H =
      CongruentMixedReducedOp<backend_t, std::remove_cvref_t<decltype(P.DT())>, std::remove_cvref_t<decltype(P.M())>,
                              std::remove_cvref_t<decltype(P.D())>, std::remove_cvref_t<decltype(P.L())>>(
          backend_t{}, P.DT(), P.M(), P.D(), P.L());

  auto g = backend_t::make_vector_like(P.q());
  auto m_f_b = backend_t::make_range_vector(P.M());
  auto dt_workspace = P.dt_workspace();
  auto m_workspace = P.m_workspace();
  auto d_workspace = P.d_workspace();
  auto l_workspace = P.l_workspace();

  impl::workspace_invalidate(dt_workspace);
  impl::workspace_invalidate(m_workspace);
  impl::workspace_invalidate(d_workspace);
  impl::workspace_invalidate(l_workspace);

  backend_t::apply(P.M(), P.f_b(), m_f_b, m_workspace);
  backend_t::apply(P.DT(), m_f_b, g, dt_workspace);                                 // g = D^T M f_b
  backend_t::axpby(static_cast<scalar_t>(1), P.q(), static_cast<scalar_t>(-1), g);  // g = q - D^T M f_b

  auto dx = backend_t::make_range_vector(P.D());
  auto mdx = backend_t::make_range_vector(P.M());
  auto lmdx = backend_t::make_range_vector(P.L());
  auto workspace = H.make_workspace(std::move(dx), std::move(mdx), std::move(lmdx), std::move(dt_workspace),
                                    std::move(m_workspace), std::move(d_workspace), std::move(l_workspace));
  return CQPPProblem(backend_t{}, H, g, P.space(), workspace);
}
//@}

//! \name Policies
//@{

struct LinfNormProjectedGradientResidual {  // Lower bound only for non-negativity constraints
  template <typename Backend, typename XVector, typename GradVector,
            typename ReductionScalar = impl::vector_scalar_t<GradVector>>
  KOKKOS_INLINE_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                                    const XVector& x,                         //
                                                    const GradVector& grad,                   //
                                                    const space::LowerBound<ReductionScalar>& convex_space) const {
    MUNDY_THROW_REQUIRE(convex_space.bound() == static_cast<ReductionScalar>(0), std::invalid_argument,
                        "LinfNormProjectedGradientResidual is only implemented for non-negativity constraints.");

    using scalar_t = ReductionScalar;

    size_t n = Backend::size(x);
    scalar_t largest_abs_gradient;
    Backend::template reduce_max<scalar_t>(
        x, n,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          // perform the projection EQ 2.2 of Dai & Fletcher 2005
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i);

          scalar_t abs_projected_grad;
          if (x_i < get_zero_tolerance<scalar_t>()) {
            abs_projected_grad = Kokkos::max(scalar_t(0), grad_i);
          } else {
            abs_projected_grad = Kokkos::abs(grad_i);
          }

          if (abs_projected_grad > max_val) {
            max_val = abs_projected_grad;
          }
        },
        largest_abs_gradient);

    return largest_abs_gradient;
  }
};

struct LinfNormProjectedDiffResidual {
  template <typename Backend, typename XVector, typename GradVector, space::ValidConvexSpace ConvexSpace,
            typename ReductionScalar = impl::vector_scalar_t<GradVector>>
  KOKKOS_INLINE_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                                    const XVector& x,                         //
                                                    const GradVector& grad,                   //
                                                    const ConvexSpace& convex_space) const {
    using scalar_t = ReductionScalar;

    // This res comes from line 17 and Eq 25 of Mazhar 2015
    // res =  1.0 / (3 * num_unknowns * gd) * norm_inf(xk - proj(xk - gd * gk))
    size_t num_unknowns = Backend::size(x);
    constexpr scalar_t small_step_size = static_cast<scalar_t>(1e-6);
    scalar_t largest_abs_diff;
    Backend::template reduce_max<scalar_t>(
        x, num_unknowns,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i);
          scalar_t x_i_proj = convex_space.project(x_i - small_step_size * grad_i);
          scalar_t abs_diff = Kokkos::abs(x_i - x_i_proj);
          if (abs_diff > max_val) {
            max_val = abs_diff;
          }
        },
        largest_abs_diff);

    return largest_abs_diff / small_step_size;
  }
};

struct BBStepStrategy {
  template <typename Backend, typename XOldVector, typename GradOldVector, typename XVector, typename GradVector,
            typename ReductionScalar = impl::vector_scalar_t<XVector>>
  KOKKOS_INLINE_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                                    const XOldVector& x_old,
                                                    const GradOldVector& grad_old,  //
                                                    const XVector& x, const GradVector& grad) const {
    using scalar_t = ReductionScalar;

    scalar_t num = Backend::template diff_dot<scalar_t>(x, x_old);  // (x - x_old) dot (x - x_old)
    scalar_t denom =
        Backend::template diff_dot<scalar_t>(x, x_old, grad, grad_old);  // (x - x_old) dot (grad - grad_old)

    // Avoid division by zero
    constexpr scalar_t eps = get_zero_tolerance<scalar_t>() * static_cast<scalar_t>(10);
    denom += eps * (Kokkos::abs(denom) < eps);

    return num / denom;
  }
};  // BBStepStrategy
//@}

template <typename Scalar>
struct PGDConfig {
  using scalar_t = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};
};

template <class Scalar>
struct SolveResult {
  using scalar_t = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};
  bool converged{false};
};

/// \brief Write SolveResult to an ostream
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const SolveResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged;
  return os;
}

template <class Scalar, class XVectorStorage, class GradVectorStorage, class XTmpVectorStorage,
          class GradTmpVectorStorage>
class PGDState {
 public:
  using scalar_t = Scalar;
  using x_vector_storage_t = XVectorStorage;
  using grad_vector_storage_t = GradVectorStorage;
  using x_tmp_vector_storage_t = XTmpVectorStorage;
  using grad_tmp_vector_storage_t = GradTmpVectorStorage;

  KOKKOS_INLINE_FUNCTION
  PGDState(x_vector_storage_t x, grad_vector_storage_t g, x_tmp_vector_storage_t x_tmp, grad_tmp_vector_storage_t g_tmp)
      : x_(std::move(x)), g_(std::move(g)), x_tmp_(std::move(x_tmp)), g_tmp_(std::move(g_tmp)) {
  }

  // Accessors (const/non-const as needed)
  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& x()      { return impl::unwrap(x_); }
  KOKKOS_INLINE_FUNCTION const auto& x() const{ return impl::unwrap(x_); }
  KOKKOS_INLINE_FUNCTION       auto& grad()      { return impl::unwrap(g_); }
  KOKKOS_INLINE_FUNCTION const auto& grad() const{ return impl::unwrap(g_); }
  KOKKOS_INLINE_FUNCTION       auto& x_tmp()      { return impl::unwrap(x_tmp_); }
  KOKKOS_INLINE_FUNCTION const auto& x_tmp() const{ return impl::unwrap(x_tmp_); }
  KOKKOS_INLINE_FUNCTION       auto& grad_tmp()      { return impl::unwrap(g_tmp_); }
  KOKKOS_INLINE_FUNCTION const auto& grad_tmp() const{ return impl::unwrap(g_tmp_); }
  // clang-format on

  // Iteration locals with accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION unsigned& iter()         { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&     converged()    { return converged_; }
  KOKKOS_INLINE_FUNCTION scalar_t& residual()     { return residual_; }
  KOKKOS_INLINE_FUNCTION scalar_t& step_size()        { return step_size_; }

  KOKKOS_INLINE_FUNCTION unsigned  iter()    const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool      converged() const { return converged_; }
  KOKKOS_INLINE_FUNCTION scalar_t  residual() const  { return residual_; }
  KOKKOS_INLINE_FUNCTION scalar_t  step_size()    const  { return step_size_; }
  // clang-format on

 private:
  x_vector_storage_t x_;
  grad_vector_storage_t g_;
  x_tmp_vector_storage_t x_tmp_;
  grad_tmp_vector_storage_t g_tmp_;
  unsigned iter_{0};
  bool converged_{false};
  scalar_t residual_{0};
  scalar_t step_size_{1};
};

template <class StepPolicy, class ResidualPolicy, class Config>
class PGDStrategy {
 public:
  using scalar_t = typename Config::scalar_t;

  using step_policy_t = StepPolicy;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = SolveResult<scalar_t>;

  KOKKOS_INLINE_FUNCTION
  PGDStrategy(step_policy_t step, residual_policy_t resid, config_t cfg = {}) : step_(step), resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  KOKKOS_INLINE_FUNCTION void initialize(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr scalar_t one = static_cast<scalar_t>(1);
    auto& workspace = prob.workspace();

    // x_tmp = x
    backend_t::deep_copy(state.x_tmp(), state.x());

    // grad_tmp = A x_tmp + q
    impl::workspace_invalidate(workspace);
    backend_t::apply(prob.A(), state.x_tmp(), state.grad_tmp(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad_tmp());

    // Dai-Fletcher Sec. 5 initial step
    state.residual() = resid_(backend, state.x_tmp(), state.grad_tmp(), prob.space());

    // Initialize iteration state (allow for early exit)
    state.iter() = 0;
    state.converged() = (state.residual() <= static_cast<scalar_t>(cfg_.tol));
    if (state.converged()) {
      state.step_size() = one;
      // If already converged, copy grad_tmp to grad
      backend_t::deep_copy(state.grad(), state.grad_tmp());
      impl::workspace_commit(workspace);
    } else {
      state.step_size() = one / state.residual();
    }
  }

  template <class Problem, class State>
  KOKKOS_INLINE_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr scalar_t one = static_cast<scalar_t>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    // x = Proj(x_tmp - step_size * grad_tmp)
    backend_t::wrapped_axpbyz(one, state.x_tmp(), -state.step_size(), state.grad_tmp(), state.x(), prob.space());

    // grad = A x + q
    impl::workspace_invalidate(workspace);
    backend_t::apply(prob.A(), state.x(), state.grad(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad());

    // residual & test
    state.residual() = resid_(backend, state.x(), state.grad(), prob.space());
    if (state.residual() <= static_cast<scalar_t>(cfg_.tol)) {
      state.converged() = true;
      impl::workspace_commit(workspace);
      return true;
    }

    // update step size and roll x_tmp/grad_tmp forward
    state.step_size() = step_(backend, state.x_tmp(), state.grad_tmp(), state.x(), state.grad());
    backend_t::deep_copy(state.x_tmp(), state.x());
    backend_t::deep_copy(state.grad_tmp(), state.grad());
    ++state.iter();
    return false;
  }

  template <class State>
  KOKKOS_INLINE_FUNCTION bool done(const State& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_INLINE_FUNCTION result_t result(const State& state) const {
    return {state.iter(), state.residual(), state.converged()};
  }

 private:
  step_policy_t step_;
  residual_policy_t resid_;
  config_t cfg_;
};

template <class Strategy, class Problem, class State>
concept CQPPSolverStrategy = requires(const Strategy& s, const Problem& prob, State& state) {
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.iterate(prob, state) } -> std::same_as<bool>;
  { s.done(state) } -> std::same_as<bool>;
  s.result(state);
};

//! \name Deduction guides
//@{

/// \brief Deduction guide for CQPPProblem
template <typename Backend, typename LinearOp, typename QVector, space::ValidConvexSpace ConvexSpace>
CQPPProblem(Backend, const LinearOp&, const QVector&, const ConvexSpace&)
    -> CQPPProblem<Backend, LinearOp, QVector, ConvexSpace>;

template <typename Backend, typename LinearOp, typename QVector, space::ValidConvexSpace ConvexSpace,
          typename Workspace>
CQPPProblem(Backend, const LinearOp&, const QVector&, const ConvexSpace&, const Workspace&)
    -> CQPPProblem<Backend, LinearOp, QVector, ConvexSpace, Workspace>;

/// \brief Deduction guide for LCPProblem
template <typename Backend, typename LinearOp, typename QVector>
LCPProblem(Backend, const LinearOp&, const QVector&) -> LCPProblem<Backend, LinearOp, QVector>;

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
LCPProblem(Backend, const LinearOp&, const QVector&, const Workspace&)
    -> LCPProblem<Backend, LinearOp, QVector, Workspace>;

/// \brief Deduction guide for MCQPPProblem
template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          space::ValidConvexSpace ConvexSpace>
MCQPPProblem(Backend, const LinearOpA&, const QVector&, const LinearOpL&, const FVector&, const ConvexSpace&)
    -> MCQPPProblem<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace>;

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          space::ValidConvexSpace ConvexSpace, typename AWorkspace, typename LWorkspace>
MCQPPProblem(Backend, const LinearOpA&, const QVector&, const LinearOpL&, const FVector&, const ConvexSpace&,
             const AWorkspace&, const LWorkspace&)
    -> MCQPPProblem<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace, AWorkspace, LWorkspace>;

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, space::ValidConvexSpace ConvexSpace>
CongruentMCQPPProblem(Backend, const LinearOpDT&, const LinearOpM&, const LinearOpD&, const QVector&, const LinearOpL&,
                      const FVector&, const ConvexSpace&)
    -> CongruentMCQPPProblem<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace>;

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, space::ValidConvexSpace ConvexSpace, typename DTWorkspace,
          typename MWorkspace, typename DWorkspace, typename LWorkspace>
CongruentMCQPPProblem(Backend, const LinearOpDT&, const LinearOpM&, const LinearOpD&, const QVector&, const LinearOpL&,
                      const FVector&, const ConvexSpace&, const DTWorkspace&, const MWorkspace&, const DWorkspace&,
                      const LWorkspace&)
    -> CongruentMCQPPProblem<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace,
                             DTWorkspace, MWorkspace, DWorkspace, LWorkspace>;

/// \brief Deduction guide for PGDConfig
template <typename Scalar>
PGDConfig(unsigned, Scalar) -> PGDConfig<Scalar>;

/// \brief Deduction guide for PGDState
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
PGDState(XVector&, GradVector&, XTmpVector&, GradTmpVector&)
    -> PGDState<impl::vector_scalar_t<XVector>, XVector, GradVector, XTmpVector, GradTmpVector>;

/// \brief Deduction guide for PGDStrategy
template <class StepPolicy, class ResidualPolicy, class Config>
PGDStrategy(StepPolicy, ResidualPolicy, Config = {}) -> PGDStrategy<StepPolicy, ResidualPolicy, Config>;
//@}

}  // namespace convex

template <typename Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
KOKKOS_INLINE_FUNCTION auto make_quadratic_form(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  return convex::QuadraticFormOp(Backend{}, std::move(DT_storage), std::move(M_storage), std::move(D_storage));
}

template <typename Backend, typename LinearOp, typename QVector, convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOp&& A, QVector&& q, ConvexSpace&& space) {
  auto A_storage = convex::impl::to_storage(std::forward<LinearOp>(A));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  return convex::CQPPProblem(Backend{}, std::move(A_storage), std::move(q_storage), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, ConvexSpace&& space) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto A = make_quadratic_form<Backend>(std::move(DT_storage), std::move(M_storage), std::move(D_storage));
  return convex::CQPPProblem(Backend{}, std::move(A), std::move(q_storage), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          convex::space::ValidConvexSpace ConvexSpace, typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, FVector&& f,
                                      UVector&& u, ConvexSpace&& space) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto f_storage = convex::impl::to_storage(std::forward<FVector>(f));
  auto u_storage = convex::impl::to_storage(std::forward<UVector>(u));

  auto A = make_quadratic_form<Backend>(std::move(DT_storage), std::move(M_storage), std::move(D_storage));
  auto workspace = A.make_workspace(std::move(f_storage), std::move(u_storage));
  return convex::CQPPProblem(Backend{}, std::move(A), std::move(q_storage), std::forward<ConvexSpace>(space),
                             std::move(workspace));
}

template <typename Backend, typename LinearOp, typename QVector, convex::space::ValidConvexSpace ConvexSpace,
          typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOp&& A, QVector&& q, ConvexSpace&& space, Workspace&& workspace) {
  auto A_storage = convex::impl::to_storage(std::forward<LinearOp>(A));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  return convex::CQPPProblem(Backend{}, std::move(A_storage), std::move(q_storage), std::forward<ConvexSpace>(space),
                             std::forward<Workspace>(workspace));
}

template <typename Backend, typename LinearOp, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOp&& A, QVector&& q) {
  auto A_storage = convex::impl::to_storage(std::forward<LinearOp>(A));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  return convex::LCPProblem(Backend{}, std::move(A_storage), std::move(q_storage));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto A = make_quadratic_form<Backend>(std::move(DT_storage), std::move(M_storage), std::move(D_storage));
  return convex::LCPProblem(Backend{}, std::move(A), std::move(q_storage));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, FVector&& f,
                                     UVector&& u) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto f_storage = convex::impl::to_storage(std::forward<FVector>(f));
  auto u_storage = convex::impl::to_storage(std::forward<UVector>(u));

  auto A = make_quadratic_form<Backend>(std::move(DT_storage), std::move(M_storage), std::move(D_storage));
  auto workspace = A.make_workspace(std::move(f_storage), std::move(u_storage));
  return convex::LCPProblem(Backend{}, std::move(A), std::move(q_storage), std::move(workspace));
}

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOp&& A, QVector&& q, Workspace&& workspace) {
  auto A_storage = convex::impl::to_storage(std::forward<LinearOp>(A));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  return convex::LCPProblem(Backend{}, std::move(A_storage), std::move(q_storage), std::forward<Workspace>(workspace));
}

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b,
                                            ConvexSpace&& space) {
  auto A_storage = convex::impl::to_storage(std::forward<LinearOpA>(A));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto L_storage = convex::impl::to_storage(std::forward<LinearOpL>(L));
  auto f_b_storage = convex::impl::to_storage(std::forward<FVector>(f_b));

  return convex::MCQPPProblem(Backend{}, std::move(A_storage), std::move(q_storage), std::move(L_storage),
                              std::move(f_b_storage), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          convex::space::ValidConvexSpace ConvexSpace, typename AWorkspace, typename LWorkspace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpA&& A, QVector&& q,    //
                                            LinearOpL&& L, FVector&& f_b,  //
                                            ConvexSpace&& space,           //
                                            AWorkspace&& a_workspace,      //
                                            LWorkspace&& l_workspace) {
  auto A_storage = convex::impl::to_storage(std::forward<LinearOpA>(A));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto L_storage = convex::impl::to_storage(std::forward<LinearOpL>(L));
  auto f_b_storage = convex::impl::to_storage(std::forward<FVector>(f_b));
  auto a_workspace_storage = convex::impl::to_storage(std::forward<AWorkspace>(a_workspace));
  auto l_workspace_storage = convex::impl::to_storage(std::forward<LWorkspace>(l_workspace));

  return convex::MCQPPProblem(Backend{}, std::move(A_storage), std::move(q_storage), std::move(L_storage),
                              std::move(f_b_storage), std::forward<ConvexSpace>(space), std::move(a_workspace_storage),
                              std::move(l_workspace_storage));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L,
                                            FVector&& f_b, ConvexSpace&& space) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto L_storage = convex::impl::to_storage(std::forward<LinearOpL>(L));
  auto f_b_storage = convex::impl::to_storage(std::forward<FVector>(f_b));

  return convex::CongruentMCQPPProblem(Backend{}, std::move(DT_storage), std::move(M_storage), std::move(D_storage),
                                       std::move(q_storage), std::move(L_storage), std::move(f_b_storage),
                                       std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpB, typename LinearOpS, typename LinearOpBT, typename BVector,
          convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpB&& B,
                                            LinearOpS&& S, LinearOpBT&& BT, BVector&& b, ConvexSpace&& space) {
  auto DT_storage = convex::impl::to_storage(std::forward<LinearOpDT>(DT));
  auto M_storage = convex::impl::to_storage(std::forward<LinearOpM>(M));
  auto D_storage = convex::impl::to_storage(std::forward<LinearOpD>(D));
  auto q_storage = convex::impl::to_storage(std::forward<QVector>(q));
  auto B_storage = convex::impl::to_storage(std::forward<LinearOpB>(B));
  auto S_storage = convex::impl::to_storage(std::forward<LinearOpS>(S));
  auto BT_storage = convex::impl::to_storage(std::forward<LinearOpBT>(BT));
  auto b_storage = convex::impl::to_storage(std::forward<BVector>(b));

  using backend_t = Backend;
  auto tmp = backend_t::make_range_vector(convex::impl::unwrap(S_storage));
  auto f_b = backend_t::make_range_vector(convex::impl::unwrap(B_storage));
  backend_t::apply(convex::impl::unwrap(S_storage), convex::impl::unwrap(b_storage), tmp);
  backend_t::apply(convex::impl::unwrap(B_storage), tmp, f_b);

  auto L = make_quadratic_form<backend_t>(std::move(B_storage), std::move(S_storage), std::move(BT_storage));

  return convex::CongruentMCQPPProblem(backend_t{}, std::move(DT_storage), std::move(M_storage), std::move(D_storage),
                                       std::move(q_storage), std::move(L), std::move(f_b),
                                       std::forward<ConvexSpace>(space));
}

template <class StepPolicy, class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(StepPolicy&& step_policy,          //
                                                       ResidualPolicy&& residual_policy,  //
                                                       const convex::PGDConfig<Scalar>& cfg = {}) {
  return convex::PGDStrategy(std::forward<StepPolicy>(step_policy), std::forward<ResidualPolicy>(residual_policy), cfg);
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const convex::PGDConfig<Scalar>& cfg = {}) {
  using DefaultStepPolicy = convex::BBStepStrategy;
  using DefaultResidualPolicy = convex::LinfNormProjectedDiffResidual;
  return convex::PGDStrategy(DefaultStepPolicy{}, DefaultResidualPolicy{}, cfg);
}
//
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
KOKKOS_INLINE_FUNCTION auto make_pgd_state(XVector&& x,         //
                                           GradVector&& grad,   //
                                           XTmpVector&& x_tmp,  //
                                           GradTmpVector&& grad_tmp) {
  auto x_storage = convex::impl::to_storage(std::forward<XVector>(x));
  auto grad_storage = convex::impl::to_storage(std::forward<GradVector>(grad));
  auto x_tmp_storage = convex::impl::to_storage(std::forward<XTmpVector>(x_tmp));
  auto grad_tmp_storage = convex::impl::to_storage(std::forward<GradTmpVector>(grad_tmp));

  using scalar_t = convex::impl::vector_scalar_t<XVector>;
  using x_storage_t = decltype(x_storage);
  using grad_storage_t = decltype(grad_storage);
  using x_tmp_storage_t = decltype(x_tmp_storage);
  using grad_tmp_storage_t = decltype(grad_tmp_storage);

  return convex::PGDState<scalar_t, x_storage_t, grad_storage_t, x_tmp_storage_t, grad_tmp_storage_t>(
      std::move(x_storage), std::move(grad_storage), std::move(x_tmp_storage), std::move(grad_tmp_storage));
}

/// \brief Solve a constrained quadratic programming problem (CQPP)
///
/// \param prob The constrained quadratic programming problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
  requires convex::CQPPSolverStrategy<Strategy, Problem, State>
KOKKOS_INLINE_FUNCTION auto solve_cqpp(const Problem& prob, const Strategy& strat, State& state) {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  auto result = strat.result(state);
  MUNDY_THROW_ASSERT(!result.converged || convex::impl::workspace_is_committed(prob.workspace()), std::logic_error,
                     "solve_cqpp: converged solution requires committed operator workspace.");
  return result;
}

/// \brief Solve a mixed constrained convex quadratic programming problem (MCQPP)
///
/// This is for a mixed constrained convex quadratic programming problem such as:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// where M and K^{-1} are symmetric positive definite matrices, D and B are linear operators,
/// q and b are vectors, and Omega_x is a convex space.
///
/// This can be mapped onto a reduced CQPP in x alone via the Schur complement.
/// Define:
///   S^{-1} := B^T M B + K^{-1} (symmetric positive definite)
///   H := D^T M D - D^T M B S B^T M D
///   g := q - D^T M B S b
/// Then the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y^* = -S^{-1}(b + B^T M D x^*)
///
/// In more general terms, the mixed problem exists because of an affine operator mapping x -> f:
///   f = (I + L M) D x + f_b,
/// where f_b = f(x = 0) and L is an spsd linear operator
///
/// In the less-general problem above: L = B S B^T and f_b = B S b. In this form, the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := D^T M (D - L M D)
///   g := q - D^T M f_b
/// y^*, f^*, and u^* are all intermediates that may or may not be explicitly formed during the solution
/// process depending on the structure of L. We refer to these variables as your "workspace" variables. They are updated
/// by the linear operators during the solution process, and their final values are "committed" at the end of the solve
/// to indicate that they are the correct values corresponding to the optimal x^*.
///
/// We may also have the simpler problem where D = I, in which case the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := A (I - L A)
///   g := q - A f_b
///
/// Just like the CQPP, we will either accept the simple case of (A, q, L, f_b) or the more general case of (DT, M, D,
/// q, BT, S, B, b).
///
/// \param prob The mixed constrained convex quadratic programming problem to solve.
/// \param strat The solution strategy to use (for the reduced CQPP in x and the linear solve in y).
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy>
  requires requires(const Problem& p) {
    { to_cqpp(p) };
  }
KOKKOS_INLINE_FUNCTION auto solve_mixed_cqpp(const Problem& prob, const Strategy& strat,
                                             typename Strategy::state_t& state) -> typename Strategy::result_t {
  // Convert MCQPP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

/// \brief Solve a linear complementarity problem (LCP) using a constrained quadratic programming solver
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
///
/// Example 1:
/// \code{.cpp}
///    // Problem setup
///    Matrix3d A = {/*...*/};
///    Vector3d q = {/*...*/};
///    Vector3d x{/* initial_guess */}, grad{}, x_tmp{}, grad_tmp{};
///
///    // Build the problem (no template args at callsite)
///    const auto lcp = make_mundy_math_lcp(A, q);
///
///    // Reuse the backend token from the problem
///    const auto backend = lcp.backend();
///
///    // Strategy + state
///    PGDConfig cfg{1000, 1e-6};
///
///    auto pgd = make_pgd_solution_strategy(cfg);  // Use default step/residual strategies
///    // auto pgd = make_pgd_solution_strategy(    //
///        MyStepStrat{}, MyResidualStrat{}, cfg);  // Custom step/residual strategies
///    auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);
///
///    // Solve (can reuse "lcp" and "pgd" across many states)
///    auto result = solve_lcp(lcp, pgd, pgd_state);
/// \endcode
///
/// Example 2: Congruent LCP
/// \code{.cpp}
///    // Problem setup
///    Matrix3d D = {/*...*/};
///    Matrix3d M = {/*...*/};
///    Vector3d q = {/*...*/};
///    Vector3d x{/* initial_guess */}, grad{}, x_tmp{}, grad_tmp{};
///
///    // Build quadratic-form operator and problem
///    using backend_t = convex::MundyMathBackend<double, 3>;
///    auto A = make_quadratic_form<backend_t>(D.view_transpose(), M, D);
///    const auto lcp = make_mundy_math_lcp(A, q);
///
///   // Reuse the backend token from the problem
///    const auto backend = lcp.backend();
///
///    // Strategy + state
///    PGDConfig cfg{1000, 1e-6};
///
///    auto pgd = make_pgd_solution_strategy(backend, cfg);
///    auto pgd_state = make_pgd_state(backend, x, grad, x_tmp, grad_tmp);
///
///    // Solve (can reuse "lcp" and "pgd" across many states)
///    auto result = solve_lcp(lcp, pgd, pgd_state);
///
///    // Workspace (f/u) now lives on the linear operator in the problem
///    const auto& ws = lcp.workspace();
///    MUNDY_THROW_ASSERT(ws.is_committed(), std::logic_error, "workspace must be committed after convergence");
/// \endcode
///
/// \param prob The linear complementarity problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
  requires requires(const Problem& p) {
    { to_cqpp(p) };
  }
KOKKOS_INLINE_FUNCTION auto solve_lcp(const Problem& prob, const Strategy& strat, State& state) {
  // Convert LCP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_HPP_