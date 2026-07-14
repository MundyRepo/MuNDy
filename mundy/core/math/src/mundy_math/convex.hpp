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

// Mundy
#include <mundy_math/Matrix.hpp>     // for mundy::is_matrix_v
#include <mundy_math/Tolerance.hpp>  // for mundy::get_zero_tolerance<T>
#include <mundy_math/Vector.hpp>     // for mundy::Vector
#include <mundy_math/cmath.hpp>
#include <mundy_utils/reference_wrapper.hpp>
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>            // for mundy::storage, mundy::store
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>
#include <mundy_utils/tuple.hpp>  // for mundy::tuple, mundy::for_each, mundy::all_of

namespace mundy {

namespace convex {

namespace space {

// These are 1d convex spaces, which will be applied to each element of a vector assuming a separable convex space

/// \brief Proj(x) = x for all x in R
template <typename Scalar>
struct Unconstrained {
  using value_type = Scalar;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return x;
  }
};

/// \brief Proj(x) = max(x, lower_bound) for all x in R
template <typename Scalar>
struct LowerBound {
  using value_type = Scalar;

  value_type lower_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return max(x, lower_bound);
  }
};

/// \brief Proj(x) = min(x, upper_bound) for all x in R
template <typename Scalar>
struct UpperBound {
  using value_type = Scalar;

  value_type upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return min(x, upper_bound);
  }
};

/// \brief Proj(x) = min(max(x, lower_bound), upper_bound) for all x in R
template <typename Scalar>
struct Bounded {
  using value_type = Scalar;

  value_type lower_bound;
  value_type upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return min(max(x, lower_bound), upper_bound);
  }
};

/// \brief Concept for a valid space
template <class Space>
concept ValidConvexSpace = requires {
  typename std::remove_cvref_t<Space>::value_type;  // make the nested type check explicit
} && requires(const std::remove_cvref_t<Space>& s, typename std::remove_cvref_t<Space>::value_type x) {
  { s.project(x) } -> std::same_as<typename std::remove_cvref_t<Space>::value_type>;
  { s(x) } -> std::same_as<typename std::remove_cvref_t<Space>::value_type>;
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
using vector_value_type =
    std::remove_cvref_t<decltype(std::declval<const std::remove_reference_t<Vector>&>()(size_t{}))>;

/// \brief Aggregates commit()/invalidate()/is_committed() over a pack of child workspace storages.
///
/// A composite operator's Workspace derives from this for its nested per-sub-operator workspaces, so it only needs
/// to declare its own scratch-vector storage; the commit/invalidate/is_committed contract is inherited.
template <class... ChildWorkspace>
class CommitGroup {
 public:
  // For an empty ChildWorkspace... pack this is a zero-argument constructor (committed defaults to false).
  KOKKOS_INLINE_FUNCTION explicit CommitGroup(ChildWorkspace&&... children, bool committed = false)
      : children_(::mundy::store(std::forward<ChildWorkspace>(children))...), committed_(committed) {
  }

  KOKKOS_FUNCTION void commit() {
    committed_ = true;
    ::mundy::for_each(children_, [](auto& c) { workspace_commit(c.get()); });
  }

  KOKKOS_FUNCTION void invalidate() {
    committed_ = false;
    ::mundy::for_each(children_, [](auto& c) { workspace_invalidate(c.get()); });
  }

  KOKKOS_FUNCTION bool is_committed() const {
    return committed_ && ::mundy::all_of(children_, [](const auto& c) { return workspace_is_committed(c.get()); });
  }

 protected:
  template <size_t I>
  KOKKOS_INLINE_FUNCTION auto& child() {
    return children_.template get<I>().get();
  }
  template <size_t I>
  KOKKOS_INLINE_FUNCTION const auto& child() const {
    return children_.template get<I>().get();
  }

 private:
  ::mundy::tuple<::mundy::storage<ChildWorkspace>...> children_;
  bool committed_{false};
};

}  // namespace impl

/// \brief Declares a named accessor pair for a Workspace's own scratch-vector storage member.
#define MUNDY_WORKSPACE_FIELD(name, storage_member) \
  KOKKOS_INLINE_FUNCTION auto& name() {             \
    return storage_member.get();                    \
  }                                                 \
  KOKKOS_INLINE_FUNCTION const auto& name() const { \
    return storage_member.get();                    \
  }

/// \brief Declares a named accessor pair for a Workspace's I'th CommitGroup child (a nested sub-operator workspace).
#define MUNDY_WORKSPACE_CHILD(name, index)          \
  KOKKOS_INLINE_FUNCTION auto& name() {             \
    return this->template child<index>();           \
  }                                                 \
  KOKKOS_INLINE_FUNCTION const auto& name() const { \
    return this->template child<index>();           \
  }

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
    MUNDY_WORKSPACE_FIELD(f, f_storage_)
    MUNDY_WORKSPACE_FIELD(u, u_storage_)
    // commit()/invalidate()/is_committed() come from CommitGroup.

   private:
    ::mundy::storage<FVector> f_storage_;
    ::mundy::storage<UVector> u_storage_;
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

/// \brief The operator that perform Op x for Op := A (I - L A)
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
    MUNDY_WORKSPACE_FIELD(ax, ax_storage_)
    MUNDY_WORKSPACE_FIELD(lax, lax_storage_)
    MUNDY_WORKSPACE_CHILD(a_workspace, 0)
    MUNDY_WORKSPACE_CHILD(l_workspace, 1)
    // commit()/invalidate()/is_committed() come from CommitGroup.

   private:
    ::mundy::storage<AxVector> ax_storage_;
    ::mundy::storage<LAxVector> lax_storage_;
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

/// \brief The operator that performs Op x for Op := D^T M (D - L M D)
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
    MUNDY_WORKSPACE_FIELD(dx, dx_storage_)
    MUNDY_WORKSPACE_FIELD(mdx, mdx_storage_)
    MUNDY_WORKSPACE_FIELD(lmdx, lmdx_storage_)
    MUNDY_WORKSPACE_CHILD(dt_workspace, 0)
    MUNDY_WORKSPACE_CHILD(m_workspace, 1)
    MUNDY_WORKSPACE_CHILD(d_workspace, 2)
    MUNDY_WORKSPACE_CHILD(l_workspace, 3)
    // commit()/invalidate()/is_committed() come from CommitGroup.

   private:
    ::mundy::storage<DxVector> dx_storage_;
    ::mundy::storage<MDxVector> mdx_storage_;
    ::mundy::storage<LMDxVector> lmdx_storage_;
  };
#undef MUNDY_WORKSPACE_FIELD
#undef MUNDY_WORKSPACE_CHILD

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

/// \brief Backend for Kokkos single process execution
template <typename ExecSpace>
struct KokkosBackend {
 public:
  using exec_space = ExecSpace;

 public:
  template <class Vector>
  static auto make_vector_like(const Vector& q) {
    return Vector("make_vector_like",
                  q.extent(0));  // Kokkos::View has no label-less allocating ctor; a size alone resolves to
                                 // the (incompatible) pointer-wrapping overload instead.
  }

  // make_domain/range_vector is host only, but may be called from KOKKOS_FUNCTION code being called on the host
  // This will cause warnings, but is otherwise perfectly valid, so we suppress the warnings for these functions
  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_domain_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeDomainVectorMember<LinearOp>) {
      return op.make_domain_vector();
    } else if constexpr (impl::DenseMatView<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      using value_type = typename op_t::non_const_value_type;
      using mem_space = typename op_t::memory_space;
      using layout_t = typename Kokkos::View<value_type*, mem_space>::array_layout;
      using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
      return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "domain_vector"), domain_size(op));
    } else {
      static_assert(impl::dependent_false_v<LinearOp>,
                    "KokkosBackend::make_domain_vector requires DenseMatView or op.make_domain_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_range_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeRangeVectorMember<LinearOp>) {
      return op.make_range_vector();
    } else if constexpr (impl::DenseMatView<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      using value_type = typename op_t::non_const_value_type;
      using mem_space = typename op_t::memory_space;
      using layout_t = typename Kokkos::View<value_type*, mem_space>::array_layout;
      using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
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
  MUNDY_REQUIRES(impl::DenseMatView<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.extent(1);
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> && impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.domain_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> && !impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp&) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::domain_size: op must be a rank-2 Kokkos::View or provide size_t domain_size().");
    return 0;
  }

  template <class LinearOp>
  MUNDY_REQUIRES(impl::DenseMatView<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.extent(0);
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> && impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.range_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> && !impl::HasRangeSizeMember<LinearOp>)
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
  MUNDY_REQUIRES(impl::DenseMatView<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y) {
    using value_type = impl::vector_value_type<YVector>;
    MUNDY_THROW_ASSERT(A.extent(1) == x.extent(0), std::invalid_argument, "gemv: dimension mismatch A(:,1) vs x");
    MUNDY_THROW_ASSERT(A.extent(0) == y.extent(0), std::invalid_argument, "gemv: dimension mismatch A(0,:) vs y");
    KokkosBlas::gemv(exec_space{}, "N", value_type(1), A, x, value_type(0), y);
  }

  // Path 1: If op is a dense 2D Kokkos::View, call BLAS gemv.
  // y = alpha * A * x + beta * y
  template <class Scalar, class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(impl::DenseMatView<LinearOp>)
  static void apply(Scalar alpha, const LinearOp& A, const XVector& x, Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(A.extent(1) == x.extent(0), std::invalid_argument, "gemv: dimension mismatch A(:,1) vs x");
    MUNDY_THROW_ASSERT(A.extent(0) == y.extent(0), std::invalid_argument, "gemv: dimension mismatch A(0,:) vs y");
    KokkosBlas::gemv(exec_space{}, "N", alpha, A, x, beta, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  // Path 2: If op has member `apply(x,y)`
  template <class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> && impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    op.apply(x, y);
  }

  // Path 3: Otherwise, runtime error.
  template <typename LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> && !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::apply: op must be a rank-2 Kokkos::View or provide void apply(x,y).");
  }

  /// \brief Dispatch to the matching apply_impl overload below, invalidating the workspace exactly once first.
  template <class LinearOp, class XVector, class YVector, class Workspace>
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    impl::workspace_invalidate(workspace);
    apply_impl(op, x, y, workspace);
  }

 private:
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(impl::DenseMatView<LinearOp>)
  static void apply_impl(const LinearOp& A, const XVector& x, YVector& y, Workspace&) {
    apply(A, x, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> &&
                 impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>)
  static void apply_impl(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    op.apply(x, y, workspace);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> &&
                 !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
                 impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply_impl(const LinearOp& op, const XVector& x, YVector& y, Workspace&) {
    op.apply(x, y);
  }

  template <typename LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!impl::DenseMatView<LinearOp> &&
                 !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
                 !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply_impl(const LinearOp&, const XVector&, YVector&, Workspace&) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::apply: op must be a rank-2 Kokkos::View or provide void apply(x,y[,workspace]).");
  }

 public:
  template <class Scalar, class XVector, class YVector>
  static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    const bool alpha_is_zero = abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = abs(beta) < get_zero_tolerance<Scalar>();

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
    const bool alpha_is_zero = abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = abs(beta) < get_zero_tolerance<Scalar>();

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
  KOKKOS_INLINE_FUNCTION static auto make_vector_like(const Vector& /*x*/) {
    return Vector();
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_domain_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeDomainVectorMember<LinearOp>) {
      return op.make_domain_vector();
    } else if constexpr (requires {
                           typename std::remove_reference_t<LinearOp>::value_type;
                           std::remove_reference_t<LinearOp>::num_cols;
                         }) {
      using op_t = std::remove_reference_t<LinearOp>;
      return Vector<typename op_t::value_type, op_t::num_cols>{};
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
                           typename std::remove_reference_t<LinearOp>::value_type;
                           std::remove_reference_t<LinearOp>::num_rows;
                         }) {
      using op_t = std::remove_reference_t<LinearOp>;
      return Vector<typename op_t::value_type, op_t::num_rows>{};
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
  MUNDY_REQUIRES(is_matrix_v<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& /*op*/) {
    return std::remove_reference_t<LinearOp>::num_cols;
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_matrix_v<LinearOp> && impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.domain_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_matrix_v<LinearOp> && !impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& /*op*/) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::domain_size: op must be a rank-2 Kokkos::View or provide size_t domain_size().");
    return 0;
  }

  template <class LinearOp>
  MUNDY_REQUIRES(is_matrix_v<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& /*op*/) {
    return std::remove_reference_t<LinearOp>::num_rows;
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_matrix_v<LinearOp> && impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.range_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_matrix_v<LinearOp> && !impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& /*op*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "MundyBackend::range_size: op must be a mundy::Matrix or provide size_t range_size().");
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

  /// \brief Dispatch to apply_impl below, invalidating the workspace exactly once first.
  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    impl::workspace_invalidate(workspace);
    apply_impl(op, x, y, workspace);
  }

 private:
  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply_impl(const LinearOp& op, const XVector& x, YVector& y,
                                                Workspace& workspace) {
    if constexpr (impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>) {
      op.apply(x, y, workspace);
    } else if constexpr (impl::HasApplyMember<LinearOp, XVector, YVector>) {
      op.apply(x, y);
    } else {
      y = op * x;
    }
  }

 public:
  template <class Scalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    y = alpha * x + beta * y;
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  KOKKOS_INLINE_FUNCTION static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta,
                                                    const YVector& y, ZVector& z, const Wrapper& wrapper) {
    z = ::mundy::apply(wrapper, alpha * x + beta * y);
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
template <typename Backend, typename LinearOp, typename QVector, space::ValidConvexSpace ConvexSpace,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
class CQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using vector_storage_t = ::mundy::storage<QVector>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using vector_t = typename vector_storage_t::value_type;
  using space_t = ConvexSpace;
  using workspace_t = Workspace;
  using value_type = impl::vector_value_type<vector_t>;

  CQPPProblem(Backend, LinearOp&& A, QVector&& q, const space_t& space)
      : A_(std::forward<LinearOp>(A)),
        q_(std::forward<QVector>(q)),
        space_(space),
        workspace_(impl::make_workspace(A_.get())) {
  }

  CQPPProblem(Backend, LinearOp&& A, QVector&& q, const space_t& space, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), q_(std::forward<QVector>(q)), space_(space), workspace_(std::move(workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  /// \brief This problem's cached scratch state for evaluating A (see Backend::apply and Op::Workspace types).
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
template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          space::ValidConvexSpace ConvexSpace,
          typename WorkspaceA = impl::workspace_for_t<std::remove_cvref_t<LinearOpA>>,
          typename WorkspaceL = impl::workspace_for_t<std::remove_cvref_t<LinearOpL>>>
class MCQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_a_t = ::mundy::storage<LinearOpA>;
  using linear_op_storage_l_t = ::mundy::storage<LinearOpL>;
  using q_vector_storage_t = ::mundy::storage<QVector>;
  using f_vector_storage_t = ::mundy::storage<FVector>;
  using q_vector_t = typename q_vector_storage_t::value_type;
  using f_vector_t = typename f_vector_storage_t::value_type;
  using space_t = ConvexSpace;
  using a_workspace_t = WorkspaceA;
  using l_workspace_t = WorkspaceL;
  using value_type = impl::vector_value_type<q_vector_t>;

  MCQPPProblem(Backend, LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b, const space_t& space)
      : A_(std::forward<LinearOpA>(A)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        a_workspace_(impl::make_workspace(A_.get())),
        l_workspace_(impl::make_workspace(L_.get())) {
  }

  MCQPPProblem(Backend, LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b, const space_t& space,
               a_workspace_t a_workspace, l_workspace_t l_workspace)
      : A_(std::forward<LinearOpA>(A)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        a_workspace_(std::move(a_workspace)),
        l_workspace_(std::move(l_workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return L_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& f_b() const { return f_b_.get(); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  /// \brief This problem's cached scratch state for evaluating A and L (see Backend::apply and Op::Workspace types).
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

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, space::ValidConvexSpace ConvexSpace,
          typename WorkspaceDT = impl::workspace_for_t<std::remove_cvref_t<LinearOpDT>>,
          typename WorkspaceM = impl::workspace_for_t<std::remove_cvref_t<LinearOpM>>,
          typename WorkspaceD = impl::workspace_for_t<std::remove_cvref_t<LinearOpD>>,
          typename WorkspaceL = impl::workspace_for_t<std::remove_cvref_t<LinearOpL>>>
class CongruentMCQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_dt_t = ::mundy::storage<LinearOpDT>;
  using linear_op_storage_m_t = ::mundy::storage<LinearOpM>;
  using linear_op_storage_d_t = ::mundy::storage<LinearOpD>;
  using q_vector_storage_t = ::mundy::storage<QVector>;
  using linear_op_storage_l_t = ::mundy::storage<LinearOpL>;
  using f_vector_storage_t = ::mundy::storage<FVector>;
  using q_vector_t = typename q_vector_storage_t::value_type;
  using space_t = ConvexSpace;
  using dt_workspace_t = WorkspaceDT;
  using m_workspace_t = WorkspaceM;
  using d_workspace_t = WorkspaceD;
  using l_workspace_t = WorkspaceL;
  using value_type = impl::vector_value_type<q_vector_t>;

  CongruentMCQPPProblem(Backend, LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L,
                        FVector&& f_b, const space_t& space)
      : DT_(std::forward<LinearOpDT>(DT)),
        M_(std::forward<LinearOpM>(M)),
        D_(std::forward<LinearOpD>(D)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        dt_workspace_(impl::make_workspace(DT_.get())),
        m_workspace_(impl::make_workspace(M_.get())),
        d_workspace_(impl::make_workspace(D_.get())),
        l_workspace_(impl::make_workspace(L_.get())) {
  }

  CongruentMCQPPProblem(Backend, LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L,
                        FVector&& f_b, const space_t& space, dt_workspace_t dt_workspace, m_workspace_t m_workspace,
                        d_workspace_t d_workspace, l_workspace_t l_workspace)
      : DT_(std::forward<LinearOpDT>(DT)),
        M_(std::forward<LinearOpM>(M)),
        D_(std::forward<LinearOpD>(D)),
        q_(std::forward<QVector>(q)),
        L_(std::forward<LinearOpL>(L)),
        f_b_(std::forward<FVector>(f_b)),
        space_(space),
        dt_workspace_(std::move(dt_workspace)),
        m_workspace_(std::move(m_workspace)),
        d_workspace_(std::move(d_workspace)),
        l_workspace_(std::move(l_workspace)) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& DT() const { return DT_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& M() const { return M_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& D() const { return D_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& L() const { return L_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& f_b() const { return f_b_.get(); }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  /// \brief This problem's cached scratch state for evaluating DT, M, D, and L (see Backend::apply and Op::Workspace
  /// types).
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
template <typename Backend, typename LinearOp, typename QVector,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
class LCPProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using q_vector_storage_t = ::mundy::storage<QVector>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using q_vector_t = typename q_vector_storage_t::value_type;
  using workspace_t = Workspace;
  using value_type = impl::vector_value_type<q_vector_t>;

  LCPProblem(Backend, LinearOp&& A, QVector&& q)
      : A_(std::forward<LinearOp>(A)), q_(std::forward<QVector>(q)), workspace_(impl::make_workspace(A_.get())) {
  }

  LCPProblem(Backend, LinearOp&& A, QVector&& q, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), q_(std::forward<QVector>(q)), workspace_(std::move(workspace)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  /// \brief This problem's cached scratch state for evaluating A (see Backend::apply and Op::Workspace types).
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  q_vector_storage_t q_;
  mutable workspace_t workspace_;
};

template <class Backend, class LinearOp, class QVector>
KOKKOS_FUNCTION auto to_cqpp(const LCPProblem<Backend, LinearOp, QVector>& P) {
  using value_type = typename LCPProblem<Backend, LinearOp, QVector>::value_type;
  static constexpr space::LowerBound Rn_plus{static_cast<value_type>(0)};
  auto A_copy = P.A();
  auto q_copy = P.q();
  return CQPPProblem(P.backend(), std::move(A_copy), std::move(q_copy), Rn_plus, P.workspace());
}

template <class Backend, class LinearOpA, class QVector, class LinearOpL, class FVector, class ConvexSpace,
          class AWorkspace, class LWorkspace>
KOKKOS_FUNCTION auto to_cqpp(
    const MCQPPProblem<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace, AWorkspace, LWorkspace>& P) {
  // get the type of P no ref
  using value_type = std::remove_reference_t<decltype(P)>::value_type;

  auto backend = P.backend();
  using backend_t = decltype(backend);

  // H owns an independent copy of A and L, so it stays valid after P is destroyed.
  auto A_copy = P.A();
  auto L_copy = P.L();
  auto H = MixedReducedOp(backend_t{}, std::move(A_copy), std::move(L_copy));

  auto g = backend_t::make_vector_like(P.q());
  auto a_workspace = P.a_workspace();
  auto l_workspace = P.l_workspace();

  backend_t::apply(P.A(), P.f_b(), g, a_workspace);                                     // g = A f_b
  backend_t::axpby(static_cast<value_type>(1), P.q(), static_cast<value_type>(-1), g);  // g = q - A f_b

  auto ax = backend_t::make_range_vector(P.A());
  auto lax = backend_t::make_range_vector(P.L());
  auto workspace = H.make_workspace(std::move(ax), std::move(lax), std::move(a_workspace), std::move(l_workspace));
  return CQPPProblem(backend_t{}, std::move(H), std::move(g), P.space(), workspace);
}

template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD, class QVector, class LinearOpL,
          class FVector, class ConvexSpace, class DTWorkspace, class MWorkspace, class DWorkspace, class LWorkspace>
KOKKOS_FUNCTION auto to_cqpp(
    const CongruentMCQPPProblem<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace,
                                DTWorkspace, MWorkspace, DWorkspace, LWorkspace>& P) {
  using value_type = std::remove_reference_t<decltype(P)>::value_type;

  auto backend = P.backend();
  using backend_t = decltype(backend);

  // H owns an independent copy of DT, M, D, and L, so it stays valid after P is destroyed.
  auto DT_copy = P.DT();
  auto M_copy = P.M();
  auto D_copy = P.D();
  auto L_copy = P.L();
  auto H =
      CongruentMixedReducedOp(backend_t{}, std::move(DT_copy), std::move(M_copy), std::move(D_copy), std::move(L_copy));

  auto g = backend_t::make_vector_like(P.q());
  auto m_f_b = backend_t::make_range_vector(P.M());
  auto dt_workspace = P.dt_workspace();
  auto m_workspace = P.m_workspace();
  auto d_workspace = P.d_workspace();
  auto l_workspace = P.l_workspace();

  backend_t::apply(P.M(), P.f_b(), m_f_b, m_workspace);
  backend_t::apply(P.DT(), m_f_b, g, dt_workspace);                                     // g = D^T M f_b
  backend_t::axpby(static_cast<value_type>(1), P.q(), static_cast<value_type>(-1), g);  // g = q - D^T M f_b

  auto dx = backend_t::make_range_vector(P.D());
  auto mdx = backend_t::make_range_vector(P.M());
  auto lmdx = backend_t::make_range_vector(P.L());
  auto workspace = H.make_workspace(std::move(dx), std::move(mdx), std::move(lmdx), std::move(dt_workspace),
                                    std::move(m_workspace), std::move(d_workspace), std::move(l_workspace));
  return CQPPProblem(backend_t{}, std::move(H), std::move(g), P.space(), workspace);
}
//@}

//! \name Policies
//@{

struct LinfNormProjectedGradientResidual {  // Lower bound only for non-negativity constraints
  template <typename Backend, typename XVector, typename GradVector,
            typename ReductionScalar = impl::vector_value_type<GradVector>>
  KOKKOS_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                             const XVector& x,                         //
                                             const GradVector& grad,                   //
                                             const space::LowerBound<ReductionScalar>& convex_space) const {
    MUNDY_THROW_REQUIRE(convex_space.bound() == static_cast<ReductionScalar>(0), std::invalid_argument,
                        "LinfNormProjectedGradientResidual is only implemented for non-negativity constraints.");

    using value_type = ReductionScalar;

    size_t n = Backend::size(x);
    value_type largest_abs_gradient;
    Backend::template reduce_max<value_type>(
        x, n,
        KOKKOS_LAMBDA(const int i, value_type& max_val) {
          // perform the projection EQ 2.2 of Dai & Fletcher 2005
          value_type x_i = Backend::vector_data(x, i);
          value_type grad_i = Backend::vector_data(grad, i);

          value_type abs_projected_grad;
          if (x_i < get_zero_tolerance<value_type>()) {
            abs_projected_grad = max(value_type(0), grad_i);
          } else {
            abs_projected_grad = abs(grad_i);
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
            typename ReductionScalar = impl::vector_value_type<GradVector>>
  KOKKOS_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                             const XVector& x,                         //
                                             const GradVector& grad,                   //
                                             const ConvexSpace& convex_space) const {
    using value_type = ReductionScalar;

    // This res comes from line 17 and Eq 25 of Mazhar 2015
    // res =  1.0 / (3 * num_unknowns * gd) * norm_inf(xk - proj(xk - gd * gk))
    size_t num_unknowns = Backend::size(x);
    constexpr value_type small_step_size = static_cast<value_type>(1e-6);
    value_type largest_abs_diff;
    Backend::template reduce_max<value_type>(
        x, num_unknowns,
        KOKKOS_LAMBDA(const int i, value_type& max_val) {
          value_type x_i = Backend::vector_data(x, i);
          value_type grad_i = Backend::vector_data(grad, i);
          value_type x_i_proj = convex_space.project(x_i - small_step_size * grad_i);
          value_type abs_diff = abs(x_i - x_i_proj);
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
            typename ReductionScalar = impl::vector_value_type<XVector>>
  KOKKOS_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                             const XOldVector& x_old,
                                             const GradOldVector& grad_old,  //
                                             const XVector& x, const GradVector& grad) const {
    using value_type = ReductionScalar;

    value_type num = Backend::template diff_dot<value_type>(x, x_old);  // (x - x_old) dot (x - x_old)
    value_type denom =
        Backend::template diff_dot<value_type>(x, x_old, grad, grad_old);  // (x - x_old) dot (grad - grad_old)

    // Avoid division by zero
    constexpr value_type eps = get_zero_tolerance<value_type>() * static_cast<value_type>(10);
    denom += eps * (abs(denom) < eps);

    return num / denom;
  }
};  // BBStepStrategy
//@}

template <typename Scalar>
struct PGDConfig {
  using value_type = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};
};

template <class Scalar>
struct SolveResult {
  using value_type = Scalar;

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

template <class Scalar, class XVector, class GradVector, class XTmpVector, class GradTmpVector>
class PGDState {
 public:
  using value_type = Scalar;
  using x_vector_storage_t = ::mundy::storage<XVector>;
  using grad_vector_storage_t = ::mundy::storage<GradVector>;
  using x_tmp_vector_storage_t = ::mundy::storage<XTmpVector>;
  using grad_tmp_vector_storage_t = ::mundy::storage<GradTmpVector>;

  KOKKOS_INLINE_FUNCTION
  PGDState(XVector&& x, GradVector&& g, XTmpVector&& x_tmp, GradTmpVector&& g_tmp)
      : x_(std::forward<XVector>(x)),
        g_(std::forward<GradVector>(g)),
        x_tmp_(std::forward<XTmpVector>(x_tmp)),
        g_tmp_(std::forward<GradTmpVector>(g_tmp)) {
  }

  // Accessors (const/non-const as needed)
  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& x()      { return x_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& x() const{ return x_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& grad()      { return g_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& grad() const{ return g_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& x_tmp()      { return x_tmp_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& x_tmp() const{ return x_tmp_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& grad_tmp()      { return g_tmp_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& grad_tmp() const{ return g_tmp_.get(); }
  // clang-format on

  // Iteration locals with accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION unsigned& iter()         { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&     converged()    { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type& residual()     { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type& step_size()        { return step_size_; }

  KOKKOS_INLINE_FUNCTION unsigned  iter()    const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool      converged() const { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type  residual() const  { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type  step_size()    const  { return step_size_; }
  // clang-format on

 private:
  x_vector_storage_t x_;
  grad_vector_storage_t g_;
  x_tmp_vector_storage_t x_tmp_;
  grad_tmp_vector_storage_t g_tmp_;
  unsigned iter_{0};
  bool converged_{false};
  value_type residual_{0};
  value_type step_size_{1};
};

template <class StepPolicy, class ResidualPolicy, class Config>
class PGDStrategy {
 public:
  using value_type = typename Config::value_type;

  using step_policy_t = StepPolicy;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = SolveResult<value_type>;

  KOKKOS_INLINE_FUNCTION
  PGDStrategy(step_policy_t step, residual_policy_t resid, config_t cfg = {}) : step_(step), resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    // x_tmp = x
    backend_t::deep_copy(state.x_tmp(), state.x());

    // grad_tmp = A x_tmp + q
    backend_t::apply(prob.A(), state.x_tmp(), state.grad_tmp(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad_tmp());

    // Dai-Fletcher Sec. 5 initial step
    state.residual() = resid_(backend, state.x_tmp(), state.grad_tmp(), prob.space());

    // Initialize iteration state (allow for early exit)
    state.iter() = 0;
    state.converged() = (state.residual() <= static_cast<value_type>(cfg_.tol));
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
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    // x = Proj(x_tmp - step_size * grad_tmp)
    backend_t::wrapped_axpbyz(one, state.x_tmp(), -state.step_size(), state.grad_tmp(), state.x(), prob.space());

    // grad = A x + q
    backend_t::apply(prob.A(), state.x(), state.grad(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad());

    // residual & test
    state.residual() = resid_(backend, state.x(), state.grad(), prob.space());
    if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
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
  KOKKOS_FUNCTION bool done(const State& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_FUNCTION result_t result(const State& state) const {
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

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

/// \brief Deduction guide for QuadraticFormOp
template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
QuadraticFormOp(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&)
    -> QuadraticFormOp<Backend, LinearOpDT, LinearOpM, LinearOpD>;

/// \brief Deduction guide for MixedReducedOp
template <class Backend, class LinearOpA, class LinearOpL>
MixedReducedOp(Backend, LinearOpA&&, LinearOpL&&) -> MixedReducedOp<Backend, LinearOpA, LinearOpL>;

/// \brief Deduction guide for CongruentMixedReducedOp
template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD, class LinearOpL>
CongruentMixedReducedOp(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&, LinearOpL&&)
    -> CongruentMixedReducedOp<Backend, LinearOpDT, LinearOpM, LinearOpD, LinearOpL>;

/// \brief Deduction guide for CQPPProblem
template <typename Backend, typename LinearOp, typename QVector, space::ValidConvexSpace ConvexSpace>
CQPPProblem(Backend, LinearOp&&, QVector&&, const ConvexSpace&) -> CQPPProblem<Backend, LinearOp, QVector, ConvexSpace>;

template <typename Backend, typename LinearOp, typename QVector, space::ValidConvexSpace ConvexSpace,
          typename Workspace>
CQPPProblem(Backend, LinearOp&&, QVector&&, const ConvexSpace&, const Workspace&)
    -> CQPPProblem<Backend, LinearOp, QVector, ConvexSpace, Workspace>;

/// \brief Deduction guide for LCPProblem
template <typename Backend, typename LinearOp, typename QVector>
LCPProblem(Backend, LinearOp&&, QVector&&) -> LCPProblem<Backend, LinearOp, QVector>;

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
LCPProblem(Backend, LinearOp&&, QVector&&, const Workspace&) -> LCPProblem<Backend, LinearOp, QVector, Workspace>;

/// \brief Deduction guide for MCQPPProblem
template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          space::ValidConvexSpace ConvexSpace>
MCQPPProblem(Backend, LinearOpA&&, QVector&&, LinearOpL&&, FVector&&, const ConvexSpace&)
    -> MCQPPProblem<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace>;

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          space::ValidConvexSpace ConvexSpace, typename AWorkspace, typename LWorkspace>
MCQPPProblem(Backend, LinearOpA&&, QVector&&, LinearOpL&&, FVector&&, const ConvexSpace&, const AWorkspace&,
             const LWorkspace&)
    -> MCQPPProblem<Backend, LinearOpA, QVector, LinearOpL, FVector, ConvexSpace, AWorkspace, LWorkspace>;

/// \brief Deduction guide for CongruentMCQPPProblem
template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, space::ValidConvexSpace ConvexSpace>
CongruentMCQPPProblem(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&, QVector&&, LinearOpL&&, FVector&&,
                      const ConvexSpace&)
    -> CongruentMCQPPProblem<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace>;

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, space::ValidConvexSpace ConvexSpace, typename DTWorkspace,
          typename MWorkspace, typename DWorkspace, typename LWorkspace>
CongruentMCQPPProblem(Backend, LinearOpDT&&, LinearOpM&&, LinearOpD&&, QVector&&, LinearOpL&&, FVector&&,
                      const ConvexSpace&, const DTWorkspace&, const MWorkspace&, const DWorkspace&, const LWorkspace&)
    -> CongruentMCQPPProblem<Backend, LinearOpDT, LinearOpM, LinearOpD, QVector, LinearOpL, FVector, ConvexSpace,
                             DTWorkspace, MWorkspace, DWorkspace, LWorkspace>;

/// \brief Deduction guide for PGDConfig
template <typename Scalar>
PGDConfig(unsigned, Scalar) -> PGDConfig<Scalar>;

/// \brief Deduction guide for PGDState
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
PGDState(XVector&&, GradVector&&, XTmpVector&&, GradTmpVector&&)
    -> PGDState<impl::vector_value_type<XVector>, XVector, GradVector, XTmpVector, GradTmpVector>;

/// \brief Deduction guide for PGDStrategy
template <class StepPolicy, class ResidualPolicy, class Config>
PGDStrategy(StepPolicy, ResidualPolicy, Config = {}) -> PGDStrategy<StepPolicy, ResidualPolicy, Config>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace convex

template <typename Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
KOKKOS_INLINE_FUNCTION auto make_quadratic_form(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D) {
  return convex::QuadraticFormOp(Backend{}, std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                 std::forward<LinearOpD>(D));
}

template <typename Backend, typename LinearOp, typename QVector, convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOp&& A, QVector&& q, ConvexSpace&& space) {
  return convex::CQPPProblem(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q),
                             std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, ConvexSpace&& space) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  return convex::CQPPProblem(Backend{}, std::move(A), std::forward<QVector>(q), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          convex::space::ValidConvexSpace ConvexSpace, typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, FVector&& f,
                                      UVector&& u, ConvexSpace&& space) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  auto workspace = A.make_workspace(std::forward<FVector>(f), std::forward<UVector>(u));
  return convex::CQPPProblem(Backend{}, std::move(A), std::forward<QVector>(q), std::forward<ConvexSpace>(space),
                             std::move(workspace));
}

template <typename Backend, typename LinearOp, typename QVector, convex::space::ValidConvexSpace ConvexSpace,
          typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(LinearOp&& A, QVector&& q, ConvexSpace&& space, Workspace&& workspace) {
  return convex::CQPPProblem(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q),
                             std::forward<ConvexSpace>(space), std::forward<Workspace>(workspace));
}

template <typename Backend, typename LinearOp, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOp&& A, QVector&& q) {
  return convex::LCPProblem(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  return convex::LCPProblem(Backend{}, std::move(A), std::forward<QVector>(q));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, FVector&& f,
                                     UVector&& u) {
  auto A = make_quadratic_form<Backend>(std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                        std::forward<LinearOpD>(D));
  auto workspace = A.make_workspace(std::forward<FVector>(f), std::forward<UVector>(u));
  return convex::LCPProblem(Backend{}, std::move(A), std::forward<QVector>(q), std::move(workspace));
}

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_lcp(LinearOp&& A, QVector&& q, Workspace&& workspace) {
  return convex::LCPProblem(Backend{}, std::forward<LinearOp>(A), std::forward<QVector>(q),
                            std::forward<Workspace>(workspace));
}

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpA&& A, QVector&& q, LinearOpL&& L, FVector&& f_b,
                                            ConvexSpace&& space) {
  return convex::MCQPPProblem(Backend{}, std::forward<LinearOpA>(A), std::forward<QVector>(q),
                              std::forward<LinearOpL>(L), std::forward<FVector>(f_b), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpA, typename QVector, typename LinearOpL, typename FVector,
          convex::space::ValidConvexSpace ConvexSpace, typename AWorkspace, typename LWorkspace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpA&& A, QVector&& q,    //
                                            LinearOpL&& L, FVector&& f_b,  //
                                            ConvexSpace&& space,           //
                                            AWorkspace&& a_workspace,      //
                                            LWorkspace&& l_workspace) {
  return convex::MCQPPProblem(Backend{}, std::forward<LinearOpA>(A), std::forward<QVector>(q),
                              std::forward<LinearOpL>(L), std::forward<FVector>(f_b), std::forward<ConvexSpace>(space),
                              std::forward<AWorkspace>(a_workspace), std::forward<LWorkspace>(l_workspace));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpL, typename FVector, convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpL&& L,
                                            FVector&& f_b, ConvexSpace&& space) {
  return convex::CongruentMCQPPProblem(Backend{}, std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                       std::forward<LinearOpD>(D), std::forward<QVector>(q), std::forward<LinearOpL>(L),
                                       std::forward<FVector>(f_b), std::forward<ConvexSpace>(space));
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename LinearOpB, typename LinearOpS, typename LinearOpBT, typename BVector,
          convex::space::ValidConvexSpace ConvexSpace>
KOKKOS_FUNCTION auto make_mixed_cqpp(LinearOpDT&& DT, LinearOpM&& M, LinearOpD&& D, QVector&& q, LinearOpB&& B,
                                     LinearOpS&& S, LinearOpBT&& BT, BVector&& b, ConvexSpace&& space) {
  using backend_t = Backend;

  // Backend::apply/make_range_vector operate on the raw operators directly -- no need to wrap B/S/b in storage
  // just to read from them here. B and S are only forwarded (not read again) after this, into make_quadratic_form,
  // which is the one place that decides how they end up stored.
  auto tmp = backend_t::make_range_vector(S);
  auto f_b = backend_t::make_range_vector(B);
  backend_t::apply(S, b, tmp);
  backend_t::apply(B, tmp, f_b);

  auto L = make_quadratic_form<backend_t>(std::forward<LinearOpB>(B), std::forward<LinearOpS>(S),
                                          std::forward<LinearOpBT>(BT));

  return convex::CongruentMCQPPProblem(backend_t{}, std::forward<LinearOpDT>(DT), std::forward<LinearOpM>(M),
                                       std::forward<LinearOpD>(D), std::forward<QVector>(q), std::move(L),
                                       std::move(f_b), std::forward<ConvexSpace>(space));
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
  return convex::PGDState(std::forward<XVector>(x), std::forward<GradVector>(grad), std::forward<XTmpVector>(x_tmp),
                          std::forward<GradTmpVector>(grad_tmp));
}

/// \brief Solve a constrained quadratic programming problem (CQPP)
///
/// \param prob The constrained quadratic programming problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(convex::CQPPSolverStrategy<Strategy, Problem, State>)
KOKKOS_FUNCTION auto solve_cqpp(const Problem& prob, const Strategy& strat, State& state) {
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
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(requires(const Problem& p) {
  { to_cqpp(p) };
})
KOKKOS_FUNCTION auto solve_mixed_cqpp(const Problem& prob, const Strategy& strat, State& state) {
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
///    Matrix3d A{};
///    Vector3d q{};
///    Vector3d x{}, grad{}, x_tmp{}, grad_tmp{};
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
///    Matrix3d D{};
///    Matrix3d M{};
///    Vector3d q{};
///    Vector3d x{}, grad{}, x_tmp{}, grad_tmp{};
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
MUNDY_REQUIRES(requires(const Problem& p) {
  { to_cqpp(p) };
})
KOKKOS_FUNCTION auto solve_lcp(const Problem& prob, const Strategy& strat, State& state) {
  // Convert LCP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_HPP_
