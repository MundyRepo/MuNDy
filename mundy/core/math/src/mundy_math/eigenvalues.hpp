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

#ifndef MUNDY_MATH_EIGENVALUES_HPP_
#define MUNDY_MATH_EIGENVALUES_HPP_

/// \file eigenvalues.hpp
/// \brief Extremal eigenvalues of a square operator via the power method or, for a symmetric one, the Lanczos method.
///
/// The power method iterates q <- A q / ||A q|| from a caller-supplied start vector and reports the Rayleigh quotient
/// lambda = q . A q of the unit iterate. It converges to the eigenvalue of largest magnitude at rate
/// |lambda_2 / lambda_1| per iteration, provided the start vector has a component along its eigenvector.
///
/// The Lanczos method builds the tridiagonal T that a symmetric A takes on the Krylov space of a start vector. T's
/// extreme eigenvalues approach the ends of A's spectrum (of P A, preconditioned by an SPD P) from inside, each with a
/// residual bounding its distance to an eigenvalue of A.

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/NumTraits.hpp>        // for mundy::NumTraits
#include <mundy_math/Tolerance.hpp>        // for mundy::get_relaxed_zero_tolerance<T>
#include <mundy_math/cmath.hpp>            // for mundy::sqrt
#include <mundy_math/impl/eigenvalues_impl.hpp>  // for mundy::impl::{cold_start, extreme_eigenvalue, ...}
#include <mundy_math/linear_ops.hpp>       // for mundy::make_shifted_op
#include <mundy_math/preconditioners.hpp>  // for mundy::{NoPreconditioner, Preconditioner}
#include <mundy_math/residuals.hpp>        // for the vector and change residual policies
#include <mundy_math/solver_backends.hpp>  // for mundy::impl::{vector_value_type, ...}
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Solve result
//@{

/// \brief One eigenvalue of a solve: its iteration count, residual, eigenvalue, and whether it converged.
///
/// A default EigenResult (no iterations, NaN residual and eigenvalue, not converged) is one the solve did not compute.
template <class Scalar>
struct EigenResult {
  using value_type = Scalar;

  unsigned num_iters{0};
  Scalar residual{NumTraits<Scalar>::quiet_NaN()};
  Scalar eigenvalue{NumTraits<Scalar>::quiet_NaN()};
  bool converged{false};
};

/// \brief Write an EigenResult to an ostream.
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const EigenResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", eigenvalue: " << result.eigenvalue
     << ", converged?: " << result.converged;
  return os;
}

/// \brief The largest and the smallest eigenvalue of a solve.
template <class Scalar>
struct EigenBounds {
  using value_type = Scalar;

  EigenResult<Scalar> largest;
  EigenResult<Scalar> smallest;
};

/// \brief Write an EigenBounds to an ostream.
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const EigenBounds<Scalar> bounds) {
  os << "largest: {" << bounds.largest << "}, smallest: {" << bounds.smallest << "}";
  return os;
}
//@}

template <typename Scalar>
struct PowerConfig {
  using value_type = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};  // compared directly against whatever ResidualPolicy reports
};

template <typename Scalar>
struct LanczosConfig {
  using value_type = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};  // an end converges once rho <= tol |theta|, or rho is rounding
  bool largest{true};                                // solve for the largest eigenvalue
  bool smallest{true};                               // solve for the smallest eigenvalue
};

/// \brief The eigenvalue problem A q = lambda q for a square operator A, paired with a mutable workspace.
template <typename Backend, typename LinearOp,
          typename Workspace = impl::workspace_for_t<std::remove_cvref_t<LinearOp>>>
class EigenProblem {
 public:
  using backend_t = Backend;
  using linear_op_storage_t = ::mundy::storage<LinearOp>;
  using linear_op_t = typename linear_op_storage_t::value_type;
  using workspace_t = Workspace;

  KOKKOS_INLINE_FUNCTION
  EigenProblem(Backend, LinearOp&& A) : A_(std::forward<LinearOp>(A)), workspace_(impl::make_workspace(A_.get())) {
    MUNDY_THROW_ASSERT(Backend::domain_size(A_.get()) == Backend::range_size(A_.get()), std::invalid_argument,
                       "EigenProblem: operator must be square.");
  }

  KOKKOS_INLINE_FUNCTION
  EigenProblem(Backend, LinearOp&& A, workspace_t workspace)
      : A_(std::forward<LinearOp>(A)), workspace_(std::move(workspace)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(A_.get()) == Backend::range_size(A_.get()), std::invalid_argument,
                       "EigenProblem: operator must be square.");
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return A_.get(); }
  /// \brief Cached scratch state for evaluating A (mutated during a solve).
  ///
  /// One Problem must not back two concurrent solves; construct a fresh Problem per concurrent solve.
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_storage_t A_;
  mutable workspace_t workspace_;
};

/// \brief The power-method state: the q/z/r vectors and the iteration scalars.
///
/// q is the start vector on entry (zero: the cold start) and the unit eigenvector estimate on exit; z holds A q and r
/// the eigen residual A q - lambda q. eigenvalue() is the Rayleigh quotient of the q that residual() was measured at,
/// and NaN before the first iterate.
template <class Scalar, class QVector, class ZVector, class RVector>
class PowerState {
 public:
  using value_type = Scalar;

  KOKKOS_INLINE_FUNCTION
  PowerState(QVector&& q, ZVector&& z, RVector&& r)
      : q_(std::forward<QVector>(q)), z_(std::forward<ZVector>(z)), r_(std::forward<RVector>(r)) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& q()       { return q_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& q() const { return q_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& z()       { return z_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& z() const { return z_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& r()       { return r_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& r() const { return r_.get(); }

  KOKKOS_INLINE_FUNCTION unsigned&   iter()             { return iter_; }
  KOKKOS_INLINE_FUNCTION unsigned    iter()       const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&       converged()        { return converged_; }
  KOKKOS_INLINE_FUNCTION bool        converged()  const { return converged_; }
  KOKKOS_INLINE_FUNCTION value_type& residual()         { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type  residual()   const { return residual_; }
  KOKKOS_INLINE_FUNCTION value_type& eigenvalue()       { return eigenvalue_; }
  KOKKOS_INLINE_FUNCTION value_type  eigenvalue() const { return eigenvalue_; }
  // clang-format on

 private:
  ::mundy::storage<QVector> q_;
  ::mundy::storage<ZVector> z_;
  ::mundy::storage<RVector> r_;
  unsigned iter_{0};
  bool converged_{false};
  Scalar residual_{NumTraits<Scalar>::quiet_NaN()};
  Scalar eigenvalue_{NumTraits<Scalar>::quiet_NaN()};
};

/// \brief Stands for an absent vector of a Lanczos state: an end without an estimate, Bv without a preconditioner, or
/// the basis without estimates.
struct NoVector {};

/// \brief Whether T is NoVector.
template <class T>
constexpr bool is_no_vector_v = std::same_as<std::remove_cvref_t<T>, NoVector>;

/// \brief The Lanczos state: the recurrence's vectors, T and its scratch, the eigenvector estimates, and each end.
///
/// The vectors follow Saad's Lanczos for P A in the B inner product, B = P^-1: v holds the Lanczos vector v_j, Bv holds
/// B v_j, Bv_prev holds B v_{j-1}, and Av holds A v_j, all of A's size. Without a preconditioner B = I, and Bv is
/// NoVector, as v holds B v_j.
///
/// The scalar array holds at least (4 + e) max_iters scalars, for e the number of estimate vectors. alpha()[i] and
/// beta()[i] are alpha_{i+1} and beta_{i+1}: after k = iter() iterations, T_k has diagonal alpha_1..alpha_k and
/// off-diagonal beta_1..beta_{k-1}, and beta_k couples it to v_{k+1}. The rest of the array is scratch.
///
/// Each estimate vector is its end's start on entry (zero: the cold start) and its eigenvector estimate on exit, or
/// NoVector for an end without one. The basis keeps the first Lanczos vectors, from which the estimates are formed; it
/// is NoVector without estimate vectors. No two of these vectors may alias.
///
/// largest() and smallest() are each end's latest measurement.
template <class VVector, class BvPrevVector, class AvVector, class BvVector, class ScalarArray,
          class LargestVector = NoVector, class SmallestVector = NoVector, class Basis = NoVector>
class LanczosState {
 public:
  using value_type = impl::vector_value_type<std::remove_cvref_t<VVector>>;
  static constexpr bool has_largest_estimate = !is_no_vector_v<LargestVector>;
  static constexpr bool has_smallest_estimate = !is_no_vector_v<SmallestVector>;

  KOKKOS_INLINE_FUNCTION
  LanczosState(VVector&& v, BvPrevVector&& Bv_prev, AvVector&& Av, BvVector&& Bv, ScalarArray&& scalars,
               LargestVector&& largest_estimate, SmallestVector&& smallest_estimate, Basis&& basis)
      : v_(std::forward<VVector>(v)),
        Bv_prev_(std::forward<BvPrevVector>(Bv_prev)),
        Av_(std::forward<AvVector>(Av)),
        Bv_(std::forward<BvVector>(Bv)),
        scalars_(std::forward<ScalarArray>(scalars)),
        largest_estimate_(std::forward<LargestVector>(largest_estimate)),
        smallest_estimate_(std::forward<SmallestVector>(smallest_estimate)),
        basis_(std::forward<Basis>(basis)) {
  }

  /// \brief The iterations the scalar array has room for.
  KOKKOS_INLINE_FUNCTION size_t capacity() const {
    return impl::scalar_array_size(scalars()) / num_sections;
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION       auto& v()                       { return v_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& v()                 const { return v_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& Bv_prev()                 { return Bv_prev_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& Bv_prev()           const { return Bv_prev_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& Av()                      { return Av_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& Av()                const { return Av_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& Bv()                      { return Bv_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& Bv()                const { return Bv_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& scalars()                 { return scalars_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& scalars()           const { return scalars_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& largest_estimate()        { return largest_estimate_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& largest_estimate()  const { return largest_estimate_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& smallest_estimate()       { return smallest_estimate_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& smallest_estimate() const { return smallest_estimate_.get(); }
  KOKKOS_INLINE_FUNCTION       auto& basis()                   { return basis_.get(); }
  KOKKOS_INLINE_FUNCTION const auto& basis()             const { return basis_.get(); }

  KOKKOS_INLINE_FUNCTION       value_type* alpha()                 { return section(0); }
  KOKKOS_INLINE_FUNCTION const value_type* alpha()           const { return section(0); }
  KOKKOS_INLINE_FUNCTION       value_type* beta()                  { return section(1); }
  KOKKOS_INLINE_FUNCTION const value_type* beta()            const { return section(1); }
  KOKKOS_INLINE_FUNCTION       value_type* forward_pivots()        { return section(2); }
  KOKKOS_INLINE_FUNCTION       value_type* backward_pivots()       { return section(3); }
  // s(largest): for an end with an estimate vector, T's unit eigenvector at that end's eigenvalue
  KOKKOS_INLINE_FUNCTION value_type* s(bool largest) { return section(4 + (!largest && has_largest_estimate)); }

  KOKKOS_INLINE_FUNCTION unsigned&                     iter()              { return iter_; }
  KOKKOS_INLINE_FUNCTION unsigned                      iter()        const { return iter_; }
  KOKKOS_INLINE_FUNCTION EigenResult<value_type>&       largest()           { return largest_; }
  KOKKOS_INLINE_FUNCTION const EigenResult<value_type>& largest()     const { return largest_; }
  KOKKOS_INLINE_FUNCTION EigenResult<value_type>&       smallest()          { return smallest_; }
  KOKKOS_INLINE_FUNCTION const EigenResult<value_type>& smallest()    const { return smallest_; }
  KOKKOS_INLINE_FUNCTION impl::GershgorinBounds<value_type>&       gershgorin()       { return gershgorin_; }
  KOKKOS_INLINE_FUNCTION const impl::GershgorinBounds<value_type>& gershgorin() const { return gershgorin_; }
  // start_scale(): 1 / ||x||_P for the start x, which turns x into B v_1.
  KOKKOS_INLINE_FUNCTION value_type&                   start_scale()       { return start_scale_; }
  KOKKOS_INLINE_FUNCTION value_type                    start_scale() const { return start_scale_; }
  // clang-format on

 private:
  // alpha, beta, the forward and backward pivots, then s for each end with an estimate vector
  static constexpr size_t num_sections = 4 + has_largest_estimate + has_smallest_estimate;

  KOKKOS_INLINE_FUNCTION value_type* section(size_t i) {
    return scalars().data() + i * capacity();
  }
  KOKKOS_INLINE_FUNCTION const value_type* section(size_t i) const {
    return scalars().data() + i * capacity();
  }

  ::mundy::storage<VVector> v_;
  ::mundy::storage<BvPrevVector> Bv_prev_;
  ::mundy::storage<AvVector> Av_;
  ::mundy::storage<BvVector> Bv_;
  ::mundy::storage<ScalarArray> scalars_;
  ::mundy::storage<LargestVector> largest_estimate_;
  ::mundy::storage<SmallestVector> smallest_estimate_;
  ::mundy::storage<Basis> basis_;
  unsigned iter_{0};
  EigenResult<value_type> largest_;
  EigenResult<value_type> smallest_;
  impl::GershgorinBounds<value_type> gershgorin_;
  value_type start_scale_{0};
};

/// \brief The power-method strategy: initialize/iterate/done/result over (Problem, State).
///
/// The iteration itself is fixed; only how convergence is measured is pluggable. ResidualPolicy is either a
/// VectorResidualPolicy, measuring r = A q - lambda q against A q, or a ChangeResidualPolicy, measuring lambda against
/// its previous estimate.
///
/// On return, the problem's workspace holds the intermediates of the apply at the returned q, and the eigenvalue and
/// result.residual are those of that q.
template <class ResidualPolicy, class Config>
class PowerStrategy {
 public:
  using value_type = typename Config::value_type;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = EigenResult<value_type>;

  KOKKOS_INLINE_FUNCTION
  PowerStrategy(residual_policy_t resid, config_t cfg = {}) : resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize([[maybe_unused]] const Problem& prob, State& state) const {
    using backend_t = decltype(prob.backend());
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);

    // q / m scaled to unit norm, for m the power of two at most max |q|; a zero q becomes the cold start c, with
    // m = 1, by the exact select q + [q = 0] c
    value_type q_max;
    backend_t::reduce("PowerStrategy::start_max", Kokkos::Max<value_type>(q_max), abs(backend_t::entries(state.q())));
    const value_type is_zero = q_max == zero ? one : zero;
    const value_type q_scale = impl::power_of_two_at_most(q_max);
    backend_t::assign("PowerStrategy::start", state.q(),
                      (backend_t::entries(state.q()) + is_zero * impl::cold_start<backend_t>(state.q())) / q_scale);
    const value_type q_squared = dot(backend_t{}, state.q(), state.q());
    MUNDY_THROW_REQUIRE(q_squared > zero && q_squared < NumTraits<value_type>::infinity(), std::invalid_argument,
                        "PowerStrategy: the start vector must be finite.");
    backend_t::assign("PowerStrategy::normalize", state.q(), backend_t::entries(state.q()) * (one / sqrt(q_squared)));

    state.iter() = 0;
    state.converged() = false;
    state.residual() = NumTraits<value_type>::quiet_NaN();
    state.eigenvalue() = NumTraits<value_type>::quiet_NaN();
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    backend_t::apply(prob.A(), state.q(), state.z(), workspace);  // z = A q
    const value_type lambda_prev = state.eigenvalue();
    const value_type lambda = dot(backend_t{}, state.q(), state.z());  // q is unit
    backend_t::assign("PowerStrategy::residual", state.r(),
                      backend_t::entries(state.z()) - lambda * backend_t::entries(state.q()));  // r = A q - lambda q

    state.eigenvalue() = lambda;
    state.residual() = measure(backend, state, lambda, lambda_prev);
    ++state.iter();

    if (state.residual() <= static_cast<value_type>(cfg_.tol)) {
      state.converged() = true;
      return true;
    }
    if (state.iter() >= cfg_.max_iters) {
      return false;  // the solve ends at q, the vector its eigenvalue and residual were measured at
    }

    const value_type z_norm = sqrt(dot(backend_t{}, state.z(), state.z()));
    MUNDY_THROW_REQUIRE(z_norm > zero, std::runtime_error,
                        "PowerStrategy: A q = 0 before convergence; the iterate has no direction to continue along.");
    backend_t::assign("PowerStrategy::step", state.q(),
                      backend_t::entries(state.z()) * (one / z_norm));  // q = A q / ||A q||
    return false;
  }

  template <class State>
  KOKKOS_FUNCTION bool done(const State& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_FUNCTION result_t result(const State& state) const {
    return {state.iter(), state.residual(), state.eigenvalue(), state.converged()};
  }

 private:
  template <class Backend, class State>
  KOKKOS_FUNCTION value_type measure(const Backend& backend, const State& state, value_type lambda,
                                     value_type lambda_prev) const {
    using r_vector_t = std::remove_cvref_t<decltype(state.r())>;
    using z_vector_t = std::remove_cvref_t<decltype(state.z())>;
    if constexpr (VectorResidualPolicy<residual_policy_t, Backend, r_vector_t, z_vector_t>) {
      return resid_(backend, state.r(), state.z());
    } else {
      static_assert(ChangeResidualPolicy<residual_policy_t, value_type>,
                    "PowerStrategy: ResidualPolicy must be a VectorResidualPolicy or a ChangeResidualPolicy.");
      return resid_(lambda, lambda_prev);
    }
  }

  residual_policy_t resid_;
  config_t cfg_;
};

/// \brief The Lanczos method for the largest and smallest eigenvalues of a symmetric operator A.
///
/// Each iteration applies A once and adds a row to a small symmetric tridiagonal matrix T: A restricted to the vectors
/// the iterations have visited. T's extreme eigenvalues (Ritz values) approach A's from inside the spectrum, usually
/// long before the iteration count nears A's size.
///
/// For each end the config asks for, the solve reports a Ritz value theta and its residual rho: after k iterations, A
/// has an eigenvalue within rho + k eps ||T|| of theta, for eps the machine epsilon. From a cold start that eigenvalue
/// is the extreme one. An end converges once rho <= tol |theta|, or once rho is at that rounding level; the solve may
/// run up to k / 8 iterations past it.
///
/// The solve starts from the sum of one unit vector per wanted end: along the end's estimate vector, or along a fixed
/// cold start for an end without one or with a zero estimate, which the cold start then replaces. On exit each
/// estimate vector holds its end's eigenvector estimate, ready to start the next solve. Forming an estimate past the
/// basis repeats the solve's applies, so A and P must depend on their input alone.
///
/// With an SPD preconditioner P the ends are those of P A, and P times an estimate is an eigenvector estimate of P A.
///
/// The solve stops unconverged, on the T it has, when a step's arithmetic is not finite. It also stops unconverged when
/// P proves not to be positive definite; no end is then converged. A start that is not finite, or on which P is not
/// positive, stops it at its first iteration. A true return of iterate ends the solve. The problem's workspace is
/// unspecified on return.
template <class Config, class Precond = NoPreconditioner>
class LanczosStrategy {
 public:
  using value_type = typename Config::value_type;
  using config_t = Config;
  using preconditioner_t = Precond;
  using result_t = EigenBounds<value_type>;

  KOKKOS_INLINE_FUNCTION
  explicit LanczosStrategy(config_t cfg = {}) : cfg_(cfg) {
  }

  KOKKOS_INLINE_FUNCTION
  LanczosStrategy(config_t cfg, Precond&& precond) : cfg_(cfg), precond_(std::forward<Precond>(precond)) {
  }

  KOKKOS_INLINE_FUNCTION const auto& precond() const {
    return precond_.get();
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION void initialize(const Problem& prob, State& state) const {
    using backend_t = decltype(prob.backend());
    using scalar_array_t = std::remove_cvref_t<decltype(state.scalars())>;
    static_assert(Preconditioner<preconditioner_t, backend_t, std::remove_cvref_t<decltype(state.v())>>,
                  "LanczosStrategy: Precond must be a Preconditioner.");
    static_assert(is_no_vector_v<decltype(state.Bv())> != is_preconditioned,
                  "LanczosStrategy: the state holds Bv exactly when the strategy has a preconditioner.");
    static_assert(backend_t::template is_scalar_array<scalar_array_t>,
                  "LanczosStrategy: the scalar array must be one the backend's solve drivers read and write.");
    static_assert(std::same_as<std::remove_cvref_t<decltype(*state.scalars().data())>, value_type>,
                  "LanczosStrategy: the scalar array holds the solve's scalar type.");
    static_assert(has_basis<State> == has_estimates<State>,
                  "LanczosStrategy: the state holds a basis exactly when it holds estimate vectors.");
    MUNDY_THROW_REQUIRE(cfg_.largest || cfg_.smallest, std::invalid_argument,
                        "LanczosStrategy: the config must want the largest end, the smallest end, or both.");
    MUNDY_THROW_REQUIRE(cfg_.largest || !State::has_largest_estimate, std::invalid_argument,
                        "LanczosStrategy: an estimate vector for the largest end needs cfg.largest.");
    MUNDY_THROW_REQUIRE(cfg_.smallest || !State::has_smallest_estimate, std::invalid_argument,
                        "LanczosStrategy: an estimate vector for the smallest end needs cfg.smallest.");
    MUNDY_THROW_REQUIRE(state.capacity() >= cfg_.max_iters, std::invalid_argument,
                        "LanczosStrategy: the scalar array must have room for (4 + e) max_iters scalars.");

    state.iter() = 0;
    state.largest() = EigenResult<value_type>{};
    state.smallest() = EigenResult<value_type>{};
    state.gershgorin() = impl::GershgorinBounds<value_type>{};
    start(prob, state);
  }

  template <class Problem, class State>
  KOKKOS_FUNCTION bool iterate(const Problem& prob, State& state) const {
    using backend_t = decltype(prob.backend());
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type infinity = NumTraits<value_type>::infinity();

    if (done(state)) {
      return converged(state);
    }

    // Row k of T: alpha_k = v_k^T (A v_k - beta_{k-1} B v_{k-1}), then beta_k^2 = w_k^T P w_k for w_k = A v_k -
    // beta_{k-1} B v_{k-1} - alpha_k B v_k. beta_k^2 is a reduction of its own: ||A v_k - beta_{k-1} B v_{k-1}||_P^2 -
    // alpha_k^2 cancels once the v_j lose orthogonality.
    const unsigned k = state.iter() + 1;
    const value_type beta_prev = beta_before(state, k);
    value_type alpha;
    value_type beta_squared;
    backend_t::apply(prob.A(), state.v(), state.Av(), prob.workspace());
    backend_t::reduce("LanczosStrategy::alpha", Kokkos::Sum<value_type>(alpha),
                      backend_t::entries(state.v()) *
                          (backend_t::entries(state.Av()) - beta_prev * backend_t::entries(state.Bv_prev())));
    form_w<backend_t>(state, alpha, beta_prev);
    backend_t::reduce("LanczosStrategy::beta", Kokkos::Sum<value_type>(beta_squared),
                      w<backend_t>(state, alpha, beta_prev) * Pw<backend_t>(state, alpha, beta_prev));

    // A step that is not finite, or w_k^T P w_k below zero beyond T_k's rounding level, which only an indefinite P
    // allows, ends the solve on T_{k-1}.
    state.alpha()[k - 1] = alpha;
    const auto gershgorin = state.gershgorin().grown(state.alpha(), state.beta(), k);
    const value_type floor = rounding_level(k, gershgorin.norm_bound);
    const bool finite = abs(alpha) < infinity && abs(beta_squared) < infinity;
    const bool indefinite = is_preconditioned && finite && beta_squared < -floor * floor;
    if (!finite || indefinite) {
      stop_on_breakdown(prob, state, indefinite);
      return true;
    }

    const value_type beta = beta_squared > zero ? sqrt(beta_squared) : zero;
    state.beta()[k - 1] = beta;
    state.gershgorin() = gershgorin;
    state.iter() = k;

    // rho <= beta_k, and measure_end accepts rho at this level, so an invariant T_k converges every end
    const bool invariant = beta <= floor;
    const bool stops = invariant || k >= cfg_.max_iters;
    if (stops || measurement_due(state, k)) {
      measure_ends(state);
    }
    if (converged(state) || stops) {
      form_estimates(prob, state);
      return converged(state);
    }

    advance<backend_t>(state, alpha, beta_prev, beta);
    record_column<backend_t>(state, k);
    return false;
  }

  template <class State>
  KOKKOS_FUNCTION bool done(const State& state) const {
    return converged(state) || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_FUNCTION result_t result(const State& state) const {
    return {state.largest(), state.smallest()};
  }

 private:
  static constexpr bool is_preconditioned = !std::same_as<std::remove_cvref_t<Precond>, NoPreconditioner>;

  template <class State>
  static constexpr bool has_estimates = State::has_largest_estimate || State::has_smallest_estimate;

  template <class State>
  static constexpr bool has_basis = !is_no_vector_v<decltype(std::declval<State&>().basis())>;

  template <class Backend, class State>
  KOKKOS_FUNCTION static constexpr size_t num_columns() {
    return Backend::template num_columns<decltype(std::declval<State&>().basis())>();
  }

  /// \brief k eps ||T||, the rounding level of T_k for a bound norm on ||T||.
  KOKKOS_FUNCTION static value_type rounding_level(unsigned k, value_type norm) {
    return k * NumTraits<value_type>::epsilon() * norm;
  }

  /// \brief beta_{k-1}, the coupling of row k of T to row k - 1, and 0 for k = 1.
  template <class State>
  KOKKOS_FUNCTION static value_type beta_before(const State& state, unsigned k) {
    return k == 1 ? static_cast<value_type>(0) : state.beta()[k - 2];
  }

  /// \brief The index of an end's entry in a per-end pair: 0 for the largest, 1 for the smallest.
  KOKKOS_FUNCTION static constexpr unsigned slot(bool largest) {
    return largest ? 0 : 1;
  }

  /// \brief The largest or the smallest end's measurement.
  template <class State>
  KOKKOS_FUNCTION static auto& end_of(State& state, bool largest) {
    return largest ? state.largest() : state.smallest();
  }

  /// \brief Whether every wanted end converged.
  template <class State>
  KOKKOS_FUNCTION bool converged(const State& state) const {
    return (!cfg_.largest || state.largest().converged) && (!cfg_.smallest || state.smallest().converged);
  }

  /// \brief The k of the latest measurement: that of any wanted end not yet converged, as such ends are measured
  /// together.
  template <class State>
  KOKKOS_FUNCTION unsigned last_measurement(const State& state) const {
    return cfg_.largest && !state.largest().converged ? state.largest().num_iters : state.smallest().num_iters;
  }

  /// \brief Whether T_k's ends are measured: every k up to 16, then once k / 8 iterations have passed since the last
  /// measurement, which keeps the measurements' total work proportional to k.
  template <class State>
  KOKKOS_FUNCTION bool measurement_due(const State& state, unsigned k) const {
    return k <= 16 || k - last_measurement(state) >= k / 8;
  }

  //! \name The optional estimate vectors
  //@{
  // Each helper visits the present estimate vectors, passing std::true_type for the largest end's and
  // std::false_type for the smallest end's.

  /// \brief One assign over the present estimates, each := source(estimate, largest), and the trailing pairs.
  template <class Backend, class State, class Source, class... TargetsAndSources>
  KOKKOS_FUNCTION static void estimates_assign(const char* label, State& state, const Source& source,
                                               TargetsAndSources&&... targets_and_sources) {
    if constexpr (State::has_largest_estimate && State::has_smallest_estimate) {
      Backend::assign(label, state.largest_estimate(), source(state.largest_estimate(), std::true_type{}),
                      state.smallest_estimate(), source(state.smallest_estimate(), std::false_type{}),
                      std::forward<TargetsAndSources>(targets_and_sources)...);
    } else if constexpr (State::has_largest_estimate) {
      Backend::assign(label, state.largest_estimate(), source(state.largest_estimate(), std::true_type{}),
                      std::forward<TargetsAndSources>(targets_and_sources)...);
    } else if constexpr (State::has_smallest_estimate) {
      Backend::assign(label, state.smallest_estimate(), source(state.smallest_estimate(), std::false_type{}),
                      std::forward<TargetsAndSources>(targets_and_sources)...);
    } else if constexpr (sizeof...(TargetsAndSources) > 0) {
      Backend::assign(label, std::forward<TargetsAndSources>(targets_and_sources)...);
    }
  }

  /// \brief One reduce over the present estimates of source(estimate, largest), into results[0] for the largest end
  /// and results[1] for the smallest, and the trailing pairs.
  template <template <class...> class Reducer, class Backend, class State, class Source, class... ReducersAndSources>
  KOKKOS_FUNCTION static void estimates_reduce(const char* label, const State& state,
                                               Kokkos::Array<value_type, 2>& results, const Source& source,
                                               const ReducersAndSources&... reducers_and_sources) {
    if constexpr (State::has_largest_estimate && State::has_smallest_estimate) {
      Backend::reduce(label, Reducer<value_type>(results[0]), source(state.largest_estimate(), std::true_type{}),
                      Reducer<value_type>(results[1]), source(state.smallest_estimate(), std::false_type{}),
                      reducers_and_sources...);
    } else if constexpr (State::has_largest_estimate) {
      Backend::reduce(label, Reducer<value_type>(results[0]), source(state.largest_estimate(), std::true_type{}),
                      reducers_and_sources...);
    } else if constexpr (State::has_smallest_estimate) {
      Backend::reduce(label, Reducer<value_type>(results[1]), source(state.smallest_estimate(), std::false_type{}),
                      reducers_and_sources...);
    } else if constexpr (sizeof...(ReducersAndSources) > 0) {
      Backend::reduce(label, reducers_and_sources...);
    }
  }

  /// \brief rest + sum over the present estimates of term(estimate, largest).
  template <class State, class Term, class Rest>
  KOKKOS_FUNCTION static auto estimates_sum(const State& state, const Term& term, const Rest& rest) {
    if constexpr (State::has_largest_estimate && State::has_smallest_estimate) {
      return rest + term(state.largest_estimate(), std::true_type{}) +
             term(state.smallest_estimate(), std::false_type{});
    } else if constexpr (State::has_largest_estimate) {
      return rest + term(state.largest_estimate(), std::true_type{});
    } else if constexpr (State::has_smallest_estimate) {
      return rest + term(state.smallest_estimate(), std::false_type{});
    } else {
      return rest;
    }
  }

  /// \brief f(estimate, largest) for each present estimate.
  template <class State, class F>
  KOKKOS_FUNCTION static void for_each_estimate(State& state, const F& f) {
    if constexpr (State::has_largest_estimate) {
      f(state.largest_estimate(), std::true_type{});
    }
    if constexpr (State::has_smallest_estimate) {
      f(state.smallest_estimate(), std::false_type{});
    }
  }
  //@}

  //! \name The recurrence
  //@{

  /// \brief The start x, the sum over wanted ends of a unit vector along the end's estimate or along the cold start c;
  /// then B v_1 = x / ||x||_P and v_1 = P x / ||x||_P, and column 0 of the basis = x.
  template <class Problem, class State>
  KOKKOS_FUNCTION void start(const Problem& prob, State& state) const {
    using backend_t = decltype(prob.backend());
    constexpr value_type zero = static_cast<value_type>(0);
    constexpr value_type one = static_cast<value_type>(1);

    // Each term is e / m at unit norm, for m the power of two at most max |e| of an estimate e, so that ||e / m||^2
    // lies in [1, 4n] at any scale of e; m = 1 for c, whose entries lie in (-1, 1). A zero estimate becomes c by the
    // exact select e + [e = 0] c. c is formed once, in v, so its norm and its term read the same entries.
    Kokkos::Array<value_type, 2> largest_entry{zero, zero};
    estimates_reduce<Kokkos::Max, backend_t>("LanczosStrategy::start_max", state, largest_entry,
                                             [](const auto& e, auto) { return abs(backend_t::entries(e)); });
    Kokkos::Array<value_type, 2> m{one, one};
    for (unsigned i = 0; i < 2; ++i) {
      m[i] = impl::power_of_two_at_most(largest_entry[i]);
    }
    estimates_assign<backend_t>(
        "LanczosStrategy::cold_start", state,
        [&](const auto& e, auto largest) {
          const value_type is_zero = largest_entry[slot(largest)] == zero ? one : zero;
          return backend_t::entries(e) + is_zero * impl::cold_start<backend_t>(e);
        },
        state.v(), impl::cold_start<backend_t>(state.v()));

    Kokkos::Array<value_type, 2> squared_norm{one, one};
    value_type cold_squared_norm;
    const auto C = backend_t::entries(state.v());
    estimates_reduce<Kokkos::Sum, backend_t>(
        "LanczosStrategy::start_norms", state, squared_norm,
        [&](const auto& e, auto largest) {
          const auto X = backend_t::entries(e) / m[slot(largest)];
          return X * X;
        },
        Kokkos::Sum<value_type>(cold_squared_norm), C * C);

    const unsigned num_cold =
        (cfg_.largest && !State::has_largest_estimate) + (cfg_.smallest && !State::has_smallest_estimate);
    const value_type cold_weight = num_cold * (one / sqrt(cold_squared_norm));
    backend_t::assign("LanczosStrategy::start", state.Av(),
                      estimates_sum(
                          state,
                          [&](const auto& e, auto largest) {
                            const unsigned i = slot(largest);
                            return (one / sqrt(squared_norm[i])) * (backend_t::entries(e) / m[i]);
                          },
                          cold_weight * C));
    value_type start_squared;
    if constexpr (is_preconditioned) {
      backend_t::apply(precond(), state.Av(), state.v());
      start_squared = dot(backend_t{}, state.Av(), state.v());
    } else {
      start_squared = dot(backend_t{}, state.Av(), state.Av());
    }

    // An invalid start makes v NaN, which ends the solve at its first iteration.
    const bool valid = start_squared > zero && start_squared < NumTraits<value_type>::infinity();
    state.start_scale() = valid ? one / sqrt(start_squared) : NumTraits<value_type>::quiet_NaN();
    if constexpr (has_basis<State>) {
      backend_t::assign("LanczosStrategy::basis", backend_t::column(state.basis(), 0), state.Av());
    }
    first_lanczos_vector<backend_t>(state);
  }

  /// \brief B v_1 := start_scale x and B v_0 := 0, and with a preconditioner v_1 := start_scale P x, for Av = x and,
  /// with a preconditioner, v = P x.
  template <class Backend, class State>
  KOKKOS_FUNCTION static void first_lanczos_vector(State& state) {
    // B v_0 is set, not left to beta_0 = 0: 0 times a stale entry need not be 0
    const value_type scale = state.start_scale();
    if constexpr (is_preconditioned) {
      Backend::assign("LanczosStrategy::first_lanczos_vector", state.Bv(), Backend::entries(state.Av()) * scale,
                      state.Bv_prev(), static_cast<value_type>(0), state.v(), Backend::entries(state.v()) * scale);
    } else {
      Backend::assign("LanczosStrategy::first_lanczos_vector", state.v(), Backend::entries(state.Av()) * scale,
                      state.Bv_prev(), static_cast<value_type>(0));
    }
  }

  /// \brief B v_k: Bv with a preconditioner, v without one.
  template <class State>
  KOKKOS_FUNCTION static auto& Bv_of(State& state) {
    if constexpr (is_preconditioned) {
      return state.Bv();
    } else {
      return state.v();
    }
  }

  /// \brief w_k = A v_k - beta_{k-1} B v_{k-1} - alpha_k B v_k, entry by entry from Av = A v_k.
  template <class Backend, class State>
  KOKKOS_FUNCTION static auto w_from_Av(State& state, value_type alpha, value_type beta_prev) {
    return Backend::entries(state.Av()) - beta_prev * Backend::entries(state.Bv_prev()) -
           alpha * Backend::entries(Bv_of(state));
  }

  /// \brief With a preconditioner, Av := w_k and v := P w_k; without one, w_k is never formed as a vector.
  template <class Backend, class State>
  KOKKOS_FUNCTION void form_w(State& state, value_type alpha, value_type beta_prev) const {
    if constexpr (is_preconditioned) {
      Backend::assign("LanczosStrategy::form_w", state.Av(), w_from_Av<Backend>(state, alpha, beta_prev));
      Backend::apply(precond(), state.Av(), state.v());
    }
  }

  /// \brief w_k, once form_w ran: Av with a preconditioner, else formed entry by entry.
  template <class Backend, class State>
  KOKKOS_FUNCTION static auto w(State& state, value_type alpha, value_type beta_prev) {
    if constexpr (is_preconditioned) {
      return Backend::entries(state.Av());
    } else {
      return w_from_Av<Backend>(state, alpha, beta_prev);
    }
  }

  /// \brief P w_k, once form_w ran: v with a preconditioner, else w_k.
  template <class Backend, class State>
  KOKKOS_FUNCTION static auto Pw(State& state, value_type alpha, value_type beta_prev) {
    if constexpr (is_preconditioned) {
      return Backend::entries(state.v());
    } else {
      return w<Backend>(state, alpha, beta_prev);
    }
  }

  /// \brief B v_{k-1} := B v_k and B v_{k+1} := w_k / beta_k, and with a preconditioner v_{k+1} := P w_k / beta_k, once
  /// form_w ran.
  template <class Backend, class State>
  KOKKOS_FUNCTION static void advance(State& state, value_type alpha, value_type beta_prev, value_type beta) {
    if constexpr (is_preconditioned) {
      Backend::assign("LanczosStrategy::advance", state.Bv_prev(), Backend::entries(state.Bv()), state.Bv(),
                      w<Backend>(state, alpha, beta_prev) / beta, state.v(),
                      Pw<Backend>(state, alpha, beta_prev) / beta);
    } else {
      Backend::assign("LanczosStrategy::advance", state.Bv_prev(), Backend::entries(state.v()), state.v(),
                      w<Backend>(state, alpha, beta_prev) / beta);
    }
  }

  /// \brief Column k of the basis := what step k is formed from again: w_k with a preconditioner, as v_{k+1} is
  /// P w_k / beta_k, which P (w_k / beta_k) does not reproduce; without one w_k is never formed, and the column holds
  /// v_{k+1} = w_k / beta_k.
  template <class Backend, class State>
  KOKKOS_FUNCTION static void record_column(State& state, unsigned k) {
    // its own pass, so that advance is the same pass when form_estimates repeats the step
    if constexpr (has_basis<State>) {
      if (k < num_columns<Backend, State>()) {
        if constexpr (is_preconditioned) {
          Backend::assign("LanczosStrategy::basis", Backend::column(state.basis(), k), state.Av());
        } else {
          Backend::assign("LanczosStrategy::basis", Backend::column(state.basis(), k), state.v());
        }
      }
    }
  }

  /// \brief The state as the solve held it at B v_n, for n = columns, from the basis: B v_{n-1}, B v_n, and with a
  /// preconditioner v_n.
  template <class Backend, class State>
  KOKKOS_FUNCTION void resume(State& state, unsigned columns) const {
    const auto& basis = state.basis();
    if (columns == 1) {
      Backend::assign("LanczosStrategy::resume", state.Av(), Backend::column(basis, 0));
      if constexpr (is_preconditioned) {
        Backend::apply(precond(), state.Av(), state.v());
      }
      first_lanczos_vector<Backend>(state);
      return;
    }

    // B v_{n-1}: column 0 times start_scale for n = 2, else column n - 2 (over beta_{n-2} with a preconditioner)
    const auto Bv_1 = Backend::entries(Backend::column(basis, 0)) * state.start_scale();
    const auto& last = Backend::column(basis, columns - 1);
    if constexpr (is_preconditioned) {
      // B v_n = w_{n-1} / beta_{n-1} and v_n = P w_{n-1} / beta_{n-1}, as advance formed them
      const value_type beta = state.beta()[columns - 2];
      Backend::assign("LanczosStrategy::resume", state.Av(), last);
      Backend::apply(precond(), state.Av(), state.v());
      const auto W = Backend::entries(state.Av());
      const auto PW = Backend::entries(state.v());
      if (columns == 2) {
        Backend::assign("LanczosStrategy::resume", state.Bv_prev(), Bv_1, state.Bv(), W / beta, state.v(),
                        PW / beta);
      } else {
        Backend::assign("LanczosStrategy::resume", state.Bv_prev(),
                        Backend::entries(Backend::column(basis, columns - 2)) / state.beta()[columns - 3], state.Bv(),
                        W / beta, state.v(), PW / beta);
      }
    } else {
      if (columns == 2) {
        Backend::assign("LanczosStrategy::resume", state.Bv_prev(), Bv_1, state.v(), last);
      } else {
        Backend::assign("LanczosStrategy::resume", state.Bv_prev(), Backend::column(basis, columns - 2), state.v(),
                        last);
      }
    }
  }
  //@}

  //! \name The ends
  //@{

  /// \brief The stop on a step that is not finite or shows P not positive definite: the results of the T it has.
  template <class Problem, class State>
  KOKKOS_FUNCTION void stop_on_breakdown(const Problem& prob, State& state, bool indefinite) const {
    measure_ends(state);
    if (indefinite) {
      state.largest().converged = false;
      state.smallest().converged = false;
    }
    form_estimates(prob, state);
  }

  /// \brief Each wanted end's theta and rho on T_k, k = iter(), unless it converged or was measured on T_k.
  ///
  /// An end's num_iters is the k of the T it was measured on; an unmeasured end's 0 matches the empty T_0, which is
  /// never measured.
  template <class State>
  KOKKOS_FUNCTION void measure_ends(State& state) const {
    if (cfg_.largest && !state.largest().converged && state.largest().num_iters != state.iter()) {
      measure_end(state, true, state.largest());
    }
    if (cfg_.smallest && !state.smallest().converged && state.smallest().num_iters != state.iter()) {
      measure_end(state, false, state.smallest());
    }
  }

  /// \brief theta by bisection from the end's last theta, which bounds the new one by interlacing; rho from T's
  /// eigenvector at theta.
  template <class State>
  KOKKOS_FUNCTION void measure_end(State& state, bool largest, EigenResult<value_type>& end) const {
    const auto& gershgorin = state.gershgorin();
    const unsigned k = state.iter();
    const bool measured = end.num_iters > 0;
    const value_type lower = largest && measured ? end.eigenvalue : gershgorin.lower;
    const value_type upper = !largest && measured ? end.eigenvalue : gershgorin.upper;
    const value_type theta = impl::extreme_eigenvalue(state.alpha(), state.beta(), k, largest, lower, upper,
                                                      gershgorin.norm_bound);
    const value_type rho = impl::ritz_residual(state.alpha(), state.beta(), k, theta, gershgorin.norm_bound,
                                               state.forward_pivots(), state.backward_pivots(),
                                               static_cast<value_type*>(nullptr));
    const value_type relative_floor = static_cast<value_type>(cfg_.tol) * abs(theta);
    const value_type rounding_floor = rounding_level(k, gershgorin.norm_bound);
    end = {k, rho, theta, rho <= (relative_floor > rounding_floor ? relative_floor : rounding_floor)};
  }

  /// \brief Each estimate vector := y = sum_i s_i B v_i for its end's s, scaled to y^T P y = 1.
  template <class Problem, class State>
  KOKKOS_FUNCTION void form_estimates(const Problem& prob, State& state) const {
    if constexpr (has_estimates<State>) {
      using backend_t = decltype(prob.backend());

      // y spans B v_1..B v_k, for k the longest end's; every wanted end is first measured at k = 1, so k = 0 means no
      // measurement, and the estimates stay as the start left them.
      unsigned k = 0;
      for_each_estimate(state, [&](const auto&, auto largest) {
        const unsigned end_k = end_of(state, largest).num_iters;
        k = end_k > k ? end_k : k;
      });
      if (k == 0) {
        return;
      }
      ritz_coefficients(state, k);
      const unsigned in_basis = combine_basis<backend_t>(state, k);
      extend_past_basis(prob, state, in_basis, k);
      normalize_estimates<backend_t>(state);
    }
  }

  /// \brief Each estimate's s := T's unit eigenvector at its end's theta, on the T of its measurement, and 0 past it
  /// up to k entries.
  template <class State>
  KOKKOS_FUNCTION static void ritz_coefficients(State& state, unsigned k) {
    for_each_estimate(state, [&](const auto&, auto largest) {
      const auto& end = end_of(state, largest);
      impl::GershgorinBounds<value_type> measured;
      for (unsigned j = 1; j <= end.num_iters; ++j) {
        measured = measured.grown(state.alpha(), state.beta(), j);
      }
      value_type* s = state.s(largest);
      impl::ritz_residual(state.alpha(), state.beta(), end.num_iters, end.eigenvalue, measured.norm_bound,
                          state.forward_pivots(), state.backward_pivots(), s);
      for (unsigned i = end.num_iters; i < k; ++i) {
        s[i] = static_cast<value_type>(0);
      }
    });
  }

  /// \brief Each estimate := sum_{i <= n} s_i B v_i over the basis, for n = min(k, its columns); returns n.
  ///
  /// Column 0 holds x = B v_1 / start_scale; column j >= 1 holds B v_{j+1}, or w_j = beta_j B v_{j+1} with a
  /// preconditioner.
  template <class Backend, class State>
  KOKKOS_FUNCTION static unsigned combine_basis(State& state, unsigned k) {
    constexpr size_t basis_columns = num_columns<Backend, State>();
    const unsigned in_basis = k < basis_columns ? k : static_cast<unsigned>(basis_columns);
    Kokkos::Array<Kokkos::Array<value_type, basis_columns>, 2> weights{};
    for_each_estimate(state, [&](const auto&, auto largest) {
      const value_type* s = state.s(largest);
      auto& c = weights[slot(largest)];
      c[0] = s[0] * state.start_scale();
      for (unsigned j = 1; j < in_basis; ++j) {
        c[j] = is_preconditioned ? s[j] / state.beta()[j - 1] : s[j];
      }
    });
    estimates_assign<Backend>("LanczosStrategy::combination", state, [&](const auto&, auto largest) {
      return Backend::combination(state.basis(), in_basis, weights[slot(largest)]);
    });
    return in_basis;
  }

  /// \brief Each estimate += sum_{n < i <= k} s_i B v_i, for n = in_basis, by the solve's steps n..k - 1 again.
  template <class Problem, class State>
  KOKKOS_FUNCTION void extend_past_basis(const Problem& prob, State& state, unsigned in_basis, unsigned k) const {
    using backend_t = decltype(prob.backend());
    if (k <= in_basis) {
      return;
    }
    resume<backend_t>(state, in_basis);
    for (unsigned j = in_basis; j < k; ++j) {
      const value_type alpha = state.alpha()[j - 1];
      const value_type beta_prev = beta_before(state, j);
      backend_t::apply(prob.A(), state.v(), state.Av(), prob.workspace());
      form_w<backend_t>(state, alpha, beta_prev);
      advance<backend_t>(state, alpha, beta_prev, state.beta()[j - 1]);
      const auto BV = backend_t::entries(Bv_of(state));
      estimates_assign<backend_t>("LanczosStrategy::accumulate", state, [&](const auto& e, auto largest) {
        return backend_t::entries(e) + state.s(largest)[j] * BV;
      });
    }
  }

  /// \brief P y for each estimate y, in v for the largest end and Av for the smallest, both free once y is formed.
  template <class State, class Largest>
  KOKKOS_FUNCTION static auto& Py_of(State& state, Largest) {
    if constexpr (Largest::value) {
      return state.v();
    } else {
      return state.Av();
    }
  }

  /// \brief Each estimate y := y / sqrt(y^T P y).
  template <class Backend, class State>
  KOKKOS_FUNCTION void normalize_estimates(State& state) const {
    constexpr value_type one = static_cast<value_type>(1);
    Kokkos::Array<value_type, 2> squared_norm{one, one};
    if constexpr (is_preconditioned) {
      for_each_estimate(state,
                        [&](const auto& e, auto largest) { Backend::apply(precond(), e, Py_of(state, largest)); });
      estimates_reduce<Kokkos::Sum, Backend>(
          "LanczosStrategy::estimate_norms", state, squared_norm,
          [&](const auto& e, auto largest) { return Backend::entries(e) * Backend::entries(Py_of(state, largest)); });
    } else {
      estimates_reduce<Kokkos::Sum, Backend>(
          "LanczosStrategy::estimate_norms", state, squared_norm,
          [](const auto& e, auto) { return Backend::entries(e) * Backend::entries(e); });
    }
    estimates_assign<Backend>("LanczosStrategy::normalize", state, [&](const auto& e, auto largest) {
      return Backend::entries(e) * (one / sqrt(squared_norm[slot(largest)]));
    });
  }
  //@}

  config_t cfg_;
  ::mundy::storage<Precond> precond_;
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
//! \name Deduction guides
//@{

template <class Backend, class LinearOp>
EigenProblem(Backend, LinearOp&&) -> EigenProblem<Backend, LinearOp>;

template <class Backend, class LinearOp, class Workspace>
EigenProblem(Backend, LinearOp&&, const Workspace&) -> EigenProblem<Backend, LinearOp, Workspace>;

template <class QVector, class ZVector, class RVector>
PowerState(QVector&&, ZVector&&, RVector&&) -> PowerState<impl::vector_value_type<QVector>, QVector, ZVector, RVector>;

template <class ResidualPolicy, class Config>
PowerStrategy(ResidualPolicy, Config = {}) -> PowerStrategy<ResidualPolicy, Config>;

template <class VVector, class BvPrevVector, class AvVector, class BvVector, class ScalarArray, class LargestVector,
          class SmallestVector, class Basis>
LanczosState(VVector&&, BvPrevVector&&, AvVector&&, BvVector&&, ScalarArray&&, LargestVector&&, SmallestVector&&,
             Basis&&) -> LanczosState<VVector, BvPrevVector, AvVector, BvVector, ScalarArray, LargestVector,
                                      SmallestVector, Basis>;

template <class Config>
LanczosStrategy(Config) -> LanczosStrategy<Config>;

template <class Config, class Precond>
LanczosStrategy(Config, Precond&&) -> LanczosStrategy<Config, Precond>;
//@}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

//! \name Factory functions
//@{

template <class Backend, class LinearOp>
KOKKOS_INLINE_FUNCTION auto make_eigen_problem(LinearOp&& A) {
  return EigenProblem(Backend{}, std::forward<LinearOp>(A));
}

template <class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_power_strategy(ResidualPolicy&& residual_policy, const PowerConfig<Scalar>& cfg = {}) {
  return PowerStrategy(std::forward<ResidualPolicy>(residual_policy), cfg);
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_power_strategy(const PowerConfig<Scalar>& cfg = {}) {
  return PowerStrategy(L2Residual{}, cfg);
}
//
template <class QVector, class ZVector, class RVector>
KOKKOS_INLINE_FUNCTION auto make_power_state(QVector&& q, ZVector&& z, RVector&& r) {
  return PowerState(std::forward<QVector>(q), std::forward<ZVector>(z), std::forward<RVector>(r));
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_lanczos_strategy(const LanczosConfig<Scalar>& cfg = {}) {
  return LanczosStrategy(cfg);
}
//
template <class Scalar, class Precond>
KOKKOS_INLINE_FUNCTION auto make_lanczos_strategy(const LanczosConfig<Scalar>& cfg, Precond&& precond) {
  return LanczosStrategy(cfg, std::forward<Precond>(precond));
}
//
/// \brief A Lanczos state without estimate vectors, over a scalar array of at least 4 max_iters scalars.
///
/// Bv is NoVector without a preconditioner.
template <class VVector, class BvPrevVector, class AvVector, class BvVector, class ScalarArray>
KOKKOS_INLINE_FUNCTION auto make_lanczos_state(VVector&& v, BvPrevVector&& Bv_prev, AvVector&& Av, BvVector&& Bv,
                                               ScalarArray&& scalars) {
  return LanczosState(std::forward<VVector>(v), std::forward<BvPrevVector>(Bv_prev), std::forward<AvVector>(Av),
                      std::forward<BvVector>(Bv), std::forward<ScalarArray>(scalars), NoVector{}, NoVector{},
                      NoVector{});
}
//
/// \brief A Lanczos state with an estimate vector, or NoVector, for each end, and the basis that forms them.
///
/// The scalar array holds at least (4 + e) max_iters scalars, for e the number of estimate vectors. Bv is NoVector
/// without a preconditioner.
template <class VVector, class BvPrevVector, class AvVector, class BvVector, class ScalarArray, class LargestVector,
          class SmallestVector, class Basis>
KOKKOS_INLINE_FUNCTION auto make_lanczos_state(VVector&& v, BvPrevVector&& Bv_prev, AvVector&& Av, BvVector&& Bv,
                                               ScalarArray&& scalars, LargestVector&& largest_estimate,
                                               SmallestVector&& smallest_estimate, Basis&& basis) {
  return LanczosState(std::forward<VVector>(v), std::forward<BvPrevVector>(Bv_prev), std::forward<AvVector>(Av),
                      std::forward<BvVector>(Bv), std::forward<ScalarArray>(scalars),
                      std::forward<LargestVector>(largest_estimate), std::forward<SmallestVector>(smallest_estimate),
                      std::forward<Basis>(basis));
}
//@}

/// \brief Solve the eigenvalue problem prob from state with strat.
///
/// What the problem's workspace holds on return, and what result.residual measures, are stated by the strategy.
template <class Problem, class Strategy, class State>
MUNDY_REQUIRES(requires(const Strategy& s, const Problem& prob, State& state) {
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.iterate(prob, state) } -> std::same_as<bool>;
  { s.done(state) } -> std::same_as<bool>;
  s.result(state);
})
KOKKOS_FUNCTION auto solve_eigen_problem(const Problem& prob, const Strategy& strat, State& state) {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  return strat.result(state);
}

/// \brief Both ends of the spectrum of a symmetric A via the shifted power method.
///
/// With lambda_d the dominant eigenvalue, the dominant eigenvalue of A - lambda_d I is lambda_opp - lambda_d, where
/// lambda_opp is the opposite end of the spectrum, so a second power iteration on the shifted operator recovers it.
/// Each state's q() is its start vector on entry and its unit eigenvector on exit. The problem's workspace holds the
/// dominant solve's intermediates, at its returned q; the shifted solve runs on a workspace of its own.
template <class Problem, class Strategy, class DominantState, class OppositeState>
KOKKOS_FUNCTION auto solve_eigen_bounds(const Problem& prob, const Strategy& strat, DominantState& dominant_state,
                                        OppositeState& opposite_state) {
  using backend_t = decltype(prob.backend());
  using value_type = typename Strategy::value_type;

  const auto dominant = solve_eigen_problem(prob, strat, dominant_state);
  MUNDY_THROW_REQUIRE(dominant.converged, std::runtime_error,
                      "solve_eigen_bounds: the dominant eigenvalue, which sets the shift, did not converge.");

  const auto shifted_prob = make_eigen_problem<backend_t>(make_shifted_op<backend_t>(dominant.eigenvalue, prob.A()));
  auto opposite = solve_eigen_problem(shifted_prob, strat, opposite_state);
  opposite.eigenvalue += dominant.eigenvalue;

  // a nonnegative dominant eigenvalue is the largest
  if (dominant.eigenvalue >= static_cast<value_type>(0)) {
    return EigenBounds<value_type>{dominant, opposite};
  }
  return EigenBounds<value_type>{opposite, dominant};
}

}  // namespace mundy

#endif  // MUNDY_MATH_EIGENVALUES_HPP_
