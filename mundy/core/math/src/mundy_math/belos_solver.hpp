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

#ifndef MUNDY_MATH_BELOS_SOLVER_HPP_
#define MUNDY_MATH_BELOS_SOLVER_HPP_

/// \file belos_solver.hpp
/// \brief Host-orchestrated iterative linear solves backed by Trilinos Belos.
///
/// Solves A x = b for a square (not necessarily symmetric) matrix-free operator A over Kokkos Views on a single
/// process, with no MPI. A is any Mundy LinearOperator (apply(x, y)); the right-hand side and solution are the same
/// Kokkos Views the CG path uses. This is a host-side entry point that launches device kernels; it is not
/// device-callable, and it serves the single-large-system case rather than the on-device batched case that
/// MundyMathBackend + CG serves.
///
/// This header is a no-op unless both the Belos and Tpetra TPLs are enabled (HAVE_MUNDYMATH_BELOS and
/// HAVE_MUNDYMATH_TPETRA).

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{BELOS,TPETRA}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

// Teuchos (only the two types the public escape hatch exposes):
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

// Mundy
#include <mundy_math/Tolerance.hpp>                // for mundy::get_relaxed_zero_tolerance<T>
#include <mundy_math/impl/belos_solver_impl.hpp>   // for the (self-contained) Tpetra/Belos machinery
#include <mundy_math/linear_system.hpp>            // for mundy::LinearSystem
#include <mundy_math/solver_backends.hpp>          // for the Backend contract
#include <mundy_utils/storage.hpp>                 // for value-or-reference storage
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

//! \name Solver selection
//@{

/// \brief Which Belos Krylov solver to run. The choice is a runtime field; all share one solve entry point. Each
/// enumerator is named for the Belos solver manager it maps to.
enum class BelosSolver {
  PSEUDOBLOCK_GMRES,  ///< restarted GMRES (pseudo-block); the default, non-symmetric.
  BICGSTAB,           ///< BiCGStab; non-symmetric, no restart storage.
  TFQMR,              ///< transpose-free QMR; non-symmetric.
  MINRES,             ///< MINRES; symmetric indefinite.
  PSEUDOBLOCK_CG,     ///< CG; symmetric positive definite.
  GCRODR,             ///< GCRO-DR; GMRES with subspace recycling across solves.
};
//@}

//! \name Solve result
//@{

/// \brief Result of a Belos solve: iteration count, achieved residual, and whether it converged. Same shape as
/// CGResult so results are uniform across the solver family.
template <class Scalar>
struct BelosResult {
  using value_type = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};  ///< achieved tolerance (relative residual by default).
  bool converged{false};
};

/// \brief Write a BelosResult to an ostream.
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const BelosResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual
     << ", converged?: " << result.converged;
  return os;
}
//@}

//! \name Solver configuration
//@{

/// \brief Configuration for a Belos solve.
///
/// The typed fields cover the common case and are translated to Belos solver parameters. For anything they do not
/// expose, \c extra is a Teuchos::ParameterList of Belos parameters -- keyed by Belos's own parameter names (e.g.
/// "Num Recycled Blocks" for GCRO-DR) -- that is merged in last, so its entries override the typed fields.
template <typename Scalar>
struct BelosConfig {
  using value_type = Scalar;

  BelosSolver solver{BelosSolver::PSEUDOBLOCK_GMRES};
  unsigned max_iters{200};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};  ///< convergence tolerance (relative residual by default)
  unsigned num_blocks{30};                            ///< Krylov subspace / restart length (GMRES, GCRO-DR)
  unsigned max_restarts{20};                          ///< max restarts (GMRES, GCRO-DR)
  int verbosity{1};                                   ///< errors + warnings

  /// Advanced: extra Belos parameters, merged after (and overriding) the typed fields above. Null by default.
  Teuchos::RCP<Teuchos::ParameterList> extra{};
};
//@}

namespace impl {

/// \brief The Belos solver-manager name for a BelosSolver enumerator (the enumerator's own name, spaced).
inline std::string solver_name_string(BelosSolver s) {
  switch (s) {
    case BelosSolver::PSEUDOBLOCK_GMRES: return "PSEUDOBLOCK GMRES";
    case BelosSolver::BICGSTAB: return "BICGSTAB";
    case BelosSolver::TFQMR: return "TFQMR";
    case BelosSolver::MINRES: return "MINRES";
    case BelosSolver::PSEUDOBLOCK_CG: return "PSEUDOBLOCK CG";
    case BelosSolver::GCRODR: return "GCRODR";
  }
  MUNDY_THROW_REQUIRE(false, std::logic_error, "solver_name_string: unhandled BelosSolver enumerator.");
  return {};  // unreachable; the switch above is exhaustive.
}

/// \brief Translate a BelosConfig to a parameter list; \c cfg.extra overrides the typed fields.
template <class Scalar>
Teuchos::RCP<Teuchos::ParameterList> make_parameter_list(const BelosConfig<Scalar>& cfg) {
  auto pl = Teuchos::rcp(new Teuchos::ParameterList());
  pl->set("Maximum Iterations", static_cast<int>(cfg.max_iters));
  pl->set("Convergence Tolerance", cfg.tol);
  pl->set("Verbosity", cfg.verbosity);
  if (cfg.solver == BelosSolver::PSEUDOBLOCK_GMRES || cfg.solver == BelosSolver::GCRODR) {
    pl->set("Num Blocks", static_cast<int>(cfg.num_blocks));
    pl->set("Maximum Restarts", static_cast<int>(cfg.max_restarts));
  }
  if (Teuchos::nonnull(cfg.extra)) {
    pl->setParameters(*cfg.extra);
  }
  return pl;
}

}  // namespace impl

//! \name Solve
//@{

/// \brief Solve A x = b for a square operator A via Belos, returning a BelosResult.
///
/// \p prob is a LinearSystem (operator + right-hand side). \p x is the initial guess on entry (the caller decides
/// cold vs warm start) and the solution on exit. \p precond is an optional preconditioner: any Mundy matrix-free
/// operator whose apply() computes the approximate-inverse action (e.g. a Jacobi diagonal); it is applied as a
/// right preconditioner.
///
/// On non-convergence this returns a result with converged == false and leaves the best iterate in \p x; it does
/// not throw. Requires a host-orchestrating backend (KokkosBackend); it is not valid with MundyMathBackend.
template <class Problem, class XVector, class Precond = impl::NoPreconditioner>
auto belos_solve(const Problem& prob, XVector& x, const BelosConfig<typename Problem::value_type>& cfg,
                 const Precond& precond = {}) -> BelosResult<typename Problem::value_type> {
  using value_type = typename Problem::value_type;
  using backend_t = typename Problem::backend_t;
  using op_t = std::remove_cvref_t<decltype(prob.A())>;
  static_assert(requires { typename backend_t::exec_space; },
                "belos_solve requires a host-orchestrating KokkosBackend (Backend::exec_space); it cannot be used "
                "with MundyMathBackend.");
  impl::BelosSolveSession<backend_t, op_t, value_type, Precond> session(
      prob.A(), prob.b().extent(0), impl::solver_name_string(cfg.solver), impl::make_parameter_list(cfg), precond);
  const impl::BelosSolveStats stats = session.solve(prob.b(), x);
  return BelosResult<value_type>{static_cast<unsigned>(stats.num_iters),
                                 static_cast<value_type>(stats.achieved_tol), stats.converged};
}
//@}

//! \name Inverse operator
//@{
// One layer above belos_solve: an operator whose apply() is itself a Belos solve, so a (possibly non-symmetric)
// operator's inverse can slot into operator algebra wherever a LinearOperator is expected.

/// \brief Wraps a square operator as its inverse: apply(rhs, out) solves op * out = rhs via matrix-free Belos.
///
/// The Belos solve (Map, vectors, problem, and solver) is built once at construction and reused across apply()
/// calls, which only swap the right-hand side. The solution buffer starts zeroed, so the first apply is a cold
/// start even when \p warm_start is set; thereafter warm_start reuses the previous solution as the initial guess.
/// \p Precond, when not NoPreconditioner, is applied as a right preconditioner (its apply() computes the
/// approximate-inverse action). apply() throws std::runtime_error if the inner solve does not converge. Host-only.
template <typename Backend, typename Op, typename Precond = impl::NoPreconditioner>
class BelosInvOp {
 public:
  using backend_t = Backend;
  using x_vector_t = decltype(Backend::make_domain_vector(std::declval<const Op&>()));
  using value_type = impl::vector_value_type<x_vector_t>;
  using config_t = BelosConfig<value_type>;

  BelosInvOp(Backend, Op&& op, const config_t& cfg, Precond&& precond, bool warm_start = false)
      : op_storage_(std::forward<Op>(op)),
        warm_start_(warm_start),
        x_(Backend::make_domain_vector(op_storage_.get())),
        session_(op_storage_.get(), Backend::domain_size(op_storage_.get()), impl::solver_name_string(cfg.solver),
                 impl::make_parameter_list(cfg), std::forward<Precond>(precond)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(op_storage_.get()) == Backend::range_size(op_storage_.get()),
                       std::invalid_argument, "BelosInvOp: operator must be square.");
    Backend::deep_copy(x_, value_type(0));  // defined initial guess, so a warm first apply is a cold start
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  size_t domain_size() const { return Backend::domain_size(op_storage_.get()); }
  size_t range_size() const { return Backend::range_size(op_storage_.get()); }
  auto make_domain_vector() const { return Backend::make_domain_vector(op_storage_.get()); }
  auto make_range_vector() const { return Backend::make_range_vector(op_storage_.get()); }
  // clang-format on

  /// out := op^{-1} rhs, via Belos.
  template <class RhsVector, class OutVector>
  void apply(const RhsVector& rhs, OutVector& out) const {
    constexpr value_type zero = static_cast<value_type>(0);
    if (!warm_start_) {
      Backend::axpby(zero, x_, zero, x_);  // cold start: x0 = 0
    }
    // else: leave x_ at whatever the previous solve left (warm start).

    const impl::BelosSolveStats stats = session_.solve(rhs, x_);
    last_result_ = BelosResult<value_type>{static_cast<unsigned>(stats.num_iters),
                                           static_cast<value_type>(stats.achieved_tol), stats.converged};

    // A non-converged solve would silently return a wrong answer, so throw instead of returning it.
    MUNDY_THROW_REQUIRE(last_result_.converged, std::runtime_error, "BelosInvOp: inner Belos solve failed to converge.");
    Backend::deep_copy(out, x_);
  }

  const BelosResult<value_type>& last_result() const {
    return last_result_;
  }

 private:
  using session_t = impl::BelosSolveSession<Backend, std::remove_cvref_t<Op>, value_type, std::remove_cvref_t<Precond>>;

  ::mundy::storage<Op> op_storage_;
  bool warm_start_;
  mutable x_vector_t x_;
  mutable session_t session_;
  mutable BelosResult<value_type> last_result_{};
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <class Backend, class Op, class Scalar, class Precond>
BelosInvOp(Backend, Op&&, const BelosConfig<Scalar>&, Precond&&, bool) -> BelosInvOp<Backend, Op, Precond>;
#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \brief Build a BelosInvOp. \p precond defaults to no preconditioner.
template <class Backend, class Op, class Scalar, class Precond = impl::NoPreconditioner>
auto make_belos_inv_op(Op&& op, const BelosConfig<Scalar>& cfg, Precond&& precond = {}, bool warm_start = false) {
  return BelosInvOp(Backend{}, std::forward<Op>(op), cfg, std::forward<Precond>(precond), warm_start);
}
//@}

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA

#endif  // MUNDY_MATH_BELOS_SOLVER_HPP_
