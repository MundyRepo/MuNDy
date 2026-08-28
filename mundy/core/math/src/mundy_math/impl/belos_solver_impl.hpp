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

#ifndef MUNDY_MATH_IMPL_BELOS_SOLVER_IMPL_HPP_
#define MUNDY_MATH_IMPL_BELOS_SOLVER_IMPL_HPP_

/// \file belos_solver_impl.hpp
/// \brief Tpetra/Belos/Teuchos machinery behind belos_solver.hpp: a matrix-free Tpetra::Operator adapter and a
/// low-level solve that consumes an already-translated solver name plus parameter list. This header depends on no
/// public belos_solver.hpp types, so belos_solver.hpp includes it up top with its other includes.

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{BELOS,TPETRA}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

// Tpetra / Teuchos:
#include <Teuchos_DefaultSerialComm.hpp>  // for Teuchos::SerialComm (single-rank, no MPI)
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

// Belos:
#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosSolverManager.hpp>
#include <BelosTpetraAdapter.hpp>  // MultiVecTraits / OperatorTraits specializations for Tpetra
#include <BelosTypes.hpp>

// Mundy:
#include <mundy_math/solver_backends.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace impl {

/// \brief Sentinel meaning "no preconditioner" for the optional preconditioner argument.
struct NoPreconditioner {};

/// \brief Plain solve statistics returned by belos_solve_raw (kept free of the public config/result types).
struct BelosSolveStats {
  int num_iters{0};
  double achieved_tol{0.0};
  bool converged{false};
};

/// \brief Exposes a Mundy matrix-free operator to Belos as a Tpetra::Operator.
///
/// apply() forwards each MultiVector column to Backend::apply as Y = alpha (op X) + beta Y (NO_TRANS only), through
/// a persistent operator workspace. The wrapped operator's apply must be generic over the vector type so it accepts
/// the Tpetra device view types.
template <class Backend, class Op, class Scalar, class LO, class GO, class NO>
class MundyTpetraOperator : public Tpetra::Operator<Scalar, LO, GO, NO> {
 public:
  using map_type = Tpetra::Map<LO, GO, NO>;
  using mv_type = Tpetra::MultiVector<Scalar, LO, GO, NO>;

  MundyTpetraOperator(const Op& op, const Teuchos::RCP<const map_type>& map)
      : op_(op), map_(map), workspace_(Backend::make_workspace(op_)) {
    MUNDY_THROW_ASSERT(Backend::domain_size(op_) == map_->getLocalNumElements() &&
                           Backend::range_size(op_) == map_->getLocalNumElements(),
                       std::invalid_argument, "MundyTpetraOperator: operator domain/range size must match the map.");
  }

  Teuchos::RCP<const map_type> getDomainMap() const override {
    return map_;
  }
  Teuchos::RCP<const map_type> getRangeMap() const override {
    return map_;
  }

  void apply(const mv_type& X, mv_type& Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
             Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
             Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const override {
    MUNDY_THROW_REQUIRE(mode == Teuchos::NO_TRANS, std::logic_error,
                        "MundyTpetraOperator: only NO_TRANS apply is supported.");
    const Scalar one = Teuchos::ScalarTraits<Scalar>::one();
    const Scalar zero = Teuchos::ScalarTraits<Scalar>::zero();
    auto x2d = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    auto y2d = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
    const size_t ncols = X.getNumVectors();
    for (size_t j = 0; j < ncols; ++j) {
      auto xj = Kokkos::subview(x2d, Kokkos::ALL(), j);
      auto yj = Kokkos::subview(y2d, Kokkos::ALL(), j);
      if (alpha == one && beta == zero) {
        Backend::apply(op_, xj, yj, workspace_);  // Y = A x
      } else {
        Backend::apply(alpha, op_, xj, beta, yj, workspace_);  // Y = alpha (A x) + beta Y
      }
    }
  }

  bool hasTransposeApply() const override {
    return false;
  }

 private:
  Op op_;
  Teuchos::RCP<const map_type> map_;
  mutable decltype(Backend::make_workspace(std::declval<const Op&>())) workspace_;
};

/// \brief A contiguous, single-process Map of \p n rows over a SerialComm -- fully local, no MPI.
template <class LO, class GO, class NO>
Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> make_serial_map(size_t n) {
  const Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::SerialComm<int>());
  const auto n_global = static_cast<Tpetra::global_size_t>(n);
  const GO index_base = 0;
  return Teuchos::rcp(new Tpetra::Map<LO, GO, NO>(n_global, index_base, comm));
}

/// \brief Copy a rank-1 Kokkos View into column 0 of a single-column MultiVector (device-to-device).
template <class MvType, class SrcView>
void load_view_into_mv(const SrcView& src, MvType& mv) {
  auto dev = mv.getLocalViewDevice(Tpetra::Access::OverwriteAll);
  auto col0 = Kokkos::subview(dev, Kokkos::ALL(), 0);
  Kokkos::deep_copy(col0, src);
}

/// \brief Copy column 0 of a single-column MultiVector into a rank-1 Kokkos View (device-to-device).
template <class MvType, class DstView>
void extract_mv_into_view(MvType& mv, DstView& dst) {
  auto dev = mv.getLocalViewDevice(Tpetra::Access::ReadOnly);
  auto col0 = Kokkos::subview(dev, Kokkos::ALL(), 0);
  Kokkos::deep_copy(dst, col0);
}

/// \brief A reusable Belos solve for a fixed operator, preconditioner, and configuration.
///
/// Owns the Tpetra Map, the right-hand-side and solution MultiVectors, the operator (and optional preconditioner)
/// adapters, the Belos LinearProblem, and the SolverManager -- all built once at construction. solve() overwrites
/// the right-hand-side and initial-guess data and re-solves, so repeated solves of the same operator reuse the
/// Krylov machinery (and a GCRO-DR recycle subspace) rather than rebuilding it. Scalar is the operator's value type.
template <class Backend, class Op, class Scalar, class Precond>
  requires requires { typename Backend::exec_space; }
class BelosSolveSession {
 public:
  using LO = typename Tpetra::Map<>::local_ordinal_type;
  using GO = typename Tpetra::Map<>::global_ordinal_type;
  using NO = typename Tpetra::Map<>::node_type;
  using map_type = Tpetra::Map<LO, GO, NO>;
  using mv_type = Tpetra::MultiVector<Scalar, LO, GO, NO>;
  using op_type = Tpetra::Operator<Scalar, LO, GO, NO>;
  using problem_type = Belos::LinearProblem<Scalar, mv_type, op_type>;
  using solver_type = Belos::SolverManager<Scalar, mv_type, op_type>;

  static_assert(std::is_same_v<typename Backend::exec_space, typename NO::execution_space>,
                "belos solve: Backend::exec_space must match the Tpetra default Node execution space (build "
                "Trilinos and Mundy against the same Kokkos execution space).");

  BelosSolveSession(const Op& op, size_t n, const std::string& solver_name,
                    const Teuchos::RCP<Teuchos::ParameterList>& params, const Precond& precond)
      : map_(make_serial_map<LO, GO, NO>(n)),
        b_(Teuchos::rcp(new mv_type(map_, static_cast<size_t>(1)))),
        x_(Teuchos::rcp(new mv_type(map_, static_cast<size_t>(1)))),
        a_op_(Teuchos::rcp(new MundyTpetraOperator<Backend, Op, Scalar, LO, GO, NO>(op, map_))),
        problem_(Teuchos::rcp(new problem_type(a_op_, x_, b_))) {
    if constexpr (!std::is_same_v<Precond, NoPreconditioner>) {
      // The preconditioner operator's apply() computes the approximate-inverse action; set as a right preconditioner.
      m_op_ = Teuchos::rcp(new MundyTpetraOperator<Backend, Precond, Scalar, LO, GO, NO>(precond, map_));
      problem_->setRightPrec(m_op_);
    }
    Belos::SolverFactory<Scalar, mv_type, op_type> factory;
    solver_ = factory.create(solver_name, params);
  }

  template <class BView, class XView>
  BelosSolveStats solve(const BView& b, XView& x) {
    MUNDY_THROW_REQUIRE(b.extent(0) == map_->getLocalNumElements() && x.extent(0) == map_->getLocalNumElements(),
                        std::invalid_argument, "belos solve: b and x lengths must match the operator size.");
    load_view_into_mv(b, *b_);
    load_view_into_mv(x, *x_);  // initial guess (cold vs warm start is the caller's choice)
    const bool set_ok = problem_->setProblem();
    MUNDY_THROW_REQUIRE(set_ok, std::runtime_error, "belos solve: Belos::LinearProblem::setProblem() failed.");
    solver_->setProblem(problem_);
    const Belos::ReturnType ret = solver_->solve();
    extract_mv_into_view(*x_, x);
    return BelosSolveStats{static_cast<int>(solver_->getNumIters()), static_cast<double>(solver_->achievedTol()),
                           ret == Belos::Converged};
  }

 private:
  Teuchos::RCP<const map_type> map_;
  Teuchos::RCP<mv_type> b_;
  Teuchos::RCP<mv_type> x_;
  Teuchos::RCP<op_type> a_op_;
  Teuchos::RCP<op_type> m_op_;
  Teuchos::RCP<problem_type> problem_;
  Teuchos::RCP<solver_type> solver_;
};

}  // namespace impl

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA

#endif  // MUNDY_MATH_IMPL_BELOS_SOLVER_IMPL_HPP_
