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

#ifndef MUNDY_MATH_MUELU_PRECONDITIONER_HPP_
#define MUNDY_MATH_MUELU_PRECONDITIONER_HPP_

/// \file muelu_preconditioner.hpp
/// \brief Algebraic multigrid preconditioning of sparse symmetric positive definite matrices, backed by Trilinos MueLu.
///
/// The preconditioner of a sparse matrix A applies one multigrid cycle to A z = r. It is a host-side operator that
/// launches device kernels; it is not device-callable.
///
/// This header is a no-op unless the MueLu, Tpetra and KokkosKernels TPLs are enabled (HAVE_MUNDYMATH_MUELU,
/// HAVE_MUNDYMATH_TPETRA and HAVE_MUNDYMATH_KOKKOSKERNELS).

#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{MUELU,TPETRA,KOKKOSKERNELS}

#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <cstddef>
#include <iostream>
#include <string>

// Teuchos (only the two types the public escape hatch exposes):
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

// Mundy
#include <mundy_math/impl/muelu_preconditioner_impl.hpp>  // for the (self-contained) Tpetra/MueLu machinery
#include <mundy_math/solver_backends.hpp>                 // for the Backend contract

namespace mundy {

//! \name Configuration
//@{

/// \brief The smoother each level applies before and after its coarse correction.
enum class MueLuSmoother {
  JACOBI,                  ///< Jacobi relaxation; the default.
  SYMMETRIC_GAUSS_SEIDEL,  ///< symmetric Gauss-Seidel relaxation.
  CHEBYSHEV,               ///< a Chebyshev polynomial in A.
};

/// \brief How many coarse corrections a level makes per cycle: one (V) or two (W).
enum class MueLuCycle { V, W };

/// \brief How the coarsest level is solved: exactly (DIRECT) or by the smoother (SMOOTHER).
enum class MueLuCoarseSolver { DIRECT, SMOOTHER };

/// \brief Configuration for a MueLu preconditioner.
///
/// The typed fields cover the common case and default to MueLu's own defaults. For anything they do not expose,
/// \c extra is a Teuchos::ParameterList of MueLu parameters -- keyed by MueLu's own parameter names (e.g.
/// "aggregation: drop tol") -- that is merged in last, so its entries override the typed fields.
template <typename Scalar>
struct MueLuConfig {
  using value_type = Scalar;

  MueLuSmoother smoother{MueLuSmoother::JACOBI};
  unsigned smoother_sweeps{1};  ///< sweeps of a relaxation smoother, or the degree of a Chebyshev one
  MueLuCycle cycle{MueLuCycle::V};
  unsigned max_levels{10};
  unsigned coarse_max_size{2000};  ///< the coarsest level has at most this many unknowns
  MueLuCoarseSolver coarse_solver{MueLuCoarseSolver::DIRECT};

  /// Aggregate identically on every run, so the same A gives the same preconditioner to rounding.
  bool deterministic{false};

  /// Advanced: extra MueLu parameters, merged after (and overriding) the typed fields above. Null by default.
  Teuchos::RCP<Teuchos::ParameterList> extra{};
};
//@}

namespace impl {

/// \brief Translate a MueLuConfig to a parameter list; \c cfg.extra overrides the typed fields.
template <class Scalar>
Teuchos::ParameterList make_parameter_list(const MueLuConfig<Scalar>& cfg) {
  Teuchos::ParameterList params;
  params.set("verbosity", std::string("none"));
  params.set("problem: symmetric", true);
  params.set("multigrid algorithm", std::string("sa"));
  params.set("max levels", static_cast<int>(cfg.max_levels));
  params.set("coarse: max size", static_cast<int>(cfg.coarse_max_size));
  params.set("cycle type", std::string(cfg.cycle == MueLuCycle::V ? "V" : "W"));
  params.set("aggregation: deterministic", cfg.deterministic);
  params.set("use kokkos refactor", false);

  Teuchos::ParameterList smoother_params;
  std::string smoother_type = "RELAXATION";
  switch (cfg.smoother) {
    case MueLuSmoother::JACOBI:
      smoother_params.set("relaxation: type", std::string("Jacobi"));
      smoother_params.set("relaxation: sweeps", static_cast<int>(cfg.smoother_sweeps));
      break;
    case MueLuSmoother::SYMMETRIC_GAUSS_SEIDEL:
      smoother_params.set("relaxation: type", std::string("Symmetric Gauss-Seidel"));
      smoother_params.set("relaxation: sweeps", static_cast<int>(cfg.smoother_sweeps));
      break;
    case MueLuSmoother::CHEBYSHEV:
      smoother_type = "CHEBYSHEV";
      smoother_params.set("chebyshev: degree", static_cast<int>(cfg.smoother_sweeps));
      break;
  }
  params.set("smoother: type", smoother_type);
  params.sublist("smoother: params").setParameters(smoother_params);

  if (cfg.coarse_solver == MueLuCoarseSolver::DIRECT) {
    params.set("coarse: type", std::string("KLU"));
  } else {
    params.set("coarse: type", smoother_type);
    params.sublist("coarse: params").setParameters(smoother_params);
  }

  if (Teuchos::nonnull(cfg.extra)) {
    params.setParameters(*cfg.extra);
  }
  if (params.get<bool>("use kokkos refactor")) {
    std::cerr << "Warning: MueLu preconditioner: \"use kokkos refactor\" is true. As of Trilinos 16.1, MueLu's Kokkos "
                 "tentative prolongator is wrong for two or more near-null vectors. Solutions remain correct, but "
                 "performance and determinism are degraded."
              << std::endl;
  }
  return params;
}

}  // namespace impl

//! \name Preconditioner
//@{

/// \brief The multigrid preconditioner z = P r of a sparse symmetric positive definite matrix A.
///
/// P applies one multigrid cycle to A z = r from z = 0. It is symmetric, and positive definite whenever its smoother
/// converges on A; P then preconditions CG. The near-null space, the vectors on which A is small relative to their
/// length, defaults to the constant vector. With deterministic aggregation, the same A, near-null space and
/// configuration give the same P to rounding; on more than one thread, setup sums in an order that varies from run to
/// run, so P is not reproducible bit for bit.
///
/// P is that of A's values at construction or at the last update(A); A must not change otherwise. A new sparsity
/// pattern needs a new preconditioner. Copies share P and its storage, so a preconditioner and its copies must not be
/// applied concurrently. Host-only.
template <typename Backend, typename Matrix>
class MueLuPreconditioner {
 public:
  using backend_t = Backend;
  using value_type = typename Matrix::non_const_value_type;
  using config_t = MueLuConfig<value_type>;
  using vector_t = Kokkos::View<value_type*, typename Matrix::memory_space>;

  MueLuPreconditioner(Backend, const Matrix& A, const config_t& cfg) : session_(A, impl::make_parameter_list(cfg)) {
  }

  template <class NearNullSpace>
  MueLuPreconditioner(Backend, const Matrix& A, const NearNullSpace& near_null_space, const config_t& cfg)
      : session_(A, near_null_space, impl::make_parameter_list(cfg)) {
  }

  /// \brief P for A's current values; A's sparsity pattern is the one P was made with.
  void update(const Matrix& A) {
    session_.update(A);
  }

  /// \brief P for A's current values and a new near-null space; A's sparsity pattern is the one P was made with.
  template <class NearNullSpace>
  void update(const Matrix& A, const NearNullSpace& near_null_space) {
    session_.update(A, near_null_space);
  }

  // clang-format off
  Backend backend() const { return Backend{}; }
  size_t domain_size() const { return session_.size(); }
  size_t range_size() const { return session_.size(); }
  auto make_domain_vector() const { return Backend::template make_vector<vector_t>(domain_size()); }
  auto make_range_vector() const { return Backend::template make_vector<vector_t>(range_size()); }
  // clang-format on

  /// z := P r.
  template <class RVector, class ZVector>
  void apply(const RVector& r, ZVector& z) const {
    session_.apply(r, z);
  }

 private:
  impl::MueLuSession<Backend, Matrix> session_;
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <class Backend, class Matrix, class Scalar>
MueLuPreconditioner(Backend, const Matrix&, const MueLuConfig<Scalar>&) -> MueLuPreconditioner<Backend, Matrix>;

template <class Backend, class Matrix, class NearNullSpace, class Scalar>
MueLuPreconditioner(Backend, const Matrix&, const NearNullSpace&,
                    const MueLuConfig<Scalar>&) -> MueLuPreconditioner<Backend, Matrix>;
#endif  // DOXYGEN_SHOULD_SKIP_THIS

/// \brief The MueLu preconditioner of A.
template <class Backend, class Matrix, class Scalar>
auto make_muelu_preconditioner(const Matrix& A, const MueLuConfig<Scalar>& cfg) {
  return MueLuPreconditioner(Backend{}, A, cfg);
}

/// \brief The MueLu preconditioner of A with the given near-null space, an n x k rank-2 view.
template <class Backend, class Matrix, class NearNullSpace, class Scalar>
auto make_muelu_preconditioner(const Matrix& A, const NearNullSpace& near_null_space, const MueLuConfig<Scalar>& cfg) {
  return MueLuPreconditioner(Backend{}, A, near_null_space, cfg);
}
//@}

}  // namespace mundy

#endif  // HAVE_MUNDYMATH_MUELU && HAVE_MUNDYMATH_TPETRA && HAVE_MUNDYMATH_KOKKOSKERNELS

#endif  // MUNDY_MATH_MUELU_PRECONDITIONER_HPP_
