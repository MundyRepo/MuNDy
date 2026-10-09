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

#ifndef MUNDY_MBODY_KOKKOSMBODYPRECONDITIONERS_HPP_
#define MUNDY_MBODY_KOKKOSMBODYPRECONDITIONERS_HPP_

/// \file
/// \brief Schur-complement preconditioner policies: how the CG behind the bilateral block's Schur complement is
/// preconditioned.

// C++ core
#include <cstddef>    // for size_t
#include <optional>   // for std::optional
#include <stdexcept>  // for std::logic_error
#include <utility>    // for std::declval, std::in_place

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <MundyMath_config.hpp>                 // for HAVE_MUNDYMATH_{MUELU,TPETRA,KOKKOSKERNELS}
#include <mundy_math/Matrix.hpp>                // for mundy::Matrix
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3d
#include <mundy_math/preconditioners.hpp>       // for mundy::{Preconditioner, JacobiPreconditioner}
#include <mundy_math/solver_backends.hpp>       // for mundy::KokkosBackend
#include <mundy_mbody/KokkosMbodyMobility.hpp>  // for mundy::mbody::HasSelfMobility

#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
#include <KokkosSparse_CrsMatrix.hpp>                           // for KokkosSparse::CrsMatrix
#include <mundy_math/muelu_preconditioner.hpp>                  // for mundy::{MueLuConfig, MueLuPreconditioner}
#include <mundy_mbody/impl/KokkosMbodyPreconditionersImpl.hpp>  // for mundy::mbody::impl::{make_row_rods, ...}
#include <mundy_utils/host_ptr.hpp>                             // for mundy::host_ptr
#include <mundy_utils/throw_assert.hpp>                         // for MUNDY_THROW_ASSERT
#endif

namespace mundy {

namespace mbody {

//! \name Schur-complement preconditioner policies
//@{
// The bilateral block's Schur complement S = (dt B^T M B + K^{-1})^{-1} is applied by CG. A preconditioner policy
// preconditions that CG: its make_preconditioner(linearization) makes a preconditioner op once per step, the solver
// calls the op's update(linearization) before it solves at each linearization, and CG otherwise only applies it. The
// default, NoPreconditioner, leaves the CG unpreconditioned.
//
// A linearization is read-only and provides:
//   num_rows(), dt(), mobility() (M at the step's start configuration), compliance() (K^{-1} per row, a view);
//   jacobian(), a device-callable row accessor: num_bodies(row) <= 3, and for k below it body(row, k), force(row, k)
//     and torque(row, k), the force and torque that a unit multiplier on the row exerts on that body;
//   apply(x, y), y = (dt B^T M B + K^{-1}) x.

/// \brief The preconditioner op Policy makes from a Linearization.
template <class Policy, class Linearization>
using schur_preconditioner_t =
    decltype(std::declval<const Policy&>().make_preconditioner(std::declval<const Linearization&>()));

/// \brief Whether Policy makes, from a Linearization, a preconditioner op for its rows that update(linearization)
/// refreshes.
template <class Policy, class Linearization>
concept SchurPreconditionerPolicy =
    requires { typename schur_preconditioner_t<Policy, Linearization>; } &&
    requires(schur_preconditioner_t<Policy, Linearization>& preconditioner, const Linearization& linearization) {
      preconditioner.update(linearization);
    } &&
    ::mundy::Preconditioner<schur_preconditioner_t<Policy, Linearization>,
                            ::mundy::KokkosBackend<typename Linearization::execution_space>,
                            typename Linearization::view_t>;

/// \brief The Jacobi preconditioner z = r ./ d of dt B^T M B + K^{-1}, with d formed from M's self blocks.
///
/// d_i = dt sum_k v_ik^T M_{b_ik} v_ik + K^{-1}_i, for b_ik the k-th body of row i and v_ik = [force(i, k); torque(i,
/// k)]. It is the operator's exact diagonal when M couples no two rods, and otherwise its diagonal with M's coupling
/// between rods dropped.
template <typename ExecSpace>
class SelfMobilityJacobiOp {
 public:
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;
  using backend_t = ::mundy::KokkosBackend<ExecSpace>;

  explicit SelfMobilityJacobiOp(size_t num_rows)
      : diagonal_("self_mobility_jacobi_diagonal", num_rows), jacobi_(backend_t{}, view_t(diagonal_)) {
  }

  /// \brief d at the last update.
  const view_t& diagonal() const {
    return diagonal_;
  }

  size_t domain_size() const {
    return jacobi_.domain_size();
  }
  size_t range_size() const {
    return jacobi_.range_size();
  }
  auto make_domain_vector() const {
    return jacobi_.make_domain_vector();
  }
  auto make_range_vector() const {
    return jacobi_.make_range_vector();
  }

  template <class RVector, class ZVector>
  void apply(const RVector& r, ZVector& z) const {
    jacobi_.apply(r, z);
  }

  /// \brief Form d at linearization.
  template <class Linearization>
  void update(const Linearization& linearization) {
    const auto jacobian = linearization.jacobian();
    const auto mobility = linearization.mobility();
    const view_t compliance = linearization.compliance();
    const double dt = linearization.dt();
    const view_t diagonal = diagonal_;
    Kokkos::parallel_for(
        "SelfMobilityJacobiOp::update", Kokkos::RangePolicy<ExecSpace>(0, diagonal.extent(0)),
        KOKKOS_LAMBDA(const int row) {
          double v_m_v = 0.0;
          for (int k = 0; k < jacobian.num_bodies(row); ++k) {
            const Matrix<double, 6, 6> self_block = mobility.self_mobility(jacobian.body(row, k));
            const Vector3d force = jacobian.force(row, k);
            const Vector3d torque = jacobian.torque(row, k);
            const double v[6] = {force[0], force[1], force[2], torque[0], torque[1], torque[2]};
            for (int a = 0; a < 6; ++a) {
              for (int b = 0; b < 6; ++b) {
                v_m_v += v[a] * self_block(a, b) * v[b];
              }
            }
          }
          diagonal(row) = dt * v_m_v + compliance(row);
        });
  }

 private:
  view_t diagonal_;
  ::mundy::JacobiPreconditioner<backend_t, view_t> jacobi_;
};

/// \brief Jacobi preconditioning from the mobility's self blocks; the mobility must provide them (HasSelfMobility).
struct SelfMobilityJacobi {
  template <class Linearization>
  auto make_preconditioner(const Linearization& linearization) const {
    static_assert(HasSelfMobility<typename Linearization::mobility_t>,
                  "mbody::SelfMobilityJacobi: the mobility must provide self_mobility(rod).");
    return SelfMobilityJacobiOp<typename Linearization::execution_space>(linearization.num_rows());
  }
};

#if defined(HAVE_MUNDYMATH_MUELU) && defined(HAVE_MUNDYMATH_TPETRA) && defined(HAVE_MUNDYMATH_KOKKOSKERNELS)
/// \brief One algebraic multigrid cycle of A_self = dt B^T M_self B + K^{-1}, for M_self the self blocks of M.
///
/// A_self(i, j) = dt sum v_ik^T M_b v_jl over the pairs (k, l) with b_ik = b_jl = b, plus K^{-1}_i when i = j, for b_ik
/// the k-th rod of row i, v_ik = [force(i, k); torque(i, k)] and M_b rod b's self block: rows couple when they share a
/// rod. A_self is the operator itself when M couples no two rods, and otherwise the operator with M's coupling between
/// rods dropped. Each update assembles A_self at the linearization and rebuilds the cycle; copies share both.
/// Host-only.
template <typename ExecSpace>
class SelfMobilityAMGOp {
 public:
  using view_t = Kokkos::View<double*, typename ExecSpace::memory_space>;
  using backend_t = ::mundy::KokkosBackend<ExecSpace>;
  using matrix_t =
      KokkosSparse::CrsMatrix<double, int, Kokkos::Device<ExecSpace, typename ExecSpace::memory_space>, void, size_t>;

  SelfMobilityAMGOp(size_t num_rows, const ::mundy::MueLuConfig<double>& config)
      : num_rows_(num_rows), state_(std::in_place, config) {
  }

  /// \brief A_self at the last update.
  const matrix_t& matrix() const {
    return state_->matrix;
  }

  size_t domain_size() const {
    return num_rows_;
  }
  size_t range_size() const {
    return num_rows_;
  }
  auto make_domain_vector() const {
    return backend_t::template make_vector<view_t>(num_rows_);
  }
  auto make_range_vector() const {
    return backend_t::template make_vector<view_t>(num_rows_);
  }

  /// \brief z := one cycle on r, for A_self at the last update.
  template <class RVector, class ZVector>
  void apply(const RVector& r, ZVector& z) const {
    MUNDY_THROW_ASSERT(state_->cycle.has_value(), std::logic_error,
                       "mbody::SelfMobilityAMGOp: applied before its first update.");
    state_->cycle->apply(r, z);
  }

  /// \brief Assemble A_self at linearization and rebuild the cycle.
  template <class Linearization>
  void update(const Linearization& linearization) {
    State& state = *state_;
    const impl::RowRods<ExecSpace> row_rods = impl::make_row_rods<ExecSpace>(linearization.jacobian(), num_rows_);
    // Rows keep their layout from update to update but may join other rods, which couple them differently.
    const bool same_pattern = state.cycle.has_value() && impl::same_row_rods(row_rods, state.row_rods);
    if (!same_pattern) {
      state.row_rods = row_rods;
      state.matrix = impl::make_shared_rod_pattern<matrix_t>(row_rods);
    }
    impl::fill_self_mobility_values<ExecSpace>(linearization.dt(), linearization.jacobian(), linearization.mobility(),
                                               linearization.compliance(), state.matrix);
    if (same_pattern) {
      state.cycle->update(state.matrix);
    } else {
      state.cycle.emplace(backend_t{}, state.matrix, state.config);
    }
  }

 private:
  struct State {
    explicit State(const ::mundy::MueLuConfig<double>& cycle_config) : config(cycle_config) {
    }

    ::mundy::MueLuConfig<double> config;
    impl::RowRods<ExecSpace> row_rods;
    matrix_t matrix;
    std::optional<::mundy::MueLuPreconditioner<backend_t, matrix_t>> cycle;
  };

  size_t num_rows_;
  ::mundy::host_ptr<State> state_;
};

/// \brief Algebraic multigrid preconditioning from the mobility's self blocks; the mobility must provide them
/// (HasSelfMobility).
struct SelfMobilityAMG {
  ::mundy::MueLuConfig<double> config{};  ///< the multigrid cycle

  template <class Linearization>
  auto make_preconditioner(const Linearization& linearization) const {
    static_assert(HasSelfMobility<typename Linearization::mobility_t>,
                  "mbody::SelfMobilityAMG: the mobility must provide self_mobility(rod).");
    return SelfMobilityAMGOp<typename Linearization::execution_space>(linearization.num_rows(), config);
  }
};
#endif  // HAVE_MUNDYMATH_MUELU && HAVE_MUNDYMATH_TPETRA && HAVE_MUNDYMATH_KOKKOSKERNELS
//@}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODYPRECONDITIONERS_HPP_
