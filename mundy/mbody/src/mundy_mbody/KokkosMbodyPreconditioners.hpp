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
#include <cstddef>  // for size_t
#include <utility>  // for std::declval

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Matrix.hpp>                // for mundy::Matrix
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3d
#include <mundy_math/preconditioners.hpp>       // for mundy::{Preconditioner, JacobiPreconditioner}
#include <mundy_math/solver_backends.hpp>       // for mundy::KokkosBackend
#include <mundy_mbody/KokkosMbodyMobility.hpp>  // for mundy::mbody::HasSelfMobility

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
//@}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODYPRECONDITIONERS_HPP_
