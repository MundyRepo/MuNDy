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

#ifndef MUNDY_MBODY_KOKKOSMBODYMOBILITY_HPP_
#define MUNDY_MBODY_KOKKOSMBODYMOBILITY_HPP_

/// \file
/// \brief Mobilities: how rods move under the forces and torques on them.

// C++ core
#include <concepts>   // for std::convertible_to, std::copy_constructible
#include <cstddef>    // for size_t
#include <stdexcept>  // for std::invalid_argument

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/Matrix.hpp>           // for mundy::Matrix
#include <mundy_math/Tolerance.hpp>        // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>          // for mundy::{Vector3d, dot}
#include <mundy_math/solver_backends.hpp>  // for mundy::{KokkosBackend, LinearOperator, HasScaledApplyMember}
#include <mundy_mbody/KokkosMbodyTypes.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

namespace mbody {

//! \name Mobility contracts
//@{
// A mobility M maps the rods' generalized force F (force and torque, 6 entries per rod, in rod order) to their
// generalized velocity U = M F (velocity and omega) at one configuration of the rods. M is symmetric positive definite.

/// \brief Whether M is a mobility under ExecSpace: a linear operator from generalized force to generalized velocity.
template <class M, class ExecSpace>
concept Mobility =
    std::copy_constructible<M> && ::mundy::LinearOperator<::mundy::KokkosBackend<ExecSpace>, M,
                                                          Kokkos::View<double*, typename ExecSpace::memory_space>,
                                                          Kokkos::View<double*, typename ExecSpace::memory_space>>;

/// \brief Whether a mobility provides each rod's own 6x6 block M_rr: the map from the rod's own force and torque to its
/// own velocity and omega.
template <class M>
concept HasSelfMobility = requires(const M& mobility, int rod) {
  { mobility.self_mobility(rod) } -> std::convertible_to<Matrix<double, 6, 6>>;
};

/// \brief Whether Model describes a mobility under ExecSpace: make_mobility(rods) is the mobility at rods'
/// configuration.
template <class Model, class ExecSpace>
concept MobilityModel = requires(const Model& model, const RodViews<ExecSpace>& rods) {
  { model.make_mobility(rods) } -> Mobility<ExecSpace>;
};
//@}

//! \name Local drag
//@{

/// \brief Slender-body drag on each rod alone, with no hydrodynamic coupling between rods.
///
/// Drag coefficients from Lowen 1994: Brownian dynamics of hard spherocylinders 
///
/// A rod of length L and radius a, with L' = L + 2a and p = L' / 2a, moves under force F and torque T as
///   velocity = c_perp (I - t t^T) F + c_para t t^T F,  omega = c_rot T
/// along its axis t, with inverse drags
///   c_perp = (ln p + 0.839 + 0.185/p + 0.233/p^2) / (4 pi mu L'),
///   c_para = (ln p - 0.207 + 0.98/p - 0.133/p^2) / (2 pi mu L'),
///   c_rot = 3 (ln p - 0.662 + 0.917/p - 0.05/p^2) / (pi mu L'^3).
template <typename ExecSpace>
class LocalDragMobilityOp {
 public:
  using memory_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, memory_space>;

  LocalDragMobilityOp(double viscosity, const RodViews<ExecSpace>& rods) : viscosity_(viscosity), rods_(rods) {
  }

  size_t domain_size() const {
    return 6 * rods_.size();
  }
  size_t range_size() const {
    return 6 * rods_.size();
  }

  auto make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "M_domain"), domain_size());
  }
  auto make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "M_range"), range_size());
  }

  void apply(const view_t& force_torque, view_t& vel_omega) const {
    apply(1.0, force_torque, 0.0, vel_omega);
  }

  /// \brief vel_omega := alpha * M(force_torque) + beta * vel_omega.
  ///
  /// alpha == 0 skips the per-rod drag-coefficient computation entirely.
  void apply(double alpha, const view_t& force_torque, double beta, view_t& vel_omega) const {
    MUNDY_THROW_ASSERT(force_torque.extent(0) == domain_size(), std::invalid_argument,
                       "LocalDragMobilityOp: size mismatch.");
    const bool alpha_is_zero = Kokkos::abs(alpha) < get_zero_tolerance<double>();
    const bool beta_is_zero = Kokkos::abs(beta) < get_zero_tolerance<double>();

    if (alpha_is_zero) {
      if (beta_is_zero) {
        Kokkos::deep_copy(vel_omega, 0.0);
      } else {
        auto vel_omega_l = vel_omega;
        Kokkos::parallel_for(
            "LocalDragMobilityOp::apply(beta-only)", Kokkos::RangePolicy<ExecSpace>(0, vel_omega.extent(0)),
            KOKKOS_LAMBDA(const int i) { vel_omega_l(i) *= beta; });
      }
      return;
    }

    const double inv_viscosity = 1.0 / viscosity_;
    auto rods = rods_;

    Kokkos::parallel_for(
        "LocalDragMobilityOp::apply", Kokkos::RangePolicy<ExecSpace>(0, rods.size()), KOKKOS_LAMBDA(const int i) {
          const InverseDrag inv_drag = inverse_drag(rods.length(i), rods.radius(i), inv_viscosity);
          const Vector3d tangent = rods.orientation(i) * Vector3d{0.0, 0.0, 1.0};

          const Vector3d force = rod_force(force_torque, i);
          const Vector3d torque = rod_torque(force_torque, i);

          const Vector3d force_para = dot(force, tangent) * tangent;
          const Vector3d force_perp = force - force_para;

          const Vector3d velocity = inv_drag.perp * force_perp + inv_drag.para * force_para;
          const Vector3d omega = inv_drag.rot * torque;

          if (beta_is_zero) {
            rod_velocity(vel_omega, i) = alpha * velocity;
            rod_omega(vel_omega, i) = alpha * omega;
          } else {
            rod_velocity(vel_omega, i) = alpha * velocity + beta * rod_velocity(vel_omega, i);
            rod_omega(vel_omega, i) = alpha * omega + beta * rod_omega(vel_omega, i);
          }
        });
  }

  /// \brief The rod's own 6x6 block of M, [[c_perp (I - t t^T) + c_para t t^T, 0], [0, c_rot I]].
  KOKKOS_FUNCTION Matrix<double, 6, 6> self_mobility(int rod) const {
    const InverseDrag inv_drag = inverse_drag(rods_.length(rod), rods_.radius(rod), 1.0 / viscosity_);
    const Vector3d tangent = rods_.orientation(rod) * Vector3d{0.0, 0.0, 1.0};

    Matrix<double, 6, 6> block = Matrix<double, 6, 6>();
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        const double t_t = tangent[c] * tangent[r];
        block(r, c) = inv_drag.perp * ((r == c ? 1.0 : 0.0) - t_t) + inv_drag.para * t_t;
      }
      block(3 + r, 3 + r) = inv_drag.rot;
    }
    return block;
  }

 private:
  /// \brief A rod's perpendicular, parallel and rotational inverse drags.
  struct InverseDrag {
    double perp;
    double para;
    double rot;
  };

  KOKKOS_INLINE_FUNCTION static InverseDrag inverse_drag(double length, double radius, double inv_viscosity) {
    constexpr double pi = Kokkos::numbers::pi_v<double>;
    constexpr double inv_four_pi = 1.0 / (4.0 * pi);
    constexpr double inv_two_pi = 1.0 / (2.0 * pi);
    constexpr double inv_pi = 1.0 / pi;
    const double lprime = length + 2.0 * radius;
    const double p = lprime / (2.0 * radius);
    const double log_p = Kokkos::log(p);
    const double inv_p = 1.0 / p;
    const double inv_p2 = inv_p * inv_p;
    const double inv_lprime = 1.0 / lprime;
    const double inv_lprime3 = inv_lprime * inv_lprime * inv_lprime;
    return InverseDrag{(log_p + 0.839 + 0.185 * inv_p + 0.233 * inv_p2) * inv_lprime * inv_four_pi * inv_viscosity,
                       (log_p - 0.207 + 0.98 * inv_p - 0.133 * inv_p2) * inv_lprime * inv_two_pi * inv_viscosity,
                       3.0 * (log_p - 0.662 + 0.917 * inv_p - 0.05 * inv_p2) * inv_lprime3 * inv_pi * inv_viscosity};
  }

  double viscosity_;
  RodViews<ExecSpace> rods_;
};

/// \brief Local drag in a fluid of viscosity mu: each rod moves under its own slender-body drag alone.
struct LocalDragMobility {
  double viscosity = 1.0;

  template <typename ExecSpace>
  LocalDragMobilityOp<ExecSpace> make_mobility(const RodViews<ExecSpace>& rods) const {
    return LocalDragMobilityOp<ExecSpace>(viscosity, rods);
  }
};

static_assert(Mobility<LocalDragMobilityOp<Kokkos::DefaultExecutionSpace>, Kokkos::DefaultExecutionSpace>,
              "LocalDragMobilityOp must satisfy Mobility");
static_assert(HasSelfMobility<LocalDragMobilityOp<Kokkos::DefaultExecutionSpace>>,
              "LocalDragMobilityOp must satisfy HasSelfMobility");
static_assert(::mundy::HasScaledApplyMember<LocalDragMobilityOp<Kokkos::DefaultExecutionSpace>, double,
                                            Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>,
                                            Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>>,
              "LocalDragMobilityOp must satisfy ::mundy::HasScaledApplyMember");
static_assert(MobilityModel<LocalDragMobility, Kokkos::DefaultExecutionSpace>,
              "LocalDragMobility must satisfy MobilityModel");
//@}

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_KOKKOSMBODYMOBILITY_HPP_
