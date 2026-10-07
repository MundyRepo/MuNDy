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

#ifndef MUNDY_MBODY_PERIPHERY_HPP_
#define MUNDY_MBODY_PERIPHERY_HPP_

/* This class evaluates the fluid flow at interior points of the domain induced by enforcing the no-slip condition
on the periphery: given an external flow sampled on the surface it computes the surface forces f = M^{-1} u, then
evaluates the flow those forces induce at the interior points.

The periphery is described by a collection of nodes with consistently oriented surface normals (set_surface_normals
declares the orientation; it is checked against the geometry) and predefined quadrature weights. We are not in charge
of computing these quantities.

The periphery is single-device and targets N < 5000 nodes: the dense self-interaction inverse (a 5000x5000 matrix
is ~200 MB) fits on one device, so it is not distributed.

Conventions:
  T[q](x) = int_Gamma K(x, y) q(y) dS_y with K_ij(x, y) = -3 / (4 pi mu) r_i r_j (r . n(y)) / |r|^5 and r = x - y is
  the double-layer potential of a stresslet surface density q (force per length, i.e. mu times velocity).
  sigma = +1 if the normals point out of the region Gamma encloses and -1 if they point into it. The Gauss identity
  gives int_Gamma K(x, y) dS_y = sigma/mu inside, sigma/(2 mu) on Gamma (principal value), and 0 outside, so
  PV T[1] = sigma/(2 mu) I. With the normals pointing into the fluid (outward on a body, inward on the periphery),
  the fluid-side trace of T[q] is (J + T)[q] with J = -1/(2 mu) I. It is evaluated by singularity subtraction,
    (J + T)[q](x_t) = int_Gamma K(x_t, y) (q(y) - q(x_t)) dS_y + c q(x_t),
  with the analytic target coefficient c = sigma/mu for the periphery's interior trace and c = 0 for a body's
  exterior trace (for either orientation).

Consider a system containing just mobile bodies:
  for n in 1..N_{bodies}:
    (-1/(2 mu) I + T_b^n)[q_b^n](x) - U_b^n - Omega_b^n cross (x - X_b^n) + sum_{m, m != n} T_b^m[q_m](x)
        = - u_{ext}(x) - sum_m (G(x - X_b^m) dot F_b^m + R(x - X_b^m) dot tau_b^m)

  for n in 1..N_{bodies}:
    1/|Gamma_b^n| int_{Gamma_b^n} q_b^n(y) dS_y - U_b^n = 0
    1/|Gamma_b^n| int_{Gamma_b^n} (y - X_b^n) cross q_b^n(y) dS_y - Omega_b^n = 0

  Here,
    U_b^n, Omega_b^n are the unknown center of mass translational and rotational velocities of the n'th body
    q_b^n is the unknown surface density on body n
    I is the identity operator
    T_b^n is the double-layer (stresslet) operator integrated over the n'th body, with outward normals
    (-1/(2 mu) I + T_b^n)[q_b^n](x) is evaluated via singularity subtraction
    u_{ext}(x) is some known external flow
    G/R are the stokeslet and rotlet
    F_b^m and tao_b^m are the center of mass force and torque on body m applied at location X_b^m
    |Gamma_b^n| means surface area of the surface Gamma_b^n of the n'th body
  The constraint rows only fix the rigid-motion null-space component of q_b^n, which produces no exterior flow, so
  U_b^n and Omega_b^n are determined by the first row and are the physical velocities for any viscosity.

Now, place them in a periphery:
  for n in 1..N_{bodies}:
    (-1/(2 mu) I + T_b^n)[q_b^n](x)                             // Self, immotile
        - U_b^n - Omega_b^n cross (x - X_b^n)                   // Self, motile
        + sum_{m, m != n} T_b^m[q_m](x)                         // Other bodies
        + T_p[P^{-1} u_p^{slip_unknown}(x) for y in Gamma_p](x) // Unknown periphery feedback
      = - u_{ext}(x)                                                                    // Known flow field
        - T_p[P^{-1} (u_p^{slip_known}(y) for y in Gamma_p)](x)                         // Known periphery feedback
        - sum_m (G(x - X_b^m) dot F_b^m + R(x - X_b^m) dot tau_b^m) for x in Gamma_p^n  // Known body force/torque
feedback where u_p^{slip_unknown}(x) = sum_m T_b^m[q_b^m](x) u_p^{slip_known}(x) = u_{ext}(x) + sum_m (G(x - X_b^m) dot
F_b^m + R(x - X_b^m) dot tau_b^m)

  for n in 1..N_{bodies}:
    1/|Gamma_b^n| int_{Gamma_b^n} q_b^n(y) dS_y - U_b^n = 0
    1/|Gamma_b^n| int_{Gamma_b^n} (y - X_b^n) cross q_b^n(y) dS_y - Omega_b^n = 0

Now, add point spheres generating a flow through their interaction kernel K (RPY, RPYC, or Stokes; none when Dry):
  u_ext(x) = sum_s K(x - x_s) dot F_s

The order here goes
  1. Setup fixed RHS:
    1.1. Evaluate the spheres' interaction kernel at all quadrature points (bodies + periphery)
    1.2. Evaluate G/R from body force torque on all quadrature points (bodies + periphery)
    1.3. Evaluate known periphery feedback (periphery-to-bodies)
  2. Setup LHS:
    2.1. Evaluate singularity-subtracted double-layer kernel (body self-interaction)
    2.2. Evaluate double-layer kernel from all bodies to all quadrature points that are not their own (other bodies +
periphery)
    2.3. Evaluate unknown periphery feedback (periphery-to-body)
  3. Post-solve
    3.1. Evaluate periphery-to-sphere flow (known and unknown)
    3.2. Evaluate body-to-sphere flow (known and unknown)
*/

// C++ core
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/GaussLegendreSphere.hpp>  // for mundy::gauss_legendre_sphere_rule
#include <mundy_math/Quaternion.hpp>           // for mundy::Quaternion (reference->lab rotation)
#include <mundy_math/Vector3.hpp>              // for mundy::Vector3, mundy::cross
#include <mundy_math/direct_sum.hpp>           // for mundy::direct_sum
#include <mundy_math/invert.hpp>               // for mundy::invert
#include <mundy_math/matrix_market.hpp>        // for mundy::read_matrix_market, mundy::write_matrix_market
#include <mundy_utils/throw_assert.hpp>        // for MUNDY_THROW_ASSERT

// The matrix-free GMRES inverse path is only available when both the Belos and Tpetra TPLs are enabled; without
// them the periphery still offers the dense direct inverse.
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_{BELOS,TPETRA}
#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
#include <Tpetra_Map.hpp>                  // for Tpetra::Map<>::node_type (the exec space the Belos solve must run on)
#include <mundy_math/belos_solver.hpp>     // for mundy::{make_belos_inv_op, BelosInvOp, BelosConfig, BelosSolver}
#include <mundy_math/solver_backends.hpp>  // for mundy::KokkosBackend
#endif

#define DOUBLE_ZERO 1.0e-12

namespace mundy {

namespace mbody {

namespace impl {

/// \brief True iff every listed type is a rank-1 Kokkos::View of (possibly const) double.
template <class... Views>
inline constexpr bool are_double_vectors_v =
    ((Kokkos::is_view<Views>::value && Views::rank == 1 &&
      std::is_same_v<std::remove_const_t<typename Views::value_type>, double>) &&
     ...);

/// \brief True iff every listed type is a rank-2 Kokkos::View of (possibly const) double.
template <class... Matrices>
inline constexpr bool are_double_matrices_v =
    ((Kokkos::is_view<Matrices>::value && Matrices::rank == 2 &&
      std::is_same_v<std::remove_const_t<typename Matrices::value_type>, double>) &&
     ...);

/// \brief True iff every listed view shares the memory space of the first.
template <class First, class... Rest>
inline constexpr bool share_memory_space_v =
    (std::is_same_v<typename First::memory_space, typename Rest::memory_space> && ...);

/// \brief Host-side length precondition: require view.extent(0) == expected, else throw a message naming the
/// calling context, the offending view (via its Kokkos label), and the expected vs. actual length.
template <class View>
void require_length(const View& view, const size_t expected, const char* context) {
  MUNDY_THROW_ASSERT(view.extent(0) == expected, std::invalid_argument,
                     mundy::sink() << context << ": " << view.label() << " must have length " << expected
                                   << " but has length " << view.extent(0));
}

/// \brief Weighted moments of a surface quadrature, accumulated by SurfaceMomentsFunctor.
struct SurfaceMoments {
  double area;      //!< sum_s w_s
  double wx[3];     //!< sum_s w_s x_s
  double wn[3];     //!< sum_s w_s n_s
  double wn_dot_x;  //!< sum_s w_s n_s . x_s
};

/// \brief Reduction functor accumulating SurfaceMoments in one sweep over a surface quadrature.
template <class PosView, class NrmView, class WgtView>
struct SurfaceMomentsFunctor {
  using value_type = SurfaceMoments;

  PosView positions;
  NrmView normals;
  WgtView weights;

  KOKKOS_INLINE_FUNCTION void operator()(const size_t s, value_type& acc) const {
    const double w = weights(s);
    acc.area += w;
    for (int d = 0; d < 3; ++d) {
      acc.wx[d] += w * positions(3 * s + d);
      acc.wn[d] += w * normals(3 * s + d);
      acc.wn_dot_x += w * normals(3 * s + d) * positions(3 * s + d);
    }
  }

  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
    dst.area += src.area;
    for (int d = 0; d < 3; ++d) {
      dst.wx[d] += src.wx[d];
      dst.wn[d] += src.wn[d];
    }
    dst.wn_dot_x += src.wn_dot_x;
  }

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const {
    v.area = 0.0;
    for (int d = 0; d < 3; ++d) {
      v.wx[d] = 0.0;
      v.wn[d] = 0.0;
    }
    v.wn_dot_x = 0.0;
  }
};

/// \brief The area |Gamma| = sum_s w_s of a closed surface quadrature, after validating it and its normal orientation.
///
/// Throws unless |Gamma| is positive and finite and the orientation flux sum_s w_s n_s . (x_s - c) has the declared
/// sign. By the divergence theorem the flux is +3V for normals pointing out of the enclosed region (of volume V) and
/// -3V for normals pointing into it; subtracting the centroid c = sum_s w_s x_s / |Gamma| keeps it translation
/// invariant when the discrete sum_s w_s n_s is not exactly zero.
template <class ExecutionSpace, class PosView, class NrmView, class WgtView>
double validated_surface_area([[maybe_unused]] const ExecutionSpace& space, const size_t num_points,
                              const PosView& positions, const NrmView& normals, const WgtView& weights,
                              const bool outward_normal, const char* context) {
  SurfaceMomentsFunctor<PosView, NrmView, WgtView> functor{positions, normals, weights};
  SurfaceMoments moments;
  Kokkos::parallel_reduce("validated_surface_area", Kokkos::RangePolicy<ExecutionSpace>(0, num_points), functor,
                          moments);
  const double area = moments.area;
  MUNDY_THROW_REQUIRE(
      area > 0.0 && std::isfinite(area), std::invalid_argument,
      mundy::sink() << context << ": the surface area sum_s w_s = " << area << " must be positive and finite.");

  const double centroid_dot_wn =
      (moments.wx[0] * moments.wn[0] + moments.wx[1] * moments.wn[1] + moments.wx[2] * moments.wn[2]) / area;
  const double flux = moments.wn_dot_x - centroid_dot_wn;
  MUNDY_THROW_REQUIRE(
      outward_normal ? (flux > 0.0) : (flux < 0.0), std::invalid_argument,
      mundy::sink() << context << ": the normals were declared "
                    << (outward_normal ? "outward (outward_normal = true)" : "inward (outward_normal = false)")
                    << " but the orientation flux sum_s w_s n_s . (x_s - c) = " << flux
                    << " says otherwise (it is +3V for outward and -3V for inward normals).");
  return area;
}

/// \brief Host-side precondition: the viscosity must be positive.
inline void require_positive_viscosity(const double viscosity, const char* context) {
  MUNDY_THROW_REQUIRE(viscosity > 0.0, std::invalid_argument,
                      mundy::sink() << context << ": viscosity must be positive but is " << viscosity);
}

/// \brief The analytic target coefficient sigma / viscosity of the periphery's interior trace (see the file header).
inline double interior_trace_coefficient(const double viscosity, const bool outward_normal) {
  return (outward_normal ? 1.0 : -1.0) / viscosity;
}

/// \brief A view read from a Matrix Market file, which must hold a rows x cols array (cols = 1 for a vector).
///
/// Files are external input, so their extents are checked in every build, not only in debug builds.
template <class View>
View read_matrix_market_with_extents(const std::string& filename, const size_t rows, const size_t cols,
                                     const char* context) {
  View view;
  mundy::read_matrix_market(filename, view);
  const size_t file_cols = View::rank() == 1 ? 1 : view.extent(1);
  MUNDY_THROW_REQUIRE(view.extent(0) == rows && file_cols == cols, std::runtime_error,
                      mundy::sink() << context << ": " << filename << " holds a " << view.extent(0) << " x "
                                    << file_cols << " array, but a " << rows << " x " << cols << " array is expected.");
  return view;
}

/// \brief The direct_sum accumulator that adds target t's 3-vector sum to entries 3 t, 3 t + 1, 3 t + 2 of out.
template <class View>
auto add_to_vector3_entries(const View& out) {
  return KOKKOS_LAMBDA(const size_t t, const mundy::Vector3d& sum) {
    out(3 * t + 0) += sum[0];
    out(3 * t + 1) += sum[1];
    out(3 * t + 2) += sum[2];
  };
}

}  // namespace impl

/// \brief Copy a host std::vector<double> into a fresh device (LayoutLeft) view of the same length.
template <class ExecSpace = Kokkos::DefaultExecutionSpace>
Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space> to_device(const std::vector<double>& host) {
  Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space> dev(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "to_device"), host.size());
  auto mirror = Kokkos::create_mirror_view(dev);
  for (size_t i = 0; i < host.size(); ++i) {
    mirror(i) = host[i];
  }
  Kokkos::deep_copy(dev, mirror);
  return dev;
}

/// \brief Apply the stokes kernel to map source forces to target velocities: u_target += M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceForceVectorType, typename TargetVelocityVectorType>
void apply_stokes_kernel(const ExecutionSpace& space,                  //
                         const double viscosity,                       //
                         const SourcePosVectorType& source_positions,  //
                         const TargetPosVectorType& target_positions,  //
                         const SourceForceVectorType& source_forces,   //
                         const TargetVelocityVectorType& target_velocities) {
  static_assert(impl::are_double_vectors_v<SourcePosVectorType, TargetPosVectorType, SourceForceVectorType,
                                           TargetVelocityVectorType>,
                "apply_stokes_kernel: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<SourcePosVectorType, TargetPosVectorType, SourceForceVectorType,
                                           TargetVelocityVectorType>,
                "apply_stokes_kernel: all views must share one memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  impl::require_length(source_forces, 3 * num_source_points, "apply_stokes_kernel");
  impl::require_length(target_velocities, 3 * num_target_points, "apply_stokes_kernel");

  // Launch the parallel kernel
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);

  auto stokes_computation = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = r2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : r2);
    const double rinv3 = rinv * rinv * rinv;

    const double f_dot_r = fx * dx + fy * dy + fz * dz;
    const double scale_factor_rinv3 = scale_factor * rinv3;

    // Accumulate velocity contribution to local variables
    return mundy::Vector3d{scale_factor_rinv3 * (r2 * fx + dx * f_dot_r), scale_factor_rinv3 * (r2 * fy + dy * f_dot_r),
                           scale_factor_rinv3 * (r2 * fz + dz * f_dot_r)};
  };

  mundy::direct_sum(space, num_target_points, num_source_points, stokes_computation,
                    impl::add_to_vector3_entries(target_velocities));
}

/// \brief Apply the stokes kernel to map source forces to target velocities: u_target += M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceForceVectorType, typename SourceWeightVectorType, typename TargetVelocityVectorType>
void apply_weighted_stokes_kernel(const ExecutionSpace& space,                   //
                                  const double viscosity,                        //
                                  const SourcePosVectorType& source_positions,   //
                                  const TargetPosVectorType& target_positions,   //
                                  const SourceForceVectorType& source_forces,    //
                                  const SourceWeightVectorType& source_weights,  //
                                  const TargetVelocityVectorType& target_velocities) {
  static_assert(impl::are_double_vectors_v<SourcePosVectorType, TargetPosVectorType, SourceForceVectorType,
                                           SourceWeightVectorType, TargetVelocityVectorType>,
                "apply_weighted_stokes_kernel: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<SourcePosVectorType, TargetPosVectorType, SourceForceVectorType,
                                           SourceWeightVectorType, TargetVelocityVectorType>,
                "apply_weighted_stokes_kernel: all views must share one memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  impl::require_length(source_forces, 3 * num_source_points, "apply_weighted_stokes_kernel");
  impl::require_length(target_velocities, 3 * num_target_points, "apply_weighted_stokes_kernel");
  impl::require_length(source_weights, num_source_points, "apply_weighted_stokes_kernel");

  // Launch the parallel kernel
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);
  auto weighted_stokes_computation = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0) * source_weights(s);
    const double fy = source_forces(3 * s + 1) * source_weights(s);
    const double fz = source_forces(3 * s + 2) * source_weights(s);

    // const double r2 = dx * dx + dy * dy + dz * dz;
    // const double rinv = r2 < DOUBLE_ZERO ? 0.0 : 1.0 / Kokkos::sqrt(r2);
    // const double rinv3 = rinv * rinv * rinv;

    // const double inner_prod = fx * dx + fy * dy + fz * dz;
    // const double scale_factor_rinv3 = scale_factor * rinv3;

    // // Accumulate velocity contribution to local variables
    // vx_accum += scale_factor_rinv3 * (r2 * fx + dx * inner_prod);
    // vy_accum += scale_factor_rinv3 * (r2 * fy + dy * inner_prod);
    // vz_accum += scale_factor_rinv3 * (r2 * fz + dz * inner_prod);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = r2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : r2);
    const double rinv2 = rinv * rinv;

    const double f_dot_r_rinv2 = (fx * dx + fy * dy + fz * dz) * rinv2;
    const double scale_factor_rinv = scale_factor * rinv;

    // Accumulate velocity contribution to local variables
    return mundy::Vector3d{scale_factor_rinv * (fx + dx * f_dot_r_rinv2), scale_factor_rinv * (fy + dy * f_dot_r_rinv2),
                           scale_factor_rinv * (fz + dz * f_dot_r_rinv2)};
  };

  mundy::direct_sum(space, num_target_points, num_source_points, weighted_stokes_computation,
                    impl::add_to_vector3_entries(target_velocities));
}

/// \brief Apply the RPY kernel to map source forces to target velocities: u_target += M f_source
///
/// Note, this does not include self-interaction. If that is desired simply add 1/(6 pi mu) * f to u
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceRadiusVectorType, typename TargetRadiusVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_rpy_kernel(const ExecutionSpace& space,                  //
                      const double viscosity,                       //
                      const SourcePosVectorType& source_positions,  //
                      const TargetPosVectorType& target_positions,  //
                      const SourceRadiusVectorType& source_radii,   //
                      const TargetRadiusVectorType& target_radii,   //
                      const SourceForceVectorType& source_forces,   //
                      const TargetVelocityVectorType& target_velocities) {
  static_assert(impl::are_double_vectors_v<SourcePosVectorType, TargetPosVectorType, SourceRadiusVectorType,
                                           TargetRadiusVectorType, SourceForceVectorType, TargetVelocityVectorType>,
                "apply_rpy_kernel: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<SourcePosVectorType, TargetPosVectorType, SourceRadiusVectorType,
                                           TargetRadiusVectorType, SourceForceVectorType, TargetVelocityVectorType>,
                "apply_rpy_kernel: all views must share one memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  impl::require_length(source_forces, 3 * num_source_points, "apply_rpy_kernel");
  impl::require_length(target_velocities, 3 * num_target_points, "apply_rpy_kernel");
  impl::require_length(source_radii, num_source_points, "apply_rpy_kernel");
  impl::require_length(target_radii, num_target_points, "apply_rpy_kernel");

  // Launch the parallel kernel
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);
  auto rpy_computation = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);
    const double a = source_radii(s);

    constexpr double one_over_three = 1.0 / 3.0;
    constexpr double one_over_six = 1.0 / 6.0;

    const double a2_over_three = one_over_three * a * a;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = r2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : r2);
    const double rinv3 = rinv * rinv * rinv;
    const double rinv5 = rinv * rinv * rinv3;
    const double fdotr = fx * dx + fy * dy + fz * dz;

    const double three_fdotr_rinv5 = 3.0 * fdotr * rinv5;
    const double cx = fx * rinv3 - three_fdotr_rinv5 * dx;
    const double cy = fy * rinv3 - three_fdotr_rinv5 * dy;
    const double cz = fz * rinv3 - three_fdotr_rinv5 * dz;

    const double fdotr_rinv3 = fdotr * rinv3;

    // Velocity
    const double v0 = scale_factor * (fx * rinv + dx * fdotr_rinv3 + a2_over_three * cx);
    const double v1 = scale_factor * (fy * rinv + dy * fdotr_rinv3 + a2_over_three * cy);
    const double v2 = scale_factor * (fz * rinv + dz * fdotr_rinv3 + a2_over_three * cz);

    // Laplacian
    const double lap0 = 2.0 * scale_factor * cx;
    const double lap1 = 2.0 * scale_factor * cy;
    const double lap2 = 2.0 * scale_factor * cz;

    // Apply the result
    const double lap_coeff = one_over_six * target_radii(t) * target_radii(t);
    return mundy::Vector3d{v0 + lap_coeff * lap0, v1 + lap_coeff * lap1, v2 + lap_coeff * lap2};
  };

  mundy::direct_sum(space, num_target_points, num_source_points, rpy_computation,
                    impl::add_to_vector3_entries(target_velocities));
}

/// \brief Apply the corrected RPY kernel to map source forces to target velocities: u_target += M f_source
///
/// Note, this does not include self-interaction. If that is desired simply add 1/(6 pi mu) * f to u
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] source_positions The positions of the source points (size num_source_points x 3)
/// \param[in] target_positions The positions of the target points (size num_target_points x 3)
/// \param[in] source_forces The source values (size num_source_points x 3)
/// \param[out] target_values The target values (size num_target_points x 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceRadiusVectorType, typename TargetRadiusVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_rpyc_kernel(const ExecutionSpace& space,                  //
                       const double viscosity,                       //
                       const SourcePosVectorType& source_positions,  //
                       const TargetPosVectorType& target_positions,  //
                       const SourceRadiusVectorType& source_radii,   //
                       const TargetRadiusVectorType& target_radii,   //
                       const SourceForceVectorType& source_forces,   //
                       const TargetVelocityVectorType& target_velocities) {
  static_assert(impl::are_double_vectors_v<SourcePosVectorType, TargetPosVectorType, SourceRadiusVectorType,
                                           TargetRadiusVectorType, SourceForceVectorType, TargetVelocityVectorType>,
                "apply_rpyc_kernel: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<SourcePosVectorType, TargetPosVectorType, SourceRadiusVectorType,
                                           TargetRadiusVectorType, SourceForceVectorType, TargetVelocityVectorType>,
                "apply_rpyc_kernel: all views must share one memory space.");

  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  impl::require_length(source_forces, 3 * num_source_points, "apply_rpyc_kernel");
  impl::require_length(target_velocities, 3 * num_target_points, "apply_rpyc_kernel");
  impl::require_length(source_radii, num_source_points, "apply_rpyc_kernel");
  impl::require_length(target_radii, num_target_points, "apply_rpyc_kernel");

  // Launch the parallel kernel
  constexpr double one_over_eight = 1.0 / 8.0;
  constexpr double one_over_six = 1.0 / 6.0;
  constexpr double one_over_three = 1.0 / 3.0;
  constexpr double one_over_32 = 1.0 / 32.0;
  constexpr double inv_pi = 1.0 / Kokkos::numbers::pi_v<double>;
  const double inv_viscosity = 1.0 / viscosity;
  auto rpyc_computation = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);
    const double a = source_radii(s);
    const double b = target_radii(t);

    const double r2 = dx * dx + dy * dy + dz * dz;
    const double r = Kokkos::sqrt(r2);
    const double r3 = r * r2;
    const bool coincident = r2 < DOUBLE_ZERO;  // selected, not branched on
    const double rinv = coincident ? 0.0 : 1.0 / (coincident ? 1.0 : r);
    const double rinv2 = rinv * rinv;
    const double rinv3 = rinv2 * rinv;

    const double dx_hat = dx * rinv;
    const double dy_hat = dy * rinv;
    const double dz_hat = dz * rinv;

    const double f_dot_rhat = fx * dx_hat + fy * dy_hat + fz * dz_hat;

    if (a + b < r) {
      // If a + b < r, regular RPY
      // M = coeff * (tmp1 * I + tmp2 * r_hat outer r_hat)
      //   coeff = 1 / (8 pi mu r_norm)
      //   tmp1 = 1 + (a**2 + b**2) / (3 * r_norm**2)
      //   tmp2 = 1 - (a**2 + b**2) / (r_norm**2)
      //   r_hat = r / r_norm
      const double a2 = a * a;
      const double b2 = b * b;
      const double a2_plus_b2_rinv2 = (a2 + b2) * rinv2;
      const double scale_factor = one_over_eight * inv_pi * inv_viscosity * rinv;
      const double tmp1_scaled = scale_factor * (1. + a2_plus_b2_rinv2 * one_over_three);
      const double tmp2_scaled = scale_factor * (1. - a2_plus_b2_rinv2);
      return mundy::Vector3d{tmp1_scaled * fx + tmp2_scaled * (f_dot_rhat * dx_hat),
                             tmp1_scaled * fy + tmp2_scaled * (f_dot_rhat * dy_hat),
                             tmp1_scaled * fz + tmp2_scaled * (f_dot_rhat * dz_hat)};
    } else if (Kokkos::abs(a - b) < r && a > DOUBLE_ZERO && b > DOUBLE_ZERO) {
      // If neither radius is zero and if abs(a - b) < r < a + b, corrected RPY
      // M = 1/(6 pi mu a b) * (tmp1 I + tmp2 r_hat outer r_hat) f
      //  tmp1 = (16 r^3 (a + b) - ((a - b)^2 + 3 r^2)^2) / (32 r^3)
      //  tmp2 = 3 ((a - b)^2 - r^2)^2 / (32 r^3)
      //  r_hat = r / r_norm
      const double a_plus_b = a + b;
      const double a_minus_b = a - b;
      const double a_minus_b2 = a_minus_b * a_minus_b;
      const double tmp3 = a_minus_b2 + 3 * r2;
      const double tmp4 = a_minus_b2 - r2;

      const double scale_factor = one_over_six * inv_pi * inv_viscosity / (a * b);
      const double tmp1_scaled = scale_factor * (16.0 * r3 * a_plus_b - tmp3 * tmp3) * one_over_32 * rinv3;
      const double tmp2_scaled = scale_factor * 3.0 * tmp4 * tmp4 * one_over_32 * rinv3;

      return mundy::Vector3d{tmp1_scaled * fx + tmp2_scaled * (f_dot_rhat * dx_hat),
                             tmp1_scaled * fy + tmp2_scaled * (f_dot_rhat * dy_hat),
                             tmp1_scaled * fz + tmp2_scaled * (f_dot_rhat * dz_hat)};
    } else {
      //  if r < abs(a - b), Local drag
      // v = 1 / (6 pi mu max(a, b)) * f
      if (r2 < DOUBLE_ZERO) {
        // Skip self interaction
        return mundy::Vector3d{0.0, 0.0, 0.0};
      }

      const double max_a_b = Kokkos::max(a, b);
      const double scale_factor = one_over_six * inv_pi * inv_viscosity / max_a_b;
      return mundy::Vector3d{scale_factor * fx, scale_factor * fy, scale_factor * fz};
    }
  };

  mundy::direct_sum(space, num_target_points, num_source_points, rpyc_computation,
                    impl::add_to_vector3_entries(target_velocities));
}

/// \brief Accumulate the singularity-subtracted exterior trace u += (J + T)[f] of a closed body surface.
///
/// Evaluates u_t += sum_{s != t} T_{t,s} (f_s - f_t), the exterior trace of the double-layer potential with the given
/// normals; its analytic target coefficient is zero (see the file header), so it is valid for either orientation.
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_points The number of surface points
/// \param[in] positions The surface point positions (size num_points * 3)
/// \param[in] normals The surface normals (size num_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_points)
/// \param[in] forces The surface forces the operator is applied to (size num_points * 3)
/// \param[out] velocities The resulting surface velocities (size num_points * 3)
template <class ExecutionSpace, typename PosVectorType, typename NormalVectorType, typename QuadratureWeightVectorType,
          typename ForceVectorType, typename VelocityVectorType>
void apply_stokes_double_layer_kernel_ss(const ExecutionSpace& space,                           //
                                         const double viscosity,                                //
                                         const size_t num_points,                               //
                                         const PosVectorType& positions,                        //
                                         const NormalVectorType& normals,                       //
                                         const QuadratureWeightVectorType& quadrature_weights,  //
                                         const ForceVectorType& forces,                         //
                                         const VelocityVectorType& velocities) {
  static_assert(impl::are_double_vectors_v<PosVectorType, NormalVectorType, QuadratureWeightVectorType, ForceVectorType,
                                           VelocityVectorType>,
                "apply_stokes_double_layer_kernel_ss: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<PosVectorType, NormalVectorType, QuadratureWeightVectorType, ForceVectorType,
                                           VelocityVectorType>,
                "apply_stokes_double_layer_kernel_ss: all views must share one memory space.");

  impl::require_length(positions, 3 * num_points, "apply_stokes_double_layer_kernel_ss");
  impl::require_length(normals, 3 * num_points, "apply_stokes_double_layer_kernel_ss");
  impl::require_length(quadrature_weights, num_points, "apply_stokes_double_layer_kernel_ss");
  impl::require_length(forces, 3 * num_points, "apply_stokes_double_layer_kernel_ss");
  impl::require_length(velocities, 3 * num_points, "apply_stokes_double_layer_kernel_ss");
  impl::require_positive_viscosity(viscosity, "apply_stokes_double_layer_kernel_ss");

  // Launch the parallel kernel
  const double scale_factor = 3.0 / (4.0 * M_PI * viscosity);
  auto stokes_double_layer_computation = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Skip self-interaction
    if (t == s) {
      return mundy::Vector3d{0.0, 0.0, 0.0};
    }

    // Compute the distance vector
    const double dx = positions(3 * t + 0) - positions(3 * s + 0);
    const double dy = positions(3 * t + 1) - positions(3 * s + 1);
    const double dz = positions(3 * t + 2) - positions(3 * s + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = dr2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : dr2);
    const double rinv2 = rinv * rinv;
    const double rinv5 = rinv * rinv2 * rinv2;

    // The singularity-subtracted integrand K(x_t, y_s) (f_s - f_t), bounded as y_s -> x_t.
    const double fs0 = forces(3 * s + 0) - forces(3 * t + 0);
    const double fs1 = forces(3 * s + 1) - forces(3 * t + 1);
    const double fs2 = forces(3 * s + 2) - forces(3 * t + 2);
    const double sxx = normals(3 * s + 0) * fs0 * quadrature_weights(s);
    const double sxy = normals(3 * s + 0) * fs1 * quadrature_weights(s);
    const double sxz = normals(3 * s + 0) * fs2 * quadrature_weights(s);
    const double syx = normals(3 * s + 1) * fs0 * quadrature_weights(s);
    const double syy = normals(3 * s + 1) * fs1 * quadrature_weights(s);
    const double syz = normals(3 * s + 1) * fs2 * quadrature_weights(s);
    const double szx = normals(3 * s + 2) * fs0 * quadrature_weights(s);
    const double szy = normals(3 * s + 2) * fs1 * quadrature_weights(s);
    const double szz = normals(3 * s + 2) * fs2 * quadrature_weights(s);

    double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
    coeff += (sxy + syx) * dx * dy;
    coeff += (sxz + szx) * dx * dz;
    coeff += (syz + szy) * dy * dz;
    coeff *= -scale_factor * rinv5;

    return mundy::Vector3d{dx * coeff, dy * coeff, dz * coeff};
  };

  mundy::direct_sum(space, num_points, num_points, stokes_double_layer_computation,
                    impl::add_to_vector3_entries(velocities));
}

/// \brief Apply the stokes double layer kernel to map source forces to target velocities: u_target += M f_source
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
/// \param[in] source_forces The vector to apply the self-interaction matrix to (size num_nodes * 3)
/// \param[out] target_velocities The result of applying the self-interaction matrix to f (size num_nodes * 3)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename QuadratureWeightVectorType, typename SourceForceVectorType,
          typename TargetVelocityVectorType>
void apply_stokes_double_layer_kernel(const ExecutionSpace& space,                           //
                                      const double viscosity,                                //
                                      const size_t num_source_points,                        //
                                      const size_t num_target_points,                        //
                                      const SourcePosVectorType& source_positions,           //
                                      const TargetPosVectorType& target_positions,           //
                                      const SourceNormalVectorType& source_normals,          //
                                      const QuadratureWeightVectorType& quadrature_weights,  //
                                      const SourceForceVectorType& source_forces,            //
                                      const TargetVelocityVectorType& target_velocities) {
  static_assert(impl::are_double_vectors_v<SourcePosVectorType, TargetPosVectorType, SourceNormalVectorType,
                                           QuadratureWeightVectorType, SourceForceVectorType, TargetVelocityVectorType>,
                "apply_stokes_double_layer_kernel: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<SourcePosVectorType, TargetPosVectorType, SourceNormalVectorType,
                                           QuadratureWeightVectorType, SourceForceVectorType, TargetVelocityVectorType>,
                "apply_stokes_double_layer_kernel: all views must share one memory space.");

  impl::require_length(source_positions, 3 * num_source_points, "apply_stokes_double_layer_kernel");
  impl::require_length(target_positions, 3 * num_target_points, "apply_stokes_double_layer_kernel");
  impl::require_length(source_normals, 3 * num_source_points, "apply_stokes_double_layer_kernel");
  impl::require_length(quadrature_weights, num_source_points, "apply_stokes_double_layer_kernel");
  impl::require_length(source_forces, 3 * num_source_points, "apply_stokes_double_layer_kernel");
  impl::require_length(target_velocities, 3 * num_target_points, "apply_stokes_double_layer_kernel");

  // Launch the parallel kernel
  const double scale_factor = 3.0 / (4.0 * M_PI * viscosity);
  auto stokes_double_layer_contribution = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Compute the distance vector
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = dr2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : dr2);
    const double rinv2 = rinv * rinv;
    const double rinv5 = rinv * rinv2 * rinv2;

    // Compute the double layer potential
    const double sxx = source_normals(3 * s + 0) * source_forces(3 * s + 0) * quadrature_weights(s);
    const double sxy = source_normals(3 * s + 0) * source_forces(3 * s + 1) * quadrature_weights(s);
    const double sxz = source_normals(3 * s + 0) * source_forces(3 * s + 2) * quadrature_weights(s);
    const double syx = source_normals(3 * s + 1) * source_forces(3 * s + 0) * quadrature_weights(s);
    const double syy = source_normals(3 * s + 1) * source_forces(3 * s + 1) * quadrature_weights(s);
    const double syz = source_normals(3 * s + 1) * source_forces(3 * s + 2) * quadrature_weights(s);
    const double szx = source_normals(3 * s + 2) * source_forces(3 * s + 0) * quadrature_weights(s);
    const double szy = source_normals(3 * s + 2) * source_forces(3 * s + 1) * quadrature_weights(s);
    const double szz = source_normals(3 * s + 2) * source_forces(3 * s + 2) * quadrature_weights(s);

    double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
    coeff += (sxy + syx) * dx * dy;
    coeff += (sxz + szx) * dx * dz;
    coeff += (syz + szy) * dy * dz;
    coeff *= -scale_factor * rinv5;

    return mundy::Vector3d{dx * coeff, dy * coeff, dz * coeff};
  };

  mundy::direct_sum(space, num_target_points, num_source_points, stokes_double_layer_contribution,
                    impl::add_to_vector3_entries(target_velocities));
}

/// \brief Apply local drag to the sphere velocities v += 1/(6 pi mu r) f
template <class ExecutionSpace, typename SphereVelocityVectorType, typename SphereForceVectorType,
          typename SphereRadiusVectorType>
void apply_local_drag([[maybe_unused]] const ExecutionSpace& space,       //
                      const double viscosity,                             //
                      const SphereVelocityVectorType& sphere_velocities,  //
                      const SphereForceVectorType& sphere_forces,         //
                      const SphereRadiusVectorType& sphere_radii) {
  static_assert(impl::are_double_vectors_v<SphereVelocityVectorType, SphereForceVectorType, SphereRadiusVectorType>,
                "apply_local_drag: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<SphereVelocityVectorType, SphereForceVectorType, SphereRadiusVectorType>,
                "apply_local_drag: all views must share one memory space.");

  impl::require_length(sphere_velocities, 3 * sphere_radii.extent(0), "apply_local_drag");
  impl::require_length(sphere_forces, 3 * sphere_radii.extent(0), "apply_local_drag");

  const size_t num_spheres = sphere_radii.extent(0);
  const double scale = 1.0 / (6.0 * M_PI * viscosity);
  Kokkos::parallel_for(
      "apply_local_drag", Kokkos::RangePolicy<ExecutionSpace>(0, num_spheres), KOKKOS_LAMBDA(const size_t i) {
        const double r = sphere_radii(i);
        const double inv_drag_coeff = scale / r;
        sphere_velocities(3 * i) += inv_drag_coeff * sphere_forces(3 * i);
        sphere_velocities(3 * i + 1) += inv_drag_coeff * sphere_forces(3 * i + 1);
        sphere_velocities(3 * i + 2) += inv_drag_coeff * sphere_forces(3 * i + 2);
      });
}

/// \brief Fill the stokes double layer matrix times the surface normal
///
///   T is the stokes double layer kernel; its 3x3 block for target t and source s is
///     T_{t,s,ij} = -3 / (4*pi*viscosity) * r_i * r_j * r_k * normal_{s,k} / r**5 * quadrature_weight_s,
///   with r = x_t - y_s. The diagonal blocks (r = 0) are set to zero, so T is the punctured quadrature of the
///   double-layer potential (see the file header).
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_source_points The number of source points
/// \param[in] num_target_points The number of target points
/// \param[in] source_positions The positions of the source points (size num_source_points * 3)
/// \param[in] target_positions The positions of the target points (size num_target_points * 3)
/// \param[in] source_normals The normals of the source points (size num_source_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_source_points)
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceNormalVectorType, typename QuadratureWeightVectorType, typename MatrixType>
void fill_stokes_double_layer_matrix([[maybe_unused]] const ExecutionSpace& space,          //
                                     const double viscosity,                                //
                                     const size_t num_source_points,                        //
                                     const size_t num_target_points,                        //
                                     const SourcePosVectorType& source_positions,           //
                                     const TargetPosVectorType& target_positions,           //
                                     const SourceNormalVectorType& source_normals,          //
                                     const QuadratureWeightVectorType& quadrature_weights,  //
                                     const MatrixType& T) {
  static_assert(impl::are_double_vectors_v<SourcePosVectorType, TargetPosVectorType, SourceNormalVectorType,
                                           QuadratureWeightVectorType>,
                "fill_stokes_double_layer_matrix: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::are_double_matrices_v<MatrixType>,
                "fill_stokes_double_layer_matrix: the matrix must be a rank-2 Kokkos::View of double.");
  static_assert(impl::share_memory_space_v<SourcePosVectorType, TargetPosVectorType, SourceNormalVectorType,
                                           QuadratureWeightVectorType, MatrixType>,
                "fill_stokes_double_layer_matrix: all views must share one memory space.");

  impl::require_length(source_positions, 3 * num_source_points, "fill_stokes_double_layer_matrix");
  impl::require_length(target_positions, 3 * num_target_points, "fill_stokes_double_layer_matrix");
  impl::require_length(source_normals, 3 * num_source_points, "fill_stokes_double_layer_matrix");
  impl::require_length(quadrature_weights, num_source_points, "fill_stokes_double_layer_matrix");
  impl::require_length(T, 3 * num_target_points, "fill_stokes_double_layer_matrix");
  MUNDY_THROW_ASSERT(T.extent(1) == 3 * num_source_points, std::invalid_argument,
                     "fill_stokes_double_layer_matrix: T must have size 3 * num_source_points.");

  // Compute the scale factor
  const double scale_factor = -3.0 / (4.0 * M_PI * viscosity);
  Kokkos::parallel_for(
      "DoubleLayerMatrixFill", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_target_points, num_source_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        // Compute the distance vector
        const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
        const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
        const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);

        // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
        const double dr2 = dx * dx + dy * dy + dz * dz;
        const bool coincident = dr2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
        const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : dr2);
        const double rinv2 = rinv * rinv;
        const double rinv5 = rinv * rinv2 * rinv2;

        // Read in the surface normal at the source
        const double quadrature_weight = quadrature_weights(s);
        const double scaled_normal_s0 = source_normals(3 * s + 0) * quadrature_weight;
        const double scaled_normal_s1 = source_normals(3 * s + 1) * quadrature_weight;
        const double scaled_normal_s2 = source_normals(3 * s + 2) * quadrature_weight;

        // tmp_vec = (r dot normal) r, scaled by -3 / (4 pi r^5 viscosity)
        const double r_dot_scaled_normal = dx * scaled_normal_s0 + dy * scaled_normal_s1 + dz * scaled_normal_s2;
        const double tmp_vec0 = scale_factor * rinv5 * r_dot_scaled_normal * dx;
        const double tmp_vec1 = scale_factor * rinv5 * r_dot_scaled_normal * dy;
        const double tmp_vec2 = scale_factor * rinv5 * r_dot_scaled_normal * dz;

            // T (local) = r outer tmp_vec = r outer r (r dot normal) * -3 / (4 pi r^5 viscosity)
            // clang-format off
        T(t * 3 + 0, s * 3 + 0) = dx * tmp_vec0; T(t * 3 + 0, s * 3 + 1) = dx * tmp_vec1; T(t * 3 + 0, s * 3 + 2) = dx * tmp_vec2;
        T(t * 3 + 1, s * 3 + 0) = dy * tmp_vec0; T(t * 3 + 1, s * 3 + 1) = dy * tmp_vec1; T(t * 3 + 1, s * 3 + 2) = dy * tmp_vec2;
        T(t * 3 + 2, s * 3 + 0) = dz * tmp_vec0; T(t * 3 + 2, s * 3 + 1) = dz * tmp_vec1; T(t * 3 + 2, s * 3 + 2) = dz * tmp_vec2;
        // clang-format on
      });
}

/// \brief Turn the punctured stokes double layer matrix T of a closed surface into its interior trace J + T.
///
/// T is filled by fill_stokes_double_layer_matrix with source == target. On exit
///   T <- T - blockdiag(W_t) + (sigma / viscosity) I,   W_t = sum_s T_{t,s},
/// the discrete singularity-subtracted interior trace (see the file header). W_t approximates
/// PV T[1] = sigma / (2 viscosity) I, so the sign of sum_t tr(W_t) is checked against outward_normal.
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in,out] T The punctured stokes double layer matrix (size num_points * 3 x num_points * 3)
/// \param[in] outward_normal Whether the normals T was filled with point out of (true) or into (false) the enclosed
/// region
template <class ExecutionSpace, typename MatrixType>
void add_singularity_subtraction([[maybe_unused]] const ExecutionSpace& space, const double viscosity,
                                 const MatrixType& T, const bool outward_normal) {
  static_assert(impl::are_double_matrices_v<MatrixType>,
                "add_singularity_subtraction: the matrix must be a rank-2 Kokkos::View of double.");
  impl::require_positive_viscosity(viscosity, "add_singularity_subtraction");
  MUNDY_THROW_REQUIRE((T.extent(0) == T.extent(1)) && (T.extent(0) % 3 == 0), std::invalid_argument,
                      mundy::sink() << "add_singularity_subtraction: T must be square with a multiple of 3 rows but is "
                                    << T.extent(0) << " x " << T.extent(1));
  const size_t num_points = T.extent(0) / 3;

  // Add the singularity subtraction to T
  // Create a vector that has 1 in the x-component and 0 in the y and z components for each source point
  using Layout = typename MatrixType::array_layout;
  using MemorySpace = typename MatrixType::memory_space;
  Kokkos::View<double*, Layout, MemorySpace> e1("e1", 3 * num_points);
  Kokkos::View<double*, Layout, MemorySpace> e2("e2", 3 * num_points);
  Kokkos::View<double*, Layout, MemorySpace> e3("e3", 3 * num_points);
  Kokkos::parallel_for(
      "E1/2/3", Kokkos::RangePolicy<ExecutionSpace>(0, num_points), KOKKOS_LAMBDA(const size_t s) {
        e1(3 * s + 0) = 1.0;
        e1(3 * s + 1) = 0.0;
        e1(3 * s + 2) = 0.0;

        e2(3 * s + 0) = 0.0;
        e2(3 * s + 1) = 1.0;
        e2(3 * s + 2) = 0.0;

        e3(3 * s + 0) = 0.0;
        e3(3 * s + 1) = 0.0;
        e3(3 * s + 2) = 1.0;
      });

  // Compute w1 = T * e1, w2 = T * e2, w3 = T * e3
  Kokkos::View<double*, Layout, MemorySpace> w1("w1", 3 * num_points);
  Kokkos::View<double*, Layout, MemorySpace> w2("w2", 3 * num_points);
  Kokkos::View<double*, Layout, MemorySpace> w3("w3", 3 * num_points);
  KokkosBlas::gemv("N", 1.0, T, e1, 0.0, w1);
  KokkosBlas::gemv("N", 1.0, T, e2, 0.0, w2);
  KokkosBlas::gemv("N", 1.0, T, e3, 0.0, w3);

  // sum_t tr(W_t) approximates 3 num_points sigma / (2 viscosity), so its sign must match the declared orientation.
  double trace_sum = 0.0;
  Kokkos::parallel_reduce(
      "SingularitySubtractionTrace", Kokkos::RangePolicy<ExecutionSpace>(0, num_points),
      KOKKOS_LAMBDA(const size_t t, double& sum) { sum += w1(3 * t + 0) + w2(3 * t + 1) + w3(3 * t + 2); }, trace_sum);
  MUNDY_THROW_REQUIRE(
      outward_normal ? (trace_sum > 0.0) : (trace_sum < 0.0), std::invalid_argument,
      mundy::sink() << "add_singularity_subtraction: the normals were declared "
                    << (outward_normal ? "outward (outward_normal = true)" : "inward (outward_normal = false)")
                    << " but sum_t tr(W_t) = " << trace_sum
                    << " says otherwise (it approximates 3 N sigma / (2 viscosity)).");

  // Apply singularity subtraction (the interior trace of the file header).
  // T is stored target-row / source-column, so (T q)_t = sum_s T_{t,s} q_s with T_{t,s} the 3x3 block above and
  // T_{t,t} = 0 (the omitted singular node). Discretely,
  // sum_s T_{t,s} [q_s - q_t] + (sigma / viscosity) q_t
  //    = sum_s T_{t,s} q_s - sum_s T_{t,s} q_t + (sigma / viscosity) q_t
  //    = sum_s T_{t,s} q_s - (q1_t sum_s T_{t,s} e1_s + q2_t sum_s T_{t,s} e2_s + q3_t sum_s T_{t,s} e3_s)
  //      + (sigma / viscosity) q_t
  //    = sum_s T_{t,s} q_s - (q1_t w1_t + q2_t w2_t + q3_t w3_t) + (sigma / viscosity) q_t
  //
  // Note, T [q(y) - q(x)](x) is a vector of size 3.
  // w1, w2, w3 are vectors of size 3 * num_points
  // sum_s T_{t,s} e1_s = T[e1](x_t) = w1_t where e1 is a vector of size 3 * num_points with 1 in the x-component and
  // 0 in the y and z components of each source point (e1_s = [1 0 0]^T; likewise e2_s = [0 1 0]^T and
  // e3_s = [0 0 1]^T).
  //
  // q1_t w1_t + q2_t w2_t + q3_t w3_t is a vector of size 3 and equal [w1_t w2_t w3_t] [q1_t q2_t q3_t]^T,
  // which is the multiplication of a 3x3 matrix with a 3x1 vector. That matrix, W_t = [w1_t w2_t w3_t] =
  // sum_s T_{t,s}, is the punctured quadrature of PV T[1](x_t); subtracting it removes the singular part exactly and
  // replaces it with the analytic coefficient.
  //
  // Hence, the matrix form of T with singularity subtraction is
  //   T - [w1_0 w2_0 w3_0                              ] + (sigma / viscosity) I
  //       [               w1_1 w2_1 w3_1               ]
  //       [                              w1_2 w2_2 w3_2]
  //       [                                             ...]
  const double target_coefficient = impl::interior_trace_coefficient(viscosity, outward_normal);
  Kokkos::parallel_for(
      "SingularitySubtraction", Kokkos::RangePolicy<ExecutionSpace>(0, num_points), KOKKOS_LAMBDA(const size_t ts) {
        T(ts * 3 + 0, ts * 3 + 0) -= w1(3 * ts + 0);
        T(ts * 3 + 1, ts * 3 + 0) -= w1(3 * ts + 1);
        T(ts * 3 + 2, ts * 3 + 0) -= w1(3 * ts + 2);

        T(ts * 3 + 0, ts * 3 + 1) -= w2(3 * ts + 0);
        T(ts * 3 + 1, ts * 3 + 1) -= w2(3 * ts + 1);
        T(ts * 3 + 2, ts * 3 + 1) -= w2(3 * ts + 2);

        T(ts * 3 + 0, ts * 3 + 2) -= w3(3 * ts + 0);
        T(ts * 3 + 1, ts * 3 + 2) -= w3(3 * ts + 1);
        T(ts * 3 + 2, ts * 3 + 2) -= w3(3 * ts + 2);

        T(ts * 3 + 0, ts * 3 + 0) += target_coefficient;
        T(ts * 3 + 1, ts * 3 + 1) += target_coefficient;
        T(ts * 3 + 2, ts * 3 + 2) += target_coefficient;
      });
}

/// \brief Add the null-space completion N to a periphery operator.
///
///   N_{t,s,ij} = normal_{t,i} * normal_{s,j} * quadrature_weight_s / (viscosity * |Gamma|)
///
/// J + T alone has a one-dimensional null space whose density produces no interior flow, so N's scale does not change
/// the solved flow; 1/(viscosity |Gamma|) gives N the eigenvalue 1/viscosity on n, keeping M well conditioned.
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] surface_area The surface area |Gamma| = sum_s quadrature_weight_s
/// \param[in] normals The surface normals (size num_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_points)
/// \param[in,out] T The matrix N is added to (size num_points * 3 x num_points * 3)
template <class ExecutionSpace, typename NormalVectorType, typename QuadratureWeightVectorType, typename MatrixType>
void add_complementary_matrix([[maybe_unused]] const ExecutionSpace& space,          //
                              const double viscosity,                                //
                              const double surface_area,                             //
                              const NormalVectorType& normals,                       //
                              const QuadratureWeightVectorType& quadrature_weights,  //
                              const MatrixType& T) {
  static_assert(impl::are_double_vectors_v<NormalVectorType, QuadratureWeightVectorType>,
                "add_complementary_matrix: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::are_double_matrices_v<MatrixType>,
                "add_complementary_matrix: the matrix must be a rank-2 Kokkos::View of double.");
  static_assert(impl::share_memory_space_v<NormalVectorType, QuadratureWeightVectorType, MatrixType>,
                "add_complementary_matrix: all views must share one memory space.");

  impl::require_positive_viscosity(viscosity, "add_complementary_matrix");
  MUNDY_THROW_REQUIRE(
      surface_area > 0.0 && std::isfinite(surface_area), std::invalid_argument,
      mundy::sink() << "add_complementary_matrix: surface_area must be positive and finite but is " << surface_area);
  MUNDY_THROW_REQUIRE((T.extent(0) == T.extent(1)) && (T.extent(0) % 3 == 0), std::invalid_argument,
                      mundy::sink() << "add_complementary_matrix: T must be square with a multiple of 3 rows but is "
                                    << T.extent(0) << " x " << T.extent(1));
  const size_t num_points = T.extent(0) / 3;
  impl::require_length(normals, 3 * num_points, "add_complementary_matrix");
  impl::require_length(quadrature_weights, num_points, "add_complementary_matrix");

  // Add the complementary matrix
  const double scale = 1.0 / (viscosity * surface_area);
  Kokkos::parallel_for(
      "ComplementaryMatrix", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_points, num_points}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        const double normal_s0 = normals(3 * s + 0);
        const double normal_s1 = normals(3 * s + 1);
        const double normal_s2 = normals(3 * s + 2);

        const double normal_t0 = normals(3 * t + 0);
        const double normal_t1 = normals(3 * t + 1);
        const double normal_t2 = normals(3 * t + 2);

        const double weighted_normal_s0 = normal_s0 * quadrature_weights(s) * scale;
        const double weighted_normal_s1 = normal_s1 * quadrature_weights(s) * scale;
        const double weighted_normal_s2 = normal_s2 * quadrature_weights(s) * scale;

        T(t * 3 + 0, s * 3 + 0) += normal_t0 * weighted_normal_s0;
        T(t * 3 + 0, s * 3 + 1) += normal_t0 * weighted_normal_s1;
        T(t * 3 + 0, s * 3 + 2) += normal_t0 * weighted_normal_s2;

        T(t * 3 + 1, s * 3 + 0) += normal_t1 * weighted_normal_s0;
        T(t * 3 + 1, s * 3 + 1) += normal_t1 * weighted_normal_s1;
        T(t * 3 + 1, s * 3 + 2) += normal_t1 * weighted_normal_s2;

        T(t * 3 + 2, s * 3 + 0) += normal_t2 * weighted_normal_s0;
        T(t * 3 + 2, s * 3 + 1) += normal_t2 * weighted_normal_s1;
        T(t * 3 + 2, s * 3 + 2) += normal_t2 * weighted_normal_s2;
      });
}

/// \brief Fill the periphery's second-kind operator M = J + T + N (interior trace).
///
/// M f = u is the second kind Fredholm integral equation for the Stokes flow a closed periphery induces to satisfy the
/// surface velocity u (for no-slip, u = -u_slip); f is a stresslet surface density (see the file header).
///   - J + T: the singularity-subtracted interior trace of the double layer, J = sigma / (2 viscosity) I
///     (add_singularity_subtraction)
///   - N: the null-space completion (add_complementary_matrix)
/// Reversing the normals reverses J + T (N is invariant), so f changes sign and the interior flow T[f] is unchanged.
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_points The number of surface points
/// \param[in] positions The surface positions (size num_points * 3)
/// \param[in] normals The surface normals (size num_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_points)
/// \param[out] M The matrix (size num_points * 3 x num_points * 3)
/// \param[in] outward_normal Whether the normals point out of (true) or into (false) the enclosed region; checked
/// against the geometry
template <class ExecutionSpace, typename PosVectorType, typename NormalVectorType, typename QuadratureWeightVectorType,
          typename MatrixType>
void fill_skfie_matrix(const ExecutionSpace& space,                           //
                       const double viscosity,                                //
                       const size_t num_points,                               //
                       const PosVectorType& positions,                        //
                       const NormalVectorType& normals,                       //
                       const QuadratureWeightVectorType& quadrature_weights,  //
                       const MatrixType& M,                                   //
                       const bool outward_normal) {
  static_assert(impl::are_double_vectors_v<PosVectorType, NormalVectorType, QuadratureWeightVectorType>,
                "fill_skfie_matrix: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::are_double_matrices_v<MatrixType>,
                "fill_skfie_matrix: the matrix must be a rank-2 Kokkos::View of double.");
  static_assert(impl::share_memory_space_v<PosVectorType, NormalVectorType, QuadratureWeightVectorType, MatrixType>,
                "fill_skfie_matrix: all views must share one memory space.");

  impl::require_positive_viscosity(viscosity, "fill_skfie_matrix");
  impl::require_length(positions, 3 * num_points, "fill_skfie_matrix");
  impl::require_length(normals, 3 * num_points, "fill_skfie_matrix");
  impl::require_length(quadrature_weights, num_points, "fill_skfie_matrix");
  const double surface_area = impl::validated_surface_area(space, num_points, positions, normals, quadrature_weights,
                                                           outward_normal, "fill_skfie_matrix");

  // Fill the stokes double layer matrix
  fill_stokes_double_layer_matrix(space, viscosity, num_points, num_points, positions, positions, normals,
                                  quadrature_weights, M);

  // Add singularity subtraction
  add_singularity_subtraction(space, viscosity, M, outward_normal);

  // Add the complementary matrix
  add_complementary_matrix(space, viscosity, surface_area, normals, quadrature_weights, M);
}

/// \brief Accumulate u += M f, the matrix-free form of fill_skfie_matrix.
///
/// Per target t (see fill_skfie_matrix for sigma, J, T, and N):
///   u_t += sum_{s != t} T_{t,s} (f_s - f_t) + (sigma / viscosity) f_t
///          + normal_t / (viscosity |Gamma|) sum_s quadrature_weight_s normal_s . f_s
///
/// \param space The execution space
/// \param[in] viscosity The viscosity
/// \param[in] num_points The number of surface points
/// \param[in] positions The surface positions (size num_points * 3)
/// \param[in] normals The surface normals (size num_points * 3)
/// \param[in] quadrature_weights The quadrature weights (size num_points)
/// \param[in] forces The density f (size num_points * 3)
/// \param[out] velocities M f is accumulated into this vector (size num_points * 3)
/// \param[in] outward_normal Whether the normals point out of (true) or into (false) the enclosed region; checked
/// against the geometry
template <class ExecutionSpace, typename PosVectorType, typename NormalVectorType, typename QuadratureWeightVectorType,
          typename ForceVectorType, typename VelocityVectorType>
void apply_skfie(const ExecutionSpace& space,                           //
                 const double viscosity,                                //
                 const size_t num_points,                               //
                 const PosVectorType& positions,                        //
                 const NormalVectorType& normals,                       //
                 const QuadratureWeightVectorType& quadrature_weights,  //
                 const ForceVectorType& forces,                         //
                 const VelocityVectorType& velocities,                  //
                 const bool outward_normal) {
  static_assert(impl::are_double_vectors_v<PosVectorType, NormalVectorType, QuadratureWeightVectorType, ForceVectorType,
                                           VelocityVectorType>,
                "apply_skfie: inputs must be rank-1 Kokkos::Views of double.");
  static_assert(impl::share_memory_space_v<PosVectorType, NormalVectorType, QuadratureWeightVectorType, ForceVectorType,
                                           VelocityVectorType>,
                "apply_skfie: all views must share one memory space.");

  impl::require_positive_viscosity(viscosity, "apply_skfie");
  impl::require_length(positions, 3 * num_points, "apply_skfie");
  impl::require_length(normals, 3 * num_points, "apply_skfie");
  impl::require_length(quadrature_weights, num_points, "apply_skfie");
  impl::require_length(forces, 3 * num_points, "apply_skfie");
  impl::require_length(velocities, 3 * num_points, "apply_skfie");
  const double surface_area = impl::validated_surface_area(space, num_points, positions, normals, quadrature_weights,
                                                           outward_normal, "apply_skfie");
  const double complementary_scale = 1.0 / (viscosity * surface_area);

  // Launch the parallel kernel
  const double scale_factor = 3.0 / (4.0 * M_PI * viscosity);
  auto skfie_contribution = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    // Compute the distance vector
    const double dx = positions(3 * t + 0) - positions(3 * s + 0);
    const double dy = positions(3 * t + 1) - positions(3 * s + 1);
    const double dz = positions(3 * t + 2) - positions(3 * s + 2);

    // Read in the necessary values
    const double normal_s0 = normals(3 * s + 0);
    const double normal_s1 = normals(3 * s + 1);
    const double normal_s2 = normals(3 * s + 2);
    const double normal_t0 = normals(3 * t + 0);
    const double normal_t1 = normals(3 * t + 1);
    const double normal_t2 = normals(3 * t + 2);
    const double quadrature_weight_s = quadrature_weights(s);
    const double force_s0 = forces(3 * s + 0);
    const double force_s1 = forces(3 * s + 1);
    const double force_s2 = forces(3 * s + 2);
    const double force_t0 = forces(3 * t + 0);
    const double force_t1 = forces(3 * t + 1);
    const double force_t2 = forces(3 * t + 2);

    // Compute rinv5. If r is zero, set rinv5 to zero, effectively setting the diagonal of K to zero.
    const double dr2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = dr2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : dr2);
    const double rinv2 = rinv * rinv;
    const double rinv5 = rinv * rinv2 * rinv2;

    // Compute the singularity-subtracted double layer potential, K(x_t, y_s) (f_s - f_t)
    const double sxx = normal_s0 * (force_s0 - force_t0) * quadrature_weight_s;
    const double sxy = normal_s0 * (force_s1 - force_t1) * quadrature_weight_s;
    const double sxz = normal_s0 * (force_s2 - force_t2) * quadrature_weight_s;
    const double syx = normal_s1 * (force_s0 - force_t0) * quadrature_weight_s;
    const double syy = normal_s1 * (force_s1 - force_t1) * quadrature_weight_s;
    const double syz = normal_s1 * (force_s2 - force_t2) * quadrature_weight_s;
    const double szx = normal_s2 * (force_s0 - force_t0) * quadrature_weight_s;
    const double szy = normal_s2 * (force_s1 - force_t1) * quadrature_weight_s;
    const double szz = normal_s2 * (force_s2 - force_t2) * quadrature_weight_s;

    double coeff = sxx * dx * dx + syy * dy * dy + szz * dz * dz;
    coeff += (sxy + syx) * dx * dy;
    coeff += (sxz + szx) * dx * dz;
    coeff += (syz + szy) * dy * dz;
    coeff *= -scale_factor * rinv5;

    // Compute the complementarity term v += normal(t) * normal(s) dot f(s) w(s) / (viscosity |Gamma|)
    const double scaled_normal_dot_force = (normal_s0 * force_s0 + normal_s1 * force_s1 + normal_s2 * force_s2) *
                                           quadrature_weight_s * complementary_scale;

    return mundy::Vector3d{dx * coeff + scaled_normal_dot_force * normal_t0,
                           dy * coeff + scaled_normal_dot_force * normal_t1,
                           dz * coeff + scaled_normal_dot_force * normal_t2};
  };

  mundy::direct_sum(space, num_points, num_points, skfie_contribution, impl::add_to_vector3_entries(velocities));

  // The analytic target term (sigma / viscosity) f_t, added once per target node -- NOT inside the per-source sweep.
  const double target_coefficient = impl::interior_trace_coefficient(viscosity, outward_normal);
  Kokkos::parallel_for(
      "apply_skfie_target_term", Kokkos::RangePolicy<ExecutionSpace>(0, 3 * num_points),
      KOKKOS_LAMBDA(const size_t i) { velocities(i) += target_coefficient * forces(i); });
}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
/// \brief The operator M of fill_skfie_matrix as a matrix-free Mundy LinearOperator.
///
/// apply(f, u) computes u = M f by evaluating apply_skfie against the surface geometry it holds; no dense matrix is
/// formed, so mobile geometry is handled by re-applying against updated positions/normals. Its views live in the
/// solve's memory space (\c ExecSpace::memory_space), which apply_skfie requires to match the force/velocity views
/// Belos hands it. Domain and range are the 3*num_nodes flattened surface vectors.
template <class ExecSpace>
struct SkfieOp {
  using exec_space = ExecSpace;
  using memory_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, memory_space>;

  double viscosity;
  size_t num_nodes;
  view_t surface_positions;   //!< size 3*num_nodes
  view_t surface_normals;     //!< size 3*num_nodes
  view_t quadrature_weights;  //!< size num_nodes
  bool outward_normal;        //!< whether surface_normals point out of (true) or into (false) the enclosed region

  size_t domain_size() const {
    return 3 * num_nodes;
  }
  size_t range_size() const {
    return 3 * num_nodes;
  }
  auto make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "skfie_domain"), 3 * num_nodes);
  }
  auto make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "skfie_range"), 3 * num_nodes);
  }

  // u = M f. apply_skfie accumulates into u via atomic_add, so u must be zeroed first.
  template <class ForceVector, class VelocityVector>
  void apply(const ForceVector& f, VelocityVector& u) const {
    Kokkos::deep_copy(u, 0.0);
    apply_skfie(exec_space{}, viscosity, num_nodes, surface_positions, surface_normals, quadrature_weights, f, u,
                outward_normal);
  }
};
#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA

/// \brief How the periphery inverts the self-interaction operator M to recover surface forces from a slip velocity.
enum class InverseMethod {
  Direct,          //!< Precompute the dense inverse M^{-1} and apply it with a gemv.
  MatrixFreeGMRES  //!< Solve M f = u matrix-free with Belos GMRES; no dense matrix is formed (requires Belos+Tpetra).
};

template <typename ExecSpace>
class PeripheryT {
 public:
  //! \name Types
  //@{

  using DeviceExecutionSpace = ExecSpace;
  using DeviceMemorySpace = typename ExecSpace::memory_space;
  using device_vector_t = Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>;
  using device_matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, DeviceMemorySpace>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor
  PeripheryT() = delete;

  /// \brief No copy constructor
  PeripheryT(const PeripheryT&) = delete;

  /// \brief No copy assignment
  PeripheryT& operator=(const PeripheryT&) = delete;

  /// \brief Default move constructor
  PeripheryT(PeripheryT&&) = default;

  /// \brief Default move assignment
  PeripheryT& operator=(PeripheryT&&) = default;

  /// \brief Destructor
  ~PeripheryT() = default;

  /// \brief Constructor
  PeripheryT(const size_t num_surface_nodes, const double viscosity)
      : num_surface_nodes_(num_surface_nodes),
        viscosity_(viscosity),
        outward_normal_(false),
        is_surface_positions_set_(false),
        is_surface_normals_set_(false),
        is_quadrature_weights_set_(false),
        is_inverse_self_interaction_matrix_set_(false),
        surface_positions_("surface_positions", 3 * num_surface_nodes_),
        surface_normals_("surface_normals", 3 * num_surface_nodes_),
        quadrature_weights_("quadrature_weights", num_surface_nodes_),
        M_inv_("M_inv", 3 * num_surface_nodes_, 3 * num_surface_nodes_) {
    impl::require_positive_viscosity(viscosity_, "PeripheryT");

    // Initialize host Kokkos mirrors
    surface_positions_host_ = Kokkos::create_mirror_view(surface_positions_);
    surface_normals_host_ = Kokkos::create_mirror_view(surface_normals_);
    quadrature_weights_host_ = Kokkos::create_mirror_view(quadrature_weights_);
    M_inv_host_ = Kokkos::create_mirror_view(M_inv_);
  }
  //@}

  //! \name Setters
  //@{

  /// Setters initialize the periphery. Each array may be passed as a Kokkos view, a raw pointer, or a filename to
  /// be read from disk.

  /// \brief Set the surface positions
  ///
  /// \param surface_positions The surface positions (size num_nodes * 3)
  template <class MemorySpace, class Layout>
  PeripheryT& set_surface_positions(const Kokkos::View<double*, Layout, MemorySpace>& surface_positions) {
    MUNDY_THROW_ASSERT(surface_positions.extent(0) == 3 * num_surface_nodes_, std::invalid_argument,
                       "set_surface_positions: surface_positions must have size 3 * num_surface_nodes.");
    Kokkos::deep_copy(surface_positions_, surface_positions);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface positions
  ///
  /// \param surface_positions The surface positions (size num_nodes * 3)
  PeripheryT& set_surface_positions(const double* surface_positions) {
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const size_t idx = 3 * i + j;
        surface_positions_host_(idx) = surface_positions[idx];
      }
    }
    Kokkos::deep_copy(surface_positions_, surface_positions_host_);
    is_surface_positions_set_ = true;

    return *this;
  }

  /// \brief Set the surface positions
  ///
  /// \param surface_positions_filename A Matrix Market file of the surface positions (num_nodes * 3 x 1)
  PeripheryT& set_surface_positions(const std::string& surface_positions_filename) {
    return set_surface_positions(impl::read_matrix_market_with_extents<device_vector_t>(
        surface_positions_filename, 3 * num_surface_nodes_, 1, "set_surface_positions"));
  }

  /// \brief Set the surface normals and declare their orientation
  ///
  /// \param surface_normals The surface normals (size num_nodes * 3)
  /// \param outward_normal Whether the normals point out of (true) or into (false) the enclosed fluid; checked against
  /// the geometry when the inverse is built
  template <class MemorySpace, class Layout>
  PeripheryT& set_surface_normals(const Kokkos::View<double*, Layout, MemorySpace>& surface_normals,
                                  const bool outward_normal) {
    MUNDY_THROW_ASSERT(surface_normals.extent(0) == 3 * num_surface_nodes_, std::invalid_argument,
                       "set_surface_normals: surface_normals must have size 3 * num_surface_nodes.");
    Kokkos::deep_copy(surface_normals_, surface_normals);
    outward_normal_ = outward_normal;
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals and declare their orientation
  ///
  /// \param surface_normals The surface normals (size num_nodes * 3)
  /// \param outward_normal Whether the normals point out of (true) or into (false) the enclosed fluid; checked against
  /// the geometry when the inverse is built
  PeripheryT& set_surface_normals(const double* surface_normals, const bool outward_normal) {
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const size_t idx = 3 * i + j;
        surface_normals_host_(idx) = surface_normals[idx];
      }
    }
    Kokkos::deep_copy(surface_normals_, surface_normals_host_);
    outward_normal_ = outward_normal;
    is_surface_normals_set_ = true;

    return *this;
  }

  /// \brief Set the surface normals and declare their orientation
  ///
  /// \param surface_normals_filename A Matrix Market file of the surface normals (num_nodes * 3 x 1)
  /// \param outward_normal Whether the normals point out of (true) or into (false) the enclosed fluid; checked against
  /// the geometry when the inverse is built
  PeripheryT& set_surface_normals(const std::string& surface_normals_filename, const bool outward_normal) {
    return set_surface_normals(impl::read_matrix_market_with_extents<device_vector_t>(
                                   surface_normals_filename, 3 * num_surface_nodes_, 1, "set_surface_normals"),
                               outward_normal);
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights The quadrature weights (size num_nodes)
  template <class MemorySpace, class Layout>
  PeripheryT& set_quadrature_weights(const Kokkos::View<double*, Layout, MemorySpace>& quadrature_weights) {
    MUNDY_THROW_ASSERT(quadrature_weights.extent(0) == num_surface_nodes_, std::invalid_argument,
                       "set_quadrature_weights: quadrature_weights must have size num_surface_nodes.");
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights The quadrature weights (size num_nodes)
  PeripheryT& set_quadrature_weights(const double* quadrature_weights) {
    for (size_t i = 0; i < num_surface_nodes_; ++i) {
      quadrature_weights_host_(i) = quadrature_weights[i];
    }
    Kokkos::deep_copy(quadrature_weights_, quadrature_weights_host_);
    is_quadrature_weights_set_ = true;

    return *this;
  }

  /// \brief Set the quadrature weights
  ///
  /// \param quadrature_weights_filename A Matrix Market file of the quadrature weights (num_nodes x 1)
  PeripheryT& set_quadrature_weights(const std::string& quadrature_weights_filename) {
    return set_quadrature_weights(impl::read_matrix_market_with_extents<device_vector_t>(
        quadrature_weights_filename, num_surface_nodes_, 1, "set_quadrature_weights"));
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param M_inv The precomputed matrix (size 3 * num_nodes x 3 * num_nodes)
  template <class MemorySpace>
  PeripheryT& set_inverse_self_interaction_matrix(
      const Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>& M_inv) {
    MUNDY_THROW_ASSERT(
        (M_inv.extent(0) == 3 * num_surface_nodes_) && (M_inv.extent(1) == 3 * num_surface_nodes_),
        std::invalid_argument,
        "set_inverse_self_interaction_matrix: M_inv must have size 3 * num_surface_nodes x 3 * num_surface_nodes.");
    Kokkos::deep_copy(M_inv_, M_inv);
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param M_inv_flat The precomputed matrix, row-major (size 3 * num_nodes x 3 * num_nodes)
  PeripheryT& set_inverse_self_interaction_matrix(const double* M_inv_flat) {
    for (size_t i = 0; i < 3 * num_surface_nodes_; ++i) {
      for (size_t j = 0; j < 3 * num_surface_nodes_; ++j) {
        const size_t idx = i * 3 * num_surface_nodes_ + j;
        M_inv_host_(i, j) = M_inv_flat[idx];
      }
    }
    Kokkos::deep_copy(M_inv_, M_inv_host_);
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }

  /// \brief Set the precomputed matrix
  ///
  /// \param inverse_self_interaction_matrix_filename A Matrix Market file of the precomputed matrix, as written by
  /// write_inverse_self_interaction_matrix (num_nodes * 3 x num_nodes * 3)
  PeripheryT& set_inverse_self_interaction_matrix(const std::string& inverse_self_interaction_matrix_filename) {
    return set_inverse_self_interaction_matrix(impl::read_matrix_market_with_extents<device_matrix_t>(
        inverse_self_interaction_matrix_filename, 3 * num_surface_nodes_, 3 * num_surface_nodes_,
        "set_inverse_self_interaction_matrix"));
  }
  //@}

  //! \name Public member functions
  //@{

  /// \brief Build the dense inverse M^{-1} of the self-interaction matrix, for the direct inverse method.
  ///
  /// Save it with write_inverse_self_interaction_matrix and restore it with set_inverse_self_interaction_matrix.
  PeripheryT& build_inverse_self_interaction_matrix() {
    MUNDY_THROW_REQUIRE(is_surface_positions_set_ && is_surface_normals_set_ && is_quadrature_weights_set_,
                        std::runtime_error,
                        "build_inverse_self_interaction_matrix: surface_positions, surface_normals, and "
                        "quadrature_weights must be set before calling this function.");

    // Fill the self-interaction matrix using temporary storage
    Kokkos::View<double**, Kokkos::LayoutLeft, DeviceMemorySpace> M("M", 3 * num_surface_nodes_,
                                                                    3 * num_surface_nodes_);
    fill_skfie_matrix(DeviceExecutionSpace(), viscosity_, num_surface_nodes_, surface_positions_, surface_normals_,
                      quadrature_weights_, M, outward_normal_);

    // Invert M; its LU factors overwrite the temporary
    mundy::invert(DeviceExecutionSpace(), M, M_inv_);
    is_inverse_self_interaction_matrix_set_ = true;

    return *this;
  }

  /// \brief Write the dense inverse M^{-1} to a Matrix Market file, exactly.
  ///
  /// set_inverse_self_interaction_matrix reads it back. Like every Matrix Market write, this is serial: under MPI,
  /// call it from one process only.
  const PeripheryT& write_inverse_self_interaction_matrix(
      const std::string& inverse_self_interaction_matrix_filename) const {
    MUNDY_THROW_REQUIRE(is_inverse_self_interaction_matrix_set_, std::runtime_error,
                        "write_inverse_self_interaction_matrix: build_inverse_self_interaction_matrix() or "
                        "set_inverse_self_interaction_matrix() must be called first.");
    mundy::write_matrix_market(inverse_self_interaction_matrix_filename, M_inv_);

    return *this;
  }

  /// \brief Compute the surface forces induced by external flow on the surface
  ///
  /// \param[in] external_flow_velocity The external flow velocity (size num_nodes x 3)
  /// \param[out] surface_forces The surface forces induced by enforcing no-slip on the surface (size num_nodes x 3)
  PeripheryT& compute_surface_forces(
      const Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>& external_flow_velocity,
      Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>& surface_forces) {
    // Both paths accumulate surface_forces += -M^{-1} u_slip: the negative balances the imposed slip velocity, and
    // the result is added onto whatever surface_forces already holds.
    if (inverse_method_ == InverseMethod::Direct) {
      MUNDY_THROW_REQUIRE(is_inverse_self_interaction_matrix_set_, std::runtime_error,
                          "compute_surface_forces: build_inverse_self_interaction_matrix() or "
                          "set_inverse_self_interaction_matrix() must be called before using the direct inverse.");
      KokkosBlas::gemv(DeviceExecutionSpace(), "N", -1.0, M_inv_, external_flow_velocity, 1.0, surface_forces);
      return *this;
    }

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
    // MatrixFreeGMRES: solve M f = u_slip for f = M^{-1} u_slip, then accumulate -f into surface_forces.
    MUNDY_THROW_REQUIRE(is_matrix_free_inverse_set_, std::runtime_error,
                        "compute_surface_forces: build_matrix_free_inverse() must be called before using the "
                        "matrix-free inverse.");
    belos_inv_op_->apply(external_flow_velocity, mf_solution_);
    auto solution = mf_solution_;
    auto forces = surface_forces;
    Kokkos::parallel_for(
        "compute_surface_forces_accumulate", Kokkos::RangePolicy<DeviceExecutionSpace>(0, forces.extent(0)),
        KOKKOS_LAMBDA(const size_t i) { forces(i) -= solution(i); });
    return *this;
#else
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "compute_surface_forces: MatrixFreeGMRES requires the Belos and Tpetra TPLs, which are not "
                        "enabled in this build.");
    return *this;
#endif
  }

  /// \brief Select how compute_surface_forces inverts M (direct dense inverse vs. matrix-free GMRES).
  PeripheryT& set_inverse_method(InverseMethod method) {
    inverse_method_ = method;
    return *this;
  }

  InverseMethod get_inverse_method() const {
    return inverse_method_;
  }

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
  /// \brief Set the solver configuration used by the matrix-free inverse (applied at the next build).
  PeripheryT& set_belos_config(const mundy::BelosConfig<double>& belos_config) {
    belos_config_ = belos_config;
    return *this;
  }

  /// \brief Build (or rebuild) the matrix-free GMRES inverse over the current surface geometry.
  ///
  /// Builds a matrix-free GMRES inverse of the surface self-interaction operator; no dense matrix is formed.
  /// Re-call after the geometry moves to refresh the operator against the new positions/normals. Requires
  /// surface_positions, surface_normals, and quadrature_weights to be set. The declared normal orientation is checked
  /// here, so a mismatch fails at build time rather than inside the first GMRES apply.
  PeripheryT& build_matrix_free_inverse() {
    MUNDY_THROW_REQUIRE(is_surface_positions_set_ && is_surface_normals_set_ && is_quadrature_weights_set_,
                        std::runtime_error,
                        "build_matrix_free_inverse: surface_positions, surface_normals, and quadrature_weights must "
                        "be set before calling this function.");
    impl::validated_surface_area(DeviceExecutionSpace(), num_surface_nodes_, surface_positions_, surface_normals_,
                                 quadrature_weights_, outward_normal_, "build_matrix_free_inverse");

    // The operator's geometry lives in the solve's memory space; copy from the periphery's device views (a
    // same-space, cheap copy on a host build). SkfieOp shares these handles, so a later re-copy refreshes it.
    mf_surface_positions_ =
        MatrixFreeView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mf_surface_positions"), 3 * num_surface_nodes_);
    mf_surface_normals_ =
        MatrixFreeView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mf_surface_normals"), 3 * num_surface_nodes_);
    mf_quadrature_weights_ =
        MatrixFreeView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mf_quadrature_weights"), num_surface_nodes_);
    Kokkos::deep_copy(mf_surface_positions_, surface_positions_);
    Kokkos::deep_copy(mf_surface_normals_, surface_normals_);
    Kokkos::deep_copy(mf_quadrature_weights_, quadrature_weights_);

    // M^{-1} u, kept in the periphery's device space so the accumulate in compute_surface_forces stays in one space.
    // Each solve starts from it, so it starts at zero.
    mf_solution_ = Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>("mf_solution", 3 * num_surface_nodes_);

    SkfieOpType op{viscosity_,          num_surface_nodes_,     mf_surface_positions_,
                   mf_surface_normals_, mf_quadrature_weights_, outward_normal_};
    belos_inv_op_.emplace(MatrixFreeBackend{}, std::move(op), belos_config_, mundy::NoPreconditioner{});
    is_matrix_free_inverse_set_ = true;
    return *this;
  }

  /// \brief The iteration count / residual / convergence of the most recent matrix-free solve.
  const mundy::BelosResult<double>& get_last_matrix_free_result() const {
    MUNDY_THROW_REQUIRE(belos_inv_op_.has_value(), std::runtime_error,
                        "get_last_matrix_free_result: the matrix-free inverse has not been built.");
    return belos_inv_op_->last_result();
  }
#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA
  //@}

  //! \name Getters
  //@{

  /// \brief Get the number of nodes
  size_t get_num_nodes() const {
    return num_surface_nodes_;
  }

  /// \brief Get the viscosity
  ///
  /// \return The viscosity
  double get_viscosity() const {
    return viscosity_;
  }

  /// \brief Get the surface positions
  ///
  /// \return The surface positions
  const Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>& get_surface_positions() const {
    return surface_positions_;
  }

  /// \brief Get the surface normals
  const Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>& get_surface_normals() const {
    return surface_normals_;
  }

  /// \brief Get the quadrature weights
  const Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>& get_quadrature_weights() const {
    return quadrature_weights_;
  }

  /// \brief Get the dense inverse of the self-interaction matrix; throws unless it was built or set
  const Kokkos::View<double**, Kokkos::LayoutLeft, DeviceMemorySpace>& get_M_inv() const {
    MUNDY_THROW_REQUIRE(is_inverse_self_interaction_matrix_set_, std::runtime_error,
                        "get_M_inv: build_inverse_self_interaction_matrix() or set_inverse_self_interaction_matrix() "
                        "must be called first.");
    return M_inv_;
  }
  //@}

 private:
  //! \name Private member variables
  //@{

  size_t num_surface_nodes_;  //!< The number of nodes
  double viscosity_;          //!< The viscosity
  bool outward_normal_;       //!< Whether the surface normals point out of (true) or into (false) the enclosed fluid
                              //!< (declared by set_surface_normals)

  bool is_surface_positions_set_;                //!< Whether the surface positions have been set
  bool is_surface_normals_set_;                  //!< Whether the surface normals have been set
  bool is_quadrature_weights_set_;               //!< Whether the quadrature weights have been set
  bool is_inverse_self_interaction_matrix_set_;  //!< Whether the inverse of the self-interaction matrix has been set

  // Host Kokkos views
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>
      surface_positions_host_;                                                         //!< The surface positions (host)
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_normals_host_;  //!< The surface normals (host)
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>
      quadrature_weights_host_;  //!< The quadrature weights (host)
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>
      M_inv_host_;  //!< The inverse of the self-interaction matrix (host)

  // Device Kokkos views
  Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace> surface_positions_;  //!< The node positions (device)
  Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace> surface_normals_;    //!< The surface normals (device)
  Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace>
      quadrature_weights_;  //!< The quadrature weights (device)
  Kokkos::View<double**, Kokkos::LayoutLeft, DeviceMemorySpace>
      M_inv_;  //!< The inverse of the self-interaction matrix (device)

  InverseMethod inverse_method_ = InverseMethod::Direct;  //!< Which inverse compute_surface_forces uses

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
  // BelosSolveSession requires the backend execution space to equal the Tpetra default Node execution space, so the
  // matrix-free types below are pinned to that space independent of the class's ExecSpace template parameter.
  using MatrixFreeExecSpace = typename Tpetra::Map<>::node_type::execution_space;
  using MatrixFreeBackend = mundy::KokkosBackend<MatrixFreeExecSpace>;
  using SkfieOpType = SkfieOp<MatrixFreeExecSpace>;
  using MatrixFreeView = typename SkfieOpType::view_t;
  using BelosInvOpType = mundy::BelosInvOp<MatrixFreeBackend, SkfieOpType, mundy::NoPreconditioner>;

  mundy::BelosConfig<double> belos_config_{};   //!< Belos GMRES configuration for the matrix-free inverse
  bool is_matrix_free_inverse_set_ = false;     //!< Whether build_matrix_free_inverse has been called
  MatrixFreeView mf_surface_positions_;         //!< Surface geometry in the solve's memory space (3*num_nodes)
  MatrixFreeView mf_surface_normals_;           //!< (3*num_nodes)
  MatrixFreeView mf_quadrature_weights_;        //!< (num_nodes)
  std::optional<BelosInvOpType> belos_inv_op_;  //!< The matrix-free GMRES inverse over SkfieOp
  Kokkos::View<double*, Kokkos::LayoutLeft, DeviceMemorySpace> mf_solution_;  //!< Scratch for M^{-1} u
#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA
  //@}
};  // class PeripheryT

using Periphery = PeripheryT<Kokkos::DefaultExecutionSpace>;  //!< Default periphery type

// ---------------------------------------------------------------------------------------------------------------
// Motile rigid bodies (surface quadrature) coupled to point spheres + periphery, solved matrix-free with Belos GMRES.
// See the block comment at the top of this file for the formulation.
// Only available when the Belos + Tpetra TPLs are enabled (the outer solve uses mundy::belos_solve).
// ---------------------------------------------------------------------------------------------------------------
#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)

/// \brief Fused Stokeslet + Rotlet in one source->target sweep:
///        target_velocities += sum_s [ G(x_t - x_s) . F_s + R(x_t - x_s) . tau_s ],
///        G = 1/(8 pi mu) (I/r + r r / r^3),   R . tau = 1/(8 pi mu) (tau x r) / r^3.
///
/// The body's net force/torque enter as a single point source at the body center, so this is typically called
/// with one source point. Accumulates into target_velocities (zero it first).
template <class ExecutionSpace, typename SourcePosVectorType, typename TargetPosVectorType,
          typename SourceForceVectorType, typename SourceTorqueVectorType, typename TargetVelocityVectorType>
void apply_stokeslet_rotlet_kernel(const ExecutionSpace& space, const double viscosity,  //
                                   const SourcePosVectorType& source_positions,          //
                                   const TargetPosVectorType& target_positions,          //
                                   const SourceForceVectorType& source_forces,           //
                                   const SourceTorqueVectorType& source_torques,         //
                                   const TargetVelocityVectorType& target_velocities) {
  const size_t num_source_points = source_positions.extent(0) / 3;
  const size_t num_target_points = target_positions.extent(0) / 3;
  const double scale_factor = 1.0 / (8.0 * M_PI * viscosity);
  auto contribution = KOKKOS_LAMBDA(const size_t t, const size_t s) {
    const double dx = target_positions(3 * t + 0) - source_positions(3 * s + 0);
    const double dy = target_positions(3 * t + 1) - source_positions(3 * s + 1);
    const double dz = target_positions(3 * t + 2) - source_positions(3 * s + 2);
    const double r2 = dx * dx + dy * dy + dz * dz;
    const bool coincident = r2 < DOUBLE_ZERO;  // selected, not branched on, so direct_sum vectorizes
    const double rinv = coincident ? 0.0 : 1.0 / Kokkos::sqrt(coincident ? 1.0 : r2);
    const double rinv3 = rinv * rinv * rinv;

    // Stokeslet: (I/r + r r / r^3) . F
    const double fx = source_forces(3 * s + 0);
    const double fy = source_forces(3 * s + 1);
    const double fz = source_forces(3 * s + 2);
    const double f_dot_r = fx * dx + fy * dy + fz * dz;
    const mundy::Vector3d stokeslet{scale_factor * (rinv * fx + rinv3 * dx * f_dot_r),
                                    scale_factor * (rinv * fy + rinv3 * dy * f_dot_r),
                                    scale_factor * (rinv * fz + rinv3 * dz * f_dot_r)};

    // Rotlet: (tau x r) / r^3
    const double tx = source_torques(3 * s + 0);
    const double ty = source_torques(3 * s + 1);
    const double tz = source_torques(3 * s + 2);
    const mundy::Vector3d rotlet{scale_factor * rinv3 * (ty * dz - tz * dy), scale_factor * rinv3 * (tz * dx - tx * dz),
                                 scale_factor * rinv3 * (tx * dy - ty * dx)};
    return stokeslet + rotlet;
  };
  mundy::direct_sum(space, num_target_points, num_source_points, contribution,
                    impl::add_to_vector3_entries(target_velocities));
}

/// \brief Body self-interaction LHS block: y = (-1/(2 viscosity) I + T_b)[q] - (U + Omega x (x - X)).
///
/// T_b (with the -1/(2 viscosity) I jump) is the singularity-subtracted double layer (source == target == this body;
/// see apply_stokes_double_layer_kernel_ss). The rigid-body-motion term is subtracted once per target node -- NOT
/// inside the per-source sweep. Accumulates into y, which the caller must have zeroed.
template <class ExecutionSpace, class PosView, class NrmView, class WgtView, class CenterView, class QView, class UView,
          class OmView, class OutView>
void apply_body_self_lhs(const ExecutionSpace& space, const double viscosity, const size_t num_quadrature_points,
                         const PosView& positions, const NrmView& normals, const WgtView& weights,
                         const CenterView& center, const QView& q, const UView& U, const OmView& Om, const OutView& y) {
  // (-1/(2 viscosity) I + T_b)[q] via singularity subtraction; accumulates into y.
  apply_stokes_double_layer_kernel_ss(space, viscosity, num_quadrature_points, positions, normals, weights, q, y);

  // Subtract the rigid-body motion U + Omega x (x - X) once per target node.
  Kokkos::parallel_for(
      "apply_body_self_lhs_rigid_motion", Kokkos::RangePolicy<ExecutionSpace>(0, num_quadrature_points),
      KOKKOS_LAMBDA(const size_t k) {
        const double rx = positions(3 * k + 0) - center(0);
        const double ry = positions(3 * k + 1) - center(1);
        const double rz = positions(3 * k + 2) - center(2);
        y(3 * k + 0) -= U(0) + (Om(1) * rz - Om(2) * ry);
        y(3 * k + 1) -= U(1) + (Om(2) * rx - Om(0) * rz);
        y(3 * k + 2) -= U(2) + (Om(0) * ry - Om(1) * rx);
      });
}

/// \brief Fused reduction value: the mean force (3) and mean torque (3) accumulated over a body's quadrature.
struct BodyMeanForceTorque {
  double force[3];
  double torque[3];
};

/// \brief Reduction functor accumulating BodyMeanForceTorque in one sweep over a body's quadrature.
///
/// Accumulates the (un-normalized) mean force sum_k w_k q_k and mean torque sum_k w_k (x_k - X) x q_k.
template <class PosView, class WgtView, class QView, class CenterView>
struct BodyMeanForceTorqueFunctor {
  using value_type = BodyMeanForceTorque;

  PosView positions;
  WgtView weights;
  QView q;
  CenterView center;

  KOKKOS_INLINE_FUNCTION void operator()(const size_t k, value_type& acc) const {
    const double w = weights(k);
    const double qx = q(3 * k + 0);
    const double qy = q(3 * k + 1);
    const double qz = q(3 * k + 2);
    acc.force[0] += w * qx;
    acc.force[1] += w * qy;
    acc.force[2] += w * qz;

    const double rx = positions(3 * k + 0) - center(0);
    const double ry = positions(3 * k + 1) - center(1);
    const double rz = positions(3 * k + 2) - center(2);
    acc.torque[0] += w * (ry * qz - rz * qy);
    acc.torque[1] += w * (rz * qx - rx * qz);
    acc.torque[2] += w * (rx * qy - ry * qx);
  }

  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
    for (int i = 0; i < 3; ++i) {
      dst.force[i] += src.force[i];
      dst.torque[i] += src.torque[i];
    }
  }

  KOKKOS_INLINE_FUNCTION void init(value_type& v) const {
    for (int i = 0; i < 3; ++i) {
      v.force[i] = 0.0;
      v.torque[i] = 0.0;
    }
  }
};

/// \brief The mean force and mean torque of a body's surface density, in one reduction sweep.
///
///   mean_force = (1/|Gamma|) integral q dS,   mean_torque = (1/|Gamma|) integral (y - X) x q dS.
/// These recover the body's rigid translational and angular velocity from the surface density.
template <class ExecutionSpace, class PosView, class WgtView, class QView, class CenterView>
void body_mean_force_and_torque([[maybe_unused]] const ExecutionSpace& space, const size_t num_quadrature_points,
                                const PosView& positions, const WgtView& weights, const QView& q,
                                const CenterView& center, const double area, mundy::Vector3d& mean_force,
                                mundy::Vector3d& mean_torque) {
  BodyMeanForceTorqueFunctor<PosView, WgtView, QView, CenterView> functor{positions, weights, q, center};
  BodyMeanForceTorque acc;
  Kokkos::parallel_reduce("body_mean_force_and_torque", Kokkos::RangePolicy<ExecutionSpace>(0, num_quadrature_points),
                          functor, acc);
  mean_force = mundy::Vector3d(acc.force[0] / area, acc.force[1] / area, acc.force[2] / area);
  mean_torque = mundy::Vector3d(acc.torque[0] / area, acc.torque[1] / area, acc.torque[2] / area);
}

/// \brief One motile rigid body with its own surface quadrature.
///
/// The reference-frame geometry is placed into the lab frame by the center and orientation (both fixed during a
/// solve; the velocities U and Omega are the unknowns). The pose and load views are subviews of the owning BodySet's
/// flat arrays, wired by BodySet::wire_pose_load.
template <class ExecSpace>
struct MotileBody {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  size_t num_quadrature_points = 0;            //!< number of surface quadrature points on this body
  double area = 0.0;                           //!< |Gamma| = sum of weights (rigid-invariant)
  view_t positions_ref, normals_ref, weights;  //!< reference-frame quadrature (3N, 3N outward, N)
  view_t positions, normals;                   //!< lab-frame quadrature (materialized by place_body_in_lab_frame)
  view_t center;                               //!< X (translation), length 3; subview of BodySet::centers
  view_t orientation;  //!< reference -> lab rotation (w, x, y, z), length 4; subview of BodySet::orientations
  view_t net_force;    //!< F (prescribed), length 3; subview of BodySet::net_forces
  view_t net_torque;   //!< tau (prescribed), length 3; subview of BodySet::net_torques
};

/// \brief Materialize the lab-frame quadrature from the reference frame: x = X + q * x_ref, n = q * n_ref.
template <class ExecSpace>
void place_body_in_lab_frame(const ExecSpace&, MotileBody<ExecSpace>& b) {
  auto center = b.center;
  auto orientation = b.orientation;
  auto positions_ref = b.positions_ref;
  auto normals_ref = b.normals_ref;
  auto positions = b.positions;
  auto normals = b.normals;
  Kokkos::parallel_for(
      "place_body_in_lab_frame", Kokkos::RangePolicy<ExecSpace>(0, b.num_quadrature_points),
      KOKKOS_LAMBDA(const size_t k) {
        const mundy::Quaternion<double> q(orientation(0), orientation(1), orientation(2), orientation(3));
        const mundy::Vector3d p_ref(positions_ref(3 * k + 0), positions_ref(3 * k + 1), positions_ref(3 * k + 2));
        const mundy::Vector3d n_ref(normals_ref(3 * k + 0), normals_ref(3 * k + 1), normals_ref(3 * k + 2));
        const auto p_lab = q * p_ref;  // a quaternion rotates a vector via multiplication
        const auto n_lab = q * n_ref;
        positions(3 * k + 0) = center(0) + p_lab[0];
        positions(3 * k + 1) = center(1) + p_lab[1];
        positions(3 * k + 2) = center(2) + p_lab[2];
        normals(3 * k + 0) = n_lab[0];
        normals(3 * k + 1) = n_lab[1];
        normals(3 * k + 2) = n_lab[2];
      });
}

/// \brief A collection of motile bodies, providing the flat-vector layout used by the block solve.
///
/// The unknown vector is x = [ q^0 | q^1 | ... | U^0 Omega^0 | U^1 Omega^1 | ... ]: all densities concatenated,
/// then each body's rigid velocities. Indices returned by density_offset/rigid_velocity_offset are flat (already
/// multiplied by 3).
template <class ExecSpace>
struct BodySet {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  std::vector<MotileBody<ExecSpace>> bodies;
  view_t centers, orientations, net_forces, net_torques;  //!< flat pose/load arrays (3 / 4 / 3 / 3 per body)

  //! \brief Allocate the flat pose/load arrays for the current bodies and point each body's subviews into them.
  //!
  //! Call after bodies (with their geometry) are populated. Fill the pose/load through set_center/set_orientation/
  //! set_net_force/set_net_torque, or by touching the flat views directly.
  void wire_pose_load() {
    centers = view_t("body_centers", 3 * num_bodies());
    orientations = view_t("body_orientations", 4 * num_bodies());
    net_forces = view_t("body_net_forces", 3 * num_bodies());
    net_torques = view_t("body_net_torques", 3 * num_bodies());
    for (size_t m = 0; m < num_bodies(); ++m) {
      bodies[m].center = Kokkos::subview(centers, Kokkos::pair<size_t, size_t>(3 * m, 3 * m + 3));
      bodies[m].orientation = Kokkos::subview(orientations, Kokkos::pair<size_t, size_t>(4 * m, 4 * m + 4));
      bodies[m].net_force = Kokkos::subview(net_forces, Kokkos::pair<size_t, size_t>(3 * m, 3 * m + 3));
      bodies[m].net_torque = Kokkos::subview(net_torques, Kokkos::pair<size_t, size_t>(3 * m, 3 * m + 3));
    }
  }

  void set_center(size_t m, const mundy::Vector3d& value) {
    set_vector3(centers, m, value, "BodySet::set_center");
  }
  void set_net_force(size_t m, const mundy::Vector3d& value) {
    set_vector3(net_forces, m, value, "BodySet::set_net_force");
  }
  void set_net_torque(size_t m, const mundy::Vector3d& value) {
    set_vector3(net_torques, m, value, "BodySet::set_net_torque");
  }
  //! \brief Set body m's orientation, stored as (w, x, y, z) -- the order place_body_in_lab_frame reads.
  void set_orientation(size_t m, const mundy::Quaternion<double>& value) {
    require_pose_slot(orientations, 4, m, "BodySet::set_orientation");
    auto sub = Kokkos::subview(orientations, Kokkos::pair<size_t, size_t>(4 * m, 4 * m + 4));
    auto host = Kokkos::create_mirror_view(sub);
    host(0) = value.w();  // operator[] uses the storage order (x, y, z, w), so use the named accessors
    host(1) = value.x();
    host(2) = value.y();
    host(3) = value.z();
    Kokkos::deep_copy(sub, host);
  }

  //! \brief Throw unless every body is wired (see wire_pose_load) and has a consistent quadrature.
  void require_wired(const char* context) const {
    for (size_t m = 0; m < num_bodies(); ++m) {
      const auto& b = bodies[m];
      const size_t n = b.num_quadrature_points;
      MUNDY_THROW_REQUIRE(
          (b.center.extent(0) == 3) && (b.orientation.extent(0) == 4) && (b.net_force.extent(0) == 3) &&
              (b.net_torque.extent(0) == 3),
          std::invalid_argument,
          mundy::sink() << context << ": body " << m << " has no pose/load views; call BodySet::wire_pose_load first.");
      MUNDY_THROW_REQUIRE(
          (b.positions_ref.extent(0) == 3 * n) && (b.normals_ref.extent(0) == 3 * n) && (b.weights.extent(0) == n) &&
              (b.positions.extent(0) == 3 * n) && (b.normals.extent(0) == 3 * n) && (b.area > 0.0),
          std::invalid_argument,
          mundy::sink() << context << ": body " << m << " has an inconsistent quadrature (views must hold " << n
                        << " points and the area must be positive).");
    }
  }

  size_t num_bodies() const {
    return bodies.size();
  }
  size_t num_quadrature_points() const {
    size_t total = 0;
    for (const auto& b : bodies) total += b.num_quadrature_points;
    return total;
  }
  size_t num_dofs() const {
    return 3 * num_quadrature_points() + 6 * num_bodies();
  }
  size_t density_offset(size_t i) const {  //!< flat start index of body i's density block
    size_t points = 0;
    for (size_t j = 0; j < i; ++j) points += bodies[j].num_quadrature_points;
    return 3 * points;
  }
  size_t rigid_velocity_offset(size_t i) const {  //!< flat start index of body i's [U, Omega] block
    return 3 * num_quadrature_points() + 6 * i;
  }

 private:
  void require_pose_slot(const view_t& flat, const size_t width, const size_t m, const char* context) const {
    MUNDY_THROW_REQUIRE((m < num_bodies()) && (flat.extent(0) == width * num_bodies()), std::invalid_argument,
                        mundy::sink() << context << ": body " << m << " is out of range or the pose/load views are "
                                      << "not wired (call wire_pose_load after populating bodies).");
  }

  void set_vector3(const view_t& flat, size_t m, const mundy::Vector3d& value, const char* context) {
    require_pose_slot(flat, 3, m, context);
    auto sub = Kokkos::subview(flat, Kokkos::pair<size_t, size_t>(3 * m, 3 * m + 3));
    auto host = Kokkos::create_mirror_view(sub);
    host(0) = value[0];
    host(1) = value[1];
    host(2) = value[2];
    Kokkos::deep_copy(sub, host);
  }
};

/// \brief A motile spherical body of the given radius with an order-`order` surface quadrature (outward normals).
///
/// The quadrature is the (order + 1)-ring Gauss-Legendre sphere rule scaled to the radius. Geometry only: the pose and
/// load live on the owning BodySet; see make_sphere_body_set.
template <class ExecSpace>
MotileBody<ExecSpace> make_sphere_body(const int order, const double radius) {
  MUNDY_THROW_REQUIRE(order >= 0, std::invalid_argument, "make_sphere_body: order must be non-negative.");
  MUNDY_THROW_REQUIRE(radius > 0.0, std::invalid_argument, "make_sphere_body: radius must be positive.");
  using view_t = typename MotileBody<ExecSpace>::view_t;
  std::vector<double> normals;  // the unit-sphere points, which are the outward unit normals
  std::vector<double> weights;
  mundy::gauss_legendre_sphere_rule(order + 1, normals, weights);
  std::vector<double> points(normals.size());
  for (size_t i = 0; i < normals.size(); ++i) {
    points[i] = radius * normals[i];
  }
  for (double& weight : weights) {
    weight *= radius * radius;
  }
  const size_t num_quadrature_points = weights.size();

  MotileBody<ExecSpace> body;
  body.num_quadrature_points = num_quadrature_points;
  body.area = std::accumulate(weights.begin(), weights.end(), 0.0);  // |Gamma| = sum of quadrature weights
  body.positions_ref = to_device<ExecSpace>(points);
  body.normals_ref = to_device<ExecSpace>(normals);
  body.weights = to_device<ExecSpace>(weights);
  body.positions = view_t("positions", 3 * num_quadrature_points);
  body.normals = view_t("normals", 3 * num_quadrature_points);
  return body;
}

/// \brief A one-body BodySet: a sphere at `center`/`orientation` with the given net force/torque, placed in the lab.
template <class ExecSpace>
BodySet<ExecSpace> make_sphere_body_set(const int order, const double radius, const mundy::Vector3d& center,
                                        const mundy::Quaternion<double>& orientation,
                                        const mundy::Vector3d& net_force = mundy::Vector3d(0.0, 0.0, 0.0),
                                        const mundy::Vector3d& net_torque = mundy::Vector3d(0.0, 0.0, 0.0)) {
  BodySet<ExecSpace> body_set;
  body_set.bodies.push_back(make_sphere_body<ExecSpace>(order, radius));
  body_set.wire_pose_load();
  body_set.set_center(0, center);
  body_set.set_orientation(0, orientation);
  body_set.set_net_force(0, net_force);
  body_set.set_net_torque(0, net_torque);
  place_body_in_lab_frame(ExecSpace{}, body_set.bodies[0]);
  return body_set;
}

/// \brief How a set of point spheres interacts hydrodynamically.
///
/// Dry: self-drag only (no mutual flow, no coupling to the periphery or resolved bodies). Stokes/RPY/RPYC: self-drag
/// plus the named mutual kernel and coupling to the periphery and any resolved bodies.
enum class SphereInteraction { Dry, Stokes, RPY, RPYC };

/// \brief Dispatch a sphere source->target flow kernel by interaction type.
///
/// Dry is a no-op; Stokes ignores the radii (Stokeslet point sources). Accumulates into target_velocities, like the
/// underlying kernels.
template <class ExecSpace, typename SourcePosVectorType, typename TargetPosVectorType, typename SourceRadiusVectorType,
          typename TargetRadiusVectorType, typename SourceForceVectorType, typename TargetVelocityVectorType>
void apply_sphere_kernel(const SphereInteraction interaction, const ExecSpace& space, const double viscosity,
                         const SourcePosVectorType& source_positions, const TargetPosVectorType& target_positions,
                         const SourceRadiusVectorType& source_radii, const TargetRadiusVectorType& target_radii,
                         const SourceForceVectorType& source_forces,
                         const TargetVelocityVectorType& target_velocities) {
  switch (interaction) {
    case SphereInteraction::RPY:
      apply_rpy_kernel(space, viscosity, source_positions, target_positions, source_radii, target_radii, source_forces,
                       target_velocities);
      break;
    case SphereInteraction::RPYC:
      apply_rpyc_kernel(space, viscosity, source_positions, target_positions, source_radii, target_radii, source_forces,
                        target_velocities);
      break;
    case SphereInteraction::Stokes:
      apply_stokes_kernel(space, viscosity, source_positions, target_positions, source_forces, target_velocities);
      break;
    case SphereInteraction::Dry:
      break;  // no long-range flow
  }
}

/// \brief Point spheres with prescribed forces that generate an ambient flow through their interaction kernel.
template <class ExecSpace>
struct SphereSet {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  view_t positions, radii, forces;                         //!< lengths 3N / N / 3N for N = num_spheres()
  SphereInteraction interaction = SphereInteraction::Dry;  //!< set by the constructor; irrelevant with no spheres

  //! \brief No spheres.
  SphereSet() = default;

  //! \brief The spheres at \p positions_in with \p radii_in and prescribed \p forces_in, coupled via \p interaction_in.
  SphereSet(const view_t& positions_in, const view_t& radii_in, const view_t& forces_in,
            const SphereInteraction interaction_in)
      : positions(positions_in), radii(radii_in), forces(forces_in), interaction(interaction_in) {
    MUNDY_THROW_REQUIRE((positions.extent(0) == 3 * num_spheres()) && (forces.extent(0) == 3 * num_spheres()),
                        std::invalid_argument,
                        mundy::sink() << "SphereSet: positions and forces must have length 3 * " << num_spheres()
                                      << " but have lengths " << positions.extent(0) << " and " << forces.extent(0));
  }

  size_t num_spheres() const {
    return radii.extent(0);
  }
};

/// \brief The block LHS operator A applied to x = [ q | U | Omega ] (matrix-free GMRES).
///
/// Domain = range = BodySet::num_dofs(). Holds the periphery geometry + dense M^{-1} (P^{-1} = -M^{-1}); the
/// periphery closure is skipped when there is no periphery (num_periphery_points == 0).
template <class ExecSpace>
struct BodyMobilityOp {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  using mat_t = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  const BodySet<ExecSpace>* body_set = nullptr;
  double viscosity = 1.0;
  size_t num_periphery_points = 0;  //!< periphery quadrature points (0 => no periphery closure)
  view_t p_positions, p_normals, p_weights;
  mat_t M_inv;                 //!< dense periphery M^{-1} (P^{-1} = -M^{-1})
  mutable view_t slip_p, q_p;  //!< scratch, size 3*num_periphery_points

  size_t domain_size() const {
    return body_set->num_dofs();
  }
  size_t range_size() const {
    return body_set->num_dofs();
  }
  view_t make_domain_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "body_mobility_x"), body_set->num_dofs());
  }
  view_t make_range_vector() const {
    return view_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "body_mobility_y"), body_set->num_dofs());
  }

  template <class XVector, class YVector>
  void apply(const XVector& x, YVector& y) const {
    const BodySet<ExecSpace>& B = *body_set;
    const size_t num_bodies = B.num_bodies();
    Kokkos::deep_copy(y, 0.0);

    auto density = [&](const auto& v, size_t i) {
      const size_t start = B.density_offset(i);
      return Kokkos::subview(v, Kokkos::pair<size_t, size_t>(start, start + 3 * B.bodies[i].num_quadrature_points));
    };
    auto trans = [&](const auto& v, size_t i) {
      const size_t start = B.rigid_velocity_offset(i);
      return Kokkos::subview(v, Kokkos::pair<size_t, size_t>(start, start + 3));
    };
    auto rot = [&](const auto& v, size_t i) {
      const size_t start = B.rigid_velocity_offset(i) + 3;
      return Kokkos::subview(v, Kokkos::pair<size_t, size_t>(start, start + 3));
    };

    // 2.1  singularity-subtracted double-layer, body self-interaction (fused with the motile term).
    for (size_t i = 0; i < num_bodies; ++i) {
      const auto& body = B.bodies[i];
      apply_body_self_lhs(ExecSpace{}, viscosity, body.num_quadrature_points, body.positions, body.normals,
                          body.weights, body.center, density(x, i), trans(x, i), rot(x, i), density(y, i));
    }

    // 2.2  Each body's double layer evaluated at every quadrature point that is NOT its own surface: the other
    //      bodies' surfaces (the inter-body coupling) and, when present, the periphery (accumulating the unknown
    //      periphery slip sum_m T_b^m[q_m]). Source and target are distinct sets here, so this is the plain kernel.
    if (num_periphery_points > 0) {
      Kokkos::deep_copy(slip_p, 0.0);
    }
    for (size_t m = 0; m < num_bodies; ++m) {
      const auto& src = B.bodies[m];
      auto q_m = density(x, m);
      for (size_t i = 0; i < num_bodies; ++i) {  // -> other bodies (empty for a single body)
        if (i == m) {
          continue;
        }
        const auto& tgt = B.bodies[i];
        apply_stokes_double_layer_kernel(ExecSpace{}, viscosity, src.num_quadrature_points, tgt.num_quadrature_points,
                                         src.positions, tgt.positions, src.normals, src.weights, q_m, density(y, i));
      }
      if (num_periphery_points > 0) {  // -> periphery
        apply_stokes_double_layer_kernel(ExecSpace{}, viscosity, src.num_quadrature_points, num_periphery_points,
                                         src.positions, p_positions, src.normals, src.weights, q_m, slip_p);
      }
    }

    // 2.3  Unknown periphery feedback: q_p = P^{-1} slip_p = -M^{-1} slip_p ;  y_q_i += T_p[q_p] on each body.
    if (num_periphery_points > 0) {
      KokkosBlas::gemv("N", -1.0, M_inv, slip_p, 0.0, q_p);
      for (size_t i = 0; i < num_bodies; ++i) {
        const auto& body = B.bodies[i];
        apply_stokes_double_layer_kernel(ExecSpace{}, viscosity, num_periphery_points, body.num_quadrature_points,
                                         p_positions, body.positions, p_normals, p_weights, q_p, density(y, i));
      }
    }

    // constraint rows:  (1/|Gamma|) integral q - U ,  (1/|Gamma|) integral (y-X) x q - Omega.
    for (size_t i = 0; i < num_bodies; ++i) {
      const auto& body = B.bodies[i];
      mundy::Vector3d mean_force;
      mundy::Vector3d mean_torque;
      body_mean_force_and_torque(ExecSpace{}, body.num_quadrature_points, body.positions, body.weights, density(x, i),
                                 body.center, body.area, mean_force, mean_torque);
      const Kokkos::Array<double, 3> mf = {mean_force[0], mean_force[1], mean_force[2]};
      const Kokkos::Array<double, 3> mt = {mean_torque[0], mean_torque[1], mean_torque[2]};
      auto U_i = trans(x, i);
      auto Om_i = rot(x, i);
      auto yU_i = trans(y, i);
      auto yO_i = rot(y, i);
      Kokkos::parallel_for(
          "body_constraint_rows", Kokkos::RangePolicy<ExecSpace>(0, 3), KOKKOS_LAMBDA(const int k) {
            yU_i(k) = mf[k] - U_i(k);
            yO_i(k) = mt[k] - Om_i(k);
          });
    }
  }
};

/// \brief Assemble the fixed RHS b for the block mobility solve (formulation step 1).
template <class ExecSpace>
void assemble_rhs(const ExecSpace& space, const BodySet<ExecSpace>& B, const SphereSet<ExecSpace>& spheres,
                  const double viscosity, const size_t num_periphery_points,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_positions,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_normals,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_weights,
                  const Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& M_inv,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& b) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  const size_t num_bodies = B.num_bodies();
  Kokkos::deep_copy(b, 0.0);  // constraint (rigid) blocks stay 0

  view_t slip_p, q_p, p_zero_radii;
  if (num_periphery_points > 0) {
    slip_p = view_t("slip_p", 3 * num_periphery_points);
    q_p = view_t("q_p", 3 * num_periphery_points);
    p_zero_radii = view_t("p_zero_radii", num_periphery_points);  // periphery points are field points (radius 0)
    Kokkos::deep_copy(slip_p, 0.0);
    Kokkos::deep_copy(p_zero_radii, 0.0);
  }

  auto b_surf = [&](size_t i) {
    const size_t start = B.density_offset(i);
    return Kokkos::subview(b, Kokkos::pair<size_t, size_t>(start, start + 3 * B.bodies[i].num_quadrature_points));
  };

  // 1.1  Sphere ambient flow to all quad points (each body + periphery), via the spheres' interaction kernel.
  //      Dry spheres induce no flow, so they contribute nothing here.
  if (spheres.num_spheres() > 0 && spheres.interaction != SphereInteraction::Dry) {
    for (size_t i = 0; i < num_bodies; ++i) {
      const auto& body = B.bodies[i];
      view_t body_zero_radii("body_zero_radii", body.num_quadrature_points);
      Kokkos::deep_copy(body_zero_radii, 0.0);
      auto bs = b_surf(i);
      apply_sphere_kernel(spheres.interaction, space, viscosity, spheres.positions, body.positions, spheres.radii,
                          body_zero_radii, spheres.forces, bs);
    }
    if (num_periphery_points > 0) {
      apply_sphere_kernel(spheres.interaction, space, viscosity, spheres.positions, p_positions, spheres.radii,
                          p_zero_radii, spheres.forces, slip_p);
    }
  }

  // 1.2  G/R from each body's net force/torque to all quad points (fused Stokeslet + Rotlet).
  for (size_t m = 0; m < num_bodies; ++m) {
    const auto& src = B.bodies[m];
    for (size_t i = 0; i < num_bodies; ++i) {
      auto bs = b_surf(i);
      apply_stokeslet_rotlet_kernel(space, viscosity, src.center, B.bodies[i].positions, src.net_force, src.net_torque,
                                    bs);
    }
    if (num_periphery_points > 0) {
      apply_stokeslet_rotlet_kernel(space, viscosity, src.center, p_positions, src.net_force, src.net_torque, slip_p);
    }
  }

  // 1.3  known periphery feedback (periphery -> bodies):  q_p = -M^{-1} slip_p ;  b_surf += T_p[q_p].
  if (num_periphery_points > 0) {
    KokkosBlas::gemv("N", -1.0, M_inv, slip_p, 0.0, q_p);
    for (size_t i = 0; i < num_bodies; ++i) {
      const auto& body = B.bodies[i];
      auto bs = b_surf(i);
      apply_stokes_double_layer_kernel(space, viscosity, num_periphery_points, body.num_quadrature_points, p_positions,
                                       body.positions, p_normals, p_weights, q_p, bs);
    }
  }

  // b_surf = -b_surf  (all known terms enter the RHS negated); rigid blocks are already 0.
  const size_t n_surf = 3 * B.num_quadrature_points();
  Kokkos::parallel_for(
      "negate_rhs_surface", Kokkos::RangePolicy<ExecSpace>(0, n_surf), KOKKOS_LAMBDA(const size_t k) { b(k) = -b(k); });
}

/// \brief Solve the block mobility system for x = [ q | U | Omega ] via matrix-free GMRES (steps 1-2).
///
/// The solution is written into \p x (allocated by the caller, length B.num_dofs()); returns the GMRES result.
template <class ExecSpace>
mundy::BelosResult<double> solve_mobility(
    const ExecSpace& space, const BodySet<ExecSpace>& B, const SphereSet<ExecSpace>& spheres, const double viscosity,
    const size_t num_periphery_points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_positions,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_weights,
    const Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& M_inv,
    const mundy::BelosConfig<double>& gmres_cfg,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& x) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  B.require_wired("solve_mobility");

  view_t b("body_mobility_rhs", B.num_dofs());
  assemble_rhs(space, B, spheres, viscosity, num_periphery_points, p_positions, p_normals, p_weights, M_inv, b);

  BodyMobilityOp<ExecSpace> op;
  op.body_set = &B;
  op.viscosity = viscosity;
  op.num_periphery_points = num_periphery_points;
  op.p_positions = p_positions;
  op.p_normals = p_normals;
  op.p_weights = p_weights;
  op.M_inv = M_inv;
  if (num_periphery_points > 0) {
    op.slip_p = view_t("op_slip_p", 3 * num_periphery_points);
    op.q_p = view_t("op_q_p", 3 * num_periphery_points);
  }

  Kokkos::deep_copy(x, 0.0);  // cold start
  auto problem = mundy::LinearSystem(mundy::KokkosBackend<ExecSpace>{}, op, b);
  return mundy::belos_solve(problem, x, gmres_cfg);
}

/// \brief Accumulate onto u_spheres the flow the periphery and bodies induce at the spheres (steps 3.1 + 3.2).
///
/// 3.1 -- the periphery's response to the full (known + unknown) slip, when a periphery is present. 3.2 -- each
/// body's Stokeslet/Rotlet (its known net force/torque) plus its double layer (its unknown solved density).
/// u_spheres is ADDED to, not overwritten: the caller pre-fills it with the spheres' own velocity (e.g. their mutual
/// interaction and self-drag). \p x is the solved [ q | U | Omega ]. Dry spheres receive no long-range flow.
template <class ExecSpace>
void evaluate_sphere_flow(
    const ExecSpace& space, const BodySet<ExecSpace>& B, const SphereSet<ExecSpace>& spheres, const double viscosity,
    const size_t num_periphery_points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_positions,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& p_weights,
    const Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& M_inv,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& x,
    const Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>& u_spheres) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  if (spheres.num_spheres() == 0 || spheres.interaction == SphereInteraction::Dry) {
    return;  // no spheres, or dry spheres that neither induce nor receive long-range flow
  }
  B.require_wired("evaluate_sphere_flow");
  const size_t num_bodies = B.num_bodies();

  // u_spheres is ACCUMULATED onto (never zeroed here): the caller pre-fills it with the spheres' own velocity.

  auto density = [&](size_t i) {
    const size_t start = B.density_offset(i);
    return Kokkos::subview(x, Kokkos::pair<size_t, size_t>(start, start + 3 * B.bodies[i].num_quadrature_points));
  };

  // 3.1  Periphery-to-sphere flow (known + unknown). Reassemble the full periphery slip -- known ambient (the
  //      spheres' kernel) + known body G/R + the now-known body double-layer slip sum_m T_b^m[q_m] -- invert it
  //      ONCE (P^{-1} = -M^{-1}), then evaluate T_p at the spheres.
  if (num_periphery_points > 0) {
    view_t slip_tot("slip_tot", 3 * num_periphery_points);
    view_t q_p_tot("q_p_tot", 3 * num_periphery_points);
    view_t p_zero_radii("p_zero_radii", num_periphery_points);
    Kokkos::deep_copy(slip_tot, 0.0);
    Kokkos::deep_copy(p_zero_radii, 0.0);

    apply_sphere_kernel(spheres.interaction, space, viscosity, spheres.positions, p_positions, spheres.radii,
                        p_zero_radii, spheres.forces, slip_tot);
    for (size_t m = 0; m < num_bodies; ++m) {
      const auto& src = B.bodies[m];
      apply_stokeslet_rotlet_kernel(space, viscosity, src.center, p_positions, src.net_force, src.net_torque, slip_tot);
    }
    for (size_t i = 0; i < num_bodies; ++i) {
      const auto& body = B.bodies[i];
      apply_stokes_double_layer_kernel(space, viscosity, body.num_quadrature_points, num_periphery_points,
                                       body.positions, p_positions, body.normals, body.weights, density(i), slip_tot);
    }
    KokkosBlas::gemv("N", -1.0, M_inv, slip_tot, 0.0, q_p_tot);
    apply_stokes_double_layer_kernel(space, viscosity, num_periphery_points, spheres.num_spheres(), p_positions,
                                     spheres.positions, p_normals, p_weights, q_p_tot, u_spheres);
  }

  // 3.2  Body-to-sphere flow (known + unknown): each body's Stokeslet/Rotlet from its (known) net force/torque
  //      plus its double layer with the (unknown) solved density, evaluated directly at the spheres.
  for (size_t m = 0; m < num_bodies; ++m) {
    const auto& body = B.bodies[m];
    apply_stokeslet_rotlet_kernel(space, viscosity, body.center, spheres.positions, body.net_force, body.net_torque,
                                  u_spheres);
    apply_stokes_double_layer_kernel(space, viscosity, body.num_quadrature_points, spheres.num_spheres(),
                                     body.positions, spheres.positions, body.normals, body.weights, density(m),
                                     u_spheres);
  }
}

/// \brief Coupling object for point spheres, resolved rigid bodies, and a periphery under one solve.
///
/// Each participant is optional; each set_* declares the views it reads and the view(s) it writes, and solve()
/// reads the inputs and writes the outputs:
///   - no periphery  -> free space (the default; pass an empty periphery shared_ptr);
///   - no spheres    -> no sphere-velocity output is written;
///   - no bodies     -> no block solve is run (the sphere velocities are a direct evaluation).
///
/// Spheres are force-only point sources whose SphereInteraction (Dry / Stokes / RPY / RPYC) selects both their
/// mutual kernel and whether they couple to the periphery and bodies; solve() overwrites the registered sphere
/// velocity view with the full hydrodynamic velocity (self-drag + mutual + periphery + body coupling), and the
/// caller composes any other contribution (e.g. Brownian). Resolved bodies carry orientation + net force/torque
/// and are solved for their rigid velocity (written to the registered velocity / angular-velocity views) via
/// matrix-free GMRES; solve() first places each body in the lab frame from its current pose.
///
/// The periphery is held by shared_ptr (shared ownership) and must hold a dense M^{-1}
/// (built via build_inverse_self_interaction_matrix or set_inverse_self_interaction_matrix).
template <class ExecSpace = Kokkos::DefaultExecutionSpace>
class MobilitySystem {
 public:
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  using mat_t = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  explicit MobilitySystem(const double viscosity) : viscosity_(viscosity) {
    impl::require_positive_viscosity(viscosity_, "MobilitySystem");
  }

  //! \brief Set the (optional) periphery; the system shares ownership.
  //!
  //! An empty shared_ptr leaves the system in free space.
  MobilitySystem& set_periphery(std::shared_ptr<const PeripheryT<ExecSpace>> periphery) {
    periphery_ = std::move(periphery);
    return *this;
  }

  //! \brief Register the point spheres.
  //!
  //! Force-only sources: reads positions/forces (3N) and radii (N); solve() writes the full hydrodynamic
  //! velocity into \p velocities (3N, overwritten).
  MobilitySystem& set_spheres(const view_t& positions, const view_t& radii, const view_t& forces,
                              const view_t& velocities, const SphereInteraction interaction) {
    spheres_ = SphereSet<ExecSpace>(positions, radii, forces, interaction);
    MUNDY_THROW_REQUIRE(velocities.extent(0) == 3 * spheres_.num_spheres(), std::invalid_argument,
                        mundy::sink() << "MobilitySystem::set_spheres: velocities must have length 3 * "
                                      << spheres_.num_spheres() << " but has length " << velocities.extent(0));
    sphere_velocities_ = velocities;
    return *this;
  }

  //! \brief Register the resolved rigid bodies.
  //!
  //! Their net_force/net_torque are the prescribed loads. solve() writes each body's rigid velocity into
  //! \p velocities (3 per body) and angular velocity into \p angular_velocities (3 per body), indexed by body.
  MobilitySystem& set_bodies(const BodySet<ExecSpace>& bodies, const view_t& velocities,
                             const view_t& angular_velocities) {
    bodies.require_wired("MobilitySystem::set_bodies");
    MUNDY_THROW_REQUIRE(
        (velocities.extent(0) == 3 * bodies.num_bodies()) && (angular_velocities.extent(0) == 3 * bodies.num_bodies()),
        std::invalid_argument,
        mundy::sink() << "MobilitySystem::set_bodies: velocities and angular_velocities must have "
                      << "length 3 * " << bodies.num_bodies() << " but have lengths " << velocities.extent(0) << " and "
                      << angular_velocities.extent(0));
    bodies_ = bodies;
    body_velocities_ = velocities;
    body_angular_velocities_ = angular_velocities;
    return *this;
  }

  //! \brief GMRES configuration for the block solve (only used when resolved bodies are present).
  MobilitySystem& set_belos_config(const mundy::BelosConfig<double>& cfg) {
    belos_config_ = cfg;
    return *this;
  }

  //! \brief Read the input views and write the output views.
  //!
  //! With resolved bodies this places each body in the lab frame, runs the matrix-free GMRES block solve for
  //! x = [ q | U | Omega ], and scatters each body's rigid velocity into the registered velocity /
  //! angular-velocity views. With spheres it overwrites the registered sphere-velocity view with the full
  //! hydrodynamic velocity: self-drag for every sphere, plus (for wet spheres) the mutual kernel and the periphery +
  //! resolved-body coupling. Empty participants are skipped.
  MobilitySystem& solve() {
    size_t num_periphery_points = 0;
    view_t p_positions, p_normals, p_weights;
    mat_t M_inv;
    if (periphery_) {
      num_periphery_points = periphery_->get_num_nodes();
      p_positions = periphery_->get_surface_positions();
      p_normals = periphery_->get_surface_normals();
      p_weights = periphery_->get_quadrature_weights();
      M_inv = periphery_->get_M_inv();
    }

    // Resolved bodies: solve the block system, then scatter x's rigid blocks into the registered output views.
    has_body_solve_ = false;
    if (bodies_.num_bodies() > 0) {
      for (auto& body : bodies_.bodies) {
        place_body_in_lab_frame(ExecSpace{}, body);
      }
      x_ = view_t("mobility_system_x", bodies_.num_dofs());
      last_result_ = solve_mobility(ExecSpace{}, bodies_, spheres_, viscosity_, num_periphery_points, p_positions,
                                    p_normals, p_weights, M_inv, belos_config_, x_);
      has_body_solve_ = true;
      const size_t rigid_base = bodies_.rigid_velocity_offset(0);  // body i's [U, Omega] block is at base + 6 i
      const view_t x = x_;
      const view_t bv = body_velocities_;
      const view_t ba = body_angular_velocities_;
      Kokkos::parallel_for(
          "mobility_system_scatter_body_vel", Kokkos::RangePolicy<ExecSpace>(0, bodies_.num_bodies()),
          KOKKOS_LAMBDA(const size_t i) {
            for (int c = 0; c < 3; ++c) {
              bv(3 * i + c) = x(rigid_base + 6 * i + c);
              ba(3 * i + c) = x(rigid_base + 6 * i + 3 + c);
            }
          });
    }

    // Spheres: overwrite the registered velocity view with the full hydrodynamic velocity.
    if (spheres_.num_spheres() > 0) {
      // Self-drag u_i = F_i / (6 pi mu a_i) for every sphere (dry and wet).
      Kokkos::deep_copy(sphere_velocities_, 0.0);
      apply_local_drag(ExecSpace{}, viscosity_, sphere_velocities_, spheres_.forces, spheres_.radii);
      if (spheres_.interaction != SphereInteraction::Dry) {
        // Mutual sphere-sphere flow (kernels exclude the i == i self term) plus periphery + body coupling; both
        // accumulate onto the self-drag-seeded output view.
        apply_sphere_kernel(spheres_.interaction, ExecSpace{}, viscosity_, spheres_.positions, spheres_.positions,
                            spheres_.radii, spheres_.radii, spheres_.forces, sphere_velocities_);
        evaluate_sphere_flow(ExecSpace{}, bodies_, spheres_, viscosity_, num_periphery_points, p_positions, p_normals,
                             p_weights, M_inv, x_, sphere_velocities_);
      }
    }
    return *this;
  }

  size_t num_spheres() const {
    return spheres_.num_spheres();
  }
  size_t num_bodies() const {
    return bodies_.num_bodies();
  }
  //! \brief The GMRES result of the most recent block solve; throws unless solve() ran with resolved bodies.
  const mundy::BelosResult<double>& body_solve_result() const {
    MUNDY_THROW_REQUIRE(has_body_solve_, std::runtime_error,
                        "MobilitySystem::body_solve_result: no body solve has run (set_bodies + solve first).");
    return last_result_;
  }

 private:
  double viscosity_ = 1.0;
  std::shared_ptr<const PeripheryT<ExecSpace>> periphery_;  //!< shared ownership; empty => free space
  SphereSet<ExecSpace> spheres_;
  BodySet<ExecSpace> bodies_;
  view_t sphere_velocities_;        //!< registered sphere velocity output (3N); solve() overwrites
  view_t body_velocities_;          //!< registered body translational velocity output (3 per body); solve() writes
  view_t body_angular_velocities_;  //!< registered body angular velocity output (3 per body); solve() writes
  mundy::BelosConfig<double> belos_config_;
  view_t x_;  //!< solved [ q | U | Omega ] (empty when there are no resolved bodies)
  mundy::BelosResult<double> last_result_;
  bool has_body_solve_ = false;  //!< whether last_result_ holds a body solve
};

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA

}  // namespace mbody

}  // namespace mundy

#endif  // MUNDY_MBODY_PERIPHERY_HPP_