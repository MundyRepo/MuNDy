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

/// \file
/// \brief Accuracy studies of the periphery and the bodies and spheres inside it, each written to a CSV.
///
/// Usage: PeripheryStudies <study> [<study> ...]; run it without arguments for the list of studies. Each study writes
/// periphery_studies_<study>.csv to the working directory, for the analyze_*.py scripts beside this file.

// External libs
#include <openrand/philox.h>

// C++ core
#include <algorithm>         // for std::find, std::find_if, std::min, std::max
#include <array>             // for std::array
#include <chrono>            // for std::chrono::steady_clock
#include <cmath>             // for std::abs, std::sqrt, std::acos, std::atan2, std::sin, std::cos
#include <fstream>           // for std::ofstream
#include <initializer_list>  // for std::initializer_list
#include <iomanip>           // for std::setw, std::setprecision
#include <iostream>          // for std::cout, std::cerr
#include <limits>            // for std::numeric_limits
#include <memory>            // for std::make_shared
#include <sstream>           // for std::ostringstream
#include <string>            // for std::string
#include <utility>           // for std::pair
#include <vector>            // for std::vector

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// STK
#include <stk_util/parallel/Parallel.hpp>

// Mundy
#include <mundy_geom/randomize.hpp>             // for mundy::generate_random_unit_quaternion
#include <mundy_math/GaussLegendreSphere.hpp>  // for mundy::gauss_legendre_sphere_rule
#include <mundy_math/Vector3.hpp>              // for Vector3
#include <mundy_math/cmath.hpp>                // for mundy::rsqrt
#include <mundy_mbody/Periphery.hpp>           // for fill_skfie_matrix, apply_skfie, PeripheryT, ...
#include <mundy_utils/rng.hpp>                 // for mundy::make_philox

namespace mundy {

namespace mbody {

namespace {

/// \brief  Keep track of what kind of motile body we are using
enum class SphereType { RPY, Inclusion };

const char* to_string(const SphereType type) {
  return type == SphereType::RPY ? "RPY" : "Inclusion";
}

// ====================================================================================================================
// Motile-body BIE mobility: resolved rigid bodies (each with its own surface quadrature) + RPY spheres + periphery,
// solved as one block matrix-free GMRES system. The studies' shared helpers (make_rpy_body, solve_cavity, sphere_mix,
// wilson_three_sphere) come first, then the mobility studies built on them.
// ====================================================================================================================

// A single RPY body (center point-force)
template <class ExecSpace>
SphereSet<ExecSpace> make_rpy_body(const mundy::Vector3d& center, const double radius, const mundy::Vector3d& force) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  view_t positions("sphere_positions", 3);
  view_t radii("sphere_radii", 1);
  view_t forces("sphere_forces", 3);
  auto h_pos = Kokkos::create_mirror_view(positions);
  auto h_rad = Kokkos::create_mirror_view(radii);
  auto h_frc = Kokkos::create_mirror_view(forces);
  for (int j = 0; j < 3; ++j) {
    h_pos(j) = center[j];
    h_frc(j) = force[j];
  }
  h_rad(0) = radius;
  Kokkos::deep_copy(positions, h_pos);
  Kokkos::deep_copy(radii, h_rad);
  Kokkos::deep_copy(forces, h_frc);
  return SphereSet<ExecSpace>(positions, radii, forces, SphereInteraction::RPYC);
}

// Build the cavity periphery at quadrature order `order`, with its dense inverse self-interaction matrix. This
// is the expensive half of a cavity solve -- a dense O((3 np)^3) inverse -- and depends only on
// (order, cavity_radius, mu), so the sweeps below build it once per periphery order and reuse it across every
// body order.
template <class ExecSpace>
std::shared_ptr<PeripheryT<ExecSpace>> make_cavity_periphery(const int order, const double cavity_radius,
                                                             const double mu) {
  // The (order + 1)-ring sphere rule scaled to the cavity, with inward unit normals.
  std::vector<double> p_pts;
  std::vector<double> p_wts;
  gauss_legendre_sphere_rule(order + 1, p_pts, p_wts);
  std::vector<double> p_nrm(p_pts.size());
  for (size_t i = 0; i < p_pts.size(); ++i) {
    p_nrm[i] = -p_pts[i];
    p_pts[i] *= cavity_radius;
  }
  for (double& w : p_wts) {
    w *= cavity_radius * cavity_radius;
  }
  auto periphery = std::make_shared<PeripheryT<ExecSpace>>(p_wts.size(), mu);
  periphery->set_surface_positions(p_pts.data())
      .set_quadrature_weights(p_wts.data())
      .set_surface_normals(p_nrm.data(), /*outward_normal=*/false);
  periphery->build_inverse_self_interaction_matrix();
  return periphery;
}

// Solve one motile participant of radius `body_radius`, driven by force F and torque tau, at the center of the
// stationary cavity `periphery`. Returns its rigid velocity U and angular velocity Omega both unbounded (no
// cavity) and confined. The unbounded solve isolates the participant's own quadrature error from the
// wall-coupling error. `type` selects the participant: an RPY point sphere (force-only, so `body_order` is
// unused) or a resolved Inclusion carrying its own order-`body_order` surface quadrature.
template <class ExecSpace>
void solve_cavity(const SphereType type, const double mu, const double body_radius, const int body_order,
                  const std::shared_ptr<const PeripheryT<ExecSpace>>& periphery, const mundy::BelosConfig<double>& cfg,
                  const mundy::Vector3d& F, const mundy::Vector3d& tau, mundy::Vector3d& U_unbounded,
                  mundy::Vector3d& Om_unbounded, mundy::Vector3d& U_confined, mundy::Vector3d& Om_confined) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  SphereSet<ExecSpace> spheres;
  BodySet<ExecSpace> bodies;
  if (type == SphereType::RPY) {
    spheres = make_rpy_body<ExecSpace>(mundy::Vector3d(0.0, 0.0, 0.0), body_radius, F);
  } else {
    bodies = make_sphere_body_set<ExecSpace>(body_order, body_radius, mundy::Vector3d(0.0, 0.0, 0.0),
                                             mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0), F, tau);
  }

  auto read_vector3 = [](const view_t& v) {
    auto host = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(host, v);
    return mundy::Vector3d(host(0), host(1), host(2));
  };
  auto run = [&](std::shared_ptr<const PeripheryT<ExecSpace>> p, mundy::Vector3d& U, mundy::Vector3d& Om) {
    view_t u("u", 3);
    view_t om("om", 3);
    MobilitySystem<ExecSpace> mobility(mu);
    mobility.set_periphery(std::move(p)).set_belos_config(cfg);
    if (type == SphereType::RPY) {
      // RPYC and RPY agree here (the sphere never overlaps the wall nodes); RPYC keeps an off-center sweep valid.
      mobility.set_spheres(spheres.positions, spheres.radii, spheres.forces, u, SphereInteraction::RPYC);
    } else {
      mobility.set_bodies(bodies, u, om);
    }
    mobility.solve();
    U = read_vector3(u);
    Om = read_vector3(om);
  };

  run(nullptr, U_unbounded, Om_unbounded);
  run(periphery, U_confined, Om_confined);
}

/// A rigid sphere of radius a translating concentrically inside a stationary
/// spherical cavity of radius b experiences enhanced hydrodynamic drag.
///
/// Let lambda = a / b, with 0 <= lambda < 1. The exact mobility is
///
///   U = F / (6 pi mu a)
///       * (1 - 9 lambda / 4 + 5 lambda^3 / 2
///            - 9 lambda^5 / 4 + lambda^6)
///       / (1 - lambda^5).
///
/// Equivalently,
///
///   F = 6 pi mu a U
///       * (1 - lambda^5)
///       / (1 - 9 lambda / 4 + 5 lambda^3 / 2
///            - 9 lambda^5 / 4 + lambda^6).
///
/// See Happel & Brenner, Low Reynolds Number Hydrodynamics,
/// Sec. 4-22, Eq. (4-22.11).
void motile_body_in_spherical_cavity_drag(std::ostream& csv, const SphereType type, const double body_radius,
                                          const double cavity_radius, const std::vector<int>& body_orders,
                                          const std::vector<int>& periphery_orders) {
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;

  const double mu = 1.0;
  const double lambda = body_radius / cavity_radius;

  // Exact concentric-cavity translation mobility: U = U_stokes * poly(lambda) / (1 - lambda^5).
  const double u_stokes = 1.0 / (6.0 * M_PI * mu * body_radius);
  const double l3 = lambda * lambda * lambda;
  const double l5 = l3 * lambda * lambda;
  const double l6 = l5 * lambda;
  const double poly = 1.0 - 2.25 * lambda + 2.5 * l3 - 2.25 * l5 + l6;
  const double U_exact = u_stokes * poly / (1.0 - l5);

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 1000;
  cfg.num_blocks = 150;
  cfg.max_restarts = 30;

  const mundy::Vector3d F(1.0, 0.0, 0.0);
  const mundy::Vector3d no_torque(0.0, 0.0, 0.0);

  // An RPY point sphere has no surface quadrature, so the body-order axis is meaningless for it: one pass,
  // reported as body order 0.
  const std::vector<int> b_orders = (type == SphereType::RPY) ? std::vector<int>{0} : body_orders;
  const size_t nb = b_orders.size();
  const size_t np = periphery_orders.size();
  std::vector<std::vector<double>> u_conf(nb, std::vector<double>(np, 0.0));  // for the Richardson extrapolation

  std::cout << "MotileBodyInSphericalCavityDrag[" << to_string(type) << "]: body_radius=" << body_radius
            << " cavity_radius=" << cavity_radius << " lambda=" << lambda << "  U_stokes=" << u_stokes
            << "  U_exact=" << U_exact << "\n";
  for (size_t ip = 0; ip < np; ++ip) {
    auto periphery = make_cavity_periphery<ExecSpace>(periphery_orders[ip], cavity_radius, mu);
    for (size_t ib = 0; ib < nb; ++ib) {
      mundy::Vector3d U_unb;
      mundy::Vector3d Om_unb;
      mundy::Vector3d U_conf;
      mundy::Vector3d Om_conf;
      solve_cavity<ExecSpace>(type, mu, body_radius, b_orders[ib], periphery, cfg, F, no_torque, U_unb, Om_unb, U_conf,
                              Om_conf);
      // Every error reported here and below is RELATIVE: |computed - exact| / exact.
      const double rel_err_unb = std::abs(U_unb[0] - u_stokes) / u_stokes;
      const double rel_err_conf = std::abs(U_conf[0] - U_exact) / U_exact;
      u_conf[ib][ip] = U_conf[0];
      csv << "Drag," << to_string(type) << ',' << body_radius << ',' << cavity_radius << ',' << lambda << ','
          << b_orders[ib] << ',' << periphery_orders[ip] << ',' << periphery->get_num_nodes() << ',' << U_unb[0] << ','
          << U_conf[0] << ',' << U_exact << ',' << rel_err_unb << ',' << rel_err_conf << '\n';
    }
  }

  // Richardson in 1/periphery_order at fixed body order: the body's own quadrature represents rigid sphere
  // translation exactly (U_unb is machine-exact at every order), so the periphery axis carries the whole
  // wall-coupling error.
  const double o1 = periphery_orders[np - 2];
  const double o2 = periphery_orders[np - 1];
  for (size_t ib = 0; ib < nb; ++ib) {
    const double slope = (u_conf[ib][np - 2] - u_conf[ib][np - 1]) / (1.0 / o1 - 1.0 / o2);
    const double u_inf = u_conf[ib][np - 1] - slope / o2;  // extrapolate 1/order -> 0 (infinite resolution)
    const double rel_err_inf = std::abs(u_inf - U_exact) / U_exact;
    std::cout << "  body_order=" << b_orders[ib] << "  Richardson (1/periphery_order) limit = " << u_inf
              << "  rel_err(U vs exact) = " << rel_err_inf << "\n";
  }
  std::cout << std::flush;
}

/// A rigid sphere of radius a rotating concentrically inside a stationary
/// spherical cavity of radius b experiences enhanced hydrodynamic resistance.
///
/// Let lambda = a / b, with 0 <= lambda < 1. For an applied torque T, the
/// exact rotational mobility is
///
///   Omega = T / (8 pi mu a^3) * (1 - lambda^3).
///
/// Equivalently, the hydrodynamic torque opposing rotation is
///
///   T_hydro = -8 pi mu a^3 Omega / (1 - lambda^3).
///
/// Thus, the magnitude of the applied torque required to maintain angular
/// velocity Omega is
///
///   T = 8 pi mu a^3 Omega / (1 - lambda^3).
///
/// In the unbounded-fluid limit, lambda -> 0, this reduces to the standard
/// rotational Stokes law,
///
///   T = 8 pi mu a^3 Omega.
///
/// See Happel & Brenner, Low Reynolds Number Hydrodynamics,
/// Sec. 7-8, Eqs. (7-8.18)--(7-8.20).
void motile_body_in_spherical_cavity_rotation(std::ostream& csv, const SphereType type, const double body_radius,
                                              const double cavity_radius, const std::vector<int>& body_orders,
                                              const std::vector<int>& periphery_orders) {
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;

  if (type == SphereType::RPY) {
    std::cout << "MotileBodyInSphericalCavityRotation[RPY]: skipped -- RPY spheres are force-only (no torque in, "
                 "no angular velocity out)."
              << std::endl;
    return;
  }

  const double mu = 1.0;
  const double lambda = body_radius / cavity_radius;

  // Exact concentric-cavity rotational mobility: Omega = Omega_stokes * (1 - lambda^3).
  const double om_stokes =
      1.0 / (8.0 * M_PI * mu * body_radius * body_radius * body_radius);  // per unit torque, unbounded
  const double l3 = lambda * lambda * lambda;
  const double Om_exact = om_stokes * (1.0 - l3);

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 1000;
  cfg.num_blocks = 150;
  cfg.max_restarts = 30;

  const mundy::Vector3d no_force(0.0, 0.0, 0.0);
  const mundy::Vector3d tau(0.0, 0.0, 1.0);  // unit torque about z

  const size_t nb = body_orders.size();
  const size_t np = periphery_orders.size();
  std::vector<std::vector<double>> om_conf(nb, std::vector<double>(np, 0.0));  // for the Richardson extrapolation

  std::cout << "MotileBodyInSphericalCavityRotation[" << to_string(type) << "]: body_radius=" << body_radius
            << " cavity_radius=" << cavity_radius << " lambda=" << lambda << "  Om_stokes=" << om_stokes
            << "  Om_exact=" << Om_exact << "\n";
  for (size_t ip = 0; ip < np; ++ip) {
    auto periphery = make_cavity_periphery<ExecSpace>(periphery_orders[ip], cavity_radius, mu);
    for (size_t ib = 0; ib < nb; ++ib) {
      mundy::Vector3d U_unb;
      mundy::Vector3d Om_unb;
      mundy::Vector3d U_conf;
      mundy::Vector3d Om_conf;
      solve_cavity<ExecSpace>(type, mu, body_radius, body_orders[ib], periphery, cfg, no_force, tau, U_unb, Om_unb,
                              U_conf, Om_conf);
      // Every error reported here and below is RELATIVE: |computed - exact| / exact.
      const double rel_err_unb = std::abs(Om_unb[2] - om_stokes) / om_stokes;
      const double rel_err_conf = std::abs(Om_conf[2] - Om_exact) / Om_exact;
      om_conf[ib][ip] = Om_conf[2];
      csv << "Rotation," << to_string(type) << ',' << body_radius << ',' << cavity_radius << ',' << lambda << ','
          << body_orders[ib] << ',' << periphery_orders[ip] << ',' << periphery->get_num_nodes() << ',' << Om_unb[2]
          << ',' << Om_conf[2] << ',' << Om_exact << ',' << rel_err_unb << ',' << rel_err_conf << '\n';
    }
  }

  // Richardson in 1/periphery_order at fixed body order; unbounded rotational drag is machine-exact at every
  // order, so all of the confined error is wall coupling.
  const double o1 = periphery_orders[np - 2];
  const double o2 = periphery_orders[np - 1];
  for (size_t ib = 0; ib < nb; ++ib) {
    const double slope = (om_conf[ib][np - 2] - om_conf[ib][np - 1]) / (1.0 / o1 - 1.0 / o2);
    const double om_inf = om_conf[ib][np - 1] - slope / o2;  // extrapolate 1/order -> 0 (infinite resolution)
    const double rel_err_inf = std::abs(om_inf - Om_exact) / Om_exact;
    std::cout << "  body_order=" << body_orders[ib] << "  Richardson (1/periphery_order) limit = " << om_inf
              << "  rel_err(Om vs exact) = " << rel_err_inf << "\n";
  }
  std::cout << std::flush;
}

// Solve N equal spheres of radius r at `pos` with forces `frc` (zero torques), each independently typed as a point
// sphere or a resolved rigid inclusion (a BIE body), through MobilitySystem -- the same path HP1 uses. Point spheres
// interact via `interaction` (HP1's default is RPYC). Inclusions are placed with `orient` (only their quadrature grid
// rotates; the rigid velocities are lab-frame). An optional `periphery` confines everything; empty = free space.
// Returns the N translational velocities; GMRES iterations go to *num_iters (0 when there are no inclusions).
template <class ExecSpace, size_t N>
std::array<mundy::Vector3d, N> sphere_mix(const double mu, const double r, const std::array<mundy::Vector3d, N>& pos,
                                          const std::array<mundy::Vector3d, N>& frc, const int order,
                                          const mundy::BelosConfig<double>& cfg, const std::array<SphereType, N>& types,
                                          const std::array<mundy::Quaternion<double>, N>& orient,
                                          unsigned* num_iters = nullptr,
                                          const SphereInteraction interaction = SphereInteraction::RPYC,
                                          const std::shared_ptr<const PeripheryT<ExecSpace>>& periphery = nullptr) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  // Partition the objects into resolved bodies (Inclusion) and RPY spheres, tracking their global indices.
  std::vector<int> body_gi;
  std::vector<int> sph_gi;
  for (int j = 0; j < static_cast<int>(N); ++j) {
    if (types[j] == SphereType::Inclusion) {
      body_gi.push_back(j);
    } else {
      sph_gi.push_back(j);
    }
  }

  BodySet<ExecSpace> body_set;
  for (size_t m = 0; m < body_gi.size(); ++m) {
    body_set.bodies.push_back(make_sphere_body<ExecSpace>(order, r));
  }
  body_set.wire_pose_load();
  for (size_t m = 0; m < body_gi.size(); ++m) {
    const int gi = body_gi[m];
    body_set.set_center(m, pos[gi]);
    body_set.set_orientation(m, orient[gi]);
    body_set.set_net_force(m, frc[gi]);
    body_set.set_net_torque(m, mundy::Vector3d(0.0, 0.0, 0.0));
    place_body_in_lab_frame(ExecSpace{}, body_set.bodies[m]);
  }

  const size_t nb = body_gi.size();
  const size_t ns = sph_gi.size();
  MobilitySystem<ExecSpace> mobility(mu);
  mobility.set_belos_config(cfg).set_periphery(periphery);

  view_t body_vel("body_vel", 3 * nb);
  view_t body_omega("body_omega", 3 * nb);
  mobility.set_bodies(body_set, body_vel, body_omega);

  view_t u_spheres;
  if (ns > 0) {
    view_t sph_pos("sph_pos", 3 * ns);
    view_t sph_rad("sph_rad", ns);
    view_t sph_frc("sph_frc", 3 * ns);
    auto hp = Kokkos::create_mirror_view(sph_pos);
    auto hr = Kokkos::create_mirror_view(sph_rad);
    auto hf = Kokkos::create_mirror_view(sph_frc);
    for (size_t a = 0; a < ns; ++a) {
      hr(a) = r;
      for (int d = 0; d < 3; ++d) {
        hp(3 * a + d) = pos[sph_gi[a]][d];
        hf(3 * a + d) = frc[sph_gi[a]][d];
      }
    }
    Kokkos::deep_copy(sph_pos, hp);
    Kokkos::deep_copy(sph_rad, hr);
    Kokkos::deep_copy(sph_frc, hf);
    u_spheres = view_t("u_spheres", 3 * ns);
    mobility.set_spheres(sph_pos, sph_rad, sph_frc, u_spheres, interaction);
  }

  mobility.solve();

  std::array<mundy::Vector3d, N> v;
  if (num_iters) {
    *num_iters = 0;
  }
  if (nb > 0) {
    const auto& res = mobility.body_solve_result();
    if (!res.converged) {
      std::cerr << "GMRES failed: " << res << "\n";
    }
    if (num_iters) {
      *num_iters = res.num_iters;
    }
    auto hv = Kokkos::create_mirror_view(body_vel);
    Kokkos::deep_copy(hv, body_vel);
    for (size_t bi = 0; bi < nb; ++bi) {
      v[body_gi[bi]] = mundy::Vector3d(hv(3 * bi), hv(3 * bi + 1), hv(3 * bi + 2));
    }
  }
  if (ns > 0) {
    auto hu = Kokkos::create_mirror_view(u_spheres);
    Kokkos::deep_copy(hu, u_spheres);
    for (size_t a = 0; a < ns; ++a) {
      v[sph_gi[a]] = mundy::Vector3d(hu(3 * a), hu(3 * a + 1), hu(3 * a + 2));
    }
  }
  return v;
}

// Orientations of the N spheres' quadrature grids for ensemble sample `sample`. Sample 0 is the identity (the
// unrandomized grid). Otherwise each is uniform on SO(3): mundy's random quaternion carries the grid pole z to a
// uniform direction, and a uniform spin about the pole first randomizes the azimuthal phase that parallel transport
// leaves fixed. Seeded only by (sample, sphere), so every s and order sees the same ensemble.
template <size_t N>
std::array<mundy::Quaternion<double>, N> sphere_orientations(const int sample) {
  std::array<mundy::Quaternion<double>, N> q;
  for (size_t m = 0; m < N; ++m) {
    if (sample == 0) {
      q[m] = mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0);
      continue;
    }
    openrand::Philox rng(1234, static_cast<uint32_t>(sample * N + m));
    const auto pole = mundy::generate_random_unit_quaternion<double>(rng);
    const double half_phi = M_PI * rng.rand<double>();
    q[m] = pole * mundy::Quaternion<double>(std::cos(half_phi), 0.0, 0.0, std::sin(half_phi));
  }
  return q;
}

// Wilson triangle: equilateral, side s*r, in the x-y plane with s1 at the origin (see the diagram below).
std::array<mundy::Vector3d, 3> wilson_triangle(const double s, const double r) {
  const double sqrt3_2 = std::sqrt(3.0) / 2.0;
  std::array<mundy::Vector3d, 3> pos;
  pos[0] = mundy::Vector3d(0.0, 0.0, 0.0);
  pos[1] = mundy::Vector3d(-s * r / 2.0, -sqrt3_2 * s * r, 0.0);
  pos[2] = mundy::Vector3d(s * r / 2.0, -sqrt3_2 * s * r, 0.0);
  return pos;
}

// Solve the Wilson three-sphere problem at separation s: equilateral triangle of side s*r, force F1 along y on s1.
// Returns the three velocities in the s1/s2/s3 slots.
template <class ExecSpace>
std::array<mundy::Vector3d, 3> three_sphere_wilson_mix(const double mu, const double r, const double F1, const double s,
                                                       const int order, const mundy::BelosConfig<double>& cfg,
                                                       const std::array<SphereType, 3>& types,
                                                       const std::array<mundy::Quaternion<double>, 3>& orient) {
  std::array<mundy::Vector3d, 3> frc;
  frc[0] = mundy::Vector3d(0.0, F1, 0.0);
  frc[1] = mundy::Vector3d(0.0, 0.0, 0.0);
  frc[2] = mundy::Vector3d(0.0, 0.0, 0.0);
  return sphere_mix<ExecSpace, 3>(mu, r, wilson_triangle(s, r), frc, order, cfg, types, orient);
}

// The most accurate pseudo-analytical results for the hydrodynamic interaction between three spheres is given in
// Wilson 2013 Stokes flow past three spheres.

// The three spheres are located at the corners of an equilateral triangle in the x-y plane.
// The side length of the triangle is s * r, and the spheres have radius r.
/*     s1
//      o
//     / \
// s2 o---o s3
//
// s1 is located at x = 0, y = 0, z = 0
// s2 is located at x = -d/2, y = -sqrt(3)/2 * d, z = 0
// s3 is located at x = d/2, y = -sqrt(3)/2 * d, z = 0
//
// Apply a single force to s1 of f_s1 = (0, -6 pi mu r, 0), and zero forces to s2 and s3.
// We will run this test for various values of s, but RPY should only be accurate for large s.
//
// If we define U1, U2 as the component of velocity of s1 and s2 in the y-direction and
// Y3 as the component of velocity of s3 in the x-direction. We also let omega1, omega2, omega3
// denote the magnitude of rotational velocity of s1, s2, s3.
*/
//
// One test, run over the full matrix of sphere types -- each of the three spheres independently an RPY point
// sphere (R) or a resolved rigid inclusion (I). As more spheres are resolved the result approaches Wilson's exact
// reference; with all three resolved it matches to quadrature accuracy. This covers the former standalone
// inclusion/three-body tests without duplication:
//   {R, R, R}  point-RPY reference (misses the back-reaction on the forced sphere: U1 == 1)
//   {R, I, R}  one passive inclusion (its resolved stresslet slows the forced sphere)
//   {I, R, R}  the forced sphere resolved (feels no back-reaction from force-free RPY neighbors: bare drag)
//   {I, I, I}  all resolved: reproduces Wilson's exact reference to quadrature accuracy
// The stdout table is sample 0 (unrotated grids); the CSV has every orientation sample for configs with an inclusion.
void wilson_three_sphere(std::ostream& csv, const int order, const int num_samples) {
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;

  const double mu = 1.0;
  const double r = 12.34;
  const double F1 = -6.0 * M_PI * mu * r;  // force on s1; bare velocity = -1

  const std::vector<double> svals = {2.5, 3.0, 4.0, 6.0};
  const std::vector<double> W_U1 = {0.87765, 0.93905, 0.97964, 0.99581};
  const std::vector<double> W_U2 = {0.49545, 0.41694, 0.31859, 0.21586};
  const std::vector<double> W_U3 = {0.07393, 0.07824, 0.06925, 0.05078};

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 2000;
  cfg.num_blocks = 200;
  cfg.max_restarts = 40;

  using T = SphereType;
  struct Config {
    const char* label;
    std::array<T, 3> types;
  };
  const std::array<Config, 4> configs = {{{"RRR", {{T::RPY, T::RPY, T::RPY}}},
                                          {"RIR", {{T::RPY, T::Inclusion, T::RPY}}},
                                          {"IRR", {{T::Inclusion, T::RPY, T::RPY}}},
                                          {"III", {{T::Inclusion, T::Inclusion, T::Inclusion}}}}};

  auto U1 = [](const std::array<mundy::Vector3d, 3>& v) { return -v[0][1]; };
  auto U2 = [](const std::array<mundy::Vector3d, 3>& v) { return -v[1][1]; };
  auto U3 = [](const std::array<mundy::Vector3d, 3>& v) { return v[2][0]; };
  auto prow = [&](const char* name, double w, const std::array<double, 4>& c) {
    std::cout << std::setw(8) << std::left << (std::string("  ") + name) << std::right << " | " << std::setw(11) << w;
    for (int k = 0; k < 4; ++k) {
      std::cout << " | " << std::setw(11) << c[k];
    }
    std::cout << "\n";
  };

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "WilsonThreeSphere: U1 = -vy(s1), U2 = -vy(s2), U3 = vx(s3)\n";
  std::cout << "  order = " << order << std::endl;
  std::cout << std::setw(8) << std::left << "" << std::right << " | " << std::setw(11) << "Wilson";
  for (const auto& config : configs) {
    std::cout << " | " << std::setw(11) << config.label;
  }
  std::cout << "\n";

  for (size_t i = 0; i < svals.size(); ++i) {
    const double s = svals[i];
    std::array<std::array<mundy::Vector3d, 3>, 4> vel;
    for (int c = 0; c < 4; ++c) {
      const bool has_inclusion =
          std::find(configs[c].types.begin(), configs[c].types.end(), T::Inclusion) != configs[c].types.end();
      for (int sample = 0; sample < (has_inclusion ? num_samples : 1); ++sample) {
        const auto v = three_sphere_wilson_mix<ExecSpace>(mu, r, F1, s, order, cfg, configs[c].types,
                                                          sphere_orientations<3>(sample));
        if (sample == 0) {
          vel[c] = v;
        }
        csv << order << ',' << s << ',' << configs[c].label << ',' << sample << ',' << U1(v) << ',' << U2(v) << ','
            << U3(v) << '\n';
      }
    }

    std::cout << "s = " << s << "\n";
    prow("U1", W_U1[i], {U1(vel[0]), U1(vel[1]), U1(vel[2]), U1(vel[3])});
    prow("U2", W_U2[i], {U2(vel[0]), U2(vel[1]), U2(vel[2]), U2(vel[3])});
    prow("U3", W_U3[i], {U3(vel[0]), U3(vel[1]), U3(vel[2]), U3(vel[3])});
  }
  std::cout << std::flush;
}

// Cholesky of the symmetric part (M + M^T)/2 of an n x n row-major matrix. Returns the smallest pivot; the matrix is
// SPD iff it is > 0. Sets *asym to ||M - M^T||_F / ||M||_F.
double sym_part_min_pivot(const std::vector<double>& M, const int n, double* asym) {
  double num = 0.0;
  double den = 0.0;
  std::vector<double> A(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double diff = M[i * n + j] - M[j * n + i];
      num += diff * diff;
      den += M[i * n + j] * M[i * n + j];
      A[i * n + j] = 0.5 * (M[i * n + j] + M[j * n + i]);
    }
  }
  *asym = std::sqrt(num / den);

  double min_pivot = std::numeric_limits<double>::max();
  for (int k = 0; k < n; ++k) {
    double pivot = A[k * n + k];
    for (int j = 0; j < k; ++j) {
      pivot -= A[k * n + j] * A[k * n + j];
    }
    min_pivot = std::min(min_pivot, pivot);
    if (pivot <= 0.0) {
      return min_pivot;
    }
    const double lkk = std::sqrt(pivot);
    A[k * n + k] = lkk;
    for (int i = k + 1; i < n; ++i) {
      double sum = A[i * n + k];
      for (int j = 0; j < k; ++j) {
        sum -= A[i * n + j] * A[k * n + j];
      }
      A[i * n + k] = sum / lkk;
    }
  }
  return min_pivot;
}

template <size_t N>
struct MobilityConfig {
  const char* label;
  std::array<SphereType, N> types;
  SphereInteraction interaction;
};

// Comma-joined CSV key columns at the study CSVs' precision.
std::string csv_key(std::initializer_list<double> vals) {
  std::ostringstream key;
  key << std::setprecision(12);
  for (auto it = vals.begin(); it != vals.end(); ++it) {
    key << (it == vals.begin() ? "" : ",") << *it;
  }
  return key.str();
}

// For N spheres of radius r at `pos_of_sample(sample)` and each type combination, assemble the 3N x 3N translational
// mobility (zero torques) column by column from unit-force solves, nondimensionalized by the bare mobility
// 1/(6 pi mu r), and write it to csv as one row per (config, sample), led by the `key` columns. Configs with an
// inclusion are repeated over `num_samples` quadrature-grid orientations (sphere_orientations; sample 0 is the
// unrotated grid) to ensemble-average over grid alignment; in free space all-RPY configs are orientation-independent
// and run once. With a `periphery` every config runs every sample, since pos_of_sample also moves the spheres relative
// to the periphery grid. No overlap guard: overlapping configurations are kept only to see how they break down.
// Prints one progress line with the largest asymmetry and smallest Cholesky pivot over the samples of each config.
template <size_t N, size_t C, class PosFn>
void mobility_matrix_study(
    std::ostream& csv, const std::string& key, const int order, const double r, const PosFn& pos_of_sample,
    const std::array<MobilityConfig<N>, C>& configs, const int num_samples,
    const std::shared_ptr<const PeripheryT<Tpetra::Map<>::node_type::execution_space>>& periphery = nullptr) {
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  constexpr int n = 3 * N;

  const double mu = 1.0;
  const double drag = 6.0 * M_PI * mu * r;

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 2000;
  cfg.num_blocks = 200;
  cfg.max_restarts = 40;

  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "[" << key << "]" << std::scientific << std::setprecision(2);
  for (const auto& config : configs) {
    const bool has_inclusion =
        std::find(config.types.begin(), config.types.end(), SphereType::Inclusion) != config.types.end();
    const int config_samples = (has_inclusion || periphery) ? num_samples : 1;

    double max_asym = 0.0;
    double min_pivot = std::numeric_limits<double>::max();
    unsigned total_iters = 0;
    for (int sample = 0; sample < config_samples; ++sample) {
      const auto orient = sphere_orientations<N>(sample);
      const std::array<mundy::Vector3d, N> pos = pos_of_sample(sample);
      std::vector<double> M(n * n);
      unsigned sample_iters = 0;
      for (int k = 0; k < n; ++k) {
        std::array<mundy::Vector3d, N> frc;
        for (auto& f : frc) {
          f = mundy::Vector3d(0.0, 0.0, 0.0);
        }
        frc[k / 3][k % 3] = 1.0;
        unsigned iters = 0;
        const auto v = sphere_mix<ExecSpace, N>(mu, r, pos, frc, order, cfg, config.types, orient, &iters,
                                                config.interaction, periphery);
        sample_iters += iters;
        for (int i = 0; i < n; ++i) {
          M[i * n + k] = drag * v[i / 3][i % 3];
        }
      }

      double asym = 0.0;
      min_pivot = std::min(min_pivot, sym_part_min_pivot(M, n, &asym));
      max_asym = std::max(max_asym, asym);
      total_iters += sample_iters;

      csv << key << ',' << config.label << ',' << sample << ',' << sample_iters;
      for (const double m : M) {
        csv << ',' << m;
      }
      csv << '\n';
    }
    std::cout << "  " << config.label << " asym=" << max_asym << " piv=" << min_pivot << " it=" << total_iters << " |";
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << std::fixed << std::setprecision(1) << "  (" << secs << " s)" << std::endl;
}

// Two spheres on the x-axis at center separation s*r. RR is run with both the uncorrected RPY kernel and HP1's
// overlap-corrected RPYC; IR/II use RPYC as in HP1.
void two_sphere_spd(std::ostream& csv, const int order, const double s, const double r, const int num_samples) {
  using T = SphereType;
  using SI = SphereInteraction;
  const std::array<MobilityConfig<2>, 4> configs = {{{"RR_RPY", {{T::RPY, T::RPY}}, SI::RPY},
                                                     {"RR_RPYC", {{T::RPY, T::RPY}}, SI::RPYC},
                                                     {"IR", {{T::Inclusion, T::RPY}}, SI::RPYC},
                                                     {"II", {{T::Inclusion, T::Inclusion}}, SI::RPYC}}};
  std::array<mundy::Vector3d, 2> pos;
  pos[0] = mundy::Vector3d(0.0, 0.0, 0.0);
  pos[1] = mundy::Vector3d(s * r, 0.0, 0.0);
  mobility_matrix_study(
      csv, csv_key({static_cast<double>(order), s, r}), order, r, [&](int) { return pos; }, configs, num_samples);
}

// Three spheres at the Wilson triangle (side s*r). The triangle is symmetric under relabeling, so one of each
// inclusion count covers every type combination: RRR (RPY and RPYC), IRR, IIR, III (RPYC as in HP1).
void three_sphere_spd(std::ostream& csv, const int order, const double s, const double r, const int num_samples) {
  using T = SphereType;
  using SI = SphereInteraction;
  const std::array<MobilityConfig<3>, 5> configs = {{{"RRR_RPY", {{T::RPY, T::RPY, T::RPY}}, SI::RPY},
                                                     {"RRR_RPYC", {{T::RPY, T::RPY, T::RPY}}, SI::RPYC},
                                                     {"IRR", {{T::Inclusion, T::RPY, T::RPY}}, SI::RPYC},
                                                     {"IIR", {{T::Inclusion, T::Inclusion, T::RPY}}, SI::RPYC},
                                                     {"III", {{T::Inclusion, T::Inclusion, T::Inclusion}}, SI::RPYC}}};
  const auto pos = wilson_triangle(s, r);
  mobility_matrix_study(
      csv, csv_key({static_cast<double>(order), s, r}), order, r, [&](int) { return pos; }, configs, num_samples);
}

// The two extreme approach directions against a Gauss-Legendre sphere periphery grid: straight at the node nearest +x,
// and at the center of the grid cell above it (half an azimuthal step, 2 pi / (2p + 2), and halfway in elevation to
// the next Gauss-Legendre latitude on that meridian).
template <class ExecSpace>
std::pair<mundy::Vector3d, mundy::Vector3d> periphery_node_and_cell_dirs(const PeripheryT<ExecSpace>& periphery,
                                                                         const int periphery_order) {
  auto h = Kokkos::create_mirror_view(periphery.get_surface_positions());
  Kokkos::deep_copy(h, periphery.get_surface_positions());
  const size_t np = periphery.get_num_nodes();
  size_t n0 = 0;
  for (size_t i = 1; i < np; ++i) {
    if (h(3 * i) > h(3 * n0)) {
      n0 = i;
    }
  }
  const mundy::Vector3d node(h(3 * n0), h(3 * n0 + 1), h(3 * n0 + 2));
  const double R = mundy::norm(node);
  const double e0 = std::asin(node[2] / R);
  double e1 = M_PI / 2.0;  // next latitude up on the phi = 0 meridian (the pole if n0 is on the top row)
  for (size_t i = 0; i < np; ++i) {
    if (h(3 * i) > 0.0 && std::abs(h(3 * i + 1)) < 1.0e-12 * R) {
      const double e = std::asin(h(3 * i + 2) / R);
      if (e > e0 + 1.0e-12 && e < e1) {
        e1 = e;
      }
    }
  }
  const double el = 0.5 * (e0 + e1);
  const double az = M_PI / (2.0 * periphery_order + 2.0);
  const mundy::Vector3d cell(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
  return {node / R, cell};
}

// One sphere of radius r inside the cavity `periphery` of radius R, its center at distance s*r from the wall (s = 1
// touches it). The full 3x3 mobility M, which is SPD exactly when F . (M F) > 0 for every force F (the deleted
// SphereQuadPeripheryRPYC diagnostic checked F . V along a few directions). The approach direction depends on the
// sample: 0 = straight at a periphery node, 1 = at a
// grid-cell center (the two extremes of alignment with the periphery grid), k >= 2 = uniformly random (+x rotated by
// sphere_orientations<1>(k)). The inclusion's own grid is oriented by sphere_orientations<1>(k) in every sample.
// R_RPY and R_RPYC differ only once the sphere overlaps a periphery node.
void periphery_sphere_spd(std::ostream& csv, const int order,
                          const std::shared_ptr<const PeripheryT<Tpetra::Map<>::node_type::execution_space>>& periphery,
                          const int periphery_order, const double periphery_radius, const double s, const double r,
                          const int num_samples) {
  using T = SphereType;
  using SI = SphereInteraction;
  const std::array<MobilityConfig<1>, 3> configs = {
      {{"R_RPY", {{T::RPY}}, SI::RPY}, {"R_RPYC", {{T::RPY}}, SI::RPYC}, {"I", {{T::Inclusion}}, SI::RPYC}}};
  const auto dirs = periphery_node_and_cell_dirs(*periphery, periphery_order);
  auto pos_of_sample = [&](int sample) {
    const mundy::Vector3d dir = sample == 0   ? dirs.first
                                : sample == 1 ? dirs.second
                                              : sphere_orientations<1>(sample)[0] * mundy::Vector3d(1.0, 0.0, 0.0);
    return std::array<mundy::Vector3d, 1>{(periphery_radius - s * r) * dir};
  };
  const std::string key = csv_key({static_cast<double>(order), static_cast<double>(periphery_order),
                                   static_cast<double>(periphery->get_num_nodes()), periphery_radius, s, r});
  mobility_matrix_study(csv, key, order, r, pos_of_sample, configs, num_samples, periphery);
}

// ====================================================================================================================
// The double layer on a sphere, apart from any mobility: the operators behind the periphery's second-kind equation.
// ====================================================================================================================

using StudyExecSpace = Kokkos::DefaultExecutionSpace;
using StudyVector = Kokkos::View<double*, Kokkos::LayoutLeft, StudyExecSpace::memory_space>;
using StudyMatrix = Kokkos::View<double**, Kokkos::LayoutLeft, StudyExecSpace::memory_space>;

/// \brief The double-layer operators the double_layer study compares (see Periphery.hpp's header).
enum class DoubleLayerOperator {
  TDense,                   //!< The punctured double layer T, filled by fill_stokes_double_layer_matrix
  TMatrixFree,              //!< T, applied by apply_stokes_double_layer_kernel
  InteriorTraceDense,       //!< The interior trace J + T, by add_singularity_subtraction on T
  ExteriorTraceMatrixFree,  //!< The singularity-subtracted exterior trace, by apply_stokes_double_layer_kernel_ss
  SkfieDense,               //!< The second-kind operator M = J + T + N, filled by fill_skfie_matrix
  SkfieMatrixFree           //!< M, applied by apply_skfie
};

constexpr std::array<DoubleLayerOperator, 6> kDoubleLayerOperators = {
    DoubleLayerOperator::TDense,     DoubleLayerOperator::TMatrixFree,
    DoubleLayerOperator::InteriorTraceDense, DoubleLayerOperator::ExteriorTraceMatrixFree,
    DoubleLayerOperator::SkfieDense, DoubleLayerOperator::SkfieMatrixFree};

const char* to_string(const DoubleLayerOperator op) {
  switch (op) {
    case DoubleLayerOperator::TDense:
      return "T_dense";
    case DoubleLayerOperator::TMatrixFree:
      return "T_matrix_free";
    case DoubleLayerOperator::InteriorTraceDense:
      return "interior_trace_dense";
    case DoubleLayerOperator::ExteriorTraceMatrixFree:
      return "exterior_trace_matrix_free";
    case DoubleLayerOperator::SkfieDense:
      return "skfie_dense";
    case DoubleLayerOperator::SkfieMatrixFree:
      return "skfie_matrix_free";
  }
  return "unknown";
}

/// \brief The densities the double_layer study applies the operators to.
enum class SurfaceField {
  Constant,  //!< (1, 1, 1)
  Normal,    //!< The outward unit normal x / |x|
  Mixed      //!< See surface_field_value
};

constexpr std::array<SurfaceField, 3> kSurfaceFields = {SurfaceField::Constant, SurfaceField::Normal,
                                                        SurfaceField::Mixed};

const char* to_string(const SurfaceField field) {
  switch (field) {
    case SurfaceField::Constant:
      return "constant";
    case SurfaceField::Normal:
      return "normal";
    case SurfaceField::Mixed:
      return "mixed";
  }
  return "unknown";
}

/// \brief The value of field at the sphere point (x, y, z).
///
/// Mixed is the field of the deleted StokesDoubleLayerSmoothForces diagnostic. With r = |x| and theta and phi the polar
/// and azimuthal angles, it is
///   (sin(theta) cos(theta) + cos(theta) cos(phi) / r, sin^2(theta) sin(phi) - cos(theta) sin(phi) / r,
///    sin^2(theta) cos(phi)),
/// which that diagnostic labeled sin(theta) n + cos(theta) t_theta + cos^2(theta) t_phi. Its cos(phi) / r and
/// sin(phi) / r terms do not vanish at the poles, where phi is undefined, so it is not smooth there.
Vector3d surface_field_value(const SurfaceField field, const double x, const double y, const double z) {
  const double inv_radius = rsqrt(x * x + y * y + z * z);
  switch (field) {
    case SurfaceField::Constant:
      return Vector3d(1.0, 1.0, 1.0);
    case SurfaceField::Normal:
      return Vector3d(x * inv_radius, y * inv_radius, z * inv_radius);
    case SurfaceField::Mixed: {
      const double theta = std::acos(z * inv_radius);
      const double phi = std::atan2(y, x);
      const double sin_theta = std::sin(theta);
      const double cos_theta = std::cos(theta);
      return Vector3d(sin_theta * z * inv_radius + cos_theta * std::cos(phi) * inv_radius,
                      sin_theta * y * inv_radius - cos_theta * std::sin(phi) * inv_radius, sin_theta * x * inv_radius);
    }
  }
  return Vector3d(0.0, 0.0, 0.0);
}

/// \brief A sphere quadrature with inward normals, the periphery's convention, on the host and in StudyExecSpace.
struct DoubleLayerSurface {
  size_t num_nodes;
  std::vector<double> points;   //!< 3 N node positions, on the host
  std::vector<double> weights;  //!< N weights, on the host
  StudyVector d_points;         //!< points in StudyExecSpace's memory
  StudyVector d_weights;        //!< weights in StudyExecSpace's memory
  StudyVector d_normals;        //!< 3 N inward unit normals in StudyExecSpace's memory
};

/// \brief The (order + 1)-ring Gauss-Legendre rule on the sphere of the given radius, with inward normals.
DoubleLayerSurface make_double_layer_surface(const int order, const double radius) {
  std::vector<double> normals;  // the unit-sphere points, which are its outward unit normals
  std::vector<double> weights;
  gauss_legendre_sphere_rule(order + 1, normals, weights);
  std::vector<double> points(normals.size());
  for (size_t i = 0; i < normals.size(); ++i) {
    points[i] = radius * normals[i];
    normals[i] = -normals[i];
  }
  for (double& weight : weights) {
    weight *= radius * radius;
  }
  return {weights.size(),
          points,
          weights,
          to_device<StudyExecSpace>(points),
          to_device<StudyExecSpace>(weights),
          to_device<StudyExecSpace>(normals)};
}

/// \brief The field's values at the surface's nodes, in StudyExecSpace's memory.
StudyVector make_surface_field(const SurfaceField field, const DoubleLayerSurface& surface) {
  std::vector<double> values(3 * surface.num_nodes);
  for (size_t s = 0; s < surface.num_nodes; ++s) {
    const Vector3d value =
        surface_field_value(field, surface.points[3 * s], surface.points[3 * s + 1], surface.points[3 * s + 2]);
    for (size_t d = 0; d < 3; ++d) {
      values[3 * s + d] = value[d];
    }
  }
  return to_device<StudyExecSpace>(values);
}

/// \brief op q on the surface.
StudyVector apply_double_layer_operator(const DoubleLayerOperator op, const double viscosity,
                                        const DoubleLayerSurface& surface, const StudyVector& q) {
  const StudyExecSpace space;
  const size_t n = surface.num_nodes;
  const bool outward_normal = false;
  StudyVector u("u", 3 * n);
  auto apply_dense = [&](const auto& fill) {
    StudyMatrix A("A", 3 * n, 3 * n);
    fill(A);
    KokkosBlas::gemv(space, "N", 1.0, A, q, 0.0, u);
  };
  auto fill_t = [&](const StudyMatrix& A) {
    fill_stokes_double_layer_matrix(space, viscosity, n, n, surface.d_points, surface.d_points, surface.d_normals,
                                    surface.d_weights, A);
  };
  switch (op) {
    case DoubleLayerOperator::TDense:
      apply_dense(fill_t);
      break;
    case DoubleLayerOperator::TMatrixFree:
      apply_stokes_double_layer_kernel(space, viscosity, n, n, surface.d_points, surface.d_points, surface.d_normals,
                                       surface.d_weights, q, u);
      break;
    case DoubleLayerOperator::InteriorTraceDense:
      apply_dense([&](const StudyMatrix& A) {
        fill_t(A);
        add_singularity_subtraction(space, viscosity, A, outward_normal);
      });
      break;
    case DoubleLayerOperator::ExteriorTraceMatrixFree:
      apply_stokes_double_layer_kernel_ss(space, viscosity, n, surface.d_points, surface.d_normals, surface.d_weights,
                                          q, u);
      break;
    case DoubleLayerOperator::SkfieDense:
      apply_dense([&](const StudyMatrix& A) {
        fill_skfie_matrix(space, viscosity, n, surface.d_points, surface.d_normals, surface.d_weights, A,
                          outward_normal);
      });
      break;
    case DoubleLayerOperator::SkfieMatrixFree:
      apply_skfie(space, viscosity, n, surface.d_points, surface.d_normals, surface.d_weights, q, u, outward_normal);
      break;
  }
  return u;
}

/// \brief op q without forming a matrix: a dense operator's matrix-free twin, which equals it to roundoff.
///
/// The interior trace's twin is the exterior trace plus its analytic target term (sigma / viscosity) q, sigma = -1 for
/// inward normals, as the SkfieDenseMatchesMatrixFree unit test asserts.
StudyVector apply_matrix_free_twin(const DoubleLayerOperator op, const double viscosity,
                                   const DoubleLayerSurface& surface, const StudyVector& q) {
  switch (op) {
    case DoubleLayerOperator::TDense:
      return apply_double_layer_operator(DoubleLayerOperator::TMatrixFree, viscosity, surface, q);
    case DoubleLayerOperator::InteriorTraceDense: {
      StudyVector u =
          apply_double_layer_operator(DoubleLayerOperator::ExteriorTraceMatrixFree, viscosity, surface, q);
      KokkosBlas::axpy(StudyExecSpace{}, -1.0 / viscosity, q, u);
      return u;
    }
    case DoubleLayerOperator::SkfieDense:
      return apply_double_layer_operator(DoubleLayerOperator::SkfieMatrixFree, viscosity, surface, q);
    default:
      return apply_double_layer_operator(op, viscosity, surface, q);
  }
}

/// \brief The exact c with op 1 = c 1 for a constant density on a sphere with inward normals (sigma = -1).
///
/// The constant makes the subtracted integrand vanish: the interior trace and M return their target term
/// (sigma / viscosity) 1 (N 1 = 0 because sum_s w_s n_s = 0), and the exterior trace returns 0. The punctured sum T
/// only approximates PV T[1] = sigma / (2 viscosity) 1, since its kernel is O(1/r) at the omitted node.
double constant_density_coefficient(const DoubleLayerOperator op, const double viscosity) {
  const double sigma = -1.0;
  switch (op) {
    case DoubleLayerOperator::TDense:
    case DoubleLayerOperator::TMatrixFree:
      return sigma / (2.0 * viscosity);
    case DoubleLayerOperator::ExteriorTraceMatrixFree:
      return 0.0;
    default:
      return sigma / viscosity;
  }
}

/// \brief sqrt(sum_s w_s |u_s|^2), the surface L2 norm of a 3-vector field by the quadrature.
double surface_l2_norm(const StudyVector& u, const std::vector<double>& weights) {
  const auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  double sum = 0.0;
  for (size_t s = 0; s < weights.size(); ++s) {
    for (size_t d = 0; d < 3; ++d) {
      sum += weights[s] * h_u(3 * s + d) * h_u(3 * s + d);
    }
  }
  return std::sqrt(sum);
}

/// \brief The root-mean-square entry of u - c 1.
double rms_difference(const StudyVector& u, const double c) {
  const auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u);
  double sum = 0.0;
  for (size_t i = 0; i < h_u.extent(0); ++i) {
    sum += (h_u(i) - c) * (h_u(i) - c);
  }
  return std::sqrt(sum / static_cast<double>(h_u.extent(0)));
}

/// The double_layer study: each double-layer operator applied to each field on a sphere of radius 12.34 with inward
/// normals, at viscosity 0.5305, over quadrature orders 2 to 22. It writes two kinds of rows:
///   - Reference "analytic", for the constant field: the RMS error against constant_density_coefficient, with that
///     coefficient's magnitude as the reference norm. The singularity-subtracted operators are exact up to roundoff
///     (and asserted so by the unit tests); the bare T converges slowly, since the constant excites its singular
///     diagonal.
///   - Reference "order_64", for every field: a self-convergence, | ||op q|| - ||op q||_64 | in the surface L2 norm,
///     with the order-64 norm as the reference norm. The quadrature nodes of different orders do not nest, so only the
///     norms are compared. The reference is computed matrix-free (by apply_matrix_free_twin), which spares the dense
///     operators' 25350 x 25350 matrices at order 64. The normal field spans the interior trace's null space (the one
///     N lifts), so the interior trace of it is 0 up to discretization error; its reference norm is that error, and
///     only its absolute Error is meaningful. Likewise, the exterior trace of the constant field is 0.
void double_layer_study(std::ostream& csv) {
  const double radius = 12.34;
  const double viscosity = 0.5305;
  const int reference_order = 64;
  const std::vector<int> orders = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};

  // The surface L2 norm of each operator applied to each field at the reference order.
  const DoubleLayerSurface reference_surface = make_double_layer_surface(reference_order, radius);
  std::array<std::array<double, kDoubleLayerOperators.size()>, kSurfaceFields.size()> reference_norms;
  for (size_t f = 0; f < kSurfaceFields.size(); ++f) {
    const StudyVector q = make_surface_field(kSurfaceFields[f], reference_surface);
    for (size_t o = 0; o < kDoubleLayerOperators.size(); ++o) {
      reference_norms[f][o] = surface_l2_norm(
          apply_matrix_free_twin(kDoubleLayerOperators[o], viscosity, reference_surface, q), reference_surface.weights);
    }
  }

  csv << std::setprecision(12) << "Operator,Field,Reference,Order,NumNodes,Error,ReferenceNorm\n";
  for (const int order : orders) {
    const DoubleLayerSurface surface = make_double_layer_surface(order, radius);
    std::cout << "double_layer: order " << order << " (" << surface.num_nodes << " nodes)" << std::endl;
    for (size_t f = 0; f < kSurfaceFields.size(); ++f) {
      const StudyVector q = make_surface_field(kSurfaceFields[f], surface);
      for (size_t o = 0; o < kDoubleLayerOperators.size(); ++o) {
        const DoubleLayerOperator op = kDoubleLayerOperators[o];
        const StudyVector u = apply_double_layer_operator(op, viscosity, surface, q);
        const std::string row_key = std::string(to_string(op)) + ',' + to_string(kSurfaceFields[f]);
        if (kSurfaceFields[f] == SurfaceField::Constant) {
          const double c = constant_density_coefficient(op, viscosity);
          csv << row_key << ",analytic," << order << ',' << surface.num_nodes << ',' << rms_difference(u, c) << ','
              << std::abs(c) << '\n';
        }
        csv << row_key << ",order_" << reference_order << ',' << order << ',' << surface.num_nodes << ','
            << std::abs(surface_l2_norm(u, surface.weights) - reference_norms[f][o]) << ',' << reference_norms[f][o]
            << '\n';
      }
    }
  }
}

// ====================================================================================================================
// Overlapping spheres: the pair mobility of the sphere kernels as two spheres pass through each other.
// ====================================================================================================================

/// The overlapping_pair study: Fig. 1 of Zuk, Wajnryb, Mizerski, and Szymczak, Rotne-Prager-Yamakawa approximation for
/// different-sized particles in application to macromolecular bead models, J. Fluid Mech. 741, R5 (2014).
///
/// Spheres of radii a1 = 1 and a2 = a1 / 2 start coincident and move apart along a random unit direction r_hat. A unit
/// force on sphere 2 along r_hat, or along a unit vector perpendicular to it, moves sphere 1 along the same direction;
/// the parallel and perpendicular coefficients are that velocity's component for the RPY, RPYC, and Stokes kernels.
/// As in the paper, the coefficients are in units of 1 / (6 pi viscosity (a1 + a2)) and the distance is
/// d / (a1 + a2). The kernels exclude a sphere's self-interaction, so sphere 1's velocity is the pair term alone.
void overlapping_pair_study(std::ostream& csv) {
  const double viscosity = 0.1;
  const double a1 = 1.0;
  const double a2 = a1 / 2.0;
  const double dr = 0.001;
  const size_t num_distances = 2000;
  const double coefficient_unit = 6.0 * Kokkos::numbers::pi_v<double> * viscosity * (a1 + a2);

  // A random position for sphere 1 and a random direction, from a seeded stream.
  openrand::Philox rng = make_philox(1234, 0);
  const Vector3d x1(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
  Vector3d r_hat(rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0));
  r_hat = r_hat / norm(r_hat);
  Vector3d r_perp(r_hat[1], -r_hat[0], 0.0);  // perpendicular to r_hat, which is never along z here
  r_perp = r_perp / norm(r_perp);

  const StudyExecSpace space;
  const StudyVector radii = to_device<StudyExecSpace>({a1, a2});
  csv << std::setprecision(12)
      << "Distance,ParallelRPY,PerpendicularRPY,ParallelRPYC,PerpendicularRPYC,ParallelStokes,PerpendicularStokes\n";
  for (size_t p = 0; p < num_distances; ++p) {
    const Vector3d x2 = x1 + (dr * static_cast<double>(p)) * r_hat;
    const StudyVector positions = to_device<StudyExecSpace>({x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]});

    // The coefficients (RPY, RPYC, Stokes) along one direction.
    auto coefficients = [&](const Vector3d& direction) {
      const StudyVector forces =
          to_device<StudyExecSpace>({0.0, 0.0, 0.0, direction[0], direction[1], direction[2]});
      std::array<StudyVector, 3> velocities = {StudyVector("v_rpy", 6), StudyVector("v_rpyc", 6),
                                               StudyVector("v_stokes", 6)};
      apply_rpy_kernel(space, viscosity, positions, positions, radii, radii, forces, velocities[0]);
      apply_rpyc_kernel(space, viscosity, positions, positions, radii, radii, forces, velocities[1]);
      apply_stokes_kernel(space, viscosity, positions, positions, forces, velocities[2]);
      std::array<double, 3> result;
      for (size_t k = 0; k < 3; ++k) {
        const auto v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, velocities[k]);
        result[k] = coefficient_unit * (v(0) * direction[0] + v(1) * direction[1] + v(2) * direction[2]);
      }
      return result;
    };
    const std::array<double, 3> parallel = coefficients(r_hat);
    const std::array<double, 3> perpendicular = coefficients(r_perp);
    csv << dr * static_cast<double>(p) / (a1 + a2) << ',' << parallel[0] << ',' << perpendicular[0] << ','
        << parallel[1] << ',' << perpendicular[1] << ',' << parallel[2] << ',' << perpendicular[2] << '\n';
  }
}

// ====================================================================================================================
// The studies PeripheryStudies runs, each writing periphery_studies_<name>.csv.
// ====================================================================================================================

// Quadrature-grid orientation samples per inclusion config (sample 0 is the unrotated grid).
constexpr int kNumOrientationSamples = 16;

/// \brief cavity: a sphere, point or resolved, at the center of a spherical cavity, against Happel & Brenner.
void run_cavity_study() {
  const std::vector<std::pair<double, double>> radii = {{1.0, 1.0}, {0.95, 1.0}, {0.99, 1.0}, {1.0, 2.0}, {1.0, 2.5},
                                                        {1.0, 4.0}, {1.0, 5.0},  {1.0, 10.0}, {2.0, 10.0}};
  const std::vector<int> body_orders = {4, 8, 12, 16, 20, 24};
  const std::vector<int> periphery_orders = {4, 6, 8, 10, 12, 16, 20, 24};

  std::ofstream csv("periphery_studies_cavity.csv");
  csv << "Study,Type,BodyRadius,CavityRadius,Lambda,BodyOrder,PeripheryOrder,NumPeripheryNodes,ValueUnbounded,"
         "ValueConfined,ValueExact,RelErrUnbounded,RelErrConfined\n";
  for (const auto& [a, b] : radii) {
    motile_body_in_spherical_cavity_drag(csv, SphereType::RPY, a, b, body_orders, periphery_orders);
    motile_body_in_spherical_cavity_drag(csv, SphereType::Inclusion, a, b, body_orders, periphery_orders);
    motile_body_in_spherical_cavity_rotation(csv, SphereType::Inclusion, a, b, body_orders, periphery_orders);
  }
}

/// \brief two_sphere: the symmetry, positive definiteness, and accuracy of the 6x6 mobility of two spheres.
void run_two_sphere_study() {
  const std::vector<int> body_orders = {8, 12, 16, 20, 24 /*, 28, 32 */};
  // Uncorrected RPY loses SPD near s ~ 1.137 (perpendicular relative mode, 3/(4s) + 1/(2s^3) = 1).
  const std::vector<double> svals = {1.0, 1.13, 1.14, 1.2, 1.5, 2.0, 2.0001, 2.001, 2.01, 2.05,
                                     2.1, 2.15, 2.2,  2.3, 2.4, 2.5, 3.0,    4.0,   6.0};
  const double r = 12.34;
  std::ofstream csv("periphery_studies_two_sphere.csv");
  csv << std::setprecision(12);
  csv << "Order,S,R,Config,Sample,NumIters";
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      csv << ",M" << i << j;
    }
  }
  csv << "\n";
  for (const auto& order : body_orders) {
    for (const auto& sval : svals) {
      two_sphere_spd(csv, order, sval, r, kNumOrientationSamples);
    }
  }
}

/// \brief wilson: the accuracy of three spheres against Wilson (2013).
void run_wilson_study() {
  const std::vector<int> body_orders = {8, 12, 16, 20, 24 /*, 28, 32 */};
  std::ofstream csv("periphery_studies_wilson.csv");
  csv << std::setprecision(12) << "Order,S,Config,Sample,U1,U2,U3\n";
  for (const auto& order : body_orders) {
    wilson_three_sphere(csv, order, kNumOrientationSamples);
  }
}

/// \brief three_sphere: the symmetry, positive definiteness, and accuracy of the 9x9 mobility of the Wilson triangle.
void run_three_sphere_study() {
  const std::vector<int> body_orders = {8, 12, 16, 20, 24 /*, 28, 32 */};

  // Wilson's s values are included for the accuracy check. Uncorrected RPY loses SPD near s ~ 1.218 for the triangle.
  const std::vector<double> svals = {1.0, 1.2,  1.25, 1.5, 2.0, 2.0001, 2.001, 2.01, 2.05,
                                     2.1, 2.15, 2.2,  2.3, 2.4, 2.5,    3.0,   4.0,  6.0};
  const double r = 12.34;
  std::ofstream csv("periphery_studies_three_sphere.csv");
  csv << std::setprecision(12);
  csv << "Order,S,R,Config,Sample,NumIters";
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      csv << ",M" << i << j;
    }
  }
  csv << "\n";
  for (const auto& order : body_orders) {
    for (const auto& sval : svals) {
      three_sphere_spd(csv, order, sval, r, kNumOrientationSamples);
    }
  }
}

/// \brief periphery_sphere: the symmetry and positive definiteness of the 3x3 mobility of a sphere near the periphery.
///
/// A sphere of radius 0.1 inside a periphery of radius 13.5 (the radius of HP1's R135 quadrature files); HP1's default
/// periphery order is 32. s = center-to-wall distance / r: s = 1 touches the wall, and s = 135 is the center.
void run_periphery_sphere_study() {
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  const std::vector<int> periphery_orders = {8, 12, 16, 20, 24, 32};
  const std::vector<int> body_orders = {8, 12, 16};
  const std::vector<double> svals = {1.0,  1.01, 1.1,  1.25, 1.5,  2.0,  2.5,  3.0,  4.0,  5.0,   6.0,  8.0,
                                     10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 135.0};
  const double r = 0.1;
  const double periphery_radius = 13.5;
  std::ofstream csv("periphery_studies_periphery_sphere.csv");
  csv << std::setprecision(12);
  csv << "Order,PeripheryOrder,NumPeripheryNodes,PeripheryRadius,S,R,Config,Sample,NumIters";
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      csv << ",M" << i << j;
    }
  }
  csv << "\n";
  for (const auto& periphery_order : periphery_orders) {
    const auto periphery = make_cavity_periphery<ExecSpace>(periphery_order, periphery_radius, 1.0);
    for (const auto& order : body_orders) {
      for (const auto& sval : svals) {
        periphery_sphere_spd(csv, order, periphery, periphery_order, periphery_radius, sval, r,
                             kNumOrientationSamples);
      }
    }
  }
}

/// \brief double_layer: the double-layer operators' accuracy on a sphere (see double_layer_study).
void run_double_layer_study() {
  std::ofstream csv("periphery_studies_double_layer.csv");
  double_layer_study(csv);
}

/// \brief overlapping_pair: the pair mobility of two overlapping spheres (see overlapping_pair_study).
void run_overlapping_pair_study() {
  std::ofstream csv("periphery_studies_overlapping_pair.csv");
  overlapping_pair_study(csv);
}

/// \brief A study PeripheryStudies can run: its command-line name, the function that runs it, and a summary.
struct Study {
  const char* name;
  void (*run)();
  const char* summary;
};

constexpr std::array<Study, 7> kStudies = {{
    {"cavity", run_cavity_study, "a sphere at the center of a spherical cavity, against Happel & Brenner"},
    {"two_sphere", run_two_sphere_study, "the 6x6 mobility of two spheres: symmetry, SPD, and accuracy"},
    {"wilson", run_wilson_study, "three spheres against Wilson (2013)"},
    {"three_sphere", run_three_sphere_study, "the 9x9 mobility of the Wilson triangle: symmetry, SPD, and accuracy"},
    {"periphery_sphere", run_periphery_sphere_study, "the 3x3 mobility of a sphere near the periphery: symmetry, SPD"},
    {"double_layer", run_double_layer_study, "the double-layer operators' accuracy on a sphere"},
    {"overlapping_pair", run_overlapping_pair_study, "the pair mobility of overlapping spheres (Zuk et al. 2014)"},
}};

/// \brief The usage message, listing every study.
void print_usage(std::ostream& os) {
  os << "Usage: PeripheryStudies <study> [<study> ...]\n"
     << "Each study writes periphery_studies_<study>.csv to the working directory. The studies:\n";
  for (const Study& study : kStudies) {
    os << "  " << std::left << std::setw(18) << study.name << study.summary << "\n";
  }
}

}  // namespace

}  // namespace mbody

}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  using mundy::mbody::kStudies;
  std::vector<const mundy::mbody::Study*> selected;
  bool valid = argc > 1;
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    const auto study =
        std::find_if(kStudies.begin(), kStudies.end(), [&](const mundy::mbody::Study& s) { return name == s.name; });
    if (study == kStudies.end()) {
      std::cerr << "PeripheryStudies: unknown study '" << name << "'\n";
      valid = false;
    } else {
      selected.push_back(&*study);
    }
  }

  if (valid) {
    Kokkos::print_configuration(std::cout);
    for (const mundy::mbody::Study* study : selected) {
      std::cout << "PeripheryStudies: running " << study->name << std::endl;
      study->run();
    }
  } else {
    mundy::mbody::print_usage(std::cerr);
  }

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return valid ? 0 : 1;
}
