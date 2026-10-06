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

// External libs
#include <openrand/philox.h>

// C++ core
#include <algorithm>         // for std::find, std::min, std::max
#include <array>             // for std::array
#include <chrono>            // for std::chrono::steady_clock
#include <fstream>           // for std::ofstream
#include <initializer_list>  // for std::initializer_list
#include <iomanip>           // for std::setw, std::setprecision
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
#include <mundy_geom/randomize.hpp>   // for mundy::generate_random_unit_quaternion
#include <mundy_math/Vector3.hpp>     // for Vector3
#include <mundy_mbody/Periphery.hpp>  // for gen_sphere_quadrature, fill_skfie_matrix, apply_skfie, PeripheryT, ...
#include <mundy_utils/rng.hpp>        // for mundy::make_philox

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
// solved as one block matrix-free GMRES system. Shared helpers (make_rpy_body, solve_cavity, sphere_mix,
// wilson_three_sphere) followed by their asserting tests.
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
  std::vector<double> p_pts;
  std::vector<double> p_wts;
  std::vector<double> p_nrm;
  gen_sphere_quadrature(order, cavity_radius, &p_pts, &p_wts, &p_nrm, /*include_poles=*/false,
                        /*outward_normal=*/false);
  auto periphery = std::make_shared<PeripheryT<ExecSpace>>(p_wts.size(), mu);
  periphery->set_surface_positions(p_pts.data())
      .set_quadrature_weights(p_wts.data())
      .set_surface_normals(p_nrm.data(), /*outward_normal=*/false);
  periphery->build_inverse_self_interaction_matrix(/*write_to_file=*/false);
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

// The two extreme approach directions against a gen_sphere_quadrature periphery grid: straight at the node nearest +x,
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
// touches it). Same measurement as PeripheryDiagnostic.SphereQuadPeripheryRPYC (F.V < 0 along a few directions), but
// as the full 3x3 mobility. The approach direction depends on the sample: 0 = straight at a periphery node, 1 = at a
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

}  // namespace

}  // namespace mbody

}  // namespace mundy

int main(int argc, char** argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  bool do_body_cavity = false;
  bool do_two_sphere = false;
  bool do_wilson = false;
  bool do_three_sphere = false;
  bool do_periphery_sphere = true;

  // Quadrature-grid orientation samples per inclusion config (sample 0 is the unrotated grid).
  const int num_orientation_samples = 16;

  if (do_body_cavity) {
    // Comparison of accuracy of a mobility solve inside of a cavity
    using ::mundy::mbody::SphereType;
    const std::vector<std::pair<double, double>> radii = {{1.0, 1.0}, {0.95, 1.0}, {0.99, 1.0}, {1.0, 2.0}, {1.0, 2.5},
                                                          {1.0, 4.0}, {1.0, 5.0},  {1.0, 10.0}, {2.0, 10.0}};
    const std::vector<int> body_orders = {4, 8, 12, 16, 20, 24};
    const std::vector<int> periphery_orders = {4, 6, 8, 10, 12, 16, 20, 24};

    std::ofstream csv("hydro_studies_cavity.csv");
    csv << "Study,Type,BodyRadius,CavityRadius,Lambda,BodyOrder,PeripheryOrder,NumPeripheryNodes,ValueUnbounded,"
           "ValueConfined,ValueExact,RelErrUnbounded,RelErrConfined\n";
    for (const auto& [a, b] : radii) {
      ::mundy::mbody::motile_body_in_spherical_cavity_drag(csv, SphereType::RPY, a, b, body_orders, periphery_orders);
      ::mundy::mbody::motile_body_in_spherical_cavity_drag(csv, SphereType::Inclusion, a, b, body_orders,
                                                           periphery_orders);
      ::mundy::mbody::motile_body_in_spherical_cavity_rotation(csv, SphereType::Inclusion, a, b, body_orders,
                                                               periphery_orders);
    }
  }

  if (do_two_sphere) {
    // Symmetric positive definiteness and accuracy of two sphere comparisons
    const std::vector<int> body_orders = {8, 12, 16, 20, 24 /*, 28, 32 */};
    // Uncorrected RPY loses SPD near s ~ 1.137 (perpendicular relative mode, 3/(4s) + 1/(2s^3) = 1).
    const std::vector<double> svals = {1.0, 1.13, 1.14, 1.2, 1.5, 2.0, 2.0001, 2.001, 2.01, 2.05,
                                       2.1, 2.15, 2.2,  2.3, 2.4, 2.5, 3.0,    4.0,   6.0};
    const double r = 12.34;
    std::ofstream csv("hydro_studies_twosphere.csv");
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
        ::mundy::mbody::two_sphere_spd(csv, order, sval, r, num_orientation_samples);
      }
    }
  }

  if (do_wilson) {
    // Accuracy of Wilson's three spheres
    const std::vector<int> body_orders = {8, 12, 16, 20, 24 /*, 28, 32 */};
    std::ofstream wilson_csv("hydro_studies_wilson.csv");
    wilson_csv << std::setprecision(12) << "Order,S,Config,Sample,U1,U2,U3\n";
    for (const auto& order : body_orders) {
      ::mundy::mbody::wilson_three_sphere(wilson_csv, order, num_orientation_samples);
    }
  }

  if (do_three_sphere) {
    // Symmetric positive definiteness and accuracy of two/three sphere comparisons
    const std::vector<int> body_orders = {8, 12, 16, 20, 24 /*, 28, 32 */};

    // Full 9x9 mobility of the Wilson triangle, including Wilson's s values for the accuracy check. Uncorrected RPY
    // loses SPD near s ~ 1.218 for the triangle.
    const std::vector<double> svals = {1.0, 1.2,  1.25, 1.5, 2.0, 2.0001, 2.001, 2.01, 2.05,
                                       2.1, 2.15, 2.2,  2.3, 2.4, 2.5,    3.0,   4.0,  6.0};
    const double r = 12.34;
    std::ofstream csv("hydro_studies_threesphere.csv");
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
        ::mundy::mbody::three_sphere_spd(csv, order, sval, r, num_orientation_samples);
      }
    }
  }

  if (do_periphery_sphere) {
    // Symmetric positive definiteness of one sphere approaching the periphery. Geometry of
    // PeripheryDiagnostic.SphereQuadPeripheryRPYC (sphere radius 0.1, periphery radius 13.5 = HP1's R135 files);
    // HP1's default periphery order is 32. s = center-to-wall distance / r, s = 1 touches the wall, s = 135 is the
    // center.
    using ExecSpace = Tpetra::Map<>::node_type::execution_space;
    const std::vector<int> periphery_orders = {8, 12, 16, 20, 24, 32};
    const std::vector<int> body_orders = {8, 12, 16};
    const std::vector<double> svals = {1.0,  1.01, 1.1,  1.25, 1.5,  2.0,  2.5,  3.0,  4.0,  5.0,   6.0,  8.0,
                                       10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 135.0};
    const double r = 0.1;
    const double periphery_radius = 13.5;
    std::ofstream csv("hydro_studies_periphery.csv");
    csv << std::setprecision(12);
    csv << "Order,PeripheryOrder,NumPeripheryNodes,PeripheryRadius,S,R,Config,Sample,NumIters";
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        csv << ",M" << i << j;
      }
    }
    csv << "\n";
    for (const auto& periphery_order : periphery_orders) {
      const auto periphery = ::mundy::mbody::make_cavity_periphery<ExecSpace>(periphery_order, periphery_radius, 1.0);
      for (const auto& order : body_orders) {
        for (const auto& sval : svals) {
          ::mundy::mbody::periphery_sphere_spd(csv, order, periphery, periphery_order, periphery_radius, sval, r,
                                               num_orientation_samples);
        }
      }
    }
  }

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}