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
/// \brief Unit tests for the periphery: its setters, its second-kind operator, and the bodies and spheres inside it.

// External
#include <gtest/gtest.h>      // for TEST, EXPECT_NEAR, SCOPED_TRACE, testing::PrintToString, etc
#include <openrand/philox.h>  // for openrand::Philox

#include <KokkosBlas.hpp>        // for KokkosBlas::gemv, KokkosBlas::gemm, KokkosBlas::axpy
#include <Kokkos_Core.hpp>       // for Kokkos::View, Kokkos::create_mirror_view_and_copy, Kokkos::numbers::pi_v
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_BELOS, HAVE_MUNDYMATH_TPETRA

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
#include <Tpetra_Map.hpp>  // for Tpetra::Map<>::node_type
#endif

// C++ core
#include <algorithm>  // for std::max
#include <array>      // for std::array
#include <cmath>      // for std::abs, std::sqrt, std::log, std::sin, std::cos
#include <cstdio>     // for std::remove
#include <limits>     // for std::numeric_limits
#include <memory>     // for std::make_shared
#include <numeric>    // for std::accumulate
#include <stdexcept>  // for std::invalid_argument, std::runtime_error
#include <string>     // for std::string
#include <utility>    // for std::pair
#include <vector>     // for std::vector

// Mundy
#include <mundy_math/GaussLegendreSphere.hpp>  // for mundy::gauss_legendre_sphere_rule
#include <mundy_math/Matrix3.hpp>              // for mundy::Matrix3d, mundy::transpose, mundy::frobenius_norm
#include <mundy_math/Quaternion.hpp>           // for mundy::Quaterniond
#include <mundy_math/Vector3.hpp>              // for mundy::Vector3d, mundy::norm
#include <mundy_math/invert.hpp>               // for mundy::invert
#include <mundy_math/matrix_market.hpp>        // for mundy::write_matrix_market
#include <mundy_mbody/Periphery.hpp>           // for mundy::mbody::PeripheryT, fill_skfie_matrix, apply_skfie, ...
#include <mundy_utils/rng.hpp>                 // for mundy::make_philox
#include <mundy_utils/throw_assert.hpp>        // for MUNDY_THROW_REQUIRE

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
#include <mundy_math/belos_solver.hpp>  // for mundy::BelosConfig, mundy::BelosResult, mundy::BelosSolver
#endif

namespace mundy {

namespace mbody {

namespace {

//! \name Device staging
//@{

// Library calls run on TestExecSpace; inputs are built and results checked on the host.
using TestExecSpace = Kokkos::DefaultExecutionSpace;
using TestMemSpace = TestExecSpace::memory_space;
using TestPeriphery = PeripheryT<TestExecSpace>;
using DeviceVector = Kokkos::View<double*, Kokkos::LayoutLeft, TestMemSpace>;
using DeviceMatrix = Kokkos::View<double**, Kokkos::LayoutLeft, TestMemSpace>;

/// \brief A host copy of a vector or matrix in any memory space.
template <class View>
auto to_host(const View& view) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
}

//@}

//! \name Constants
//@{

constexpr double kViscosity = 0.5305;
constexpr std::array<double, 2> kViscosities = {1.0, kViscosity};
constexpr double kPeripheryRadius = 13.5;  //!< The radius of the periphery the operator tests use
constexpr double kPi = Kokkos::numbers::pi_v<double>;

//@}

//! \name Sphere surfaces and peripheries
//@{

/// \brief A Gauss-Legendre sphere rule in ExecSpace's memory, with its normals of one orientation.
template <class ExecSpace = TestExecSpace>
struct SphereSurface {
  using vector_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  size_t num_nodes;     //!< N
  vector_t points;      //!< 3 N node positions
  vector_t weights;     //!< N quadrature weights, which sum to the area
  vector_t normals;     //!< 3 N unit normals
  bool outward_normal;  //!< Whether the normals point out of (true) or into (false) the sphere
};

/// \brief The (order + 1)-ring Gauss-Legendre rule on the sphere of the given radius about the origin.
template <class ExecSpace = TestExecSpace>
SphereSurface<ExecSpace> make_sphere_surface(const int order, const double radius, const bool outward_normal) {
  std::vector<double> normals;  // the unit-sphere points, which are its outward unit normals
  std::vector<double> weights;
  gauss_legendre_sphere_rule(order + 1, normals, weights);
  std::vector<double> points(normals.size());
  for (size_t i = 0; i < normals.size(); ++i) {
    points[i] = radius * normals[i];
    normals[i] *= outward_normal ? 1.0 : -1.0;
  }
  for (double& weight : weights) {
    weight *= radius * radius;
  }
  return {weights.size(), to_device<ExecSpace>(points), to_device<ExecSpace>(weights), to_device<ExecSpace>(normals),
          outward_normal};
}

/// \brief A periphery on surface with its dense inverse built.
template <class ExecSpace>
PeripheryT<ExecSpace> make_periphery(const SphereSurface<ExecSpace>& surface, const double viscosity) {
  PeripheryT<ExecSpace> periphery(surface.num_nodes, viscosity);
  periphery.set_surface_positions(surface.points)
      .set_surface_normals(surface.normals, surface.outward_normal)
      .set_quadrature_weights(surface.weights)
      .build_inverse_self_interaction_matrix();
  return periphery;
}

/// \brief n values drawn uniformly from [-1, 1) by the Philox stream of the given seed.
std::vector<double> make_random_values(const size_t n, const size_t seed) {
  openrand::Philox rng = make_philox(seed, 0);
  std::vector<double> values(n);
  for (double& value : values) {
    value = rng.uniform(-1.0, 1.0);
  }
  return values;
}

//@}

//! \name Comparisons
//@{

/// \brief max_i |a_i - b_i| / max_i |a_i|.
template <class ViewA, class ViewB>
double max_relative_difference(const ViewA& a_view, const ViewB& b_view) {
  const auto a = to_host(a_view);
  const auto b = to_host(b_view);
  MUNDY_THROW_REQUIRE(a.extent(0) == b.extent(0), std::invalid_argument, "max_relative_difference: length mismatch.");
  double max_difference = 0.0;
  double max_magnitude = 0.0;
  for (size_t i = 0; i < a.extent(0); ++i) {
    max_difference = std::max(max_difference, std::abs(a(i) - b(i)));
    max_magnitude = std::max(max_magnitude, std::abs(a(i)));
  }
  return max_difference / max_magnitude;
}

/// \brief Whether each error falls below the one before it, unless either has fallen below floor.
bool decreases_or_reaches(const std::vector<double>& errors, const double floor) {
  for (size_t i = 1; i < errors.size(); ++i) {
    const bool at_floor = errors[i - 1] < floor || errors[i] < floor;
    if (!(errors[i] < errors[i - 1] || at_floor)) {
      return false;
    }
  }
  return true;
}

//@}

//! \name The periphery's setters
//@{

/// \brief How many entries of two equally sized matrices differ in value.
template <class MatrixA, class MatrixB>
size_t count_value_differences(const MatrixA& a_view, const MatrixB& b_view) {
  const auto a = to_host(a_view);
  const auto b = to_host(b_view);
  MUNDY_THROW_REQUIRE(a.extent(0) == b.extent(0) && a.extent(1) == b.extent(1), std::invalid_argument,
                      "count_value_differences: size mismatch.");
  size_t differences = 0;
  for (size_t i = 0; i < a.extent(0); ++i) {
    for (size_t j = 0; j < a.extent(1); ++j) {
      differences += (a(i, j) != b(i, j)) ? 1 : 0;
    }
  }
  return differences;
}

// The file setters are exact: files reproduce the view setters' inverse bit for bit, and a file of the wrong length
// throws.
TEST(Periphery, FileSettersMatchViewSetters) {
  const auto surface = make_sphere_surface(6, 1.7, /*outward_normal=*/false);
  const std::string prefix = "FileSettersMatchViewSetters_";
  write_matrix_market(prefix + "points.mtx", surface.points);
  write_matrix_market(prefix + "normals.mtx", surface.normals);
  write_matrix_market(prefix + "weights.mtx", surface.weights);

  const TestPeriphery from_views = make_periphery(surface, kViscosity);
  TestPeriphery from_files(surface.num_nodes, kViscosity);
  from_files.set_surface_positions(prefix + "points.mtx")
      .set_surface_normals(prefix + "normals.mtx", surface.outward_normal)
      .set_quadrature_weights(prefix + "weights.mtx")
      .build_inverse_self_interaction_matrix();
  from_views.write_inverse_self_interaction_matrix(prefix + "M_inv.mtx");
  TestPeriphery from_inverse_file(surface.num_nodes, kViscosity);
  from_inverse_file.set_inverse_self_interaction_matrix(prefix + "M_inv.mtx");

  EXPECT_EQ(count_value_differences(from_files.get_M_inv(), from_views.get_M_inv()), 0u)
      << "entries of M^{-1} that differ when the geometry is read from files";
  EXPECT_EQ(count_value_differences(from_inverse_file.get_M_inv(), from_views.get_M_inv()), 0u)
      << "entries of M^{-1} that differ after a write and read";

  TestPeriphery larger(surface.num_nodes + 1, kViscosity);
  EXPECT_THROW(larger.set_surface_positions(prefix + "points.mtx"), std::runtime_error);
  EXPECT_THROW(larger.set_inverse_self_interaction_matrix(prefix + "M_inv.mtx"), std::runtime_error);
  for (const char* name : {"points.mtx", "normals.mtx", "weights.mtx", "M_inv.mtx"}) {
    std::remove((prefix + name).c_str());
  }
}

// The flat-pointer setter reads a row-major 3N x 3N matrix.
TEST(Periphery, FlatPointerInverseIsReadRowMajor) {
  const size_t num_nodes = 18;  // an order-2 sphere rule
  const size_t n = 3 * num_nodes;
  std::vector<double> flat(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      flat[i * n + j] = 1000.0 * static_cast<double>(i) + static_cast<double>(j);
    }
  }

  TestPeriphery periphery(num_nodes, kViscosity);
  periphery.set_inverse_self_interaction_matrix(flat.data());
  const auto M_inv = to_host(periphery.get_M_inv());
  size_t num_mismatches = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      num_mismatches += (M_inv(i, j) != flat[i * n + j]) ? 1 : 0;
    }
  }
  EXPECT_EQ(num_mismatches, 0u) << "M_inv(i, j) must equal M_inv_flat[i * 3N + j]";
}

// compute_surface_density throws before an inverse exists.
TEST(Periphery, SurfaceDensityRequiresAnInverse) {
  const size_t num_nodes = 18;
  TestPeriphery periphery(num_nodes, kViscosity);
  DeviceVector slip("slip", 3 * num_nodes);
  DeviceVector surface_density("surface_density", 3 * num_nodes);
  EXPECT_THROW(periphery.compute_surface_density(slip, surface_density), std::runtime_error);
}

//@}

//! \name The second-kind operator
// These tests check the periphery's second-kind operator M = J + T + N, derived in Periphery.hpp's header.
//@{

// M M^{-1} = I to 1e-10 for both normal orientations: N lifts the null space of J + T, so M is well conditioned.
TEST(Periphery, SkfieTimesItsInverseIsTheIdentity) {
  for (const bool outward_normal : {false, true}) {
    const auto surface = make_sphere_surface(12, 12.34, outward_normal);
    const size_t n = 3 * surface.num_nodes;
    DeviceMatrix M("M", n, n);
    fill_skfie_matrix(TestExecSpace{}, kViscosity, surface.num_nodes, surface.points, surface.normals, surface.weights,
                      M, outward_normal);
    DeviceMatrix factors("factors", n, n);  // invert overwrites its input with its LU factors
    Kokkos::deep_copy(factors, M);
    DeviceMatrix M_inv("M_inv", n, n);
    invert(TestExecSpace{}, factors, M_inv);
    DeviceMatrix product("product", n, n);
    KokkosBlas::gemm(TestExecSpace{}, "N", "N", 1.0, M, M_inv, 0.0, product);

    const auto h_product = to_host(product);
    double max_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        max_error = std::max(max_error, std::abs(h_product(i, j) - (i == j ? 1.0 : 0.0)));
      }
    }
    EXPECT_LE(max_error, 1.0e-10) << "max |M M^{-1} - I|, outward_normal=" << outward_normal;
  }
}

/// \brief How far each operator is from its analytic value on the constant densities e_1, e_2, e_3.
struct ConstantDensityErrors {
  double dense;            //!< max |M c - (sigma / viscosity) c| for fill_skfie_matrix's M
  double matrix_free;      //!< max |M c - (sigma / viscosity) c| for apply_skfie
  double exterior;         //!< max |apply_stokes_double_layer_kernel_ss[c]|, whose exact value is 0
  double principal_value;  //!< The relative error of T c's area-weighted mean against PV T[1] = sigma / (2 viscosity)
};

/// \brief The ConstantDensityErrors of the order-`order` periphery of radius kPeripheryRadius.
ConstantDensityErrors constant_density_errors(const int order, const double viscosity, const bool outward_normal) {
  const auto surface = make_sphere_surface(order, kPeripheryRadius, outward_normal);
  const size_t num_nodes = surface.num_nodes;
  DeviceMatrix M("M", 3 * num_nodes, 3 * num_nodes);
  fill_skfie_matrix(TestExecSpace{}, viscosity, num_nodes, surface.points, surface.normals, surface.weights, M,
                    outward_normal);
  DeviceMatrix T("T", 3 * num_nodes, 3 * num_nodes);
  fill_stokes_double_layer_matrix(TestExecSpace{}, viscosity, num_nodes, num_nodes, surface.points, surface.points,
                                  surface.normals, surface.weights, T);
  const auto weights = to_host(surface.weights);

  const double sigma = outward_normal ? 1.0 : -1.0;
  const double jump = sigma / viscosity;
  ConstantDensityErrors errors{0.0, 0.0, 0.0, 0.0};
  double weighted_pv_diagonal = 0.0;  // sum_k sum_t w_t (T e_k)_{3t+k}
  for (size_t k = 0; k < 3; ++k) {
    std::vector<double> c(3 * num_nodes, 0.0);
    for (size_t t = 0; t < num_nodes; ++t) {
      c[3 * t + k] = 1.0;
    }
    const DeviceVector c_device = to_device(c);
    DeviceVector u_dense("u_dense", 3 * num_nodes);
    DeviceVector u_matrix_free("u_matrix_free", 3 * num_nodes);
    DeviceVector u_exterior("u_exterior", 3 * num_nodes);
    DeviceVector u_pv("u_pv", 3 * num_nodes);
    KokkosBlas::gemv(TestExecSpace{}, "N", 1.0, M, c_device, 0.0, u_dense);
    apply_skfie(TestExecSpace{}, viscosity, num_nodes, surface.points, surface.normals, surface.weights, c_device,
                u_matrix_free, outward_normal);
    apply_stokes_double_layer_kernel_ss(TestExecSpace{}, viscosity, num_nodes, surface.points, surface.normals,
                                        surface.weights, c_device, u_exterior);
    KokkosBlas::gemv(TestExecSpace{}, "N", 1.0, T, c_device, 0.0, u_pv);

    const auto h_dense = to_host(u_dense);
    const auto h_matrix_free = to_host(u_matrix_free);
    const auto h_exterior = to_host(u_exterior);
    const auto h_pv = to_host(u_pv);
    for (size_t i = 0; i < 3 * num_nodes; ++i) {
      errors.dense = std::max(errors.dense, std::abs(h_dense(i) - jump * c[i]));
      errors.matrix_free = std::max(errors.matrix_free, std::abs(h_matrix_free(i) - jump * c[i]));
      errors.exterior = std::max(errors.exterior, std::abs(h_exterior(i)));
    }
    for (size_t t = 0; t < num_nodes; ++t) {
      weighted_pv_diagonal += weights(t) * h_pv(3 * t + k);
    }
  }
  const double area = std::accumulate(weights.data(), weights.data() + num_nodes, 0.0);
  const double pv_expected = sigma / (2.0 * viscosity);
  errors.principal_value = std::abs(weighted_pv_diagonal / (3.0 * area) - pv_expected) / std::abs(pv_expected);
  return errors;
}

// A constant density c makes the subtracted integrand vanish, so M and the exterior trace return their jumps to
// roundoff, with sigma = +1 for outward normals and -1 for inward:
//    M c = (sigma / viscosity) c,    exterior trace of c = 0.
// The bare punctured sum T c only approximates its principal value, (sigma / (2 viscosity)) c, so it is checked only to
// converge toward it and to come within 10%.
TEST(Periphery, SkfieMapsAConstantDensityToItsJump) {
  const std::vector<int> orders = {4, 8, 16};
  for (const bool outward_normal : {false, true}) {
    for (const double viscosity : {1.0, kViscosity, 3.7}) {
      SCOPED_TRACE(testing::Message() << "outward_normal=" << outward_normal << " viscosity=" << viscosity);
      std::vector<double> pv_errors;
      for (const int order : orders) {
        const ConstantDensityErrors errors = constant_density_errors(order, viscosity, outward_normal);
        const double tol = 1.0e-11 / viscosity;
        EXPECT_LT(errors.dense, tol) << "dense (J + T + N)[c] != (sigma / viscosity) c, order=" << order;
        EXPECT_LT(errors.matrix_free, tol) << "apply_skfie[c] != (sigma / viscosity) c, order=" << order;
        EXPECT_LT(errors.exterior, tol) << "apply_stokes_double_layer_kernel_ss[c] != 0, order=" << order;
        pv_errors.push_back(errors.principal_value);
      }
      EXPECT_TRUE(decreases_or_reaches(pv_errors, 0.0))
          << "PV T[1] must converge: orders=" << testing::PrintToString(orders)
          << " errors=" << testing::PrintToString(pv_errors);
      EXPECT_LT(pv_errors.back(), 0.1) << "PV T[1] must approach sigma / (2 viscosity)";
    }
  }
}

// The dense and matrix-free operators agree to roundoff: fill_skfie_matrix with apply_skfie, and the dense interior
// trace with the exterior trace plus its jump:
//    (J + T)_interior q = (J + T)_exterior q + (sigma / viscosity) q.
TEST(Periphery, SkfieDenseMatchesMatrixFree) {
  for (const bool outward_normal : {false, true}) {
    for (const int order : {4, 8, 12}) {
      SCOPED_TRACE(testing::Message() << "outward_normal=" << outward_normal << " order=" << order);
      const auto surface = make_sphere_surface(order, kPeripheryRadius, outward_normal);
      const size_t n = 3 * surface.num_nodes;
      const DeviceVector q = to_device(make_random_values(n, static_cast<size_t>(order)));

      DeviceMatrix M("M", n, n);
      fill_skfie_matrix(TestExecSpace{}, kViscosity, surface.num_nodes, surface.points, surface.normals,
                        surface.weights, M, outward_normal);
      DeviceVector u_dense("u_dense", n);
      DeviceVector u_matrix_free("u_matrix_free", n);
      KokkosBlas::gemv(TestExecSpace{}, "N", 1.0, M, q, 0.0, u_dense);
      apply_skfie(TestExecSpace{}, kViscosity, surface.num_nodes, surface.points, surface.normals, surface.weights, q,
                  u_matrix_free, outward_normal);
      EXPECT_LT(max_relative_difference(u_dense, u_matrix_free), 1.0e-12) << "fill_skfie_matrix vs apply_skfie";

      DeviceMatrix T("T", n, n);
      fill_stokes_double_layer_matrix(TestExecSpace{}, kViscosity, surface.num_nodes, surface.num_nodes, surface.points,
                                      surface.points, surface.normals, surface.weights, T);
      add_singularity_subtraction(TestExecSpace{}, kViscosity, T, outward_normal);
      DeviceVector u_interior("u_interior", n);
      DeviceVector u_exterior("u_exterior", n);
      KokkosBlas::gemv(TestExecSpace{}, "N", 1.0, T, q, 0.0, u_interior);
      apply_stokes_double_layer_kernel_ss(TestExecSpace{}, kViscosity, surface.num_nodes, surface.points,
                                          surface.normals, surface.weights, q, u_exterior);
      const double sigma = outward_normal ? 1.0 : -1.0;
      KokkosBlas::axpy(TestExecSpace{}, sigma / kViscosity, q, u_exterior);
      EXPECT_LT(max_relative_difference(u_interior, u_exterior), 1.0e-12)
          << "add_singularity_subtraction vs apply_stokes_double_layer_kernel_ss";
    }
  }
}

// A declared normal orientation that contradicts the geometry throws, since it would flip the sign of the jump.
TEST(Periphery, SkfieRejectsMismatchedNormals) {
  for (const bool outward_normal : {false, true}) {
    SCOPED_TRACE(testing::Message() << "outward_normal=" << outward_normal);
    const auto surface = make_sphere_surface(6, kPeripheryRadius, outward_normal);
    const size_t num_nodes = surface.num_nodes;
    const bool wrong = !outward_normal;
    DeviceMatrix M("M", 3 * num_nodes, 3 * num_nodes);
    DeviceVector q("q", 3 * num_nodes);
    DeviceVector u("u", 3 * num_nodes);
    Kokkos::deep_copy(q, 1.0);

    EXPECT_THROW(fill_skfie_matrix(TestExecSpace{}, kViscosity, num_nodes, surface.points, surface.normals,
                                   surface.weights, M, wrong),
                 std::invalid_argument);
    EXPECT_THROW(apply_skfie(TestExecSpace{}, kViscosity, num_nodes, surface.points, surface.normals, surface.weights,
                             q, u, wrong),
                 std::invalid_argument);

    // add_singularity_subtraction cannot see the normals; it infers their orientation from T's block row sums.
    fill_stokes_double_layer_matrix(TestExecSpace{}, kViscosity, num_nodes, num_nodes, surface.points, surface.points,
                                    surface.normals, surface.weights, M);
    EXPECT_THROW(add_singularity_subtraction(TestExecSpace{}, kViscosity, M, wrong), std::invalid_argument);

    TestPeriphery periphery(num_nodes, kViscosity);
    periphery.set_surface_positions(surface.points)
        .set_surface_normals(surface.normals, wrong)
        .set_quadrature_weights(surface.weights);
    EXPECT_THROW(periphery.build_inverse_self_interaction_matrix(), std::invalid_argument);

    // The matching declaration is accepted.
    EXPECT_NO_THROW(fill_skfie_matrix(TestExecSpace{}, kViscosity, num_nodes, surface.points, surface.normals,
                                      surface.weights, M, outward_normal));
  }
}

/// \brief A point force outside the periphery and 50 points inside it, within 0.6 R.
struct InteriorFlowProblem {
  DeviceVector source_position;  //!< 3
  DeviceVector source_force;     //!< 3
  DeviceVector bulk_points;      //!< 3 x 50
};

/// \brief The InteriorFlowProblem, its bulk points drawn by a seeded Philox stream.
InteriorFlowProblem make_interior_flow_problem() {
  const size_t num_bulk_points = 50;
  std::vector<double> bulk_points(3 * num_bulk_points);
  openrand::Philox rng = make_philox(1234, 0);
  for (size_t i = 0; i < num_bulk_points; ++i) {
    const double theta = rng.uniform(0.0, 2.0 * kPi);
    const double phi = rng.uniform(0.0, kPi);
    const double r = rng.uniform(0.0, 0.6 * kPeripheryRadius);
    bulk_points[3 * i] = r * std::sin(phi) * std::cos(theta);
    bulk_points[3 * i + 1] = r * std::sin(phi) * std::sin(theta);
    bulk_points[3 * i + 2] = r * std::cos(phi);
  }
  return {to_device({1.7 * kPeripheryRadius, 0.4 * kPeripheryRadius, -0.9 * kPeripheryRadius}),
          to_device({0.3, -1.0, 0.6}), to_device(bulk_points)};
}

/// \brief ||u_h + u_exact|| / ||u_exact|| at the bulk points, for the flow u_h the order-`order` periphery induces.
double interior_flow_error(const InteriorFlowProblem& problem, const int order, const double viscosity,
                           const bool outward_normal) {
  const auto surface = make_sphere_surface(order, kPeripheryRadius, outward_normal);
  const size_t num_bulk_points = problem.bulk_points.extent(0) / 3;
  DeviceVector u_exact("u_exact", 3 * num_bulk_points);
  apply_stokes_kernel(TestExecSpace{}, viscosity, problem.source_position, problem.bulk_points, problem.source_force,
                      u_exact);

  DeviceVector slip("slip", 3 * surface.num_nodes);
  apply_stokes_kernel(TestExecSpace{}, viscosity, problem.source_position, surface.points, problem.source_force, slip);
  TestPeriphery periphery = make_periphery(surface, viscosity);
  DeviceVector surface_density("surface_density", 3 * surface.num_nodes);
  periphery.compute_surface_density(slip, surface_density);
  DeviceVector u_h("u_h", 3 * num_bulk_points);
  apply_stokes_double_layer_kernel(TestExecSpace{}, viscosity, surface.num_nodes, num_bulk_points, surface.points,
                                   problem.bulk_points, surface.normals, surface.weights, surface_density, u_h);

  const auto h_exact = to_host(u_exact);
  const auto h_u_h = to_host(u_h);
  double error = 0.0;
  double exact_norm = 0.0;
  for (size_t i = 0; i < 3 * num_bulk_points; ++i) {
    error += (h_u_h(i) + h_exact(i)) * (h_u_h(i) + h_exact(i));
    exact_norm += h_exact(i) * h_exact(i);
  }
  return std::sqrt(error) / std::sqrt(exact_norm);
}

// A no-slip periphery cancels an outside point force's flow everywhere inside: its flow and the force's are interior
// Stokes flows with opposite boundary values. The relative error falls to 1e-5 by order 28 for both normal orientations
// and matches across viscosities.
TEST(Periphery, PeripheryCancelsAnOutsideFlow) {
  const std::vector<int> orders = {8, 12, 16, 28};
  const InteriorFlowProblem problem = make_interior_flow_problem();
  const double num_unknowns = 3.0 * make_sphere_surface(orders.back(), kPeripheryRadius, false).num_nodes;
  const double roundoff = num_unknowns * std::numeric_limits<double>::epsilon();

  for (const bool outward_normal : {false, true}) {
    std::vector<std::vector<double>> errors_by_viscosity;
    for (const double viscosity : kViscosities) {
      SCOPED_TRACE(testing::Message() << "outward_normal=" << outward_normal << " viscosity=" << viscosity);
      std::vector<double> errors;
      for (const int order : orders) {
        errors.push_back(interior_flow_error(problem, order, viscosity, outward_normal));
      }
      EXPECT_TRUE(decreases_or_reaches(errors, 0.0))
          << "interior flow must converge under refinement: orders=" << testing::PrintToString(orders)
          << " errors=" << testing::PrintToString(errors);
      EXPECT_LT(errors.back(), 1.0e-5) << "finest periphery must reproduce the interior Stokes flow";
      errors_by_viscosity.push_back(errors);
    }
    for (size_t i = 0; i < orders.size(); ++i) {
      EXPECT_NEAR(errors_by_viscosity[1][i], errors_by_viscosity[0][i], roundoff)
          << "interior-flow error must not depend on the viscosity, outward_normal=" << outward_normal
          << ", order=" << orders[i];
    }
  }
}

//@}

//! \name A sphere inside the periphery
//@{

/// \brief The flow the periphery induces at RPYC spheres to cancel their flow on it.
DeviceVector periphery_response(TestPeriphery& periphery, const DeviceVector& positions, const DeviceVector& radii,
                                const DeviceVector& forces) {
  const size_t num_nodes = periphery.get_num_nodes();
  const double viscosity = periphery.get_viscosity();
  DeviceVector node_radii("node_radii", num_nodes);
  DeviceVector slip("slip", 3 * num_nodes);
  apply_rpyc_kernel(TestExecSpace{}, viscosity, positions, periphery.get_surface_positions(), radii, node_radii, forces,
                    slip);
  DeviceVector surface_density("surface_density", 3 * num_nodes);
  periphery.compute_surface_density(slip, surface_density);
  DeviceVector velocities("velocities", positions.extent(0));
  apply_stokes_double_layer_kernel(TestExecSpace{}, viscosity, num_nodes, radii.extent(0),
                                   periphery.get_surface_positions(), positions, periphery.get_surface_normals(),
                                   periphery.get_quadrature_weights(), surface_density, velocities);
  return velocities;
}

/// \brief The periphery's correction C to the 3x3 mobility of one sphere of the given radius at position.
Matrix3d sphere_periphery_correction(TestPeriphery& periphery, const double sphere_radius, const Vector3d& position) {
  const DeviceVector positions = to_device({position[0], position[1], position[2]});
  const DeviceVector radii = to_device({sphere_radius});
  Matrix3d correction;
  for (size_t k = 0; k < 3; ++k) {
    std::vector<double> force(3, 0.0);
    force[k] = 1.0;
    const auto velocity = to_host(periphery_response(periphery, positions, radii, to_device(force)));
    for (size_t i = 0; i < 3; ++i) {
      correction(i, k) = velocity(i);
    }
  }
  return correction;
}

// The correction C that a periphery of radius R adds to the mobility of a sphere of radius a converges.
//
// At the center, it converges to its known value, the image of the sphere's flow:
//    -3 / (8 pi viscosity R) (1 - 5 a^2 / (9 R^2)) I.
// Halfway to the wall, where we have no closed form, it converges to a symmetric matrix (Lorentz reciprocity)
// independent of the normals' orientation. Each error falls with order to below the a^2 term, the finest effect
// resolved, and C scales as 1 / viscosity to roundoff.
TEST(Periphery, SphereMobilityCorrectionConverges) {
  const double a = 0.5;  // large enough for its a^2 term to stand out from the discretization error
  const double R = kPeripheryRadius;
  const double leading = -3.0 / (8.0 * kPi * kViscosity * R);          // the image of a point force
  const double exact = leading * (1.0 - 5.0 * a * a / (9.0 * R * R));  // plus the image of the RPY flow's a^2 term
  const double a2_term = std::abs(exact - leading) / std::abs(exact);
  const Vector3d center(0.0, 0.0, 0.0);
  const Vector3d off_center = Vector3d(1.0, 2.0, 3.0) * (0.5 * R / std::sqrt(14.0));

  std::vector<double> center_errors;
  std::vector<double> asymmetries;
  std::vector<double> orientation_differences;
  for (const int order : {8, 12, 16}) {
    TestPeriphery inward = make_periphery(make_sphere_surface(order, R, false), kViscosity);
    TestPeriphery outward = make_periphery(make_sphere_surface(order, R, true), kViscosity);
    const Matrix3d C0 = sphere_periphery_correction(inward, a, center);
    double center_error = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        center_error = std::max(center_error, std::abs(C0(i, j) - (i == j ? exact : 0.0)) / std::abs(exact));
      }
    }
    center_errors.push_back(center_error);
    const Matrix3d C1 = sphere_periphery_correction(inward, a, off_center);
    const Matrix3d C2 = sphere_periphery_correction(outward, a, off_center);
    asymmetries.push_back(frobenius_norm(C1 - transpose(C1)) / frobenius_norm(C1));
    orientation_differences.push_back(frobenius_norm(C2 - C1) / frobenius_norm(C1));
  }
  for (const auto& [name, errors] : {std::pair{"center error", center_errors}, std::pair{"asymmetry", asymmetries},
                                     std::pair{"inward vs outward", orientation_differences}}) {
    EXPECT_TRUE(decreases_or_reaches(errors, 0.0)) << name << ": " << testing::PrintToString(errors);
    EXPECT_LT(errors.back(), a2_term) << name << " must resolve the a^2 term";
  }

  TestPeriphery unit = make_periphery(make_sphere_surface(8, R, false), 1.0);
  TestPeriphery non_unit = make_periphery(make_sphere_surface(8, R, false), kViscosity);
  const Matrix3d C_unit = sphere_periphery_correction(unit, a, off_center);
  const Matrix3d C_non_unit = kViscosity * sphere_periphery_correction(non_unit, a, off_center);
  const double roundoff = 3.0 * static_cast<double>(unit.get_num_nodes()) * std::numeric_limits<double>::epsilon();
  EXPECT_LT(frobenius_norm(C_non_unit - C_unit) / frobenius_norm(C_unit), roundoff)
      << "viscosity C must not depend on the viscosity";
}

//@}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)

//! \name Belos solves
//@{

// Belos solves on Tpetra's node space, which is TestExecSpace in the CUDA and OpenMP builds.
using SolveExecSpace = Tpetra::Map<>::node_type::execution_space;
using SolvePeriphery = PeripheryT<SolveExecSpace>;
using SolveVector = Kokkos::View<double*, Kokkos::LayoutLeft, SolveExecSpace::memory_space>;
using SolveMatrix = Kokkos::View<double**, Kokkos::LayoutLeft, SolveExecSpace::memory_space>;

constexpr double kCavityRadius = 5.0;  //!< The radius of the spherical cavity the body tests use
constexpr Quaterniond kNoRotation(1.0, 0.0, 0.0, 0.0);

/// \brief Pseudo-block GMRES to a relative residual of 1e-10, within the given iteration budget.
BelosConfig<double> make_gmres_config(const unsigned max_iters = 500, const unsigned num_blocks = 100,
                                      const unsigned max_restarts = 20) {
  BelosConfig<double> cfg;
  cfg.solver = BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = max_iters;
  cfg.num_blocks = num_blocks;
  cfg.max_restarts = max_restarts;
  return cfg;
}

/// \brief A body's translational and angular velocity.
struct RigidVelocity {
  Vector3d velocity;
  Vector3d angular_velocity;
};

/// \brief Body b's RigidVelocity in the solution x = [ q | U | Omega ] of a block mobility solve.
RigidVelocity rigid_velocity(const BodySet<SolveExecSpace>& bodies, const SolveVector& x, const size_t b) {
  const auto h_x = to_host(x);
  const size_t offset = bodies.rigid_velocity_offset(b);
  return {Vector3d(h_x(offset), h_x(offset + 1), h_x(offset + 2)),
          Vector3d(h_x(offset + 3), h_x(offset + 4), h_x(offset + 5))};
}

/// \brief A block mobility solve: GMRES's result and the solution x = [ q | U | Omega ].
struct BodySolve {
  BelosResult<double> result;
  SolveVector x;
};

/// \brief The block mobility solve for bodies among spheres in free space.
BodySolve run_body_solve(const BodySet<SolveExecSpace>& bodies, const SphereSet<SolveExecSpace>& spheres,
                         const double viscosity, const BelosConfig<double>& cfg) {
  SolveVector x("x", bodies.num_dofs());
  const SolveVector none;
  const BelosResult<double> result =
      solve_mobility(SolveExecSpace{}, bodies, spheres, viscosity, 0, none, none, none, SolveMatrix{}, cfg, x);
  return {result, x};
}

/// \brief The block mobility solve for bodies among spheres inside periphery, at its viscosity.
BodySolve run_confined_body_solve(const SolvePeriphery& periphery, const BodySet<SolveExecSpace>& bodies,
                                  const SphereSet<SolveExecSpace>& spheres, const BelosConfig<double>& cfg) {
  SolveVector x("x", bodies.num_dofs());
  const BelosResult<double> result =
      solve_mobility(SolveExecSpace{}, bodies, spheres, periphery.get_viscosity(), periphery.get_num_nodes(),
                     periphery.get_surface_positions(), periphery.get_surface_normals(),
                     periphery.get_quadrature_weights(), periphery.get_M_inv(), cfg, x);
  return {result, x};
}

//@}

//! \name The matrix-free inverse
//@{

// The matrix-free GMRES inverse matches the dense direct inverse well within 1e-6: both invert the same
// well-conditioned M, and GMRES solves to a 1e-10 residual.
TEST(Periphery, MatrixFreeMatchesDirectInverse) {
  for (const bool outward_normal : {false, true}) {
    SCOPED_TRACE(testing::Message() << "outward_normal=" << outward_normal);
    const auto surface = make_sphere_surface<SolveExecSpace>(10, 1.0, outward_normal);
    const SolveVector slip = to_device<SolveExecSpace>(make_random_values(3 * surface.num_nodes, 0));

    SolvePeriphery direct = make_periphery(surface, kViscosity);
    direct.set_inverse_method(InverseMethod::Direct);
    SolveVector q_direct("q_direct", 3 * surface.num_nodes);
    direct.compute_surface_density(slip, q_direct);

    SolvePeriphery matrix_free(surface.num_nodes, kViscosity);
    matrix_free.set_surface_positions(surface.points)
        .set_surface_normals(surface.normals, outward_normal)
        .set_quadrature_weights(surface.weights)
        .set_belos_config(make_gmres_config())
        .set_inverse_method(InverseMethod::MatrixFreeGMRES)
        .build_matrix_free_inverse();
    SolveVector q_matrix_free("q_matrix_free", 3 * surface.num_nodes);
    matrix_free.compute_surface_density(slip, q_matrix_free);

    const BelosResult<double>& result = matrix_free.get_last_matrix_free_result();
    EXPECT_TRUE(result.converged) << "GMRES failed to converge: " << result;
    EXPECT_LT(max_relative_difference(q_direct, q_matrix_free), 1.0e-6) << "matrix-free vs direct, GMRES: " << result;
  }
}

// A mismatched normal orientation throws when the matrix-free inverse is built, not inside its first GMRES apply.
TEST(Periphery, MatrixFreeInverseRejectsMismatchedNormals) {
  for (const bool outward_normal : {false, true}) {
    const auto surface = make_sphere_surface<SolveExecSpace>(6, kPeripheryRadius, outward_normal);
    SolvePeriphery periphery(surface.num_nodes, kViscosity);
    periphery.set_surface_positions(surface.points)
        .set_surface_normals(surface.normals, !outward_normal)
        .set_quadrature_weights(surface.weights);
    EXPECT_THROW(periphery.build_matrix_free_inverse(), std::invalid_argument) << "outward_normal=" << outward_normal;
  }
}

//@}

//! \name Motile bodies
//@{

// Resolved rigid bodies among RPY spheres are solved as one block system by matrix-free GMRES.

// set_orientation stores (w, x, y, z), and place_body_in_lab_frame rotates by it, then translates: a quarter turn about
// z maps x to y and y to -x.
TEST(Periphery, PlaceBodyInLabFrameAppliesOrientation) {
  MotileBody<SolveExecSpace> body;
  body.num_quadrature_points = 2;
  body.area = 1.0;
  body.positions_ref = to_device<SolveExecSpace>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  body.normals_ref = to_device<SolveExecSpace>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  body.weights = to_device<SolveExecSpace>({0.5, 0.5});
  body.positions = SolveVector("positions", 6);
  body.normals = SolveVector("normals", 6);
  BodySet<SolveExecSpace> bodies;
  bodies.bodies.push_back(body);
  bodies.wire_pose_load();

  const double c = std::sqrt(0.5);
  bodies.set_center(0, Vector3d(0.3, -0.2, 0.5));
  bodies.set_orientation(0, Quaterniond(c, 0.0, 0.0, c));  // (w, x, y, z): a quarter turn about z
  place_body_in_lab_frame(SolveExecSpace{}, bodies.bodies[0]);

  const auto orientation = to_host(bodies.orientations);
  const std::array<double, 4> expected_orientation = {c, 0.0, 0.0, c};
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(orientation(i), expected_orientation[i]) << "orientations must be stored as (w, x, y, z), index " << i;
  }

  const auto positions = to_host(bodies.bodies[0].positions);
  const auto normals = to_host(bodies.bodies[0].normals);
  const std::array<double, 6> expected_positions = {0.3, 0.8, 0.5, -0.7, -0.2, 0.5};
  const std::array<double, 6> expected_normals = {0.0, 1.0, 0.0, -1.0, 0.0, 0.0};
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(positions(i), expected_positions[i], 1.0e-14) << "position component " << i;
    EXPECT_NEAR(normals(i), expected_normals[i], 1.0e-14) << "normal component " << i;
  }
}

// A lone sphere of radius a moves at the Stokes drag, whatever its viscosity, position, and orientation:
//    U = F / (6 pi viscosity a),    Omega = tau / (8 pi viscosity a^3).
// The completed double layer represents rigid motion exactly, so only the GMRES tolerance remains.
TEST(Periphery, FreeBodyMatchesStokesDrag) {
  const double a = 1.0;
  const double c = std::sqrt(0.5);
  for (const double viscosity : kViscosities) {
    SCOPED_TRACE(testing::Message() << "viscosity=" << viscosity);
    const BodySet<SolveExecSpace> bodies = make_sphere_body_set<SolveExecSpace>(
        8, a, Vector3d(0.3, -0.2, 0.5), Quaterniond(c, 0.0, 0.0, c), Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 0.0, 1.0));
    const BodySolve solve = run_body_solve(bodies, SphereSet<SolveExecSpace>{}, viscosity, make_gmres_config());
    EXPECT_TRUE(solve.result.converged) << "GMRES did not converge: " << solve.result;

    const RigidVelocity rigid = rigid_velocity(bodies, solve.x, 0);
    const double u_expected = 1.0 / (6.0 * kPi * viscosity * a);
    const double omega_expected = 1.0 / (8.0 * kPi * viscosity * a * a * a);
    const double u_tol = 1.0e-8 * u_expected;
    const double omega_tol = 1.0e-8 * omega_expected;
    EXPECT_NEAR(rigid.velocity[0], u_expected, u_tol);
    EXPECT_NEAR(rigid.velocity[1], 0.0, u_tol);
    EXPECT_NEAR(rigid.velocity[2], 0.0, u_tol);
    EXPECT_NEAR(rigid.angular_velocity[0], 0.0, omega_tol);
    EXPECT_NEAR(rigid.angular_velocity[1], 0.0, omega_tol);
    EXPECT_NEAR(rigid.angular_velocity[2], omega_expected, omega_tol);
  }
}

/// \brief One RPY sphere of the given radius at center under force: the source of an ambient flow.
SphereSet<SolveExecSpace> make_rpy_source(const Vector3d& center, const double radius, const Vector3d& force) {
  return SphereSet<SolveExecSpace>(to_device<SolveExecSpace>({center[0], center[1], center[2]}),
                                   to_device<SolveExecSpace>({radius}),
                                   to_device<SolveExecSpace>({force[0], force[1], force[2]}), SphereInteraction::RPY);
}

// A force- and torque-free sphere of radius a at distance r from a point force F moves at exactly the Faxen velocity,
// the RPY mobility:
//    U = (1 + a^2 / 6 grad^2) G F = M(r; 0, a) F.
// The body solve must converge to it: the error falls at every order until the solver floor, and by at least three
// decades in all.
TEST(Periphery, ForceFreeBodyConvergesToTheFaxenVelocity) {
  const double a = 1.0;
  const SphereSet<SolveExecSpace> source = make_rpy_source(Vector3d(3.0, 0.0, 0.0), 0.0, Vector3d(0.0, 1.0, 0.0));

  // The Faxen velocity M(r; 0, a) F: the source's RPY flow at a radius-a target at the body's center.
  SolveVector expected("expected", 3);
  apply_rpy_kernel(SolveExecSpace{}, kViscosity, source.positions, to_device<SolveExecSpace>({0.0, 0.0, 0.0}),
                   source.radii, to_device<SolveExecSpace>({a}), source.forces, expected);
  const auto h_expected = to_host(expected);
  const Vector3d u_expected(h_expected(0), h_expected(1), h_expected(2));
  ASSERT_GT(norm(u_expected), 0.0) << "the reference velocity is degenerate";

  const BelosConfig<double> cfg = make_gmres_config();
  const std::vector<int> orders = {4, 6, 8, 10, 12, 14};
  std::vector<double> errors;
  for (const int order : orders) {
    const BodySet<SolveExecSpace> bodies =
        make_sphere_body_set<SolveExecSpace>(order, a, Vector3d(0.0, 0.0, 0.0), kNoRotation);
    const BodySolve solve = run_body_solve(bodies, source, kViscosity, cfg);
    EXPECT_TRUE(solve.result.converged) << "GMRES did not converge at order=" << order << ": " << solve.result;
    errors.push_back(norm(rigid_velocity(bodies, solve.x, 0).velocity - u_expected) / norm(u_expected));
  }

  EXPECT_TRUE(decreases_or_reaches(errors, 5.0 * cfg.tol))
      << "error must fall under refinement: orders=" << testing::PrintToString(orders)
      << " errors=" << testing::PrintToString(errors);
  EXPECT_LT(errors.back(), 100.0 * cfg.tol) << "finest quadrature must reach the Faxen velocity";
  EXPECT_LT(errors.back(), errors.front() / 1.0e3) << "refinement must collapse the error by at least three decades";
}

// The block system is linear in its loads, so the responses to the body's load and to the ambient flow add up to the
// response to both. This guards the right-hand side's assembly against a missed zero or an overwrite.
TEST(Periphery, BodyAndAmbientFlowsSuperpose) {
  const BelosConfig<double> cfg = make_gmres_config();
  BodySet<SolveExecSpace> bodies = make_sphere_body_set<SolveExecSpace>(8, 1.0, Vector3d(0.0, 0.0, 0.0), kNoRotation);
  const SphereSet<SolveExecSpace> source = make_rpy_source(Vector3d(5.0, 0.0, 0.0), 0.5, Vector3d(0.0, 1.0, 0.0));
  const SphereSet<SolveExecSpace> no_spheres;

  bodies.set_net_force(0, Vector3d(1.0, 0.0, 0.0));
  bodies.set_net_torque(0, Vector3d(0.0, 0.0, 1.0));
  const BodySolve body_only = run_body_solve(bodies, no_spheres, kViscosity, cfg);
  const BodySolve both = run_body_solve(bodies, source, kViscosity, cfg);
  bodies.set_net_force(0, Vector3d(0.0, 0.0, 0.0));
  bodies.set_net_torque(0, Vector3d(0.0, 0.0, 0.0));
  const BodySolve ambient_only = run_body_solve(bodies, source, kViscosity, cfg);
  ASSERT_TRUE(body_only.result.converged) << body_only.result;
  ASSERT_TRUE(ambient_only.result.converged) << ambient_only.result;
  ASSERT_TRUE(both.result.converged) << both.result;

  const RigidVelocity u_body = rigid_velocity(bodies, body_only.x, 0);
  const RigidVelocity u_ambient = rigid_velocity(bodies, ambient_only.x, 0);
  const RigidVelocity u_both = rigid_velocity(bodies, both.x, 0);
  const double tol = 100.0 * cfg.tol;
  for (size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(u_both.velocity[k], u_body.velocity[k] + u_ambient.velocity[k], tol) << "velocity component " << k;
    EXPECT_NEAR(u_both.angular_velocity[k], u_body.angular_velocity[k] + u_ambient.angular_velocity[k], tol)
        << "angular velocity component " << k;
  }
}

/// \brief The rigid velocities of a sphere under a load, alone and at the center of the spherical cavity.
struct CavityResponse {
  RigidVelocity unbounded;  //!< In free space
  RigidVelocity confined;   //!< At the center of the cavity of radius kCavityRadius
};

/// \brief The CavityResponse of a sphere of radius a under force and torque, with sphere and cavity of order `order`.
CavityResponse run_cavity(const double viscosity, const double a, const int order, const BelosConfig<double>& cfg,
                          const Vector3d& force, const Vector3d& torque) {
  const SolvePeriphery cavity =
      make_periphery(make_sphere_surface<SolveExecSpace>(order, kCavityRadius, /*outward_normal=*/false), viscosity);
  const BodySet<SolveExecSpace> bodies =
      make_sphere_body_set<SolveExecSpace>(order, a, Vector3d(0.0, 0.0, 0.0), kNoRotation, force, torque);
  const SphereSet<SolveExecSpace> no_spheres;
  const BodySolve unbounded = run_body_solve(bodies, no_spheres, viscosity, cfg);
  const BodySolve confined = run_confined_body_solve(cavity, bodies, no_spheres, cfg);
  EXPECT_TRUE(unbounded.result.converged) << "unbounded GMRES, order=" << order << ": " << unbounded.result;
  EXPECT_TRUE(confined.result.converged) << "confined GMRES, order=" << order << ": " << confined.result;
  return {rigid_velocity(bodies, unbounded.x, 0), rigid_velocity(bodies, confined.x, 0)};
}

// A sphere of radius a at the center of a spherical cavity of radius b, with lambda = a / b, moves at Happel &
// Brenner's exact mobilities (Eqs. 4-22.11 and 7-8.18):
//    U = F / (6 pi viscosity a) (1 - 9 lambda / 4 + 5 lambda^3 / 2 - 9 lambda^5 / 4 + lambda^6) / (1 - lambda^5),
//    Omega = T / (8 pi viscosity a^3) (1 - lambda^3).
// The unbounded drag is exact, so all of the error is wall coupling. Translation converges faster than 1 / order^2,
// independent of the viscosity. Rotation converges to the solver floor, since the subtraction cancels its rigid slip
// exactly.
TEST(Periphery, BodyInSphericalCavityMatchesHappelAndBrenner) {
  const double a = 1.0;
  const double lambda = a / kCavityRadius;  // 0.2
  const double l3 = lambda * lambda * lambda;
  const double l5 = l3 * lambda * lambda;
  const double l6 = l5 * lambda;
  const BelosConfig<double> cfg = make_gmres_config(1000, 150, 30);
  const std::vector<int> orders = {4, 6, 8, 10, 12, 16, 20, 24};

  std::vector<std::vector<double>> translation_errors_by_viscosity;
  for (const double viscosity : kViscosities) {
    SCOPED_TRACE(testing::Message() << "viscosity=" << viscosity);
    const double u_stokes = 1.0 / (6.0 * kPi * viscosity * a);
    const double omega_stokes = 1.0 / (8.0 * kPi * viscosity * a * a * a);
    const double u_exact = u_stokes * (1.0 - 2.25 * lambda + 2.5 * l3 - 2.25 * l5 + l6) / (1.0 - l5);
    const double omega_exact = omega_stokes * (1.0 - l3);

    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    for (const int order : orders) {
      const CavityResponse response =
          run_cavity(viscosity, a, order, cfg, Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 0.0, 1.0));
      EXPECT_LT(std::abs(response.unbounded.velocity[0] - u_stokes) / u_stokes, 1.0e-8)
          << "unbounded Stokes drag must be exact, order=" << order;
      EXPECT_LT(std::abs(response.unbounded.angular_velocity[2] - omega_stokes) / omega_stokes, 1.0e-8)
          << "unbounded rotational Stokes drag must be exact, order=" << order;
      translation_errors.push_back(std::abs(response.confined.velocity[0] - u_exact) / u_exact);
      rotation_errors.push_back(std::abs(response.confined.angular_velocity[2] - omega_exact) / omega_exact);
    }

    EXPECT_TRUE(decreases_or_reaches(translation_errors, 0.0))
        << "confined translation must converge under refinement: orders=" << testing::PrintToString(orders)
        << " errors=" << testing::PrintToString(translation_errors);
    EXPECT_LT(translation_errors.back(), 1.0e-4) << "finest quadrature must reach the cavity's translational mobility";
    const size_t middle = orders.size() / 2;
    const double observed_rate = std::log(translation_errors[middle] / translation_errors.back()) /
                                 std::log(static_cast<double>(orders.back()) / orders[middle]);
    EXPECT_GT(observed_rate, 2.0) << "confined translation must converge faster than 1 / order^2";

    EXPECT_TRUE(decreases_or_reaches(rotation_errors, 5.0 * cfg.tol))
        << "confined rotation must converge under refinement: orders=" << testing::PrintToString(orders)
        << " errors=" << testing::PrintToString(rotation_errors);
    EXPECT_LT(rotation_errors.back(), 100.0 * cfg.tol)
        << "finest quadrature must reach the cavity's rotational mobility";
    translation_errors_by_viscosity.push_back(translation_errors);
  }

  for (size_t i = 0; i < orders.size(); ++i) {
    EXPECT_NEAR(translation_errors_by_viscosity[1][i], translation_errors_by_viscosity[0][i],
                1.0e-2 * translation_errors_by_viscosity[0][i])
        << "confined translation error must not depend on the viscosity, order=" << orders[i];
  }
}

/// \brief Whether a Wilson sphere is an RPY point sphere or a resolved rigid inclusion.
enum class SphereType { RPY, Inclusion };

/// \brief Wilson's three reported velocities: U1 and U2 of s1 and s2 along -y, and U3 of s3 along x.
struct WilsonVelocities {
  double u1;
  double u2;
  double u3;
};

/// \brief The corners s1, s2, s3 of Wilson's equilateral triangle of side s r.
std::array<Vector3d, 3> wilson_triangle(const double s, const double r) {
  const double height = std::sqrt(3.0) / 2.0 * s * r;
  return {Vector3d(0.0, 0.0, 0.0), Vector3d(-s * r / 2.0, -height, 0.0), Vector3d(s * r / 2.0, -height, 0.0)};
}

/// \brief The WilsonVelocities of the three spheres as RPY point spheres, by direct evaluation of the RPY tensor.
WilsonVelocities expected_rpy_wilson_velocities(const double viscosity, const double r, const double f1,
                                                const double s) {
  const std::array<Vector3d, 3> centers = wilson_triangle(s, r);
  std::vector<double> positions;
  for (const Vector3d& center : centers) {
    positions.insert(positions.end(), {center[0], center[1], center[2]});
  }
  const SolveVector d_positions = to_device<SolveExecSpace>(positions);
  const SolveVector radii = to_device<SolveExecSpace>({r, r, r});
  const SolveVector forces = to_device<SolveExecSpace>({0.0, f1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  SolveVector velocities("velocities", 9);
  apply_rpy_kernel(SolveExecSpace{}, viscosity, d_positions, d_positions, radii, radii, forces, velocities);
  apply_local_drag(SolveExecSpace{}, viscosity, velocities, forces, radii);
  const auto v = to_host(velocities);
  return {-v(1), -v(4), v(6)};
}

/// \brief The WilsonVelocities at separation s, with each sphere an RPY point sphere or a resolved inclusion.
WilsonVelocities run_wilson(const double viscosity, const double r, const double f1, const double s, const int order,
                            const BelosConfig<double>& cfg, const std::array<SphereType, 3>& types) {
  const std::array<Vector3d, 3> centers = wilson_triangle(s, r);
  const std::array<Vector3d, 3> forces = {Vector3d(0.0, f1, 0.0), Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0)};
  std::vector<size_t> body_indices;
  std::vector<size_t> sphere_indices;
  for (size_t j = 0; j < 3; ++j) {
    (types[j] == SphereType::Inclusion ? body_indices : sphere_indices).push_back(j);
  }

  BodySet<SolveExecSpace> bodies;
  for (size_t m = 0; m < body_indices.size(); ++m) {
    bodies.bodies.push_back(make_sphere_body<SolveExecSpace>(order, r));
  }
  bodies.wire_pose_load();
  for (size_t m = 0; m < body_indices.size(); ++m) {
    bodies.set_center(m, centers[body_indices[m]]);
    bodies.set_orientation(m, kNoRotation);
    bodies.set_net_force(m, forces[body_indices[m]]);
    bodies.set_net_torque(m, Vector3d(0.0, 0.0, 0.0));
    place_body_in_lab_frame(SolveExecSpace{}, bodies.bodies[m]);
  }

  std::vector<double> sphere_positions;
  std::vector<double> sphere_radii;
  std::vector<double> sphere_forces;
  for (const size_t j : sphere_indices) {
    sphere_positions.insert(sphere_positions.end(), {centers[j][0], centers[j][1], centers[j][2]});
    sphere_radii.push_back(r);
    sphere_forces.insert(sphere_forces.end(), {forces[j][0], forces[j][1], forces[j][2]});
  }
  const SphereSet<SolveExecSpace> spheres(to_device<SolveExecSpace>(sphere_positions),
                                          to_device<SolveExecSpace>(sphere_radii),
                                          to_device<SolveExecSpace>(sphere_forces), SphereInteraction::RPY);

  // The RPY spheres' own velocity, their mutual RPY flow and self-drag; evaluate_sphere_flow adds the bodies' flow.
  SolveVector sphere_velocities("sphere_velocities", sphere_positions.size());
  apply_rpy_kernel(SolveExecSpace{}, viscosity, spheres.positions, spheres.positions, spheres.radii, spheres.radii,
                   spheres.forces, sphere_velocities);
  apply_local_drag(SolveExecSpace{}, viscosity, sphere_velocities, spheres.forces, spheres.radii);

  std::array<Vector3d, 3> velocities;
  if (!body_indices.empty()) {
    const BodySolve solve = run_body_solve(bodies, spheres, viscosity, cfg);
    EXPECT_TRUE(solve.result.converged) << "GMRES failed at s=" << s << ": " << solve.result;
    const SolveVector none;
    evaluate_sphere_flow(SolveExecSpace{}, bodies, spheres, viscosity, 0, none, none, none, SolveMatrix{}, solve.x,
                         sphere_velocities);
    for (size_t m = 0; m < body_indices.size(); ++m) {
      velocities[body_indices[m]] = rigid_velocity(bodies, solve.x, m).velocity;
    }
  }
  const auto h_sphere_velocities = to_host(sphere_velocities);
  for (size_t i = 0; i < sphere_indices.size(); ++i) {
    velocities[sphere_indices[i]] =
        Vector3d(h_sphere_velocities(3 * i), h_sphere_velocities(3 * i + 1), h_sphere_velocities(3 * i + 2));
  }
  return {-velocities[0][1], -velocities[1][1], velocities[2][0]};
}

// Three spheres sit at the corners of an equilateral triangle, with a force on one (Wilson 2013, Stokes flow past three
// spheres). With each sphere an RPY point (R) or a resolved inclusion (I), all-resolved (III) matches Wilson to 1e-3,
// all-RPY (RRR) matches the RPY tensor, and the mixed cases fall between, since RPY misses the back-reaction.
TEST(Periphery, ThreeSpheresMatchWilson) {
  const double r = 12.34;
  const int order = 12;
  const double f1 = -6.0 * kPi * kViscosity * r;
  const BelosConfig<double> cfg = make_gmres_config(2000, 200, 40);

  // Wilson's U1, U2, U3 at each separation s.
  const std::array<double, 4> separations = {2.5, 3.0, 4.0, 6.0};
  const std::array<WilsonVelocities, 4> wilson = {{{0.87765, 0.49545, 0.07393},
                                                   {0.93905, 0.41694, 0.07824},
                                                   {0.97964, 0.31859, 0.06925},
                                                   {0.99581, 0.21586, 0.05078}}};

  using T = SphereType;
  for (size_t i = 0; i < separations.size(); ++i) {
    const double s = separations[i];
    SCOPED_TRACE(testing::Message() << "s=" << s);
    const WilsonVelocities rpy = expected_rpy_wilson_velocities(kViscosity, r, f1, s);
    const WilsonVelocities rrr = run_wilson(kViscosity, r, f1, s, order, cfg, {T::RPY, T::RPY, T::RPY});
    const WilsonVelocities rir = run_wilson(kViscosity, r, f1, s, order, cfg, {T::RPY, T::Inclusion, T::RPY});
    const WilsonVelocities irr = run_wilson(kViscosity, r, f1, s, order, cfg, {T::Inclusion, T::RPY, T::RPY});
    const WilsonVelocities iii =
        run_wilson(kViscosity, r, f1, s, order, cfg, {T::Inclusion, T::Inclusion, T::Inclusion});

    // RRR matches the RPY tensor.
    EXPECT_NEAR(rrr.u1, rpy.u1, 1.0e-9) << "RRR U1 == RPY";
    EXPECT_NEAR(rrr.u2, rpy.u2, 1.0e-9) << "RRR U2 == RPY";
    EXPECT_NEAR(rrr.u3, rpy.u3, 1.0e-9) << "RRR U3 == RPY";

    // III matches Wilson's values.
    EXPECT_NEAR(iii.u1, wilson[i].u1, 1.0e-3) << "III U1 == Wilson";
    EXPECT_NEAR(iii.u2, wilson[i].u2, 1.0e-3) << "III U2 == Wilson";
    EXPECT_NEAR(iii.u3, wilson[i].u3, 1.0e-3) << "III U3 == Wilson";

    // A force-free inclusion in RPY flow moves at the RPY velocity (Faxen's law).
    EXPECT_NEAR(rir.u2, rpy.u2, 3.0e-3 * std::abs(rpy.u2)) << "RIR inclusion U2 == RPY";

    // A passive inclusion slows the forced sphere below RPY's 1, but not past Wilson's.
    EXPECT_GT(rir.u1, wilson[i].u1) << "RIR U1 above Wilson";
    EXPECT_LT(rir.u1, 1.0) << "RIR U1 below the RPY value";

    // A forced inclusion among force-free RPY spheres moves at its bare drag.
    EXPECT_NEAR(irr.u1, 1.0, 1.0e-3) << "IRR forced-inclusion bare drag";

    // At the widest separation, every configuration approaches Wilson's.
    if (i + 1 == separations.size()) {
      const std::array<std::pair<const char*, WilsonVelocities>, 4> configurations = {
          {{"RRR", rrr}, {"RIR", rir}, {"IRR", irr}, {"III", iii}}};
      for (const auto& [label, v] : configurations) {
        EXPECT_NEAR(v.u1, wilson[i].u1, 1.0e-2) << label << " U1 -> Wilson";
        EXPECT_NEAR(v.u2, wilson[i].u2, 1.0e-2) << label << " U2 -> Wilson";
        EXPECT_NEAR(v.u3, wilson[i].u3, 1.0e-2) << label << " U3 -> Wilson";
      }
    }
  }
}

//@}

//! \name The mobility system
//@{

// MobilitySystem reproduces the building blocks it wires together to roundoff, and a Dry sphere moves at its bare drag
// even inside a periphery.
TEST(Periphery, MobilitySystemMatchesBuildingBlocks) {
  const BelosConfig<double> cfg = make_gmres_config();
  const auto periphery = std::make_shared<SolvePeriphery>(
      make_periphery(make_sphere_surface<SolveExecSpace>(8, kCavityRadius, /*outward_normal=*/false), kViscosity));

  // A resolved body: MobilitySystem against solve_mobility.
  {
    const BodySet<SolveExecSpace> bodies = make_sphere_body_set<SolveExecSpace>(
        8, 1.0, Vector3d(0.4, -0.3, 0.2), kNoRotation, Vector3d(1.0, 0.0, 0.0), Vector3d(0.0, 0.0, 1.0));
    SolveVector velocity("velocity", 3);
    SolveVector angular_velocity("angular_velocity", 3);
    MobilitySystem<SolveExecSpace> mobility(kViscosity);
    mobility.set_periphery(periphery).set_belos_config(cfg).set_bodies(bodies, velocity, angular_velocity).solve();
    EXPECT_TRUE(mobility.body_solve_result().converged) << mobility.body_solve_result();

    const BodySolve reference = run_confined_body_solve(*periphery, bodies, SphereSet<SolveExecSpace>{}, cfg);
    const size_t offset = bodies.rigid_velocity_offset(0);
    const auto expected_velocity = Kokkos::subview(reference.x, Kokkos::pair<size_t, size_t>(offset, offset + 3));
    const auto expected_angular_velocity =
        Kokkos::subview(reference.x, Kokkos::pair<size_t, size_t>(offset + 3, offset + 6));
    EXPECT_LT(max_relative_difference(expected_velocity, velocity), 1.0e-12)
        << "MobilitySystem body velocity != solve_mobility";
    EXPECT_LT(max_relative_difference(expected_angular_velocity, angular_velocity), 1.0e-12)
        << "MobilitySystem body angular velocity != solve_mobility";
  }

  // Point spheres: the dry drag plus the periphery's no-slip response (RPY), and the bare drag alone (Dry).
  const SolveVector positions = to_device<SolveExecSpace>({0.5, 1.0, 1.5});
  const SolveVector radii = to_device<SolveExecSpace>({0.1});
  const SolveVector forces = to_device<SolveExecSpace>({0.3, -1.0, 0.6});
  SolveVector dry_drag("dry_drag", 3);
  apply_local_drag(SolveExecSpace{}, kViscosity, dry_drag, forces, radii);
  {
    SolveVector velocity("velocity", 3);
    MobilitySystem<SolveExecSpace> mobility(kViscosity);
    mobility.set_periphery(periphery).set_spheres(positions, radii, forces, velocity, SphereInteraction::RPY).solve();

    const size_t num_nodes = periphery->get_num_nodes();
    SolveVector slip("slip", 3 * num_nodes);
    apply_rpy_kernel(SolveExecSpace{}, kViscosity, positions, periphery->get_surface_positions(), radii,
                     SolveVector("node_radii", num_nodes), forces, slip);
    SolveVector surface_density("surface_density", 3 * num_nodes);
    periphery->compute_surface_density(slip, surface_density);
    SolveVector expected("expected", 3);
    Kokkos::deep_copy(expected, dry_drag);
    apply_stokes_double_layer_kernel(SolveExecSpace{}, kViscosity, num_nodes, 1, periphery->get_surface_positions(),
                                     positions, periphery->get_surface_normals(), periphery->get_quadrature_weights(),
                                     surface_density, expected);
    EXPECT_LT(max_relative_difference(expected, velocity), 1.0e-12)
        << "MobilitySystem RPY sphere velocity != drag + periphery response";
  }
  {
    SolveVector velocity("velocity", 3);
    MobilitySystem<SolveExecSpace> mobility(kViscosity);
    mobility.set_periphery(periphery).set_spheres(positions, radii, forces, velocity, SphereInteraction::Dry).solve();
    EXPECT_LT(max_relative_difference(dry_drag, velocity), 1.0e-15) << "a Dry sphere must move at its bare drag";
  }
}

// MobilitySystem and its inputs throw on inconsistent or incomplete inputs.
TEST(Periphery, MobilitySystemRejectsInvalidInputs) {
  SolveVector three_a("three_a", 3);
  SolveVector three_b("three_b", 3);
  SolveVector three_c("three_c", 3);
  SolveVector six("six", 6);
  SolveVector one("one", 1);

  // Mismatched sphere lengths.
  EXPECT_THROW((SphereSet<SolveExecSpace>(six, one, three_a, SphereInteraction::RPY)), std::invalid_argument);
  MobilitySystem<SolveExecSpace> mobility(kViscosity);
  EXPECT_THROW(mobility.set_spheres(three_a, one, three_b, six, SphereInteraction::RPY), std::invalid_argument);

  // An unwired BodySet, and its setters.
  BodySet<SolveExecSpace> unwired;
  unwired.bodies.push_back(make_sphere_body<SolveExecSpace>(4, 1.0));
  EXPECT_THROW(mobility.set_bodies(unwired, three_a, three_b), std::invalid_argument);
  EXPECT_THROW(unwired.set_center(0, Vector3d(0.0, 0.0, 0.0)), std::invalid_argument);

  // Mismatched body-velocity lengths.
  const BodySet<SolveExecSpace> wired =
      make_sphere_body_set<SolveExecSpace>(4, 1.0, Vector3d(0.0, 0.0, 0.0), kNoRotation);
  EXPECT_THROW(mobility.set_bodies(wired, six, three_b), std::invalid_argument);

  // No body solve has run yet.
  EXPECT_THROW(mobility.body_solve_result(), std::runtime_error);

  // A periphery without a dense inverse.
  const auto surface = make_sphere_surface<SolveExecSpace>(4, kCavityRadius, /*outward_normal=*/false);
  auto periphery = std::make_shared<SolvePeriphery>(surface.num_nodes, kViscosity);
  periphery->set_surface_positions(surface.points)
      .set_surface_normals(surface.normals, surface.outward_normal)
      .set_quadrature_weights(surface.weights);
  mobility.set_periphery(periphery).set_bodies(wired, three_b, three_c);
  EXPECT_THROW(mobility.solve(), std::runtime_error);

  // A periphery of another viscosity, whose M^{-1} carries that viscosity.
  EXPECT_THROW(MobilitySystem<SolveExecSpace>(1.0).set_periphery(periphery), std::invalid_argument);
}

//@}

#else

// The matrix-free inverse, the motile-body, and the mobility-system tests solve with Belos on Tpetra.
TEST(Periphery, MatrixFreeAndBodyTestsNeedBelosAndTpetra) {
  GTEST_SKIP() << "The matrix-free inverse, motile-body, and mobility-system tests require the Belos and Tpetra TPLs.";
}

#endif  // HAVE_MUNDYMATH_BELOS && HAVE_MUNDYMATH_TPETRA

}  // namespace

}  // namespace mbody

}  // namespace mundy
