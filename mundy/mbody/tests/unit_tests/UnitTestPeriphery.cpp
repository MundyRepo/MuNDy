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

// I realized that UnitTestPeriphery is kinda a mess. I'm the one who made it years ago and it lacks the current level
// of cleanlyness, organization, quality, and seperation of concerns seen throughout mundy. It also generally feels
// beyond verbose.

// External
#include <gmock/gmock.h>      // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <array>    // for std::array
#include <cstdio>   // for std::remove
#include <fstream>  // for std::ofstream
#include <iomanip>  // for std::setw, std::setprecision
#include <numeric>  // for std::accumulate
#include <vector>   // for std::vector

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_math/GaussLegendreSphere.hpp>  // for mundy::gauss_legendre_sphere_rule
#include <mundy_math/Vector3.hpp>              // for Vector3
#include <mundy_math/invert.hpp>               // for mundy::invert
#include <mundy_math/matrix_market.hpp>        // for mundy::read_matrix_market, mundy::write_matrix_market
#include <mundy_mbody/Periphery.hpp>           // for fill_skfie_matrix, apply_skfie, PeripheryT, ...
#include <mundy_utils/rng.hpp>                 // for mundy::make_philox

namespace mundy {

namespace mbody {

namespace {

//! \name Helper functions
//@{

/// \brief Compute the slope of a log-log plot of y(x) vs x
double compute_log_log_slope(const std::vector<double>& x, const std::vector<double>& y) {
  const size_t data_size = x.size();
  std::vector<double> log_x(data_size);
  std::vector<double> log_y(data_size);

  for (size_t i = 0; i < data_size; ++i) {
    assert(x[i] > 0.0 && "x values must be positive");
    log_x[i] = std::log(x[i]);
    log_y[i] = std::log(std::fabs(y[i]));  // Use fabs to avoid log of negative numbers
  }

  // Compute the slope of log-log plot (log_y vs log_x)
  double sum_log_x = std::accumulate(log_x.begin(), log_x.end(), 0.0);
  double sum_log_y = std::accumulate(log_y.begin(), log_y.end(), 0.0);
  double sum_log_x_log_y = 0.0;
  double sum_log_x_squared = 0.0;

  for (size_t i = 0; i < data_size; ++i) {
    sum_log_x_log_y += log_x[i] * log_y[i];
    sum_log_x_squared += log_x[i] * log_x[i];
  }

  double slope = (static_cast<double>(data_size) * sum_log_x_log_y - sum_log_x * sum_log_y) /
                 (static_cast<double>(data_size) * sum_log_x_squared - sum_log_x * sum_log_x);
  return slope;
}

double fd_l2_norm(const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& a) {
  const size_t num_elements = a.extent(0);
  double l2_norm = 0.0;
  for (size_t i = 0; i < num_elements; ++i) {
    l2_norm += a(i) * a(i);
  }
  return std::sqrt(l2_norm / static_cast<double>(num_elements));
}

double fd_l2_norm_difference(const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& a,
                             const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& b) {
  const size_t num_elements = a.extent(0);
  double l2_norm = 0.0;
  for (size_t i = 0; i < num_elements; ++i) {
    const double diff = a(i) - b(i);
    l2_norm += diff * diff;
  }
  return std::sqrt(l2_norm / static_cast<double>(num_elements));
}

template <int field_dim>
double spherical_l2_norm(const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& a,
                         const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights) {
  assert(a.extent(0) == field_dim * weights.extent(0));

  double l2_norm = 0.0;
  for (size_t i = 0; i < weights.extent(0); ++i) {
    for (size_t j = 0; j < field_dim; ++j) {
      l2_norm += a(i * field_dim + j) * a(i * field_dim + j) * weights(i);
    }
  }
  return std::sqrt(l2_norm);
}

/// \brief A function for generating a quadrature rule
using QuadGenerationFunc =
    std::function<std::array<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>, 3>(const int order)>;

/// \brief A function for generating a Kokkos vector based on a quadrature rule
using QuadVectorFunc = std::function<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>(
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights)>;

/// \brief A function that accepts a quadrature rule, an in field, and an out field.
using QuadInOutFunc = std::function<void(
    const double viscosity, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& in_field,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& out_field)>;

/// \brief A struct for storing the results of a convergence study
struct ConvergenceResults {
  std::vector<double> num_quadrature_points;
  std::vector<double> abs_error;
  double abs_slope;
  std::vector<double> rel_error;
  double rel_slope;
};

/// \brief Perform a convergence study against an expected result
ConvergenceResults perform_convergence_study(const double& viscosity, const QuadGenerationFunc& quad_gen,
                                             const QuadInOutFunc& func, const QuadVectorFunc& input_field_gen,
                                             const QuadVectorFunc& expected_results_gen) {
  std::vector<double> stashed_num_quadrature_points;
  std::vector<double> stashed_error_abs;
  std::vector<double> stashed_error_rel;
  for (int order = 2; order <= 24; order += 4) {
    // Generate the quadrature rule
    auto [points, weights, normals] = quad_gen(order);
    const size_t num_quadrature_points = weights.extent(0);

    // Fetch the input field and the expected results
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field =
        input_field_gen(points, normals, weights);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results =
        expected_results_gen(points, normals, weights);

    // Apply the function and compute the finite difference error
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> result_vector("result_vector",
                                                                               expected_results.extent(0));
    func(viscosity, points, normals, weights, input_field, result_vector);

    // Absolute error
    const double fd_l2_error = fd_l2_norm_difference(result_vector, expected_results);

    // Relative error
    const double fd_l2_norm_expected = fd_l2_norm(expected_results);
    const double l2_rel_error = fd_l2_error / fd_l2_norm_expected;

    // Stash the error
    stashed_error_abs.push_back(fd_l2_error);
    stashed_error_rel.push_back(l2_rel_error);
    stashed_num_quadrature_points.push_back(static_cast<double>(num_quadrature_points));
  }

  // Check that the error converges to zero at the expected rate
  const double slope_absolute = compute_log_log_slope(stashed_num_quadrature_points, stashed_error_abs);
  const double slope_relative = compute_log_log_slope(stashed_num_quadrature_points, stashed_error_rel);
  return {stashed_num_quadrature_points, stashed_error_abs, slope_absolute, stashed_error_rel, slope_relative};
}

/// \brief Perform a self-convergence study
/// Note, when we say "self-convergence study", we mean that we are comparing the error between a low-order and an
/// extremely high-order quadrature rule.
ConvergenceResults perform_self_convergence_study(const double& viscosity, const QuadGenerationFunc& quad_gen,
                                                  const QuadInOutFunc& func, const QuadVectorFunc& input_field_gen) {
  // Generate high-order quadrature rule
  auto [points_ho, weights_ho, normals_ho] = quad_gen(64);

  // Fetch the high_order input field and the corresponding expected results
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field_ho =
      input_field_gen(points_ho, normals_ho, weights_ho);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results_ho("expected_results_ho",
                                                                                   3 * weights_ho.extent(0));
  func(viscosity, points_ho, normals_ho, weights_ho, input_field_ho, expected_results_ho);
  const double expected_l2_norm = spherical_l2_norm<3>(expected_results_ho, weights_ho);

  // Perform the self-convergence study
  std::vector<double> stashed_num_quadrature_points;
  std::vector<double> stashed_error_abs;
  std::vector<double> stashed_error_rel;
  for (int order = 2; order <= 16; order *= 2) {
    // Generate the quadrature rule
    auto [points, weights, normals] = quad_gen(order);
    const size_t num_quadrature_points = weights.extent(0);

    // Fetch the input field
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field =
        input_field_gen(points, normals, weights);

    // Apply the function and compute the finite difference error
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> result_vector("result_vector",
                                                                               3 * num_quadrature_points);
    func(viscosity, points, normals, weights, input_field, result_vector);

    // It's not trivial to find the error between the expected and current results without resorting to interpolation
    // onto a common grid, as the locations of quadrature points for lower orders is not necessarily a subset of the
    // higher order quadrature points. So, we'll just look at the error between the l2 norm of the current results and
    // the high-order results. Note, this is the spherical l2 norm, not the Euclidean l2 norm.
    const double current_l2_norm = spherical_l2_norm<3>(result_vector, weights);
    const double l2_error = std::fabs(current_l2_norm - expected_l2_norm);
    const double l2_error_relative = l2_error / expected_l2_norm;

    // Stash the error
    stashed_error_abs.push_back(l2_error);
    stashed_error_rel.push_back(l2_error_relative);
    stashed_num_quadrature_points.push_back(static_cast<double>(num_quadrature_points));
  }

  // Check that the error converges to zero at the expected rate
  const double slope_absolute = compute_log_log_slope(stashed_num_quadrature_points, stashed_error_abs);
  const double slope_relative = compute_log_log_slope(stashed_num_quadrature_points, stashed_error_rel);
  return {stashed_num_quadrature_points, stashed_error_abs, slope_absolute, stashed_error_rel, slope_relative};
}

/// \brief Apply the stokes double layer matrix to surface forces to get surface velocities
void apply_stokes_double_layer_matrix_wrapper(
    const double viscosity, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_forces,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_velocities) {
  // Fill the stokes_double_layer_matrix
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> stokes_double_layer_matrix(
      "stokes_double_layer_matrix", 3 * num_quadrature_points, 3 * num_quadrature_points);
  fill_stokes_double_layer_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                  num_quadrature_points, points, points, normals, weights, stokes_double_layer_matrix);
  // Apply the stokes_double_layer_matrix to a the surface forces
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, stokes_double_layer_matrix, surface_forces, 0.0,
                   surface_velocities);
}

/// \brief Apply the periphery's dense singularity-subtracted interior trace (J + T) for inward normals
void apply_stokes_double_layer_matrix_ss_wrapper(
    const double viscosity, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_forces,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_velocities) {
  // Fill the stokes_double_layer_matrix and apply the singularity subtraction
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> stokes_double_layer_matrix(
      "stokes_double_layer_matrix", 3 * num_quadrature_points, 3 * num_quadrature_points);
  fill_stokes_double_layer_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                  num_quadrature_points, points, points, normals, weights, stokes_double_layer_matrix);
  add_singularity_subtraction(Kokkos::DefaultHostExecutionSpace(), viscosity, stokes_double_layer_matrix,
                              /*outward_normal=*/false);

  // Apply the stokes_double_layer_matrix to a the surface forces
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, stokes_double_layer_matrix, surface_forces, 0.0,
                   surface_velocities);
}

/// \brief Apply the stokes double layer operator to surface forces to get surface velocities
void apply_stokes_double_layer_kernel_wrapper(
    const double viscosity, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_forces,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_velocities) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_stokes_double_layer_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                   num_quadrature_points, points, points, normals, weights, surface_forces,
                                   surface_velocities);
}

/// \brief Apply the singularity-subtracted exterior trace (J + T) of a body surface (either orientation)
void apply_stokes_double_layer_kernel_ss_wrapper(
    const double viscosity, const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_forces,
    const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_velocities) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_stokes_double_layer_kernel_ss(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points, points,
                                      normals, weights, surface_forces, surface_velocities);
}

/// \brief Apply the periphery's dense second kind Fredholm integral operator for inward normals
void apply_skfie_matrix_wrapper(const double viscosity,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& input_field,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& output_field) {
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> M("M", 3 * num_quadrature_points,
                                                                  3 * num_quadrature_points);
  fill_skfie_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points, points, normals, weights, M,
                    /*outward_normal=*/false);
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, M, input_field, 0.0, output_field);
}

/// \brief Apply the periphery's matrix-free second kind Fredholm integral operator for inward normals
void apply_skfie_wrapper(const double viscosity,
                         const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
                         const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
                         const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights,
                         const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& input_field,
                         const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& output_field) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_skfie(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points, points, normals, weights,
              input_field, output_field, /*outward_normal=*/false);
}

/// \brief Solve for the velocity on a bulk point given an imposed slip velocity on the periphery
///
/// Mathematically U_bulk = G_{periphery -> bulk} * M^{-1} * U_slip
///
/// \param outward_normal Whether surface_normals point out of (true) or into (false) the periphery
void apply_resistance(const double viscosity,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_points,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_normals,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_weights,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_velocities,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& surface_forces,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& bulk_points,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& bulk_velocities,
                      const bool outward_normal) {
  const size_t num_surface_points = surface_weights.extent(0);
  const size_t num_bulk_points = bulk_points.extent(0) / 3;

  // Fill the SKFIE matrix M
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> M("M", 3 * num_surface_points, 3 * num_surface_points);
  fill_skfie_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_points, surface_points, surface_normals,
                    surface_weights, M, outward_normal);

  // Invert the SKFIE matrix
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> M_inv("M_inv", 3 * num_surface_points,
                                                                      3 * num_surface_points);
  mundy::invert(Kokkos::DefaultHostExecutionSpace(), M, M_inv);

  // F = M^{-1} * U_slip
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, M_inv, surface_velocities, 0.0, surface_forces);

  // U_bulk = G^{dl}_{periphery -> bulk} * F
  apply_stokes_double_layer_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_points, num_bulk_points,
                                   surface_points, bulk_points, surface_normals, surface_weights, surface_forces,
                                   bulk_velocities);
}

/// \brief A sphere's quadrature: points, weights, and unit normals.
struct SphereQuadrature {
  std::vector<double> points;
  std::vector<double> weights;
  std::vector<double> normals;
};

/// \brief The (order + 1)-ring Gauss-Legendre sphere rule scaled to a radius, with unit normals pointing out or in.
SphereQuadrature sphere_quadrature(const int order, const double radius, const bool outward_normal) {
  SphereQuadrature quadrature;
  gauss_legendre_sphere_rule(order + 1, quadrature.normals, quadrature.weights);  // unit points = outward normals
  quadrature.points.resize(quadrature.normals.size());
  const double sign = outward_normal ? 1.0 : -1.0;
  for (size_t i = 0; i < quadrature.normals.size(); ++i) {
    quadrature.points[i] = radius * quadrature.normals[i];
    quadrature.normals[i] *= sign;
  }
  for (double& weight : quadrature.weights) {
    weight *= radius * radius;
  }
  return quadrature;
}

/// \brief A functor for generating a quadrature rule on the sphere
class SphereQuadFunctor {
 public:
  explicit SphereQuadFunctor(const double sphere_radius, const bool outward_normal = true)
      : sphere_radius_(sphere_radius), outward_normal_(outward_normal) {
  }

  std::array<Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>, 3> operator()(const int& order) {
    const auto [points_vec, weights_vec, normals_vec] = sphere_quadrature(order, sphere_radius_, outward_normal_);
    const size_t num_quadrature_points = weights_vec.size();

    // Convert the points, weights, and normals to Kokkos views
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> points("points", num_quadrature_points * 3);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> weights("weights", num_quadrature_points);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> normals("normals", num_quadrature_points * 3);
    for (size_t i = 0; i < num_quadrature_points; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        points(3 * i + j) = points_vec[3 * i + j];
        normals(3 * i + j) = normals_vec[3 * i + j];
      }
      weights(i) = weights_vec[i];
    }

    return {points, weights, normals};
  }

 private:
  double sphere_radius_;
  bool outward_normal_;
};  // class SphereQuadFunctor

//@}

//! \name Periphery auxilary function tests
//@{

TEST(PeripheryTest, FileSettersMatchViewSetters) {
  // The file setters read Matrix Market files exactly: a periphery set from files matches one set from views bit for
  // bit, and a written inverse reads back bit for bit.
  using HostPeripheryType = PeripheryT<Kokkos::DefaultHostExecutionSpace>;
  const double viscosity = 0.5305;
  const bool outward_normal = false;
  auto [points, weights, normals] = SphereQuadFunctor(1.7, outward_normal)(6);
  const size_t num_surface_nodes = weights.extent(0);
  const std::string prefix = "FileSettersMatchViewSetters_";
  mundy::write_matrix_market(prefix + "points.mtx", points);
  mundy::write_matrix_market(prefix + "normals.mtx", normals);
  mundy::write_matrix_market(prefix + "weights.mtx", weights);

  HostPeripheryType from_views(num_surface_nodes, viscosity);
  from_views.set_surface_positions(points)
      .set_surface_normals(normals, outward_normal)
      .set_quadrature_weights(weights)
      .build_inverse_self_interaction_matrix();
  HostPeripheryType from_files(num_surface_nodes, viscosity);
  from_files.set_surface_positions(prefix + "points.mtx")
      .set_surface_normals(prefix + "normals.mtx", outward_normal)
      .set_quadrature_weights(prefix + "weights.mtx")
      .build_inverse_self_interaction_matrix();

  from_views.write_inverse_self_interaction_matrix(prefix + "M_inv.mtx");
  HostPeripheryType from_inverse_file(num_surface_nodes, viscosity);
  from_inverse_file.set_inverse_self_interaction_matrix(prefix + "M_inv.mtx");

  const auto& expected = from_views.get_M_inv();
  size_t file_mismatches = 0;
  size_t inverse_file_mismatches = 0;
  for (size_t i = 0; i < expected.extent(0); ++i) {
    for (size_t j = 0; j < expected.extent(1); ++j) {
      file_mismatches += from_files.get_M_inv()(i, j) != expected(i, j);
      inverse_file_mismatches += from_inverse_file.get_M_inv()(i, j) != expected(i, j);
    }
  }
  EXPECT_EQ(file_mismatches, 0u) << "entries of M^{-1} that differ when the geometry is read from files";
  EXPECT_EQ(inverse_file_mismatches, 0u) << "entries of M^{-1} that differ after a write and read";

  // A file of the wrong length is rejected.
  HostPeripheryType larger(num_surface_nodes + 1, viscosity);
  EXPECT_THROW(larger.set_surface_positions(prefix + "points.mtx"), std::runtime_error);
  EXPECT_THROW(larger.set_inverse_self_interaction_matrix(prefix + "M_inv.mtx"), std::runtime_error);
  for (const char* name : {"points.mtx", "normals.mtx", "weights.mtx", "M_inv.mtx"}) {
    std::remove((prefix + name).c_str());
  }
}

// DIAGNOSTIC (prints for inspection; no pass/fail assertions).
TEST(PeripheryDiagnostic, RPYKernelThreeSpheres) {
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

  std::vector<double> s = {2.01, 2.10, 2.50, 3.00, 4.00, 6.00};
  std::vector<double> U1 = {0.65528, 0.73857, 0.87765, 0.93905, 0.97964, 0.99581};
  std::vector<double> U2 = {0.63461, 0.59718, 0.49545, 0.41694, 0.31859, 0.21586};
  std::vector<double> U3 = {0.00498, 0.03517, 0.07393, 0.07824, 0.06925, 0.05078};
  std::vector<double> Omega3 = {0.037336, 0.052035, 0.045466, 0.035022, 0.021634, 0.010159};

  const double viscosity = 1.0;
  const double radius = 12.34;
  for (size_t i = 0; i < s.size(); i++) {
    const double s_current = s[i];
    const double U1_current = U1[i];
    const double U2_current = U2[i];
    const double U3_current = U3[i];
    const double Omega3_current = Omega3[i];

    // Setup the kokkos vectors for position, radius, force, and velocity
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> positions("positions", 9);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> radii("radii", 3);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> forces("forces", 9);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpy("velocities_rpy", 9);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpyc("velocities_rpyc", 9);

    Kokkos::deep_copy(forces, 0.0);
    Kokkos::deep_copy(velocities_rpy, 0.0);
    Kokkos::deep_copy(velocities_rpyc, 0.0);
    Kokkos::deep_copy(radii, radius);

    // s1:
    positions(0) = 0.0;
    positions(1) = 0.0;
    positions(2) = 0.0;
    forces(1) = -6.0 * M_PI * radius;

    // s2:
    positions(3) = -s_current * radius / 2.0;
    positions(4) = -std::sqrt(3.0) * s_current * radius / 2.0;
    positions(5) = 0.0;

    // s3:
    positions(6) = s_current * radius / 2.0;
    positions(7) = -std::sqrt(3.0) * s_current * radius / 2.0;
    positions(8) = 0.0;

    // Apply the RPY kernel
    apply_rpy_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, radii, radii, forces,
                     velocities_rpy);
    apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, radii, radii, forces,
                      velocities_rpyc);

    // Add self-interaction
    const double inv_drag_coeff = 1.0 / (6.0 * M_PI * radius * viscosity);
    for (size_t j = 0; j < 9; j++) {
      velocities_rpy(j) += inv_drag_coeff * forces(j);
      velocities_rpyc(j) += inv_drag_coeff * forces(j);
    }

    // Check the results
    std::cout << "s: " << s_current << std::endl;
    std::cout << "RPY:" << std::endl;
    std::cout << "  U1: " << velocities_rpy(1) << " | expected: " << -U1_current
              << ") | relative error: " << std::fabs(velocities_rpy(1) + U1_current) / std::fabs(U1_current)
              << std::endl;
    std::cout << "  U2: " << velocities_rpy(4) << " | expected: " << -U2_current
              << ") | relative error: " << std::fabs(velocities_rpy(4) + U2_current) / std::fabs(U2_current)
              << std::endl;
    std::cout << "  U3: " << velocities_rpy(6) << " | expected: " << U3_current
              << ") | relative error: " << std::fabs(velocities_rpy(6) - U3_current) / std::fabs(U3_current)
              << std::endl;

    std::cout << "RPYC:" << std::endl;
    std::cout << "  U1: " << velocities_rpyc(1) << " | expected: " << -U1_current
              << ") | relative error: " << std::fabs(velocities_rpyc(1) + U1_current) / std::fabs(U1_current)
              << std::endl;
    std::cout << "  U2: " << velocities_rpyc(4) << " | expected: " << -U2_current
              << ") | relative error: " << std::fabs(velocities_rpyc(4) + U2_current) / std::fabs(U2_current)
              << std::endl;
    std::cout << "  U3: " << velocities_rpyc(6) << " | expected: " << U3_current
              << ") | relative error: " << std::fabs(velocities_rpyc(6) - U3_current) / std::fabs(U3_current)
              << std::endl;
  }
}

TEST(PeripheryTest, OverlappingRpySpheres) {
  // Fig 1 of Rotne–Prager–Yamakawa approximation for different-sized particles
  //  in application to macromolecular bead models JFM Rapids 2014
  //
  // Compute the parallel and perpendicular coefficients for two overlapping spheres with radii
  //  a2 = a1 / 2.
  //
  // Move the spheres increasingly closer to one another and measure parallel and perpendicular
  // components of the mobility matrix of 1 acting on 2.
  //
  // Because we use a kernel and not a matrix, we can get this my applying a 3 different unit forces to sphere 2
  // and measuring the velocity of sphere 1 in each direction.

  const double viscosity = 0.1;
  const double radius1 = 1.0;
  const double radius2 = radius1 / 2.0;
  const double dr = 0.001;
  const size_t num_points = 2000;

  std::vector<double> coeff_para_rpy(num_points);
  std::vector<double> coeff_perp_rpy(num_points);
  std::vector<double> coeff_para_rpyc(num_points);
  std::vector<double> coeff_perp_rpyc(num_points);
  std::vector<double> coeff_para_stokes(num_points);
  std::vector<double> coeff_perp_stokes(num_points);

  // Setup the kokkos vectors for position, radius, force, and velocity
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> positions("positions", 6);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> radii("radii", 2);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> forces("forces", 6);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpy("velocities_rpy", 6);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpyc("velocities_rpyc", 6);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_stokes("velocities_stokes", 6);

  // Randomize the position of the first sphere and choose a random r_hat
  positions(0) = static_cast<double>(rand()) / RAND_MAX;
  positions(1) = static_cast<double>(rand()) / RAND_MAX;
  positions(2) = static_cast<double>(rand()) / RAND_MAX;
  std::vector<double> r_hat = {static_cast<double>(rand()) / RAND_MAX, static_cast<double>(rand()) / RAND_MAX,
                               static_cast<double>(rand()) / RAND_MAX};

  double r_hat_mag = std::sqrt(r_hat[0] * r_hat[0] + r_hat[1] * r_hat[1] + r_hat[2] * r_hat[2]);
  r_hat[0] /= r_hat_mag;
  r_hat[1] /= r_hat_mag;
  r_hat[2] /= r_hat_mag;
  ASSERT_NEAR(r_hat[0] * r_hat[0] + r_hat[1] * r_hat[1] + r_hat[2] * r_hat[2], 1.0, 1.0e-8);

  // Generate a vector perpendicular to r_hat. statistically speaking the following should be perpendicular
  std::vector<double> r_hat_perp = {r_hat[1], -r_hat[0], 0.0};
  ASSERT_NEAR(r_hat_perp[0] * r_hat[0] + r_hat_perp[1] * r_hat[1] + r_hat_perp[2] * r_hat[2], 0.0, 1.0e-8);

  // Set the radii
  radii(0) = radius1;
  radii(1) = radius2;

  for (size_t p = 0; p < num_points; p++) {
    Kokkos::deep_copy(forces, 0.0);
    Kokkos::deep_copy(velocities_rpy, 0.0);
    Kokkos::deep_copy(velocities_rpyc, 0.0);
    Kokkos::deep_copy(velocities_stokes, 0.0);

    positions(3) = positions(0) + r_hat[0] * dr * p;
    positions(4) = positions(1) + r_hat[1] * dr * p;
    positions(5) = positions(2) + r_hat[2] * dr * p;

    // Parallel force
    forces(3) = r_hat[0];
    forces(4) = r_hat[1];
    forces(5) = r_hat[2];

    // Compute the velocities:
    // No need to add self-interaction as we are only interested in the flow of sphere 1
    apply_rpy_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, radii, radii, forces,
                     velocities_rpy);
    apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, radii, radii, forces,
                      velocities_rpyc);
    apply_stokes_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, forces,
                        velocities_stokes);

    // Compute the parallel and component of the mobility matrix by dotting with r_hat.
    // The velocity should lie entirely along r_hat.
    //
    // RPY
    coeff_para_rpy[p] = velocities_rpy(0) * r_hat[0] + velocities_rpy(1) * r_hat[1] + velocities_rpy(2) * r_hat[2];

    // RPYC:
    coeff_para_rpyc[p] = velocities_rpyc(0) * r_hat[0] + velocities_rpyc(1) * r_hat[1] + velocities_rpyc(2) * r_hat[2];

    // Stokes:
    coeff_para_stokes[p] =
        velocities_stokes(0) * r_hat[0] + velocities_stokes(1) * r_hat[1] + velocities_stokes(2) * r_hat[2];

    // Zero the velocity and switch to a perpendicular force
    Kokkos::deep_copy(velocities_rpy, 0.0);
    Kokkos::deep_copy(velocities_rpyc, 0.0);
    Kokkos::deep_copy(velocities_stokes, 0.0);
    forces(3) = r_hat_perp[0];
    forces(4) = r_hat_perp[1];
    forces(5) = r_hat_perp[2];

    // Compute the velocities:
    apply_rpy_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, radii, radii, forces,
                     velocities_rpy);
    apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, radii, radii, forces,
                      velocities_rpyc);
    apply_stokes_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, positions, positions, forces,
                        velocities_stokes);

    // Compute the perpendicular component of the mobility matrix by dotting with r_hat_perp.
    // The velocity should lie entirely along r_hat_perp.
    //
    // RPY
    coeff_perp_rpy[p] =
        velocities_rpy(0) * r_hat_perp[0] + velocities_rpy(1) * r_hat_perp[1] + velocities_rpy(2) * r_hat_perp[2];

    // RPYC:
    coeff_perp_rpyc[p] =
        velocities_rpyc(0) * r_hat_perp[0] + velocities_rpyc(1) * r_hat_perp[1] + velocities_rpyc(2) * r_hat_perp[2];

    // Stokes:
    coeff_perp_stokes[p] = velocities_stokes(0) * r_hat_perp[0] + velocities_stokes(1) * r_hat_perp[1] +
                           velocities_stokes(2) * r_hat_perp[2];
  }

  // Write the results to a file
  // Use the same normalization as the paper.
  //  The coefficients are scaled by 6 pi mu (a1 + a2)
  //  The distance is presented as (dr * p) / (a1 + a2)
  const double normalization_factor = 6.0 * M_PI * viscosity * (radius1 + radius2);
  std::ofstream file("OverlappingRpySpheres.dat");
  file << "dr coeff_para_rpy coeff_perp_rpy coeff_para_rpyc coeff_perp_rpyc coeff_para_stokes coeff_perp_stokes"
       << std::endl;
  for (size_t p = 0; p < num_points; p++) {
    file << dr * p / (radius1 + radius2)                        //
         << " " << coeff_para_rpy[p] * normalization_factor     //
         << " " << coeff_perp_rpy[p] * normalization_factor     //
         << " " << coeff_para_rpyc[p] * normalization_factor    //
         << " " << coeff_perp_rpyc[p] * normalization_factor    //
         << " " << coeff_para_stokes[p] * normalization_factor  //
         << " " << coeff_perp_stokes[p] * normalization_factor << std::endl;
  }
  file.close();
}

// DIAGNOSTIC (prints convergence for inspection; no pass/fail assertions -- the bare double layer of a constant force
// excites the singular diagonal, so its convergence is slow/algebraic and no tight tolerance is meaningful; the
// singularity-subtracted operators are exact for a constant force and are asserted in SkfieConstantDensityJump).
TEST(PeripheryDiagnostic, StokesDoubleLayerConstantForce) {
  // For x on a closed surface D whose normals n point into the enclosed region (the periphery), with T(x - y) the
  // double layer kernel:
  //   PV int_{partial D} n(y) dot T(x - y) dS(y) = -I / (2 viscosity)
  // For the singularity-subtracted interior trace (J + T) with J = -I / (2 viscosity),
  //    (J + T)[F](x) = int_{partial D} n(y) dot T(x - y) dot (F(y) - F(x)) dS(y) + (J + PV T[1](x)) F(x),
  //    J + PV T[1](x) = -I / viscosity,
  // so a constant F makes the integrand vanish and (J + T)[F] = -F / viscosity. The complementary term
  // N[F] = n (F . sum_s w_s n_s) / (viscosity |D|) vanishes for a constant F, so the SKFIE also returns -F / viscosity.

  // Setup the convergence study
  const double sphere_radius = 12.34;
  const double viscosity = 0.5305;

  const QuadGenerationFunc quad_gen =
      SphereQuadFunctor(sphere_radius, /*outward_normal=*/false);
  const QuadVectorFunc in_field_gen =
      []([[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
         [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
         [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights) {
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field("input_field", 3 * weights.extent(0));
        Kokkos::deep_copy(input_field, 1.0);
        return input_field;
      };
  const QuadVectorFunc expected_results_gen =
      [viscosity]([[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
                  [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
                  [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights) {
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("expected_results",
                                                                                      3 * weights.extent(0));
        Kokkos::deep_copy(expected_results, -0.5 / viscosity);
        return expected_results;
      };

  // Run the convergence study for the stokes double layer matrix
  const ConvergenceResults matrix_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_matrix_wrapper, in_field_gen, expected_results_gen);

  // Run the same convergence study for the stokes double layer kernel
  const ConvergenceResults kernel_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_kernel_wrapper, in_field_gen, expected_results_gen);

  // Run the convergence study for the stokes double layer matrix with singularity subtraction
  const QuadVectorFunc new_expected_results_gen =
      [viscosity]([[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
                  [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
                  [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights) {
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("new_expected_results",
                                                                                      3 * weights.extent(0));
        Kokkos::deep_copy(expected_results, -1.0 / viscosity);
        return expected_results;
      };
  const ConvergenceResults matrix_ss_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_matrix_ss_wrapper, in_field_gen, new_expected_results_gen);

  const ConvergenceResults matrix_skfie_results = perform_convergence_study(
      viscosity, quad_gen, apply_skfie_matrix_wrapper, in_field_gen, new_expected_results_gen);

  const ConvergenceResults kernel_skfie_results =
      perform_convergence_study(viscosity, quad_gen, apply_skfie_wrapper, in_field_gen, new_expected_results_gen);

  // For now, just print the results
  std::cout << "matrix_results.abs_slope = " << matrix_results.abs_slope << " rel_slope = " << matrix_results.rel_slope
            << std::endl;
  for (size_t i = 0; i < matrix_results.num_quadrature_points.size(); ++i) {
    std::cout << "  num_quadrature_points[" << i << "] = " << matrix_results.num_quadrature_points[i];
    std::cout << ", abs_error[" << i << "] = " << matrix_results.abs_error[i];
    std::cout << ", rel_error[" << i << "] = " << matrix_results.rel_error[i] << std::endl;
  }
  std::cout << "kernel_results.abs_slope = " << kernel_results.abs_slope << " rel_slope = " << kernel_results.rel_slope
            << std::endl;
  for (size_t i = 0; i < kernel_results.num_quadrature_points.size(); ++i) {
    std::cout << "  num_quadrature_points[" << i << "] = " << kernel_results.num_quadrature_points[i];
    std::cout << ", abs_error[" << i << "] = " << kernel_results.abs_error[i];
    std::cout << ", rel_error[" << i << "] = " << kernel_results.rel_error[i] << std::endl;
  }
  std::cout << "matrix_ss_results.abs_slope = " << matrix_ss_results.abs_slope
            << " rel_slope = " << matrix_ss_results.rel_slope << std::endl;
  for (size_t i = 0; i < matrix_ss_results.num_quadrature_points.size(); ++i) {
    std::cout << "  num_quadrature_points[" << i << "] = " << matrix_ss_results.num_quadrature_points[i];
    std::cout << ", abs_error[" << i << "] = " << matrix_ss_results.abs_error[i];
    std::cout << ", rel_error[" << i << "] = " << matrix_ss_results.rel_error[i] << std::endl;
  }
  std::cout << "matrix_skfie_results.abs_slope = " << matrix_skfie_results.abs_slope
            << " rel_slope = " << matrix_skfie_results.rel_slope << std::endl;
  for (size_t i = 0; i < matrix_skfie_results.num_quadrature_points.size(); ++i) {
    std::cout << "  num_quadrature_points[" << i << "] = " << matrix_skfie_results.num_quadrature_points[i];
    std::cout << ", abs_error[" << i << "] = " << matrix_skfie_results.abs_error[i];
    std::cout << ", rel_error[" << i << "] = " << matrix_skfie_results.rel_error[i] << std::endl;
  }
  std::cout << "kernel_skfie_results.abs_slope = " << kernel_skfie_results.abs_slope
            << " rel_slope = " << kernel_skfie_results.rel_slope << std::endl;
  for (size_t i = 0; i < kernel_skfie_results.num_quadrature_points.size(); ++i) {
    std::cout << "  num_quadrature_points[" << i << "] = " << kernel_skfie_results.num_quadrature_points[i];
    std::cout << ", abs_error[" << i << "] = " << kernel_skfie_results.abs_error[i];
    std::cout << ", rel_error[" << i << "] = " << kernel_skfie_results.rel_error[i] << std::endl;
  }
}

// DIAGNOSTIC (prints convergence for inspection; no pass/fail assertions -- convergence is mode/operator
// dependent, ranging from machine-exact to non-monotone, so no single tolerance is meaningful).
TEST(PeripheryDiagnostic, StokesDoubleLayerSmoothForces) {
  // Self-convergence against the integral of a smooth function
  // In this case, we use the smooth function
  auto run_test_for_function =
      [](const std::string& function_name,
         const std::function<std::array<double, 3>(double, double, double)>& func_to_integrate) {
        // Setup the convergence study (inward normals, the periphery's convention; kernel_ss is the exterior trace)
        const double sphere_radius = 12.34;
        const double viscosity = 0.5305;

        const QuadGenerationFunc quad_gen =
            SphereQuadFunctor(sphere_radius, /*outward_normal=*/false);
        const QuadVectorFunc in_field_gen =
            [&func_to_integrate](
                [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& points,
                [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& normals,
                [[maybe_unused]] const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& weights) {
              Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field("input_field",
                                                                                       3 * weights.extent(0));
              for (size_t i = 0; i < weights.extent(0); ++i) {
                const double x = points(3 * i);
                const double y = points(3 * i + 1);
                const double z = points(3 * i + 2);
                const auto result = func_to_integrate(x, y, z);
                input_field(3 * i) = result[0];
                input_field(3 * i + 1) = result[1];
                input_field(3 * i + 2) = result[2];
              }
              return input_field;
            };

        const ConvergenceResults matrix_results =
            perform_self_convergence_study(viscosity, quad_gen, apply_stokes_double_layer_matrix_wrapper, in_field_gen);

        const ConvergenceResults kernel_results =
            perform_self_convergence_study(viscosity, quad_gen, apply_stokes_double_layer_kernel_wrapper, in_field_gen);

        const ConvergenceResults matrix_ss_results = perform_self_convergence_study(
            viscosity, quad_gen, apply_stokes_double_layer_matrix_ss_wrapper, in_field_gen);

        const ConvergenceResults kernel_ss_results = perform_self_convergence_study(
            viscosity, quad_gen, apply_stokes_double_layer_kernel_ss_wrapper, in_field_gen);

        const ConvergenceResults matrix_skfie_results =
            perform_self_convergence_study(viscosity, quad_gen, apply_skfie_matrix_wrapper, in_field_gen);

        const ConvergenceResults kernel_skfie_results =
            perform_self_convergence_study(viscosity, quad_gen, apply_skfie_wrapper, in_field_gen);

        // For now, just print the results
        std::cout << "###########################################################" << std::endl;
        std::cout << "function_name = " << function_name << std::endl;
        std::cout << "matrix_results.abs_slope = " << matrix_results.abs_slope
                  << " rel_slope = " << matrix_results.rel_slope << std::endl;
        for (size_t i = 0; i < matrix_results.num_quadrature_points.size(); ++i) {
          std::cout << "  num_quadrature_points[" << i << "] = " << matrix_results.num_quadrature_points[i];
          std::cout << ", abs_error[" << i << "] = " << matrix_results.abs_error[i];
          std::cout << ", rel_error[" << i << "] = " << matrix_results.rel_error[i] << std::endl;
        }
        std::cout << "kernel_results.abs_slope = " << kernel_results.abs_slope
                  << " rel_slope = " << kernel_results.rel_slope << std::endl;
        for (size_t i = 0; i < kernel_results.num_quadrature_points.size(); ++i) {
          std::cout << "  num_quadrature_points[" << i << "] = " << kernel_results.num_quadrature_points[i];
          std::cout << ", abs_error[" << i << "] = " << kernel_results.abs_error[i];
          std::cout << ", rel_error[" << i << "] = " << kernel_results.rel_error[i] << std::endl;
        }
        std::cout << "matrix_ss_results.abs_slope = " << matrix_ss_results.abs_slope
                  << " rel_slope = " << matrix_ss_results.rel_slope << std::endl;
        for (size_t i = 0; i < matrix_ss_results.num_quadrature_points.size(); ++i) {
          std::cout << "  num_quadrature_points[" << i << "] = " << matrix_ss_results.num_quadrature_points[i];
          std::cout << ", abs_error[" << i << "] = " << matrix_ss_results.abs_error[i];
          std::cout << ", rel_error[" << i << "] = " << matrix_ss_results.rel_error[i] << std::endl;
        }
        std::cout << "kernel_ss_results.abs_slope = " << kernel_ss_results.abs_slope
                  << " rel_slope = " << kernel_ss_results.rel_slope << std::endl;
        for (size_t i = 0; i < kernel_ss_results.num_quadrature_points.size(); ++i) {
          std::cout << "  num_quadrature_points[" << i << "] = " << kernel_ss_results.num_quadrature_points[i];
          std::cout << ", abs_error[" << i << "] = " << kernel_ss_results.abs_error[i];
          std::cout << ", rel_error[" << i << "] = " << kernel_ss_results.rel_error[i] << std::endl;
        }
        std::cout << "matrix_skfie_results.abs_slope = " << matrix_skfie_results.abs_slope
                  << " rel_slope = " << matrix_skfie_results.rel_slope << std::endl;
        for (size_t i = 0; i < matrix_skfie_results.num_quadrature_points.size(); ++i) {
          std::cout << "  num_quadrature_points[" << i << "] = " << matrix_skfie_results.num_quadrature_points[i];
          std::cout << ", abs_error[" << i << "] = " << matrix_skfie_results.abs_error[i];
          std::cout << ", rel_error[" << i << "] = " << matrix_skfie_results.rel_error[i] << std::endl;
        }
        std::cout << "kernel_skfie_results.abs_slope = " << kernel_skfie_results.abs_slope
                  << " rel_slope = " << kernel_skfie_results.rel_slope << std::endl;
        for (size_t i = 0; i < kernel_skfie_results.num_quadrature_points.size(); ++i) {
          std::cout << "  num_quadrature_points[" << i << "] = " << kernel_skfie_results.num_quadrature_points[i];
          std::cout << ", abs_error[" << i << "] = " << kernel_skfie_results.abs_error[i];
          std::cout << ", rel_error[" << i << "] = " << kernel_skfie_results.rel_error[i] << std::endl;
        }
      };  // run_test_for_function

  run_test_for_function("f(x, y, z) = (1, 1, 1)", [](double, double, double) { return std::array{1.0, 1.0, 1.0}; });

  run_test_for_function("f(x, y, z) = nhat(x, y, z)", [](double x, double y, double z) {
    const double inv_radius = 1.0 / std::sqrt(x * x + y * y + z * z);
    return std::array{x * inv_radius, y * inv_radius, z * inv_radius};
  });

  run_test_for_function("f(x, y, z) = sin(theta) * nhat + cos(theta) * tangent_theta + cos^2(theta) tangent_phi",
                        [](double x, double y, double z) {
                          const double inv_radius = 1.0 / std::sqrt(x * x + y * y + z * z);
                          const double theta = std::acos(z * inv_radius);
                          const double phi = std::atan2(y, x);
                          const double sin_theta = std::sin(theta);
                          const double cos_theta = std::cos(theta);
                          const double sin_phi = std::sin(phi);
                          const double cos_phi = std::cos(phi);
                          return std::array{sin_theta * z * inv_radius + cos_theta * cos_phi * inv_radius,
                                            sin_theta * y * inv_radius - cos_theta * sin_phi * inv_radius,
                                            sin_theta * x * inv_radius};
                        });
}

// [TEMP] Quick test only: read periphery quadrature straight from hfirouznia's dir (icosahedron + octahedron
// families, all available sizes), ordered finest-first. Delete this block and revert the from-file tests after.
// The files are read as Matrix Market arrays, so each needs the banner "%%MatrixMarket matrix array real general"
// and the extents line "n 1" in place of its leading count.
struct TempQuad {
  size_t n;
  std::string label, pts, nrm, wgt;
};
inline std::vector<TempQuad> temp_quad_files() {
  const std::string base =
      "/mnt/home/hfirouznia/Desktop/Chromatin_Code-master/Surface_quadrature_points/quadrature_error_analysis/"
      "dat_files/R135e-1";
  std::vector<TempQuad> v;
  for (size_t n : {5120u, 3840u, 960u, 240u}) {  // [TEMP] capped at 5120 (dense inverse cost)
    const std::string d = base + "/icosahedron/quad_", s = "_R135_" + std::to_string(n) + ".dat";
    v.push_back({n, "ico_" + std::to_string(n), d + "pos" + s, d + "normvec" + s, d + "weight" + s});
  }
  for (size_t n : {1536u, 384u}) {
    const std::string d = base + "/octahedron/quad_", s = "_R135e-1_N" + std::to_string(n) + ".dat";
    v.push_back({n, "octa_" + std::to_string(n), d + "pos" + s, d + "normvec" + s, d + "weight" + s});
  }
  return v;
}

// [TEMP] Orientation of the normals in temp_quad_files(); checked against the geometry when the periphery is built.
constexpr bool temp_quad_files_outward_normal = true;

// DIAGNOSTIC (prints self-convergence for inspection; no pass/fail assertions). [TEMP] reads hfirouznia's
// quadrature files -- see temp_quad_files().
TEST(PeripheryDiagnostic, SKFIESelfConvFromFile) {
  // Test the convergence for the flow induced by N points within the bulk of the sphere
  // Assign to each point in the bulk a smoothly varying force field
  // f(x,y,z) = (y, -x, z) / sqrt(x^2 + y^2 + z^2)

  // Setup the convergence study
  const double sphere_radius = 13.5;
  const double viscosity = 0.5305;
  const size_t num_bulk_points = 1000;

  // Setup the bulk points at the corner of the cube with side length cude_side_length
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_points("bulk_points", 3 * num_bulk_points);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_forces("bulk_forces", 3 * num_bulk_points);
  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng = make_philox(seed, counter);
  for (size_t i = 0; i < num_bulk_points; ++i) {
    const double theta = rng.uniform(0.0, 2.0 * M_PI);
    const double phi = rng.uniform(0.0, M_PI);
    const double r = rng.uniform(0.0, 0.9 * sphere_radius - 1e-12);  // Avoid landing on the surface
    bulk_points(3 * i) = r * std::sin(phi) * std::cos(theta);
    bulk_points(3 * i + 1) = r * std::sin(phi) * std::sin(theta);
    bulk_points(3 * i + 2) = r * std::cos(phi);

    const double x = bulk_points(3 * i);
    const double y = bulk_points(3 * i + 1);
    const double z = bulk_points(3 * i + 2);
    const double inv_radius = 1.0 / std::sqrt(x * x + y * y + z * z);
    bulk_forces(3 * i) = y * inv_radius;
    bulk_forces(3 * i + 1) = -x * inv_radius;
    bulk_forces(3 * i + 2) = z * inv_radius;
  }

  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_bulk_velocity("expected_bulk_velocity",
                                                                                      3 * num_bulk_points);

  std::cout << "###########################################################" << std::endl;
  std::cout << "Self-conv from file" << std::endl;
  const auto quad_files = temp_quad_files();  // [TEMP] both families, all available sizes (finest first)
  for (size_t i = 0; i < quad_files.size(); ++i) {
    // Read in the normals, points, and weights to kokkos views
    const size_t num_quad_points = quad_files[i].n;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> normals("normal", 3 * num_quad_points);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> points("points", 3 * num_quad_points);
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> weights("weights", num_quad_points);
    mundy::read_matrix_market(quad_files[i].nrm, normals);
    mundy::read_matrix_market(quad_files[i].pts, points);
    mundy::read_matrix_market(quad_files[i].wgt, weights);

    // Compute the surface slip velocity induced by the bulk forces
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_slip_velocity("surface_slip_velocity",
                                                                                       3 * num_quad_points);
    apply_stokes_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, bulk_points, points, bulk_forces,
                        surface_slip_velocity);

    // Use apply_resistance to map the surface velocities to the bulk surface velocities
    // The surface forces are unknown and will be computed by apply_resistance
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_forces("surface_forces", points.extent(0));
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_velocity("bulk_velocity", 3 * num_bulk_points);
    Kokkos::deep_copy(surface_forces, 0.0);
    Kokkos::deep_copy(bulk_velocity, 0.0);
    apply_resistance(viscosity, points, normals, weights, surface_slip_velocity, surface_forces, bulk_points,
                     bulk_velocity, temp_quad_files_outward_normal);

    // Save the self-conv results
    if (i == 0) {
      Kokkos::deep_copy(expected_bulk_velocity, bulk_velocity);
    }

    // Absolute error
    const double fd_l2_error = fd_l2_norm_difference(bulk_velocity, expected_bulk_velocity);

    // Relative error
    const double fd_l2_norm_expected = fd_l2_norm(expected_bulk_velocity);
    const double l2_rel_error = fd_l2_error / fd_l2_norm_expected;
    std::cout << "  num_quadrature_points[" << i << "] = " << num_quad_points;
    std::cout << ", abs_error[" << i << "] = " << fd_l2_error;
    std::cout << ", rel_error[" << i << "] = " << l2_rel_error << std::endl;
  }
}

TEST(PeripheryTest, SKFIEIsInvertible) {
  // Check that the second kind Fredholm integral equation matrix is invertable, for both normal orientations

  const double viscosity = 0.5305;
  const double sphere_radius = 12.34;
  const size_t spectral_order = 12;
  for (const bool outward_normal : {false, true}) {
    auto [host_points, host_weights, host_normals] = SphereQuadFunctor(sphere_radius, outward_normal)(spectral_order);
    const size_t num_surface_nodes = host_weights.extent(0);

    // Fill the self-interaction matrix and take its inverse
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> M("M", 3 * num_surface_nodes, 3 * num_surface_nodes);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> M_original("M", 3 * num_surface_nodes,
                                                                             3 * num_surface_nodes);
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> M_inv("M_inv", 3 * num_surface_nodes,
                                                                        3 * num_surface_nodes);
    fill_skfie_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_nodes, host_points, host_normals,
                      host_weights, M, outward_normal);

    // invert overwrites M with its LU factors, so stash the original matrix
    Kokkos::deep_copy(M_original, M);
    mundy::invert(Kokkos::DefaultHostExecutionSpace(), M, M_inv);

    // Multiply the matrix by its inverse and check that it is the m_m_inv matrix
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> m_m_inv("m_m_inv", 3 * num_surface_nodes,
                                                                          3 * num_surface_nodes);
    KokkosBlas::gemm(Kokkos::DefaultHostExecutionSpace(), "N", "N", 1.0, M_original, M_inv, 0.0, m_m_inv);
    for (size_t i = 0; i < 3 * num_surface_nodes; ++i) {
      for (size_t j = 0; j < 3 * num_surface_nodes; ++j) {
        if (i == j) {
          ASSERT_NEAR(m_m_inv(i, j), 1.0, 1.0e-10) << "i = " << i << ", j = " << j;
        } else {
          ASSERT_NEAR(m_m_inv(i, j), 0.0, 1.0e-10) << "i = " << i << ", j = " << j;
        }
      }
    }
  }
}

// The flat-pointer setter reads a row-major 3N x 3N matrix.
TEST(PeripheryTest, SetInverseFromFlatPointer) {
  const size_t num_nodes = 18;  // an order-2 sphere rule
  const size_t n = 3 * num_nodes;
  std::vector<double> flat(n * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      flat[i * n + j] = 1000.0 * static_cast<double>(i) + static_cast<double>(j);
    }
  }

  PeripheryT<Kokkos::DefaultHostExecutionSpace> periphery(num_nodes, 0.5305);
  periphery.set_inverse_self_interaction_matrix(flat.data());
  const auto& M_inv = periphery.get_M_inv();
  size_t num_mismatches = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      num_mismatches += (M_inv(i, j) != flat[i * n + j]) ? 1 : 0;
    }
  }
  EXPECT_EQ(num_mismatches, 0u) << "M_inv(i, j) must equal M_inv_flat[i * 3N + j]";
}

// compute_surface_forces throws, rather than silently returning, before the inverse exists.
TEST(PeripheryTest, ComputeSurfaceForcesRequiresInverse) {
  const size_t num_nodes = 18;
  PeripheryT<Kokkos::DefaultHostExecutionSpace> periphery(num_nodes, 0.5305);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> u("u", 3 * num_nodes);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> f("f", 3 * num_nodes);
  EXPECT_THROW(periphery.compute_surface_forces(u, f), std::runtime_error);
}

// The matrix-free GMRES inverse must recover the same surface forces as the dense direct inverse: both invert the
// same second-kind operator M, so PeripheryT with InverseMethod::MatrixFreeGMRES and InverseMethod::Direct should
// agree on f = -M^{-1} u for an arbitrary slip velocity u.
TEST(PeripheryTest, MatrixFreeMatchesDirectInverse) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MatrixFreeMatchesDirectInverse requires the Belos and Tpetra TPLs.";
#else
  using mundy::mbody::InverseMethod;
  using mundy::mbody::Periphery;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;

  const double viscosity = 0.5305;
  const double sphere_radius = 1.0;
  const int order = 10;

  for (const bool outward_normal : {false, true}) {
    // Generate a sphere quadrature rule.
    const auto [points_vec, weights_vec, normals_vec] = sphere_quadrature(order, sphere_radius, outward_normal);
    const size_t num_nodes = weights_vec.size();

    view_t points("points", 3 * num_nodes), normals("normals", 3 * num_nodes), weights("weights", num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      weights(i) = weights_vec[i];
      for (int j = 0; j < 3; ++j) {
        points(3 * i + j) = points_vec[3 * i + j];
        normals(3 * i + j) = normals_vec[3 * i + j];
      }
    }

    // An arbitrary imposed slip velocity on the surface.
    view_t u("u", 3 * num_nodes);
    openrand::Philox rng = make_philox(0, 0);
    for (size_t i = 0; i < 3 * num_nodes; ++i) {
      u(i) = rng.uniform(-1.0, 1.0);
    }

    // Direct dense inverse.
    Periphery periphery_direct(num_nodes, viscosity);
    periphery_direct.set_surface_positions(points)
        .set_surface_normals(normals, outward_normal)
        .set_quadrature_weights(weights);
    periphery_direct.build_inverse_self_interaction_matrix();
    periphery_direct.set_inverse_method(InverseMethod::Direct);
    view_t f_direct("f_direct", 3 * num_nodes);
    Kokkos::deep_copy(f_direct, 0.0);
    periphery_direct.compute_surface_forces(u, f_direct);

    // Matrix-free GMRES inverse over the same geometry.
    Periphery periphery_mf(num_nodes, viscosity);
    periphery_mf.set_surface_positions(points)
        .set_surface_normals(normals, outward_normal)
        .set_quadrature_weights(weights);
    mundy::BelosConfig<double> cfg;
    cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
    cfg.tol = 1.0e-10;
    cfg.max_iters = 500;
    cfg.num_blocks = 100;
    cfg.max_restarts = 20;
    periphery_mf.set_belos_config(cfg).set_inverse_method(InverseMethod::MatrixFreeGMRES).build_matrix_free_inverse();
    view_t f_mf("f_mf", 3 * num_nodes);
    Kokkos::deep_copy(f_mf, 0.0);
    periphery_mf.compute_surface_forces(u, f_mf);

    const auto& result = periphery_mf.get_last_matrix_free_result();
    EXPECT_TRUE(result.converged) << "GMRES failed to converge: " << result;
    std::cout << "MatrixFreeMatchesDirectInverse: outward_normal=" << outward_normal
              << "  GMRES iters=" << result.num_iters << std::endl;

    // Compare the two force fields.
    double max_abs_diff = 0.0;
    double max_abs_direct = 0.0;
    for (size_t i = 0; i < 3 * num_nodes; ++i) {
      const double diff = std::fabs(f_direct(i) - f_mf(i));
      if (diff > max_abs_diff) {
        max_abs_diff = diff;
      }
      const double mag = std::fabs(f_direct(i));
      if (mag > max_abs_direct) {
        max_abs_direct = mag;
      }
    }
    const double rel_diff = max_abs_diff / (max_abs_direct + 1.0e-300);
    EXPECT_LT(rel_diff, 1.0e-6) << "outward_normal=" << outward_normal
                                << ": matrix-free vs direct relative diff = " << rel_diff
                                << " (GMRES iters = " << result.num_iters << ", residual = " << result.residual << ")";
  }
#endif
}

// ====================================================================================================================
// Operator-level checks of the periphery second-kind operator M = J + T + N, independent of any mobility
// construction. Every term of M carries 1/viscosity, so a coefficient that silently assumes viscosity = 1 is
// invisible at viscosity = 1; these tests therefore always include a non-unit viscosity.
// ====================================================================================================================

// A constant density c makes the subtracted integrand vanish, so each singularity-subtracted operator must return
// exactly its analytic target term (see the file header): (sigma/viscosity) c for the periphery (N c = 0 because
// sum_s w_s n_s = 0 on the sphere rule) and 0 for the exterior trace. The bare punctured sum T c only approximates
// PV T[c] = sigma/(2 viscosity) c (its kernel is O(1/r) at the omitted node), so it is checked loosely.
TEST(PeripheryTest, SkfieConstantDensityJump) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  const auto space = Kokkos::DefaultHostExecutionSpace();
  const double sphere_radius = 13.5;
  const std::vector<int> orders = {4, 8, 16};

  for (const bool outward_normal : {false, true}) {
    const double sigma = outward_normal ? 1.0 : -1.0;
    for (const double viscosity : {1.0, 0.5305, 3.7}) {
      std::vector<double> pv_rel_err;
      for (const int order : orders) {
        auto [points, weights, normals] = SphereQuadFunctor(sphere_radius, outward_normal)(order);
        const size_t num_nodes = weights.extent(0);
        matrix_t M("M", 3 * num_nodes, 3 * num_nodes);
        fill_skfie_matrix(space, viscosity, num_nodes, points, normals, weights, M, outward_normal);
        matrix_t T("T", 3 * num_nodes, 3 * num_nodes);
        fill_stokes_double_layer_matrix(space, viscosity, num_nodes, num_nodes, points, points, normals, weights, T);

        const double expected_skfie = sigma / viscosity;
        double max_err_dense = 0.0;
        double max_err_matrix_free = 0.0;
        double max_err_ss = 0.0;
        double weighted_pv_diagonal = 0.0;  // sum_k sum_t w_t (T e_k)_{3t+k}
        for (int k = 0; k < 3; ++k) {
          view_t c("c", 3 * num_nodes);
          for (size_t i = 0; i < num_nodes; ++i) {
            c(3 * i + k) = 1.0;
          }
          view_t u_dense("u_dense", 3 * num_nodes);
          view_t u_matrix_free("u_matrix_free", 3 * num_nodes);
          view_t u_ss("u_ss", 3 * num_nodes);
          view_t u_pv("u_pv", 3 * num_nodes);
          KokkosBlas::gemv(space, "N", 1.0, M, c, 0.0, u_dense);
          apply_skfie(space, viscosity, num_nodes, points, normals, weights, c, u_matrix_free, outward_normal);
          apply_stokes_double_layer_kernel_ss(space, viscosity, num_nodes, points, normals, weights, c, u_ss);
          KokkosBlas::gemv(space, "N", 1.0, T, c, 0.0, u_pv);
          for (size_t i = 0; i < 3 * num_nodes; ++i) {
            max_err_dense = std::max(max_err_dense, std::fabs(u_dense(i) - expected_skfie * c(i)));
            max_err_matrix_free = std::max(max_err_matrix_free, std::fabs(u_matrix_free(i) - expected_skfie * c(i)));
            max_err_ss = std::max(max_err_ss, std::fabs(u_ss(i)));
          }
          for (size_t t = 0; t < num_nodes; ++t) {
            weighted_pv_diagonal += weights(t) * u_pv(3 * t + k);
          }
        }
        const double area = std::accumulate(weights.data(), weights.data() + num_nodes, 0.0);
        const double pv_mean = weighted_pv_diagonal / (3.0 * area);
        const double pv_expected = sigma / (2.0 * viscosity);
        pv_rel_err.push_back(std::fabs(pv_mean - pv_expected) / std::fabs(pv_expected));

        std::cout << "SkfieConstantDensityJump: outward_normal=" << outward_normal << " viscosity=" << viscosity
                  << " order=" << order << "  max_err dense=" << max_err_dense << " matrix_free=" << max_err_matrix_free
                  << " ss=" << max_err_ss << "  area-mean PV T[1] rel_err=" << pv_rel_err.back() << std::endl;

        // The difference form is algebraically exact for a constant density; only roundoff remains.
        const double tol = 1.0e-11 / viscosity;
        EXPECT_LT(max_err_dense, tol) << "dense (J + T + N)[c] != (sigma/viscosity) c, order " << order;
        EXPECT_LT(max_err_matrix_free, tol) << "apply_skfie[c] != (sigma/viscosity) c, order " << order;
        EXPECT_LT(max_err_ss, tol) << "apply_stokes_double_layer_kernel_ss[c] != 0, order " << order;
      }

      // The bare punctured sum has the right sign and scale and improves under refinement.
      for (size_t i = 1; i < pv_rel_err.size(); ++i) {
        EXPECT_LT(pv_rel_err[i], pv_rel_err[i - 1]) << "PV T[1] must converge, order " << orders[i];
      }
      EXPECT_LT(pv_rel_err.back(), 0.1) << "PV T[1] must approach sigma/(2 viscosity)";
    }
  }
}

// The periphery must reproduce an exact interior Stokes flow: a Stokeslet outside the cavity, imposed as the slip and
// evaluated at interior points (compute_surface_forces returns -M^{-1} u, so the evaluated flow is -u_exact). The error
// must not depend on the viscosity or the normal orientation.
TEST(PeripheryTest, SkfieReproducesInteriorStokesFlow) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using HostPeripheryType = PeripheryT<Kokkos::DefaultHostExecutionSpace>;
  const auto space = Kokkos::DefaultHostExecutionSpace();
  const double sphere_radius = 13.5;
  const std::vector<int> orders = {8, 12, 16, 24};

  // An exterior point force and a deterministic set of interior evaluation points within 0.6 R.
  view_t source_position("source_position", 3);
  view_t source_force("source_force", 3);
  source_position(0) = 1.7 * sphere_radius;
  source_position(1) = 0.4 * sphere_radius;
  source_position(2) = -0.9 * sphere_radius;
  source_force(0) = 0.3;
  source_force(1) = -1.0;
  source_force(2) = 0.6;
  const size_t num_bulk_points = 50;
  view_t bulk_points("bulk_points", 3 * num_bulk_points);
  openrand::Philox rng = make_philox(1234, 0);
  for (size_t i = 0; i < num_bulk_points; ++i) {
    const double theta = rng.uniform(0.0, 2.0 * M_PI);
    const double phi = rng.uniform(0.0, M_PI);
    const double r = rng.uniform(0.0, 0.6 * sphere_radius);
    bulk_points(3 * i) = r * std::sin(phi) * std::cos(theta);
    bulk_points(3 * i + 1) = r * std::sin(phi) * std::sin(theta);
    bulk_points(3 * i + 2) = r * std::cos(phi);
  }

  std::vector<std::vector<double>> all_errors;
  for (const bool outward_normal : {false, true}) {
    for (const double viscosity : {1.0, 0.5305}) {
      view_t u_exact("u_exact", 3 * num_bulk_points);
      apply_stokes_kernel(space, viscosity, source_position, bulk_points, source_force, u_exact);
      double u_exact_norm = 0.0;
      for (size_t i = 0; i < 3 * num_bulk_points; ++i) {
        u_exact_norm += u_exact(i) * u_exact(i);
      }
      u_exact_norm = std::sqrt(u_exact_norm);

      std::vector<double> errors;
      for (const int order : orders) {
        auto [points, weights, normals] = SphereQuadFunctor(sphere_radius, outward_normal)(order);
        const size_t num_nodes = weights.extent(0);

        view_t slip("slip", 3 * num_nodes);
        apply_stokes_kernel(space, viscosity, source_position, points, source_force, slip);

        HostPeripheryType periphery(num_nodes, viscosity);
        periphery.set_surface_positions(points)
            .set_surface_normals(normals, outward_normal)
            .set_quadrature_weights(weights);
        periphery.build_inverse_self_interaction_matrix();
        view_t surface_forces("surface_forces", 3 * num_nodes);
        periphery.compute_surface_forces(slip, surface_forces);

        view_t u_h("u_h", 3 * num_bulk_points);
        apply_stokes_double_layer_kernel(space, viscosity, num_nodes, num_bulk_points, points, bulk_points, normals,
                                         weights, surface_forces, u_h);
        double err = 0.0;
        for (size_t i = 0; i < 3 * num_bulk_points; ++i) {
          err += (u_h(i) + u_exact(i)) * (u_h(i) + u_exact(i));
        }
        errors.push_back(std::sqrt(err) / u_exact_norm);
        std::cout << "SkfieReproducesInteriorStokesFlow: outward_normal=" << outward_normal
                  << " viscosity=" << viscosity << " order=" << order << " N=" << num_nodes
                  << "  rel_err=" << errors.back() << std::endl;
      }

      for (size_t i = 1; i < errors.size(); ++i) {
        EXPECT_LT(errors[i], errors[i - 1]) << "interior flow must converge under refinement, order " << orders[i];
      }
      EXPECT_LT(errors.back(), 3.0e-5) << "finest periphery must reproduce the interior Stokes flow";
      all_errors.push_back(errors);
    }
  }

  // Viscosity and orientation invariance: every error sequence matches the first.
  for (size_t c = 1; c < all_errors.size(); ++c) {
    for (size_t i = 0; i < orders.size(); ++i) {
      EXPECT_NEAR(all_errors[c][i], all_errors[0][i], 1.0e-6 * all_errors[0][i])
          << "interior-flow error must not depend on viscosity or normal orientation, order " << orders[i];
    }
  }
}

// The dense and matrix-free paths implement the same operators: fill_skfie_matrix = apply_skfie, and the dense interior
// trace is the exterior trace plus its analytic target term, (J + T)_interior q = (J + T)_exterior q + (sigma/mu) q.
TEST(PeripheryTest, SkfieDenseMatchesMatrixFree) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  const auto space = Kokkos::DefaultHostExecutionSpace();
  const double sphere_radius = 13.5;
  const double viscosity = 0.5305;

  auto max_rel_diff = [](const view_t& a, const view_t& b) {
    double max_diff = 0.0;
    double max_mag = 0.0;
    for (size_t i = 0; i < a.extent(0); ++i) {
      max_diff = std::max(max_diff, std::fabs(a(i) - b(i)));
      max_mag = std::max(max_mag, std::fabs(a(i)));
    }
    return max_diff / max_mag;
  };

  for (const bool outward_normal : {false, true}) {
    for (const int order : {4, 8, 12}) {
      auto [points, weights, normals] = SphereQuadFunctor(sphere_radius, outward_normal)(order);
      const size_t num_nodes = weights.extent(0);
      view_t q("q", 3 * num_nodes);
      openrand::Philox rng = make_philox(static_cast<size_t>(order), 0);
      for (size_t i = 0; i < 3 * num_nodes; ++i) {
        q(i) = rng.uniform(-1.0, 1.0);
      }

      matrix_t M("M", 3 * num_nodes, 3 * num_nodes);
      fill_skfie_matrix(space, viscosity, num_nodes, points, normals, weights, M, outward_normal);
      view_t u_dense("u_dense", 3 * num_nodes);
      view_t u_matrix_free("u_matrix_free", 3 * num_nodes);
      KokkosBlas::gemv(space, "N", 1.0, M, q, 0.0, u_dense);
      apply_skfie(space, viscosity, num_nodes, points, normals, weights, q, u_matrix_free, outward_normal);
      const double skfie_diff = max_rel_diff(u_dense, u_matrix_free);
      std::cout << "SkfieDenseMatchesMatrixFree: outward_normal=" << outward_normal << " order=" << order
                << "  skfie rel diff=" << skfie_diff;
      EXPECT_LT(skfie_diff, 1.0e-12) << "fill_skfie_matrix vs apply_skfie, order " << order;

      matrix_t T("T", 3 * num_nodes, 3 * num_nodes);
      fill_stokes_double_layer_matrix(space, viscosity, num_nodes, num_nodes, points, points, normals, weights, T);
      add_singularity_subtraction(space, viscosity, T, outward_normal);
      view_t u_interior("u_interior", 3 * num_nodes);
      view_t u_exterior("u_exterior", 3 * num_nodes);
      KokkosBlas::gemv(space, "N", 1.0, T, q, 0.0, u_interior);
      apply_stokes_double_layer_kernel_ss(space, viscosity, num_nodes, points, normals, weights, q, u_exterior);
      const double sigma_over_viscosity = (outward_normal ? 1.0 : -1.0) / viscosity;
      for (size_t i = 0; i < 3 * num_nodes; ++i) {
        u_exterior(i) += sigma_over_viscosity * q(i);
      }
      const double trace_diff = max_rel_diff(u_interior, u_exterior);
      std::cout << "  interior vs exterior + target term rel diff=" << trace_diff << std::endl;
      EXPECT_LT(trace_diff, 1.0e-12) << "add_singularity_subtraction vs apply_stokes_double_layer_kernel_ss, order "
                                     << order;
    }
  }
}

// A declared normal orientation that contradicts the geometry must throw: the analytic target coefficient
// sigma / viscosity would otherwise have the wrong sign.
TEST(PeripheryTest, SkfieRejectsMismatchedNormals) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using matrix_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using HostPeripheryType = PeripheryT<Kokkos::DefaultHostExecutionSpace>;
  const auto space = Kokkos::DefaultHostExecutionSpace();
  const double sphere_radius = 13.5;
  const double viscosity = 0.5305;
  const int order = 6;

  for (const bool outward_normal : {false, true}) {
    auto [points, weights, normals] = SphereQuadFunctor(sphere_radius, outward_normal)(order);
    const size_t num_nodes = weights.extent(0);
    const bool wrong = !outward_normal;
    matrix_t M("M", 3 * num_nodes, 3 * num_nodes);
    view_t q("q", 3 * num_nodes);
    view_t u("u", 3 * num_nodes);
    Kokkos::deep_copy(q, 1.0);

    EXPECT_THROW(fill_skfie_matrix(space, viscosity, num_nodes, points, normals, weights, M, wrong),
                 std::invalid_argument);
    EXPECT_THROW(apply_skfie(space, viscosity, num_nodes, points, normals, weights, q, u, wrong),
                 std::invalid_argument);

    // add_singularity_subtraction cannot see the normals; it checks the flag against the sign of sum_t tr(W_t).
    fill_stokes_double_layer_matrix(space, viscosity, num_nodes, num_nodes, points, points, normals, weights, M);
    EXPECT_THROW(add_singularity_subtraction(space, viscosity, M, wrong), std::invalid_argument);

    HostPeripheryType periphery(num_nodes, viscosity);
    periphery.set_surface_positions(points).set_surface_normals(normals, wrong).set_quadrature_weights(weights);
    EXPECT_THROW(periphery.build_inverse_self_interaction_matrix(), std::invalid_argument);
#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
    EXPECT_THROW(periphery.build_matrix_free_inverse(), std::invalid_argument);
#endif

    // The matching declaration is accepted.
    EXPECT_NO_THROW(fill_skfie_matrix(space, viscosity, num_nodes, points, normals, weights, M, outward_normal));
  }
}

// ====================================================================================================================
// Motile-body BIE mobility: resolved rigid bodies (each with its own surface quadrature) + RPY spheres + periphery,
// solved as one block matrix-free GMRES system. Shared helpers (make_rpy_source, solve_cavity, three_sphere_all_rpy,
// wilson_three_sphere) followed by their asserting tests.
// ====================================================================================================================

// set_orientation stores (w, x, y, z) and place_body_in_lab_frame rotates the reference frame by that quaternion: a
// 90-degree turn about z maps (1, 0, 0) -> (0, 1, 0) and (0, 1, 0) -> (-1, 0, 0) before translating by the center.
TEST(PeripheryTest, PlaceBodyInLabFrameAppliesOrientation) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "PlaceBodyInLabFrameAppliesOrientation requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  MotileBody<ExecSpace> body;
  body.num_quadrature_points = 2;
  body.area = 1.0;
  body.positions_ref = to_device<ExecSpace>(std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  body.normals_ref = to_device<ExecSpace>(std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0});
  body.weights = to_device<ExecSpace>(std::vector<double>{0.5, 0.5});
  body.positions = view_t("positions", 6);
  body.normals = view_t("normals", 6);
  BodySet<ExecSpace> body_set;
  body_set.bodies.push_back(body);
  body_set.wire_pose_load();

  const double c = std::sqrt(0.5);
  body_set.set_center(0, mundy::Vector3d(0.3, -0.2, 0.5));
  body_set.set_orientation(0, mundy::Quaternion<double>(c, 0.0, 0.0, c));  // (w, x, y, z): 90 degrees about z
  place_body_in_lab_frame(ExecSpace{}, body_set.bodies[0]);

  auto orientation = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, body_set.orientations);
  const std::array<double, 4> expected_orientation = {c, 0.0, 0.0, c};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(orientation(i), expected_orientation[i]) << "orientations must be stored as (w, x, y, z), index " << i;
  }

  auto positions = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, body_set.bodies[0].positions);
  auto normals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, body_set.bodies[0].normals);
  const std::array<double, 6> expected_positions = {0.3, 0.8, 0.5, -0.7, -0.2, 0.5};
  const std::array<double, 6> expected_normals = {0.0, 1.0, 0.0, -1.0, 0.0, 0.0};
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(positions(i), expected_positions[i], 1.0e-14) << "position component " << i;
    EXPECT_NEAR(normals(i), expected_normals[i], 1.0e-14) << "normal component " << i;
  }
#endif
}

// A bare motile sphere (no periphery, no ambient spheres) under a prescribed force/torque must recover the
// analytic Stokes drag: U = F/(6 pi mu a), Omega = tau/(8 pi mu a^3). This is the Stage-A go/no-go gate for the
// block mobility solve; a non-identity orientation + offset center also exercises place_body_in_lab_frame (a
// sphere's drag is rotation/translation invariant, so the analytic answer is unchanged).
TEST(PeripheryTest, MotileBodyStokesDrag) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MotileBodyStokesDrag requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  using mem_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>;

  // The body self-block is (-1/(2 mu) I + T_b): both terms scale as 1/mu, so the drag must be exact at any viscosity.
  for (const double mu : {1.0, 0.5305}) {
    const double a = 1.0;  // body radius
    const int order = 8;   // sphere quadrature order

    mundy::BelosConfig<double> cfg;
    cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
    cfg.tol = 1.0e-10;
    cfg.max_iters = 500;
    cfg.num_blocks = 100;
    cfg.max_restarts = 20;

    // A sphere body with OUTWARD normals at an offset center and 90-deg-about-z orientation.
    BodySet<ExecSpace> body_set = make_sphere_body_set<ExecSpace>(
        order, a, mundy::Vector3d(0.3, -0.2, 0.5), mundy::Quaternion<double>(0.70710678, 0.0, 0.0, 0.70710678),
        mundy::Vector3d(1.0, 0.0, 0.0), mundy::Vector3d(0.0, 0.0, 1.0));
    const size_t num_quadrature_points = body_set.bodies[0].num_quadrature_points;

    SphereSet<ExecSpace> spheres;  // none
    view_t p_pos;                  // no periphery
    view_t p_nrm;
    view_t p_wgt;
    Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> M_inv;

    view_t x("x", body_set.num_dofs());
    auto result = solve_mobility(ExecSpace{}, body_set, spheres, mu, /*num_periphery_points=*/0, p_pos, p_nrm, p_wgt,
                                 M_inv, cfg, x);

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);
    const size_t rig = body_set.rigid_velocity_offset(0);
    const mundy::Vector3d U(h_x(rig + 0), h_x(rig + 1), h_x(rig + 2));
    const mundy::Vector3d Om(h_x(rig + 3), h_x(rig + 4), h_x(rig + 5));

    const double U_expected = 1.0 / (6.0 * M_PI * mu * a);           // U = F / (6 pi mu a)
    const double Om_expected = 1.0 / (8.0 * M_PI * mu * a * a * a);  // Omega = tau / (8 pi mu a^3)
    std::cout << "MotileBodyStokesDrag: mu=" << mu << " num_quadrature_points=" << num_quadrature_points
              << " converged=" << result.converged << " iters=" << result.num_iters << " residual=" << result.residual
              << "\n"
              << "  U  = (" << U[0] << ", " << U[1] << ", " << U[2] << ")  expected Ux=" << U_expected << "\n"
              << "  Om = (" << Om[0] << ", " << Om[1] << ", " << Om[2] << ")  expected Oz=" << Om_expected << std::endl;

    // The completed double layer represents rigid translation and rotation exactly.
    const double u_tol = 1.0e-8 * U_expected;
    const double om_tol = 1.0e-8 * Om_expected;
    EXPECT_TRUE(result.converged) << "GMRES did not converge: " << result;
    EXPECT_NEAR(U[0], U_expected, u_tol);
    EXPECT_NEAR(U[1], 0.0, u_tol);
    EXPECT_NEAR(U[2], 0.0, u_tol);
    EXPECT_NEAR(Om[2], Om_expected, om_tol);
    EXPECT_NEAR(Om[0], 0.0, om_tol);
    EXPECT_NEAR(Om[1], 0.0, om_tol);
  }
#endif
}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
// A single RPY source sphere (one point force) used as the ambient driver in the Stage-B tests.
template <class ExecSpace>
SphereSet<ExecSpace> make_rpy_source(const mundy::Vector3d& center, const double radius, const mundy::Vector3d& force) {
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
  return SphereSet<ExecSpace>(positions, radii, forces, SphereInteraction::RPY);
}
#endif

// Faxen's law is exact for a rigid sphere in a Stokes ambient: a force- and torque-free sphere of radius a in a
// flow u_ext translates at U = (1 + a^2/6 grad^2) u_ext(X), exactly (to all orders, for any singularity outside
// the sphere). Driving with a POINT force (source radius 0) makes u_ext a pure Stokeslet G.F, so the exact
// response is U = (1 + a^2/6 grad^2) G.F = M(r; 0, a).F -- the RPY mobility with a point source, which carries no
// finite-source cross-term and is therefore the EXACT anchor. (A finite source radius would add an O(a^2 a_src^2)
// grad^4 G term that RPY drops, so the BIE would converge to a value offset from M(r; a_src, a) -- correct
// physics, wrong reference. The point source removes that ambiguity.) The completed double-layer BIE reaches U by
// a wholly independent route -- sample u_ext on the body surface, GMRES the block system, read U = mean(q). This
// is a refinement study: refining the surface quadrature must collapse the BIE velocity onto M(r; 0, a).F. The
// source sits at r=3 so low orders carry real error and the spectral collapse is visible. (Stage-B: no periphery.)
TEST(PeripheryTest, MotileBodyAmbientFaxenConvergence) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MotileBodyAmbientFaxenConvergence requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  using mem_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>;

  const double mu = 1.0;
  const double a = 1.0;      // body radius
  const double a_src = 0.0;  // point-force source => M(r; 0, a) is the exact Faxen anchor

  // One point-force source outside the body (r = 3 > a, so Faxen is exact; close enough that low quadrature
  // orders carry visible error).
  const mundy::Vector3d src_center(3.0, 0.0, 0.0);
  const mundy::Vector3d src_force(0.0, 1.0, 0.0);
  SphereSet<ExecSpace> spheres = make_rpy_source<ExecSpace>(src_center, a_src, src_force);

  // Analytic reference, independent of the body quadrature: U = M(r; 0, a) . F_src, one RPY evaluation from
  // the source to a single radius-a target at the body center (origin).
  view_t ref_pos("ref_pos", 3);
  view_t ref_radius("ref_radius", 1);
  view_t ref_vel("ref_vel", 3);
  auto h_rpos = Kokkos::create_mirror_view(ref_pos);
  auto h_rrad = Kokkos::create_mirror_view(ref_radius);
  for (int j = 0; j < 3; ++j) {
    h_rpos(j) = 0.0;
  }
  h_rrad(0) = a;
  Kokkos::deep_copy(ref_pos, h_rpos);
  Kokkos::deep_copy(ref_radius, h_rrad);
  Kokkos::deep_copy(ref_vel, 0.0);
  apply_rpy_kernel(ExecSpace{}, mu, spheres.positions, ref_pos, spheres.radii, ref_radius, spheres.forces, ref_vel);
  auto h_rvel = Kokkos::create_mirror_view(ref_vel);
  Kokkos::deep_copy(h_rvel, ref_vel);
  const mundy::Vector3d U_rpy(h_rvel(0), h_rvel(1), h_rvel(2));
  const double U_rpy_mag = std::sqrt(U_rpy[0] * U_rpy[0] + U_rpy[1] * U_rpy[1] + U_rpy[2] * U_rpy[2]);
  ASSERT_GT(U_rpy_mag, 0.0) << "analytic reference velocity is degenerate";

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 500;
  cfg.num_blocks = 100;
  cfg.max_restarts = 20;

  view_t p_pos;  // no periphery
  view_t p_nrm;
  view_t p_wgt;
  Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> M_inv;

  // Solve the force-free/torque-free ambient problem at a given surface-quadrature order; return the body velocity.
  auto ambient_velocity = [&](const int order) -> mundy::Vector3d {
    BodySet<ExecSpace> body_set = make_sphere_body_set<ExecSpace>(order, a, mundy::Vector3d(0.0, 0.0, 0.0),
                                                                  mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0));
    view_t x("x", body_set.num_dofs());
    const auto result = solve_mobility(ExecSpace{}, body_set, spheres, mu, /*num_periphery_points=*/0, p_pos, p_nrm,
                                       p_wgt, M_inv, cfg, x);
    EXPECT_TRUE(result.converged) << "GMRES did not converge at order " << order << ": " << result;
    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);
    const size_t rig = body_set.rigid_velocity_offset(0);
    return mundy::Vector3d(h_x(rig + 0), h_x(rig + 1), h_x(rig + 2));
  };

  const std::vector<int> orders = {4, 6, 8, 10, 12, 14};
  std::vector<double> rel_err;
  rel_err.reserve(orders.size());
  std::cout << "MotileBodyAmbientFaxenConvergence: analytic U = (" << U_rpy[0] << ", " << U_rpy[1] << ", " << U_rpy[2]
            << ")\n";
  for (const int order : orders) {
    const mundy::Vector3d U = ambient_velocity(order);
    const mundy::Vector3d e = U - U_rpy;
    const double err = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) / U_rpy_mag;
    rel_err.push_back(err);
    std::cout << "  order=" << order << "  U=(" << U[0] << ", " << U[1] << ", " << U[2] << ")  rel_err=" << err << "\n";
  }
  std::cout << std::flush;

  // A refinement study, not a single-point tolerance: the error must (1) decrease monotonically until it reaches
  // the solver floor, (2) collapse by at least two decades across the sweep, and (3) reach the analytic mobility
  // at the finest order. The collapse itself is the evidence -- there are no magnitude or finiteness sanity checks.
  const double floor = 5.0 * cfg.tol;  // below this the GMRES residual, not the quadrature, sets the error
  for (size_t i = 1; i < rel_err.size(); ++i) {
    const bool at_floor = rel_err[i - 1] < floor || rel_err[i] < floor;
    EXPECT_TRUE(rel_err[i] < rel_err[i - 1] || at_floor)
        << "error must fall under refinement: order " << orders[i - 1] << " -> " << orders[i] << " gave "
        << rel_err[i - 1] << " -> " << rel_err[i];
  }
  // The finest quadrature must resolve the mobility below the solver tolerance, and the sweep must exhibit a
  // genuine multi-decade spectral collapse -- a flat error curve (e.g. an approximate rsqrt flooring the kernels)
  // fails the second check even though it would pass a loose single-point tolerance.
  EXPECT_LT(rel_err.back(), 100.0 * cfg.tol) << "finest quadrature must reach the analytic Faxen/RPY mobility";
  EXPECT_LT(rel_err.back(), rel_err.front() / 1.0e3) << "refinement must collapse the error by >= 3 decades";
#endif
}

// The block system is linear in the loads (F_b, tau_b, F_source), so its solution must superpose: the response to
// (body force/torque only) plus the response to (ambient only) must equal the response to (both), to the GMRES
// tolerance. This is not a physics check -- it is a guard on the RHS assembly, which accumulates the body G/R and
// the sphere RPY contributions into shared buffers; a missed zero or an overwrite (rather than +=) breaks it.
TEST(PeripheryTest, MotileBodyAmbientSuperposition) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MotileBodyAmbientSuperposition requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  using mem_space = typename ExecSpace::memory_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>;

  const double mu = 1.0;
  const double a = 1.0;
  const int order = 8;

  BodySet<ExecSpace> body_set = make_sphere_body_set<ExecSpace>(order, a, mundy::Vector3d(0.0, 0.0, 0.0),
                                                                mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0));

  SphereSet<ExecSpace> spheres =
      make_rpy_source<ExecSpace>(mundy::Vector3d(5.0, 0.0, 0.0), 0.5, mundy::Vector3d(0.0, 1.0, 0.0));
  SphereSet<ExecSpace> no_spheres;  // num_spheres() == 0

  view_t p_pos;  // no periphery
  view_t p_nrm;
  view_t p_wgt;
  Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> M_inv;

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 500;
  cfg.num_blocks = 100;
  cfg.max_restarts = 20;

  auto solve = [&](const mundy::Vector3d& F, const mundy::Vector3d& tau, const SphereSet<ExecSpace>& sph,
                   const view_t& x) {
    body_set.set_net_force(0, F);
    body_set.set_net_torque(0, tau);
    return solve_mobility(ExecSpace{}, body_set, sph, mu, /*num_periphery_points=*/0, p_pos, p_nrm, p_wgt, M_inv, cfg,
                          x);
  };
  auto rigid_velocities = [&](const view_t& x, mundy::Vector3d& U, mundy::Vector3d& Om) {
    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);
    const size_t rig = body_set.rigid_velocity_offset(0);
    U = mundy::Vector3d(h_x(rig + 0), h_x(rig + 1), h_x(rig + 2));
    Om = mundy::Vector3d(h_x(rig + 3), h_x(rig + 4), h_x(rig + 5));
  };

  const mundy::Vector3d F(1.0, 0.0, 0.0);
  const mundy::Vector3d tau(0.0, 0.0, 1.0);
  const mundy::Vector3d zero(0.0, 0.0, 0.0);

  view_t x_body("x_body", body_set.num_dofs());
  view_t x_amb("x_amb", body_set.num_dofs());
  view_t x_both("x_both", body_set.num_dofs());
  const auto r_body = solve(F, tau, no_spheres, x_body);  // body force/torque only
  const auto r_amb = solve(zero, zero, spheres, x_amb);   // ambient (spheres) only
  const auto r_both = solve(F, tau, spheres, x_both);     // both
  ASSERT_TRUE(r_body.converged) << r_body;
  ASSERT_TRUE(r_amb.converged) << r_amb;
  ASSERT_TRUE(r_both.converged) << r_both;

  mundy::Vector3d U_body, Om_body, U_amb, Om_amb, U_both, Om_both;
  rigid_velocities(x_body, U_body, Om_body);
  rigid_velocities(x_amb, U_amb, Om_amb);
  rigid_velocities(x_both, U_both, Om_both);

  // Superposition holds to the GMRES tolerance (the loads add linearly through the RHS).
  const double tol = 100.0 * cfg.tol;
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(U_both[k], U_body[k] + U_amb[k], tol) << "U component " << k;
    EXPECT_NEAR(Om_both[k], Om_body[k] + Om_amb[k], tol) << "Omega component " << k;
  }
#endif
}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
// Solve one rigid sphere (radius a) driven by force F and torque tau at the center of a stationary spherical cavity
// (radius b), at surface-quadrature order `order`. Returns its rigid velocity U and angular velocity Omega both
// unbounded (no periphery) and confined (with the cavity's dense periphery inverse). The unbounded solve
// isolates the body's own quadrature error from the wall-coupling error.
template <class ExecSpace>
void solve_cavity(const double mu, const double a, const double b, const int order,
                  const mundy::BelosConfig<double>& cfg, const mundy::Vector3d& F, const mundy::Vector3d& tau,
                  mundy::Vector3d& U_unbounded, mundy::Vector3d& Om_unbounded, mundy::Vector3d& U_confined,
                  mundy::Vector3d& Om_confined) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  using mat_t = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;

  auto [p_pts, p_wts, p_nrm] = sphere_quadrature(order, b, /*outward_normal=*/false);
  const size_t num_periphery_points = p_wts.size();
  PeripheryT<ExecSpace> periphery(num_periphery_points, mu);
  periphery.set_surface_positions(p_pts.data())
      .set_quadrature_weights(p_wts.data())
      .set_surface_normals(p_nrm.data(), /*outward_normal=*/false);
  periphery.build_inverse_self_interaction_matrix();
  auto M_inv = periphery.get_M_inv();
  auto p_positions = periphery.get_surface_positions();
  auto p_normals = periphery.get_surface_normals();
  auto p_weights = periphery.get_quadrature_weights();

  BodySet<ExecSpace> body_set = make_sphere_body_set<ExecSpace>(order, a, mundy::Vector3d(0.0, 0.0, 0.0),
                                                                mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0), F, tau);

  SphereSet<ExecSpace> no_spheres;
  view_t empty_p;
  mat_t empty_M;
  auto extract = [&](const view_t& x, mundy::Vector3d& U, mundy::Vector3d& Om) {
    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);
    const size_t rig = body_set.rigid_velocity_offset(0);
    U = mundy::Vector3d(h_x(rig + 0), h_x(rig + 1), h_x(rig + 2));
    Om = mundy::Vector3d(h_x(rig + 3), h_x(rig + 4), h_x(rig + 5));
  };
  view_t x0("x0", body_set.num_dofs());
  solve_mobility(ExecSpace{}, body_set, no_spheres, mu, /*num_periphery_points=*/0, empty_p, empty_p, empty_p, empty_M,
                 cfg, x0);
  extract(x0, U_unbounded, Om_unbounded);
  view_t x("x", body_set.num_dofs());
  solve_mobility(ExecSpace{}, body_set, no_spheres, mu, num_periphery_points, p_positions, p_normals, p_weights, M_inv,
                 cfg, x);
  extract(x, U_confined, Om_confined);
}
#endif

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
TEST(PeripheryTest, MotileBodyInSphericalCavityDrag) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MotileBodyInSphericalCavityDrag requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;

  const double a = 1.0;  // body radius
  const double b = 5.0;  // cavity radius (lambda = a/b = 0.2)
  const double lambda = a / b;

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 1000;
  cfg.num_blocks = 150;
  cfg.max_restarts = 30;

  const mundy::Vector3d F(1.0, 0.0, 0.0);
  const mundy::Vector3d no_torque(0.0, 0.0, 0.0);

  const std::vector<int> orders = {4, 6, 8, 10, 12, 16, 20, 24};
  std::vector<std::vector<double>> rel_err_by_viscosity;
  for (const double mu : {1.0, 0.5305}) {
    // Exact concentric-cavity translation mobility: U = U_stokes * poly(lambda) / (1 - lambda^5).
    const double u_stokes = 1.0 / (6.0 * M_PI * mu * a);
    const double l3 = lambda * lambda * lambda;
    const double l5 = l3 * lambda * lambda;
    const double l6 = l5 * lambda;
    const double poly = 1.0 - 2.25 * lambda + 2.5 * l3 - 2.25 * l5 + l6;
    const double U_exact = u_stokes * poly / (1.0 - l5);

    std::vector<double> rel_err_conf;
    rel_err_conf.reserve(orders.size());
    std::cout << "MotileBodyInSphericalCavityDrag: mu=" << mu << " a=" << a << " b=" << b << " lambda=" << lambda
              << "  U_stokes=" << u_stokes << "  U_exact=" << U_exact << "\n";
    for (const int order : orders) {
      mundy::Vector3d U_unb;
      mundy::Vector3d Om_unb;
      mundy::Vector3d U_conf;
      mundy::Vector3d Om_conf;
      solve_cavity<ExecSpace>(mu, a, b, order, cfg, F, no_torque, U_unb, Om_unb, U_conf, Om_conf);
      const double err_unb = std::abs(U_unb[0] - u_stokes) / u_stokes;
      const double err_conf = std::abs(U_conf[0] - U_exact) / U_exact;
      rel_err_conf.push_back(err_conf);
      // Unbounded drag is machine-exact at every order (the completed double layer represents rigid sphere
      // translation exactly), so all of the confined error is wall coupling.
      EXPECT_LT(err_unb, 1.0e-8) << "unbounded Stokes drag must be exact at order " << order << " (err " << err_unb
                                 << ")";
      std::cout << "  order=" << order << "  U_unb_err=" << err_unb << "   U_conf=" << U_conf[0] << "  err=" << err_conf
                << "\n";
    }
    std::cout << std::flush;

    // The confined error converges algebraically (the subtracted integrand is bounded but not smooth at the target)
    // but faster than 1/order, because the jump term is analytic. The gate: monotone decrease, the finest order within
    // 1e-4 of the exact cavity value, and an observed rate above 2 between the middle and finest orders.
    for (size_t i = 1; i < rel_err_conf.size(); ++i) {
      EXPECT_LT(rel_err_conf[i], rel_err_conf[i - 1])
          << "confined mobility must converge under refinement at order " << orders[i] << " (mu = " << mu << ")";
    }
    EXPECT_LT(rel_err_conf.back(), 1.0e-4) << "finest quadrature must reach the analytic cavity mobility";

    const size_t i_ref = orders.size() / 2;
    const double observed_rate = std::log(rel_err_conf[i_ref] / rel_err_conf.back()) /
                                 std::log(static_cast<double>(orders.back()) / orders[i_ref]);
    std::cout << "  observed convergence rate (orders " << orders[i_ref] << " -> " << orders.back()
              << ") = " << observed_rate << std::endl;
    EXPECT_GT(observed_rate, 2.0) << "confined mobility must converge faster than ~1/order";
    rel_err_by_viscosity.push_back(rel_err_conf);
  }

  // Every operator in the coupled solve scales as 1/mu, so the relative error must not depend on the viscosity.
  for (size_t i = 0; i < orders.size(); ++i) {
    EXPECT_NEAR(rel_err_by_viscosity[1][i], rel_err_by_viscosity[0][i], 1.0e-2 * rel_err_by_viscosity[0][i])
        << "confined-mobility error must not depend on the viscosity, order " << orders[i];
  }
#endif
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
TEST(PeripheryTest, MotileBodyInSphericalCavityRotation) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MotileBodyInSphericalCavityRotation requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;

  const double a = 1.0;  // body radius
  const double b = 5.0;  // cavity radius (lambda = a/b = 0.2)
  const double lambda = a / b;

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 1000;
  cfg.num_blocks = 150;
  cfg.max_restarts = 30;

  const mundy::Vector3d no_force(0.0, 0.0, 0.0);
  const mundy::Vector3d tau(0.0, 0.0, 1.0);  // unit torque about z

  const std::vector<int> orders = {4, 6, 8, 10, 12, 16, 20, 24};
  for (const double mu : {1.0, 0.5305}) {
    // Exact concentric-cavity rotational mobility: Omega = Omega_stokes * (1 - lambda^3).
    const double om_stokes = 1.0 / (8.0 * M_PI * mu * a * a * a);  // angular velocity per unit torque, unbounded
    const double l3 = lambda * lambda * lambda;
    const double Om_exact = om_stokes * (1.0 - l3);

    std::vector<double> rel_err_conf;
    rel_err_conf.reserve(orders.size());
    std::cout << "MotileBodyInSphericalCavityRotation: mu=" << mu << " a=" << a << " b=" << b << " lambda=" << lambda
              << "  Om_stokes=" << om_stokes << "  Om_exact=" << Om_exact << "\n";
    for (const int order : orders) {
      mundy::Vector3d U_unb;
      mundy::Vector3d Om_unb;
      mundy::Vector3d U_conf;
      mundy::Vector3d Om_conf;
      solve_cavity<ExecSpace>(mu, a, b, order, cfg, no_force, tau, U_unb, Om_unb, U_conf, Om_conf);
      const double err_unb = std::abs(Om_unb[2] - om_stokes) / om_stokes;
      const double err_conf = std::abs(Om_conf[2] - Om_exact) / Om_exact;
      rel_err_conf.push_back(err_conf);
      // Unbounded rotational drag is machine-exact at every order, so all of the confined error is wall coupling.
      EXPECT_LT(err_unb, 1.0e-8) << "unbounded rotational Stokes drag must be exact at order " << order << " (err "
                                 << err_unb << ")";
      std::cout << "  order=" << order << "  Om_unb_err=" << err_unb << "   Om_conf=" << Om_conf[2]
                << "  err=" << err_conf << "\n";
    }
    std::cout << std::flush;

    // The confined rotation converges spectrally: the rotlet slip is a rigid rotation, for which the subtracted
    // integrand vanishes identically (K(x, y) v is proportional to r (r . v), and v = -Omega x r is orthogonal to r).
    // The gate: monotone decrease until the solver floor, and the finest order at the solver tolerance.
    const double floor = 5.0 * cfg.tol;
    for (size_t i = 1; i < rel_err_conf.size(); ++i) {
      const bool at_floor = rel_err_conf[i - 1] < floor || rel_err_conf[i] < floor;
      EXPECT_TRUE(rel_err_conf[i] < rel_err_conf[i - 1] || at_floor)
          << "confined rotational mobility must converge under refinement at order " << orders[i] << " (mu = " << mu
          << "): " << rel_err_conf[i - 1] << " -> " << rel_err_conf[i];
    }
    EXPECT_LT(rel_err_conf.back(), 100.0 * cfg.tol) << "finest quadrature must reach the analytic cavity rotation";
  }
#endif
}

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
/// \brief A spherical cavity periphery (inward normals) with its dense inverse built.
template <class ExecSpace>
std::shared_ptr<PeripheryT<ExecSpace>> make_cavity_periphery(const int order, const double cavity_radius,
                                                             const double viscosity) {
  auto [points, weights, normals] = sphere_quadrature(order, cavity_radius, /*outward_normal=*/false);
  auto periphery = std::make_shared<PeripheryT<ExecSpace>>(weights.size(), viscosity);
  periphery->set_surface_positions(points.data())
      .set_quadrature_weights(weights.data())
      .set_surface_normals(normals.data(), /*outward_normal=*/false);
  periphery->build_inverse_self_interaction_matrix();
  return periphery;
}
#endif

// MobilitySystem is a front end: its outputs must equal the building blocks it wires together -- solve_mobility for a
// resolved body, and dry drag + the periphery's no-slip response for a point sphere -- and a Dry sphere moves at its
// bare drag even inside a periphery.
TEST(PeripheryTest, MobilitySystemMatchesBuildingBlocks) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MobilitySystemMatchesBuildingBlocks requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  const double mu = 0.5305;
  const int order = 8;
  auto periphery = make_cavity_periphery<ExecSpace>(order, /*cavity_radius=*/5.0, mu);
  const size_t num_periphery_points = periphery->get_num_nodes();

  mundy::BelosConfig<double> cfg;
  cfg.solver = mundy::BelosSolver::PSEUDOBLOCK_GMRES;
  cfg.tol = 1.0e-10;
  cfg.max_iters = 500;
  cfg.num_blocks = 100;
  cfg.max_restarts = 20;

  auto to_host = [](const view_t& v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v); };
  auto max_rel_diff = [&](const view_t& a, const view_t& b) {
    auto ha = to_host(a);
    auto hb = to_host(b);
    double max_diff = 0.0;
    double max_mag = 0.0;
    for (size_t i = 0; i < ha.extent(0); ++i) {
      max_diff = std::max(max_diff, std::fabs(ha(i) - hb(i)));
      max_mag = std::max(max_mag, std::fabs(ha(i)));
    }
    return max_diff / max_mag;
  };

  // Resolved body: MobilitySystem vs solve_mobility.
  {
    BodySet<ExecSpace> body_set = make_sphere_body_set<ExecSpace>(
        order, 1.0, mundy::Vector3d(0.4, -0.3, 0.2), mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0),
        mundy::Vector3d(1.0, 0.0, 0.0), mundy::Vector3d(0.0, 0.0, 1.0));
    view_t U("U", 3);
    view_t Om("Om", 3);
    MobilitySystem<ExecSpace> mobility(mu);
    mobility.set_periphery(periphery).set_belos_config(cfg).set_bodies(body_set, U, Om).solve();
    EXPECT_TRUE(mobility.body_solve_result().converged) << mobility.body_solve_result();

    view_t x("x", body_set.num_dofs());
    solve_mobility(ExecSpace{}, body_set, SphereSet<ExecSpace>{}, mu, num_periphery_points,
                   periphery->get_surface_positions(), periphery->get_surface_normals(),
                   periphery->get_quadrature_weights(), periphery->get_M_inv(), cfg, x);
    const size_t rig = body_set.rigid_velocity_offset(0);
    view_t U_ref = Kokkos::subview(x, Kokkos::pair<size_t, size_t>(rig, rig + 3));
    view_t Om_ref = Kokkos::subview(x, Kokkos::pair<size_t, size_t>(rig + 3, rig + 6));
    EXPECT_LT(max_rel_diff(U_ref, U), 1.0e-12) << "MobilitySystem body velocity != solve_mobility";
    EXPECT_LT(max_rel_diff(Om_ref, Om), 1.0e-12) << "MobilitySystem body angular velocity != solve_mobility";
  }

  // Point spheres: dry drag + the periphery's no-slip response (RPY), and the bare drag alone (Dry).
  const double a = 0.1;
  view_t positions = to_device<ExecSpace>(std::vector<double>{0.5, 1.0, 1.5});
  view_t radii = to_device<ExecSpace>(std::vector<double>{a});
  view_t forces = to_device<ExecSpace>(std::vector<double>{0.3, -1.0, 0.6});
  view_t dry_drag("dry_drag", 3);
  apply_local_drag(ExecSpace{}, mu, dry_drag, forces, radii);
  {
    view_t u("u", 3);
    MobilitySystem<ExecSpace> mobility(mu);
    mobility.set_periphery(periphery).set_spheres(positions, radii, forces, u, SphereInteraction::RPY).solve();

    view_t slip("slip", 3 * num_periphery_points);
    view_t node_radii("node_radii", num_periphery_points);
    apply_rpy_kernel(ExecSpace{}, mu, positions, periphery->get_surface_positions(), radii, node_radii, forces, slip);
    view_t surface_forces("surface_forces", 3 * num_periphery_points);
    periphery->compute_surface_forces(slip, surface_forces);
    view_t u_ref("u_ref", 3);
    Kokkos::deep_copy(u_ref, dry_drag);
    apply_stokes_double_layer_kernel(ExecSpace{}, mu, num_periphery_points, 1, periphery->get_surface_positions(),
                                     positions, periphery->get_surface_normals(), periphery->get_quadrature_weights(),
                                     surface_forces, u_ref);
    EXPECT_LT(max_rel_diff(u_ref, u), 1.0e-12) << "MobilitySystem RPY sphere velocity != drag + periphery response";
  }
  {
    view_t u("u", 3);
    MobilitySystem<ExecSpace> mobility(mu);
    mobility.set_periphery(periphery).set_spheres(positions, radii, forces, u, SphereInteraction::Dry).solve();
    EXPECT_LT(max_rel_diff(dry_drag, u), 1.0e-15) << "a Dry sphere must move at its bare drag";
  }
#endif
}

// MobilitySystem and its inputs throw on inconsistent or incomplete inputs instead of reading invalid memory.
TEST(PeripheryTest, MobilitySystemRejectsInvalidInputs) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "MobilitySystemRejectsInvalidInputs requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  const double mu = 0.5305;
  view_t three_a("three_a", 3);
  view_t three_b("three_b", 3);
  view_t three_c("three_c", 3);
  view_t six("six", 6);
  view_t one("one", 1);

  // Mismatched sphere lengths.
  EXPECT_THROW((SphereSet<ExecSpace>(six, one, three_a, SphereInteraction::RPY)), std::invalid_argument);
  MobilitySystem<ExecSpace> mobility(mu);
  EXPECT_THROW(mobility.set_spheres(three_a, one, three_b, six, SphereInteraction::RPY), std::invalid_argument);

  // An unwired BodySet, and its setters.
  BodySet<ExecSpace> unwired;
  unwired.bodies.push_back(make_sphere_body<ExecSpace>(4, 1.0));
  EXPECT_THROW(mobility.set_bodies(unwired, three_a, three_b), std::invalid_argument);
  EXPECT_THROW(unwired.set_center(0, mundy::Vector3d(0.0, 0.0, 0.0)), std::invalid_argument);

  // Mismatched body-velocity lengths.
  BodySet<ExecSpace> wired = make_sphere_body_set<ExecSpace>(4, 1.0, mundy::Vector3d(0.0, 0.0, 0.0),
                                                             mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0));
  EXPECT_THROW(mobility.set_bodies(wired, six, three_b), std::invalid_argument);

  // No body solve has run yet.
  EXPECT_THROW(mobility.body_solve_result(), std::runtime_error);

  // A periphery without a dense inverse.
  auto [points, weights, normals] = sphere_quadrature(4, 5.0, /*outward_normal=*/false);
  auto periphery = std::make_shared<PeripheryT<ExecSpace>>(weights.size(), mu);
  periphery->set_surface_positions(points.data())
      .set_quadrature_weights(weights.data())
      .set_surface_normals(normals.data(), /*outward_normal=*/false);
  mobility.set_periphery(periphery).set_bodies(wired, three_b, three_c);
  EXPECT_THROW(mobility.solve(), std::runtime_error);
#endif
}

// Velocities of three RPY point spheres at the Wilson triangle (equilateral, side s*r, radius r), with force F1
// along y on s1 only, including the self-drag term. Returns v[0..2] = (s1, s2, s3). Reference for the resolved
// inclusion/body tests below.
template <class ExecSpace>
std::array<mundy::Vector3d, 3> three_sphere_all_rpy(const double mu, const double r, const double F1, const double s) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  const double sqrt3_2 = std::sqrt(3.0) / 2.0;
  const double bare = 1.0 / (6.0 * M_PI * mu * r);

  std::array<mundy::Vector3d, 3> pos;
  pos[0] = mundy::Vector3d(0.0, 0.0, 0.0);
  pos[1] = mundy::Vector3d(-s * r / 2.0, -sqrt3_2 * s * r, 0.0);
  pos[2] = mundy::Vector3d(s * r / 2.0, -sqrt3_2 * s * r, 0.0);

  view_t positions("positions", 9);
  view_t radii("radii", 3);
  view_t forces("forces", 9);
  view_t vel("vel", 9);
  auto hp = Kokkos::create_mirror_view(positions);
  auto hr = Kokkos::create_mirror_view(radii);
  auto hf = Kokkos::create_mirror_view(forces);
  Kokkos::deep_copy(hf, 0.0);
  for (int j = 0; j < 3; ++j) {
    hr(j) = r;
    for (int d = 0; d < 3; ++d) {
      hp(3 * j + d) = pos[j][d];
    }
  }
  hf(1) = F1;  // force on s1, y-component
  Kokkos::deep_copy(positions, hp);
  Kokkos::deep_copy(radii, hr);
  Kokkos::deep_copy(forces, hf);
  Kokkos::deep_copy(vel, 0.0);
  apply_rpy_kernel(ExecSpace{}, mu, positions, positions, radii, radii, forces, vel);
  Kokkos::parallel_for(
      "three_sphere_rpy_self", Kokkos::RangePolicy<ExecSpace>(0, 9),
      KOKKOS_LAMBDA(const int k) { vel(k) += bare * forces(k); });
  auto hv = Kokkos::create_mirror_view(vel);
  Kokkos::deep_copy(hv, vel);
  std::array<mundy::Vector3d, 3> v;
  for (int j = 0; j < 3; ++j) {
    v[j] = mundy::Vector3d(hv(3 * j), hv(3 * j + 1), hv(3 * j + 2));
  }
  return v;
}

enum class SphereType { RPY, Inclusion };

#if defined(HAVE_MUNDYMATH_BELOS) && defined(HAVE_MUNDYMATH_TPETRA)
// Solve the Wilson three-sphere problem at separation s with each of the three spheres independently typed as an
// RPY point sphere or a resolved rigid inclusion (a BIE body). RPY spheres carry their mutual RPY interaction plus
// self-drag; inclusions are resolved bodies in the coupled block solve; evaluate_sphere_flow adds each body's
// induced flow at the RPY spheres. Returns the three velocities in the s1/s2/s3 slots.
template <class ExecSpace>
std::array<mundy::Vector3d, 3> wilson_three_sphere(const double mu, const double r, const double F1, const double s,
                                                   const int order, const mundy::BelosConfig<double>& cfg,
                                                   const std::array<SphereType, 3>& types) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  using mat_t = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
  const double sqrt3_2 = std::sqrt(3.0) / 2.0;
  const double bare = 1.0 / (6.0 * M_PI * mu * r);

  std::array<mundy::Vector3d, 3> pos;
  pos[0] = mundy::Vector3d(0.0, 0.0, 0.0);
  pos[1] = mundy::Vector3d(-s * r / 2.0, -sqrt3_2 * s * r, 0.0);
  pos[2] = mundy::Vector3d(s * r / 2.0, -sqrt3_2 * s * r, 0.0);
  std::array<mundy::Vector3d, 3> frc;
  frc[0] = mundy::Vector3d(0.0, F1, 0.0);
  frc[1] = mundy::Vector3d(0.0, 0.0, 0.0);
  frc[2] = mundy::Vector3d(0.0, 0.0, 0.0);

  // Partition the three objects into resolved bodies (Inclusion) and RPY spheres, tracking their global indices.
  std::vector<int> body_gi;
  std::vector<int> sph_gi;
  for (int j = 0; j < 3; ++j) {
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
    body_set.set_center(m, pos[body_gi[m]]);
    body_set.set_orientation(m, mundy::Quaternion<double>(1.0, 0.0, 0.0, 0.0));
    body_set.set_net_force(m, frc[body_gi[m]]);
    body_set.set_net_torque(m, mundy::Vector3d(0.0, 0.0, 0.0));
    place_body_in_lab_frame(ExecSpace{}, body_set.bodies[m]);
  }

  const size_t num_spheres = sph_gi.size();
  SphereSet<ExecSpace> sph;
  view_t u_spheres;
  if (num_spheres > 0) {
    view_t sph_pos("sph_pos", 3 * num_spheres);
    view_t sph_rad("sph_rad", num_spheres);
    view_t sph_frc("sph_frc", 3 * num_spheres);
    auto hp = Kokkos::create_mirror_view(sph_pos);
    auto hr = Kokkos::create_mirror_view(sph_rad);
    auto hf = Kokkos::create_mirror_view(sph_frc);
    for (size_t a = 0; a < num_spheres; ++a) {
      hr(a) = r;
      for (int d = 0; d < 3; ++d) {
        hp(3 * a + d) = pos[sph_gi[a]][d];
        hf(3 * a + d) = frc[sph_gi[a]][d];
      }
    }
    Kokkos::deep_copy(sph_pos, hp);
    Kokkos::deep_copy(sph_rad, hr);
    Kokkos::deep_copy(sph_frc, hf);
    sph = SphereSet<ExecSpace>(sph_pos, sph_rad, sph_frc, SphereInteraction::RPY);

    // RPY spheres' own velocity: mutual RPY + self-drag. evaluate_sphere_flow accumulates the bodies' flow onto it.
    u_spheres = view_t("u_spheres", 3 * num_spheres);
    Kokkos::deep_copy(u_spheres, 0.0);
    apply_rpy_kernel(ExecSpace{}, mu, sph.positions, sph.positions, sph.radii, sph.radii, sph.forces, u_spheres);
    view_t sph_forces = sph.forces;
    Kokkos::parallel_for(
        "wilson_sph_self", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(3 * num_spheres)),
        KOKKOS_LAMBDA(const int i) { u_spheres(i) += bare * sph_forces(i); });
  }

  std::array<mundy::Vector3d, 3> v;

  // Bodies (if any): solve the coupled block system, then add their induced flow at the RPY spheres.
  if (!body_gi.empty()) {
    view_t empty_p;
    mat_t empty_M;
    view_t x("x", body_set.num_dofs());
    const auto res = solve_mobility(ExecSpace{}, body_set, sph, mu, /*num_periphery_points=*/0, empty_p, empty_p,
                                    empty_p, empty_M, cfg, x);
    EXPECT_TRUE(res.converged) << "GMRES failed at s=" << s << ": " << res;
    if (num_spheres > 0) {
      evaluate_sphere_flow(ExecSpace{}, body_set, sph, mu, /*num_periphery_points=*/0, empty_p, empty_p, empty_p,
                           empty_M, x, u_spheres);
    }
    auto hx = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(hx, x);
    for (size_t bi = 0; bi < body_gi.size(); ++bi) {
      const size_t rig = body_set.rigid_velocity_offset(bi);
      v[body_gi[bi]] = mundy::Vector3d(hx(rig + 0), hx(rig + 1), hx(rig + 2));
    }
  }

  // RPY spheres.
  if (num_spheres > 0) {
    auto hu = Kokkos::create_mirror_view(u_spheres);
    Kokkos::deep_copy(hu, u_spheres);
    for (size_t a = 0; a < num_spheres; ++a) {
      v[sph_gi[a]] = mundy::Vector3d(hu(3 * a), hu(3 * a + 1), hu(3 * a + 2));
    }
  }

  return v;
}
#endif

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
TEST(PeripheryTest, WilsonThreeSphere) {
#if !defined(HAVE_MUNDYMATH_BELOS) || !defined(HAVE_MUNDYMATH_TPETRA)
  GTEST_SKIP() << "WilsonThreeSphere requires the Belos and Tpetra TPLs.";
#else
  using ExecSpace = Tpetra::Map<>::node_type::execution_space;

  const double mu = 1.0;
  const double r = 12.34;
  const int order = 12;
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
  std::cout << std::setw(8) << std::left << "" << std::right << " | " << std::setw(11) << "Wilson";
  for (const auto& config : configs) {
    std::cout << " | " << std::setw(11) << config.label;
  }
  std::cout << "\n";

  for (size_t i = 0; i < svals.size(); ++i) {
    const double s = svals[i];
    std::array<std::array<mundy::Vector3d, 3>, 4> vel;
    for (int c = 0; c < 4; ++c) {
      vel[c] = wilson_three_sphere<ExecSpace>(mu, r, F1, s, order, cfg, configs[c].types);
    }
    const auto rpy = three_sphere_all_rpy<ExecSpace>(mu, r, F1, s);

    std::cout << "s = " << s << "\n";
    prow("U1", W_U1[i], {U1(vel[0]), U1(vel[1]), U1(vel[2]), U1(vel[3])});
    prow("U2", W_U2[i], {U2(vel[0]), U2(vel[1]), U2(vel[2]), U2(vel[3])});
    prow("U3", W_U3[i], {U3(vel[0]), U3(vel[1]), U3(vel[2]), U3(vel[3])});

    // (a) {R,R,R} reproduces the independent analytic RPY reference (validates the sphere-only path of the helper).
    EXPECT_NEAR(U1(vel[0]), U1(rpy), 1.0e-9) << "RRR == analytic RPY, s=" << s;
    EXPECT_NEAR(U2(vel[0]), U2(rpy), 1.0e-9) << "RRR == analytic RPY, s=" << s;
    EXPECT_NEAR(U3(vel[0]), U3(rpy), 1.0e-9) << "RRR == analytic RPY, s=" << s;

    // (b) {I,I,I} reproduces Wilson's EXACT reference to quadrature accuracy at every separation.
    EXPECT_NEAR(U1(vel[3]), W_U1[i], 1.0e-3) << "III -> Wilson U1, s=" << s;
    EXPECT_NEAR(U2(vel[3]), W_U2[i], 1.0e-3) << "III -> Wilson U2, s=" << s;
    EXPECT_NEAR(U3(vel[3]), W_U3[i], 1.0e-3) << "III -> Wilson U3, s=" << s;

    // (c) A force-free inclusion driven only by an RPY ambient reproduces the RPY sphere velocity (Faxen of an RPY
    //     field is the RPY tensor): {R,I,R}'s inclusion (s2) matches the all-RPY sphere velocity.
    EXPECT_NEAR(U2(vel[1]), U2(rpy), 3.0e-3 * std::abs(U2(rpy))) << "RIR inclusion == RPY, s=" << s;
    // (d) The resolved passive inclusion slows the forced sphere -- {R,I,R} U1 lies strictly between Wilson and the
    //     RPY value (1), improving on RPY which misses the back-reaction.
    EXPECT_GT(U1(vel[1]), W_U1[i]) << "RIR U1 above Wilson, s=" << s;
    EXPECT_LT(U1(vel[1]), 1.0) << "RIR U1 below the RPY value, s=" << s;
    // (e) The forced inclusion feels no back-reaction from force-free RPY neighbors, so it moves at the bare drag.
    EXPECT_NEAR(U1(vel[2]), 1.0, 1.0e-3) << "IRR forced-inclusion bare drag, s=" << s;

    // (f) Every config converges to Wilson at the largest separation (a coarse universal sanity floor).
    if (i + 1 == svals.size()) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(U1(vel[c]), W_U1[i], 1.0e-2) << configs[c].label << " U1 -> Wilson, s=" << s;
        EXPECT_NEAR(U2(vel[c]), W_U2[i], 1.0e-2) << configs[c].label << " U2 -> Wilson, s=" << s;
        EXPECT_NEAR(U3(vel[c]), W_U3[i], 1.0e-2) << configs[c].label << " U3 -> Wilson, s=" << s;
      }
    }
  }
  std::cout << std::flush;
#endif
}

// ====================================================================================================================
// Diagnostics: exploratory sweeps that write CSVs / print convergence for inspection (the PeripheryDiagnostic
// suite). These make no pass/fail assertions -- they generate data for offline analysis (e.g. SPD-loss studies).
// ====================================================================================================================

// These diagnostics run entirely on the host (they build host views, fill element-by-element, and write CSVs), so
// the periphery is pinned to the host execution space -- keeping every view in HostSpace on both host and GPU
// builds. (The default-execution-space Periphery would be Cuda under CUDA and mismatch the host views/kernels.)
using HostPeriphery = PeripheryT<Kokkos::DefaultHostExecutionSpace>;

// Run a single sphere with a given periphery
auto run_periphery_rpyc(const double& viscosity, const int num_spheres,
                        const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& sphere_positions,
                        const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& sphere_radii,
                        const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& sphere_forces,
                        const Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>& sphere_velocities,
                        const std::shared_ptr<HostPeriphery>& periphery_ptr) {
  // Apply the RPYC kernel
  mundy::mbody::apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                  sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

  // Now engage the periphery
  const size_t num_surface_nodes = periphery_ptr->get_num_nodes();
  auto surface_positions = periphery_ptr->get_surface_positions();
  auto surface_weights = periphery_ptr->get_quadrature_weights();
  auto surface_normals = periphery_ptr->get_surface_normals();
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_radii("surface_radii", num_surface_nodes);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_velocities("surface_velocities",
                                                                                  3 * num_surface_nodes);
  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_forces("surface_forces", 3 * num_surface_nodes);
  Kokkos::deep_copy(surface_radii, 0.0);

  // Apply the RPY kernel from spheres to periphery
  mundy::mbody::apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, sphere_positions, surface_positions,
                                  sphere_radii, surface_radii, sphere_forces, surface_velocities);

  // Apply no-slip boundary conditions
  // This is done in two steps: first, we compute the forces on the periphery necessary to enforce no-slip
  // Then we evaluate the flow these forces induce on the spheres.
  periphery_ptr->compute_surface_forces(surface_velocities, surface_forces);
  mundy::mbody::apply_stokes_double_layer_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_nodes,
                                                 num_spheres, surface_positions, sphere_positions, surface_normals,
                                                 surface_weights, surface_forces, sphere_velocities);

  mundy::Vector3d final_sphere_velocity(sphere_velocities(0), sphere_velocities(1), sphere_velocities(2));
  mundy::Vector3d final_sphere_force(sphere_forces(0), sphere_forces(1), sphere_forces(2));
  return std::make_tuple(final_sphere_velocity, final_sphere_force);
}

/// \brief The periphery's contribution C to the 3x3 mobility of one RPY sphere (radius a) at \p position.
///
/// Column k is the velocity run_periphery_rpyc returns for a unit force e_k from zero initial velocity, so the dry drag
/// is excluded; with one sphere, its sphere-to-sphere RPYC term skips r = 0 and contributes nothing.
std::array<std::array<double, 3>, 3> sphere_periphery_correction(const double viscosity, const double sphere_radius,
                                                                 const mundy::Vector3d& position,
                                                                 const std::shared_ptr<HostPeriphery>& periphery_ptr) {
  using view_t = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  std::array<std::array<double, 3>, 3> correction;
  for (int k = 0; k < 3; ++k) {
    view_t sphere_positions("sphere_positions", 3);
    view_t sphere_radii("sphere_radii", 1);
    view_t sphere_forces("sphere_forces", 3);
    view_t sphere_velocities("sphere_velocities", 3);
    for (int d = 0; d < 3; ++d) {
      sphere_positions(d) = position[d];
    }
    sphere_radii(0) = sphere_radius;
    sphere_forces(k) = 1.0;
    auto [velocity, force] = run_periphery_rpyc(viscosity, 1, sphere_positions, sphere_radii, sphere_forces,
                                                sphere_velocities, periphery_ptr);
    for (int i = 0; i < 3; ++i) {
      correction[i][k] = velocity[i];
    }
  }
  return correction;
}

/// \brief Frobenius norms of the antisymmetric part A - A^T and of A itself.
std::pair<double, double> antisymmetric_and_total_norms(const std::array<std::array<double, 3>, 3>& A) {
  double asym = 0.0;
  double total = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      asym += (A[i][j] - A[j][i]) * (A[i][j] - A[j][i]);
      total += A[i][j] * A[i][j];
    }
  }
  return {std::sqrt(asym), std::sqrt(total)};
}

/// \brief Build a host periphery on the Gauss-Legendre sphere rule of the given order and radius.
std::shared_ptr<HostPeriphery> make_sphere_periphery(const int order, const double periphery_radius,
                                                     const double viscosity, const bool outward_normal) {
  auto [points_vec, weights_vec, normals_vec] = sphere_quadrature(order, periphery_radius, outward_normal);
  auto periphery_ptr = std::make_shared<HostPeriphery>(weights_vec.size(), viscosity);
  periphery_ptr->set_surface_positions(points_vec.data())
      .set_quadrature_weights(weights_vec.data())
      .set_surface_normals(normals_vec.data(), outward_normal);
  periphery_ptr->build_inverse_self_interaction_matrix();
  return periphery_ptr;
}

// By Lorentz reciprocity the confined mobility is symmetric, so its antisymmetric part is discretization error. At the
// cavity center the correction is C = -3/(8 pi viscosity R) I to leading order in a/R (Happel & Brenner Eq. 4-22.11;
// the O((a/R)^2) remainder is ~5e-5 here). viscosity * C must not depend on the viscosity, and C must not depend on the
// normal orientation.
TEST(PeripheryTest, SphereInPeripheryMobilitySymmetry) {
  const double sphere_radius = 0.1;
  const double periphery_radius = 13.5;
  const std::vector<int> orders = {8, 16, 24};
  const mundy::Vector3d center(0.0, 0.0, 0.0);
  const mundy::Vector3d off_center = mundy::Vector3d(1.0, 2.0, 3.0) * (0.5 * periphery_radius / std::sqrt(14.0));

  std::vector<std::array<std::array<double, 3>, 3>> scaled_corrections;  // viscosity * C at the finest order
  for (const double viscosity : {1.0, 0.5305}) {
    const double c_exact = -3.0 / (8.0 * M_PI * viscosity * periphery_radius);
    for (const int order : orders) {
      auto periphery_ptr = make_sphere_periphery(order, periphery_radius, viscosity, /*outward_normal=*/false);

      const auto C0 = sphere_periphery_correction(viscosity, sphere_radius, center, periphery_ptr);
      double center_err = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          const double expected = (i == j) ? c_exact : 0.0;
          center_err = std::max(center_err, std::fabs(C0[i][j] - expected) / std::fabs(c_exact));
        }
      }

      const auto C1 = sphere_periphery_correction(viscosity, sphere_radius, off_center, periphery_ptr);
      const auto [asym, total] = antisymmetric_and_total_norms(C1);
      const double asym_rel = asym / total;

      std::cout << "SphereInPeripheryMobilitySymmetry: viscosity=" << viscosity << " order=" << order
                << " N=" << periphery_ptr->get_num_nodes() << "  center C rel_err=" << center_err
                << "  off-center |C - C^T|/|C|=" << asym_rel << std::endl;

      if (order == orders.back()) {
        EXPECT_LT(center_err, 2.0e-4) << "center correction must match -3/(8 pi viscosity R) I";
        EXPECT_LT(asym_rel, 2.0e-5) << "off-center correction must be symmetric";
        auto scaled = C1;
        for (auto& row : scaled) {
          for (auto& entry : row) {
            entry *= viscosity;
          }
        }
        scaled_corrections.push_back(scaled);
      }
    }
  }

  // viscosity * C is viscosity-independent.
  const double reference_norm = antisymmetric_and_total_norms(scaled_corrections[0]).second;
  for (size_t c = 1; c < scaled_corrections.size(); ++c) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(scaled_corrections[c][i][j], scaled_corrections[0][i][j], 1.0e-10 * reference_norm)
            << "viscosity * C must not depend on the viscosity, entry (" << i << ", " << j << ")";
      }
    }
  }

  // The mobility is physical, so it must not depend on which way the periphery normals point.
  {
    const double viscosity = 0.5305;
    const int order = 16;
    const auto inward_ptr = make_sphere_periphery(order, periphery_radius, viscosity, /*outward_normal=*/false);
    const auto outward_ptr = make_sphere_periphery(order, periphery_radius, viscosity, /*outward_normal=*/true);
    const auto C_inward = sphere_periphery_correction(viscosity, sphere_radius, off_center, inward_ptr);
    const auto C_outward = sphere_periphery_correction(viscosity, sphere_radius, off_center, outward_ptr);
    // J + T flips with the normals but N does not, so the two agree to the discrete flux sum_s w_s n_s . u_s of the
    // slip data, which is small but not zero -- hence a tolerance above roundoff.
    const double norm = antisymmetric_and_total_norms(C_inward).second;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(C_outward[i][j], C_inward[i][j], 1.0e-8 * norm)
            << "inward and outward periphery normals must give the same mobility, entry (" << i << ", " << j << ")";
      }
    }
  }
}

// DIAGNOSTIC (writes a CSV sweep for inspection; no pass/fail assertions). The full 3x3 sphere mobility M inside the
// periphery and its antisymmetric part, as functions of the periphery quadrature order, the viscosity, the direction,
// and the distance from the center (equivalently, from the wall).
TEST(PeripheryDiagnostic, SphereQuadPeripheryMobilitySymmetry) {
  const double sphere_radius = 0.1;
  const double periphery_radius = 13.5;
  const double dr = 0.1;

  const std::string output_filename = "SphereQuadPeripheryMobilitySymmetry.csv";
  std::ofstream mfile(output_filename);
  ASSERT_TRUE(mfile.is_open()) << "Could not open file " << output_filename;
  mfile << std::setprecision(16);
  mfile << "Order, NumSurfacePoints, Viscosity, Direction, R, WallDistance, M00, M01, M02, M10, M11, M12, M20, M21, "
           "M22, AsymAbs, AsymRelM, AsymRelC\n";

  const std::vector<std::string> direction_names = {"x", "xyz123"};
  const std::vector<mundy::Vector3d> directions = {mundy::Vector3d(1.0, 0.0, 0.0),
                                                   mundy::Vector3d(1.0, 2.0, 3.0) / std::sqrt(14.0)};
  for (int order = 4; order <= 36; order += 4) {
    for (const double viscosity : {1.0, 0.5305}) {
      auto periphery_ptr = make_sphere_periphery(order, periphery_radius, viscosity, /*outward_normal=*/false);
      for (size_t idir = 0; idir < directions.size(); ++idir) {
        double max_asym_rel_c_bulk = 0.0;  // over r <= 0.8 R
        for (double r = 0.0; r < periphery_radius; r += dr) {
          const auto C = sphere_periphery_correction(viscosity, sphere_radius, directions[idir] * r, periphery_ptr);
          auto M = C;
          for (int i = 0; i < 3; ++i) {
            M[i][i] += 1.0 / (6.0 * M_PI * viscosity * sphere_radius);  // the dry drag
          }
          const auto [asym_m, total_m] = antisymmetric_and_total_norms(M);
          const auto [asym_c, total_c] = antisymmetric_and_total_norms(C);
          mfile << order << ", " << periphery_ptr->get_num_nodes() << ", " << viscosity << ", " << direction_names[idir]
                << ", " << r << ", " << periphery_radius - r;
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              mfile << ", " << M[i][j];
            }
          }
          mfile << ", " << asym_m << ", " << asym_m / total_m << ", " << asym_c / total_c << "\n";
          if (r <= 0.8 * periphery_radius) {
            max_asym_rel_c_bulk = std::max(max_asym_rel_c_bulk, asym_c / total_c);
          }
        }
        std::cout << "SphereQuadPeripheryMobilitySymmetry: order=" << order << " N=" << periphery_ptr->get_num_nodes()
                  << " viscosity=" << viscosity << " direction=" << direction_names[idir]
                  << "  max_{r <= 0.8R} |C - C^T|/|C| = " << max_asym_rel_c_bulk << std::endl;
      }
    }
  }
  mfile.close();
}

// DIAGNOSTIC (writes a CSV sweep for inspection; no pass/fail assertions).
TEST(PeripheryDiagnostic, SphereQuadPeripheryRPYC) {
  // Apply an RPY kernel to a single sphere with unit force towards a periphery at some distance, measuring when
  // we get the non-SPD nature of the mobility for a given periphery quadrature scheme
  const int num_spheres = 1;
  const double viscosity = 0.5305;
  const double sphere_radius = 0.1;
  const double periphery_radius = 13.5;
  const double dr = 0.1;

  // Write the results directly to a CSV file
  std::string output_filename = "SphereQuadPeripheryRPYC.csv";
  std::ofstream mfile(output_filename);
  ASSERT_TRUE(mfile.is_open()) << "Could not open file " << output_filename;

  // Create a set of tests where we move the sphere towards the periphery in various directions
  std::vector<std::string> test_types = {"SphereQuadXFx", "SphereQuadYFy", "SphereQuadZFz", "Random"};
  std::vector<mundy::Vector3d> director = {mundy::Vector3d(1.0, 0.0, 0.0), mundy::Vector3d(0.0, 1.0, 0.0),
                                           mundy::Vector3d(0.0, 0.0, 1.0), mundy::Vector3d(1.0, 1.0, 1.0)};
  std::vector<mundy::Vector3d> force_director = {mundy::Vector3d(1.0, 0.0, 0.0), mundy::Vector3d(0.0, 1.0, 0.0),
                                                 mundy::Vector3d(0.0, 0.0, 1.0), mundy::Vector3d(1.0, 1.0, 1.0)};

  mfile << "TestType, Order, NumSurfacePoints, Viscosity, SphereRadius, PeripheryRadius, X, Y, Z, Fx, Fy, Fz, Vx, "
           "Vy, Vz, FdotV\n";

  // Build a periphery of a given spectral order (or later, from an external file)
  for (int order = 2; order <= 36; order += 4) {
    std::cout << "Order = " << order;
    // Create a periphery according to the constructor
    const bool outward_normal = false;
    auto [points_vec, weights_vec, normals_vec] = sphere_quadrature(order, periphery_radius, outward_normal);
    // Create the periphery object
    const size_t num_surface_nodes = weights_vec.size();
    std::cout << " | N = " << num_surface_nodes << std::endl;
    auto periphery_ptr = std::make_shared<HostPeriphery>(num_surface_nodes, viscosity);
    periphery_ptr->set_surface_positions(points_vec.data())
        .set_quadrature_weights(weights_vec.data())
        .set_surface_normals(normals_vec.data(), outward_normal);
    periphery_ptr->build_inverse_self_interaction_matrix();

    // Loop over the test types (what direction we are moving and the force director)
    for (auto itest = 0; itest < test_types.size(); ++itest) {
      // Loop over the sphere positions moving from the center to the periphery
      // If the test is the random one, pick a quadrature point at random to look at
      int random_index = 0;
      if (itest == 3) {
        openrand::Philox rng = make_philox(0, 0);
        random_index = rng.uniform(0, static_cast<int>(num_surface_nodes) - 1);
      }
      // Sweep from the center outward and flag the first radius where the tested force does negative work
      // (F.V < 0) -- i.e. where the discrete periphery mobility stops being SPD for this direction.
      bool spd_lost = false;
      for (double r = 0.0; r < periphery_radius; r += dr) {
        // Allocate the sphere views inside the loop so each sweep step starts from a clean state.
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_positions("sphere_positions",
                                                                                      3 * num_spheres);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_radii("sphere_radii", 1 * num_spheres);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_forces("sphere_forces", 3 * num_spheres);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_velocities("sphere_velocities",
                                                                                       3 * num_spheres);

        // If the test is the random one, pick a quadrature point at random to look at
        if (itest == 3) {
          director[itest][0] = points_vec[3 * random_index];
          director[itest][1] = points_vec[3 * random_index + 1];
          director[itest][2] = points_vec[3 * random_index + 2];
          // Normalize the director for use later with the periphery radius
          // director[itest] = director[itest] / director[itest].norm();
          director[itest] = director[itest] / norm(director[itest]);
          // Force in the same direction as this
          force_director[itest][0] = director[itest][0];
          force_director[itest][1] = director[itest][1];
          force_director[itest][2] = director[itest][2];
        }
        sphere_positions(0) = director[itest][0] * r;
        sphere_positions(1) = director[itest][1] * r;
        sphere_positions(2) = director[itest][2] * r;
        sphere_radii(0) = sphere_radius;
        // Unit force along the X-axis
        sphere_forces(0) = force_director[itest][0];
        sphere_forces(1) = force_director[itest][1];
        sphere_forces(2) = force_director[itest][2];
        // Include the dry velocity along the X-axis
        sphere_velocities(0) = force_director[itest][0] / (6.0 * M_PI * viscosity * sphere_radius);
        sphere_velocities(1) = force_director[itest][1] / (6.0 * M_PI * viscosity * sphere_radius);
        sphere_velocities(2) = force_director[itest][2] / (6.0 * M_PI * viscosity * sphere_radius);

        // Run the RPYC-Periphery interaction
        auto [final_sphere_velocity, final_sphere_force] = run_periphery_rpyc(
            viscosity, num_spheres, sphere_positions, sphere_radii, sphere_forces, sphere_velocities, periphery_ptr);
        // F.V is the rate of work (the mobility quadratic form F^T M F); a negative value flags a non-SPD mobility.
        const double force_dot_velocity = final_sphere_force[0] * final_sphere_velocity[0] +
                                          final_sphere_force[1] * final_sphere_velocity[1] +
                                          final_sphere_force[2] * final_sphere_velocity[2];
        mfile << test_types[itest] << ", " << order << ", " << num_surface_nodes << ", " << viscosity << ", "
              << sphere_radius << ", " << periphery_radius << ", " << sphere_positions(0) << ", " << sphere_positions(1)
              << ", " << sphere_positions(2) << ", " << final_sphere_force[0] << ", " << final_sphere_force[1] << ", "
              << final_sphere_force[2] << ", " << final_sphere_velocity[0] << ", " << final_sphere_velocity[1] << ", "
              << final_sphere_velocity[2] << ", " << force_dot_velocity << std::endl;

        // Report the first radius at which SPD-ness is lost (negative work) for this order/direction.
        if (!spd_lost && force_dot_velocity < 0.0) {
          spd_lost = true;
          std::cout << "  SPD-ness lost: " << test_types[itest] << ", order = " << order << ", r = " << r << "/"
                    << periphery_radius << std::endl;
        }
      }
    }
  }
  mfile.close();
}

// DIAGNOSTIC (writes a CSV sweep for inspection; no pass/fail assertions).
TEST(PeripheryDiagnostic, ExternalQuadPeripheryRPYC) {
  // Apply an RPY kernel to a single sphere with unit force towards a periphery at some distance, measuring when
  // we get the non-SPD nature of the mobility for a given periphery quadrature scheme (external schemes read in)

  const int num_spheres = 1;
  const double viscosity = 0.5305;
  const double sphere_radius = 0.1;
  const double periphery_radius = 13.5;  // [TEMP] match R135 quadrature files
  const double dr = 0.1;

  // Write the results directly to a CSV file
  std::string output_filename = "ExternalQuadPeripheryRPYC.csv";
  std::ofstream mfile(output_filename);
  ASSERT_TRUE(mfile.is_open()) << "Could not open file " << output_filename;

  // Create a set of tests where we move the sphere towards the periphery in various directions
  std::vector<std::string> test_types = {"SphereQuadXFx", "SphereQuadYFy", "SphereQuadZFz", "Random"};
  // [TEMP] populate from hfirouznia's files (both families, all available sizes)
  std::vector<std::string> external_names, external_quadrature_points_filename, external_quadrature_weights_filename,
      external_quadrature_normals_filename;
  std::vector<int> external_num_quadrature_points;
  for (const auto& q : temp_quad_files()) {
    external_names.push_back(q.label);
    external_num_quadrature_points.push_back(static_cast<int>(q.n));
    external_quadrature_points_filename.push_back(q.pts);
    external_quadrature_weights_filename.push_back(q.wgt);
    external_quadrature_normals_filename.push_back(q.nrm);
  }
  std::vector<mundy::Vector3d> director = {mundy::Vector3d(1.0, 0.0, 0.0), mundy::Vector3d(0.0, 1.0, 0.0),
                                           mundy::Vector3d(0.0, 0.0, 1.0), mundy::Vector3d(1.0, 1.0, 1.0)};
  std::vector<mundy::Vector3d> force_director = {mundy::Vector3d(1.0, 0.0, 0.0), mundy::Vector3d(0.0, 1.0, 0.0),
                                                 mundy::Vector3d(0.0, 0.0, 1.0), mundy::Vector3d(1.0, 1.0, 1.0)};

  mfile << "TestType, QuadFile, NumSurfacePoints, Viscosity, SphereRadius, PeripheryRadius, X, Y, Z, Fx, Fy, Fz, Vx, "
           "Vy, Vz, FdotV\n";

  // Loop over the external names, rather than the order of the scheme, for these
  for (auto itype = 0; itype < external_names.size(); ++itype) {
    auto periphery_ptr = std::make_shared<HostPeriphery>(external_num_quadrature_points[itype], viscosity);
    periphery_ptr->set_surface_positions(external_quadrature_points_filename[itype].c_str())
        .set_quadrature_weights(external_quadrature_weights_filename[itype].c_str())
        .set_surface_normals(external_quadrature_normals_filename[itype], temp_quad_files_outward_normal);
    periphery_ptr->build_inverse_self_interaction_matrix();

    // Loop over the test types (what direction we are moving and the force director)
    for (auto itest = 0; itest < test_types.size(); ++itest) {
      // Loop over the sphere positions moving from the center to the periphery
      // If the test is the random one, pick a quadrature point at random to look at
      int random_index = 0;
      if (itest == 3) {
        openrand::Philox rng = make_philox(0, 0);
        random_index = rng.uniform(0, static_cast<int>(external_num_quadrature_points[itype]) - 1);
      }
      // Sweep from the center outward and flag the first radius where the tested force does negative work
      // (F.V < 0) -- i.e. where the discrete periphery mobility stops being SPD for this direction.
      bool spd_lost = false;
      for (double r = 0.0; r < periphery_radius; r += dr) {
        // Allocate the sphere views inside the loop so each sweep step starts from a clean state.
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_positions("sphere_positions",
                                                                                      3 * num_spheres);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_radii("sphere_radii", 1 * num_spheres);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_forces("sphere_forces", 3 * num_spheres);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_velocities("sphere_velocities",
                                                                                       3 * num_spheres);

        // If the test is the random one, pick a quadrature point at random to look at
        if (itest == 3) {
          auto surface_positions = periphery_ptr->get_surface_positions();
          director[itest][0] = surface_positions[3 * random_index];
          director[itest][1] = surface_positions[3 * random_index + 1];
          director[itest][2] = surface_positions[3 * random_index + 2];
          // Normalize the director for use later with the periphery radius
          // director[itest] = director[itest] / director[itest].norm();
          director[itest] = director[itest] / norm(director[itest]);
          // Force in the same direction as this
          force_director[itest][0] = director[itest][0];
          force_director[itest][1] = director[itest][1];
          force_director[itest][2] = director[itest][2];
        }

        sphere_positions(0) = director[itest][0] * r;
        sphere_positions(1) = director[itest][1] * r;
        sphere_positions(2) = director[itest][2] * r;
        sphere_radii(0) = sphere_radius;
        // Unit force along the X-axis
        sphere_forces(0) = force_director[itest][0];
        sphere_forces(1) = force_director[itest][1];
        sphere_forces(2) = force_director[itest][2];
        // Include the dry velocity along the X-axis
        sphere_velocities(0) = force_director[itest][0] / (6.0 * M_PI * viscosity * sphere_radius);
        sphere_velocities(1) = force_director[itest][1] / (6.0 * M_PI * viscosity * sphere_radius);
        sphere_velocities(2) = force_director[itest][2] / (6.0 * M_PI * viscosity * sphere_radius);

        // Run the RPYC-Periphery interaction
        auto [final_sphere_velocity, final_sphere_force] = run_periphery_rpyc(
            viscosity, num_spheres, sphere_positions, sphere_radii, sphere_forces, sphere_velocities, periphery_ptr);
        // F.V is the rate of work (the mobility quadratic form F^T M F); a negative value flags a non-SPD mobility.
        const double force_dot_velocity = final_sphere_force[0] * final_sphere_velocity[0] +
                                          final_sphere_force[1] * final_sphere_velocity[1] +
                                          final_sphere_force[2] * final_sphere_velocity[2];
        mfile << test_types[itest] << ", " << external_names[itype] << ", " << external_num_quadrature_points[itype]
              << ", " << viscosity << ", " << sphere_radius << ", " << periphery_radius << ", " << sphere_positions(0)
              << ", " << sphere_positions(1) << ", " << sphere_positions(2) << ", " << final_sphere_force[0] << ", "
              << final_sphere_force[1] << ", " << final_sphere_force[2] << ", " << final_sphere_velocity[0] << ", "
              << final_sphere_velocity[1] << ", " << final_sphere_velocity[2] << ", " << force_dot_velocity
              << std::endl;

        // Report the first radius at which SPD-ness is lost (negative work) for this quad file/direction.
        if (!spd_lost && force_dot_velocity < 0.0) {
          spd_lost = true;
          std::cout << "  SPD-ness lost: " << test_types[itest] << ", quad = " << external_names[itype] << ", r = " << r
                    << "/" << periphery_radius << std::endl;
        }
      }
    }
  }
  mfile.close();
}
//@}

}  // namespace

}  // namespace mbody

}  // namespace mundy