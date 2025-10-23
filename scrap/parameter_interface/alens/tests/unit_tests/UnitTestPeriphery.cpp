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

// External
#include <gmock/gmock.h>      // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>      // for TEST, ASSERT_NO_THROW, etc
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <fstream>  // for std::ofstream
#include <numeric>  // for std::accumulate
#include <vector>   // for std::vector

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_alens/periphery/Periphery.hpp>  // for gen_sphere_quadrature
#include <mundy_math/Vector3.hpp>               // for Vector3

namespace mundy {

namespace alens {

namespace periphery {

namespace {

//! \name Helper functions
//@{

/// \brief Compute the slope of a log-log plot of y(x) vs x
double compute_log_log_slope(const std::vector<double> &x, const std::vector<double> &y) {
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

double fd_l2_norm(const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &a) {
  const size_t num_elements = a.extent(0);
  double l2_norm = 0.0;
  for (size_t i = 0; i < num_elements; ++i) {
    l2_norm += a(i) * a(i);
  }
  return std::sqrt(l2_norm / static_cast<double>(num_elements));
}

double fd_l2_norm_difference(const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &a,
                             const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &b) {
  const size_t num_elements = a.extent(0);
  double l2_norm = 0.0;
  for (size_t i = 0; i < num_elements; ++i) {
    const double diff = a(i) - b(i);
    l2_norm += diff * diff;
  }
  return std::sqrt(l2_norm / static_cast<double>(num_elements));
}

template <int field_dim>
double spherical_l2_norm(const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &a,
                         const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
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
    std::function<std::array<Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>, 3>(const int order)>;

/// \brief A function for generating a Kokkos vector based on a quadrature rule
using QuadVectorFunc = std::function<Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>(
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights)>;

/// \brief A function that accepts a quadrature rule, an in field, and an out field.
using QuadInOutFunc = std::function<void(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &in_field,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &out_field)>;

/// \brief A struct for storing the results of a convergence study
struct ConvergenceResults {
  std::vector<double> num_quadrature_points;
  std::vector<double> abs_error;
  double abs_slope;
  std::vector<double> rel_error;
  double rel_slope;
};

/// \brief Perform a convergence study against an expected result
ConvergenceResults perform_convergence_study(const double &viscosity, const QuadGenerationFunc &quad_gen,
                                             const QuadInOutFunc &func, const QuadVectorFunc &input_field_gen,
                                             const QuadVectorFunc &expected_results_gen) {
  std::vector<double> stashed_num_quadrature_points;
  std::vector<double> stashed_error_abs;
  std::vector<double> stashed_error_rel;
  for (int order = 2; order <= 16; order *= 2) {
    // Generate the quadrature rule
    auto [points, weights, normals] = quad_gen(order);
    const size_t num_quadrature_points = weights.extent(0);

    // Fetch the input field and the expected results
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field =
        input_field_gen(points, normals, weights);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results =
        expected_results_gen(points, normals, weights);

    // Apply the function and compute the finite difference error
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> result_vector("result_vector",
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
ConvergenceResults perform_self_convergence_study(const double &viscosity, const QuadGenerationFunc &quad_gen,
                                                  const QuadInOutFunc &func, const QuadVectorFunc &input_field_gen) {
  // Generate high-order quadrature rule
  auto [points_ho, weights_ho, normals_ho] = quad_gen(64);

  // Fetch the high_order input field and the corresponding expected results
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field_ho =
      input_field_gen(points_ho, normals_ho, weights_ho);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results_ho("expected_results_ho",
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
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field =
        input_field_gen(points, normals, weights);

    // Apply the function and compute the finite difference error
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> result_vector("result_vector",
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
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  // Fill the stokes_double_layer_matrix
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> stokes_double_layer_matrix(
      "stokes_double_layer_matrix", 3 * num_quadrature_points, 3 * num_quadrature_points);
  fill_stokes_double_layer_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                  num_quadrature_points, points, points, normals, weights, stokes_double_layer_matrix);
  // Apply the stokes_double_layer_matrix to a the surface forces
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, stokes_double_layer_matrix, surface_forces, 0.0,
                   surface_velocities);
}

/// Apply the stokes double layer matrix with singularity subtraction to surface forces to get surface velocities
void apply_stokes_double_layer_matrix_ss_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  // Fill the stokes_double_layer_matrix and apply the singularity subtraction
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> stokes_double_layer_matrix(
      "stokes_double_layer_matrix", 3 * num_quadrature_points, 3 * num_quadrature_points);
  fill_stokes_double_layer_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                  num_quadrature_points, points, points, normals, weights, stokes_double_layer_matrix);
  add_singularity_subtraction(Kokkos::DefaultHostExecutionSpace(), stokes_double_layer_matrix);

  // Apply the stokes_double_layer_matrix to a the surface forces
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, stokes_double_layer_matrix, surface_forces, 0.0,
                   surface_velocities);
}

/// \brief Apply the stokes double layer operator to surface forces to get surface velocities
void apply_stokes_double_layer_kernel_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_stokes_double_layer_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                   num_quadrature_points, points, points, normals, weights, surface_forces,
                                   surface_velocities);
}

/// \brief Apply the stokes double layer operator (with singularity subtraction) to surface forces to get surface
/// velocities
void apply_stokes_double_layer_kernel_ss_wrapper(
    const double viscosity, const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
    const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_stokes_double_layer_kernel_ss(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points,
                                      num_quadrature_points, points, points, normals, weights, surface_forces,
                                      surface_velocities);
}

/// \brief Apply the second kind Fredholm integral operator to a function
void apply_skfie_matrix_wrapper(const double viscosity,
                                const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
                                const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
                                const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
                                const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &input_field,
                                const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &output_field) {
  const size_t num_quadrature_points = weights.extent(0);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> M("M", 3 * num_quadrature_points,
                                                                   3 * num_quadrature_points);
  fill_skfie_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points, num_quadrature_points,
                    points, points, normals, weights, M);
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, M, input_field, 0.0, output_field);
}

/// \brief Apply the second kind Fredholm integral operator with singularity subtraction to a function
void apply_skfie_wrapper(const double viscosity,
                         const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
                         const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
                         const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights,
                         const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &input_field,
                         const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &output_field) {
  const size_t num_quadrature_points = weights.extent(0);
  apply_skfie(Kokkos::DefaultHostExecutionSpace(), viscosity, num_quadrature_points, num_quadrature_points, points,
              points, normals, normals, weights, input_field, output_field);
}

/// \brief Solve for the velocity on a bulk point given an imposed slip velocity on the periphery
///
/// Mathematically U_bulk = G_{periphery -> bulk} * M^{-1} * U_slip
void apply_resistance(const double viscosity,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_points,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_normals,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_weights,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_velocities,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_forces,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &bulk_points,
                      const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &bulk_velocities) {
  const size_t num_surface_points = surface_weights.extent(0);
  const size_t num_bulk_points = bulk_points.extent(0) / 3;

  // Fill the SKFIE matrix M
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> M("M", 3 * num_surface_points, 3 * num_surface_points);
  fill_skfie_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_points, num_surface_points,
                    surface_points, surface_points, surface_normals, surface_weights, M);

  // Invert the SKFIE matrix
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> M_inv("M_inv", 3 * num_surface_points,
                                                                       3 * num_surface_points);
  invert_matrix(Kokkos::DefaultHostExecutionSpace(), M, M_inv);

  // F = M^{-1} * U_slip
  KokkosBlas::gemv(Kokkos::DefaultHostExecutionSpace(), "N", 1.0, M_inv, surface_velocities, 0.0, surface_forces);

  // U_bulk = G^{dl}_{periphery -> bulk} * F
  apply_stokes_double_layer_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_points, num_bulk_points,
                                   surface_points, bulk_points, surface_normals, surface_weights, surface_forces,
                                   bulk_velocities);
}

/// \brief A wrapper for the apply_resistance function that tests the error within randomly sampled bulk points
struct ApplyResistanceWrapper {
  ApplyResistanceWrapper(const double &sphere_radius, const size_t num_bulk_points)
      : num_bulk_points_(num_bulk_points), bulk_points_("bulk_points", 3 * num_bulk_points) {
    // Generate random bulk points
    openrand::Philox rng(0, 0);
    for (size_t i = 0; i < num_bulk_points; ++i) {
      const double theta = rng.uniform(0.0, 2.0 * M_PI);
      const double phi = rng.uniform(0.0, M_PI);
      const double r = rng.uniform(0.1 * sphere_radius, 0.2 * sphere_radius);  // Avoid landing on the surface
      bulk_points_(3 * i) = r * std::sin(phi) * std::cos(theta);
      bulk_points_(3 * i + 1) = r * std::sin(phi) * std::sin(theta);
      bulk_points_(3 * i + 2) = r * std::cos(phi);
    }
  }

  ApplyResistanceWrapper(const std::vector<double> &bulk_points)
      : num_bulk_points_(bulk_points.size() / 3), bulk_points_("bulk_points", 3 * num_bulk_points_) {
    // Copy the bulk points
    for (size_t i = 0; i < num_bulk_points_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        bulk_points_(3 * i + j) = bulk_points[3 * i + j];
      }
    }
  }

  ApplyResistanceWrapper(const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &bulk_points)
      : num_bulk_points_(bulk_points.extent(0) / 3), bulk_points_("bulk_points", 3 * num_bulk_points_) {
    // Copy the bulk points
    for (size_t i = 0; i < num_bulk_points_; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        bulk_points_(3 * i + j) = bulk_points(3 * i + j);
      }
    }
  }

  void operator()(const double &viscosity,
                  const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_points,
                  const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_normals,
                  const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &surface_weights,
                  const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &input_field,
                  const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &output_field) {
    // The input field is the slip velocity on the periphery
    // The output field is the velocity on the bulk points
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_forces("surface_forces",
                                                                                 surface_points.extent(0));
    apply_resistance(viscosity, surface_points, surface_normals, surface_weights, input_field, surface_forces,
                     bulk_points_, output_field);
  }

 private:
  double num_bulk_points_;
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_points_;

};  // struct ApplyResistanceWrapper

/// \brief A functor for generate a quadrature rule on the sphere
class SphereQuadFunctor {
 public:
  SphereQuadFunctor(const double sphere_radius, const bool include_pole = false, const bool invert = false)
      : sphere_radius_(sphere_radius), include_pole_(include_pole), invert_(invert) {
  }

  std::array<Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace>, 3> operator()(const int &order) {
    std::vector<double> weights_vec;
    std::vector<double> points_vec;
    std::vector<double> normals_vec;
    gen_sphere_quadrature(order, sphere_radius_, &points_vec, &weights_vec, &normals_vec, include_pole_, invert_);
    const size_t num_quadrature_points = weights_vec.size();

    // Convert the points, weights, and normals to Kokkos views
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> points("points", num_quadrature_points * 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> weights("weights", num_quadrature_points);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> normals("normals", num_quadrature_points * 3);
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
  bool include_pole_;
  bool invert_;
};  // class SphereQuadFunctor

//@}

//! \name Quadrature tests
//@{

TEST(PeripheryTest, SphereQuadBasicChecks) {
  // Test that the sum of the quadrature weights is equal to the surface area of the sphere
  // There's no need to check convergence, as this result should converge to the exact value
  // almost immediately.

  // Define the sphere radius
  const double sphere_radius = 12.34;
  const double sphere_surface_area = 4.0 * M_PI * sphere_radius * sphere_radius;
  for (int order = 1; order <= 16; order *= 2) {
    std::vector<double> weights;
    std::vector<double> points;
    std::vector<double> normals;
    gen_sphere_quadrature(order, sphere_radius, &points, &weights, &normals);

    // Check that the number of quadrature points is correct
    EXPECT_EQ(weights.size() * 3, points.size());
    EXPECT_EQ(weights.size() * 3, normals.size());

    for (size_t i = 0; i < points.size(); i += 3) {
      // Check that the points have the right magnitude
      const double x = points[i];
      const double y = points[i + 1];
      const double z = points[i + 2];
      const double magnitude = std::sqrt(x * x + y * y + z * z);
      EXPECT_NEAR(magnitude, sphere_radius, 1.0e-10);

      // Check that the normals are equal to the normalized points
      const double nx = normals[i];
      const double ny = normals[i + 1];
      const double nz = normals[i + 2];
      const double normal_magnitude = std::sqrt(nx * nx + ny * ny + nz * nz);
      EXPECT_NEAR(normal_magnitude, 1.0, 1.0e-10);
      EXPECT_NEAR(nx, x / sphere_radius, 1.0e-10);
      EXPECT_NEAR(ny, y / sphere_radius, 1.0e-10);
      EXPECT_NEAR(nz, z / sphere_radius, 1.0e-10);
    }

    // Check that the sum of the weights is equal to the surface area of the sphere
    const double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    EXPECT_NEAR(sum_weights, sphere_surface_area, 1.0e-10) << "failed for order = " << order << std::endl;
  }
}

TEST(PeripheryTest, SphereQuadAnalyticallyIntegrableFunctions) {
  // Test that we converge to the correct values for analytically integrable functions
  // If you know of any other functions that can be integrated over the sphere analytically without adding additional
  // dependencies like special functions (e.g., spherical harmonics), let us know and we can add them to this
  // test!

  auto run_test_for_function = [](const std::string &function_name, const double &known_integral,
                                  const std::function<double(double, double, double)> &func_to_integrate) {
    const double sphere_radius = 1.0;

    // Weird behavior can occur for order = 1, so we start at order = 2
    std::vector<double> num_quadrature_points;
    std::vector<double> errors;
    for (int order = 2; order <= 16; order *= 2) {
      std::vector<double> weights;
      std::vector<double> points;
      std::vector<double> normals;
      gen_sphere_quadrature(order, sphere_radius, &points, &weights, &normals);

      // Compute the integral
      double integral = 0.0;
      for (size_t i = 0; i < weights.size(); ++i) {
        const double x = points[3 * i];
        const double y = points[3 * i + 1];
        const double z = points[3 * i + 2];
        integral += weights[i] * func_to_integrate(x, y, z);
      }

      // Stash the error
      const double error = std::fabs(integral - known_integral);
      num_quadrature_points.push_back(static_cast<double>(weights.size()));
      errors.push_back(error);

      std::cout << "function_name = " << function_name << ", order = " << order << ", integral = " << integral
                << ", known_integral = " << known_integral << std::endl;
    }

    // Check that the error converges to zero at the expected rate
    const double slope = compute_log_log_slope(num_quadrature_points, errors);
    std::cout << "function_name = " << function_name << ", slope = " << slope << std::endl;
  };

  // Test the function f(x, y, z) = 1, which integrates to 4 * pi
  run_test_for_function("f(x, y, z) = 1", 4.0 * M_PI, [](double, double, double) { return 1.0; });

  // Test the function f(x, y, z) = x, which integrates to 0
  run_test_for_function("f(x, y, z) = x", 0.0, [](double x, double, double) { return x; });

  // Test the function f(x, y, z) = exp(x-y), which integrates to 2^(3/2) * pi * sinh(sqrt(2))
  // https://math.stackexchange.com/questions/717202/surface-integral-on-unit-sphere
  run_test_for_function("f(x, y, z) = exp(x-y)", 2.0 * std::sqrt(2.0) * M_PI * std::sinh(std::sqrt(2.0)),
                        [](double x, double y, double) { return std::exp(x - y); });

  // Test the function f(x, y, z) = max(0, x, x * cos(t) + y * sin(t)), which integrates
  // to pi + pi / 2 * sqrt( (1 - cos t)^2 + (sin t)^2 )
  // https://math.stackexchange.com/questions/3246357/a-tricky-integration-over-the-unit-sphere
  const double t = M_PI / 4.0;  // Arbitrary value of t simply needs to be in [0, 2pi]
  run_test_for_function(
      "f(x, y, z) = max(0, x, x * cos(t) + y * sin(t))",
      M_PI + M_PI / 2.0 * std::sqrt((1.0 - std::cos(t)) * (1.0 - std::cos(t)) + std::sin(t) * std::sin(t)),
      [&t](double x, double y, double) { return std::max({0.0, x, x * std::cos(t) + y * std::sin(t)}); });
}
//@}

//! \name Periphery auxilary function tests
//@{

TEST(PeripheryTest, KokkosInvertMatrix) {
  // This test tests our understanding of how to use Kokkos to invert a matrix.
  // It explicitly tests that the matrix_inverse function works as expected.

  // Statistically speaking, nearly all randomly generated matrices are invertible, so we'll just generate a random
  // matrix and invert it!

  // Generate a random matrix with random normally distributed entities with mean 0 and variance 1
  const int matrix_size = 10;
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_scratch("matrix_scratch", matrix_size,
                                                                                matrix_size);
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_original("matrix_original", matrix_size,
                                                                                 matrix_size);

  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      matrix_scratch(i, j) = rng.randn<double>();
      matrix_original(i, j) = matrix_scratch(i, j);
    }
  }

  // Invert the matrix
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_inverse("matrix_inverse", matrix_size,
                                                                                matrix_size);
  invert_matrix(Kokkos::DefaultHostExecutionSpace(), matrix_scratch, matrix_inverse);

  // Multiply the original matrix by the inverse matrix and store the result in matrix_scratch
  KokkosBlas::gemm(Kokkos::DefaultHostExecutionSpace(), "N", "N", 1.0, matrix_original, matrix_inverse, 0.0,
                   matrix_scratch);

  // Check that the result (stored in matrix_scratch) is the identity matrix
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      if (i == j) {
        ASSERT_NEAR(matrix_scratch(i, j), 1.0, 1.0e-10) << "i = " << i << ", j = " << j;
      } else {
        ASSERT_NEAR(matrix_scratch(i, j), 0.0, 1.0e-10) << "i = " << i << ", j = " << j;
      }
    }
  }
}

TEST(PeripheryTest, ReadWriteKokkosMatrixToFromFile) {
  // Test that we can write a Kokkos matrix to a file and read it back in

  // Generate a random matrix with random normally distributed entities with mean 0 and variance 1
  const int matrix_size = 10;
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix("matrix", matrix_size, matrix_size);

  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      matrix(i, j) = rng.randn<double>();
    }
  }

  // Write the matrix to a file
  const std::string file_name = "ReadWriteKokkosMatrixToFromFile_matrix.dat";
  write_matrix_to_file(file_name, matrix);

  // Read the matrix back in
  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> matrix_read("matrix_read", matrix_size, matrix_size);
  read_matrix_from_file(file_name, matrix_size, matrix_size, matrix_read);

  // Check that the matrix read in is the same as the original matrix, to machine precision
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      EXPECT_NEAR(matrix(i, j), matrix_read(i, j), 1e-12);
    }
  }
}

TEST(PeripheryTest, ReadWriteKokkosVectorToFromFile) {
  // Test that we can write a Kokkos vector to a file and read it back in

  // Generate a random vector with random normally distributed entities with mean 0 and variance 1
  const int vector_size = 10;
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> vector("vector", vector_size);

  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
  for (int i = 0; i < vector_size; ++i) {
    vector(i) = rng.randn<double>();
  }

  // Write the vector to a file
  const std::string file_name = "ReadWriteKokkosVectorToFromFile_vector.dat";
  write_vector_to_file(file_name, vector);

  // Read the vector back in
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> vector_read("vector_read", vector_size);
  read_vector_from_file(file_name, vector_size, vector_read);

  // Check that the vector read in is the same as the original vector, to machine precision
  for (int i = 0; i < vector_size; ++i) {
    EXPECT_NEAR(vector(i), vector_read(i), 1e-12);
  }
}

TEST(PeripheryTest, RPYKernelThreeSpheres) {
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
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> positions("positions", 9);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> radii("radii", 3);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> forces("forces", 9);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpy("velocities_rpy", 9);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpyc("velocities_rpyc", 9);

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
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> positions("positions", 6);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> radii("radii", 2);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> forces("forces", 6);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpy("velocities_rpy", 6);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_rpyc("velocities_rpyc", 6);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> velocities_stokes("velocities_stokes", 6);

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

TEST(PeripheryTest, StokesDoubleLayerConstantForce) {
  // We have the following identity for flow exterior to a surface D:
  //   int_{partial D} n(y) dot T(x - y) dS(y) = I / 2
  //   T(x - y) = - T(y - x)
  // This is a commonly used to check the convergence of the double layer integral.
  //
  // For the singularity subtraction, we use
  //    int_{partial D} n(y) dot T(x - y) dot (F(x) + F(y)) dS(y)
  //    = int_{partial D} n(y) dot T(x - y) dS(y) dot F(x) + int_{partial D} n(y) dot T(x - y) dot F(y) dS(y)
  //    = F(x) / 2 + int_{partial D} n(y) dot T(x - y) dot F(y) dS(y)
  //    if F(x) = F, then
  //    = F / 2 + int_{partial D} n(y) dot T(x - y) dS(y)  dot F
  //    = F

  // Setup the convergence study
  const double sphere_radius = 12.34;
  const double viscosity = 1.0;

  const QuadGenerationFunc quad_gen = SphereQuadFunctor(sphere_radius);
  const QuadVectorFunc in_field_gen =
      []([[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field("input_field", 3 * weights.extent(0));
        Kokkos::deep_copy(input_field, 1.0);
        return input_field;
      };
  const QuadVectorFunc expected_results_gen =
      []([[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("expected_results",
                                                                                       3 * weights.extent(0));
        Kokkos::deep_copy(expected_results, 0.5);
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
      []([[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
         [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("new_expected_results",
                                                                                       3 * weights.extent(0));
        Kokkos::deep_copy(expected_results, 1.0);
        return expected_results;
      };
  const ConvergenceResults matrix_ss_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_matrix_ss_wrapper, in_field_gen, new_expected_results_gen);

  const ConvergenceResults kernel_ss_results = perform_convergence_study(
      viscosity, quad_gen, apply_stokes_double_layer_kernel_ss_wrapper, in_field_gen, new_expected_results_gen);

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
}

TEST(PeripheryTest, StokesDoubleLayerSmoothForces) {
  // Self-convergence against the integral of a smooth function
  // In this case, we use the smooth function
  auto run_test_for_function =
      [](const std::string &function_name,
         const std::function<std::array<double, 3>(double, double, double)> &func_to_integrate) {
        // Setup the convergence study
        const double sphere_radius = 12.34;
        const double viscosity = 1.0;

        const QuadGenerationFunc quad_gen = SphereQuadFunctor(sphere_radius);
        const QuadVectorFunc in_field_gen =
            [&func_to_integrate](
                [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
                [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
                [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
              Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field("input_field",
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

TEST(PeripheryTest, SKFIERigidBodyMotion) {
  // Test the convergence for a point not on the periphery.
  //
  // The test is based on the following fact:
  //   If the velocity on the periphery is given by v = omega x p for some angular velocity omega in R^3 and p on the
  //   periphery, then the velocity at a point b in the bulk is also v = omega x b.
  //
  // Given the rigid body surface slip velocity U, the velocity at points in the bulk is given by G_{periphery to bulk}
  // M^{-1} U

  // Setup the convergence study
  const double sphere_radius = 1.0;
  const double viscosity = 1.0;
  const size_t num_bulk_points = 8;
  const mundy::math::Vector3d omega(0.0, 0.0, 1.0);
  const double cube_side_half_length = sphere_radius / 3;

  // Setup the bulk points at the corner of the cube with side length cude_side_length
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_points("bulk_points", 3 * num_bulk_points);
  bulk_points(0) = cube_side_half_length;
  bulk_points(1) = cube_side_half_length;
  bulk_points(2) = cube_side_half_length;

  bulk_points(3) = -cube_side_half_length;
  bulk_points(4) = cube_side_half_length;
  bulk_points(5) = cube_side_half_length;

  bulk_points(6) = cube_side_half_length;
  bulk_points(7) = -cube_side_half_length;
  bulk_points(8) = cube_side_half_length;

  bulk_points(12) = cube_side_half_length;
  bulk_points(13) = cube_side_half_length;
  bulk_points(14) = -cube_side_half_length;

  bulk_points(9) = -cube_side_half_length;
  bulk_points(10) = -cube_side_half_length;
  bulk_points(11) = cube_side_half_length;

  bulk_points(15) = -cube_side_half_length;
  bulk_points(16) = cube_side_half_length;
  bulk_points(17) = -cube_side_half_length;

  bulk_points(18) = cube_side_half_length;
  bulk_points(19) = -cube_side_half_length;
  bulk_points(20) = -cube_side_half_length;

  bulk_points(21) = -cube_side_half_length;
  bulk_points(22) = -cube_side_half_length;
  bulk_points(23) = -cube_side_half_length;

  // size_t seed = 1234;
  // size_t counter = 0;
  // openrand::Philox rng(seed, counter);
  // for (size_t i = 0; i < num_bulk_points; ++i) {
  //   const double theta = rng.uniform(0.0, 2.0 * M_PI);
  //   const double phi = rng.uniform(0.0, M_PI);
  //   const double r = rng.uniform(0.1 * sphere_radius, 0.2 * sphere_radius);  // Avoid landing on the surface
  //   bulk_points(3 * i) = r * std::sin(phi) * std::cos(theta);
  //   bulk_points(3 * i + 1) = r * std::sin(phi) * std::sin(theta);
  //   bulk_points(3 * i + 2) = r * std::cos(phi);
  // }

  for (size_t i = 0; i < num_bulk_points; ++i) {
    const double x = bulk_points(3 * i);
    const double y = bulk_points(3 * i + 1);
    const double z = bulk_points(3 * i + 2);
    const mundy::math::Vector3d p(x, y, z);
    const auto v = mundy::math::cross(omega, p);
  }

  // Setup the test
  const bool include_poles = false;
  const bool invert = true;
  const QuadGenerationFunc quad_gen = SphereQuadFunctor(sphere_radius, include_poles, invert);
  const QuadVectorFunc in_field_gen =
      [&omega](const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
               [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
               [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> input_field("input_field", 3 * weights.extent(0));
        for (size_t i = 0; i < weights.extent(0); ++i) {
          const double x = points(3 * i);
          const double y = points(3 * i + 1);
          const double z = points(3 * i + 2);
          const mundy::math::Vector3d p(x, y, z);
          const auto v = mundy::math::cross(omega, p);
          input_field(3 * i) = v[0];
          input_field(3 * i + 1) = v[1];
          input_field(3 * i + 2) = v[2];
        }
        return input_field;
      };
  const QuadVectorFunc expected_results_gen =
      [&omega, &bulk_points, &num_bulk_points](
          [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &points,
          [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &normals,
          [[maybe_unused]] const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &weights) {
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_results("expected_results",
                                                                                       3 * num_bulk_points);
        for (size_t i = 0; i < num_bulk_points; ++i) {
          const double x = bulk_points(3 * i);
          const double y = bulk_points(3 * i + 1);
          const double z = bulk_points(3 * i + 2);
          const mundy::math::Vector3d b(x, y, z);
          const auto v = mundy::math::cross(omega, b);
          expected_results(3 * i) = v[0];
          expected_results(3 * i + 1) = v[1];
          expected_results(3 * i + 2) = v[2];
        }
        return expected_results;
      };

  // Use apply_resistance to map the in_field (surface velocities) to the bulk surface velocities
  ApplyResistanceWrapper apply_resistance_wrapper(bulk_points);
  const ConvergenceResults rigid_body_results =
      perform_convergence_study(viscosity, quad_gen, apply_resistance_wrapper, in_field_gen, expected_results_gen);

  // For now, just print the results
  std::cout << "###########################################################" << std::endl;
  std::cout << "Rigid body motion" << std::endl;
  std::cout << "rigid_body_results.abs_slope = " << rigid_body_results.abs_slope
            << " rel_slope = " << rigid_body_results.rel_slope << std::endl;
  for (size_t i = 0; i < rigid_body_results.num_quadrature_points.size(); ++i) {
    std::cout << "  num_quadrature_points[" << i << "] = " << rigid_body_results.num_quadrature_points[i];
    std::cout << ", abs_error[" << i << "] = " << rigid_body_results.abs_error[i];
    std::cout << ", rel_error[" << i << "] = " << rigid_body_results.rel_error[i] << std::endl;
  }
}

TEST(PeripheryTest, SKFIERigidBodyMotionFromFile) {
  // Test the convergence for a point not on the periphery.
  //
  // The test is based on the following fact:
  //   If the velocity on the periphery is given by v = omega x p for some angular velocity omega in R^3 and p on the
  //   periphery, then the velocity at a point b in the bulk is also v = omega x b.
  //
  // Given the rigid body surface slip velocity U, the velocity at points in the bulk is given by G_{periphery to bulk}
  // M^{-1} U

  // Skip this test until we get the files read in properly
  // GTEST_SKIP() << "Skipping until we get the files read in properly";

  // Setup the convergence study
  const double sphere_radius = 28.0;
  const double viscosity = 1.0;
  const size_t num_bulk_points = 8;
  const mundy::math::Vector3d omega(1.0, 1.0, 1.0);
  const double cube_side_half_length = sphere_radius / 3;

  // Setup the bulk points at the corner of the cube with side length cude_side_length
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_points("bulk_points", 3 * num_bulk_points);
  bulk_points(0) = cube_side_half_length;
  bulk_points(1) = cube_side_half_length;
  bulk_points(2) = cube_side_half_length;

  bulk_points(3) = -cube_side_half_length;
  bulk_points(4) = cube_side_half_length;
  bulk_points(5) = cube_side_half_length;

  bulk_points(6) = cube_side_half_length;
  bulk_points(7) = -cube_side_half_length;
  bulk_points(8) = cube_side_half_length;

  bulk_points(12) = cube_side_half_length;
  bulk_points(13) = cube_side_half_length;
  bulk_points(14) = -cube_side_half_length;

  bulk_points(9) = -cube_side_half_length;
  bulk_points(10) = -cube_side_half_length;
  bulk_points(11) = cube_side_half_length;

  bulk_points(15) = -cube_side_half_length;
  bulk_points(16) = cube_side_half_length;
  bulk_points(17) = -cube_side_half_length;

  bulk_points(18) = cube_side_half_length;
  bulk_points(19) = -cube_side_half_length;
  bulk_points(20) = -cube_side_half_length;

  bulk_points(21) = -cube_side_half_length;
  bulk_points(22) = -cube_side_half_length;
  bulk_points(23) = -cube_side_half_length;

  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_bulk_velocity("expected_bulk_velocity",
                                                                                       3 * num_bulk_points);
  for (size_t i = 0; i < num_bulk_points; ++i) {
    const double x = bulk_points(3 * i);
    const double y = bulk_points(3 * i + 1);
    const double z = bulk_points(3 * i + 2);
    const mundy::math::Vector3d b(x, y, z);
    const auto v = mundy::math::cross(omega, b);
    expected_bulk_velocity(3 * i) = v[0];
    expected_bulk_velocity(3 * i + 1) = v[1];
    expected_bulk_velocity(3 * i + 2) = v[2];
  }

  // Print the bulk points and the expected velocity
  std::cout << "bulk_points = " << std::endl;
  for (size_t i = 0; i < num_bulk_points; ++i) {
    std::cout << "  " << bulk_points(3 * i) << " " << bulk_points(3 * i + 1) << " " << bulk_points(3 * i + 2)
              << std::endl;
  }

  std::cout << "expected_bulk_velocity = " << std::endl;
  for (size_t i = 0; i < num_bulk_points; ++i) {
    std::cout << "  " << expected_bulk_velocity(3 * i) << " " << expected_bulk_velocity(3 * i + 1) << " "
              << expected_bulk_velocity(3 * i + 2) << std::endl;
  }

  std::cout << "###########################################################" << std::endl;
  std::cout << "Rigid body motion from file" << std::endl;
  std::vector<size_t> vec_num_quad_points = {1280, 3840, 5120};  // 15360, 20480, 30720
  for (size_t i = 0; i < vec_num_quad_points.size(); ++i) {
    // Read in the normals, points, and weights to kokkos views
    const size_t num_quad_points = vec_num_quad_points[i];
    std::string filename_normals = "./sphere_triangle_normals_" + std::to_string(num_quad_points) + ".dat";
    std::string filename_points = "./sphere_triangle_points_" + std::to_string(num_quad_points) + ".dat";
    std::string filename_weights = "./sphere_triangle_weights_" + std::to_string(num_quad_points) + ".dat";
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> normals("normal", 3 * num_quad_points);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> points("points", 3 * num_quad_points);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> weights("weights", num_quad_points);
    read_vector_from_file(filename_normals, 3 * num_quad_points, normals);
    read_vector_from_file(filename_points, 3 * num_quad_points, points);
    read_vector_from_file(filename_weights, num_quad_points, weights);

    // Setup the test
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_slip_velocity("surface_slip_velocity",
                                                                                        3 * num_quad_points);
    for (size_t j = 0; j < weights.extent(0); ++j) {
      const double x = points(3 * j);
      const double y = points(3 * j + 1);
      const double z = points(3 * j + 2);
      const mundy::math::Vector3d p(x, y, z);
      const auto v = mundy::math::cross(omega, p);
      surface_slip_velocity(3 * j) = v[0];
      surface_slip_velocity(3 * j + 1) = v[1];
      surface_slip_velocity(3 * j + 2) = v[2];
    }

    // Use apply_resistance to map the surface velocities to the bulk surface velocities
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_forces("surface_forces", points.extent(0));
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_velocity("bulk_velocity", 3 * num_bulk_points);
    Kokkos::deep_copy(surface_forces, 0.0);
    Kokkos::deep_copy(bulk_velocity, 0.0);
    apply_resistance(viscosity, points, normals, weights, surface_slip_velocity, surface_forces, bulk_points,
                     bulk_velocity);

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

TEST(PeripheryTest, SKFIESelfConvFromFile) {
  // Test the convergence for the flow induced by N points within the bulk of the sphere
  // Assign to each point in the bulk a smoothly varying force field
  // f(x,y,z) = (y, -x, z) / sqrt(x^2 + y^2 + z^2)

  // Skip this test until we get the files read in properly
  GTEST_SKIP() << "Skipping until we get the files read in properly";

  // Setup the convergence study
  const double sphere_radius = 28.0;
  const double viscosity = 1.0;
  const size_t num_bulk_points = 1000;

  // Setup the bulk points at the corner of the cube with side length cude_side_length
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_points("bulk_points", 3 * num_bulk_points);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_forces("bulk_forces", 3 * num_bulk_points);
  size_t seed = 1234;
  size_t counter = 0;
  openrand::Philox rng(seed, counter);
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

  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> expected_bulk_velocity("expected_bulk_velocity",
                                                                                       3 * num_bulk_points);

  std::cout << "###########################################################" << std::endl;
  std::cout << "Self-conv from file" << std::endl;
  std::vector<size_t> vec_num_quad_points = {5120, 3840, 1280};  // 30720, 20480, 15360,
  for (size_t i = 0; i < vec_num_quad_points.size(); ++i) {
    // Read in the normals, points, and weights to kokkos views
    const size_t num_quad_points = vec_num_quad_points[i];
    std::string filename_normals = "./sphere_triangle_normals_" + std::to_string(num_quad_points) + ".dat";
    std::string filename_points = "./sphere_triangle_points_" + std::to_string(num_quad_points) + ".dat";
    std::string filename_weights = "./sphere_triangle_weights_" + std::to_string(num_quad_points) + ".dat";
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> normals("normal", 3 * num_quad_points);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> points("points", 3 * num_quad_points);
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> weights("weights", num_quad_points);
    read_vector_from_file(filename_normals, 3 * num_quad_points, normals);
    read_vector_from_file(filename_points, 3 * num_quad_points, points);
    read_vector_from_file(filename_weights, num_quad_points, weights);

    // Compute the surface slip velocity induced by the bulk forces
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_slip_velocity("surface_slip_velocity",
                                                                                        3 * num_quad_points);
    apply_stokes_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, bulk_points, points, bulk_forces,
                        surface_slip_velocity);

    // Use apply_resistance to map the surface velocities to the bulk surface velocities
    // The surface forces are unknown and will be computed by apply_resistance
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_forces("surface_forces", points.extent(0));
    Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> bulk_velocity("bulk_velocity", 3 * num_bulk_points);
    Kokkos::deep_copy(surface_forces, 0.0);
    Kokkos::deep_copy(bulk_velocity, 0.0);
    apply_resistance(viscosity, points, normals, weights, surface_slip_velocity, surface_forces, bulk_points,
                     bulk_velocity);

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
  // Check that the second kind Fredholm integral equation matrix is invertable

  const double viscosity = 1.0;
  const double sphere_radius = 12.34;
  const size_t spectral_order = 12;
  const bool include_poles = false;
  for (bool invert : {true, false}) {
    auto [host_points, host_weights, host_normals] =
        SphereQuadFunctor(sphere_radius, include_poles, invert)(spectral_order);
    const size_t num_surface_nodes = host_weights.extent(0);

    // Fill the self-interaction matrix and take its inverse
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> M("M", 3 * num_surface_nodes, 3 * num_surface_nodes);
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> M_original("M", 3 * num_surface_nodes,
                                                                              3 * num_surface_nodes);
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> M_inv("M_inv", 3 * num_surface_nodes,
                                                                         3 * num_surface_nodes);
    fill_skfie_matrix(Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_nodes, num_surface_nodes, host_points,
                      host_points, host_normals, host_weights, M);

    // The inverse function will replace M with its pivot matrix, so we need to stash the original matrix
    Kokkos::deep_copy(M_original, M);
    invert_matrix(Kokkos::DefaultHostExecutionSpace(), M, M_inv);

    // Multiply the matrix by its inverse and check that it is the m_m_inv matrix
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> m_m_inv("m_m_inv", 3 * num_surface_nodes,
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

// Run a single sphere with a given periphery
auto run_periphery_rpyc(const double &viscosity, const int num_spheres,
                        const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_positions,
                        const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_radii,
                        const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_forces,
                        const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_velocities,
                        const std::shared_ptr<mundy::alens::periphery::Periphery> &periphery_ptr) {
  // Apply the RPYC kernel
  mundy::alens::periphery::apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, sphere_positions,
                                             sphere_positions, sphere_radii, sphere_radii, sphere_forces,
                                             sphere_velocities);

  // Now engage the periphery
  const size_t num_surface_nodes = periphery_ptr->get_num_nodes();
  auto surface_positions = periphery_ptr->get_surface_positions();
  auto surface_weights = periphery_ptr->get_quadrature_weights();
  auto surface_normals = periphery_ptr->get_surface_normals();
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_radii("surface_radii", num_surface_nodes);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_velocities("surface_velocities",
                                                                                   3 * num_surface_nodes);
  Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> surface_forces("surface_forces", 3 * num_surface_nodes);
  Kokkos::deep_copy(surface_radii, 0.0);

  // Apply the RPY kernel from spheres to periphery
  mundy::alens::periphery::apply_rpyc_kernel(Kokkos::DefaultHostExecutionSpace(), viscosity, sphere_positions,
                                             surface_positions, sphere_radii, surface_radii, sphere_forces,
                                             surface_velocities);

  // Apply no-slip boundary conditions
  // This is done in two steps: first, we compute the forces on the periphery necessary to enforce no-slip
  // Then we evaluate the flow these forces induce on the spheres.
  periphery_ptr->compute_surface_forces(surface_velocities, surface_forces);
  mundy::alens::periphery::apply_stokes_double_layer_kernel(
      Kokkos::DefaultHostExecutionSpace(), viscosity, num_surface_nodes, num_spheres, surface_positions,
      sphere_positions, surface_normals, surface_weights, surface_forces, sphere_velocities);

  mundy::math::Vector3d final_sphere_velocity(sphere_velocities(0), sphere_velocities(1), sphere_velocities(2));
  mundy::math::Vector3d final_sphere_force(sphere_forces(0), sphere_forces(1), sphere_forces(2));
  return std::make_tuple(final_sphere_velocity, final_sphere_force);
}

TEST(PeripheryRPYC, SphereQuadPeripheryRPYC) {
  // Apply an RPY kernel to a single sphere with unit force towards a periphery at some distance, measuring when
  // we get the non-SPD nature of the mobility for a given periphery quadrature scheme
  const int num_spheres = 1;
  const double viscosity = 1.0 / (3.0 * M_PI);
  const double sphere_radius = 0.5;
  const double periphery_radius = 28.0;
  const double dr = 0.1;

  // Write the results directly to a CSV file
  std::string output_filename = "SphereQuadPeripheryRPYC.csv";
  std::ofstream mfile(output_filename);
  if (!mfile.is_open()) {
    std::cerr << "Could not open file " << output_filename << std::endl;
    return;
  }

  // Create a set of tests where we move the sphere towards the periphery in various directions
  std::vector<std::string> test_types = {"SphereQuadXFx", "SphereQuadYFy", "SphereQuadZFz", "Random"};
  std::vector<mundy::math::Vector3d> director = {
      mundy::math::Vector3d(1.0, 0.0, 0.0), mundy::math::Vector3d(0.0, 1.0, 0.0), mundy::math::Vector3d(0.0, 0.0, 1.0),
      mundy::math::Vector3d(1.0, 1.0, 1.0)};
  std::vector<mundy::math::Vector3d> force_director = {
      mundy::math::Vector3d(1.0, 0.0, 0.0), mundy::math::Vector3d(0.0, 1.0, 0.0), mundy::math::Vector3d(0.0, 0.0, 1.0),
      mundy::math::Vector3d(1.0, 1.0, 1.0)};

  mfile << "TestType, Order, NumSurfacePoints, Viscosity, SphereRadius, PeripheryRadius, X, Y, Z, Fx, Fy, Fz, Vx, "
           "Vy, Vz\n";

  // Build a periphery of a given spectral order (or later, from an external file)
  for (int order = 2; order <= 16; order *= 2) {
    std::cout << "Order = " << order << std::endl;
    // Create a periphery according to the constructor
    std::vector<double> points_vec;
    std::vector<double> weights_vec;
    std::vector<double> normals_vec;
    const bool invert = true;
    const bool include_poles = false;
    const size_t spectral_order = order;
    mundy::alens::periphery::gen_sphere_quadrature(spectral_order, periphery_radius, &points_vec, &weights_vec,
                                                   &normals_vec, include_poles, invert);
    // Create the periphery object
    const size_t num_surface_nodes = weights_vec.size();
    auto periphery_ptr = std::make_shared<mundy::alens::periphery::Periphery>(num_surface_nodes, viscosity);
    periphery_ptr->set_surface_positions(points_vec.data())
        .set_quadrature_weights(weights_vec.data())
        .set_surface_normals(normals_vec.data());
    periphery_ptr->build_inverse_self_interaction_matrix(false);

    // Loop over the test types (what direction we are moving and the force director)
    for (auto itest = 0; itest < test_types.size(); ++itest) {
      // Loop over the sphere positions moving from the center to the periphery
      // If the test is the random one, pick a quadrature point at random to look at
      int random_index = 0;
      if (itest == 3) {
        openrand::Philox rng(0, 0);
        random_index = rng.uniform(0, static_cast<int>(num_surface_nodes) - 1);
      }
      for (double r = periphery_radius - 5.0; r < periphery_radius; r += dr) {
        // Setup the kokkos vectors for position, radius, force, and velocity, do it here to prevent things from leaking
        // out of scope...
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_positions("sphere_positions",
                                                                                       3 * num_spheres);
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_radii("sphere_radii", 1 * num_spheres);
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_forces("sphere_forces", 3 * num_spheres);
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_velocities("sphere_velocities",
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
        mfile << test_types[itest] << ", " << order << ", " << num_surface_nodes << ", " << viscosity << ", "
              << sphere_radius << ", " << periphery_radius << ", " << sphere_positions(0) << ", " << sphere_positions(1)
              << ", " << sphere_positions(2) << ", " << final_sphere_force[0] << ", " << final_sphere_force[1] << ", "
              << final_sphere_force[2] << ", " << final_sphere_velocity[0] << ", " << final_sphere_velocity[1] << ", "
              << final_sphere_velocity[2] << std::endl;
      }
    }
  }
  mfile.close();
}

TEST(PeripheryRPYC, ExternalQuadPeripheryRPYC) {
  // Apply an RPY kernel to a single sphere with unit force towards a periphery at some distance, measuring when
  // we get the non-SPD nature of the mobility for a given periphery quadrature scheme (external schemes read in)
  const int num_spheres = 1;
  const double viscosity = 1.0 / (3.0 * M_PI);
  const double sphere_radius = 0.5;
  const double periphery_radius = 28.0;
  const double dr = 0.1;

  // Write the results directly to a CSV file
  std::string output_filename = "ExternalQuadPeripheryRPYC.csv";
  std::ofstream mfile(output_filename);
  if (!mfile.is_open()) {
    std::cerr << "Could not open file " << output_filename << std::endl;
    return;
  }

  // Create a set of tests where we move the sphere towards the periphery in various directions
  std::vector<std::string> test_types = {"SphereQuadXFx", "SphereQuadYFy", "SphereQuadZFz", "Random"};
  std::vector<std::string> external_names = {"Ndiv3Nelm1280", "Ndiv3Nelm3840", "Ndiv3Nelm5120"};
  std::vector<int> external_num_quadrature_points = {1280, 3840, 5120};
  std::vector<std::string> external_quadrature_points_filename = {
      "./sphere_triangle_points_1280.dat", "./sphere_triangle_points_3840.dat", "./sphere_triangle_points_5120.dat"};
  std::vector<std::string> external_quadrature_weights_filename = {
      "./sphere_triangle_weights_1280.dat", "./sphere_triangle_weights_3840.dat", "./sphere_triangle_weights_5120.dat"};
  std::vector<std::string> external_quadrature_normals_filename = {
      "./sphere_triangle_normals_1280.dat", "./sphere_triangle_normals_3840.dat", "./sphere_triangle_normals_5120.dat"};
  std::vector<mundy::math::Vector3d> director = {
      mundy::math::Vector3d(1.0, 0.0, 0.0), mundy::math::Vector3d(0.0, 1.0, 0.0), mundy::math::Vector3d(0.0, 0.0, 1.0),
      mundy::math::Vector3d(1.0, 1.0, 1.0)};
  std::vector<mundy::math::Vector3d> force_director = {
      mundy::math::Vector3d(1.0, 0.0, 0.0), mundy::math::Vector3d(0.0, 1.0, 0.0), mundy::math::Vector3d(0.0, 0.0, 1.0),
      mundy::math::Vector3d(1.0, 1.0, 1.0)};

  mfile << "TestType, QuadFile, NumSurfacePoints, Viscosity, SphereRadius, PeripheryRadius, X, Y, Z, Fx, Fy, Fz, Vx, "
           "Vy, Vz\n";

  // Loop over the external names, rather than the order of the scheme, for these
  for (auto itype = 0; itype < external_names.size(); ++itype) {
    auto periphery_ptr =
        std::make_shared<mundy::alens::periphery::Periphery>(external_num_quadrature_points[itype], viscosity);
    periphery_ptr->set_surface_positions(external_quadrature_points_filename[itype].c_str())
        .set_quadrature_weights(external_quadrature_weights_filename[itype].c_str())
        .set_surface_normals(external_quadrature_normals_filename[itype].c_str());
    periphery_ptr->build_inverse_self_interaction_matrix(false);

    // Loop over the test types (what direction we are moving and the force director)
    for (auto itest = 0; itest < test_types.size(); ++itest) {
      // Loop over the sphere positions moving from the center to the periphery
      // If the test is the random one, pick a quadrature point at random to look at
      int random_index = 0;
      if (itest == 3) {
        openrand::Philox rng(0, 0);
        random_index = rng.uniform(0, static_cast<int>(external_num_quadrature_points[itype]) - 1);
      }
      for (double r = periphery_radius - 5.0; r < periphery_radius; r += dr) {
        // Setup the kokkos vectors for position, radius, force, and velocity, do it here to prevent things from
        // leaking out of scope...
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_positions("sphere_positions",
                                                                                       3 * num_spheres);
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_radii("sphere_radii", 1 * num_spheres);
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_forces("sphere_forces", 3 * num_spheres);
        Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> sphere_velocities("sphere_velocities",
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
        mfile << test_types[itest] << ", " << external_names[itype] << ", " << external_num_quadrature_points[itype]
              << ", " << viscosity << ", " << sphere_radius << ", " << periphery_radius << ", " << sphere_positions(0)
              << ", " << sphere_positions(1) << ", " << sphere_positions(2) << ", " << final_sphere_force[0] << ", "
              << final_sphere_force[1] << ", " << final_sphere_force[2] << ", " << final_sphere_velocity[0] << ", "
              << final_sphere_velocity[1] << ", " << final_sphere_velocity[2] << std::endl;
      }
    }
  }
  mfile.close();
}
//@}

}  // namespace

}  // namespace periphery

}  // namespace alens

}  // namespace mundy
