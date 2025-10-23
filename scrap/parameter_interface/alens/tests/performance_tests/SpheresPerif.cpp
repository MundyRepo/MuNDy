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
#include <openrand/philox.h>  // for openrand::Philox

// C++ core
#include <fstream>  // for std::ofstream
#include <numeric>  // for std::accumulate
#include <vector>   // for std::vector

// Kokkos and Kokkos-Kernels
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

// Trilinos
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_alens/periphery/Periphery.hpp>  // for gen_sphere_quadrature

using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

namespace mundy {

namespace alens {

namespace periphery {

namespace {

void init_spheres_on_device(const double periphery_radius, const double sphere_radius_min,
                            const double sphere_radius_max,
                            Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &sphere_positions,
                            Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &sphere_velocities,
                            Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &sphere_forces,
                            Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &sphere_radii) {
  const size_t num_spheres = sphere_radii.extent(0);
  size_t seed = 1234;
  const double periphery_radius_sq = periphery_radius * periphery_radius;
  Kokkos::parallel_for(
      "init_spheres", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_spheres),
      KOKKOS_LAMBDA(const size_t i) {
        openrand::Philox rng(seed, i);
        double distance_from_origin_sq = 2 * periphery_radius_sq;
        while (distance_from_origin_sq > periphery_radius_sq) {
          sphere_positions(3 * i) = (2 * rng.rand<double>() - 1.0) * periphery_radius;
          sphere_positions(3 * i + 1) = (2 * rng.rand<double>() - 1.0) * periphery_radius;
          sphere_positions(3 * i + 2) = (2 * rng.rand<double>() - 1.0) * periphery_radius;
          distance_from_origin_sq = sphere_positions(3 * i) * sphere_positions(3 * i) +
                                    sphere_positions(3 * i + 1) * sphere_positions(3 * i + 1) +
                                    sphere_positions(3 * i + 2) * sphere_positions(3 * i + 2);
        }
        sphere_velocities(3 * i) = 0.0;
        sphere_velocities(3 * i + 1) = 0.0;
        sphere_velocities(3 * i + 2) = 0.0;
        sphere_forces(3 * i) = 0.0;
        sphere_forces(3 * i + 1) = 0.0;
        sphere_forces(3 * i + 2) = 0.0;
        sphere_radii(i) = rng.rand<double>() * (sphere_radius_max - sphere_radius_min) + sphere_radius_min;
      });
}

void apply_force_field(Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions,
                       Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces) {
  const size_t num_spheres = sphere_positions.extent(0) / 3;
  Kokkos::parallel_for(
      "apply_force_field", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_spheres),
      KOKKOS_LAMBDA(const size_t i) {
        // f(t) = [-x_1(t) + x_2(t), x_0(t) - x_2(t), -x_0(t) + x_1(t)]
        sphere_forces(3 * i) = -sphere_positions(3 * i + 1) + sphere_positions(3 * i + 2);
        sphere_forces(3 * i + 1) = sphere_positions(3 * i) - sphere_positions(3 * i + 2);
        sphere_forces(3 * i + 2) = -sphere_positions(3 * i) + sphere_positions(3 * i + 1);
      });
}

void apply_elastic_collision_with_periphery(
    const double periphery_radius, Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions,
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities,
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii) {
  const size_t num_spheres = sphere_radii.extent(0);
  Kokkos::parallel_for(
      "apply_inelastic_collision_with_periphery", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_spheres),
      KOKKOS_LAMBDA(const size_t i) {
        const double r = sphere_radii(i);
        const double x = sphere_positions(3 * i);
        const double y = sphere_positions(3 * i + 1);
        const double z = sphere_positions(3 * i + 2);
        const double distance_from_origin = std::sqrt(x * x + y * y + z * z);
        const double overlap = periphery_radius - (distance_from_origin + r);
        if (overlap < 0.0) {
          const double inv_distance_from_origin = 1.0 / distance_from_origin;
          const double normal_x = x * inv_distance_from_origin;
          const double normal_y = y * inv_distance_from_origin;
          const double normal_z = z * inv_distance_from_origin;
          const double dot_product = sphere_velocities(3 * i) * normal_x + sphere_velocities(3 * i + 1) * normal_y +
                                     sphere_velocities(3 * i + 2) * normal_z;
          sphere_velocities(3 * i) -= 2.0 * dot_product * normal_x;
          sphere_velocities(3 * i + 1) -= 2.0 * dot_product * normal_y;
          sphere_velocities(3 * i + 2) -= 2.0 * dot_product * normal_z;

          sphere_positions(3 * i) = x + overlap * normal_x;
          sphere_positions(3 * i + 1) = y + overlap * normal_y;
          sphere_positions(3 * i + 2) = z + overlap * normal_z;
        }
      });
}

void apply_hertzian_contact_between_spheres(
    const double youngs_modulus, const double poisson_ratio,
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions,
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces,
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii) {
  const size_t num_spheres = sphere_radii.extent(0);
  const double effective_youngs_modulus =
      (youngs_modulus * youngs_modulus) / (youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio +
                                           youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio);
  const double four_thirds = 4.0 / 3.0;

  Kokkos::parallel_for(
      "apply_hertzian_contact_between_spheres",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_spheres, num_spheres}),
      KOKKOS_LAMBDA(const size_t t, const size_t s) {
        // Avoid double counting and self-interaction
        if (t < s) {
          const double r_t = sphere_radii(t);
          const double r_s = sphere_radii(s);
          const double x_t = sphere_positions(3 * t);
          const double y_t = sphere_positions(3 * t + 1);
          const double z_t = sphere_positions(3 * t + 2);
          const double x_s = sphere_positions(3 * s);
          const double y_s = sphere_positions(3 * s + 1);
          const double z_s = sphere_positions(3 * s + 2);

          const double source_to_target_x = x_t - x_s;
          const double source_to_target_y = y_t - y_s;
          const double source_to_target_z = z_t - z_s;

          const double distance_between_centers =
              std::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                        source_to_target_z * source_to_target_z);
          const double signed_separation_distance = distance_between_centers - r_t - r_s;
          if (signed_separation_distance < 0.0) {
            const double effective_radius = (r_t * r_s) / (r_t + r_s);
            const double normal_force_magnitude_scaled =
                four_thirds * effective_youngs_modulus * std::sqrt(effective_radius) *
                std::pow(-signed_separation_distance, 1.5) / distance_between_centers;
            sphere_forces(3 * t) += normal_force_magnitude_scaled * source_to_target_x;
            sphere_forces(3 * t + 1) += normal_force_magnitude_scaled * source_to_target_y;
            sphere_forces(3 * t + 2) += normal_force_magnitude_scaled * source_to_target_z;
            sphere_forces(3 * s) -= normal_force_magnitude_scaled * source_to_target_x;
            sphere_forces(3 * s + 1) -= normal_force_magnitude_scaled * source_to_target_y;
            sphere_forces(3 * s + 2) -= normal_force_magnitude_scaled * source_to_target_z;
          }
        }
      });
}

void update_sphere_positions(const double time_step,
                             Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions,
                             Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities) {
  const size_t num_spheres = sphere_positions.extent(0) / 3;
  Kokkos::parallel_for(
      "update_sphere_positions", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_spheres),
      KOKKOS_LAMBDA(const size_t i) {
        sphere_positions(3 * i) += time_step * sphere_velocities(3 * i);
        sphere_positions(3 * i + 1) += time_step * sphere_velocities(3 * i + 1);
        sphere_positions(3 * i + 2) += time_step * sphere_velocities(3 * i + 2);
      });
}

double max_speed(const Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> &sphere_velocities) {
  const size_t num_spheres = sphere_velocities.extent(0) / 3;
  double max_speed = 0.0;
  Kokkos::parallel_reduce(
      "max_speed", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_spheres),
      KOKKOS_LAMBDA(const size_t i, double &max_speed_local) {
        const double speed = std::sqrt(sphere_velocities(3 * i) * sphere_velocities(3 * i) +
                                       sphere_velocities(3 * i + 1) * sphere_velocities(3 * i + 1) +
                                       sphere_velocities(3 * i + 2) * sphere_velocities(3 * i + 2));
        max_speed_local = speed > max_speed_local ? speed : max_speed_local;
      },
      Kokkos::Max<double>(max_speed));
  return max_speed;
}

void write_spheres_to_vtp(const std::string &filename,
                          const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_positions_host,
                          const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_velocities_host,
                          const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_forces_host,
                          const Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace> &sphere_radius_host) {
  const size_t num_spheres = sphere_positions_host.extent(0) / 3;
  std::ofstream file(filename);
  if (!file.is_open()) {
    // Throw an exception
    std::stringstream ss;
    ss << "Failed to open file: " << filename;
    throw std::runtime_error(ss.str());
    return;
  }

  // Buffer the output
  std::ostringstream buffer;

  // VTP file header
  buffer << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  buffer << "<PolyData>\n";

  buffer << "<Piece NumberOfPoints=\"" << num_spheres << "\" NumberOfLines=\"0\">\n";

  // Write points
  buffer << "<Points>\n<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (size_t i = 0; i < num_spheres; i++) {
    buffer << sphere_positions_host(3 * i + 0) << " " << sphere_positions_host(3 * i + 1) << " "
           << sphere_positions_host(3 * i + 2) << "\n";
  }
  buffer << "</DataArray>\n</Points>\n";

  // Write sphere point data
  buffer << "<PointData Scalars=\"sphere_gids\">\n";

  buffer << "<DataArray type=\"UInt32\" Name=\"sphere_gids\" format=\"ascii\">\n";
  for (size_t i = 0; i < num_spheres; i++) {
    buffer << i << "\n";
  }
  buffer << "</DataArray>\n";

  buffer << "<DataArray type=\"Float32\" Name=\"radius\" format=\"ascii\">\n";
  for (size_t i = 0; i < num_spheres; i++) {
    buffer << sphere_radius_host(i) << "\n";
  }
  buffer << "</DataArray>\n";

  buffer << "<DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (size_t i = 0; i < num_spheres; i++) {
    buffer << sphere_velocities_host(3 * i + 0) << " " << sphere_velocities_host(3 * i + 1) << " "
           << sphere_velocities_host(3 * i + 2) << "\n";
  }
  buffer << "</DataArray>\n";

  buffer << "<DataArray type=\"Float32\" Name=\"force\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (size_t i = 0; i < num_spheres; i++) {
    buffer << sphere_forces_host(3 * i + 0) << " " << sphere_forces_host(3 * i + 1) << " "
           << sphere_forces_host(3 * i + 2) << "\n";
  }
  buffer << "</DataArray>\n</PointData>\n";

  buffer << "</Piece>\n";
  buffer << "</PolyData>\n";
  buffer << "</VTKFile>\n";

  // Write the buffer to the file
  file << buffer.str();
  file.close();
}

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

}  // namespace

}  // namespace periphery

}  // namespace alens

}  // namespace mundy

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    // Simulation of N spheres in a spherical periphery with reflection boundary conditions for collisions.
    // const double viscosity = 1.0;
    // const double youngs_modulus = 10000.0;
    // const double poisson_ratio = 0.3;
    // const double periphery_radius = 100.0;
    // const double sphere_radius_min = 1.0;
    // const double sphere_radius_max = 3.0;
    // const double num_spheres = 30000;
    // const double time_step = 0.00001;
    // const size_t num_time_steps = 1000 / time_step;
    // const size_t num_equilibriation_steps = 1000;
    // const size_t io_frequency = 0.001 / time_step;
    const double viscosity = 1.0;
    const double youngs_modulus = 1000.0;
    const double poisson_ratio = 0.3;
    const double periphery_radius = 28.0;
    const double sphere_radius_min = 0.1;
    const double sphere_radius_max = 0.11;
    const double num_spheres = 1305;
    const double time_step = 0.000005;
    const size_t num_time_steps = 101;
    const size_t num_equilibriation_steps = 1000;
    const size_t io_frequency = 20;

    // Setup the periphery
    const size_t spectral_order = 28;
    const bool include_poles = false;
    const bool invert = true;
    auto [host_points, host_weights, host_normals] =
        mundy::alens::periphery::SphereQuadFunctor(periphery_radius, include_poles, invert)(spectral_order);
    const size_t num_surface_nodes = host_weights.extent(0);
    std::cout << "Num surface nodes: " << num_surface_nodes << std::endl;
    auto perif = mundy::alens::periphery::Periphery(num_surface_nodes, viscosity);
    perif.set_surface_positions(host_points).set_surface_normals(host_normals).set_quadrature_weights(host_weights);
    const bool write_to_file = false;
    perif.build_inverse_self_interaction_matrix(write_to_file);

    auto surface_positions = perif.get_surface_positions();
    auto surface_normals = perif.get_surface_normals();
    auto surface_weights = perif.get_quadrature_weights();
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_radii("surface_radii", num_surface_nodes);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_velocities("surface_velocities",
                                                                                     3 * num_surface_nodes);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> surface_forces("surface_forces",
                                                                                 3 * num_surface_nodes);
    Kokkos::deep_copy(surface_radii, 0.0);

    // Setup the spheres
    size_t seed = 1234;
    size_t counter = 0;
    openrand::Philox rng(seed, counter);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_positions("sphere_positions", 3 * num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_velocities("sphere_velocities",
                                                                                    3 * num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_forces("sphere_forces", 3 * num_spheres);
    Kokkos::View<double *, Kokkos::LayoutLeft, DeviceMemorySpace> sphere_radii("sphere_radii", num_spheres);
    auto sphere_positions_host = Kokkos::create_mirror_view(sphere_positions);
    auto sphere_velocities_host = Kokkos::create_mirror_view(sphere_velocities);
    auto sphere_forces_host = Kokkos::create_mirror_view(sphere_forces);
    auto sphere_radii_host = Kokkos::create_mirror_view(sphere_radii);

    mundy::alens::periphery::init_spheres_on_device(periphery_radius, sphere_radius_min, sphere_radius_max,
                                                    sphere_positions, sphere_velocities, sphere_forces, sphere_radii);

    // Equilibriate the system
    for (size_t t = 0; t < num_equilibriation_steps; ++t) {
      // Collision with the periphery
      mundy::alens::periphery::apply_elastic_collision_with_periphery(periphery_radius, sphere_positions,
                                                                      sphere_velocities, sphere_radii);

      // Collision between spheres
      mundy::alens::periphery::apply_hertzian_contact_between_spheres(youngs_modulus, poisson_ratio, sphere_positions,
                                                                      sphere_forces, sphere_radii);

      // Use local drag during equilibriation
      mundy::alens::periphery::apply_local_drag(DeviceExecutionSpace(), viscosity, sphere_velocities, sphere_forces,
                                                sphere_radii);

      // x(t + dt) = x(t) + dt * v(t)
      mundy::alens::periphery::update_sphere_positions(time_step, sphere_positions, sphere_velocities);

      // Zero out the forces and velocities
      Kokkos::deep_copy(sphere_forces, 0.0);
      Kokkos::deep_copy(sphere_velocities, 0.0);
    }

    // Run the simulation
    Kokkos::Timer timer;
    Kokkos::Timer main_loop_timer;
    for (size_t t = 0; t < num_time_steps; ++t) {
      timer.reset();

      // Zero out the forces and velocities
      Kokkos::deep_copy(sphere_forces, 0.0);
      Kokkos::deep_copy(sphere_velocities, 0.0);
      Kokkos::deep_copy(surface_velocities, 0.0);
      Kokkos::deep_copy(surface_forces, 0.0);

      // f(t) = [-x_1(t) + x_2(t), x_0(t) - x_2(t), -x_0(t) + x_1(t)]
      mundy::alens::periphery::apply_force_field(sphere_positions, sphere_forces);

      // Collision with the periphery
      mundy::alens::periphery::apply_elastic_collision_with_periphery(periphery_radius, sphere_positions,
                                                                      sphere_velocities, sphere_radii);

      // Collision between spheres
      mundy::alens::periphery::apply_hertzian_contact_between_spheres(youngs_modulus, poisson_ratio, sphere_positions,
                                                                      sphere_forces, sphere_radii);

      // Apply the RPY kernel from spheres to spheres
      mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, sphere_positions,
                                                sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

      // Apply the RPY kernel from spheres to periphery
      mundy::alens::periphery::apply_rpy_kernel(DeviceExecutionSpace(), viscosity, sphere_positions, surface_positions,
                                                sphere_radii, surface_radii, sphere_forces, surface_velocities);

      // Apply no-slip boundary conditions
      // This is done in two steps: first, we compute the forces on the periphery necessary to enforce no-slip
      // Then we evaluate the flow these forces induce on the spheres.
      perif.compute_surface_forces(surface_velocities, surface_forces);

      // If we evaluate the flow these forces induce on the periphery, do they satisfy no-slip?
      Kokkos::View<double **, Kokkos::LayoutLeft, DeviceMemorySpace> M("Mnew", 3 * num_surface_nodes,
                                                                       3 * num_surface_nodes);
      mundy::alens::periphery::fill_skfie_matrix(DeviceExecutionSpace(), viscosity, num_surface_nodes,
                                                 num_surface_nodes, surface_positions, surface_positions,
                                                 surface_normals, surface_weights, M);
      KokkosBlas::gemv(DeviceExecutionSpace(), "N", 1.0, M, surface_forces, 1.0, surface_velocities);
      MUNDY_THROW_REQUIRE(std::abs(mundy::alens::periphery::max_speed(surface_velocities)) < 1.0e-10,
                          std::runtime_error, "No-slip boundary conditions not satisfied");

      mundy::alens::periphery::apply_weighted_stokes_kernel(DeviceExecutionSpace(), viscosity, surface_positions,
                                                            sphere_positions, surface_forces, surface_weights,
                                                            sphere_velocities);

      // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
      mundy::alens::periphery::apply_local_drag(DeviceExecutionSpace(), viscosity, sphere_velocities, sphere_forces,
                                                sphere_radii);

      // x(t + dt) = x(t) + dt * v(t)
      mundy::alens::periphery::update_sphere_positions(time_step, sphere_positions, sphere_velocities);

      // Write the spheres to a VTP file
      if (t % io_frequency == 0) {
        std::cout << "Writing spheres to file: " << t
                  << "\t |\t max_speed: " << mundy::alens::periphery::max_speed(sphere_velocities)
                  << "\t |\t tps (avg): " << static_cast<double>(t) / main_loop_timer.seconds() << std::endl;
        Kokkos::deep_copy(sphere_positions_host, sphere_positions);
        Kokkos::deep_copy(sphere_velocities_host, sphere_velocities);
        Kokkos::deep_copy(sphere_forces_host, sphere_forces);
        Kokkos::deep_copy(sphere_radii_host, sphere_radii);
        mundy::alens::periphery::write_spheres_to_vtp("SphereSimulationWithPeriphery_" + std::to_string(t) + ".vtp",
                                                      sphere_positions_host, sphere_velocities_host, sphere_forces_host,
                                                      sphere_radii_host);
      }
    }
  }

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
