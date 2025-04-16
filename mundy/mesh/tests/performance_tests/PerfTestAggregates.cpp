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

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy libs
#include <mundy_math/Vector3.hpp>      // for mundy::math::Vector3
#include <mundy_mesh/Aggregate.hpp>    // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace {

struct TimingResults {
  double force_time;
  double velocity_time;
  double move_time;
  double overall_time;
};

std::pair<TimingResults, TimingResults> run_test_for_fields(const stk::mesh::BulkData& bulk_data,
                                                            stk::mesh::Part& sphere_part,
                                                            stk::mesh::Field<double>& node_center_field,
                                                            stk::mesh::Field<double>& node_force_field,
                                                            stk::mesh::Field<double>& node_velocity_field,
                                                            stk::mesh::Field<double>& elem_radius_field) {
  double dt = 1e-5;
  double viscosity = 0.1;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  constexpr double one_over_6pi = 1.0 / (6.0 * pi);
  const double one_over_6pi_mu = one_over_6pi / viscosity;

  // begin test for fields
  Kokkos::Timer overall_fields_timer;

  // Apply a random force to each sphere (not truly random, but good enough for this test)
  Kokkos::Timer field_force_timer;
  stk::mesh::for_each_entity_run(
      bulk_data, stk::topology::NODE_RANK, sphere_part,
      [&node_force_field]([[maybe_unused]] const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& node) {
        double* force = stk::mesh::field_data(node_force_field, node);
        force[0] = 0.1 * (rand() % 10);
        force[1] = 0.1 * (rand() % 10);
        force[2] = 0.1 * (rand() % 10);
      });
  double field_force_time = field_force_timer.seconds();

  // Compute the velocity of each sphere according to drag v = f / (6 * pi * r * mu)
  Kokkos::Timer field_velocity_timer;
  stk::mesh::for_each_entity_run(bulk_data, stk::topology::ELEM_RANK, sphere_part,
                                 [one_over_6pi_mu, &node_force_field, &node_velocity_field, &elem_radius_field](
                                     const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& sphere) {
                                   stk::mesh::Entity node = bulk_data.begin_nodes(sphere)[0];
                                   double* force = stk::mesh::field_data(node_force_field, node);
                                   double* velocity = stk::mesh::field_data(node_velocity_field, node);
                                   double radius = stk::mesh::field_data(elem_radius_field, sphere)[0];
                                   double inv_radius = 1.0 / radius;
                                   velocity[0] = one_over_6pi_mu * force[0] * inv_radius;
                                   velocity[1] = one_over_6pi_mu * force[1] * inv_radius;
                                   velocity[2] = one_over_6pi_mu * force[2] * inv_radius;
                                 });
  double field_velocity_time = field_velocity_timer.seconds();

  // Move the spheres according to their velocity
  Kokkos::Timer field_move_timer;
  stk::mesh::for_each_entity_run(bulk_data, stk::topology::NODE_RANK, sphere_part,
                                 [dt, &node_center_field, &node_velocity_field]([[maybe_unused]] const stk::mesh::BulkData& bulk_data,
                                                                                const stk::mesh::Entity& node) {
                                   double* center = stk::mesh::field_data(node_center_field, node);
                                   double* velocity = stk::mesh::field_data(node_velocity_field, node);
                                   center[0] += dt * velocity[0];
                                   center[1] += dt * velocity[1];
                                   center[2] += dt * velocity[2];
                                 });
  double field_move_time = field_move_timer.seconds();

  double overall_fields_time = overall_fields_timer.seconds();
  // end test for fields

  // begin test for accessors

  Kokkos::Timer overall_accessors_timer;

  // Create the accessors
  auto center_accessor = Vector3FieldComponent(node_center_field);
  auto force_accessor = Vector3FieldComponent(node_force_field);
  auto velocity_accessor = Vector3FieldComponent(node_velocity_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);

  // Create an aggregate for the spheres
  const auto sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
                               .add_component<CENTER, stk::topology::NODE_RANK>(center_accessor)
                               .add_component<FORCE, stk::topology::NODE_RANK>(force_accessor)
                               .add_component<VELOCITY, stk::topology::NODE_RANK>(velocity_accessor)
                               .add_component<RADIUS, stk::topology::ELEM_RANK>(radius_accessor);

  // Apply a random force to each sphere
  Kokkos::Timer acc_force_timer;
  sphere_data.for_each([](auto& sphere_view) {
    auto force = sphere_view.template get<FORCE>(0);
    force.set(0.1 * (rand() % 10), 0.1 * (rand() % 10), 0.1 * (rand() % 10));
  });
  double acc_force_time = acc_force_timer.seconds();

  // Compute the velocity of each sphere according to drag v = f / (6 * pi * r * mu)
  Kokkos::Timer acc_velocity_timer;
  sphere_data.for_each([one_over_6pi_mu](auto& sphere_view) {
    auto force = sphere_view.template get<FORCE>(0);
    auto velocity = sphere_view.template get<VELOCITY>(0);
    auto radius = sphere_view.template get<RADIUS>();
    velocity = one_over_6pi_mu * force / radius[0];
  });
  double acc_velocity_time = acc_velocity_timer.seconds();

  // Move the spheres according to their velocity
  Kokkos::Timer acc_move_timer;
  sphere_data.for_each([dt](auto& sphere_view) {
    auto center = sphere_view.template get<CENTER>(0);
    auto velocity = sphere_view.template get<VELOCITY>(0);
    center += dt * velocity;
  });
  double acc_move_time = acc_move_timer.seconds();

  double overall_accessors_time = overall_accessors_timer.seconds();
  // end test for accessors

  // Stash and return the timing results
  TimingResults fields_results{field_force_time, field_velocity_time, field_move_time, overall_fields_time};
  TimingResults accessors_results{acc_force_time, acc_velocity_time, acc_move_time, overall_accessors_time};
  return std::make_pair(fields_results, accessors_results);
}

void run_test() {
  // The aggregate will be a sphere with elem radius, node center, node velocity, and node force.
  // We will apply a random force to each sphere, use it to compute their velocity according to drag,
  // and then we will move the spheres according to their velocity.

  // Set cout to use 6 digits
  std::cout << std::fixed << std::setprecision(6);

  // Setup
  size_t num_spheres = 100000;
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  using DoubleField = stk::mesh::Field<double>;
  DoubleField& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "CENTER");
  DoubleField& node_force_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
  DoubleField& node_velocity_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");
  DoubleField& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "RADIUS");

  stk::mesh::Part& sphere_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, meta_data.universal_part(), 1, nullptr);
  meta_data.commit();

  // Create the spheres and populate the fields
  bulk_data.modification_begin();
  std::vector<stk::mesh::Entity> spheres(num_spheres);
  for (size_t i = 0; i < num_spheres; ++i) {
    stk::mesh::Entity node = bulk_data.declare_node(i + 1);  // 1-based indexing
    stk::mesh::Entity elem = bulk_data.declare_element(i + 1, stk::mesh::PartVector{&sphere_part});
    bulk_data.declare_relation(elem, node, 0);

    // Populate the fields
    vector3_field_data(node_center_field, node).set(1.1 * i, 2.2 * i, 3.3);
    vector3_field_data(node_force_field, node).set(5.0, 6.0, 7.0);
    vector3_field_data(node_velocity_field, node).set(1.0, 2.0, 3.0);
    scalar_field_data(elem_radius_field, elem).set(0.5);
  }
  bulk_data.modification_end();

  // Run the test multiple times and average the timing results.
  // Because of "first touch" memory allocation, we ignore the first run.
  TimingResults fields_avg{0.0, 0.0, 0.0, 0.0};
  TimingResults accessors_avg{0.0, 0.0, 0.0, 0.0};
  size_t num_runs = 10;
  for (size_t r = 0; r < num_runs + 1; r++) {
    // run the test
    auto [fields_results, accessors_results] = run_test_for_fields(
        bulk_data, sphere_part, node_center_field, node_force_field, node_velocity_field, elem_radius_field);

    if (r != 0) {  // Skip the first run
      fields_avg.force_time += fields_results.force_time;
      fields_avg.velocity_time += fields_results.velocity_time;
      fields_avg.move_time += fields_results.move_time;
      fields_avg.overall_time += fields_results.overall_time;

      accessors_avg.force_time += accessors_results.force_time;
      accessors_avg.velocity_time += accessors_results.velocity_time;
      accessors_avg.move_time += accessors_results.move_time;
      accessors_avg.overall_time += accessors_results.overall_time;
    }
  }

  double field_force_time = fields_avg.force_time /= num_runs;
  double field_velocity_time = fields_avg.velocity_time /= num_runs;
  double field_move_time = fields_avg.move_time /= num_runs;
  double overall_fields_time = fields_avg.overall_time /= num_runs;

  double acc_force_time = accessors_avg.force_time /= num_runs;
  double acc_velocity_time = accessors_avg.velocity_time /= num_runs;
  double acc_move_time = accessors_avg.move_time /= num_runs;
  double overall_accessors_time = accessors_avg.overall_time /= num_runs;

  std::cout << "Fields:    Force time: " << field_force_time  //
            << ", Velocity time: " << field_velocity_time     //
            << ", Move time: " << field_move_time             //
            << ", Overall time: " << overall_fields_time << std::endl;

  std::cout << "Accessors: Force time: " << acc_force_time  //
            << ", Velocity time: " << acc_velocity_time     //
            << ", Move time: " << acc_move_time             //
            << ", Overall time: " << overall_accessors_time << std::endl;

  std::cout << "Ratio (accessors/fields): " << std::endl;
  std::cout << "           Force time: " << acc_force_time / field_force_time
            << ", Velocity time: " << acc_velocity_time / field_velocity_time
            << ", Move time: " << acc_move_time / field_move_time
            << ", Overall time: " << overall_accessors_time / overall_fields_time << std::endl;
}

}  // namespace

}  // namespace mesh

}  // namespace mundy

int main(int argc, char** argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  mundy::mesh::run_test();

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}