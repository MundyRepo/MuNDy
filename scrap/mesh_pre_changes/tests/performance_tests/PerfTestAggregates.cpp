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

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// External
#include "nanobench.h"

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

void test_direct(const stk::mesh::BulkData& bulk_data, stk::mesh::Part& sphere_part,
                 stk::mesh::Field<double>& node_center_field, stk::mesh::Field<double>& node_force_field,
                 stk::mesh::Field<double>& node_velocity_field, stk::mesh::Field<double>& elem_radius_field) {
  double viscosity = 0.1;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  constexpr double one_over_6pi = 1.0 / (6.0 * pi);
  const double one_over_6pi_mu = one_over_6pi / viscosity;

  // Compute the velocity of each sphere according to drag v = f / (6 * pi * r * mu)
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
  Kokkos::fence();
  ankerl::nanobench::doNotOptimizeAway(node_velocity_field);  // Prevent optimization of the result
}

void test_aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Part& sphere_part,
                    stk::mesh::Field<double>& node_center_field, stk::mesh::Field<double>& node_force_field,
                    stk::mesh::Field<double>& node_velocity_field, stk::mesh::Field<double>& elem_radius_field) {
  double viscosity = 0.1;
  constexpr double pi = Kokkos::numbers::pi_v<double>;
  constexpr double one_over_6pi = 1.0 / (6.0 * pi);
  const double one_over_6pi_mu = one_over_6pi / viscosity;

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

  // Compute the velocity of each sphere according to drag v = f / (6 * pi * r * mu)
  sphere_data.for_each([one_over_6pi_mu](auto& sphere_view) {
    auto force = sphere_view.template get<FORCE>(0);
    auto velocity = sphere_view.template get<VELOCITY>(0);
    auto radius = sphere_view.template get<RADIUS>();
    double inv_radius = 1.0 / radius[0];
    velocity = (one_over_6pi_mu * inv_radius) * force;
  });

  Kokkos::fence();
  ankerl::nanobench::doNotOptimizeAway(node_velocity_field);  // Prevent optimization of the result
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

  // Run the test using nanobench
  ankerl::nanobench::Bench bench;
  bench.relative(true).title("Agg").unit("op").performanceCounters(true).minEpochIterations(1000);

  bench.run("direct", [&] {
    test_direct(bulk_data, sphere_part, node_center_field, node_force_field, node_velocity_field,
                elem_radius_field);
  });
  bench.run("aggregates", [&] {
    test_aggregate(bulk_data, sphere_part, node_center_field, node_force_field, node_velocity_field,
                  elem_radius_field);
  });

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