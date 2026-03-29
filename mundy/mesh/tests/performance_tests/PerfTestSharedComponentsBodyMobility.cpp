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

#include "PerfTestSharedComponentsSupport.hpp"
#include "nanobench.h"

namespace mundy::mesh::perf_test_shared_components {

void run_body_mobility_benchmarks() {
  RigidBodyPerfState state;

  Vector3FieldComponent<scalar_t> force_component(state.force_field());
  QuaternionFieldComponent<scalar_t> orientation_component(state.orientation_field());
  Vector3FieldComponent<scalar_t> velocity_component(state.velocity_field());
  Vector3FieldComponent<scalar_t> ambient_field_component(state.ambient_field());
  Matrix3FieldComponent<scalar_t> drag_field_component(state.drag_field());
  SharedVector3Component<scalar_t> ambient_shared_component(state.shared_ambient());
  SharedMatrix3Component<scalar_t> drag_shared_component(state.shared_drag());

  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(state.bulk_data());
  auto ngp_force_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.force_field());
  auto ngp_orientation_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.orientation_field());
  auto ngp_velocity_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.velocity_field());
  auto ngp_ambient_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.ambient_field());
  auto ngp_drag_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.drag_field());

  auto ngp_force_component = get_updated_ngp_component(force_component);
  auto ngp_orientation_component = get_updated_ngp_component(orientation_component);
  auto ngp_velocity_component = get_updated_ngp_component(velocity_component);
  auto ngp_ambient_field_component = get_updated_ngp_component(ambient_field_component);
  auto ngp_drag_field_component = get_updated_ngp_component(drag_field_component);
  auto& ngp_ambient_shared_component = get_updated_ngp_component(ambient_shared_component);
  auto& ngp_drag_shared_component = get_updated_ngp_component(drag_shared_component);

  auto sync_direct_ngp_body_mobility_field = [&] {
    ngp_force_field.sync_to_device();
    ngp_orientation_field.sync_to_device();
    ngp_velocity_field.sync_to_device();
    ngp_ambient_field.sync_to_device();
    ngp_drag_field.sync_to_device();
  };

  auto sync_direct_ngp_body_mobility_shared = [&] {
    ngp_force_field.sync_to_device();
    ngp_orientation_field.sync_to_device();
    ngp_velocity_field.sync_to_device();
  };

  auto sync_ngp_body_mobility_field_components = [&] {
    ngp_force_component.sync_to_device();
    ngp_orientation_component.sync_to_device();
    ngp_velocity_component.sync_to_device();
    ngp_ambient_field_component.sync_to_device();
    ngp_drag_field_component.sync_to_device();
  };

  auto sync_ngp_body_mobility_shared_components = [&] {
    ngp_force_component.sync_to_device();
    ngp_orientation_component.sync_to_device();
    ngp_velocity_component.sync_to_device();
    ngp_ambient_shared_component.sync_to_device();
    ngp_drag_shared_component.sync_to_device();
  };

  auto mark_direct_ngp_velocity_modified = [&] {
    ngp_velocity_field.modify_on_device();
    Kokkos::fence();
  };

  auto mark_ngp_velocity_component_modified = [&] {
    ngp_velocity_component.modify_on_device();
    Kokkos::fence();
  };

  auto run_host_direct_body_mobility_field = [&] {
    run_host_body_mobility_kernel(state.bulk_data(), state.selector(),
                                  HostVector3FieldAccessor<double_field_t>{&state.ambient_field()},
                                  HostMatrix3FieldAccessor<double_field_t>{&state.drag_field()},
                                  HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                                  HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                                  HostVector3FieldAccessor<double_field_t>{&state.velocity_field()});
  };

  auto run_host_component_body_mobility_field = [&] {
    run_host_body_mobility_kernel(state.bulk_data(), state.selector(), ambient_field_component, drag_field_component,
                                  force_component, orientation_component, velocity_component);
  };

  auto run_host_direct_body_mobility_shared = [&] {
    run_host_body_mobility_kernel(state.bulk_data(), state.selector(),
                                  HostRawSharedAccessor<vector3_t>{&state.shared_ambient()},
                                  HostRawSharedAccessor<matrix3_t>{&state.shared_drag()},
                                  HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                                  HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                                  HostVector3FieldAccessor<double_field_t>{&state.velocity_field()});
  };

  auto run_host_component_body_mobility_shared = [&] {
    run_host_body_mobility_kernel(state.bulk_data(), state.selector(), ambient_shared_component, drag_shared_component,
                                  force_component, orientation_component, velocity_component);
  };

  auto run_ngp_direct_body_mobility_field = [&] {
    sync_direct_ngp_body_mobility_field();
    run_ngp_body_mobility_kernel(ngp_mesh, state.selector(),
                                 NgpVector3FieldAccessor<decltype(ngp_ambient_field)>{ngp_ambient_field},
                                 NgpMatrix3FieldAccessor<decltype(ngp_drag_field)>{ngp_drag_field},
                                 NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
                                 NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
                                 NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field});
    mark_direct_ngp_velocity_modified();
  };

  auto run_ngp_component_body_mobility_field = [&] {
    sync_ngp_body_mobility_field_components();
    run_ngp_body_mobility_kernel(ngp_mesh, state.selector(), ngp_ambient_field_component, ngp_drag_field_component,
                                 ngp_force_component, ngp_orientation_component, ngp_velocity_component);
    mark_ngp_velocity_component_modified();
  };

  auto run_ngp_direct_body_mobility_shared = [&] {
    sync_direct_ngp_body_mobility_shared();
    run_ngp_body_mobility_kernel(
        ngp_mesh, state.selector(),
        NgpRawVector3Accessor<std::decay_t<decltype(state.ngp_ambient_view())>>{state.ngp_ambient_view()},
        NgpRawMatrix3Accessor<std::decay_t<decltype(state.ngp_drag_view())>>{state.ngp_drag_view()},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field});
    mark_direct_ngp_velocity_modified();
  };

  auto run_ngp_component_body_mobility_shared = [&] {
    sync_ngp_body_mobility_shared_components();
    run_ngp_body_mobility_kernel(ngp_mesh, state.selector(), ngp_ambient_shared_component, ngp_drag_shared_component,
                                 ngp_force_component, ngp_orientation_component, ngp_velocity_component);
    mark_ngp_velocity_component_modified();
  };

  validate_equal("host body mobility field-backed controls", run_host_direct_body_mobility_field,
                 run_host_component_body_mobility_field, state, false);
  validate_equal("host body mobility shared-backed controls", run_host_direct_body_mobility_shared,
                 run_host_component_body_mobility_shared, state, false);
  validate_equal("ngp body mobility field-backed controls", run_ngp_direct_body_mobility_field,
                 run_ngp_component_body_mobility_field, state, true);
  validate_equal("ngp body mobility shared-backed controls", run_ngp_direct_body_mobility_shared,
                 run_ngp_component_body_mobility_shared, state, true);

  ankerl::nanobench::Bench host_body_mobility_bench;
  host_body_mobility_bench.relative(true)
      .title("Shared vs Field Components / Host rigid-body mobility")
      .unit("entity")
      .batch(kNumEntities)
      .performanceCounters(true)
      .minEpochIterations(200);

  host_body_mobility_bench.run("direct field-backed controls", [&] {
    run_host_direct_body_mobility_field();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });
  host_body_mobility_bench.run("components field-backed controls", [&] {
    run_host_component_body_mobility_field();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });
  host_body_mobility_bench.run("direct shared-backed controls", [&] {
    run_host_direct_body_mobility_shared();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });
  host_body_mobility_bench.run("components shared-backed controls", [&] {
    run_host_component_body_mobility_shared();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });

  ankerl::nanobench::Bench ngp_body_mobility_bench;
  ngp_body_mobility_bench.relative(true)
      .title("Shared vs Field Components / NGP rigid-body mobility")
      .unit("entity")
      .batch(kNumEntities)
      .performanceCounters(true)
      .minEpochIterations(200);

  ngp_body_mobility_bench.run("direct field-backed controls", [&] {
    run_ngp_direct_body_mobility_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_field);
  });
  ngp_body_mobility_bench.run("components field-backed controls", [&] {
    run_ngp_component_body_mobility_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_component);
  });
  ngp_body_mobility_bench.run("direct shared-backed controls", [&] {
    run_ngp_direct_body_mobility_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_field);
  });
  ngp_body_mobility_bench.run("components shared-backed controls", [&] {
    run_ngp_component_body_mobility_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_component);
  });
}

}  // namespace mundy::mesh::perf_test_shared_components
