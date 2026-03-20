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

void run_drag_benchmarks() {
  RigidBodyPerfState state;

  Vector3FieldComponent<scalar_t> force_component(state.force_field());
  Vector3FieldComponent<scalar_t> velocity_component(state.velocity_field());
  ScalarFieldComponent<scalar_t> drag_scalar_field_component(state.drag_scalar_field());
  SharedScalarComponent<scalar_t> drag_scalar_shared_component(state.shared_drag_scalar());

  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(state.bulk_data());
  auto ngp_force_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.force_field());
  auto ngp_velocity_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.velocity_field());
  auto ngp_drag_scalar_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.drag_scalar_field());

  auto ngp_force_component = get_updated_ngp_component(force_component);
  auto ngp_velocity_component = get_updated_ngp_component(velocity_component);
  auto ngp_drag_scalar_field_component = get_updated_ngp_component(drag_scalar_field_component);
  auto ngp_drag_scalar_shared_component = get_updated_ngp_component(drag_scalar_shared_component);

  auto sync_direct_ngp_drag_velocity_field = [&] {
    ngp_force_field.sync_to_device();
    ngp_velocity_field.sync_to_device();
    ngp_drag_scalar_field.sync_to_device();
  };

  auto sync_direct_ngp_drag_velocity_shared = [&] {
    ngp_force_field.sync_to_device();
    ngp_velocity_field.sync_to_device();
  };

  auto sync_ngp_drag_velocity_field_components = [&] {
    ngp_force_component.sync_to_device();
    ngp_velocity_component.sync_to_device();
    ngp_drag_scalar_field_component.sync_to_device();
  };

  auto sync_ngp_drag_velocity_shared_components = [&] {
    ngp_force_component.sync_to_device();
    ngp_velocity_component.sync_to_device();
    ngp_drag_scalar_shared_component.sync_to_device();
  };

  auto mark_direct_ngp_velocity_modified = [&] {
    ngp_velocity_field.modify_on_device();
    Kokkos::fence();
  };

  auto mark_ngp_velocity_component_modified = [&] {
    ngp_velocity_component.modify_on_device();
    Kokkos::fence();
  };

  auto run_host_direct_drag_velocity_field = [&] {
    run_host_drag_velocity_kernel(state.bulk_data(), state.selector(),
                                  HostScalarFieldAccessor<double_field_t>{&state.drag_scalar_field()},
                                  HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                                  HostVector3FieldAccessor<double_field_t>{&state.velocity_field()});
  };

  auto run_host_component_drag_velocity_field = [&] {
    run_host_drag_velocity_kernel(state.bulk_data(), state.selector(), drag_scalar_field_component, force_component,
                                  velocity_component);
  };

  auto run_host_direct_drag_velocity_shared = [&] {
    run_host_drag_velocity_kernel(state.bulk_data(), state.selector(),
                                  HostRawSharedAccessor<scalar_t>{&state.shared_drag_scalar()},
                                  HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                                  HostVector3FieldAccessor<double_field_t>{&state.velocity_field()});
  };

  auto run_host_component_drag_velocity_shared = [&] {
    run_host_drag_velocity_kernel(state.bulk_data(), state.selector(), drag_scalar_shared_component, force_component,
                                  velocity_component);
  };

  auto run_ngp_direct_drag_velocity_field = [&] {
    sync_direct_ngp_drag_velocity_field();
    run_ngp_drag_velocity_kernel(ngp_mesh, state.selector(),
                                 NgpScalarFieldAccessor<decltype(ngp_drag_scalar_field)>{ngp_drag_scalar_field},
                                 NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
                                 NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field});
    mark_direct_ngp_velocity_modified();
  };

  auto run_ngp_component_drag_velocity_field = [&] {
    sync_ngp_drag_velocity_field_components();
    run_ngp_drag_velocity_kernel(ngp_mesh, state.selector(), ngp_drag_scalar_field_component, ngp_force_component,
                                 ngp_velocity_component);
    mark_ngp_velocity_component_modified();
  };

  auto run_ngp_direct_drag_velocity_shared = [&] {
    sync_direct_ngp_drag_velocity_shared();
    run_ngp_drag_velocity_kernel(
        ngp_mesh, state.selector(),
        NgpRawScalarAccessor<std::decay_t<decltype(state.ngp_drag_scalar_view())>>{state.ngp_drag_scalar_view()},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field});
    mark_direct_ngp_velocity_modified();
  };

  auto run_ngp_component_drag_velocity_shared = [&] {
    sync_ngp_drag_velocity_shared_components();
    run_ngp_drag_velocity_kernel(ngp_mesh, state.selector(), ngp_drag_scalar_shared_component, ngp_force_component,
                                 ngp_velocity_component);
    mark_ngp_velocity_component_modified();
  };

  validate_equal("host particle drag field-backed control", run_host_direct_drag_velocity_field,
                 run_host_component_drag_velocity_field, state, false);
  validate_equal("host particle drag shared-backed control", run_host_direct_drag_velocity_shared,
                 run_host_component_drag_velocity_shared, state, false);
  validate_equal("ngp particle drag field-backed control", run_ngp_direct_drag_velocity_field,
                 run_ngp_component_drag_velocity_field, state, true);
  validate_equal("ngp particle drag shared-backed control", run_ngp_direct_drag_velocity_shared,
                 run_ngp_component_drag_velocity_shared, state, true);

  ankerl::nanobench::Bench host_drag_bench;
  host_drag_bench.relative(true)
      .title("Shared vs Field Components / Host particle drag")
      .unit("entity")
      .batch(kNumEntities)
      .performanceCounters(true)
      .minEpochIterations(200);

  host_drag_bench.run("direct field-backed control", [&] {
    run_host_direct_drag_velocity_field();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });
  host_drag_bench.run("components field-backed control", [&] {
    run_host_component_drag_velocity_field();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });
  host_drag_bench.run("direct shared-backed control", [&] {
    run_host_direct_drag_velocity_shared();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });
  host_drag_bench.run("components shared-backed control", [&] {
    run_host_component_drag_velocity_shared();
    ankerl::nanobench::doNotOptimizeAway(state.velocity_field());
  });

  ankerl::nanobench::Bench ngp_drag_bench;
  ngp_drag_bench.relative(true)
      .title("Shared vs Field Components / NGP particle drag")
      .unit("entity")
      .batch(kNumEntities)
      .performanceCounters(true)
      .minEpochIterations(200);

  ngp_drag_bench.run("direct field-backed control", [&] {
    run_ngp_direct_drag_velocity_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_field);
  });
  ngp_drag_bench.run("components field-backed control", [&] {
    run_ngp_component_drag_velocity_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_component);
  });
  ngp_drag_bench.run("direct shared-backed control", [&] {
    run_ngp_direct_drag_velocity_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_field);
  });
  ngp_drag_bench.run("components shared-backed control", [&] {
    run_ngp_component_drag_velocity_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_velocity_component);
  });
}

}  // namespace mundy::mesh::perf_test_shared_components
