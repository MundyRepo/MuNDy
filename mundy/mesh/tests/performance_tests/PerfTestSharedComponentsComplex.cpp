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

void run_complex_benchmarks() {
  RigidBodyPerfState state;

  Vector3FieldComponent<scalar_t> force_component(state.force_field());
  Vector3FieldComponent<scalar_t> torque_component(state.torque_field());
  QuaternionFieldComponent<scalar_t> orientation_component(state.orientation_field());
  Matrix3FieldComponent<scalar_t> mobility_component(state.mobility_field());
  Vector3FieldComponent<scalar_t> velocity_component(state.velocity_field());
  Matrix3FieldComponent<scalar_t> stress_component(state.stress_field());
  QuaternionFieldComponent<scalar_t> orientation_out_component(state.orientation_out_field());
  ScalarFieldComponent<scalar_t> energy_component(state.energy_field());
  ScalarFieldComponent<scalar_t> dt_field_component(state.dt_field());
  Vector3FieldComponent<scalar_t> ambient_field_component(state.ambient_field());
  Matrix3FieldComponent<scalar_t> drag_field_component(state.drag_field());
  QuaternionFieldComponent<scalar_t> target_orientation_field_component(state.target_orientation_field());

  SharedScalarComponent<scalar_t> dt_shared_component(state.shared_dt());
  SharedVector3Component<scalar_t> ambient_shared_component(state.shared_ambient());
  SharedMatrix3Component<scalar_t> drag_shared_component(state.shared_drag());
  SharedQuaternionComponent<scalar_t> target_orientation_shared_component(state.shared_target_orientation());

  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(state.bulk_data());
  auto ngp_force_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.force_field());
  auto ngp_torque_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.torque_field());
  auto ngp_orientation_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.orientation_field());
  auto ngp_mobility_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.mobility_field());
  auto ngp_velocity_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.velocity_field());
  auto ngp_stress_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.stress_field());
  auto ngp_orientation_out_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.orientation_out_field());
  auto ngp_energy_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.energy_field());
  auto ngp_dt_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.dt_field());
  auto ngp_ambient_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.ambient_field());
  auto ngp_drag_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.drag_field());
  auto ngp_target_orientation_field = stk::mesh::get_updated_ngp_field<scalar_t>(state.target_orientation_field());

  auto ngp_force_component = get_updated_ngp_component(force_component);
  auto ngp_torque_component = get_updated_ngp_component(torque_component);
  auto ngp_orientation_component = get_updated_ngp_component(orientation_component);
  auto ngp_mobility_component = get_updated_ngp_component(mobility_component);
  auto ngp_velocity_component = get_updated_ngp_component(velocity_component);
  auto ngp_stress_component = get_updated_ngp_component(stress_component);
  auto ngp_orientation_out_component = get_updated_ngp_component(orientation_out_component);
  auto ngp_energy_component = get_updated_ngp_component(energy_component);
  auto ngp_dt_field_component = get_updated_ngp_component(dt_field_component);
  auto ngp_ambient_field_component = get_updated_ngp_component(ambient_field_component);
  auto ngp_drag_field_component = get_updated_ngp_component(drag_field_component);
  auto ngp_target_orientation_field_component = get_updated_ngp_component(target_orientation_field_component);

  auto ngp_dt_shared_component = get_updated_ngp_component(dt_shared_component);
  auto& ngp_ambient_shared_component = get_updated_ngp_component(ambient_shared_component);
  auto& ngp_drag_shared_component = get_updated_ngp_component(drag_shared_component);
  auto& ngp_target_orientation_shared_component = get_updated_ngp_component(target_orientation_shared_component);

  auto sync_direct_ngp_complex_fields = [&] {
    ngp_force_field.sync_to_device();
    ngp_torque_field.sync_to_device();
    ngp_orientation_field.sync_to_device();
    ngp_mobility_field.sync_to_device();
    ngp_velocity_field.sync_to_device();
    ngp_stress_field.sync_to_device();
    ngp_orientation_out_field.sync_to_device();
    ngp_energy_field.sync_to_device();
    ngp_dt_field.sync_to_device();
    ngp_ambient_field.sync_to_device();
    ngp_drag_field.sync_to_device();
    ngp_target_orientation_field.sync_to_device();
  };

  auto sync_ngp_complex_components = [&] {
    ngp_force_component.sync_to_device();
    ngp_torque_component.sync_to_device();
    ngp_orientation_component.sync_to_device();
    ngp_mobility_component.sync_to_device();
    ngp_velocity_component.sync_to_device();
    ngp_stress_component.sync_to_device();
    ngp_orientation_out_component.sync_to_device();
    ngp_energy_component.sync_to_device();
    ngp_dt_field_component.sync_to_device();
    ngp_ambient_field_component.sync_to_device();
    ngp_drag_field_component.sync_to_device();
    ngp_target_orientation_field_component.sync_to_device();
    ngp_dt_shared_component.sync_to_device();
    ngp_ambient_shared_component.sync_to_device();
    ngp_drag_shared_component.sync_to_device();
    ngp_target_orientation_shared_component.sync_to_device();
  };

  auto mark_direct_ngp_outputs_modified = [&] {
    ngp_velocity_field.modify_on_device();
    ngp_stress_field.modify_on_device();
    ngp_orientation_out_field.modify_on_device();
    ngp_energy_field.modify_on_device();
    Kokkos::fence();
  };

  auto mark_ngp_component_outputs_modified = [&] {
    ngp_velocity_component.modify_on_device();
    ngp_stress_component.modify_on_device();
    ngp_orientation_out_component.modify_on_device();
    ngp_energy_component.modify_on_device();
    Kokkos::fence();
  };

  auto run_host_direct_all_field = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(),
                               HostScalarFieldAccessor<double_field_t>{&state.dt_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.ambient_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.drag_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.target_orientation_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.torque_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.mobility_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.velocity_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.stress_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_out_field()},
                               HostScalarFieldAccessor<double_field_t>{&state.energy_field()});
  };

  auto run_host_component_all_field = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), dt_field_component, ambient_field_component,
                               drag_field_component, target_orientation_field_component, force_component,
                               torque_component, orientation_component, mobility_component, velocity_component,
                               stress_component, orientation_out_component, energy_component);
  };

  auto run_host_direct_mixed = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), HostRawSharedAccessor<scalar_t>{&state.shared_dt()},
                               HostVector3FieldAccessor<double_field_t>{&state.ambient_field()},
                               HostRawSharedAccessor<matrix3_t>{&state.shared_drag()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.target_orientation_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.torque_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.mobility_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.velocity_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.stress_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_out_field()},
                               HostScalarFieldAccessor<double_field_t>{&state.energy_field()});
  };

  auto run_host_direct_mixed_ref = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), HostRefSharedAccessor<scalar_t>{state.shared_dt()},
                               HostVector3FieldAccessor<double_field_t>{&state.ambient_field()},
                               HostRefSharedAccessor<matrix3_t>{state.shared_drag()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.target_orientation_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.torque_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.mobility_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.velocity_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.stress_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_out_field()},
                               HostScalarFieldAccessor<double_field_t>{&state.energy_field()});
  };

  auto run_host_component_mixed = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), dt_shared_component, ambient_field_component,
                               drag_shared_component, target_orientation_field_component, force_component,
                               torque_component, orientation_component, mobility_component, velocity_component,
                               stress_component, orientation_out_component, energy_component);
  };

  auto run_host_direct_all_shared = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), HostRawSharedAccessor<scalar_t>{&state.shared_dt()},
                               HostRawSharedAccessor<vector3_t>{&state.shared_ambient()},
                               HostRawSharedAccessor<matrix3_t>{&state.shared_drag()},
                               HostRawSharedAccessor<quaternion_t>{&state.shared_target_orientation()},
                               HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.torque_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.mobility_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.velocity_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.stress_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_out_field()},
                               HostScalarFieldAccessor<double_field_t>{&state.energy_field()});
  };

  auto run_host_direct_all_shared_ref = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), HostRefSharedAccessor<scalar_t>{state.shared_dt()},
                               HostRefSharedAccessor<vector3_t>{state.shared_ambient()},
                               HostRefSharedAccessor<matrix3_t>{state.shared_drag()},
                               HostRefSharedAccessor<quaternion_t>{state.shared_target_orientation()},
                               HostVector3FieldAccessor<double_field_t>{&state.force_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.torque_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.mobility_field()},
                               HostVector3FieldAccessor<double_field_t>{&state.velocity_field()},
                               HostMatrix3FieldAccessor<double_field_t>{&state.stress_field()},
                               HostQuaternionFieldAccessor<double_field_t>{&state.orientation_out_field()},
                               HostScalarFieldAccessor<double_field_t>{&state.energy_field()});
  };

  auto run_host_component_all_shared = [&] {
    run_host_rigid_body_kernel(state.bulk_data(), state.selector(), dt_shared_component, ambient_shared_component,
                               drag_shared_component, target_orientation_shared_component, force_component,
                               torque_component, orientation_component, mobility_component, velocity_component,
                               stress_component, orientation_out_component, energy_component);
  };

  auto run_ngp_direct_all_field = [&] {
    sync_direct_ngp_complex_fields();
    run_ngp_rigid_body_kernel(
        ngp_mesh, state.selector(), NgpScalarFieldAccessor<decltype(ngp_dt_field)>{ngp_dt_field},
        NgpVector3FieldAccessor<decltype(ngp_ambient_field)>{ngp_ambient_field},
        NgpMatrix3FieldAccessor<decltype(ngp_drag_field)>{ngp_drag_field},
        NgpQuaternionFieldAccessor<decltype(ngp_target_orientation_field)>{ngp_target_orientation_field},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpVector3FieldAccessor<decltype(ngp_torque_field)>{ngp_torque_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
        NgpMatrix3FieldAccessor<decltype(ngp_mobility_field)>{ngp_mobility_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field},
        NgpMatrix3FieldAccessor<decltype(ngp_stress_field)>{ngp_stress_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_out_field)>{ngp_orientation_out_field},
        NgpScalarFieldAccessor<decltype(ngp_energy_field)>{ngp_energy_field});
    mark_direct_ngp_outputs_modified();
  };

  auto run_ngp_component_all_field = [&] {
    sync_ngp_complex_components();
    run_ngp_rigid_body_kernel(ngp_mesh, state.selector(), ngp_dt_field_component, ngp_ambient_field_component,
                              ngp_drag_field_component, ngp_target_orientation_field_component, ngp_force_component,
                              ngp_torque_component, ngp_orientation_component, ngp_mobility_component,
                              ngp_velocity_component, ngp_stress_component, ngp_orientation_out_component,
                              ngp_energy_component);
    mark_ngp_component_outputs_modified();
  };

  auto run_ngp_direct_mixed = [&] {
    sync_direct_ngp_complex_fields();
    run_ngp_rigid_body_kernel(
        ngp_mesh, state.selector(),
        NgpRawScalarAccessor<std::decay_t<decltype(state.ngp_dt_view())>>{state.ngp_dt_view()},
        NgpVector3FieldAccessor<decltype(ngp_ambient_field)>{ngp_ambient_field},
        NgpRawMatrix3Accessor<std::decay_t<decltype(state.ngp_drag_view())>>{state.ngp_drag_view()},
        NgpQuaternionFieldAccessor<decltype(ngp_target_orientation_field)>{ngp_target_orientation_field},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpVector3FieldAccessor<decltype(ngp_torque_field)>{ngp_torque_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
        NgpMatrix3FieldAccessor<decltype(ngp_mobility_field)>{ngp_mobility_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field},
        NgpMatrix3FieldAccessor<decltype(ngp_stress_field)>{ngp_stress_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_out_field)>{ngp_orientation_out_field},
        NgpScalarFieldAccessor<decltype(ngp_energy_field)>{ngp_energy_field});
    mark_direct_ngp_outputs_modified();
  };

  auto run_ngp_direct_mixed_owning_shared = [&] {
    sync_direct_ngp_complex_fields();
    run_ngp_rigid_body_kernel(ngp_mesh, state.selector(), NgpOwningSharedAccessor<scalar_t>{state.shared_dt()},
                              NgpVector3FieldAccessor<decltype(ngp_ambient_field)>{ngp_ambient_field},
                              NgpOwningSharedAccessor<matrix3_t>{state.shared_drag()},
                              NgpQuaternionFieldAccessor<decltype(ngp_target_orientation_field)>{
                                  ngp_target_orientation_field},
                              NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
                              NgpVector3FieldAccessor<decltype(ngp_torque_field)>{ngp_torque_field},
                              NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
                              NgpMatrix3FieldAccessor<decltype(ngp_mobility_field)>{ngp_mobility_field},
                              NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field},
                              NgpMatrix3FieldAccessor<decltype(ngp_stress_field)>{ngp_stress_field},
                              NgpQuaternionFieldAccessor<decltype(ngp_orientation_out_field)>{ngp_orientation_out_field},
                              NgpScalarFieldAccessor<decltype(ngp_energy_field)>{ngp_energy_field});
    mark_direct_ngp_outputs_modified();
  };

  auto run_ngp_component_mixed = [&] {
    sync_ngp_complex_components();
    run_ngp_rigid_body_kernel(ngp_mesh, state.selector(), ngp_dt_shared_component, ngp_ambient_field_component,
                              ngp_drag_shared_component, ngp_target_orientation_field_component, ngp_force_component,
                              ngp_torque_component, ngp_orientation_component, ngp_mobility_component,
                              ngp_velocity_component, ngp_stress_component, ngp_orientation_out_component,
                              ngp_energy_component);
    mark_ngp_component_outputs_modified();
  };

  auto run_ngp_direct_all_shared = [&] {
    sync_direct_ngp_complex_fields();
    run_ngp_rigid_body_kernel(
        ngp_mesh, state.selector(),
        NgpRawScalarAccessor<std::decay_t<decltype(state.ngp_dt_view())>>{state.ngp_dt_view()},
        NgpRawVector3Accessor<std::decay_t<decltype(state.ngp_ambient_view())>>{state.ngp_ambient_view()},
        NgpRawMatrix3Accessor<std::decay_t<decltype(state.ngp_drag_view())>>{state.ngp_drag_view()},
        NgpRawQuaternionAccessor<std::decay_t<decltype(state.ngp_target_orientation_view())>>{
            state.ngp_target_orientation_view()},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpVector3FieldAccessor<decltype(ngp_torque_field)>{ngp_torque_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
        NgpMatrix3FieldAccessor<decltype(ngp_mobility_field)>{ngp_mobility_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field},
        NgpMatrix3FieldAccessor<decltype(ngp_stress_field)>{ngp_stress_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_out_field)>{ngp_orientation_out_field},
        NgpScalarFieldAccessor<decltype(ngp_energy_field)>{ngp_energy_field});
    mark_direct_ngp_outputs_modified();
  };

  auto run_ngp_direct_all_shared_owning_shared = [&] {
    sync_direct_ngp_complex_fields();
    run_ngp_rigid_body_kernel(
        ngp_mesh, state.selector(), NgpOwningSharedAccessor<scalar_t>{state.shared_dt()},
        NgpOwningSharedAccessor<vector3_t>{state.shared_ambient()},
        NgpOwningSharedAccessor<matrix3_t>{state.shared_drag()},
        NgpOwningSharedAccessor<quaternion_t>{state.shared_target_orientation()},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpVector3FieldAccessor<decltype(ngp_torque_field)>{ngp_torque_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
        NgpMatrix3FieldAccessor<decltype(ngp_mobility_field)>{ngp_mobility_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field},
        NgpMatrix3FieldAccessor<decltype(ngp_stress_field)>{ngp_stress_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_out_field)>{ngp_orientation_out_field},
        NgpScalarFieldAccessor<decltype(ngp_energy_field)>{ngp_energy_field});
    mark_direct_ngp_outputs_modified();
  };

  auto run_ngp_component_all_shared = [&] {
    sync_ngp_complex_components();
    run_ngp_rigid_body_kernel(ngp_mesh, state.selector(), ngp_dt_shared_component, ngp_ambient_shared_component,
                              ngp_drag_shared_component, ngp_target_orientation_shared_component, ngp_force_component,
                              ngp_torque_component, ngp_orientation_component, ngp_mobility_component,
                              ngp_velocity_component, ngp_stress_component, ngp_orientation_out_component,
                              ngp_energy_component);
    mark_ngp_component_outputs_modified();
  };

  validate_equal("host complex field-backed controls", run_host_direct_all_field, run_host_component_all_field, state,
                 false);
  validate_equal("host complex mixed controls", run_host_direct_mixed, run_host_component_mixed, state, false);
  validate_equal("host complex mixed controls (ref-backed custom direct)", run_host_direct_mixed_ref,
                 run_host_component_mixed, state, false);
  validate_equal("host complex shared-backed controls", run_host_direct_all_shared, run_host_component_all_shared,
                 state, false);
  validate_equal("host complex shared-backed controls (ref-backed custom direct)", run_host_direct_all_shared_ref,
                 run_host_component_all_shared, state, false);
  validate_equal("ngp complex field-backed controls", run_ngp_direct_all_field, run_ngp_component_all_field, state,
                 true);
  validate_equal("ngp complex mixed controls", run_ngp_direct_mixed, run_ngp_component_mixed, state, true);
  validate_equal("ngp complex mixed controls (owning-shared custom direct)", run_ngp_direct_mixed_owning_shared,
                 run_ngp_component_mixed, state, true);
  validate_equal("ngp complex shared-backed controls", run_ngp_direct_all_shared, run_ngp_component_all_shared, state,
                 true);
  validate_equal("ngp complex shared-backed controls (owning-shared custom direct)",
                 run_ngp_direct_all_shared_owning_shared, run_ngp_component_all_shared, state, true);

  ankerl::nanobench::Bench host_complex_bench;
  host_complex_bench.relative(true)
      .title("Shared vs Field Components / Host complex mixed-output step")
      .unit("entity")
      .batch(kNumEntities)
      .performanceCounters(true)
      .minEpochIterations(200);

  host_complex_bench.run("direct field-backed controls", [&] {
    run_host_direct_all_field();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("components field-backed controls", [&] {
    run_host_component_all_field();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("direct mixed controls", [&] {
    run_host_direct_mixed();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("direct mixed controls (ref-backed custom)", [&] {
    run_host_direct_mixed_ref();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("components mixed controls", [&] {
    run_host_component_mixed();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("direct shared-backed controls", [&] {
    run_host_direct_all_shared();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("direct shared-backed controls (ref-backed custom)", [&] {
    run_host_direct_all_shared_ref();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });
  host_complex_bench.run("components shared-backed controls", [&] {
    run_host_component_all_shared();
    ankerl::nanobench::doNotOptimizeAway(state.energy_field());
  });

  ankerl::nanobench::Bench ngp_complex_bench;
  ngp_complex_bench.relative(true)
      .title("Shared vs Field Components / NGP complex mixed-output step")
      .unit("entity")
      .batch(kNumEntities)
      .performanceCounters(true)
      .minEpochIterations(200);

  ngp_complex_bench.run("direct field-backed controls", [&] {
    run_ngp_direct_all_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
  });
  ngp_complex_bench.run("components field-backed controls", [&] {
    run_ngp_component_all_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_component);
  });
  ngp_complex_bench.run("direct mixed controls", [&] {
    run_ngp_direct_mixed();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
  });
  ngp_complex_bench.run("direct mixed controls (owning-shared custom)", [&] {
    run_ngp_direct_mixed_owning_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
  });
  ngp_complex_bench.run("components mixed controls", [&] {
    run_ngp_component_mixed();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_component);
  });
  ngp_complex_bench.run("direct shared-backed controls", [&] {
    run_ngp_direct_all_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
  });
  ngp_complex_bench.run("direct shared-backed controls (owning-shared custom)", [&] {
    run_ngp_direct_all_shared_owning_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
  });
  ngp_complex_bench.run("components shared-backed controls", [&] {
    run_ngp_component_all_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_component);
  });
}

}  // namespace mundy::mesh::perf_test_shared_components
