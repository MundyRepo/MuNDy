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

/// \file PerfTestSharedComponentsNew.cpp
/// \brief Nanobench performance benchmarks for Mundy shared components.
///
/// Usage: PerfTestSharedComponentsNew [--simple] [--case drag|mobility|complex|all]
///                                    [--space host|ngp|all] [--n COUNT]
///   (default) Full nanobench output for all cases on host and NGP.
///   --simple         Compact summary table per benchmark.
///   --case VALUE     Workload: drag, mobility, complex, or all (default: all).
///   --space VALUE    Execution path: host, ngp, or all (default: all).
///   --n COUNT        Number of node entities (default: 100000).
///
/// Each benchmark compares two access strategies for control inputs (parameters uniform
/// across all entities: scalars, vectors, matrices, quaternions):
///   field   -- control inputs stored as per-entity STK fields, accessed via FieldComponent
///   shared  -- control inputs stored as a single shared value, accessed via SharedComponent

#define ANKERL_NANOBENCH_IMPLEMENT

// C++ core
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// External
#include <nanobench.h>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/parallel/Parallel.hpp>

// Mundy math
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Vector3.hpp>

// Mundy mesh
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/FieldViews.hpp>
#include <mundy_mesh/ForEachEntity.hpp>
#include <mundy_mesh/SharedComponent.hpp>

// =============================================================================
// Constants
// =============================================================================

static constexpr size_t kDefaultNumEntities = 100000;

// =============================================================================
// Type aliases
// =============================================================================

using value_type     = double;
using vector3_t    = mundy::Vector3<value_type>;
using matrix3_t    = mundy::Matrix3<value_type>;
using quaternion_t = mundy::Quaternion<value_type>;
using field_t      = stk::mesh::Field<value_type>;

using mundy::normalize;
using mundy::mesh::get_updated_ngp_component;
using mundy::mesh::scalar_field_data;
using mundy::mesh::vector3_field_data;
using mundy::mesh::matrix3_field_data;
using mundy::mesh::quaternion_field_data;

using FC_s = mundy::mesh::ScalarFieldComponent<value_type>;
using FC_v = mundy::mesh::Vector3FieldComponent<value_type>;
using FC_m = mundy::mesh::Matrix3FieldComponent<value_type>;
using FC_q = mundy::mesh::QuaternionFieldComponent<value_type>;
using SC_s = mundy::mesh::SharedScalarComponent<value_type>;
using SC_v = mundy::mesh::SharedVector3Component<value_type>;
using SC_m = mundy::mesh::SharedMatrix3Component<value_type>;
using SC_q = mundy::mesh::SharedQuaternionComponent<value_type>;

// =============================================================================
// Fixture
// =============================================================================

KOKKOS_INLINE_FUNCTION
static matrix3_t make_spd_matrix3(value_type b) {
  return matrix3_t(1.25 + b,           0.08 + 0.05*b,  -0.04 + 0.01*b,
                   0.08 + 0.05*b,      1.75 + 0.50*b,   0.06 - 0.02*b,
                  -0.04 + 0.01*b,      0.06 - 0.02*b,   2.25 + 0.75*b);
}

struct RigidBodyFixture {
  std::shared_ptr<stk::mesh::MetaData> meta;
  std::shared_ptr<stk::mesh::BulkData> bulk;
  stk::mesh::Selector                  selector;
  std::vector<stk::mesh::Entity>       entities;

  // Per-entity varying fields
  field_t* force;
  field_t* torque;
  field_t* orientation;
  field_t* mobility;
  field_t* velocity;
  field_t* stress;
  field_t* orientation_out;
  field_t* energy;

  // Control fields (uniform value replicated into every entity, for the "field" variant)
  field_t* dt;
  field_t* drag_scalar;
  field_t* ambient;
  field_t* drag;
  field_t* target_orientation;

  // Shared control values (for the "shared" variant)
  value_type     shared_dt                 = 0.015;
  value_type     shared_drag_scalar        = 0.65;
  vector3_t    shared_ambient            = {0.25, -0.15, 0.35};
  matrix3_t    shared_drag               = make_spd_matrix3(0.35);
  quaternion_t shared_target_orientation = normalize(quaternion_t(1.0, 0.04, -0.08, 0.03));
};

static RigidBodyFixture build_fixture(size_t n) {
  RigidBodyFixture f;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  f.meta = builder.create_meta_data();
  f.meta->use_simple_fields();

  auto& meta = *f.meta;
  f.force            = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "FORCE");
  f.torque           = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "TORQUE");
  f.orientation      = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "ORIENTATION");
  f.mobility         = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "MOBILITY");
  f.velocity         = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "VELOCITY");
  f.stress           = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "STRESS");
  f.orientation_out  = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "ORIENTATION_OUT");
  f.energy           = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "ENERGY");
  f.dt               = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "DT");
  f.drag_scalar      = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "DRAG_SCALAR");
  f.ambient          = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "AMBIENT");
  f.drag             = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "DRAG");
  f.target_orientation = &meta.declare_field<value_type>(stk::topology::NODE_RANK, "TARGET_ORIENTATION");

  const auto& universal = meta.universal_part();
  stk::mesh::put_field_on_mesh(*f.force,            universal, 3, nullptr);
  stk::mesh::put_field_on_mesh(*f.torque,           universal, 3, nullptr);
  stk::mesh::put_field_on_mesh(*f.orientation,      universal, 4, nullptr);
  stk::mesh::put_field_on_mesh(*f.mobility,         universal, 9, nullptr);
  stk::mesh::put_field_on_mesh(*f.velocity,         universal, 3, nullptr);
  stk::mesh::put_field_on_mesh(*f.stress,           universal, 9, nullptr);
  stk::mesh::put_field_on_mesh(*f.orientation_out,  universal, 4, nullptr);
  stk::mesh::put_field_on_mesh(*f.energy,           universal, 1, nullptr);
  stk::mesh::put_field_on_mesh(*f.dt,               universal, 1, nullptr);
  stk::mesh::put_field_on_mesh(*f.drag_scalar,      universal, 1, nullptr);
  stk::mesh::put_field_on_mesh(*f.ambient,          universal, 3, nullptr);
  stk::mesh::put_field_on_mesh(*f.drag,             universal, 9, nullptr);
  stk::mesh::put_field_on_mesh(*f.target_orientation, universal, 4, nullptr);
  meta.commit();

  f.selector = universal;
  f.bulk = builder.create(f.meta);
  auto& bulk = *f.bulk;

  bulk.modification_begin();
  f.entities.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const stk::mesh::Entity node = bulk.declare_node(i + 1);
    f.entities.push_back(node);

    const value_type p = 1.0e-5 * static_cast<value_type>(i);
    vector3_field_data(*f.force,  node) = vector3_t(1.25 + 0.25*p, -0.85 + 0.15*p,  0.65 - 0.10*p);
    vector3_field_data(*f.torque, node) = vector3_t(0.35 + 0.12*p, -0.28 + 0.05*p,  0.18 - 0.02*p);
    quaternion_field_data(*f.orientation, node) = normalize(quaternion_t(1.0, 0.18*p, -0.11*p, 0.09*p));
    matrix3_field_data(*f.mobility, node) = make_spd_matrix3(0.20 + 0.05*p);

    vector3_field_data(*f.velocity, node)        = vector3_t(0.0, 0.0, 0.0);
    matrix3_field_data(*f.stress, node)          = matrix3_t(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    quaternion_field_data(*f.orientation_out, node) = quaternion_t(1.0, 0.0, 0.0, 0.0);
    scalar_field_data(*f.energy, node)[0]        = 0.0;

    scalar_field_data(*f.dt, node)[0]            = f.shared_dt;
    scalar_field_data(*f.drag_scalar, node)[0]   = f.shared_drag_scalar;
    vector3_field_data(*f.ambient, node)         = f.shared_ambient;
    matrix3_field_data(*f.drag, node)            = f.shared_drag;
    quaternion_field_data(*f.target_orientation, node) = f.shared_target_orientation;
  }
  bulk.modification_end();

  return f;
}

// =============================================================================
// Physics step functions
// =============================================================================

template <typename DragT, typename ForceT, typename VelT>
KOKKOS_INLINE_FUNCTION void drag_velocity_step(const DragT& drag, const ForceT& force, VelT vel) {
  vel = force / static_cast<value_type>(drag);
}

template <typename AmbientT, typename MobilityT, typename ForceT, typename OrientT, typename VelT>
KOKKOS_INLINE_FUNCTION void rigid_body_mobility_step(const AmbientT& ambient, const MobilityT& mobility,
                                                      const ForceT& force, const OrientT& orient, VelT vel) {
  vel = ambient + orient * (mobility * (conjugate(orient) * force));
}

template <typename DtT, typename AmbientT, typename DragT, typename TargetT,
          typename ForceT, typename TorqueT, typename OrientT, typename MobilityT,
          typename VelT, typename StressT, typename OrientOutT, typename EnergyT>
KOKKOS_INLINE_FUNCTION void rigid_body_step(const DtT& dt, const AmbientT& ambient, const DragT& drag,
                                             const TargetT& target, const ForceT& force, const TorqueT& torque,
                                             const OrientT& orient, const MobilityT& mobility,
                                             VelT vel, StressT stress, OrientOutT orient_out, EnergyT energy) {
  const auto total_force  = force + ambient;
  const auto body_force   = conjugate(orient) * total_force;
  const auto body_spin    = torque + cross(body_force, ambient);
  const quaternion_t spin_q(0.0, body_spin[0], body_spin[1], body_spin[2]);
  const auto trial_orient = normalize(target * orient + (0.5 * static_cast<value_type>(dt)) * (spin_q * orient));
  const auto eff_mobility = mobility + drag;
  const auto lab_vel      = orient * (eff_mobility * body_force);
  const value_type inv_scale = 1.0 / (1.0 + mundy::norm(body_force));
  vel        = lab_vel + (static_cast<value_type>(dt) * inv_scale) * cross(body_spin, total_force);
  stress     = trial_orient * eff_mobility + 0.25 * (mobility - drag);
  orient_out = trial_orient;
  energy     = mundy::dot(lab_vel, lab_vel) + 0.1 * mundy::dot(body_spin, body_spin)
               + mundy::dot(trial_orient, target);
}

// =============================================================================
// Validation helpers
// =============================================================================

static void sync_outputs_to_host(const RigidBodyFixture& f) {
  f.velocity->sync_to_host();
  f.stress->sync_to_host();
  f.orientation_out->sync_to_host();
  f.energy->sync_to_host();
}

static value_type checksum(const RigidBodyFixture& f) {
  const size_t stride = std::max<size_t>(1, f.entities.size() / 32);
  value_type sum = 0.0;
  for (size_t i = 0; i < f.entities.size(); i += stride) {
    const auto e      = f.entities[i];
    const auto vel    = vector3_field_data(*f.velocity, e);
    const auto str    = matrix3_field_data(*f.stress, e);
    const auto ori    = quaternion_field_data(*f.orientation_out, e);
    const auto eng    = scalar_field_data(*f.energy, e);
    sum += vel[0] + 2.0*vel[1] + 3.0*vel[2];
    sum += 0.5*(str[0] + str[4] + str[8]);
    sum += ori.w() + ori.x() - ori.y() + 0.25*ori.z();
    sum += eng[0];
  }
  return sum;
}

template <typename RunnerA, typename RunnerB>
static void validate_equal(const std::string& label, RunnerA&& a, RunnerB&& b,
                            const RigidBodyFixture& f, bool sync_from_device) {
  a();
  if (sync_from_device) sync_outputs_to_host(f);
  const value_type sum_a = checksum(f);

  b();
  if (sync_from_device) sync_outputs_to_host(f);
  const value_type sum_b = checksum(f);

  const value_type scale = 1.0 + std::max(std::abs(sum_a), std::abs(sum_b));
  if (std::abs(sum_a - sum_b) > 1.0e-11 * scale)
    throw std::runtime_error("validate_equal failed for \"" + label + "\": "
                             + std::to_string(sum_a) + " vs " + std::to_string(sum_b));
}

// =============================================================================
// Options
// =============================================================================

enum class BenchmarkCase { Drag, Mobility, Complex, All };
enum class ExecutionSpace { Host, Ngp, All };

struct BenchOptions {
  bool           simple          = false;
  BenchmarkCase  benchmark_case  = BenchmarkCase::All;
  ExecutionSpace execution_space = ExecutionSpace::All;
  size_t         num_entities    = kDefaultNumEntities;
};

static bool should_run_case(const BenchOptions& opts, BenchmarkCase c) {
  return opts.benchmark_case == BenchmarkCase::All || opts.benchmark_case == c;
}
static bool should_run_space(const BenchOptions& opts, ExecutionSpace s) {
  return opts.execution_space == ExecutionSpace::All || opts.execution_space == s;
}

// =============================================================================
// --simple mode helpers
// =============================================================================

struct VariantSummary {
  std::string label;
  double      ns_per_entity = -1.0;
};

static std::string fmt_time(double ns) {
  if (ns < 0.0) return "   ---   ";
  char buf[24];
  if      (ns < 1e3) std::snprintf(buf, sizeof(buf), "%7.2f ns", ns);
  else if (ns < 1e6) std::snprintf(buf, sizeof(buf), "%7.2f us", ns * 1e-3);
  else               std::snprintf(buf, sizeof(buf), "%7.2f ms", ns * 1e-6);
  return buf;
}

static void print_simple_table(const std::string& title, const std::vector<VariantSummary>& rows) {
  constexpr int kLabelW = 12, kValW = 14;
  std::cout << "\n[" << title << "]  (median ns/entity)\n"
            << "  " << std::left  << std::setw(kLabelW) << "variant"
            << std::right << std::setw(kValW) << "time" << "\n"
            << "  " << std::string(kLabelW + kValW, '-') << "\n";
  for (const auto& r : rows)
    std::cout << "  " << std::left  << std::setw(kLabelW) << r.label
              << std::right << std::setw(kValW) << fmt_time(r.ns_per_entity) << "\n";
}

static ankerl::nanobench::Bench& configure_bench(ankerl::nanobench::Bench& b, std::ostringstream& out,
                                                  const BenchOptions& opts, const char* title) {
  return b.output(opts.simple ? nullptr : static_cast<std::ostream*>(&out))
          .title(title).unit("entity").batch(opts.num_entities)
          .relative(true).performanceCounters(true)
          .warmup(2).epochs(10).minEpochIterations(10)
          .minEpochTime(std::chrono::milliseconds(5));
}

static std::vector<VariantSummary> collect_summaries(const ankerl::nanobench::Bench& b, size_t n) {
  std::vector<VariantSummary> rows;
  for (const auto& r : b.results())
    rows.push_back({std::string(r.config().mBenchmarkName),
                    r.median(ankerl::nanobench::Result::Measure::elapsed) * 1e9 / static_cast<double>(n)});
  return rows;
}

// =============================================================================
// Drag benchmarks
// =============================================================================

static void bench_drag_host(const BenchOptions& opts) {
  auto f = build_fixture(opts.num_entities);
  FC_v force_comp(*f.force), vel_comp(*f.velocity);
  FC_s drag_fc(*f.drag_scalar);
  SC_s drag_sc(f.shared_drag_scalar);

  auto run = [&](auto drag) {
    mundy::mesh::for_each_entity_run(*f.bulk, stk::topology::NODE_RANK, f.selector,
        [=](const stk::mesh::BulkData&, stk::mesh::Entity e) {
            drag_velocity_step(drag(e), force_comp(e), vel_comp(e));
        });
  };
  auto run_field  = [&] { run(drag_fc); };
  auto run_shared = [&] { run(drag_sc); };

  validate_equal("host drag", run_field, run_shared, f, false);

  std::ostringstream out; ankerl::nanobench::Bench b;
  configure_bench(b, out, opts, "Shared components / Host drag");
  b.run("field",  [&] { run_field();  ankerl::nanobench::doNotOptimizeAway(*f.velocity); });
  b.run("shared", [&] { run_shared(); ankerl::nanobench::doNotOptimizeAway(*f.velocity); });

  if (opts.simple) print_simple_table("Shared components / Host drag", collect_summaries(b, opts.num_entities));
  else std::cout << out.str();
}

template <typename NgpMesh, typename ForceComp, typename VelComp, typename DragT>
void eval_drag_velocity_step(NgpMesh ngp_mesh, stk::mesh::Selector &selector, ForceComp ngp_force_comp,
                             VelComp ngp_vel_comp, DragT drag) {
    mundy::mesh::for_each_entity_run(ngp_mesh, stk::topology::NODE_RANK, selector,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& idx) {
            drag_velocity_step(drag(idx), ngp_force_comp(idx), ngp_vel_comp(idx));
        });
  }

static void bench_drag_ngp(const BenchOptions& opts) {
  auto f = build_fixture(opts.num_entities);
  FC_v force_comp(*f.force), vel_comp(*f.velocity);
  FC_s drag_fc(*f.drag_scalar);
  SC_s drag_sc(f.shared_drag_scalar);

  auto ngp_mesh       = stk::mesh::get_updated_ngp_mesh(*f.bulk);
  auto ngp_force_comp = get_updated_ngp_component(force_comp);
  auto ngp_vel_comp   = get_updated_ngp_component(vel_comp);
  auto ngp_drag_fc    = get_updated_ngp_component(drag_fc);
  auto ngp_drag_sc    = get_updated_ngp_component(drag_sc);  // by value: NgpSharedScalarComponent

  auto run = [&](auto drag) {
   // Can't have a KOKKOS_LAMBDA inside a generic lambda (CUDA restriction), so we need a separate function for the NGP drag step.
    eval_drag_velocity_step(ngp_mesh, f.selector, ngp_force_comp, ngp_vel_comp, drag);
  };
  auto run_field = [&] {
    ngp_force_comp.sync_to_device(); ngp_vel_comp.sync_to_device(); ngp_drag_fc.sync_to_device();
    run(ngp_drag_fc);
    ngp_vel_comp.modify_on_device(); Kokkos::fence();
  };
  auto run_shared = [&] {
    ngp_force_comp.sync_to_device(); ngp_vel_comp.sync_to_device(); ngp_drag_sc.sync_to_device();
    run(ngp_drag_sc);
    ngp_vel_comp.modify_on_device(); Kokkos::fence();
  };

  validate_equal("ngp drag", run_field, run_shared, f, true);

  std::ostringstream out; ankerl::nanobench::Bench b;
  configure_bench(b, out, opts, "Shared components / NGP drag");
  b.run("field",  [&] { run_field();  ankerl::nanobench::doNotOptimizeAway(ngp_vel_comp); });
  b.run("shared", [&] { run_shared(); ankerl::nanobench::doNotOptimizeAway(ngp_vel_comp); });

  if (opts.simple) print_simple_table("Shared components / NGP drag", collect_summaries(b, opts.num_entities));
  else std::cout << out.str();
}

// =============================================================================
// Mobility benchmarks
// =============================================================================

static void bench_mobility_host(const BenchOptions& opts) {
  auto f = build_fixture(opts.num_entities);
  FC_v force_comp(*f.force), vel_comp(*f.velocity);
  FC_q orient_comp(*f.orientation);
  FC_v ambient_fc(*f.ambient);
  FC_m mobility_fc(*f.drag);
  SC_v ambient_sc(f.shared_ambient);
  SC_m mobility_sc(f.shared_drag);

  auto run = [&](auto ambient, auto mobility) {
    mundy::mesh::for_each_entity_run(*f.bulk, stk::topology::NODE_RANK, f.selector,
        [=](const stk::mesh::BulkData&, stk::mesh::Entity e) {
            rigid_body_mobility_step(ambient(e), mobility(e), force_comp(e), orient_comp(e), vel_comp(e));
        });
  };
  auto run_field  = [&] { run(ambient_fc, mobility_fc); };
  auto run_shared = [&] { run(ambient_sc, mobility_sc); };

  validate_equal("host mobility", run_field, run_shared, f, false);

  std::ostringstream out; ankerl::nanobench::Bench b;
  configure_bench(b, out, opts, "Shared components / Host mobility");
  b.run("field",  [&] { run_field();  ankerl::nanobench::doNotOptimizeAway(*f.velocity); });
  b.run("shared", [&] { run_shared(); ankerl::nanobench::doNotOptimizeAway(*f.velocity); });

  if (opts.simple) print_simple_table("Shared components / Host mobility", collect_summaries(b, opts.num_entities));
  else std::cout << out.str();
}

template <typename NgpMesh, typename AmbientT, typename MobilityT, typename ForceComp, typename OrientComp,
          typename VelComp>
void eval_rigid_body_mobility_step(NgpMesh ngp_mesh, stk::mesh::Selector &selector, AmbientT ngp_ambient,
                                   MobilityT ngp_mobility, ForceComp ngp_force_comp, OrientComp ngp_orient_comp,
                                   VelComp ngp_vel_comp) {
    mundy::mesh::for_each_entity_run(ngp_mesh, stk::topology::NODE_RANK, selector,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& idx) {
            rigid_body_mobility_step(ngp_ambient(idx), ngp_mobility(idx),
                ngp_force_comp(idx), ngp_orient_comp(idx), ngp_vel_comp(idx));
        });
  }

static void bench_mobility_ngp(const BenchOptions& opts) {
  auto f = build_fixture(opts.num_entities);
  FC_v force_comp(*f.force), vel_comp(*f.velocity);
  FC_q orient_comp(*f.orientation);
  FC_v ambient_fc(*f.ambient);
  FC_m mobility_fc(*f.drag);
  SC_v ambient_sc(f.shared_ambient);
  SC_m mobility_sc(f.shared_drag);

  auto ngp_mesh        = stk::mesh::get_updated_ngp_mesh(*f.bulk);
  auto ngp_force_comp  = get_updated_ngp_component(force_comp);
  auto ngp_orient_comp = get_updated_ngp_component(orient_comp);
  auto ngp_vel_comp    = get_updated_ngp_component(vel_comp);
  auto ngp_ambient_fc  = get_updated_ngp_component(ambient_fc);
  auto ngp_mobility_fc = get_updated_ngp_component(mobility_fc);
  auto& ngp_ambient_sc  = get_updated_ngp_component(ambient_sc);
  auto& ngp_mobility_sc = get_updated_ngp_component(mobility_sc);

  auto run = [&](auto ambient, auto mobility) {
    eval_rigid_body_mobility_step(ngp_mesh, f.selector, ambient, mobility, ngp_force_comp, ngp_orient_comp, ngp_vel_comp);
  };
  auto run_field = [&] {
    ngp_force_comp.sync_to_device(); ngp_orient_comp.sync_to_device(); ngp_vel_comp.sync_to_device();
    ngp_ambient_fc.sync_to_device(); ngp_mobility_fc.sync_to_device();
    run(ngp_ambient_fc, ngp_mobility_fc);
    ngp_vel_comp.modify_on_device(); Kokkos::fence();
  };
  auto run_shared = [&] {
    ngp_force_comp.sync_to_device(); ngp_orient_comp.sync_to_device(); ngp_vel_comp.sync_to_device();
    ngp_ambient_sc.sync_to_device(); ngp_mobility_sc.sync_to_device();
    run(ngp_ambient_sc, ngp_mobility_sc);
    ngp_vel_comp.modify_on_device(); Kokkos::fence();
  };

  validate_equal("ngp mobility", run_field, run_shared, f, true);

  std::ostringstream out; ankerl::nanobench::Bench b;
  configure_bench(b, out, opts, "Shared components / NGP mobility");
  b.run("field",  [&] { run_field();  ankerl::nanobench::doNotOptimizeAway(ngp_vel_comp); });
  b.run("shared", [&] { run_shared(); ankerl::nanobench::doNotOptimizeAway(ngp_vel_comp); });

  if (opts.simple) print_simple_table("Shared components / NGP mobility", collect_summaries(b, opts.num_entities));
  else std::cout << out.str();
}

// =============================================================================
// Complex benchmarks
// =============================================================================

static void bench_complex_host(const BenchOptions& opts) {
  auto f = build_fixture(opts.num_entities);
  // Per-entity fields
  FC_v force_comp(*f.force), torque_comp(*f.torque), vel_comp(*f.velocity);
  FC_q orient_comp(*f.orientation), orient_out_comp(*f.orientation_out);
  FC_m mobility_comp(*f.mobility), stress_comp(*f.stress);
  FC_s energy_comp(*f.energy);
  // Control inputs
  FC_s dt_fc(*f.dt);      FC_v ambient_fc(*f.ambient);
  FC_m drag_fc(*f.drag);  FC_q target_fc(*f.target_orientation);
  SC_s dt_sc(f.shared_dt);     SC_v ambient_sc(f.shared_ambient);
  SC_m drag_sc(f.shared_drag); SC_q target_sc(f.shared_target_orientation);

  auto run = [&](auto dt, auto ambient, auto drag, auto target) {
    mundy::mesh::for_each_entity_run(*f.bulk, stk::topology::NODE_RANK, f.selector,
        [=](const stk::mesh::BulkData&, stk::mesh::Entity e) {
            rigid_body_step(dt(e), ambient(e), drag(e), target(e),
                force_comp(e), torque_comp(e), orient_comp(e), mobility_comp(e),
                vel_comp(e), stress_comp(e), orient_out_comp(e), energy_comp(e));
        });
  };
  auto run_field  = [&] { run(dt_fc, ambient_fc, drag_fc, target_fc); };
  auto run_shared = [&] { run(dt_sc, ambient_sc, drag_sc, target_sc); };

  validate_equal("host complex", run_field, run_shared, f, false);

  std::ostringstream out; ankerl::nanobench::Bench b;
  configure_bench(b, out, opts, "Shared components / Host complex");
  b.run("field",  [&] { run_field();  ankerl::nanobench::doNotOptimizeAway(*f.energy); });
  b.run("shared", [&] { run_shared(); ankerl::nanobench::doNotOptimizeAway(*f.energy); });

  if (opts.simple) print_simple_table("Shared components / Host complex", collect_summaries(b, opts.num_entities));
  else std::cout << out.str();
}

template <typename NgpMesh, typename DtT, typename AmbientT, typename DragT, typename TargetT, typename ForceComp,
          typename TorqueComp, typename OrientComp, typename MobilityComp, typename VelComp, typename StressComp,
          typename OrientOutComp, typename EnergyComp>
void eval_rigid_body_step(NgpMesh ngp_mesh, stk::mesh::Selector &selector, DtT dt, AmbientT ambient, DragT drag,
                          TargetT target, ForceComp ngp_force_comp, TorqueComp ngp_torque_comp,
                          OrientComp ngp_orient_comp, MobilityComp ngp_mobility_comp, VelComp ngp_vel_comp,
                          StressComp ngp_stress_comp, OrientOutComp ngp_orient_out_comp, EnergyComp ngp_energy_comp) {
    mundy::mesh::for_each_entity_run(ngp_mesh, stk::topology::NODE_RANK, selector,
        KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& idx) {
            rigid_body_step(dt(idx), ambient(idx), drag(idx), target(idx),
                ngp_force_comp(idx), ngp_torque_comp(idx), ngp_orient_comp(idx),
                ngp_mobility_comp(idx), ngp_vel_comp(idx), ngp_stress_comp(idx),
                ngp_orient_out_comp(idx), ngp_energy_comp(idx));
        });
  }

static void bench_complex_ngp(const BenchOptions& opts) {
  auto f = build_fixture(opts.num_entities);
  // Per-entity fields
  FC_v force_comp(*f.force), torque_comp(*f.torque), vel_comp(*f.velocity);
  FC_q orient_comp(*f.orientation), orient_out_comp(*f.orientation_out);
  FC_m mobility_comp(*f.mobility), stress_comp(*f.stress);
  FC_s energy_comp(*f.energy);
  // Control inputs
  FC_s dt_fc(*f.dt);      FC_v ambient_fc(*f.ambient);
  FC_m drag_fc(*f.drag);  FC_q target_fc(*f.target_orientation);
  SC_s dt_sc(f.shared_dt);     SC_v ambient_sc(f.shared_ambient);
  SC_m drag_sc(f.shared_drag); SC_q target_sc(f.shared_target_orientation);

  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(*f.bulk);
  // Per-entity NGP components
  auto ngp_force_comp      = get_updated_ngp_component(force_comp);
  auto ngp_torque_comp     = get_updated_ngp_component(torque_comp);
  auto ngp_orient_comp     = get_updated_ngp_component(orient_comp);
  auto ngp_mobility_comp   = get_updated_ngp_component(mobility_comp);
  auto ngp_vel_comp        = get_updated_ngp_component(vel_comp);
  auto ngp_stress_comp     = get_updated_ngp_component(stress_comp);
  auto ngp_orient_out_comp = get_updated_ngp_component(orient_out_comp);
  auto ngp_energy_comp     = get_updated_ngp_component(energy_comp);
  // Field-backed control NGP components
  auto ngp_dt_fc      = get_updated_ngp_component(dt_fc);
  auto ngp_ambient_fc = get_updated_ngp_component(ambient_fc);
  auto ngp_drag_fc    = get_updated_ngp_component(drag_fc);
  auto ngp_target_fc  = get_updated_ngp_component(target_fc);
  // Shared control NGP components (scalar by value; others by reference to cached NgpSharedComponent)
  auto  ngp_dt_sc      = get_updated_ngp_component(dt_sc);
  auto& ngp_ambient_sc = get_updated_ngp_component(ambient_sc);
  auto& ngp_drag_sc    = get_updated_ngp_component(drag_sc);
  auto& ngp_target_sc  = get_updated_ngp_component(target_sc);

  auto sync_per_entity_comps = [&] {
    ngp_force_comp.sync_to_device();      ngp_torque_comp.sync_to_device();
    ngp_orient_comp.sync_to_device();     ngp_mobility_comp.sync_to_device();
    ngp_vel_comp.sync_to_device();        ngp_stress_comp.sync_to_device();
    ngp_orient_out_comp.sync_to_device(); ngp_energy_comp.sync_to_device();
  };
  auto mark_outputs_modified = [&] {
    ngp_vel_comp.modify_on_device();        ngp_stress_comp.modify_on_device();
    ngp_orient_out_comp.modify_on_device(); ngp_energy_comp.modify_on_device();
    Kokkos::fence();
  };

  auto run = [&](auto dt, auto ambient, auto drag, auto target) {
    eval_rigid_body_step(ngp_mesh, f.selector, dt, ambient, drag, target,
        ngp_force_comp, ngp_torque_comp, ngp_orient_comp, ngp_mobility_comp,
        ngp_vel_comp, ngp_stress_comp, ngp_orient_out_comp, ngp_energy_comp);
  };
  auto run_field = [&] {
    sync_per_entity_comps();
    ngp_dt_fc.sync_to_device(); ngp_ambient_fc.sync_to_device(); ngp_drag_fc.sync_to_device(); ngp_target_fc.sync_to_device();
    run(ngp_dt_fc, ngp_ambient_fc, ngp_drag_fc, ngp_target_fc);
    mark_outputs_modified();
  };
  auto run_shared = [&] {
    sync_per_entity_comps();
    ngp_dt_sc.sync_to_device(); ngp_ambient_sc.sync_to_device(); ngp_drag_sc.sync_to_device(); ngp_target_sc.sync_to_device();
    run(ngp_dt_sc, ngp_ambient_sc, ngp_drag_sc, ngp_target_sc);
    mark_outputs_modified();
  };

  validate_equal("ngp complex", run_field, run_shared, f, true);

  std::ostringstream out; ankerl::nanobench::Bench b;
  configure_bench(b, out, opts, "Shared components / NGP complex");
  b.run("field",  [&] { run_field();  ankerl::nanobench::doNotOptimizeAway(ngp_energy_comp); });
  b.run("shared", [&] { run_shared(); ankerl::nanobench::doNotOptimizeAway(ngp_energy_comp); });

  if (opts.simple) print_simple_table("Shared components / NGP complex", collect_summaries(b, opts.num_entities));
  else std::cout << out.str();
}

// =============================================================================
// Dispatch
// =============================================================================

static void run_benchmarks(const BenchOptions& opts) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1)
    throw std::runtime_error("PerfTestSharedComponentsNew requires MPI size 1.");

  std::cout << "=== Shared Component Performance Benchmarks ===\n"
            << "  Entities : " << opts.num_entities << "\n"
            << "  Mode     : " << (opts.simple ? "--simple" : "full") << "\n\n";

  if (should_run_case(opts, BenchmarkCase::Drag)) {
    if (should_run_space(opts, ExecutionSpace::Host)) bench_drag_host(opts);
    if (should_run_space(opts, ExecutionSpace::Ngp))  bench_drag_ngp(opts);
  }
  if (should_run_case(opts, BenchmarkCase::Mobility)) {
    if (should_run_space(opts, ExecutionSpace::Host)) bench_mobility_host(opts);
    if (should_run_space(opts, ExecutionSpace::Ngp))  bench_mobility_ngp(opts);
  }
  if (should_run_case(opts, BenchmarkCase::Complex)) {
    if (should_run_space(opts, ExecutionSpace::Host)) bench_complex_host(opts);
    if (should_run_space(opts, ExecutionSpace::Ngp))  bench_complex_ngp(opts);
  }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char** argv) {
  BenchOptions opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--simple") { opts.simple = true; continue; }
    if (arg == "--case" && i + 1 < argc) {
      const std::string v(argv[++i]);
      if      (v == "drag")     opts.benchmark_case = BenchmarkCase::Drag;
      else if (v == "mobility") opts.benchmark_case = BenchmarkCase::Mobility;
      else if (v == "complex")  opts.benchmark_case = BenchmarkCase::Complex;
      else if (v == "all")      opts.benchmark_case = BenchmarkCase::All;
      else { std::cerr << "Unknown --case value: " << v << "\n"; return 1; }
      continue;
    }
    if (arg == "--space" && i + 1 < argc) {
      const std::string v(argv[++i]);
      if      (v == "host") opts.execution_space = ExecutionSpace::Host;
      else if (v == "ngp")  opts.execution_space = ExecutionSpace::Ngp;
      else if (v == "all")  opts.execution_space = ExecutionSpace::All;
      else { std::cerr << "Unknown --space value: " << v << "\n"; return 1; }
      continue;
    }
    if (arg == "--n" && i + 1 < argc) {
      const size_t n = static_cast<size_t>(std::stoull(argv[++i]));
      if (n == 0) { std::cerr << "--n must be positive\n"; return 1; }
      opts.num_entities = n;
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    return 1;
  }

  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    run_benchmarks(opts);
  }
  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
