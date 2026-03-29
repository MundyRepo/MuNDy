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

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_mesh/FieldViews.hpp>
#include <mundy_mesh/ForEachEntity.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_utils/throw_assert.hpp>
#include <stdexcept>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <string>
#include <vector>

#include "nanobench.h"

namespace mundy {

namespace mesh {

namespace {

using scalar_t = double;
using vector3_t = Vector3<scalar_t>;
using matrix3_t = Matrix3<scalar_t>;
using quaternion_t = Quaternion<scalar_t>;
using double_field_t = stk::mesh::Field<scalar_t>;

KOKKOS_INLINE_FUNCTION
matrix3_t make_spd_matrix3(const scalar_t base) {
  return matrix3_t(1.25 + base, 0.08 + 0.05 * base, -0.04 + 0.01 * base, 0.08 + 0.05 * base, 1.75 + 0.5 * base,
                   0.06 - 0.02 * base, -0.04 + 0.01 * base, 0.06 - 0.02 * base, 2.25 + 0.75 * base);
}

template <typename AmbientType, typename DragType, typename TargetOrientationType, typename ForceType,
          typename TorqueType, typename OrientationType, typename MobilityType, typename VelocityType,
          typename StressType, typename OrientationOutType, typename EnergyType>
KOKKOS_INLINE_FUNCTION void share_issue_step(const scalar_t dt, const AmbientType& ambient, const DragType& drag,
                                             const TargetOrientationType& target_orientation, const ForceType& force,
                                             const TorqueType& torque, const OrientationType& orientation,
                                             const MobilityType& mobility, VelocityType velocity, StressType stress,
                                             OrientationOutType orientation_out, EnergyType energy) {
  const auto total_force = force + ambient;
  const auto body_force = conjugate(orientation) * total_force;
  const auto body_spin = torque + cross(body_force, ambient);
  const quaternion_t spin_quaternion(0.0, body_spin[0], body_spin[1], body_spin[2]);
  const auto trial_orientation =
      normalize(target_orientation * orientation + (0.5 * dt) * (spin_quaternion * orientation));
  const auto effective_mobility = mobility + drag;
  const auto lab_velocity = orientation * (effective_mobility * body_force);
  const scalar_t inv_force_scale = 1.0 / (1.0 + norm(body_force));
  const auto corrected_velocity = lab_velocity + (dt * inv_force_scale) * cross(body_spin, total_force);
  const auto updated_stress = trial_orientation * effective_mobility + 0.25 * (mobility - drag);

  velocity = corrected_velocity;
  stress = updated_stress;
  orientation_out = trial_orientation;
  energy = dot(corrected_velocity, corrected_velocity) + 0.1 * dot(body_spin, body_spin) +
           dot(trial_orientation, target_orientation);
}

template <typename NgpFieldType>
struct NgpScalarFieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return scalar_field_data(ngp_field, entity_index);
  }
};

template <typename NgpFieldType>
struct NgpVector3FieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return vector3_field_data(ngp_field, entity_index);
  }
};

template <typename NgpFieldType>
struct NgpMatrix3FieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return matrix3_field_data(ngp_field, entity_index);
  }
};

template <typename NgpFieldType>
struct NgpQuaternionFieldAccessor {
  NgpFieldType field;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const stk::mesh::FastMeshIndex& entity_index) const {
    auto& ngp_field = const_cast<NgpFieldType&>(field);
    return quaternion_field_data(ngp_field, entity_index);
  }
};

template <typename SharedType>
struct NgpOwningSharedAccessor {
  SharedType value;

  KOKKOS_INLINE_FUNCTION
  const SharedType& operator()(const stk::mesh::FastMeshIndex& /*entity_index*/) const {
    return value;
  }
};

template <typename DtAccessor, typename AmbientAccessor, typename DragAccessor, typename TargetAccessor,
          typename ForceAccessor, typename TorqueAccessor, typename OrientationAccessor, typename MobilityAccessor,
          typename VelocityAccessor, typename StressAccessor, typename OrientationOutAccessor, typename EnergyAccessor>
void run_ngp_share_issue_kernel(stk::mesh::NgpMesh& ngp_mesh, const stk::mesh::Selector& selector,
                                const DtAccessor& dt_accessor, const AmbientAccessor& ambient_accessor,
                                const DragAccessor& drag_accessor, const TargetAccessor& target_accessor,
                                const ForceAccessor& force_accessor, const TorqueAccessor& torque_accessor,
                                const OrientationAccessor& orientation_accessor,
                                const MobilityAccessor& mobility_accessor, const VelocityAccessor& velocity_accessor,
                                const StressAccessor& stress_accessor,
                                const OrientationOutAccessor& orientation_out_accessor,
                                const EnergyAccessor& energy_accessor) {
  for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex& entity_index) {
        share_issue_step(dt_accessor(entity_index), ambient_accessor(entity_index), drag_accessor(entity_index),
                         target_accessor(entity_index), force_accessor(entity_index), torque_accessor(entity_index),
                         orientation_accessor(entity_index), mobility_accessor(entity_index),
                         velocity_accessor(entity_index), stress_accessor(entity_index),
                         orientation_out_accessor(entity_index), energy_accessor(entity_index));
      });
}

class ShareIssueState {
 public:
  explicit ShareIssueState(const size_t num_entities)
      : meta_data_ptr_(),
        bulk_data_ptr_(),
        selector_(),
        force_field_ptr_(nullptr),
        torque_field_ptr_(nullptr),
        orientation_field_ptr_(nullptr),
        mobility_field_ptr_(nullptr),
        velocity_field_ptr_(nullptr),
        stress_field_ptr_(nullptr),
        orientation_out_field_ptr_(nullptr),
        energy_field_ptr_(nullptr),
        dt_field_ptr_(nullptr),
        ambient_field_ptr_(nullptr),
        drag_field_ptr_(nullptr),
        target_orientation_field_ptr_(nullptr),
        entities_(),
        shared_dt_(0.015),
        shared_ambient_(0.25, -0.15, 0.35),
        shared_drag_(make_spd_matrix3(0.35)),
        shared_target_orientation_(normalize(quaternion_t(1.0, 0.04, -0.08, 0.03))) {
    stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    meta_data_ptr_ = builder.create_meta_data();
    stk::mesh::MetaData& meta_data = *meta_data_ptr_;
    meta_data.use_simple_fields();

    bulk_data_ptr_ = builder.create(meta_data_ptr_);
    stk::mesh::BulkData& bulk_data = *bulk_data_ptr_;

    force_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "FORCE");
    torque_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "TORQUE");
    orientation_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "ORIENTATION");
    mobility_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "MOBILITY");
    velocity_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "VELOCITY");
    stress_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "STRESS");
    orientation_out_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "ORIENTATION_OUT");
    energy_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "ENERGY");
    dt_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "DT");
    ambient_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "AMBIENT");
    drag_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "DRAG");
    target_orientation_field_ptr_ = &meta_data.declare_field<scalar_t>(stk::topology::NODE_RANK, "TARGET_ORIENTATION");

    stk::mesh::put_field_on_mesh(*force_field_ptr_, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(*torque_field_ptr_, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(*orientation_field_ptr_, meta_data.universal_part(), 4, nullptr);
    stk::mesh::put_field_on_mesh(*mobility_field_ptr_, meta_data.universal_part(), 9, nullptr);
    stk::mesh::put_field_on_mesh(*velocity_field_ptr_, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(*stress_field_ptr_, meta_data.universal_part(), 9, nullptr);
    stk::mesh::put_field_on_mesh(*orientation_out_field_ptr_, meta_data.universal_part(), 4, nullptr);
    stk::mesh::put_field_on_mesh(*energy_field_ptr_, meta_data.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*dt_field_ptr_, meta_data.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*ambient_field_ptr_, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(*drag_field_ptr_, meta_data.universal_part(), 9, nullptr);
    stk::mesh::put_field_on_mesh(*target_orientation_field_ptr_, meta_data.universal_part(), 4, nullptr);
    meta_data.commit();

    selector_ = meta_data.universal_part();

    bulk_data.modification_begin();
    entities_.reserve(num_entities);
    for (size_t entity_index = 0; entity_index < num_entities; ++entity_index) {
      const stk::mesh::Entity node = bulk_data.declare_node(entity_index + 1);
      entities_.push_back(node);

      const scalar_t phase = 1.0e-5 * static_cast<scalar_t>(entity_index);
      const vector3_t force(1.25 + 0.25 * phase, -0.85 + 0.15 * phase, 0.65 - 0.10 * phase);
      const vector3_t torque(0.35 + 0.12 * phase, -0.28 + 0.05 * phase, 0.18 - 0.02 * phase);
      const quaternion_t orientation = normalize(quaternion_t(1.0, 0.18 * phase, -0.11 * phase, 0.09 * phase));
      const matrix3_t mobility = make_spd_matrix3(0.20 + 0.05 * phase);

      vector3_field_data(*force_field_ptr_, node) = force;
      vector3_field_data(*torque_field_ptr_, node) = torque;
      quaternion_field_data(*orientation_field_ptr_, node) = orientation;
      matrix3_field_data(*mobility_field_ptr_, node) = mobility;
      vector3_field_data(*velocity_field_ptr_, node) = vector3_t(0.0, 0.0, 0.0);
      matrix3_field_data(*stress_field_ptr_, node) = matrix3_t(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      quaternion_field_data(*orientation_out_field_ptr_, node) = quaternion_t(1.0, 0.0, 0.0, 0.0);
      scalar_field_data(*energy_field_ptr_, node)[0] = 0.0;

      // These controls are constant across entities in both benchmark paths.
      scalar_field_data(*dt_field_ptr_, node)[0] = shared_dt_;
      vector3_field_data(*ambient_field_ptr_, node) = shared_ambient_;
      matrix3_field_data(*drag_field_ptr_, node) = shared_drag_;
      quaternion_field_data(*target_orientation_field_ptr_, node) = shared_target_orientation_;
    }
    bulk_data.modification_end();
  }

  stk::mesh::BulkData& bulk_data() const {
    return *bulk_data_ptr_;
  }

  const stk::mesh::Selector& selector() const {
    return selector_;
  }

  double_field_t& force_field() const {
    return *force_field_ptr_;
  }

  double_field_t& torque_field() const {
    return *torque_field_ptr_;
  }

  double_field_t& orientation_field() const {
    return *orientation_field_ptr_;
  }

  double_field_t& mobility_field() const {
    return *mobility_field_ptr_;
  }

  double_field_t& velocity_field() const {
    return *velocity_field_ptr_;
  }

  double_field_t& stress_field() const {
    return *stress_field_ptr_;
  }

  double_field_t& orientation_out_field() const {
    return *orientation_out_field_ptr_;
  }

  double_field_t& energy_field() const {
    return *energy_field_ptr_;
  }

  double_field_t& dt_field() const {
    return *dt_field_ptr_;
  }

  double_field_t& ambient_field() const {
    return *ambient_field_ptr_;
  }

  double_field_t& drag_field() const {
    return *drag_field_ptr_;
  }

  double_field_t& target_orientation_field() const {
    return *target_orientation_field_ptr_;
  }

  const scalar_t& shared_dt() const {
    return shared_dt_;
  }

  const matrix3_t& shared_drag() const {
    return shared_drag_;
  }

  void sync_outputs_to_host() const {
    velocity_field().sync_to_host();
    stress_field().sync_to_host();
    orientation_out_field().sync_to_host();
    energy_field().sync_to_host();
  }

  scalar_t checksum() const {
    const size_t stride = std::max<size_t>(size_t(1), entities_.size() / 32);
    scalar_t sum = 0.0;

    for (size_t entity_index = 0; entity_index < entities_.size(); entity_index += stride) {
      const stk::mesh::Entity entity = entities_[entity_index];
      const auto velocity = vector3_field_data(*velocity_field_ptr_, entity);
      const auto stress = matrix3_field_data(*stress_field_ptr_, entity);
      const auto orientation = quaternion_field_data(*orientation_out_field_ptr_, entity);
      const auto energy = scalar_field_data(*energy_field_ptr_, entity);

      sum += velocity[0] + 2.0 * velocity[1] + 3.0 * velocity[2];
      sum += 0.5 * (stress[0] + stress[4] + stress[8]);
      sum += orientation.w() + orientation.x() - orientation.y() + 0.25 * orientation.z();
      sum += energy[0];
    }

    return sum;
  }

 private:
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr_;
  stk::mesh::Selector selector_;
  double_field_t* force_field_ptr_;
  double_field_t* torque_field_ptr_;
  double_field_t* orientation_field_ptr_;
  double_field_t* mobility_field_ptr_;
  double_field_t* velocity_field_ptr_;
  double_field_t* stress_field_ptr_;
  double_field_t* orientation_out_field_ptr_;
  double_field_t* energy_field_ptr_;
  double_field_t* dt_field_ptr_;
  double_field_t* ambient_field_ptr_;
  double_field_t* drag_field_ptr_;
  double_field_t* target_orientation_field_ptr_;
  std::vector<stk::mesh::Entity> entities_;
  scalar_t shared_dt_;
  vector3_t shared_ambient_;
  matrix3_t shared_drag_;
  quaternion_t shared_target_orientation_;
};

template <typename RunnerA, typename RunnerB>
void validate_equal(const std::string& label, RunnerA&& runner_a, RunnerB&& runner_b, ShareIssueState& state) {
  runner_a();
  state.sync_outputs_to_host();
  const scalar_t checksum_a = state.checksum();

  runner_b();
  state.sync_outputs_to_host();
  const scalar_t checksum_b = state.checksum();

  const scalar_t scale = 1.0 + std::max(std::abs(checksum_a), std::abs(checksum_b));
  MUNDY_THROW_REQUIRE(std::abs(checksum_a - checksum_b) <= 1.0e-11 * scale, std::runtime_error,
                      "PerfTestShareIssue validation failed for " + label + ". checksum_a = " +
                          std::to_string(checksum_a) + ", checksum_b = " + std::to_string(checksum_b));
}

void run_test() {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    std::cout << "PerfTestShareIssue requires MPI size 1, skipping." << std::endl;
    return;
  }

  constexpr size_t num_entities = 100000;
  ShareIssueState state(num_entities);

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

  auto sync_ngp_fields = [&] {
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

  auto mark_outputs_modified = [&] {
    ngp_velocity_field.modify_on_device();
    ngp_stress_field.modify_on_device();
    ngp_orientation_out_field.modify_on_device();
    ngp_energy_field.modify_on_device();
    Kokkos::fence();
  };

  auto run_ngp_direct_all_field = [&] {
    sync_ngp_fields();
    run_ngp_share_issue_kernel(
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
    mark_outputs_modified();
  };

  auto run_ngp_direct_mixed_owning_shared = [&] {
    sync_ngp_fields();
    run_ngp_share_issue_kernel(
        ngp_mesh, state.selector(), NgpOwningSharedAccessor<scalar_t>{state.shared_dt()},
        NgpVector3FieldAccessor<decltype(ngp_ambient_field)>{ngp_ambient_field},
        NgpOwningSharedAccessor<matrix3_t>{state.shared_drag()},
        NgpQuaternionFieldAccessor<decltype(ngp_target_orientation_field)>{ngp_target_orientation_field},
        NgpVector3FieldAccessor<decltype(ngp_force_field)>{ngp_force_field},
        NgpVector3FieldAccessor<decltype(ngp_torque_field)>{ngp_torque_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_field)>{ngp_orientation_field},
        NgpMatrix3FieldAccessor<decltype(ngp_mobility_field)>{ngp_mobility_field},
        NgpVector3FieldAccessor<decltype(ngp_velocity_field)>{ngp_velocity_field},
        NgpMatrix3FieldAccessor<decltype(ngp_stress_field)>{ngp_stress_field},
        NgpQuaternionFieldAccessor<decltype(ngp_orientation_out_field)>{ngp_orientation_out_field},
        NgpScalarFieldAccessor<decltype(ngp_energy_field)>{ngp_energy_field});
    mark_outputs_modified();
  };

  validate_equal("ngp field-backed vs mixed-owning-shared", run_ngp_direct_all_field,
                 run_ngp_direct_mixed_owning_shared, state);

  ankerl::nanobench::Bench bench;
  bench.relative(true)
      .title("PerfTestShareIssue / NGP field vs mixed-owning-shared")
      .unit("entity")
      .batch(num_entities)
      .performanceCounters(true)
      .minEpochIterations(200);

  bench.run("direct field-backed controls", [&] {
    run_ngp_direct_all_field();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
  });
  bench.run("direct mixed controls (owning-shared dt+drag)", [&] {
    run_ngp_direct_mixed_owning_shared();
    ankerl::nanobench::doNotOptimizeAway(ngp_energy_field);
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
