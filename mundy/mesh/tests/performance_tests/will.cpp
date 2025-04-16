
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

// Openrand
#include <openrand/philox.h>  // for openrand::Philox

// Kokkos
#include <Kokkos_Core.hpp>
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>         // stk::mesh::EntityRank
#include <stk_topology/topology.hpp>       // stk::topology
#include <stk_util/ngp/NgpSpaces.hpp>      // stk::ngp::ExecSpace, stk::ngp::RangePolicy
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_geom/distance.hpp>         // for mundy::geom::distance
#include <mundy_geom/primitives.hpp>       // for mundy::geom::Spherocylinder
#include <mundy_math/Quaternion.hpp>       // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
#include <mundy_mesh/Aggregate.hpp>        // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/LinkData.hpp>         // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>     // for mundy::mesh::field_copy

/*
Crosslinks no longer teleport to zero length when then unbind. They just unbind.
Crosslinkers are represented as two pseudo-spheres that brownian diffuse independently.
Crosslinkers always impose their linear spring forces. Assumes that angular spring at the feet
instantaneously relax upon unbinding. Crosslinkers only bind within some small cutoff distance
measured from the head... Debye screening for unbound to left/right bound.


  - Brownian diffusion of unbound crosslink heads
  - Unbound to left/right bound
  - Left/right to doubly/unbound
  - Doubly bound to left/right bound
*/

struct COORDS {};
struct VELOCITY {};
struct FORCE {};
struct TORQUE {};
struct RADIUS {};
struct LENGTH {};
struct ARCLENGTH {};
struct OMEGA {};
struct LINKED_ENTITIES {};
struct QUAT {};
struct TANGENT {};
struct REST_LENGTH {};
struct SPRING_CONSTANT {};
struct ANG_SPRING_CONSTANT {};
struct REST_COS_ANGLE {};
struct RNG_COUNTER {};
struct REST_COS_TORSION_ANGLE {};
struct TORSION_SPRING_CONSTANT {};

struct BINDING_RATE {};
struct UNBINDING_RATE {};
struct BIND_SITE_SPACING {};
struct IS_BOUND {};
struct SCRATCH_D1 {};
struct SCRATCH_D3 {};

namespace mundy {

Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace> get_local_entity_indices(
    const stk::mesh::BulkData& bulk_data, stk::mesh::EntityRank rank, stk::mesh::Selector selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace> mesh_indices("mesh_indices", local_entities.size());
  Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace>::HostMirror host_mesh_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mesh_indices);

  for (size_t i = 0; i < local_entities.size(); ++i) {
    const stk::mesh::MeshIndex& mesh_index = bulk_data.mesh_index(local_entities[i]);
    host_mesh_indices(i) = stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
  }

  Kokkos::deep_copy(mesh_indices, host_mesh_indices);
  return mesh_indices;
}

class apply_brownian_motion_rod {
 public:
  apply_brownian_motion_rod(double dt, double kbt)
      : dt_(dt), kbt_(kbt), kbt_coeff_(Kokkos::sqrt(2.0 * kbt_ / dt_)), a_small_number_(0.1 * dt_) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& rod_view) const {
    // Setup the rng. Seed by rod id and use the rod's counter
    stk::mesh::EntityId rod_id = rod_view.entity_id();
    auto rod_counter = get<RNG_COUNTER>(rod_view);
    openrand::Philox rng(rod_id, rod_counter[0]);
    rod_counter[0]++;

    // Node data
    auto velocity = get<VELOCITY>(rod_view, 0);
    auto omega = get<OMEGA>(rod_view, 0);
    auto quat = get<QUAT>(rod_view, 0);

    // Rod data
    double radius = get<RADIUS>(rod_view)[0];
    double length = get<LENGTH>(rod_view)[0];

    // Slender fiber has 0 rot drag about the long axis, regularize with identity rot mobility
    constexpr double pi = Kokkos::numbers::pi_v<double>;
    const double viscosity = 1;
    const double p = (length + 2 * radius) / (2 * radius);
    const double inv_drag_perp = 4 * pi * viscosity / (Kokkos::log(p) + 0.839 + 0.185 / p + 0.233 / (p * p));
    const double inv_drag_para = 2 * pi * viscosity / (Kokkos::log(p) - 0.207 + 0.98 / p - 0.133 / (p * p));
    const double inv_drag_rot = (pi * viscosity * p * p / 3) / (Kokkos::log(p) - 0.662 + 0.917 / p - 0.053 / (p * p));

    // RFD from Delong, JCP, 2015
    auto tangent = quat * math::Vector3<double>(0.0, 0.0, 1.0);

    auto n_mat = (inv_drag_para - inv_drag_perp) * math::outer_product(tangent, tangent) +
                 inv_drag_perp * math::Matrix3<double>::identity();
    auto n_mat_sqrt = math::cholesky(n_mat);  // This is just L in N = LL^T

    // Drift and diffusion
    math::Vector3<double> w_rot(rng.randn<double>(), rng.randn<double>(), rng.randn<double>());
    math::Vector3<double> w_pos(rng.randn<double>(), rng.randn<double>(), rng.randn<double>());
    math::Vector3<double> w_rfd_rot(rng.randn<double>(), rng.randn<double>(), rng.randn<double>());

    math::rotate_quaternion(quat, w_rfd_rot, a_small_number_);          // Update the quaternion for the rod
    tangent = quat * math::Vector3<double>(0.0, 0.0, 1.0);  // Propagate to the tangent
    auto n_mat_rfd = (inv_drag_para - inv_drag_perp) * math::outer_product(tangent, tangent) +
                     inv_drag_perp * math::Matrix3<double>::identity();  // Propagate to the mobility matrix

    auto vel_brown = kbt_coeff_ * (n_mat_sqrt * w_pos);                     // Gaussian noise
    vel_brown += (kbt_ / a_small_number_) * ((n_mat_rfd - n_mat) * w_pos);  // RFD drift
    auto omega_brown = Kokkos::sqrt(inv_drag_rot) * kbt_coeff_ * w_rot;     // Regularized identity rotation drag

    // Update the velocity and omega
    velocity += vel_brown;
    omega += omega_brown;
  }

  void apply_to(auto& rod_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Rod must be particle top.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, OMEGA, QUAT, RNG_COUNTER, RADIUS, LENGTH>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }

  void apply_to(auto& rod_agg) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Rod must be particle top.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, OMEGA, QUAT, RNG_COUNTER, RADIUS, LENGTH>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }

 private:
  double dt_;
  double kbt_;
  double kbt_coeff_;
  double a_small_number_;  // In Delong 2015, this is delta
};

class apply_brownian_motion_unbound_prc1 {
 public:
  apply_brownian_motion_unbound_prc1(double dt, double kbt) : dt_(dt), kbt_(kbt) {
  }

  template <typename EntityViewType>
  KOKKOS_INLINE_FUNCTION void operator()(const EntityViewType& sphere_view) const
    requires(EntityViewType::topology() == stk::topology::PARTICLE)
  {
    // Setup the rng. Seed by rod id and use the rod's counter
    stk::mesh::EntityId sphere_id = sphere_view.entity_id();
    auto sphere_counter = get<RNG_COUNTER>(sphere_view);
    openrand::Philox rng(sphere_id, sphere_counter[0]);
    sphere_counter[0]++;

    // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
    constexpr double pi = Kokkos::numbers::pi_v<double>;
    const double viscosity = 1;
    double radius = get<RADIUS>(sphere_view)[0];
    const double drag_coeff = 6.0 * pi * viscosity * radius;
    const double inv_drag_coeff = 1.0 / drag_coeff;
    const double coeff = std::sqrt(2.0 * kbt_ * drag_coeff / dt_) * inv_drag_coeff;

    auto velocity = get<VELOCITY>(sphere_view, 0);
    velocity[0] += coeff * rng.randn<double>() * (!get<IS_BOUND>(sphere_view, 0)[0]);
    velocity[1] += coeff * rng.randn<double>() * (!get<IS_BOUND>(sphere_view, 0)[0]);
    velocity[2] += coeff * rng.randn<double>() * (!get<IS_BOUND>(sphere_view, 0)[0]);
  }

  template <typename EntityViewType>
  KOKKOS_INLINE_FUNCTION void operator()(const EntityViewType& sphere_view) const
    requires(EntityViewType::topology() == stk::topology::NODE)
  {
    // Setup the rng. Seed by rod id and use the rod's counter
    stk::mesh::EntityId sphere_id = sphere_view.entity_id();
    auto sphere_counter = get<RNG_COUNTER>(sphere_view);
    openrand::Philox rng(sphere_id, sphere_counter[0]);
    sphere_counter[0]++;

    // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
    constexpr double pi = Kokkos::numbers::pi_v<double>;
    const double viscosity = 1;
    double radius = get<RADIUS>(sphere_view)[0];
    const double drag_coeff = 6.0 * pi * viscosity * radius;
    const double inv_drag_coeff = 1.0 / drag_coeff;
    const double coeff = std::sqrt(2.0 * kbt_ * drag_coeff / dt_) * inv_drag_coeff;

    auto velocity = get<VELOCITY>(sphere_view);
    velocity[0] += coeff * rng.randn<double>() * (!get<IS_BOUND>(sphere_view)[0]);
    velocity[1] += coeff * rng.randn<double>() * (!get<IS_BOUND>(sphere_view)[0]);
    velocity[2] += coeff * rng.randn<double>() * (!get<IS_BOUND>(sphere_view)[0]);
  }

  void apply_to(auto& sphere_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(sphere_agg.topology() == stk::topology::PARTICLE || sphere_agg.topology() == stk::topology::NODE,
                  "Sphere aggregate must be particle or node top.");

    auto ngp_sphere_agg = mesh::get_updated_ngp_aggregate(sphere_agg);
    ngp_sphere_agg.template sync_to_device<VELOCITY, RNG_COUNTER, RADIUS>();
    ngp_sphere_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_sphere_agg.template modify_on_device<VELOCITY>();
  }

  void apply_to(auto& sphere_agg) {
    static_assert(sphere_agg.topology() == stk::topology::PARTICLE || sphere_agg.topology() == stk::topology::NODE,
                  "Sphere aggregate must be particle or node top.");

    auto ngp_sphere_agg = mesh::get_updated_ngp_aggregate(sphere_agg);
    ngp_sphere_agg.template sync_to_device<VELOCITY, RNG_COUNTER, RADIUS>();
    ngp_sphere_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_sphere_agg.template modify_on_device<VELOCITY>();
  }

 private:
  double dt_;
  double kbt_;
};

class eval_prc1_force {
 public:
  eval_prc1_force() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& prc1_view) const {
    auto rA = get<COORDS>(prc1_view, 0);
    auto rB = get<COORDS>(prc1_view, 1);
    auto a1 = get<TANGENT>(prc1_view, 0);
    auto b1 = get<TANGENT>(prc1_view, 1);

    double k_lin = get<SPRING_CONSTANT>(prc1_view)[0];
    double r0_lin = get<REST_LENGTH>(prc1_view)[0];
    double k_ang = get<ANG_SPRING_CONSTANT>(prc1_view)[0];  // TODO: store on each foot
    double cos_theta0 = get<REST_COS_ANGLE>(prc1_view)[0];  // TODO: store on each foot
    double k_torsion = get<TORSION_SPRING_CONSTANT>(prc1_view)[0];
    double cos_theta0_torsion = get<REST_COS_TORSION_ANGLE>(prc1_view)[0];

    auto rAB_vec = rA - rB; /* Note the backwards convention from Allen and Germano */
    auto rAB_norm = math::norm(rAB_vec);
    auto rAB_hat = rAB_vec / rAB_norm;

    double delta_cosA = math::dot(a1, rAB_hat) - cos_theta0;
    double delta_cosB = math::dot(b1, rAB_hat) - cos_theta0;
    auto triple_crossA = math::cross(rAB_hat, math::cross(rAB_hat, a1));
    auto triple_crossB = math::cross(rAB_hat, math::cross(rAB_hat, b1));

    auto crossA = math::cross(rAB_hat, a1);
    auto normof_crossA = math::norm(crossA);
    crossA /= (normof_crossA < 1e-12) ? 1.0 : normof_crossA;

    auto crossB = math::cross(rAB_hat, b1);
    auto normof_crossB = math::norm(crossB);
    crossB /= (normof_crossB < 1e-12) ? 1.0 : normof_crossB;

    auto fA = -k_lin * (rAB_norm - r0_lin) * rAB_hat  //
              + k_ang / rAB_norm * (delta_cosA * triple_crossA + delta_cosB * triple_crossB);
    auto torqueA = k_ang * delta_cosA * math::cross(rAB_hat, a1) -  //
                   k_torsion * (math::dot(crossA, crossB) - cos_theta0_torsion) * rAB_hat;
    auto torqueB = k_ang * delta_cosB * math::cross(rAB_hat, b1) +  //
                   k_torsion * (math::dot(crossA, crossB) - cos_theta0_torsion) * rAB_hat;

    // We know that the downward connected nodes are only connected to us, therefore we do not need atomics,
    // note that we are also not summing in the forces
    get<FORCE>(prc1_view, 0) = fA;
    get<FORCE>(prc1_view, 1) = -fA;
    get<TORQUE>(prc1_view, 0) = torqueA;
    get<TORQUE>(prc1_view, 1) = torqueB;
  }

  void apply_to(auto& prc1_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(prc1_agg.topology() == stk::topology::BEAM_2, "PRC1 must be beam 2 top.");

    auto ngp_prc1_agg = mesh::get_updated_ngp_aggregate(prc1_agg);
    ngp_prc1_agg.template sync_to_device<COORDS, FORCE, TORQUE, SPRING_CONSTANT, REST_LENGTH, ANG_SPRING_CONSTANT,
                                         TORSION_SPRING_CONSTANT, REST_COS_ANGLE, REST_COS_TORSION_ANGLE>();
    ngp_prc1_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_prc1_agg.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& prc1_agg) {
    static_assert(prc1_agg.topology() == stk::topology::BEAM_2, "PRC1 must be beam 2 top.");

    auto ngp_prc1_agg = mesh::get_updated_ngp_aggregate(prc1_agg);
    ngp_prc1_agg.template sync_to_device<COORDS, FORCE, TORQUE, SPRING_CONSTANT, REST_LENGTH, ANG_SPRING_CONSTANT,
                                         TORSION_SPRING_CONSTANT, REST_COS_ANGLE, REST_COS_TORSION_ANGLE>();
    ngp_prc1_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_prc1_agg.template modify_on_device<FORCE, TORQUE>();
  }
};

template <typename RodAgg, typename SurfaceNodeAgg>
class map_surface_force_to_com_force {
 public:
  map_surface_force_to_com_force(mesh::LinkData& link_data, RodAgg& rod_agg, SurfaceNodeAgg& surface_node_agg)
      : link_data_(link_data), rod_agg_(rod_agg), surface_node_agg_(surface_node_agg) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& slink_view) const {
    // Get the two entities from the slink_view, this is already inside of a parallel loop
    auto rod_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[0]);
    auto surface_node_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[1]);

    // Go from entity view to the aggregate
    auto rod_view = ngp_rod_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(rod_entity));
    auto surface_node_view = ngp_surface_node_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(surface_node_entity));

    auto rod_force = get<FORCE>(rod_view, 0 /* center node */);
    auto rod_torque = get<TORQUE>(rod_view, 0);
    auto rod_center = get<COORDS>(rod_view, 0);
    auto surface_node_force = get<FORCE>(surface_node_view);
    auto surface_node_torque = get<TORQUE>(surface_node_view);
    auto surface_node_pos = get<COORDS>(surface_node_view);

    // Add the COM force
    Kokkos::atomic_add(&rod_force[0], surface_node_force[0]);
    Kokkos::atomic_add(&rod_force[1], surface_node_force[1]);
    Kokkos::atomic_add(&rod_force[2], surface_node_force[2]);

    // Add the torques along with the off-center force contribution
    auto torque = surface_node_torque + math::cross(surface_node_pos - rod_center, surface_node_force);
    Kokkos::atomic_add(&rod_torque[0], torque[0]);
    Kokkos::atomic_add(&rod_torque[1], torque[1]);
    Kokkos::atomic_add(&rod_torque[2], torque[2]);
  }

  void apply_to(auto& slink_agg, const stk::mesh::Selector& subset_selector) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    // Explicitly construct a local instance of the NGP rod_agg and surface_node_agg
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();
    ngp_surface_node_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);

    ngp_rod_agg_.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& slink_agg) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    // Explicitly construct a local instance of the NGP rod_agg and surface_node_agg
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();
    ngp_surface_node_agg_.template sync_to_device<FORCE, TORQUE, COORDS>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/);

    ngp_rod_agg_.template modify_on_device<FORCE, TORQUE>();
  }

 private:
  mesh::LinkData& link_data_;
  RodAgg& rod_agg_;
  SurfaceNodeAgg& surface_node_agg_;
  using rod_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<RodAgg>()));
  using surface_node_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<SurfaceNodeAgg>()));
  rod_agg_ngp_t ngp_rod_agg_;
  surface_node_agg_ngp_t ngp_surface_node_agg_;
};

class eval_mobility_rod {
 public:
  eval_mobility_rod() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& rod_view) const {
    // Center node data:
    auto force = get<FORCE>(rod_view, 0);
    auto torque = get<TORQUE>(rod_view, 0);
    auto velocity = get<VELOCITY>(rod_view, 0);
    auto omega = get<OMEGA>(rod_view, 0);
    auto tangent = get<TANGENT>(rod_view, 0);

    // Rod data:
    double length = get<LENGTH>(rod_view)[0];
    double radius = get<RADIUS>(rod_view)[0];

    // Slender fiber has 0 rot drag about the long axis, regularize with identity rot mobility
    constexpr double pi = Kokkos::numbers::pi_v<double>;
    const double viscosity = 1;
    const double p = (length + 2 * radius) / (2 * radius);
    const double inv_drag_perp = 4 * pi * viscosity / (Kokkos::log(p) + 0.839 + 0.185 / p + 0.233 / (p * p));
    const double inv_drag_para = 2 * pi * viscosity / (Kokkos::log(p) - 0.207 + 0.98 / p - 0.133 / (p * p));
    const double inv_drag_rot = (pi * viscosity * p * p / 3) / (Kokkos::log(p) - 0.662 + 0.917 / p - 0.053 / (p * p));

    // Note, += is safe here without atomic under the assumption that no two particles share a node
    auto force_para = math::dot(force, tangent) * tangent;
    auto force_perp = force - force_para;
    velocity += inv_drag_para * force_para + inv_drag_perp * force_perp;
    omega += inv_drag_rot * torque;
  }

  void apply_to(auto& rod_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected a particle topology.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<FORCE, TORQUE, VELOCITY, OMEGA, LENGTH, RADIUS, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }

  void apply_to(auto& rod_agg) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected a particle topology.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<FORCE, TORQUE, VELOCITY, OMEGA, LENGTH, RADIUS, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<VELOCITY, OMEGA>();
  }
};

class eval_mobility_unbound_prc1 {
 public:
  eval_mobility_unbound_prc1() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& prc1_head_view) const {
    auto force = get<FORCE>(prc1_head_view);
    auto velocity = get<VELOCITY>(prc1_head_view);
    double radius = get<RADIUS>(prc1_head_view)[0];

    constexpr double pi = Kokkos::numbers::pi_v<double>;
    const double viscosity = 1;
    const double inv_drag_coeff = 1.0 / (6.0 * pi * viscosity * radius);
    velocity += inv_drag_coeff * force * (!get<IS_BOUND>(prc1_head_view)[0]);
  }

  void apply_to(auto& prc1_head_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(prc1_head_agg.topology() == stk::topology::NODE, "Expected a node topology.");

    auto ngp_prc1_head_agg = mesh::get_updated_ngp_aggregate(prc1_head_agg);
    ngp_prc1_head_agg.template sync_to_device<FORCE, VELOCITY, RADIUS>();
    ngp_prc1_head_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_prc1_head_agg.template modify_on_device<VELOCITY>();
  }

  void apply_to(auto& prc1_head_agg) {
    static_assert(prc1_head_agg.topology() == stk::topology::NODE, "Expected a node topology.");

    auto ngp_prc1_head_agg = mesh::get_updated_ngp_aggregate(prc1_head_agg);
    ngp_prc1_head_agg.template sync_to_device<FORCE, VELOCITY, RADIUS>();
    ngp_prc1_head_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_prc1_head_agg.template modify_on_device<VELOCITY>();
  }
};

class update_configuration_center {
 public:
  update_configuration_center(double dt) : dt_(dt) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& node_view) const {
    // First order Euler
    auto velocity = get<VELOCITY>(node_view);
    auto position = get<COORDS>(node_view);

    // Update position
    position += dt_ * velocity;
    std::cout << "position: " << position << std::endl;
    std::cout << "velocity: " << velocity << std::endl;
    std::cout << "dt: " << dt_ << std::endl;
  }

  void apply_to(auto& rod_node_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_node_agg.rank() == stk::topology::NODE_RANK, "Expected the rod centers.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_node_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, COORDS>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<COORDS>();
  }

  void apply_to(auto& rod_node_agg) {
    static_assert(rod_node_agg.rank() == stk::topology::NODE_RANK, "Expected the rod centers.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_node_agg);
    ngp_rod_agg.template sync_to_device<VELOCITY, COORDS>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<COORDS>();
  }

  double dt_;
};

class update_configuration_orientation {
 public:
  update_configuration_orientation(double dt) : dt_(dt) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& node_view) const {
    // First order Euler
    auto omega = get<OMEGA>(node_view);
    auto quat = get<QUAT>(node_view);
    auto tangent = get<TANGENT>(node_view);

    // Update position
    std::cout << "quat 0: " << quat << std::endl;
    math::rotate_quaternion(quat, omega, dt_);
    std::cout << "quat 1: " << quat << std::endl;
    std::cout << "omega: " << omega << std::endl;
    tangent = quat * math::Vector3<double>(0.0, 0.0, 1.0);
  }

  void apply_to(auto& rod_node_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_node_agg.rank() == stk::topology::NODE_RANK, "Expected the rod centers.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_node_agg);
    ngp_rod_agg.template sync_to_device<OMEGA, QUAT, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_rod_agg.template modify_on_device<QUAT, TANGENT>();
  }

  void apply_to(auto& rod_node_agg) {
    static_assert(rod_node_agg.rank() == stk::topology::NODE_RANK, "Expected the rod centers.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_node_agg);
    ngp_rod_agg.template sync_to_device<OMEGA, QUAT, TANGENT>();
    ngp_rod_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_rod_agg.template modify_on_device<QUAT, TANGENT>();
  }

  double dt_;
};

template <typename RodAgg, typename SurfaceNodeAgg>
class reconcile_surface_nodes {
 public:
  reconcile_surface_nodes(mesh::LinkData& link_data, RodAgg& rod_agg, SurfaceNodeAgg& surface_node_agg)
      : link_data_(link_data),
        rod_agg_(rod_agg),
        surface_node_agg_(surface_node_agg),
        ngp_rod_agg_{},
        ngp_surface_node_agg_{} {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& slink_view) const {
    // Take in the surface link view and update the surface node positions/tangents using their arclength
    // and the center/tangent of the rod. Arclength is measured from the left endpoint of the rod

    // Get the two entities from the slink_view, this is already inside of a parallel loop
    auto rod_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[0]);
    auto surface_node_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[1]);

    // Go from entity view to the aggregate
    auto rod_view = ngp_rod_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(rod_entity));
    auto surface_node_view = ngp_surface_node_agg_.get_view(slink_view.ngp_mesh().fast_mesh_index(surface_node_entity));
    MUNDY_THROW_ASSERT(get<IS_BOUND>(surface_node_view)[0] == 1, std::runtime_error,
                       "Weird: A surface node is not bound to a rod!");

    // Node data:
    auto rod_center = get<COORDS>(rod_view, 0);
    auto rod_tangent = get<TANGENT>(rod_view, 0);

    // Rod data:
    double rod_length = get<LENGTH>(rod_view)[0];

    // Surface node data:
    auto surface_node_pos = get<COORDS>(surface_node_view);
    auto surface_node_tangent = get<TANGENT>(surface_node_view);
    double surface_node_arclength = get<ARCLENGTH>(surface_node_view)[0];

    // Update the position of the surface node
    surface_node_pos = rod_center + (surface_node_arclength - 0.5 * rod_length) * rod_tangent;
    surface_node_tangent = rod_tangent;
  }

  void apply_to(auto& slink_agg, const stk::mesh::Selector& subset_selector) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<COORDS, TANGENT, LENGTH>();
    ngp_surface_node_agg_.template sync_to_device<COORDS, TANGENT, ARCLENGTH>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);

    ngp_surface_node_agg_.template modify_on_device<COORDS, TANGENT>();
  }

  void apply_to(auto& slink_agg) {
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);
    ngp_rod_agg_ = mesh::get_updated_ngp_aggregate(rod_agg_);
    ngp_surface_node_agg_ = mesh::get_updated_ngp_aggregate(surface_node_agg_);

    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_rod_agg_.template sync_to_device<COORDS, TANGENT, LENGTH>();
    ngp_surface_node_agg_.template sync_to_device<COORDS, TANGENT, ARCLENGTH>();

    ngp_slink_agg.template for_each((*this) /*use my operator as a lambda*/);

    ngp_surface_node_agg_.template modify_on_device<COORDS, TANGENT>();
  }

 private:
  mesh::LinkData& link_data_;
  RodAgg& rod_agg_;
  SurfaceNodeAgg& surface_node_agg_;
  using rod_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<RodAgg>()));
  using surface_node_agg_ngp_t = decltype(mesh::get_updated_ngp_aggregate(std::declval<SurfaceNodeAgg>()));
  rod_agg_ngp_t ngp_rod_agg_;
  surface_node_agg_ngp_t ngp_surface_node_agg_;
};

//! \name Temporary N^2 binding/unbinding
//@{

struct HookeanCrosslinkerEnergy {
  KOKKOS_INLINE_FUNCTION
  double operator()(const auto& crosslinker_view, const auto& rod_view, const double arclength,
                    unsigned side /* 0 for left, 1 for right*/) const {
    // Arclength goes from 0 to length of the rod and is measured from the left endpoint of the rod

    // Get the crosslinker data
    const double spring_constant = get<SPRING_CONSTANT>(crosslinker_view)[0];
    const double rest_length = get<REST_LENGTH>(crosslinker_view)[0];

    // Get the rod data
    const auto rod_coords = get<COORDS>(rod_view, 0);
    const auto rod_tangent = get<TANGENT>(rod_view, 0);
    const double rod_length = get<LENGTH>(rod_view)[0];

    // Compute the distance between the rod and the crosslinker
    const auto other_spring_coords = rod_coords + (arclength - 0.5 * rod_length) * rod_tangent;
    const auto to_bind_spring_coords = get<COORDS>(crosslinker_view, side);
    const double spring_length = math::norm(to_bind_spring_coords - other_spring_coords);

    // Compute the energy
    return 0.5 * spring_constant * (spring_length - rest_length) * (spring_length - rest_length);
  }

  KOKKOS_INLINE_FUNCTION
  double operator()(const auto& crosslinker_view) const {
    // Arclength goes from 0 to length of the rod and is measured from the left endpoint of the rod

    // Get the crosslinker data
    const double spring_constant = get<SPRING_CONSTANT>(crosslinker_view)[0];
    const double rest_length = get<REST_LENGTH>(crosslinker_view)[0];

    // Compute the distance between the rod and the crosslinker
    const auto left_spring_coords = get<COORDS>(crosslinker_view, 0);
    const auto right_spring_coords = get<COORDS>(crosslinker_view, 1);
    const double spring_length = math::norm(right_spring_coords - left_spring_coords);

    // Compute the energy
    return 0.5 * spring_constant * (spring_length - rest_length) * (spring_length - rest_length);
  }
};

struct AngularCrosslinkerEnergy {
  double screening_length;

  KOKKOS_INLINE_FUNCTION
  double operator()(const auto& crosslinker_view, const auto& rod_view, const double arclength,
                    const unsigned side /* 0 for left, 1 for right*/) const {
    // Arclength goes from 0 to length of the rod and is measured from the left endpoint of the rod

    // Get the crosslinker data
    double k_ang = get<ANG_SPRING_CONSTANT>(crosslinker_view)[0];
    double cos_theta0 = get<REST_COS_ANGLE>(crosslinker_view)[0];
    const auto opposite_tangent = get<TANGENT>(crosslinker_view, !side);

    // Get the rod data
    const auto rod_coords = get<COORDS>(rod_view, 0);
    const auto rod_tangent = get<TANGENT>(rod_view, 0);
    const double rod_length = get<LENGTH>(rod_view)[0];

    const auto opposite_coords = get<COORDS>(crosslinker_view, !side);
    const auto pre_binding_coord = get<COORDS>(crosslinker_view, side);
    const auto post_binding_coord = rod_coords + (arclength - 0.5 * rod_length) * rod_tangent;

    // Energy is infinite if the distance moved is greater than the screening length
    const double distance_moved = math::norm(pre_binding_coord - post_binding_coord);
    if (distance_moved > screening_length) {
      return Kokkos::Experimental::finite_max_v<double> / 100.;
    }

    const auto new_spring_r = post_binding_coord - opposite_coords;
    const double new_spring_length = math::norm(new_spring_r);
    const bool spring_is_zero_length = new_spring_length < math::get_zero_tolerance<double>();
    const auto new_spring_tangent =
        spring_is_zero_length ? math::Vector3<double>(0., 0., 0.) : new_spring_r / new_spring_length;

    const double cos_theta_other = math::dot(opposite_tangent, new_spring_tangent);
    const double cos_theta_to_bind = math::dot(rod_tangent, new_spring_tangent);

    const double other_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_other - cos_theta0) * (cos_theta_other - cos_theta0);
    const double to_bind_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_to_bind - cos_theta0) * (cos_theta_to_bind - cos_theta0);

    return get<IS_BOUND>(crosslinker_view, !side)[0] * other_ang_spring_energy +
           get<IS_BOUND>(crosslinker_view, side)[0] * to_bind_ang_spring_energy;
  }

  KOKKOS_INLINE_FUNCTION
  double operator()(const auto& crosslinker_view) const {
    // Arclength goes from 0 to length of the rod and is measured from the left endpoint of the rod

    // Get the crosslinker data
    double k_ang = get<ANG_SPRING_CONSTANT>(crosslinker_view)[0];
    double cos_theta0 = get<REST_COS_ANGLE>(crosslinker_view)[0];
    const auto left_tangent = get<TANGENT>(crosslinker_view, 0);
    const auto right_tangent = get<TANGENT>(crosslinker_view, 1);

    const auto left_spring_coords = get<COORDS>(crosslinker_view, 0);
    const auto right_spring_coords = get<COORDS>(crosslinker_view, 1);

    const auto spring_r = right_spring_coords - left_spring_coords;
    const double spring_length = math::norm(spring_r);
    const bool spring_is_zero_length = spring_length < math::get_zero_tolerance<double>();
    const auto spring_tangent = spring_is_zero_length ? math::Vector3<double>(0., 0., 0.) : spring_r / spring_length;

    const double cos_theta_left = math::dot(left_tangent, spring_tangent);
    const double cos_theta_right = math::dot(right_tangent, spring_tangent);

    const double left_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_left - cos_theta0) * (cos_theta_left - cos_theta0);
    const double right_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_right - cos_theta0) * (cos_theta_right - cos_theta0);

    return get<IS_BOUND>(crosslinker_view, 0)[0] * left_ang_spring_energy +
           get<IS_BOUND>(crosslinker_view, 1)[0] * right_ang_spring_energy;
  }
};

struct HookeanPlusAngularCrosslinkerEnergy {
  double screening_length;

  KOKKOS_INLINE_FUNCTION
  double operator()(const auto& crosslinker_view, const auto& rod_view, const double arclength,
                    const unsigned side /* 0 for left, 1 for right*/) const {
    // Arclength goes from 0 to length of the rod and is measured from the left endpoint of the rod

    // Get the crosslinker data
    const double k_lin = get<SPRING_CONSTANT>(crosslinker_view)[0];
    const double l0 = get<REST_LENGTH>(crosslinker_view)[0];
    double k_ang = get<ANG_SPRING_CONSTANT>(crosslinker_view)[0];
    double cos_theta0 = get<REST_COS_ANGLE>(crosslinker_view)[0];
    const auto opposite_tangent = get<TANGENT>(crosslinker_view, !side);

    // Get the rod data
    const auto rod_coords = get<COORDS>(rod_view, 0);
    const auto rod_tangent = get<TANGENT>(rod_view, 0);
    const double rod_length = get<LENGTH>(rod_view)[0];

    const auto opposite_coords = get<COORDS>(crosslinker_view, !side);
    const auto pre_binding_coord = get<COORDS>(crosslinker_view, side);
    const auto post_binding_coord = rod_coords + (arclength - 0.5 * rod_length) * rod_tangent;

    // Energy is infinite if the distance moved is greater than the screening length
    const double distance_moved = math::norm(pre_binding_coord - post_binding_coord);
    if (distance_moved > screening_length) {
      return Kokkos::Experimental::finite_max_v<double> / 100.;
    }

    const auto new_spring_r = post_binding_coord - opposite_coords;
    const double new_spring_length = math::norm(new_spring_r);
    const bool spring_is_zero_length = new_spring_length < math::get_zero_tolerance<double>();
    const auto new_spring_tangent =
        spring_is_zero_length ? math::Vector3<double>(0., 0., 0.) : new_spring_r / new_spring_length;

    const double cos_theta_other = math::dot(opposite_tangent, new_spring_tangent);
    const double cos_theta_to_bind = math::dot(rod_tangent, new_spring_tangent);

    const double linear_spring_energy = 0.5 * k_lin * (new_spring_length - l0) * (new_spring_length - l0);
    const double other_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_other - cos_theta0) * (cos_theta_other - cos_theta0);
    const double to_bind_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_to_bind - cos_theta0) * (cos_theta_to_bind - cos_theta0);

    return linear_spring_energy + other_ang_spring_energy + to_bind_ang_spring_energy;
  }

  KOKKOS_INLINE_FUNCTION
  double operator()(const auto& crosslinker_view) const {
    // Arclength goes from 0 to length of the rod and is measured from the left endpoint of the rod

    // Get the crosslinker data
    const double k_lin = get<SPRING_CONSTANT>(crosslinker_view)[0];
    const double l0 = get<REST_LENGTH>(crosslinker_view)[0];
    double k_ang = get<ANG_SPRING_CONSTANT>(crosslinker_view)[0];
    double cos_theta0 = get<REST_COS_ANGLE>(crosslinker_view)[0];
    const auto left_tangent = get<TANGENT>(crosslinker_view, 0);
    const auto right_tangent = get<TANGENT>(crosslinker_view, 1);

    const auto left_spring_coords = get<COORDS>(crosslinker_view, 0);
    const auto right_spring_coords = get<COORDS>(crosslinker_view, 1);

    const auto spring_r = right_spring_coords - left_spring_coords;
    const double spring_length = math::norm(spring_r);
    const bool spring_is_zero_length = spring_length < math::get_zero_tolerance<double>();
    const auto spring_tangent = spring_is_zero_length ? math::Vector3<double>(0., 0., 0.) : spring_r / spring_length;

    const double cos_theta_left = math::dot(left_tangent, spring_tangent);
    const double cos_theta_right = math::dot(right_tangent, spring_tangent);

    const double linear_spring_energy = 0.5 * k_lin * (spring_length - l0) * (spring_length - l0);
    const double left_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_left - cos_theta0) * (cos_theta_left - cos_theta0);
    const double right_ang_spring_energy =
        spring_is_zero_length * 0.5 * k_ang * (cos_theta_right - cos_theta0) * (cos_theta_right - cos_theta0);

    return linear_spring_energy + left_ang_spring_energy + right_ang_spring_energy;
  }
};

class kmc_state_change_crosslinks {
 public:
  using indices_view_t = Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace>;
  using IS_BOUND_TMP = SCRATCH_D1;

  kmc_state_change_crosslinks(const double dt, const double kbt, mesh::LinkData& link_data, stk::mesh::Part& slink_part)
      : dt_(dt), kbt_(kbt), link_data_(link_data), slink_part_(slink_part) {
  }

  void bound_to_unbound(auto& ngp_crosslinker_agg, auto& ngp_crosslinker_head_agg, auto& ngp_rod_agg,
                        auto& ngp_slink_agg, const indices_view_t& crosslinker_indices,
                        const indices_view_t& rod_indices, mesh::LinkPartition& link_partition) {
    auto ngp_mesh = ngp_crosslinker_agg.ngp_mesh();
    const size_t num_crosslinkers = crosslinker_indices.extent(0);
    const size_t num_rods = rod_indices.extent(0);
    using range_policy_t = stk::ngp::DeviceRangePolicy;
    Kokkos::parallel_for(
        "apply_kmc_bound_to_unbound", range_policy_t(0, num_crosslinkers), KOKKOS_LAMBDA(const size_t n) {
          auto crosslinker_fast_mesh_index = crosslinker_indices(n);
          auto crosslinker_view = ngp_crosslinker_agg.get_view(crosslinker_fast_mesh_index);

          // Loop over left and right nodes separately
          for (unsigned side = 0; side < 2; ++side) {
            // Step 0: Check if we are already unbound
            if (!get<IS_BOUND>(crosslinker_view, side)[0]) {
              continue;  // Already unbound, skip to next side
            }

            // Step 1: Get Z-total
            double unbinding_rate = get<UNBINDING_RATE>(crosslinker_view, side)[0];
            double z_total = unbinding_rate * dt_;

            // Step 2: Determine if we unbind or not
            stk::mesh::EntityId crosslinker_id = crosslinker_view.entity_id();
            auto crosslinker_counter = get<RNG_COUNTER>(crosslinker_view);
            openrand::Philox rng(crosslinker_id, crosslinker_counter[0]);
            double rand_u01 = rng.rand<double>();
            crosslinker_counter[0]++;

            double probability_of_unbinding = 1.0 - Kokkos::exp(-z_total);
            if (rand_u01 < probability_of_unbinding) {
              get<ARCLENGTH>(crosslinker_view, side) = -1;
              get<IS_BOUND_TMP>(crosslinker_view, side) = false;
            }
          }
        });

    // Loop over each link and request its destruction if it links a surface node that is unbound.
    ngp_slink_agg.template sync_to_device<LINKED_ENTITIES>();
    ngp_slink_agg.template for_each(KOKKOS_LAMBDA(auto& slink_view) {
      stk::mesh::Entity surface_node_entity = stk::mesh::Entity(get<LINKED_ENTITIES>(slink_view)[1]);
      auto surface_node_fast_mesh_index = ngp_mesh.fast_mesh_index(surface_node_entity);
      auto surface_node_view = ngp_crosslinker_head_agg.get_view(surface_node_fast_mesh_index);

      // Loop over left and right nodes separately
      if (!get<IS_BOUND_TMP>(surface_node_view)[0]) {
        link_partition.request_destruction(
            slink_view.ngp_mesh().get_entity(stk::topology::NODE_RANK, slink_view.entity_index()));
      }
    });
  }

  template <typename EnergyFunc>
  void unbound_to_bound(auto& ngp_crosslinker_agg, auto& ngp_rod_agg, const indices_view_t& crosslinker_indices,
                        const indices_view_t& rod_indices, mesh::LinkPartition& link_partition,
                        const EnergyFunc& energy_func) {
    auto ngp_mesh = ngp_crosslinker_agg.ngp_mesh();

    const size_t num_crosslinkers = crosslinker_indices.extent(0);
    const size_t num_rods = rod_indices.extent(0);
    using range_policy_t = stk::ngp::DeviceRangePolicy;
    Kokkos::parallel_for(
        "apply_kmc_unbound_to_bound", range_policy_t(0, num_crosslinkers), KOKKOS_LAMBDA(const size_t n) {
          auto crosslinker_fast_mesh_index = crosslinker_indices(n);
          auto crosslinker_view = ngp_crosslinker_agg.get_view(crosslinker_fast_mesh_index);

          // Loop over left and right node separately
          for (unsigned side = 0; side < 2; ++side) {
            // Step 0: Check if we are already bound
            if (get<IS_BOUND>(crosslinker_view, side)[0]) {
              continue;  // Already bound, skip to next side
            }

            // Step 1: Get Z-total
            double binding_rate = get<BINDING_RATE>(crosslinker_view, side)[0];
            double z_total = 0.0;
            double old_energy = energy_func(crosslinker_view);

            for (size_t r = 0; r < num_rods; ++r) {
              auto rod_fast_mesh_index = rod_indices(r);
              auto rod_view = ngp_rod_agg.get_view(rod_fast_mesh_index);
              const double rod_length = get<LENGTH>(rod_view)[0];
              const double rod_bs_spacing = get<BIND_SITE_SPACING>(rod_view)[0];

              unsigned num_bind_sites = static_cast<unsigned>(rod_length / rod_bs_spacing) + 1;
              for (unsigned b = 0; b < num_bind_sites; b++) {
                double arclength = b * rod_bs_spacing;
                double new_energy = energy_func(crosslinker_view, rod_view, arclength, side);

                // Compute the Boltzmann weight using the change in energy
                double delta_energy = new_energy - old_energy;
                double z_i = dt_ * binding_rate * Kokkos::exp(-delta_energy / kbt_);
                z_total += z_i;
              }
            }

            // Step 2: Determine if we bind or not
            stk::mesh::EntityId crosslinker_id = crosslinker_view.entity_id();
            auto crosslinker_counter = get<RNG_COUNTER>(crosslinker_view);
            openrand::Philox rng(crosslinker_id, crosslinker_counter[0]);
            double rand_u01 = rng.rand<double>();
            crosslinker_counter[0]++;

            double probability_of_binding = 1.0 - Kokkos::exp(-z_total);
            if (rand_u01 < probability_of_binding) {
              // To avoid bias, grab a new random number
              double rand_u01 = rng.rand<double>();

              // Determine who we bind to
              double cumulative_sum = 0.0;
              for (size_t r = 0; r < num_rods; ++r) {
                auto rod_fast_mesh_index = rod_indices(r);
                auto rod_view = ngp_rod_agg.get_view(rod_fast_mesh_index);
                const double rod_length = get<LENGTH>(rod_view)[0];
                const double rod_bs_spacing = get<BIND_SITE_SPACING>(rod_view)[0];

                unsigned num_bind_sites = static_cast<unsigned>(rod_length / rod_bs_spacing) + 1;
                for (unsigned b = 0; b < num_bind_sites; b++) {
                  double arclength = b * rod_bs_spacing;
                  double new_energy = energy_func(crosslinker_view, rod_view, arclength, side);

                  // Compute the Boltzmann weight using the change in energy
                  double delta_energy = new_energy - old_energy;
                  double z_i = dt_ * binding_rate * Kokkos::exp(-delta_energy / kbt_);
                  cumulative_sum += z_i / z_total;

                  if (rand_u01 < cumulative_sum) {
                    // Bind to this rod at this bind site
                    get<ARCLENGTH>(crosslinker_view, side) = arclength;
                    get<IS_BOUND_TMP>(crosslinker_view, side) = true;
                    link_partition.request_link(
                        ngp_mesh.get_entity(stk::topology::ELEM_RANK, rod_fast_mesh_index),
                        ngp_mesh.get_nodes(stk::topology::ELEM_RANK, crosslinker_fast_mesh_index)[side]);
                    goto escape_post_binding;  // Break out of the for loop if we bind
                  }
                }
              }
            }
          escape_post_binding: {}
          }
        });
  }

  template <typename EnergyFunc>
  void apply_to(auto& crosslinker_agg, auto& crosslinker_head_agg, auto& rod_agg, auto& slink_agg,
                const EnergyFunc& energy_func) {
    auto ngp_crosslinker_agg = mesh::get_updated_ngp_aggregate(crosslinker_agg);
    auto ngp_crosslinker_head_agg = mesh::get_updated_ngp_aggregate(crosslinker_head_agg);
    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    auto ngp_slink_agg = mesh::get_updated_ngp_aggregate(slink_agg);

    using indices_view_t = Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace>;
    indices_view_t crosslinker_indices = get_local_entity_indices(
        ngp_crosslinker_agg.bulk_data(), ngp_crosslinker_agg.rank(), ngp_crosslinker_agg.selector());
    indices_view_t rod_indices =
        get_local_entity_indices(ngp_crosslinker_agg.bulk_data(), ngp_rod_agg.rank(), ngp_rod_agg.selector());

    // Get the entities we want to act on
    const size_t num_crosslinkers = crosslinker_indices.extent(0);
    const size_t num_rods = rod_indices.extent(0);

    stk::mesh::PartVector locally_owned_link_parts = {&slink_part_,
                                                      &ngp_crosslinker_agg.mesh_meta_data().locally_owned_part()};
    auto& link_partition = link_data_.get_partition(link_data_.get_partition_key(locally_owned_link_parts));
    link_partition.increase_request_link_capacity(
        2 * num_crosslinkers);  // Conservative estimate. Should never need resized between timesteps

    // Copy the IS_BOUND field to the SCRATCH_D1 field
    Kokkos::parallel_for(
        "apply_kmc_copy_bound", Kokkos::RangePolicy<stk::ngp::ExecSpace>(0, num_crosslinkers),
        KOKKOS_LAMBDA(const size_t n) {
          auto crosslinker_fast_mesh_index = crosslinker_indices(n);
          auto crosslinker_view = ngp_crosslinker_agg.get_view(crosslinker_fast_mesh_index);
          get<IS_BOUND_TMP>(crosslinker_view, 0) = get<IS_BOUND>(crosslinker_view, 0)[0];
          get<IS_BOUND_TMP>(crosslinker_view, 1) = get<IS_BOUND>(crosslinker_view, 1)[0];
        });

    // Perform the binding and unbinding (sets the IS_BOUND_TMP to the new value of IS_BOUND)
    unbound_to_bound(ngp_crosslinker_agg, ngp_rod_agg, crosslinker_indices, rod_indices, link_partition, energy_func);
    bound_to_unbound(ngp_crosslinker_agg, ngp_crosslinker_head_agg, ngp_rod_agg, ngp_slink_agg, crosslinker_indices,
                     rod_indices, link_partition);

    // Copy IS_BOUND_TMP to IS_BOUND
    Kokkos::parallel_for(
        "apply_kmc_copy_bound", Kokkos::RangePolicy<stk::ngp::ExecSpace>(0, num_crosslinkers),
        KOKKOS_LAMBDA(const size_t n) {
          auto crosslinker_fast_mesh_index = crosslinker_indices(n);
          auto crosslinker_view = ngp_crosslinker_agg.get_view(crosslinker_fast_mesh_index);
          get<IS_BOUND>(crosslinker_view, 0) = get<IS_BOUND_TMP>(crosslinker_view, 0)[0];
          get<IS_BOUND>(crosslinker_view, 1) = get<IS_BOUND_TMP>(crosslinker_view, 1)[0];
        });

    link_data_.process_requests();  // Process the link requests to create links between the freshly bound crosslinkers
                                    // and rods
  }

 private:
  double dt_;
  double kbt_;
  mesh::LinkData& link_data_;
  stk::mesh::Part& slink_part_;
};
//@}

//! \name Temporary N^2 contact
//@{

class apply_hertzian_contact {
 public:
  apply_hertzian_contact(const double youngs_modulus, const double poisson_ratio)
      : youngs_modulus_(youngs_modulus), poisson_ratio_(poisson_ratio) {
  }

  void apply_to(auto& rod_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected the rods to be particles.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<COORDS, TANGENT, RADIUS, LENGTH, FORCE, TORQUE>();

    auto ngp_rod_entities = get_local_entity_indices(rod_agg.bulk_data(), rod_agg.rank(), subset_selector);
    impl_run(ngp_rod_entities, ngp_rod_agg);

    ngp_rod_agg.template modify_on_device<FORCE, TORQUE>();
  }

  void apply_to(auto& rod_agg) {
    static_assert(rod_agg.topology() == stk::topology::PARTICLE, "Expected the rods to be particles.");

    auto ngp_rod_agg = mesh::get_updated_ngp_aggregate(rod_agg);
    ngp_rod_agg.template sync_to_device<COORDS, TANGENT, RADIUS, LENGTH, FORCE, TORQUE>();

    auto ngp_rod_entities = get_local_entity_indices(rod_agg.bulk_data(), rod_agg.rank(), rod_agg.selector());
    impl_run(ngp_rod_entities, ngp_rod_agg);

    ngp_rod_agg.template modify_on_device<FORCE, TORQUE>();
  }

  void impl_run(Kokkos::View<stk::mesh::FastMeshIndex*, stk::ngp::ExecSpace>& ngp_rod_entities, auto ngp_rod_agg) {
    // Get the entities we want to act on
    const size_t num_rods = ngp_rod_entities.extent(0);
    const double effective_youngs_modulus =
        (youngs_modulus_ * youngs_modulus_) / (youngs_modulus_ - youngs_modulus_ * poisson_ratio_ * poisson_ratio_ +
                                               youngs_modulus_ - youngs_modulus_ * poisson_ratio_ * poisson_ratio_);
    Kokkos::parallel_for(
        "apply_hertzian_contact", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_rods, num_rods}),
        KOKKOS_LAMBDA(const size_t t, const size_t s) {
          if (s >= t) {
            return;  // skip self interaction and duplication
          }
          auto rod1_fast_mesh_index = ngp_rod_entities(t);
          auto rod2_fast_mesh_index = ngp_rod_entities(s);

          auto rod1_view = ngp_rod_agg.get_view(rod1_fast_mesh_index);
          auto rod2_view = ngp_rod_agg.get_view(rod2_fast_mesh_index);

          // The signed separation distance is the distance between the centerline of the two spherocylinders minus the
          // sum of their radii
          auto rod1_coords = get<COORDS>(rod1_view, 0 /* node data*/);
          auto rod2_coords = get<COORDS>(rod2_view, 0);
          auto rod1_tangent = get<TANGENT>(rod1_view, 0);
          auto rod2_tangent = get<TANGENT>(rod2_view, 0);
          double rod1_radius = get<RADIUS>(rod1_view)[0];
          double rod2_radius = get<RADIUS>(rod2_view)[0];
          double rod1_length = get<LENGTH>(rod1_view)[0];
          double rod2_length = get<LENGTH>(rod2_view)[0];
          geom::LineSegment<double> rod1_centerline{rod1_coords - 0.5 * rod1_length * rod1_tangent,
                                                    rod1_coords + 0.5 * rod1_length * rod1_tangent};
          geom::LineSegment<double> rod2_centerline{rod2_coords - 0.5 * rod2_length * rod2_tangent,
                                                    rod2_coords + 0.5 * rod2_length * rod2_tangent};

          geom::Point<double> rod1_centerline_contact_point, rod2_centerline_contact_point;
          double rod1_contact_point_arc_length, rod2_contact_point_arc_length;
          math::Vector3<double> rod1_to_rod2_centerline_sep;
          const double signed_sep_dist =
              geom::distance(rod1_centerline, rod2_centerline,                              //
                             rod1_centerline_contact_point, rod2_centerline_contact_point,  //
                             rod1_contact_point_arc_length, rod2_contact_point_arc_length,  //
                             rod1_to_rod2_centerline_sep) -
              rod1_radius - rod2_radius;

          if (signed_sep_dist < 0.0) {
            const double effective_radius = (rod1_radius * rod2_radius) / (rod1_radius + rod2_radius);
            constexpr double four_thirds = 4.0 / 3.0;
            const double force_mag = four_thirds * effective_youngs_modulus * Kokkos::sqrt(effective_radius) *
                                     Kokkos::pow(-signed_sep_dist, 1.5);

            // Mind the signs. rod1 to rod2 is normal to the rod1 and contact forces act along the negative normal
            auto rod1_force = get<FORCE>(rod1_view, 0);
            auto rod2_force = get<FORCE>(rod2_view, 0);
            auto rod1_torque = get<TORQUE>(rod1_view, 0);
            auto rod2_torque = get<TORQUE>(rod2_view, 0);

            auto rod1_to_rod2_normal = rod1_to_rod2_centerline_sep / math::norm(rod1_to_rod2_centerline_sep);

            auto local_rod1_force = -force_mag * rod1_to_rod2_normal;
            auto local_rod2_force = force_mag * rod1_to_rod2_normal;
            auto local_rod1_torque = math::cross(rod1_centerline_contact_point - rod1_coords,  //
                                                 -force_mag * rod1_to_rod2_normal);
            auto local_rod2_torque = math::cross(rod2_centerline_contact_point - rod2_coords,  //
                                                 force_mag * rod1_to_rod2_normal);
            Kokkos::atomic_add(&rod1_force[0], local_rod1_force[0]);
            Kokkos::atomic_add(&rod1_force[1], local_rod1_force[1]);
            Kokkos::atomic_add(&rod1_force[2], local_rod1_force[2]);
            Kokkos::atomic_add(&rod2_force[0], local_rod2_force[0]);
            Kokkos::atomic_add(&rod2_force[1], local_rod2_force[1]);
            Kokkos::atomic_add(&rod2_force[2], local_rod2_force[2]);

            Kokkos::atomic_add(&rod1_torque[0], local_rod1_torque[0]);
            Kokkos::atomic_add(&rod1_torque[1], local_rod1_torque[1]);
            Kokkos::atomic_add(&rod1_torque[2], local_rod1_torque[2]);
            Kokkos::atomic_add(&rod2_torque[0], local_rod2_torque[0]);
            Kokkos::atomic_add(&rod2_torque[1], local_rod2_torque[1]);
            Kokkos::atomic_add(&rod2_torque[2], local_rod2_torque[2]);
          }
        });
  }

 private:
  double youngs_modulus_;
  double poisson_ratio_;
};

void run_main() {
  // STK usings
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::Selector;
  using stk::topology::ELEM_RANK;
  using stk::topology::NODE_RANK;

  // Mundy things
  using mesh::BulkData;
  using mesh::DeclareEntitiesHelper;
  using mesh::FieldComponent;
  using mesh::LinkData;
  using mesh::LinkMetaData;
  using mesh::MeshBuilder;
  using mesh::MetaData;
  using mesh::QuaternionFieldComponent;
  using mesh::ScalarFieldComponent;
  using mesh::Vector3FieldComponent;

  // Setup the STK mesh (boiler plate)
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder
      .set_spatial_dimension(3)  //
      .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition
                                       // to all fields are simple.
  meta_data_ptr->set_coordinate_field_name("COORDS");
  std::shared_ptr<BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  MetaData& meta_data = *meta_data_ptr;
  BulkData& bulk_data = *bulk_data_ptr;

  // Setup the link data (boilerplate)
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", NODE_RANK);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);

  // Declare parts
  //   Rod parts: PARTICLE_TOPOLOGY
  //   Spring part: BEAM_2
  //   Link part: SURFACE_LINKS
  Part& rod_part = meta_data.declare_part_with_topology("RODS", stk::topology::PARTICLE);
  Part& prc1_part = meta_data.declare_part_with_topology("PRC1", stk::topology::BEAM_2);
  Part& slink_part = link_meta_data.declare_link_part("SURFACE_LINKS", 2 /* our dimensionality */);

  // Declare all fields
  // Microtubules PARTICLE top
  //  Node fields: (Only what is logical to potentially share with other entities that connect to the node)
  //   - COORDS
  //   - QUATERNION
  //   - TANGENT
  //   - VELOCITY
  //   - OMEGA
  //   - FORCE
  //   - TORQUE
  //
  //  Elem fields:
  //   - LENGTH
  //   - RADIUS
  //   - BIND_SITE_SPACING
  //
  // PRC1: BEAM_2 top
  //  Node fields:
  //  - ARCLENGTH
  //  - RADIUS
  //  - COORDS
  //  - TANGENT
  //  - FORCE
  //  - TORQUE
  //  - IS_BOUND
  //
  //  Elem fields:
  //  - REST_LENGTH
  //  - SPRING_CONSTANT
  //  - ANG_SPRING_CONSTANT
  //  - REST_COS_ANGLE
  //
  // Surface Links: NODE rank but no fields for now
  auto& node_coords_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
  auto& node_quaternion_field = meta_data.declare_field<double>(NODE_RANK, "QUATERNION");
  auto& node_arclength_field = meta_data.declare_field<double>(NODE_RANK, "ARCLENGTH");
  auto& node_tangent_field = meta_data.declare_field<double>(NODE_RANK, "TANGENT");
  auto& node_velocity_field = meta_data.declare_field<double>(NODE_RANK, "VELOCITY");
  auto& node_omega_field = meta_data.declare_field<double>(NODE_RANK, "OMEGA");
  auto& node_force_field = meta_data.declare_field<double>(NODE_RANK, "FORCE");
  auto& node_torque_field = meta_data.declare_field<double>(NODE_RANK, "TORQUE");
  auto& node_is_bound_field = meta_data.declare_field<unsigned>(NODE_RANK, "IS_BOUND");
  auto& node_binding_rate_field = meta_data.declare_field<double>(NODE_RANK, "BINDING_RATE");
  auto& node_unbinding_rate_field = meta_data.declare_field<double>(NODE_RANK, "UNBINDING_RATE");
  auto& node_scratch_d1_field = meta_data.declare_field<double>(NODE_RANK, "SCRATCH_DOUBLE_1");
  auto& node_scratch_d3_field = meta_data.declare_field<double>(NODE_RANK, "SCRATCH_DOUBLE_3");
  auto& node_radius_field = meta_data.declare_field<double>(NODE_RANK, "RADIUS");
  auto& node_rng_counter_field = meta_data.declare_field<double>(NODE_RANK, "RNG_COUNTER");

  auto& elem_length_field = meta_data.declare_field<double>(ELEM_RANK, "LENGTH");
  auto& elem_radius_field = meta_data.declare_field<double>(ELEM_RANK, "RADIUS");
  auto& elem_rest_length_field = meta_data.declare_field<double>(ELEM_RANK, "REST_LENGTH");
  auto& elem_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "SPRING_CONSTANT");
  auto& elem_ang_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "ANG_SPRING_CONSTANT");
  auto& elem_rest_cos_angle_field = meta_data.declare_field<double>(ELEM_RANK, "REST_COS_ANGLE");
  auto& elem_torsion_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "TORSION_SPRING_CONSTANT");
  auto& elem_rest_cos_torsion_angle_field = meta_data.declare_field<double>(ELEM_RANK, "REST_COS_TORSION_ANGLE");
  auto& elem_bind_site_spacing_field = meta_data.declare_field<double>(ELEM_RANK, "BIND_SITE_SPACING");
  auto& elem_scratch_d1_field = meta_data.declare_field<double>(ELEM_RANK, "SCRATCH_DOUBLE_1");
  auto& elem_scratch_d3_field = meta_data.declare_field<double>(ELEM_RANK, "SCRATCH_DOUBLE_3");
  auto& elem_rng_counter_field = meta_data.declare_field<double>(ELEM_RANK, "RNG_COUNTER");

  // Universal fields
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_scratch_d1_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(node_scratch_d3_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_scratch_d1_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_scratch_d3_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rng_counter_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rng_counter_field, meta_data.universal_part(), 1, nullptr);

  // Assemble the microtubule parts
  stk::mesh::put_field_on_mesh(node_quaternion_field, rod_part, 4, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_omega_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_tangent_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_length_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_bind_site_spacing_field, rod_part, 1, nullptr);

  // Assemble the PRC1 parts
  stk::mesh::put_field_on_mesh(node_arclength_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_tangent_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_torque_field, prc1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_is_bound_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_binding_rate_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_unbinding_rate_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_radius_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, prc1_part, 3, nullptr);

  stk::mesh::put_field_on_mesh(elem_rest_length_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_constant_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rest_cos_angle_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_ang_spring_constant_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_torsion_spring_constant_field, prc1_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rest_cos_torsion_angle_field, prc1_part, 1, nullptr);

  // Decide IO stuff (boilerplate)
  stk::io::put_io_part_attribute(rod_part);
  stk::io::put_io_part_attribute(prc1_part);
  stk::io::put_io_part_attribute(slink_part);

  stk::io::set_field_role(node_coords_field, Ioss::Field::MESH);
  stk::io::set_field_role(node_quaternion_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_omega_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_torque_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_arclength_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_tangent_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_is_bound_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_binding_rate_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_unbinding_rate_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_rng_counter_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rng_counter_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_length_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_ang_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_cos_angle_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_bind_site_spacing_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_torsion_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rest_cos_torsion_angle_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(node_coords_field, stk::io::FieldOutputType::VECTOR_3D);
  // stk::io::set_field_output_type(node_quaternion_field, stk::io::FieldOutputType::VECTOR_4D);  // No special quat
  // format :(
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_omega_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_torque_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(elem_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_arclength_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_tangent_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_is_bound_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_binding_rate_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_unbinding_rate_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_rng_counter_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rng_counter_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_length_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_ang_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_cos_angle_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_bind_site_spacing_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_torsion_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rest_cos_torsion_angle_field, stk::io::FieldOutputType::SCALAR);

  // Commit the meta data
  meta_data.commit();

  // Create our accessors and aggregates
  auto coord_accessor = Vector3FieldComponent(node_coords_field);
  auto quaternion_accessor = QuaternionFieldComponent(node_quaternion_field);
  auto tangent_accessor = Vector3FieldComponent(node_tangent_field);
  auto arclength_accessor = ScalarFieldComponent(node_arclength_field);
  auto velocity_accessor = Vector3FieldComponent(node_velocity_field);
  auto omega_accessor = Vector3FieldComponent(node_omega_field);
  auto force_accessor = Vector3FieldComponent(node_force_field);
  auto torque_accessor = Vector3FieldComponent(node_torque_field);
  auto is_bound_accessor = ScalarFieldComponent(node_is_bound_field);
  auto binding_rate_accessor = ScalarFieldComponent(node_binding_rate_field);
  auto unbinding_rate_accessor = ScalarFieldComponent(node_unbinding_rate_field);
  auto node_radius_accessor = ScalarFieldComponent(node_radius_field);

  auto length_accessor = ScalarFieldComponent(elem_length_field);
  auto elem_radius_accessor = ScalarFieldComponent(elem_radius_field);
  auto node_rng_counter_accessor = ScalarFieldComponent(node_rng_counter_field);
  auto elem_rng_counter_accessor = ScalarFieldComponent(elem_rng_counter_field);
  auto rest_length_accessor = ScalarFieldComponent(elem_rest_length_field);
  auto spring_constant_accessor = ScalarFieldComponent(elem_spring_constant_field);
  auto ang_spring_constant_accessor = ScalarFieldComponent(elem_ang_spring_constant_field);
  auto rest_cos_angle_accessor = ScalarFieldComponent(elem_rest_cos_angle_field);
  auto linked_entities_accessor = FieldComponent(link_meta_data.linked_entities_field());
  auto bind_site_spacing_accessor = ScalarFieldComponent(elem_bind_site_spacing_field);
  auto scratch_d1_accessor = ScalarFieldComponent(node_scratch_d1_field);
  auto rest_cos_torsion_angle_accessor = ScalarFieldComponent(elem_rest_cos_torsion_angle_field);
  auto torsion_spring_constant_accessor = ScalarFieldComponent(elem_torsion_spring_constant_field);

  // Create the aggregates
  auto rod_agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, rod_part)
                     .add_component<COORDS, NODE_RANK>(coord_accessor)
                     .add_component<QUAT, NODE_RANK>(quaternion_accessor)
                     .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                     .add_component<VELOCITY, NODE_RANK>(velocity_accessor)
                     .add_component<OMEGA, NODE_RANK>(omega_accessor)
                     .add_component<FORCE, NODE_RANK>(force_accessor)
                     .add_component<TORQUE, NODE_RANK>(torque_accessor)
                     .add_component<LENGTH, ELEM_RANK>(length_accessor)
                     .add_component<RADIUS, ELEM_RANK>(elem_radius_accessor)
                     .add_component<RNG_COUNTER, ELEM_RANK>(elem_rng_counter_accessor)
                     .add_component<BIND_SITE_SPACING, ELEM_RANK>(bind_site_spacing_accessor);

  auto rod_node_agg = make_ranked_aggregate<stk::topology::NODE_RANK>(bulk_data, rod_part)
                          .add_component<COORDS, NODE_RANK>(coord_accessor)
                          .add_component<QUAT, NODE_RANK>(quaternion_accessor)
                          .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                          .add_component<VELOCITY, NODE_RANK>(velocity_accessor)
                          .add_component<OMEGA, NODE_RANK>(omega_accessor)
                          .add_component<FORCE, NODE_RANK>(force_accessor)
                          .add_component<TORQUE, NODE_RANK>(torque_accessor);

  auto prc1_agg = make_aggregate<stk::topology::BEAM_2>(bulk_data, prc1_part)
                      .add_component<ARCLENGTH, NODE_RANK>(arclength_accessor)
                      .add_component<COORDS, NODE_RANK>(coord_accessor)
                      .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                      .add_component<FORCE, NODE_RANK>(force_accessor)
                      .add_component<TORQUE, NODE_RANK>(torque_accessor)
                      .add_component<IS_BOUND, NODE_RANK>(is_bound_accessor)
                      .add_component<BINDING_RATE, NODE_RANK>(binding_rate_accessor)
                      .add_component<UNBINDING_RATE, NODE_RANK>(unbinding_rate_accessor)
                      .add_component<SCRATCH_D1, NODE_RANK>(scratch_d1_accessor)
                      .add_component<REST_LENGTH, ELEM_RANK>(rest_length_accessor)
                      .add_component<SPRING_CONSTANT, ELEM_RANK>(spring_constant_accessor)
                      .add_component<ANG_SPRING_CONSTANT, ELEM_RANK>(ang_spring_constant_accessor)
                      .add_component<REST_COS_ANGLE, ELEM_RANK>(rest_cos_angle_accessor)
                      .add_component<RNG_COUNTER, ELEM_RANK>(elem_rng_counter_accessor)
                      .add_component<TORSION_SPRING_CONSTANT, ELEM_RANK>(torsion_spring_constant_accessor)
                      .add_component<REST_COS_TORSION_ANGLE, ELEM_RANK>(rest_cos_torsion_angle_accessor);

  auto prc1_head_agg = make_aggregate<stk::topology::NODE>(bulk_data, prc1_part)
                           .add_component<ARCLENGTH, NODE_RANK>(arclength_accessor)
                           .add_component<COORDS, NODE_RANK>(coord_accessor)
                           .add_component<TANGENT, NODE_RANK>(tangent_accessor)
                           .add_component<FORCE, NODE_RANK>(force_accessor)
                           .add_component<TORQUE, NODE_RANK>(torque_accessor)
                           .add_component<VELOCITY, NODE_RANK>(velocity_accessor)
                           .add_component<RADIUS, NODE_RANK>(node_radius_accessor)
                           .add_component<IS_BOUND, NODE_RANK>(is_bound_accessor)
                           .add_component<SCRATCH_D1, NODE_RANK>(scratch_d1_accessor)
                           .add_component<BINDING_RATE, NODE_RANK>(binding_rate_accessor)
                           .add_component<UNBINDING_RATE, NODE_RANK>(unbinding_rate_accessor)
                           .add_component<RNG_COUNTER, NODE_RANK>(node_rng_counter_accessor);

  auto slink_agg = make_ranked_aggregate<NODE_RANK>(bulk_data, slink_part)
                       .add_component<LINKED_ENTITIES, NODE_RANK>(linked_entities_accessor);

  // Hard code the system params
  // Sim params
  size_t num_timesteps = 100;
  size_t io_frequency = 1;
  double dt = 0.1;

  // Brownian params
  double brownian_kbt = 0.0;

  // Collision params
  double hertz_youngs_modulus = 1.0;
  double hertz_poisson_ratio = 0.5;

  // Rod params
  double rod_radius = 0.5;
  double rod_length = 20;
  math::Quaternion<double> rod_orient1 = math::euler_to_quat(0.0, 2.5 * M_PI / 180, 0.0);
  math::Quaternion<double> rod_orient2 = math::euler_to_quat(0.0, -2.5 * M_PI / 180, 0.0);
  math::Vector3<double> rod_tangent1 = rod_orient1 * math::Vector3<double>(0.0, 0.0, 1.0);
  math::Vector3<double> rod_tangent2 = rod_orient2 * math::Vector3<double>(0.0, 0.0, 1.0);
  double init_rod_sep = 2.0;

  // Spring params
  double spring_constant = 1.0;
  double ang_spring_constant = 1.0;
  double rest_length = 1.0;
  double rest_cos_angle = 0.0;
  double rest_cos_torsion_angle = 0.9;  // parallel
  double torsion_spring_constant = 0.1;

  // Dynamic spring params
  double binding_unbinding_kbt = 1.0;
  double binding_rate = 0.0;
  double unbinding_rate = 0.0;
  double bind_site_spacing = rod_length / 100.0;
  double screening_length = rest_length / 5.0;  // Small with respect to rest_length

  DeclareEntitiesHelper dec_helper;
  dec_helper.create_node()
      .owning_proc(0)                                                 //
      .id(1)                                                          // 1 indexed
      .add_field_data<double>(&node_coords_field, {0, 0, 0.0})        //
      .add_field_data<double>(&node_quaternion_field, {1, 0, 0, 0})   //
      .add_field_data<double>(&node_tangent_field, {0, 0, 1})         //
      .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})  //
      .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})     //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})     //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0});
  dec_helper.create_node()
      .owning_proc(0)                                                 //
      .id(2)                                                          // 1 indexed
      .add_field_data<double>(&node_coords_field, {0, 1, 0.0})        //
      .add_field_data<double>(&node_quaternion_field, {1, 0, 0, 0})   //
      .add_field_data<double>(&node_tangent_field, {0, 0, 1})         //
      .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})  //
      .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})     //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})     //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0});
  dec_helper.create_node()
      .owning_proc(0)                                                     //
      .id(3)                                                              // 1 indexed
      .add_field_data<double>(&node_arclength_field, {0.5 * rod_length})  //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})         //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})        //
      .add_field_data<unsigned>(&node_is_bound_field, {1})                //
      .add_field_data<double>(&node_binding_rate_field, {binding_rate})   //
      .add_field_data<double>(&node_unbinding_rate_field, {unbinding_rate})
      .add_field_data<double>(&node_radius_field, {rod_radius});

  dec_helper.create_node()
      .owning_proc(0)  //
      .id(4)           // 1 indexed                                                             // 1 indexed
      .add_field_data<double>(&node_arclength_field, {0.5 * rod_length})       //
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})              //
      .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0})             //
      .add_field_data<unsigned>(&node_is_bound_field, {1})                     //
      .add_field_data<double>(&node_binding_rate_field, {binding_rate * (0)})  //
      .add_field_data<double>(&node_unbinding_rate_field, {unbinding_rate * (0)})
      .add_field_data<double>(&node_radius_field, {rod_radius});

  dec_helper.create_element()
      .owning_proc(0)  //
      .id(3)           //
      .topology(stk::topology::BEAM_2)
      .add_part(&prc1_part)                                                                    //
      .nodes({3, 4})                                                                           //
      .add_field_data<double>(&elem_rest_length_field, {rest_length})                          //
      .add_field_data<double>(&elem_spring_constant_field, {spring_constant})                  //
      .add_field_data<double>(&elem_ang_spring_constant_field, {ang_spring_constant})          //
      .add_field_data<double>(&elem_rest_cos_angle_field, {rest_cos_angle})                    //
      .add_field_data<double>(&elem_torsion_spring_constant_field, {torsion_spring_constant})  //
      .add_field_data<double>(&elem_rest_cos_torsion_angle_field, {rest_cos_torsion_angle})    //
      .add_field_data<size_t>(&elem_rng_counter_field, {0});


    // Rod elements
    dec_helper
        .create_element()  //
        .owning_proc(0)    //
        .id(1)      //
        .topology(stk::topology::PARTICLE)
        .add_part(&rod_part)                                                         //
        .nodes({1})                                                           //
        .add_field_data<double>(&elem_length_field, {rod_length})                    //
        .add_field_data<double>(&elem_radius_field, {rod_radius})                    //
        .add_field_data<double>(&elem_bind_site_spacing_field, {bind_site_spacing})  //
        .add_field_data<size_t>(&elem_rng_counter_field, {0});

    dec_helper
        .create_element()  //
        .owning_proc(0)    //
        .id(2)      //
        .topology(stk::topology::PARTICLE)
        .add_part(&rod_part)                                                         //
        .nodes({2})                                                           //
        .add_field_data<double>(&elem_length_field, {rod_length})                    //
        .add_field_data<double>(&elem_radius_field, {rod_radius})                    //
        .add_field_data<double>(&elem_bind_site_spacing_field, {bind_site_spacing})  //
        .add_field_data<size_t>(&elem_rng_counter_field, {0});
      
  // Declare the entities
  dec_helper.check_consistency(bulk_data);
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  stk::mesh::PartVector locally_owned_link_parts = {&slink_part, &meta_data.locally_owned_part()};
  auto& partition = link_data.get_partition(link_data.get_partition_key(locally_owned_link_parts));
  partition.increase_request_link_capacity(2);
  stk::mesh::Entity e_mt1 = bulk_data.get_entity(ELEM_RANK, 1);
  stk::mesh::Entity e_mt2 = bulk_data.get_entity(ELEM_RANK, 2);
  stk::mesh::Entity e_prc1_head1 = bulk_data.get_entity(NODE_RANK, 3);
  stk::mesh::Entity e_prc1_head2 = bulk_data.get_entity(NODE_RANK, 4);
  MUNDY_THROW_REQUIRE(bulk_data.is_valid(e_mt1) && bulk_data.is_valid(e_mt2) && bulk_data.is_valid(e_prc1_head1) &&
                          bulk_data.is_valid(e_prc1_head2),
                      std::runtime_error, "Invalid entities for link creation");
  partition.request_link(e_mt1, e_prc1_head1);
  partition.request_link(e_mt2, e_prc1_head2);
  link_data.process_requests();

  // // Fill the declare entities helper
  // DeclareEntitiesHelper dec_helper;
  // // place the MICROtubules
  // double SquareSize = 2;
  // double num_nodes = SquareSize * SquareSize;
  // double XPosition = 0;
  // double YPosition = 0;
  // for (int RodIndex = 1; RodIndex < num_nodes + 1; RodIndex++) {
  //   // Rod nodes
  //   dec_helper.create_node()
  //       .owning_proc(0)                                                           //
  //       .id(RodIndex)                                                             // 1 indexed
  //       .add_field_data<double>(&node_coords_field, {XPosition, YPosition, 0.0})  //
  //       .add_field_data<double>(&node_quaternion_field, {1, 0, 0, 0})             //
  //       .add_field_data<double>(&node_tangent_field, {0, 0, 1})                   //
  //       .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})            //
  //       .add_field_data<double>(&node_omega_field, {0.0, 0.0, 0.0})               //
  //       .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})               //
  //       .add_field_data<double>(&node_torque_field, {0.0, 0.0, 0.0});

  //   XPosition++;  // TODO(will): Hard codes the spacing as 1!!!! (bad Will Fix me)
  //   if (XPosition > (SquareSize - 1)) {
  //     // This code will be executed because x is greater than y
  //     XPosition = 0;
  //     YPosition++;
  //   }

  //   // Rod elements
  //   dec_helper
  //       .create_element()  //
  //       .owning_proc(0)    //
  //       .id(RodIndex)      //
  //       .topology(stk::topology::PARTICLE)
  //       .add_part(&rod_part)                                                         //
  //       .nodes({RodIndex})                                                           //
  //       .add_field_data<double>(&elem_length_field, {rod_length})                    //
  //       .add_field_data<double>(&elem_radius_field, {rod_radius})                    //
  //       .add_field_data<double>(&elem_bind_site_spacing_field, {bind_site_spacing})  //
  //       .add_field_data<size_t>(&elem_rng_counter_field, {0});
  // }

  // XPosition = 0;
  // YPosition = 1;
  // double PRC1HeadIndex = num_nodes + 1;
  // double MotorIndex = num_nodes + 1;
  // double MotorsPerMT = 1;
  // double TotalMotors = 2 * (SquareSize * (SquareSize - 1));
  // for (int MotorPerThisMT = 0; MotorPerThisMT < MotorsPerMT; MotorPerThisMT++) {
  //   // Make all of the motors - will get placed on MTs later on.
  //   for (int RodIndex = 1; RodIndex < TotalMotors + 1; RodIndex++) {
  //     // Declare the right facing PRC1
  //     dec_helper.create_node()
  //         .owning_proc(0)     //
  //         .id(PRC1HeadIndex)  // 1 indexed
  //         .add_field_data<double>(&node_arclength_field, {(MotorPerThisMT + 0.5) / MotorsPerMT * rod_length})  //
  //         .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0}) // .add_field_data<double>(&node_torque_field,
  //         {0.0, 0.0, 0.0})                                               //
  //         .add_field_data<unsigned>(&node_is_bound_field, {1}) // .add_field_data<double>(&node_binding_rate_field,
  //         {binding_rate})                                          //
  //         .add_field_data<double>(&node_unbinding_rate_field, {unbinding_rate}) //
  //         .add_field_data<size_t>(&node_rng_counter_field, {0})  //
  //         .add_field_data<double>(&node_radius_field, {rod_radius});
  //     PRC1HeadIndex++;

  //     dec_helper.create_node()
  //         .owning_proc(0)     //
  //         .id(PRC1HeadIndex)  // 1 indexed                                                             // 1 indexed
  //         .add_field_data<double>(&node_arclength_field, {(MotorPerThisMT + 0.5) / MotorsPerMT * rod_length})  //
  //         .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0}) // .add_field_data<double>(&node_torque_field,
  //         {0.0, 0.0, 0.0})                                                //
  //         .add_field_data<unsigned>(&node_is_bound_field, {1}) // .add_field_data<double>(&node_binding_rate_field,
  //         {binding_rate})                                           //
  //         .add_field_data<double>(&node_unbinding_rate_field, {unbinding_rate}) //
  //         .add_field_data<size_t>(&node_rng_counter_field, {0}) //
  //         .add_field_data<double>(&node_radius_field, {rod_radius});

  //     dec_helper.create_element()
  //         .owning_proc(0)  //
  //         .id(MotorIndex)  //
  //         .topology(stk::topology::BEAM_2)
  //         .add_part(&prc1_part)                                                            //
  //         .nodes({PRC1HeadIndex - 1, PRC1HeadIndex})                                       //
  //         .add_field_data<double>(&elem_rest_length_field, {rest_length})                  //
  //         .add_field_data<double>(&elem_spring_constant_field, {spring_constant})          //
  //         .add_field_data<double>(&elem_ang_spring_constant_field, {ang_spring_constant})  //
  //         .add_field_data<double>(&elem_rest_cos_angle_field, {rest_cos_angle})            //
  //         .add_field_data<double>(&elem_torsion_spring_constant_field, {torsion_spring_constant})      //
  //         .add_field_data<double>(&elem_rest_cos_torsion_angle_field,  {(MotorIndex == 1) ? 0.9: 1.0 })  //
  //         .add_field_data<size_t>(&elem_rng_counter_field, {0});
  //     PRC1HeadIndex++;
  //     MotorIndex++;
  //   }
  // }
  // // Declare the entities
  // dec_helper.check_consistency(bulk_data);
  // bulk_data.modification_begin();
  // dec_helper.declare_entities(bulk_data);
  // bulk_data.modification_end();

  // // Add the links
  // stk::mesh::PartVector locally_owned_link_parts = {&slink_part, &meta_data.locally_owned_part()};
  // auto& partition = link_data.get_partition(link_data.get_partition_key(locally_owned_link_parts));
  // partition.increase_request_link_capacity(PRC1HeadIndex);
  // // Horizontal PRC1
  // PRC1HeadIndex = num_nodes + 1;
  // for (int MotorPerThisMT = 0; MotorPerThisMT < MotorsPerMT; MotorPerThisMT++) {
  //   XPosition = 0;
  //   for (int RodIndex = 1; RodIndex < SquareSize * (SquareSize); RodIndex++) {
  //     if (XPosition < (SquareSize - 1.5)) {
  //       stk::mesh::Entity e_mt1 = bulk_data.get_entity(ELEM_RANK, RodIndex);
  //       stk::mesh::Entity e_mt2 = bulk_data.get_entity(ELEM_RANK, RodIndex + 1);
  //       stk::mesh::Entity e_prc1_head1 = bulk_data.get_entity(NODE_RANK, PRC1HeadIndex);
  //       PRC1HeadIndex++;
  //       stk::mesh::Entity e_prc1_head2 = bulk_data.get_entity(NODE_RANK, PRC1HeadIndex);
  //       PRC1HeadIndex++;
  //       MUNDY_THROW_REQUIRE(bulk_data.is_valid(e_mt1) && bulk_data.is_valid(e_mt2) &&
  //                               bulk_data.is_valid(e_prc1_head1) && bulk_data.is_valid(e_prc1_head2),
  //                           std::runtime_error, "Invalid entities for link creation");
  //       partition.request_link(e_mt1, e_prc1_head1);
  //       partition.request_link(e_mt2, e_prc1_head2);
  //     }

  //     XPosition++;
  //     if (XPosition > (SquareSize - 1)) {
  //       // This code will be executed because x is greater than y
  //       XPosition = 0;
  //       YPosition++;
  //     }
  //   }
  //   // Vertical PRC1
  //   XPosition = 0;
  //   for (int RodIndex = 1; RodIndex < SquareSize * (SquareSize - 1) + 1; RodIndex++) {
  //     if (XPosition < (SquareSize - 0.5)) {
  //       stk::mesh::Entity e_mt1 = bulk_data.get_entity(ELEM_RANK, RodIndex);
  //       stk::mesh::Entity e_mt2 = bulk_data.get_entity(ELEM_RANK, RodIndex + SquareSize);
  //       stk::mesh::Entity e_prc1_head1 = bulk_data.get_entity(NODE_RANK, PRC1HeadIndex);
  //       PRC1HeadIndex++;
  //       stk::mesh::Entity e_prc1_head2 = bulk_data.get_entity(NODE_RANK, PRC1HeadIndex);
  //       PRC1HeadIndex++;
  //       MUNDY_THROW_REQUIRE(bulk_data.is_valid(e_mt1) && bulk_data.is_valid(e_mt2) &&
  //                               bulk_data.is_valid(e_prc1_head1) && bulk_data.is_valid(e_prc1_head2),
  //                           std::runtime_error, "Invalid entities for link creation");
  //       partition.request_link(e_mt1, e_prc1_head1);
  //       partition.request_link(e_mt2, e_prc1_head2);
  //     }
  //     XPosition++;
  //     if (XPosition > (SquareSize - 1)) {
  //       // This code will be executed because x is greater than y
  //       XPosition = 0;
  //       YPosition++;
  //     }
  //   }
  // }
  // link_data.process_requests();

  // Reconcile the surface node positions/tangents using their arclengths
  reconcile_surface_nodes(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);

  for (size_t timestep_index = 0; timestep_index < num_timesteps; timestep_index++) {
    // One timestep

    // Zero the velocities, forces, and torques
    mesh::field_fill<double>(0.0, node_velocity_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_omega_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_force_field, stk::ngp::ExecSpace{});
    mesh::field_fill<double>(0.0, node_torque_field, stk::ngp::ExecSpace{});

    // Binding/unbinding first
    std::cout << "Binding/unbinding" << std::endl;
    kmc_state_change_crosslinks(dt, binding_unbinding_kbt, link_data, slink_part)
        .apply_to(prc1_agg, prc1_head_agg, rod_agg, slink_agg, AngularCrosslinkerEnergy{screening_length});
    reconcile_surface_nodes(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);  // Must update the surface
                                                                                     // nodes after binding

    // Compute Brownian motions
    std::cout << "Applying Brownian motion" << std::endl;
    // apply_brownian_motion_rod(dt, brownian_kbt).apply_to(rod_agg);
    // apply_brownian_motion_unbound_prc1(dt, brownian_kbt).apply_to(prc1_head_agg);

    // Compute spring forces
    std::cout << "Computing spring forces" << std::endl;
    eval_prc1_force().apply_to(prc1_agg);

    // Compute Hertzian contact forces
    std::cout << "Computing Hertzian contact forces" << std::endl;
    apply_hertzian_contact(hertz_youngs_modulus, hertz_poisson_ratio).apply_to(rod_agg);

    // Compute map surface forces to center forces
    std::cout << "Mapping spring surface forces to center forces" << std::endl;
    map_surface_force_to_com_force(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);

    // Map center forces to center velocity
    eval_mobility_rod().apply_to(rod_agg);
    eval_mobility_unbound_prc1().apply_to(prc1_head_agg);

    // Write to file
    if (timestep_index % io_frequency == 0) {
      std::cout << "Writing to file" << std::endl;
      stk::io::StkMeshIoBroker stk_io_broker;
      stk_io_broker.property_add(Ioss::Property("MAXIMUM_NAME_LENGTH", 180));
      stk_io_broker.set_bulk_data(bulk_data);
      size_t step = timestep_index + 1;
      double time = timestep_index * dt;

      // Left pad the timestep index with zeros to as many digits as the number of timesteps
      auto pad_on_left_with_zeros = [](int number, int width) {
        std::stringstream ss;
        ss << std::setw(width) << std::setfill('0') << number;
        return ss.str();
      };
      std::string file_name =
          "will.e-s." + pad_on_left_with_zeros(timestep_index, std::to_string(num_timesteps).size());
      stk::io::write_mesh_with_fields(file_name, stk_io_broker, step, time, stk::io::WRITE_RESULTS);
    }

    // Update center positions and orientations (quat and tangent)
    update_configuration_center(dt).apply_to(rod_node_agg);
    update_configuration_orientation(dt).apply_to(rod_node_agg);
    update_configuration_center(dt).apply_to(prc1_head_agg);

    // Reconcile surface node positions and tangents
    reconcile_surface_nodes(link_data, rod_agg, prc1_head_agg).apply_to(slink_agg);
  }
}
//@}

}  // namespace mundy

int main(int argc, char** argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  mundy::run_main();

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}