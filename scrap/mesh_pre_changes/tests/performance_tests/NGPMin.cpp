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

// STK IO
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>

// STK Mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>

// STK Search
#include <stk_search/BoxIdent.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/Point.hpp>
#include <stk_search/SearchMethod.hpp>
#include <stk_search/Sphere.hpp>

// STK Topology
#include <stk_topology/topology.hpp>

// STK Util
#include <stk_util/ngp/NgpSpaces.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// STK Balance
#include <stk_balance/balance.hpp>  // for balanceStkMesh

// Mundy
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_REQUIRE
#include <mundy_math/Vector3.hpp>          // for Vector3
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper

using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace = typename DeviceExecutionSpace::memory_space;

using IdentProc = stk::search::IdentProc<stk::mesh::EntityId, int>;
using SphereIdentProc = stk::search::BoxIdentProc<stk::search::Sphere<double>, IdentProc>;
using Intersection = stk::search::IdentProcIntersection<IdentProc, IdentProc>;
using SearchSpheresViewType = Kokkos::View<SphereIdentProc *, DeviceExecutionSpace>;
using ResultViewType = Kokkos::View<Intersection *, DeviceExecutionSpace>;
using FastMeshIndicesViewType = Kokkos::View<stk::mesh::FastMeshIndex *, DeviceExecutionSpace>;

using LocalIdentProc = stk::search::IdentProc<stk::mesh::FastMeshIndex, int>;
using LocalIntersection = stk::search::IdentProcIntersection<LocalIdentProc, LocalIdentProc>;
using LocalResultViewType = Kokkos::View<LocalIntersection *, DeviceExecutionSpace>;

//! \name Search
//@{

// Create local entities on host and copy to device
FastMeshIndicesViewType get_local_entity_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                                 stk::mesh::Selector selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(bulk_data, rank, selector, local_entities);

  FastMeshIndicesViewType mesh_indices("mesh_indices", local_entities.size());
  FastMeshIndicesViewType::HostMirror host_mesh_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mesh_indices);

  for (size_t i = 0; i < local_entities.size(); ++i) {
    const stk::mesh::MeshIndex &mesh_index = bulk_data.mesh_index(local_entities[i]);
    host_mesh_indices(i) = stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
  }

  Kokkos::deep_copy(mesh_indices, host_mesh_indices);
  return mesh_indices;
}

LocalResultViewType get_local_neighbor_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                               const ResultViewType &search_results) {
  auto host_search_results = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, search_results);

  // For each search result, get the local indices and store them in a view
  LocalResultViewType local_neighbor_indices("local_neighbor_indices", search_results.size());
  LocalResultViewType::HostMirror host_local_neighbor_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, local_neighbor_indices);

  for (size_t i = 0; i < search_results.size(); ++i) {
    const auto search_result = host_search_results(i);

    stk::mesh::Entity source_entity = bulk_data.get_entity(rank, search_result.domainIdentProc.id());
    stk::mesh::Entity target_entity = bulk_data.get_entity(rank, search_result.rangeIdentProc.id());

    const stk::mesh::MeshIndex &source_mesh_index = bulk_data.mesh_index(source_entity);
    const stk::mesh::MeshIndex &target_mesh_index = bulk_data.mesh_index(target_entity);

    const stk::mesh::FastMeshIndex source_fast_mesh_index = {source_mesh_index.bucket->bucket_id(),
                                                             source_mesh_index.bucket_ordinal};
    const stk::mesh::FastMeshIndex target_fast_mesh_index = {target_mesh_index.bucket->bucket_id(),
                                                             target_mesh_index.bucket_ordinal};

    host_local_neighbor_indices(i) =
        LocalIntersection{LocalIdentProc{source_fast_mesh_index, search_result.domainIdentProc.proc()},
                          LocalIdentProc{target_fast_mesh_index, search_result.rangeIdentProc.proc()}};
  }

  Kokkos::deep_copy(local_neighbor_indices, host_local_neighbor_indices);
  return local_neighbor_indices;
}

SearchSpheresViewType create_search_spheres(const stk::mesh::BulkData &bulk_data, const stk::mesh::NgpMesh &ngp_mesh,
                                            const double search_buffer, const stk::mesh::Selector &spheres,
                                            const stk::mesh::NgpField<double> &node_coords_field,
                                            const stk::mesh::NgpField<double> &elem_radius_field) {
  auto locally_owned_spheres = spheres & bulk_data.mesh_meta_data().locally_owned_part();
  const unsigned num_local_spheres =
      stk::mesh::count_entities(bulk_data, stk::topology::ELEMENT_RANK, locally_owned_spheres);
  SearchSpheresViewType search_spheres("search_spheres", num_local_spheres);

  // Slow host operation that is needed to get an index. There is plans to add this to the stk::mesh::NgpMesh.
  FastMeshIndicesViewType sphere_indices =
      get_local_entity_indices(bulk_data, stk::topology::ELEMENT_RANK, locally_owned_spheres);
  const int my_rank = bulk_data.parallel_rank();

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, num_local_spheres), KOKKOS_LAMBDA(const unsigned &i) {
        stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEMENT_RANK, sphere_indices(i));
        stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);

        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);

        stk::search::Point<double> center(node_coords_field(node_index, 0), node_coords_field(node_index, 1),
                                          node_coords_field(node_index, 2));
        double radius = elem_radius_field(sphere_index, 0) + search_buffer;
        search_spheres(i) = SphereIdentProc{stk::search::Sphere<double>(center, radius),
                                            IdentProc(ngp_mesh.identifier(sphere), my_rank)};
      });

  return search_spheres;
}

void ghost_neighbors(stk::mesh::BulkData &bulk_data, const ResultViewType &search_results) {
  auto host_search_results = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, search_results);
  bulk_data.modification_begin();
  stk::mesh::Ghosting &neighbor_ghosting = bulk_data.create_ghosting("neighbors");
  std::vector<stk::mesh::EntityProc> elements_to_ghost;

  const int my_parallel_rank = bulk_data.parallel_rank();

  for (size_t i = 0; i < host_search_results.size(); ++i) {
    auto result = host_search_results(i);
    const bool i_own_source = result.domainIdentProc.proc() == my_parallel_rank;
    const bool i_own_target = result.rangeIdentProc.proc() == my_parallel_rank;
    if (!i_own_source && i_own_target) {
      // Send the target to the source
      stk::mesh::Entity elem = bulk_data.get_entity(stk::topology::ELEM_RANK, result.rangeIdentProc.id());
      elements_to_ghost.emplace_back(elem, result.domainIdentProc.proc());
    } else if (i_own_source && !i_own_target) {
      // Send the source to the target
      stk::mesh::Entity elem = bulk_data.get_entity(stk::topology::ELEM_RANK, result.domainIdentProc.id());
      elements_to_ghost.emplace_back(elem, result.rangeIdentProc.proc());
    } else if (!i_own_source && !i_own_target) {
      throw std::runtime_error("Invalid search result. Somehow we received a pair of elements that we don't own.");
    }
  }

  bulk_data.change_ghosting(neighbor_ghosting, elements_to_ghost);
  bulk_data.modification_end();
}
//@}

//! \name Physics
//@{

void compute_signed_separation_distance_and_contact_normal(
    stk::mesh::NgpMesh &ngp_mesh, const LocalResultViewType &local_search_results,
    stk::mesh::NgpField<double> &node_coords_field, stk::mesh::NgpField<double> &elem_radius_field,
    const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
    const Kokkos::View<double **, DeviceMemorySpace> &con_normals_ij) {
  // Each neighbor pair will generate a constraint between the two spheres
  // Loop over each neighbor id pair, fetch each sphere's position, and compute the signed separation distance
  // defined by \|x_i - x_j\| - (r_i + r_j) where x_i and x_j are the sphere positions and r_i and r_j are the sphere
  // radii
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();

  // Loop over the neighbor pairs
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "GenerateCollisionConstraints", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_index = local_search_results(i).rangeIdentProc.id();
        if (source_index == target_index) {
          return;
        }

        stk::mesh::NgpMesh::ConnectedNodes source_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_index);
        stk::mesh::NgpMesh::ConnectedNodes target_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_index);
        stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_nodes[0]);
        stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_nodes[0]);

        // Fetch the sphere positions and radii
        const double x_i = node_coords_field(source_node_index, 0);
        const double y_i = node_coords_field(source_node_index, 1);
        const double z_i = node_coords_field(source_node_index, 2);
        const double x_j = node_coords_field(target_node_index, 0);
        const double y_j = node_coords_field(target_node_index, 1);
        const double z_j = node_coords_field(target_node_index, 2);
        const double radius_i = elem_radius_field(source_index, 0);
        const double radius_j = elem_radius_field(target_index, 0);

        // Compute the signed separation distance
        const double source_to_target_x = x_j - x_i;
        const double source_to_target_y = y_j - y_i;
        const double source_to_target_z = z_j - z_i;
        const double distance_between_centers =
            Kokkos::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                         source_to_target_z * source_to_target_z);
        signed_sep_dist(i) = distance_between_centers - radius_i - radius_j;

        // Compute the normal vector
        const double inv_distance_between_centers = 1.0 / distance_between_centers;
        con_normals_ij(i, 0) = source_to_target_x * inv_distance_between_centers;
        con_normals_ij(i, 1) = source_to_target_y * inv_distance_between_centers;
        con_normals_ij(i, 2) = source_to_target_z * inv_distance_between_centers;
      });
}

void compute_maximum_abs_projected_sep(const stk::ParallelMachine parallel,
                                       const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
                                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot, const double dt,
                                       double &maximum_abs_projected_sep) {
  // Perform parallel reduction over all linker indices
  double local_maximum_abs_projected_sep = -1.0;
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_reduce(
      "ComputeMaxAbsProjectedSep", range_policy(0, lagrange_multipliers.extent(0)),
      KOKKOS_LAMBDA(const int i, double &max_val) {
        // perform the projection EQ 2.2 of Dai & Fletcher 2005
        const double lag_mult = lagrange_multipliers(i);
        const double sep_old = signed_sep_dist(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_new = sep_old + dt * sep_dot;

        double abs_projected_sep;
        if (lag_mult < 1e-12) {
          abs_projected_sep = Kokkos::abs(Kokkos::min(sep_new, 0.0));
        } else {
          abs_projected_sep = Kokkos::abs(sep_new);
        }

        // update the maximum value
        if (abs_projected_sep > max_val) {
          max_val = abs_projected_sep;
        }
      },
      Kokkos::Max<double>(local_maximum_abs_projected_sep));

  // Global reduction
  maximum_abs_projected_sep = -1.0;
  stk::all_reduce_max(parallel, &local_maximum_abs_projected_sep, &maximum_abs_projected_sep, 1);
}

template <class Space>
struct DiffDotsReducer {
 public:
  // Required
  typedef DiffDotsReducer reducer;
  typedef mundy::math::Vector3d value_type;
  typedef Kokkos::View<value_type *, Space, Kokkos::MemoryUnmanaged> result_view_type;

 private:
  value_type &value;

 public:
  KOKKOS_INLINE_FUNCTION
  DiffDotsReducer(value_type &value_) : value(value_) {
  }

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type &dest, const value_type &src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type &val) const {
    val.set(0.0, 0.0, 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  value_type &reference() const {
    return value;
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const {
    return result_view_type(&value, 1);
  }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const {
    return true;
  }
};  // DiffDotsReducer

void compute_diff_dots(const stk::ParallelMachine parallel,
                       const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                       const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers_tmp,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot_tmp, const double dt,
                       double &dot_xkdiff_xkdiff, double &dot_xkdiff_gkdiff, double &dot_gkdiff_gkdiff) {
  // Local variables to store dot products
  mundy::math::Vector3d local_xx_xg_gg_diff = {0.0, 0.0, 0.0};

  // Perform parallel reduction to compute the dot products
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_reduce(
      "ComputeDiffDots", range_policy(0, lagrange_multipliers.extent(0)),
      KOKKOS_LAMBDA(const int i, mundy::math::Vector3d &acc_xx_xg_gg_diff) {
        const double lag_mult = lagrange_multipliers(i);
        const double lag_mult_tmp = lagrange_multipliers_tmp(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_dot_tmp = signed_sep_dot_tmp(i);

        // xkdiff = xk - xkm1
        const double xkdiff = lag_mult - lag_mult_tmp;

        // gkdiff = gk - gkm1
        const double gkdiff = dt * (sep_dot - sep_dot_tmp);

        // Compute the dot products
        acc_xx_xg_gg_diff[0] += xkdiff * xkdiff;
        acc_xx_xg_gg_diff[1] += xkdiff * gkdiff;
        acc_xx_xg_gg_diff[2] += gkdiff * gkdiff;
      },
      DiffDotsReducer<DeviceExecutionSpace>(local_xx_xg_gg_diff));

  // Global reduction
  stk::all_reduce_sum(parallel, &local_xx_xg_gg_diff[0], &dot_xkdiff_xkdiff, 1);
  stk::all_reduce_sum(parallel, &local_xx_xg_gg_diff[1], &dot_xkdiff_gkdiff, 1);
  stk::all_reduce_sum(parallel, &local_xx_xg_gg_diff[2], &dot_gkdiff_gkdiff, 1);
}

void sum_collision_force(stk::mesh::NgpMesh &ngp_mesh, const LocalResultViewType &local_search_results,
                         const Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
                         const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                         stk::mesh::NgpField<double> &node_force_field, const double alpha = 1.0,
                         const double beta = 0.0) {
  node_force_field.sync_to_device();

  // Zero out the force first
  node_force_field.set_all(ngp_mesh, 0.0);

  // Loop over the neighbor pairs
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "SumCollisionForce", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_index = local_search_results(i).rangeIdentProc.id();
        if (source_index == target_index) {
          return;
        }

        stk::mesh::NgpMesh::ConnectedNodes source_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_index);
        stk::mesh::NgpMesh::ConnectedNodes target_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_index);
        stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_nodes[0]);
        stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_nodes[0]);

        // Fetch the lagrange multiplier
        const double lag_mult = lagrange_multipliers(i);

        // Fetch the normal vector (goes from source to target)
        const double normal_x = con_normal_ij(i, 0);
        const double normal_y = con_normal_ij(i, 1);
        const double normal_z = con_normal_ij(i, 2);

        // Compute the force
        // Now, our neighbor list has both sphere i -> sphere j and sphere j -> sphere i
        // As a result, we need to take care then performing the force computation to not double count.
        //
        // We need to check this
        //
        // If the source and target are owned, then we only add into the source
        // If the source is owned and the target not, then we add into the source
        // If the target is owned and the source not, then we add into the target
        //
        // Actually, so long as auto_swap_domain_and_range = true, we don't need to worry about this
        // because the pair not be swapped if both are owned allowing us to always add into the source
        Kokkos::atomic_add(&node_force_field(source_node_index, 0),
                           -alpha * lag_mult * normal_x + beta * node_force_field(source_node_index, 0));
        Kokkos::atomic_add(&node_force_field(source_node_index, 1),
                           -alpha * lag_mult * normal_y + beta * node_force_field(source_node_index, 1));
        Kokkos::atomic_add(&node_force_field(source_node_index, 2),
                           -alpha * lag_mult * normal_z + beta * node_force_field(source_node_index, 2));
      });

  node_force_field.modify_on_device();
}

void compute_the_mobility_problem(stk::mesh::NgpMesh &ngp_mesh, const double viscosity,
                                  stk::mesh::NgpField<double> &elem_radius_field,
                                  stk::mesh::NgpField<double> &node_force_field,
                                  stk::mesh::NgpField<double> &node_velocity_field, const double alpha = 1.0,
                                  const double beta = 0.0) {
  elem_radius_field.sync_to_device();
  node_force_field.sync_to_device();
  node_velocity_field.sync_to_device();

  const stk::mesh::Selector selector(*elem_radius_field.get_field_base());

  // Self-interaction term
  const double pi = Kokkos::numbers::pi_v<double>;
  const double coeff = 1.0 / (6.0 * pi * viscosity);
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);

        const double radius = elem_radius_field(sphere_index, 0);
        const double force_x = node_force_field(node_index, 0);
        const double force_y = node_force_field(node_index, 1);
        const double force_z = node_force_field(node_index, 2);

        // Compute the velocity
        // Assume that no two spheres share the same node.
        const double inv_radius = 1.0 / radius;
        node_velocity_field(node_index, 0) =
            alpha * coeff * inv_radius * force_x + beta * node_velocity_field(node_index, 0);
        node_velocity_field(node_index, 1) =
            alpha * coeff * inv_radius * force_y + beta * node_velocity_field(node_index, 1);
        node_velocity_field(node_index, 2) =
            alpha * coeff * inv_radius * force_z + beta * node_velocity_field(node_index, 2);
      });

  node_velocity_field.modify_on_device();
}

void compute_rate_of_change_of_sep(stk::mesh::NgpMesh &ngp_mesh, const LocalResultViewType &local_search_results,
                                   stk::mesh::NgpField<double> &node_velocity_field,
                                   stk::mesh::NgpField<double> &elem_radius_dot_field,
                                   const Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
                                   const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot) {
  node_velocity_field.sync_to_device();
  elem_radius_dot_field.sync_to_device();

  // Compute the (linearized) rate of change in sep
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "ComputeRateOfChangeOfSep", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_index = local_search_results(i).rangeIdentProc.id();
        if (source_index == target_index) {
          return;
        }

        stk::mesh::NgpMesh::ConnectedNodes source_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_index);
        stk::mesh::NgpMesh::ConnectedNodes target_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_index);
        stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_nodes[0]);
        stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_nodes[0]);

        // Fetch the normal vector (goes from source to target)
        const double normal_x = con_normal_ij(i, 0);
        const double normal_y = con_normal_ij(i, 1);
        const double normal_z = con_normal_ij(i, 2);

        // Fetch the rate of change of radius of the spheres
        const double radius_dot_i = elem_radius_dot_field(source_index, 0);
        const double radius_dot_j = elem_radius_dot_field(target_index, 0);

        // Fetch the velocity of the source and target spheres
        const double source_velocity_x = node_velocity_field(source_node_index, 0);
        const double source_velocity_y = node_velocity_field(source_node_index, 1);
        const double source_velocity_z = node_velocity_field(source_node_index, 2);
        const double target_velocity_x = node_velocity_field(target_node_index, 0);
        const double target_velocity_y = node_velocity_field(target_node_index, 1);
        const double target_velocity_z = node_velocity_field(target_node_index, 2);

        // Compute the rate of change in separation
        signed_sep_dot(i) = -normal_x * (source_velocity_x - target_velocity_x) -
                            normal_y * (source_velocity_y - target_velocity_y) -
                            normal_z * (source_velocity_z - target_velocity_z) - radius_dot_i - radius_dot_j;
      });
}

void compute_rate_of_change_of_sep(stk::mesh::NgpMesh &ngp_mesh, const LocalResultViewType &local_search_results,
                                   stk::mesh::NgpField<double> &node_velocity_field,
                                   const Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
                                   const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot) {
  node_velocity_field.sync_to_device();

  // Compute the (linearized) rate of change in sep
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "ComputeRateOfChangeOfSep", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_index = local_search_results(i).rangeIdentProc.id();
        if (source_index == target_index) {
          return;
        }

        stk::mesh::NgpMesh::ConnectedNodes source_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_index);
        stk::mesh::NgpMesh::ConnectedNodes target_nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_index);
        stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_nodes[0]);
        stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_nodes[0]);

        // Fetch the normal vector (goes from source to target)
        const double normal_x = con_normal_ij(i, 0);
        const double normal_y = con_normal_ij(i, 1);
        const double normal_z = con_normal_ij(i, 2);

        // Fetch the velocity of the source and target spheres
        const double source_velocity_x = node_velocity_field(source_node_index, 0);
        const double source_velocity_y = node_velocity_field(source_node_index, 1);
        const double source_velocity_z = node_velocity_field(source_node_index, 2);
        const double target_velocity_x = node_velocity_field(target_node_index, 0);
        const double target_velocity_y = node_velocity_field(target_node_index, 1);
        const double target_velocity_z = node_velocity_field(target_node_index, 2);

        // Compute the rate of change in separation
        signed_sep_dot(i) = -normal_x * (source_velocity_x - target_velocity_x) -
                            normal_y * (source_velocity_y - target_velocity_y) -
                            normal_z * (source_velocity_z - target_velocity_z);
      });
}

void update_con_gammas(const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers,
                       const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers_tmp,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
                       const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dot, const double dt,
                       const double alpha) {
  using range_policy = Kokkos::RangePolicy<DeviceExecutionSpace>;
  Kokkos::parallel_for(
      "UpdateConGammas", range_policy(0, lagrange_multipliers.extent(0)), KOKKOS_LAMBDA(const int i) {
        // Fetch fields for the current linker
        const double sep_old = signed_sep_dist(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_new = sep_old + dt * sep_dot;

        // Update lagrange multipliers with bound projection
        lagrange_multipliers(i) = Kokkos::max(lagrange_multipliers_tmp(i) - alpha * sep_new, 0.0);
      });
}

struct CollisionResult {
  double max_abs_projected_sep;
  int ite_count;
  double max_displacement;
};

template <typename Field>
struct FieldSpeedReductionFunctor {
  KOKKOS_FUNCTION
  FieldSpeedReductionFunctor(Field &field, Kokkos::Max<double> max_reduction)
      : field_(field), max_reduction_(max_reduction) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const stk::mesh::FastMeshIndex &f, double &value) const {
    const double magnitude_sq = field_(f, 0) * field_(f, 0) + field_(f, 1) * field_(f, 1) + field_(f, 2) * field_(f, 2);
    max_reduction_.join(value, magnitude_sq);
  }

 private:
  const Field field_;
  const Kokkos::Max<double> max_reduction_;
};

template <typename Mesh, typename Field>
double get_max_speed(Mesh &ngp_mesh, Field &vel_field) {
  vel_field.sync_to_device();

  stk::mesh::Selector field_selector(*vel_field.get_field_base());
  double local_max_speed_sq = 0.0;
  Kokkos::Max<double> max_reduction(local_max_speed_sq);
  FieldSpeedReductionFunctor<Field> functor(vel_field, max_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, vel_field.get_rank(), field_selector, max_reduction, functor);

  double global_max_speed_sq = 0.0;
  stk::all_reduce_max(vel_field.get_field_base()->get_mesh().parallel(), &local_max_speed_sq, &global_max_speed_sq, 1);
  return Kokkos::sqrt(global_max_speed_sq);
}

CollisionResult resolve_collisions(
    stk::mesh::NgpMesh &ngp_mesh, const double viscosity, const double dt, const double max_allowable_overlap,
    const int max_col_iterations, const LocalResultViewType &local_search_results,
    stk::mesh::NgpField<double> &elem_radius_field, stk::mesh::NgpField<double> &elem_radius_dot_field,
    stk::mesh::NgpField<double> &node_force_field, stk::mesh::NgpField<double> &node_velocity_field,
    stk::mesh::NgpField<double> &node_collision_force_field, stk::mesh::NgpField<double> &node_collision_velocity_field,
    const Kokkos::View<double *, DeviceMemorySpace> &signed_sep_dist,
    const Kokkos::View<double **, DeviceMemorySpace> &con_normal_ij,
    const Kokkos::View<double *, DeviceMemorySpace> &lagrange_multipliers) {
  // Matrix-free BBPGD
  int ite_count = 0;
  int num_collisions = local_search_results.extent(0);
  Kokkos::View<double *, DeviceMemorySpace> lagrange_multipliers_tmp("lagrange_multipliers_tmp", num_collisions);
  Kokkos::View<double *, DeviceMemorySpace> signed_sep_dot("signed_sep_dot", num_collisions);
  Kokkos::View<double *, DeviceMemorySpace> signed_sep_dot_tmp("signed_sep_dot_tmp", num_collisions);
  const stk::ParallelMachine parallel = node_force_field.get_field_base()->get_mesh().parallel();

  // Use the given lagrange_multipliers as the initial guess
  Kokkos::deep_copy(lagrange_multipliers_tmp, lagrange_multipliers);
  Kokkos::deep_copy(signed_sep_dot_tmp, 0.0);

  // To account for external forces and external velocities, use them to update the initial signed_sep_dist
  // phi_0_corrected = phi_0 + dt * D^T U_ext

  // D^T U_ext + L^T ell_dot
  compute_rate_of_change_of_sep(ngp_mesh, local_search_results, node_velocity_field, elem_radius_dot_field,
                                con_normal_ij, signed_sep_dot);

  // signed_sep_dist += dt * signed_sep_dot
  Kokkos::parallel_for(
      "UpdateSignedSepDist", Kokkos::RangePolicy<DeviceExecutionSpace>(0, num_collisions),
      KOKKOS_LAMBDA(const int i) { signed_sep_dist(i) += dt * signed_sep_dot(i); });

  ///////////////////////
  Kokkos::deep_copy(signed_sep_dot, 0.0);

  // Compute gkm1 = D^T M D xkm1
  // Compute F = alpha D xkm1 + beta F
  sum_collision_force(ngp_mesh, local_search_results, con_normal_ij, lagrange_multipliers_tmp,
                      node_collision_force_field);

  // Compute U = alpha M F + beta U
  compute_the_mobility_problem(ngp_mesh, viscosity, elem_radius_field, node_collision_force_field,
                               node_collision_velocity_field);

  // Compute gkm1 = dt D^T U
  compute_rate_of_change_of_sep(ngp_mesh, local_search_results, node_collision_velocity_field, con_normal_ij,
                                signed_sep_dot_tmp);

  ///////////////////////
  // Check convergence //
  ///////////////////////
  // res = max(abs(projectPhi(gkm1)));
  double maximum_abs_projected_sep = -1.0;
  compute_maximum_abs_projected_sep(parallel, lagrange_multipliers_tmp, signed_sep_dist, signed_sep_dot_tmp, dt,
                                    maximum_abs_projected_sep);

  ///////////////////////
  // Loop if necessary //
  ///////////////////////
  if (maximum_abs_projected_sep < max_allowable_overlap) {
    // The initial guess was correct, nothing more is necessary
  } else {
    // Initial guess insufficient, iterate

    // First step, Dai&Fletcher2005 Section 5.
    double alpha = 1.0 / maximum_abs_projected_sep;
    while (ite_count < max_col_iterations) {
      ++ite_count;

      // Compute xk = xkm1 - alpha * gkm1 and perform the bound projection xk = boundProjection(xk)
      update_con_gammas(lagrange_multipliers, lagrange_multipliers_tmp, signed_sep_dist, signed_sep_dot, dt, alpha);

      // Compute new grad with xk: gk = dt D^T M D xk
      //   Compute F = D xk
      sum_collision_force(ngp_mesh, local_search_results, con_normal_ij, lagrange_multipliers,
                          node_collision_force_field);

      // Compute U = M F
      compute_the_mobility_problem(ngp_mesh, viscosity, elem_radius_field, node_collision_force_field,
                                   node_collision_velocity_field);

      //   Compute gk = dt D^T U
      compute_rate_of_change_of_sep(ngp_mesh, local_search_results, node_collision_velocity_field, con_normal_ij,
                                    signed_sep_dot);

      // check convergence via res = max(abs(projectPhi(gk)));
      compute_maximum_abs_projected_sep(parallel, lagrange_multipliers, signed_sep_dist, signed_sep_dot, dt,
                                        maximum_abs_projected_sep);

      if (maximum_abs_projected_sep < max_allowable_overlap) {
        // lagrange_multipliers worked.
        std::cout << "Convergence reached: " << maximum_abs_projected_sep << " < " << max_allowable_overlap
                  << std::endl;
        break;
      }

      ///////////////////////////////////////////////////////////////////////////
      // Compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff) //
      // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1                       //
      ///////////////////////////////////////////////////////////////////////////
      double global_dot_xkdiff_xkdiff = 0.0;
      double global_dot_xkdiff_gkdiff = 0.0;
      double global_dot_gkdiff_gkdiff = 0.0;
      compute_diff_dots(parallel, lagrange_multipliers, lagrange_multipliers_tmp, signed_sep_dot, signed_sep_dot_tmp,
                        dt, global_dot_xkdiff_xkdiff, global_dot_xkdiff_gkdiff, global_dot_gkdiff_gkdiff);

      ////////////////////////////////////////////
      // Compute the Barzilai-Borwein step size //
      ////////////////////////////////////////////
      // Alternating bb1 and bb2 methods
      double a;
      double b;
      if (ite_count % 2 == 0) {
        // Barzilai-Borwein step size Choice 1
        a = global_dot_xkdiff_xkdiff;
        b = global_dot_xkdiff_gkdiff;
      } else {
        // Barzilai-Borwein step size Choice 2
        a = global_dot_xkdiff_gkdiff;
        b = global_dot_gkdiff_gkdiff;
      }

      // Prevent div 0 errors.
      if (Kokkos::abs(b) < 1e-12) {
        b += 1e-12;
      }

      alpha = a / b;

      /////////////////////////////////
      // Set xkm1 = xk and gkm1 = gk //
      /////////////////////////////////
      Kokkos::deep_copy(lagrange_multipliers_tmp, lagrange_multipliers);
      Kokkos::deep_copy(signed_sep_dot_tmp, signed_sep_dot);
    }
  }

  if (ite_count == max_col_iterations) {
    throw std::runtime_error("Collision resolution did not converge!");
  }

  // Sum the collision forces and velocities into the total forces and velocities
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, stk::mesh::Selector(*node_force_field.get_field_base()),
      KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        node_force_field(node_index, 0) += node_collision_force_field(node_index, 0);
        node_force_field(node_index, 1) += node_collision_force_field(node_index, 1);
        node_force_field(node_index, 2) += node_collision_force_field(node_index, 2);

        node_velocity_field(node_index, 0) += node_collision_velocity_field(node_index, 0);
        node_velocity_field(node_index, 1) += node_collision_velocity_field(node_index, 1);
        node_velocity_field(node_index, 2) += node_collision_velocity_field(node_index, 2);
      });
  node_force_field.modify_on_device();
  node_velocity_field.modify_on_device();

  // Compute the maximum speed
  double max_speed = get_max_speed(ngp_mesh, node_velocity_field);
  CollisionResult result = {maximum_abs_projected_sep, ite_count, max_speed * dt};
  return result;
}

void check_overlap(const stk::mesh::BulkData &bulk_data, const double max_allowable_overlap,
                   const stk::mesh::Selector &spheres, const stk::mesh::Field<double> &node_coords_field,
                   const stk::mesh::Field<double> &elem_radius_field) {
  // Do the check on host for easier printing
  // Loop over all pairs of spheres via the element buckets
  bool no_overlap = true;

  const stk::mesh::BucketVector &all_sphere_buckets = bulk_data.get_buckets(stk::topology::ELEMENT_RANK, spheres);
  const size_t num_buckets = all_sphere_buckets.size();
  for (size_t source_bucket_idx = 0; source_bucket_idx < num_buckets; ++source_bucket_idx) {
    stk::mesh::Bucket &source_bucket = *all_sphere_buckets[source_bucket_idx];
    const size_t source_bucket_size = source_bucket.size();

    for (size_t source_bucket_ord = 0; source_bucket_ord < source_bucket_size; ++source_bucket_ord) {
      const stk::mesh::Entity &source_sphere = source_bucket[source_bucket_ord];
      const stk::mesh::Entity &source_node = bulk_data.begin_nodes(source_sphere)[0];
      const double *source_coords = stk::mesh::field_data(node_coords_field, source_node);
      const double source_radius = *stk::mesh::field_data(elem_radius_field, source_sphere);

      for (size_t target_bucket_idx = 0; target_bucket_idx < num_buckets; ++target_bucket_idx) {
        stk::mesh::Bucket &target_bucket = *all_sphere_buckets[target_bucket_idx];
        const size_t target_bucket_size = target_bucket.size();

        for (size_t target_bucket_ord = 0; target_bucket_ord < target_bucket_size; ++target_bucket_ord) {
          // Skip self-interactions
          if (source_bucket_idx == target_bucket_idx && source_bucket_ord == target_bucket_ord) {
            continue;
          }

          const stk::mesh::Entity &target_sphere = target_bucket[target_bucket_ord];
          const stk::mesh::Entity &target_node = bulk_data.begin_nodes(target_sphere)[0];
          const double *target_coords = stk::mesh::field_data(node_coords_field, target_node);
          const double target_radius = *stk::mesh::field_data(elem_radius_field, target_sphere);

          // Compute the distance between the centers of the spheres
          const double source_to_target_x = source_coords[0] - target_coords[0];
          const double source_to_target_y = source_coords[1] - target_coords[1];
          const double source_to_target_z = source_coords[2] - target_coords[2];
          const double distance_between_centers =
              Kokkos::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                           source_to_target_z * source_to_target_z);

          // Compute the overlap
          const double ssd = distance_between_centers - source_radius - target_radius;
          if (ssd < -max_allowable_overlap + 1e-12) {
            // The spheres are overlapping too much
            no_overlap = false;
            std::cout << "Overlap detected between spheres " << target_sphere << " and " << source_sphere << std::endl;
            std::cout << "Distance between centers: " << distance_between_centers << std::endl;
            std::cout << "Radii: " << source_radius << " and " << target_radius << std::endl;
            std::cout << "Overlap: " << ssd << std::endl;
            std::cout << "Sphere positions: (" << source_coords[0] << ", " << source_coords[1] << ", "
                      << source_coords[2] << ") and (" << target_coords[0] << ", " << target_coords[1] << ", "
                      << target_coords[2] << ")" << std::endl;
          }
        }
      }
    }
  }

  if (no_overlap) {
    std::cout << "***No overlap detected***" << std::endl;
  } else {
    std::cout << "***Overlap detected***" << std::endl;
  }
}

bool divide_large_spheres(
    stk::mesh::BulkData &bulk_data, stk::mesh::Part &spheres_part, stk::mesh::Part &springs_part,
    const double division_radius, const double growth_rate, const double rest_length, const double spring_constant,
    const stk::mesh::Field<double> &node_coords_field, const stk::mesh::Field<double> &node_force_field,
    const stk::mesh::Field<double> &node_velocity_field, const stk::mesh::Field<double> &node_collision_force_field,
    const stk::mesh::Field<double> &node_collision_velocity_field, const stk::mesh::Field<double> &elem_radius_field,
    const stk::mesh::Field<double> &elem_radius_dot_field, const stk::mesh::Field<double> &elem_rest_length_field,
    const stk::mesh::Field<double> &elem_spring_constant_field) {
  // Some comments:
  //  Each sphere that has radius larger than the division radius will divide into two spheres.
  //  Implementation-wise, we will simply keep the old sphere, halve its radius, shift its center,
  //  and create a new sphere with a shifted center and same radius. The choice of where to draw the
  //  division plane is done via a functor that is passed in. The functor will accept the sphere
  //  pre-division and will return the unit normal to the division plane. The two resulting spheres
  //  are then connected by a newly created spring.
  //
  //  I dislike the fact that this function currently takes in a HUGE number of fields merely because
  //  of copying fields to the nrely created spring and sphere. I would prefer that the functor handle this.
  //  The user gives a functor that returns if the sphere should divide or not. We simply quary this function,
  //  declare/attach the spheres, nodes, and springs. Then, we use their set fields functor, which talks in the
  //  two spheres and the spring and sets any of their fields that it wants.

  // Count the number of spheres that are ready to divide
  size_t num_divisions = 0;
  stk::mesh::EntityVector all_locally_owned_spheres;
  stk::mesh::get_selected_entities(spheres_part, bulk_data.buckets(stk::topology::ELEMENT_RANK),
                                   all_locally_owned_spheres);
  for (const stk::mesh::Entity &sphere : all_locally_owned_spheres) {
    const double radius = *stk::mesh::field_data(elem_radius_field, sphere);
    if (radius > division_radius) {
      ++num_divisions;
    }
  }

  // Create the new spheres and their nodes
  bulk_data.modification_begin();

  std::vector<size_t> requests(bulk_data.mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::NODE_RANK] = num_divisions;
  requests[stk::topology::ELEMENT_RANK] = 2 * num_divisions;
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data.generate_new_entities(requests, requested_entities);

  // Change the elements to their desired part
  std::vector<stk::mesh::Part *> add_spheres_part = {&spheres_part};
  std::vector<stk::mesh::Part *> add_springs_part = {&springs_part};

  for (int i = 0; i < num_divisions; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_divisions + i];
    bulk_data.change_entity_parts(sphere_i, add_spheres_part);
  }

  for (int i = 0; i < num_divisions; i++) {
    stk::mesh::Entity sphere_i = requested_entities[2 * num_divisions + i];
    bulk_data.change_entity_parts(sphere_i, add_springs_part);
  }

  for (int i = 0; i < num_divisions; i++) {
    stk::mesh::Entity sphere_i = requested_entities[num_divisions + i];
    bulk_data.declare_relation(sphere_i, requested_entities[i], 0);
  }

  // Loop over the buckets and perform the actual divisions
  size_t random_seed = rand();
  openrand::Philox rng(random_seed, 0);
  size_t division_counter = 0;
  for (const stk::mesh::Entity &sphere : all_locally_owned_spheres) {
    const double radius = *stk::mesh::field_data(elem_radius_field, sphere);
    if (radius > division_radius) {
      // Fetch the sphere's center
      const stk::mesh::Entity &node = bulk_data.begin_nodes(sphere)[0];
      double *coords = stk::mesh::field_data(node_coords_field, node);

      // Fetch the unit normal to the division
      // Will be replaced by a functor in the future
      const double normal[3] = {1.0, 0.0, 0.0};

      // Update the new sphere
      const double half_radius = 0.5 * radius;
      stk::mesh::Entity new_sphere = requested_entities[num_divisions + division_counter];
      stk::mesh::Entity new_node = requested_entities[division_counter];
      double *new_coords = stk::mesh::field_data(node_coords_field, new_node);
      new_coords[0] = coords[0] + half_radius * normal[0];
      new_coords[1] = coords[1] + half_radius * normal[1];
      new_coords[2] = coords[2] + half_radius * normal[2];
      stk::mesh::field_data(elem_radius_field, new_sphere)[0] = half_radius;
      stk::mesh::field_data(elem_radius_dot_field, new_sphere)[0] = growth_rate;
      stk::mesh::field_data(node_force_field, new_node)[0] = 0.0;
      stk::mesh::field_data(node_force_field, new_node)[1] = 0.0;
      stk::mesh::field_data(node_force_field, new_node)[2] = 0.0;

      stk::mesh::field_data(node_velocity_field, new_node)[0] = 0.0;
      stk::mesh::field_data(node_velocity_field, new_node)[1] = 0.0;
      stk::mesh::field_data(node_velocity_field, new_node)[2] = 0.0;

      stk::mesh::field_data(node_collision_force_field, new_node)[0] = 0.0;
      stk::mesh::field_data(node_collision_force_field, new_node)[1] = 0.0;
      stk::mesh::field_data(node_collision_force_field, new_node)[2] = 0.0;

      stk::mesh::field_data(node_collision_velocity_field, new_node)[0] = 0.0;
      stk::mesh::field_data(node_collision_velocity_field, new_node)[1] = 0.0;
      stk::mesh::field_data(node_collision_velocity_field, new_node)[2] = 0.0;

      // Update the old sphere
      coords[0] -= half_radius * normal[0];
      coords[1] -= half_radius * normal[1];
      coords[2] -= half_radius * normal[2];
      stk::mesh::field_data(elem_radius_field, sphere)[0] = half_radius;

      // // Disconnect the first spring from the old sphere and connect it to the new sphere
      // int num_connected_elems = bulk_data.num_elements(node);
      // for (int i = 0; i < num_connected_elems; i++) {
      //   stk::mesh::Entity connected_elem = bulk_data.begin_elements(node)[i];
      //   bool is_spring = bulk_data.bucket(connected_elem).member(springs_part);
      //   if (is_spring) {
      //     // Fetch the spring's connected nodes
      //     stk::mesh::Entity const *connected_nodes = bulk_data.begin_nodes(connected_elem);
      //     if (connected_nodes[0] == node) {
      //       bulk_data.destroy_relation(connected_elem, node, 0);
      //       bulk_data.declare_relation(connected_elem, new_node, 0);
      //     } else if (connected_nodes[1] == node) {
      //       bulk_data.destroy_relation(connected_elem, node, 1);
      //       bulk_data.declare_relation(connected_elem, new_node, 1);
      //     } else {
      //       throw std::runtime_error("Connected spring does not have the expected nodes");
      //     }

      //     break;
      //   }
      // }

      // One (and just one) of the springs connected to the old sphere will detach from it
      // and connect to the new sphere. To decide which one, we choose based on the energy
      // of the system in each possible configuration.
      //
      // U = 0.5 * k * (r - r0)^2
      //
      // Start by computing the total energy if all springs connect to the old sphere
      double state_total_energy = 0.0;
      int num_connected_elems = bulk_data.num_elements(node);
      for (int i = 0; i < num_connected_elems; i++) {
        stk::mesh::Entity connected_elem = bulk_data.begin_elements(node)[i];
        bool is_spring = bulk_data.bucket(connected_elem).member(springs_part);
        if (is_spring) {
          // Fetch the spring's rest length and spring constant
          double rest_length = *stk::mesh::field_data(elem_rest_length_field, connected_elem);
          double spring_constant = *stk::mesh::field_data(elem_spring_constant_field, connected_elem);

          // Fetch the spring's connected nodes
          stk::mesh::Entity const *connected_nodes = bulk_data.begin_nodes(connected_elem);
          stk::mesh::Entity other_node = connected_nodes[0] == node ? connected_nodes[1] : connected_nodes[0];
          double *other_coords = stk::mesh::field_data(node_coords_field, other_node);

          // Compute the spring's energy
          double dx = other_coords[0] - coords[0];
          double dy = other_coords[1] - coords[1];
          double dz = other_coords[2] - coords[2];
          double distance = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          state_total_energy += 0.5 * spring_constant * (distance - rest_length) * (distance - rest_length);
        }
      }

      // Compute the energy of the state corresponding to the given spring connecting to the new sphere
      // Sum this into the z_score_total variable
      double z_score_total = 0.0;
      for (int i = 0; i < num_connected_elems; i++) {
        stk::mesh::Entity connected_elem = bulk_data.begin_elements(node)[i];
        bool is_spring = bulk_data.bucket(connected_elem).member(springs_part);
        if (is_spring) {
          // Fetch the spring's rest length and spring constant
          double rest_length = *stk::mesh::field_data(elem_rest_length_field, connected_elem);
          double spring_constant = *stk::mesh::field_data(elem_spring_constant_field, connected_elem);

          // Fetch the spring's connected nodes
          stk::mesh::Entity const *connected_nodes = bulk_data.begin_nodes(connected_elem);
          stk::mesh::Entity other_node = connected_nodes[0] == node ? connected_nodes[1] : connected_nodes[0];
          double *other_coords = stk::mesh::field_data(node_coords_field, other_node);

          // Compute the spring's energy if connected to the old and new spheres
          double dx = other_coords[0] - coords[0];
          double dy = other_coords[1] - coords[1];
          double dz = other_coords[2] - coords[2];
          double distance = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          double energy = 0.5 * spring_constant * (distance - rest_length) * (distance - rest_length);

          double dx_new = other_coords[0] - new_coords[0];
          double dy_new = other_coords[1] - new_coords[1];
          double dz_new = other_coords[2] - new_coords[2];
          double distance_new = Kokkos::sqrt(dx_new * dx_new + dy_new * dy_new + dz_new * dz_new);
          double energy_new = 0.5 * spring_constant * (distance_new - rest_length) * (distance_new - rest_length);

          // Add the total probability for this state to the z_score_total
          z_score_total += Kokkos::exp(-state_total_energy + energy - energy_new);
        }
      }

      // Draw a random number between 0 and z_score_total
      double random_number = rng.rand<double>() * z_score_total;

      // Recompute the z-score for each possible state and check if the random number falls within it
      // Considering each state as a bin. Use z_score_total as a running total.
      z_score_total = 0.0;
      for (int i = 0; i < num_connected_elems; i++) {
        stk::mesh::Entity connected_elem = bulk_data.begin_elements(node)[i];
        bool is_spring = bulk_data.bucket(connected_elem).member(springs_part);
        if (is_spring) {
          // Fetch the spring's rest length and spring constant
          double rest_length = *stk::mesh::field_data(elem_rest_length_field, connected_elem);
          double spring_constant = *stk::mesh::field_data(elem_spring_constant_field, connected_elem);

          // Fetch the spring's connected nodes
          stk::mesh::Entity const *connected_nodes = bulk_data.begin_nodes(connected_elem);
          stk::mesh::Entity other_node = connected_nodes[0] == node ? connected_nodes[1] : connected_nodes[0];
          double *other_coords = stk::mesh::field_data(node_coords_field, other_node);

          // Compute the spring's energy if connected to the old and new spheres
          double dx = other_coords[0] - coords[0];
          double dy = other_coords[1] - coords[1];
          double dz = other_coords[2] - coords[2];
          double distance = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          double energy = 0.5 * spring_constant * (distance - rest_length) * (distance - rest_length);

          double dx_new = other_coords[0] - new_coords[0];
          double dy_new = other_coords[1] - new_coords[1];
          double dz_new = other_coords[2] - new_coords[2];
          double distance_new = Kokkos::sqrt(dx_new * dx_new + dy_new * dy_new + dz_new * dz_new);
          double energy_new = 0.5 * spring_constant * (distance_new - rest_length) * (distance_new - rest_length);

          // Add the total probability for this state to the z_score_total
          z_score_total += Kokkos::exp(-state_total_energy + energy - energy_new);

          // Check if the random number falls within this bin
          if (random_number < z_score_total) {
            // Connect this spring to the new sphere, disconnecting it from the old sphere
            if (connected_nodes[0] == node) {
              bulk_data.destroy_relation(connected_elem, node, 0);
              bulk_data.declare_relation(connected_elem, new_node, 0);
            } else {
              bulk_data.destroy_relation(connected_elem, node, 1);
              bulk_data.declare_relation(connected_elem, new_node, 1);
            }

            break;
          }
        }
      }

      // Attach the new spring to the new and old spheres
      // MUST happen after one of the old springs is connected to the new sphere
      stk::mesh::Entity new_spring = requested_entities[2 * num_divisions + division_counter];
      bulk_data.declare_relation(new_spring, node, 0);
      bulk_data.declare_relation(new_spring, new_node, 1);
      stk::mesh::field_data(elem_rest_length_field, new_spring)[0] = rest_length;
      stk::mesh::field_data(elem_spring_constant_field, new_spring)[0] = spring_constant;

      ++division_counter;
    }
  }

  bulk_data.modification_end();

  // Return if a division occurred or not.
  return num_divisions > 0;
}

void update_sphere_positions(stk::mesh::NgpMesh &ngp_mesh, const double time_step,
                             stk::mesh::NgpField<double> &node_coords_field,
                             stk::mesh::NgpField<double> &node_velocity_field) {
  node_coords_field.sync_to_device();
  node_velocity_field.sync_to_device();
  auto selector = stk::mesh::Selector(*node_coords_field.get_field_base());
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        node_coords_field(node_index, 0) += time_step * node_velocity_field(node_index, 0);
        node_coords_field(node_index, 1) += time_step * node_velocity_field(node_index, 1);
        node_coords_field(node_index, 2) += time_step * node_velocity_field(node_index, 2);
      });

  node_coords_field.modify_on_device();
}

void update_sphere_positions(stk::mesh::NgpMesh &ngp_mesh, const double time_step, const stk::mesh::Selector &spheres,
                             stk::mesh::NgpField<double> &node_coords_field,
                             stk::mesh::NgpField<double> &node_velocity_field,
                             stk::mesh::NgpField<double> &elem_radius_field,
                             stk::mesh::NgpField<double> &elem_radius_dot_field) {
  node_coords_field.sync_to_device();
  node_velocity_field.sync_to_device();
  elem_radius_field.sync_to_device();
  elem_radius_dot_field.sync_to_device();
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, spheres, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);
        node_coords_field(node_index, 0) += time_step * node_velocity_field(node_index, 0);
        node_coords_field(node_index, 1) += time_step * node_velocity_field(node_index, 1);
        node_coords_field(node_index, 2) += time_step * node_velocity_field(node_index, 2);
        elem_radius_field(sphere_index, 0) += time_step * elem_radius_dot_field(sphere_index, 0);
      });

  node_coords_field.modify_on_device();
  elem_radius_field.modify_on_device();
}

void compute_spring_forces(stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &springs,
                           stk::mesh::NgpField<double> &node_coords_field,
                           stk::mesh::NgpField<double> &node_force_field,
                           stk::mesh::NgpField<double> &elem_spring_constant_field,
                           stk::mesh::NgpField<double> &elem_rest_length_field) {
  node_coords_field.sync_to_device();
  node_force_field.sync_to_device();
  elem_spring_constant_field.sync_to_device();
  elem_rest_length_field.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, springs, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
        const stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
        const stk::mesh::FastMeshIndex node0_index = ngp_mesh.fast_mesh_index(nodes[0]);
        const stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(nodes[1]);

        // Compute the edge length
        const double x0 = node_coords_field(node0_index, 0);
        const double y0 = node_coords_field(node0_index, 1);
        const double z0 = node_coords_field(node0_index, 2);
        const double x1 = node_coords_field(node1_index, 0);
        const double y1 = node_coords_field(node1_index, 1);
        const double z1 = node_coords_field(node1_index, 2);

        // From node 0 to node 1
        const double dx = x1 - x0;
        const double dy = y1 - y0;
        const double dz = z1 - z0;

        const double edge_length = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

        // Compute the spring force
        const double rest_length = elem_rest_length_field(spring_index, 0);
        const double spring_constant = elem_spring_constant_field(spring_index, 0);
        const double force_magnitude = spring_constant * (edge_length - rest_length);

        // Sum the force into the nodes
        const double inv_edge_length = 1.0 / edge_length;
        const double force_x = force_magnitude * dx * inv_edge_length;
        const double force_y = force_magnitude * dy * inv_edge_length;
        const double force_z = force_magnitude * dz * inv_edge_length;

        Kokkos::atomic_add(&node_force_field(node0_index, 0), force_x);
        Kokkos::atomic_add(&node_force_field(node0_index, 1), force_y);
        Kokkos::atomic_add(&node_force_field(node0_index, 2), force_z);

        Kokkos::atomic_add(&node_force_field(node1_index, 0), -force_x);
        Kokkos::atomic_add(&node_force_field(node1_index, 1), -force_y);
        Kokkos::atomic_add(&node_force_field(node1_index, 2), -force_z);
      });

  node_force_field.modify_on_device();
}
//@}

//! \name Load Balance
//@{

class RcbSettings : public stk::balance::BalanceSettings {
 public:
  RcbSettings() {
  }
  virtual ~RcbSettings() {
  }

  virtual bool isIncrementalRebalance() const {
    return false;
  }
  virtual std::string getDecompMethod() const {
    return std::string("rcb");
  }
  virtual std::string getCoordinateFieldName() const {
    return std::string("COORDS");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings
//@}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <num_spheres_x> <num_spheres_y>" << std::endl;
    return 1;
  }
  size_t num_spheres_x = std::stoi(argv[1]);
  size_t num_spheres_y = std::stoi(argv[2]);
  size_t num_spheres = num_spheres_x * num_spheres_y;

  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  {
    // Simulation of N spheres in a cube
    const double viscosity = 1.0;
    const double sphere_radius_min = 0.5;
    const double sphere_radius_max = 0.5;
    const double sphere_division_radius = 1.0;
    const double sphere_growth_rate = 0.1;
    const double spring_constant = 1.0;
    const double spring_rest_length = 0.0;
    const double grid_spacing = 1.0;
    const int max_col_iterations = 10000;
    const double max_allowable_overlap = 1e-5;

    const double time_step_size = 0.1;
    const size_t num_time_steps = 1000;
    const size_t io_frequency = 1;
    const double search_buffer = 2.0 * sphere_radius_max;

    std::cout << "Setup: " << std::endl;
    std::cout << "  Number of spheres: " << num_spheres << std::endl;
    std::cout << "  Sphere radius min: " << sphere_radius_min << std::endl;
    std::cout << "  Sphere radius max: " << sphere_radius_max << std::endl;

    // Setup the STK mesh
    stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
    mesh_builder.set_spatial_dimension(3);
    mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
    meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition to
                                         // all fields are simple.
    meta_data_ptr->set_coordinate_field_name("COORDS");
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create(meta_data_ptr);
    stk::mesh::MetaData &meta_data = *meta_data_ptr;
    stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

    // Create the spheres
    auto &spheres_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
    auto &springs_part = meta_data.declare_part_with_topology("SPRINGS", stk::topology::BEAM_2);
    stk::io::put_io_part_attribute(spheres_part);
    stk::io::put_io_part_attribute(springs_part);

    auto &node_coords_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "COORDS");
    auto &node_displacement_since_last_rebuild_field =
        meta_data.declare_field<double>(stk::topology::NODE_RANK, "OUR_DISPLACEMENT");
    auto &node_force_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
    auto &node_velocity_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");
    auto &node_collision_force_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "COLLISION_FORCE");
    auto &node_collision_velocity_field =
        meta_data.declare_field<double>(stk::topology::NODE_RANK, "COLLISION_VELOCITY");
    auto &elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "RADIUS");
    auto &elem_radius_dot_field = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "RADIUS_DOT");
    auto &elem_rest_length_field = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "REST_LENGTH");
    auto &elem_spring_constant_field = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "SPRING_CONSTANT");

    stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(node_collision_force_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(node_collision_velocity_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(elem_radius_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(elem_radius_dot_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(elem_rest_length_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_role(elem_spring_constant_field, Ioss::Field::TRANSIENT);
    stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_output_type(node_collision_force_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_output_type(node_collision_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
    stk::io::set_field_output_type(elem_radius_field, stk::io::FieldOutputType::SCALAR);
    stk::io::set_field_output_type(elem_rest_length_field, stk::io::FieldOutputType::SCALAR);
    stk::io::set_field_output_type(elem_spring_constant_field, stk::io::FieldOutputType::SCALAR);

    // Assign fields to parts
    stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_displacement_since_last_rebuild_field, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_force_field, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_velocity_field, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_collision_force_field, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_collision_velocity_field, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(elem_radius_field, spheres_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(elem_radius_dot_field, spheres_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(elem_rest_length_field, springs_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(elem_spring_constant_field, springs_part, 1, nullptr);

    // Concretize the mesh
    meta_data.commit();

    // Generate the grid of spheres and springs
    mundy::mesh::DeclareEntitiesHelper dec_helper;
    size_t node_count = 0;
    size_t element_count = 0;

    // Generate nodes with unique IDs
    std::map<std::tuple<size_t, size_t>, size_t> grid_to_node_id_map;
    openrand::Philox rng(0, 0);
    for (size_t i = 0; i < num_spheres_x; ++i) {
      for (size_t j = 0; j < num_spheres_y; ++j) {
        double x = static_cast<double>(i) * grid_spacing;
        double y = static_cast<double>(j) * grid_spacing;

        // Create the node and store the node id in a map
        auto node = dec_helper.create_node();
        node.owning_proc(0)                                                                        //
            .id(node_count + 1)                                                                    //
            .add_parts({&spheres_part, &springs_part})                                             //
            .add_field_data<double>(&node_coords_field, {x, y, 0.0})                               //
            .add_field_data<double>(&node_displacement_since_last_rebuild_field, {0.0, 0.0, 0.0})  //
            .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                            //
            .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})                         //
            .add_field_data<double>(&node_collision_force_field, {0.0, 0.0, 0.0})                  //
            .add_field_data<double>(&node_collision_velocity_field, {0.0, 0.0, 0.0});
        grid_to_node_id_map[{i, j}] = node_count + 1;
        ++node_count;

        // Create the sphere and connect it to the node
        const double radius = rng.uniform(sphere_radius_min, sphere_radius_max);
        auto sphere = dec_helper.create_element();
        sphere
            .owning_proc(0)                                      //
            .id(element_count + 1)                               //
            .topology(stk::topology::PARTICLE)                   //
            .nodes({node_count})                                 //
            .add_part(&spheres_part)                             //
            .add_field_data<double>(&elem_radius_field, radius)  //
            .add_field_data<double>(&elem_radius_dot_field, sphere_growth_rate);
        ++element_count;
      }
    }

    // Generate springs with unique IDs
    for (const auto &[grid_indices, node_id] : grid_to_node_id_map) {
      auto [i, j] = grid_indices;

      // Horizontal spring
      if (grid_to_node_id_map.find({i + 1, j}) != grid_to_node_id_map.end()) {
        auto spring = dec_helper.create_element();
        spring
            .owning_proc(0)                                                       //
            .id(element_count + 1)                                                //
            .topology(stk::topology::BEAM_2)                                      //
            .nodes({node_id, grid_to_node_id_map[{i + 1, j}]})                    //
            .add_part(&springs_part)                                              //
            .add_field_data<double>(&elem_rest_length_field, spring_rest_length)  //
            .add_field_data<double>(&elem_spring_constant_field, spring_constant);
        ++element_count;
      }

      // Vertical spring
      if (grid_to_node_id_map.find({i, j + 1}) != grid_to_node_id_map.end()) {
        auto spring = dec_helper.create_element();
        spring
            .owning_proc(0)                                                       //
            .id(element_count + 1)                                                //
            .topology(stk::topology::BEAM_2)                                      //
            .nodes({node_id, grid_to_node_id_map[{i, j + 1}]})                    //
            .add_part(&springs_part)                                              //
            .add_field_data<double>(&elem_rest_length_field, spring_rest_length)  //
            .add_field_data<double>(&elem_spring_constant_field, spring_constant);
        ++element_count;
      }
    }

    // Declare the entities
    dec_helper.check_consistency(bulk_data);
    bulk_data.modification_begin();
    dec_helper.declare_entities(bulk_data);
    bulk_data.modification_end();

    // Balance the mesh
    // stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);

    // Get the NGP stuff
    stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    auto &ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
    auto &ngp_node_displacement_since_last_rebuild_field =
        stk::mesh::get_updated_ngp_field<double>(node_displacement_since_last_rebuild_field);
    auto &ngp_node_force_field = stk::mesh::get_updated_ngp_field<double>(node_force_field);
    auto &ngp_node_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_velocity_field);
    auto &ngp_node_collision_force_field = stk::mesh::get_updated_ngp_field<double>(node_collision_force_field);
    auto &ngp_node_collision_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_collision_velocity_field);
    auto &ngp_elem_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_radius_field);
    auto &ngp_elem_radius_dot_field = stk::mesh::get_updated_ngp_field<double>(elem_radius_dot_field);
    auto &ngp_elem_rest_length_field = stk::mesh::get_updated_ngp_field<double>(elem_rest_length_field);
    auto &ngp_elem_spring_constant_field = stk::mesh::get_updated_ngp_field<double>(elem_spring_constant_field);

    // Collision constraint memory
    size_t num_neighbor_pairs;
    Kokkos::View<double *, DeviceMemorySpace> signed_sep_dist;
    Kokkos::View<double **, DeviceMemorySpace> con_normal_ij;
    Kokkos::View<double *, DeviceMemorySpace> lagrange_multipliers;

    // Timeloop
    bool rebuild_neighbors = true;
    ResultViewType search_results;
    LocalResultViewType local_search_results;
    SearchSpheresViewType search_spheres;
    Kokkos::Timer tps_timer;
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
      if (time_step % io_frequency == 0) {
        std::cout << "Time step: " << time_step << " running at "
                  << static_cast<double>(io_frequency) / tps_timer.seconds() << " tps "
                  << " | " << tps_timer.seconds() / static_cast<double>(io_frequency) << " spt" << std::endl;
        tps_timer.reset();
        // Comm fields to host
        ngp_node_coords_field.sync_to_host();
        ngp_node_force_field.sync_to_host();
        ngp_node_velocity_field.sync_to_host();
        ngp_elem_radius_field.sync_to_host();
        ngp_elem_radius_dot_field.sync_to_host();
        ngp_elem_rest_length_field.sync_to_host();
        ngp_elem_spring_constant_field.sync_to_host();

        // Write to file using Paraview compatable naming
        stk::io::write_mesh_with_fields("growing_network.e-s." + std::to_string(time_step), bulk_data, time_step + 1,
                                        time_step * time_step_size, stk::io::WRITE_RESULTS);
      }

      // Divide before updating the neighbor list
      bool division_occured = divide_large_spheres(
          bulk_data, spheres_part, springs_part, sphere_division_radius, sphere_growth_rate, spring_rest_length,
          spring_constant, node_coords_field, node_force_field, node_velocity_field, node_collision_force_field,
          node_collision_velocity_field, elem_radius_field, elem_radius_dot_field, elem_rest_length_field,
          elem_spring_constant_field);
      if (division_occured) {
        rebuild_neighbors = true;
        std::cout << "Divided spheres" << std::endl;
      }

      // Update the displacement since the last rebuild
      ngp_node_displacement_since_last_rebuild_field.sync_to_device();
      ngp_node_velocity_field.sync_to_device();
      stk::mesh::for_each_entity_run(
          ngp_mesh, stk::topology::NODE_RANK,
          stk::mesh::Selector(*ngp_node_displacement_since_last_rebuild_field.get_field_base()),
          KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
            ngp_node_displacement_since_last_rebuild_field(node_index, 0) +=
                time_step_size * ngp_node_velocity_field(node_index, 0);
            ngp_node_displacement_since_last_rebuild_field(node_index, 1) +=
                time_step_size * ngp_node_velocity_field(node_index, 1);
            ngp_node_displacement_since_last_rebuild_field(node_index, 2) +=
                time_step_size * ngp_node_velocity_field(node_index, 2);
          });
      ngp_node_displacement_since_last_rebuild_field.modify_on_device();
      double max_disp = get_max_speed(ngp_mesh, ngp_node_displacement_since_last_rebuild_field);
      if (max_disp > search_buffer) {
        rebuild_neighbors = true;
      }

      if (rebuild_neighbors) {
        std::cout << "Rebuilding neighbors" << std::endl;
        Kokkos::Timer create_search_timer;
        search_spheres = create_search_spheres(bulk_data, ngp_mesh, search_buffer, spheres_part, ngp_node_coords_field,
                                               ngp_elem_radius_field);
        std::cout << "Create search spheres time: " << create_search_timer.seconds() << std::endl;

        Kokkos::Timer search_timer;
        stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;

        // WARNING: auto_swap_domain_and_range must be true to avoid double counting forces.
        const bool results_parallel_symmetry = true;   // create source -> target and target -> source pairs
        const bool auto_swap_domain_and_range = true;  // swap source and target if target is owned and source is not
        stk::search::coarse_search(search_spheres, search_spheres, search_method, bulk_data.parallel(), search_results,
                                   DeviceExecutionSpace{}, results_parallel_symmetry, auto_swap_domain_and_range);
        num_neighbor_pairs = search_results.extent(0);
        std::cout << "Search time: " << search_timer.seconds() << " with " << num_neighbor_pairs << " results."
                  << std::endl;

        // Ghost the non-owned spheres
        Kokkos::Timer ghost_timer;
        ghost_neighbors(bulk_data, search_results);
        std::cout << "Ghost time: " << ghost_timer.seconds() << std::endl;

        // Create local neighbor indices
        Kokkos::Timer local_index_conversion_timer;
        local_search_results = get_local_neighbor_indices(bulk_data, stk::topology::ELEMENT_RANK, search_results);
        std::cout << "Local index conversion time: " << local_index_conversion_timer.seconds() << std::endl;

        // Only resize the collision views if the number of neighbor pairs has changed
        // Otherwise we can reuse the previous lagrange multipliers as the initial guess
        signed_sep_dist = Kokkos::View<double *, DeviceMemorySpace>("signed_sep_dist", num_neighbor_pairs);
        con_normal_ij = Kokkos::View<double **, DeviceMemorySpace>("con_normal_ij", num_neighbor_pairs, 3);
        lagrange_multipliers = Kokkos::View<double *, DeviceMemorySpace>("lagrange_multipliers", num_neighbor_pairs);
        Kokkos::deep_copy(lagrange_multipliers, 0.0);  // initial guess

        // Reset the accumulated displacements and the rebuild flag
        ngp_node_displacement_since_last_rebuild_field.sync_to_device();
        ngp_node_displacement_since_last_rebuild_field.set_all(ngp_mesh, 0.0);
        ngp_node_displacement_since_last_rebuild_field.modify_on_device();
        rebuild_neighbors = false;
      }

      // Zero out the forces and velocities
      ngp_node_force_field.sync_to_device();
      ngp_node_velocity_field.sync_to_device();
      ngp_node_collision_force_field.sync_to_device();
      ngp_node_collision_velocity_field.sync_to_device();

      ngp_node_force_field.set_all(ngp_mesh, 0.0);
      ngp_node_velocity_field.set_all(ngp_mesh, 0.0);
      ngp_node_collision_force_field.set_all(ngp_mesh, 0.0);
      ngp_node_collision_velocity_field.set_all(ngp_mesh, 0.0);

      ngp_node_force_field.modify_on_device();
      ngp_node_velocity_field.modify_on_device();
      ngp_node_collision_force_field.modify_on_device();
      ngp_node_collision_velocity_field.modify_on_device();

      // Compute the spring forces
      compute_spring_forces(ngp_mesh, springs_part, ngp_node_coords_field, ngp_node_force_field,
                            ngp_elem_spring_constant_field, ngp_elem_rest_length_field);
      compute_the_mobility_problem(ngp_mesh, viscosity, ngp_elem_radius_field, ngp_node_force_field,
                                   ngp_node_velocity_field);

      // Initialize the constraints
      Kokkos::Timer init_constraints_timer;
      compute_signed_separation_distance_and_contact_normal(ngp_mesh, local_search_results, ngp_node_coords_field,
                                                            ngp_elem_radius_field, signed_sep_dist, con_normal_ij);
      std::cout << "Init constraints time: " << init_constraints_timer.seconds() << std::endl;

      Kokkos::Timer contact_timer;
      Kokkos::deep_copy(lagrange_multipliers, 0.0);  // initial guess
      CollisionResult result =
          resolve_collisions(ngp_mesh, viscosity, time_step_size, max_allowable_overlap, max_col_iterations,
                             local_search_results, ngp_elem_radius_field, ngp_elem_radius_dot_field,
                             ngp_node_force_field, ngp_node_velocity_field, ngp_node_collision_force_field,
                             ngp_node_collision_velocity_field, signed_sep_dist, con_normal_ij, lagrange_multipliers);

      std::cout << std::setprecision(8) << "Contact time: " << contact_timer.seconds() << std::endl;

      std::cout << "Result: " << std::endl;
      std::cout << "  Max abs projected sep: " << result.max_abs_projected_sep << std::endl;
      std::cout << "  Number of iterations: " << result.ite_count << std::endl;
      std::cout << "  Max displacement: " << result.max_displacement << std::endl;

      // Take an Euler step
      Kokkos::Timer update_timer;
      update_sphere_positions(ngp_mesh, time_step_size, spheres_part, ngp_node_coords_field, ngp_node_velocity_field,
                              ngp_elem_radius_field, ngp_elem_radius_dot_field);
      std::cout << "Update time: " << update_timer.seconds() << std::endl;

      // Check for overlap
      node_coords_field.sync_to_host();
      elem_radius_field.sync_to_host();
      check_overlap(bulk_data, max_allowable_overlap, spheres_part, node_coords_field, elem_radius_field);
    }
  }

  std::cout << "TODO(palmerb4): We need to update the fields of ghosted entities." << std::endl;

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
