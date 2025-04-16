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
#include <fstream>   // for std::ofstream
#include <iostream>  // for std::cout, std::endl
#include <numeric>   // for std::accumulate
#include <vector>    // for std::vector

// Trilinos libs
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

#if TRILINOS_MAJOR_MINOR_VERSION >= 160000

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

using ExecSpace = Kokkos::DefaultExecutionSpace;
using IdentProc = stk::search::IdentProc<stk::mesh::EntityId, int>;
using SphereIdentProc = stk::search::BoxIdentProc<stk::search::Sphere<double>, IdentProc>;
using Intersection = stk::search::IdentProcIntersection<IdentProc, IdentProc>;
using SearchSpheresViewType = Kokkos::View<SphereIdentProc *, ExecSpace>;
using ResultViewType = Kokkos::View<Intersection *, ExecSpace>;
using FastMeshIndicesViewType = Kokkos::View<stk::mesh::FastMeshIndex *, ExecSpace>;

using LocalIdentProc = stk::search::IdentProc<stk::mesh::FastMeshIndex, int>;
using LocalIntersection = stk::search::IdentProcIntersection<LocalIdentProc, LocalIdentProc>;
using LocalResultViewType = Kokkos::View<LocalIntersection *, ExecSpace>;

//! \name Setup
//@{

void generate_particles(stk::mesh::BulkData &bulk_data, const size_t num_particles_global,
                        stk::mesh::Part &particle_part) {
  // get the avenge number of particles per process
  size_t num_particles_local = num_particles_global / bulk_data.parallel_size();

  // num_particles_local isn't guaranteed to divide perfectly
  // add the extra workload to the first r ranks
  size_t remaining_particles = num_particles_global - num_particles_local * bulk_data.parallel_size();
  if (bulk_data.parallel_rank() < remaining_particles) {
    num_particles_local += 1;
  }

  bulk_data.modification_begin();

  std::vector<size_t> requests(bulk_data.mesh_meta_data().entity_rank_count(), 0);
  const size_t num_nodes_requested = num_particles_local;
  const size_t num_elems_requested = num_particles_local;
  requests[stk::topology::NODE_RANK] = num_nodes_requested;
  requests[stk::topology::ELEMENT_RANK] = num_elems_requested;

  // ex.
  //  requests = { 0, 4,  8}
  //  requests 0 entites of rank 0, 4 entites of rank 1, and 8 entites of rank 2
  //  requested_entities = {0 entites of rank 0, 4 entites of rank 1, 8 entites of rank 2}
  std::vector<stk::mesh::Entity> requested_entities;
  bulk_data.generate_new_entities(requests, requested_entities);

  // associate each particle with a single part
  std::vector<stk::mesh::Part *> add_particle_part(1);
  add_particle_part[0] = &particle_part;

  // set topologies of new entities
  for (int i = 0; i < num_particles_local; i++) {
    stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
    bulk_data.change_entity_parts(particle_i, add_particle_part);
  }

  // the elements should be associated with a topology before they are connected to their nodes/edges
  // set downward relations of entities
  for (int i = 0; i < num_particles_local; i++) {
    stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
    bulk_data.declare_relation(particle_i, requested_entities[i], 0);
  }

  bulk_data.modification_end();
}

void randomize_positions_and_radii(stk::mesh::NgpMesh &ngp_mesh, const stk::mesh::Selector &spheres,
                                   const double radius_min, const double radius_max,
                                   const Kokkos::Array<double, 3> &bottom_left,
                                   const Kokkos::Array<double, 3> &top_right,
                                   stk::mesh::NgpField<double> &node_coordinates,
                                   stk::mesh::NgpField<double> &element_radius) {
  node_coordinates.sync_to_device();
  element_radius.sync_to_device();

  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, spheres, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::EntityId sphere_id = ngp_mesh.identifier(sphere);
        openrand::Philox rng(sphere_id, 0);

        // Random radius
        element_radius(sphere_index, 0) = rng.rand<double>() * (radius_max - radius_min) + radius_min;

        // Random position
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);
        stk::mesh::EntityFieldData<double> node_coords = node_coordinates(node_index);
        node_coords[0] = rng.rand<double>() * (top_right[0] - bottom_left[0]) + bottom_left[0];
        node_coords[1] = rng.rand<double>() * (top_right[1] - bottom_left[1]) + bottom_left[1];
        node_coords[2] = rng.rand<double>() * (top_right[2] - bottom_left[2]) + bottom_left[2];
      });

  node_coordinates.modify_on_device();
  element_radius.modify_on_device();
}
//@}

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
                                            stk::mesh::NgpField<double> &node_coords_field,
                                            stk::mesh::NgpField<double> &elem_radius_field) {
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();

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
        stk::mesh::EntityFieldData<double> node_coords = node_coords_field(node_index);

        stk::search::Point<double> center(node_coords[0], node_coords[1], node_coords[2]);
        double search_radius = elem_radius_field(sphere_index, 0) + search_buffer;
        search_spheres(i) = SphereIdentProc{stk::search::Sphere<double>(center, search_radius),
                                            IdentProc(ngp_mesh.identifier(sphere), my_rank)};
      });

  return search_spheres;
}

void ghost_neighbors(stk::mesh::BulkData &bulk_data, const ResultViewType &search_results) {
  if (bulk_data.parallel_size() == 1) {
    return;
  }

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

void apply_hertzian_contact_between_spheres(stk::mesh::NgpMesh &ngp_mesh,
                                            const LocalResultViewType &local_search_results,
                                            const double youngs_modulus, const double poisson_ratio,
                                            const stk::mesh::NgpField<double> &node_coordinates,
                                            stk::mesh::NgpField<double> &node_forces,
                                            stk::mesh::NgpField<double> &element_radius) {
  node_forces.sync_to_device();
  element_radius.sync_to_device();

  const double effective_youngs_modulus =
      (youngs_modulus * youngs_modulus) / (youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio +
                                           youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio);
  const double four_thirds = 4.0 / 3.0;

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, local_search_results.size()), KOKKOS_LAMBDA(const unsigned &i) {
        const auto search_result = local_search_results(i);
        stk::mesh::FastMeshIndex source_entity_index = search_result.domainIdentProc.id();
        stk::mesh::FastMeshIndex target_entity_index = search_result.rangeIdentProc.id();
        if (source_entity_index == target_entity_index) {
          return;
        }

        stk::mesh::Entity source_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_entity_index)[0];
        stk::mesh::Entity target_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_entity_index)[0];

        stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_node);
        stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_node);

        stk::mesh::EntityFieldData<double> source_coords = node_coordinates(source_node_index);
        stk::mesh::EntityFieldData<double> target_coords = node_coordinates(target_node_index);
        stk::mesh::EntityFieldData<double> source_radius = element_radius(source_entity_index);
        stk::mesh::EntityFieldData<double> target_radius = element_radius(target_entity_index);
        stk::mesh::EntityFieldData<double> source_forces = node_forces(source_node_index);
        stk::mesh::EntityFieldData<double> target_forces = node_forces(target_node_index);

        const double source_to_target_x = target_coords[0] - source_coords[0];
        const double source_to_target_y = target_coords[1] - source_coords[1];
        const double source_to_target_z = target_coords[2] - source_coords[2];

        const double distance_between_centers =
            Kokkos::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                         source_to_target_z * source_to_target_z);

        const double signed_separation_distance = distance_between_centers - source_radius[0] - target_radius[0];

        if (signed_separation_distance < 0.0) {
          const double effective_radius = (source_radius[0] * target_radius[0]) / (source_radius[0] + target_radius[0]);
          const double normal_force_magnitude_scaled =
              four_thirds * effective_youngs_modulus * Kokkos::sqrt(effective_radius) *
              Kokkos::pow(-signed_separation_distance, 1.5) / distance_between_centers;
          source_forces[0] -= normal_force_magnitude_scaled * source_to_target_x;
          source_forces[1] -= normal_force_magnitude_scaled * source_to_target_y;
          source_forces[2] -= normal_force_magnitude_scaled * source_to_target_z;
          target_forces[0] += normal_force_magnitude_scaled * source_to_target_x;
          target_forces[1] += normal_force_magnitude_scaled * source_to_target_y;
          target_forces[2] += normal_force_magnitude_scaled * source_to_target_z;
        }
      });

  node_forces.modify_on_device();
}

void apply_attractive_abc_flow(stk::mesh::NgpMesh &ngp_mesh, stk::mesh::NgpField<double> &node_coordinates,
                               stk::mesh::NgpField<double> &node_forces) {
  node_coordinates.sync_to_device();
  node_forces.sync_to_device();

  const double a = 1.0;
  const double b = Kokkos::sqrt(2.0);
  const double c = Kokkos::sqrt(3.0);
  const double attraction_coeff = 2.0;

  node_coordinates.sync_to_device();
  auto selector = stk::mesh::Selector(*node_coordinates.get_field_base());
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        stk::mesh::EntityFieldData<double> node_coords = node_coordinates(node_index);
        stk::mesh::EntityFieldData<double> node_force = node_forces(node_index);

        const double coords_x = node_coords[0];
        const double coords_y = node_coords[1];
        const double coords_z = node_coords[2];
        const double r = Kokkos::sqrt(coords_x * coords_x + coords_y * coords_y + coords_z * coords_z);
        const double inv_r = 1.0 / r;

        node_force[0] += a * Kokkos::sin(coords_z) + c * Kokkos::cos(coords_y) - attraction_coeff * coords_x * inv_r;
        node_force[1] += b * Kokkos::sin(coords_x) + a * Kokkos::cos(coords_z) - attraction_coeff * coords_y * inv_r;
        node_force[2] += c * Kokkos::sin(coords_y) + b * Kokkos::cos(coords_x) - attraction_coeff * coords_z * inv_r;
      });

  node_forces.modify_on_device();
}

void update_sphere_positions(const stk::mesh::NgpMesh &ngp_mesh, const double time_step_size,
                             stk::mesh::NgpField<double> &node_coordinates,
                             stk::mesh::NgpField<double> &node_velocity) {
  node_coordinates.sync_to_device();
  node_velocity.sync_to_device();

  auto selector = stk::mesh::Selector(*node_coordinates.get_field_base());
  stk::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        stk::mesh::EntityFieldData<double> node_coords = node_coordinates(node_index);
        stk::mesh::EntityFieldData<double> node_vel = node_velocity(node_index);

        node_coords[0] += time_step_size * node_vel[0];
        node_coords[1] += time_step_size * node_vel[1];
        node_coords[2] += time_step_size * node_vel[2];
      });

  node_coordinates.modify_on_device();
}

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
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  {
    // Simulation of N spheres in a cube
    const double youngs_modulus = 100.0;
    const double poisson_ratio = 0.3;
    const double sphere_radius_min = 1.0;
    const double sphere_radius_max = 1.0;
    const double num_spheres = 100000;
    const double viscosity = 1.0 / (6.0 * Kokkos::numbers::pi * sphere_radius_max);

    const Kokkos::Array<double, 3> unit_cell_bottom_left = {-50.0, -50.0, -50.0};
    const Kokkos::Array<double, 3> unit_cell_top_right = {50.0, 50.0, 50.0};
    const double time_step_size = 0.00001;
    const size_t num_time_steps = 1000 / time_step_size;
    const size_t io_frequency = std::round(0.1 / time_step_size);
    const double search_buffer = sphere_radius_max;

    const double volume_fraction =
        4.0 / 3.0 * M_PI * sphere_radius_max * sphere_radius_max * sphere_radius_max * num_spheres /
        ((unit_cell_top_right[0] - unit_cell_bottom_left[0]) * (unit_cell_top_right[1] - unit_cell_bottom_left[1]) *
         (unit_cell_top_right[2] - unit_cell_bottom_left[2]));
    std::cout << "Setup: " << std::endl;
    std::cout << "  Number of spheres: " << num_spheres << std::endl;
    std::cout << "  Sphere radius min: " << sphere_radius_min << std::endl;
    std::cout << "  Sphere radius max: " << sphere_radius_max << std::endl;
    std::cout << "  Volume fraction: " << volume_fraction << std::endl;
    std::cout << "  IO frequency: " << io_frequency << std::endl;

    // Setup the STK mesh
    stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
    mesh_builder.set_spatial_dimension(3);
    mesh_builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
    meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition to
                                         // all fields are simple.
    meta_data_ptr->set_coordinate_field_name("NODE_COORDS");
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create(meta_data_ptr);
    stk::mesh::MetaData &meta_data = *meta_data_ptr;
    stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

    // Create the spheres
    auto &spheres_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
    stk::io::put_io_part_attribute(spheres_part);
    auto &node_coordinates = meta_data.declare_field<double>(stk::topology::NODE_RANK, "NODE_COORDS");
    auto &node_displacement_since_last_rebuild_field =
        meta_data.declare_field<double>(stk::topology::NODE_RANK, "OUR_DISP");
    auto &node_force = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
    auto &node_velocity = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");
    auto &element_radius = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "RADIUS");

    // Assign fields to parts
    double zero3[3] = {0.0, 0.0, 0.0};
    stk::mesh::put_field_on_mesh(node_coordinates, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_displacement_since_last_rebuild_field, spheres_part, 3, zero3);
    stk::mesh::put_field_on_mesh(node_force, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(node_velocity, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(element_radius, spheres_part, 1, nullptr);

    // Concretize the mesh
    meta_data.commit();

    // Get the NGP stuff
    stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    auto &ngp_node_coordinates = stk::mesh::get_updated_ngp_field<double>(node_coordinates);
    auto &ngp_node_displacement_since_last_rebuild_field =
        stk::mesh::get_updated_ngp_field<double>(node_displacement_since_last_rebuild_field);
    auto &ngp_node_force = stk::mesh::get_updated_ngp_field<double>(node_force);
    auto &ngp_node_velocity = stk::mesh::get_updated_ngp_field<double>(node_velocity);
    auto &ngp_element_radius = stk::mesh::get_updated_ngp_field<double>(element_radius);

    // Generate the particles
    generate_particles(bulk_data, num_spheres, spheres_part);

    // Randomize the positions and radii
    randomize_positions_and_radii(ngp_mesh, spheres_part, sphere_radius_min, sphere_radius_max, unit_cell_bottom_left,
                                  unit_cell_top_right, ngp_node_coordinates, ngp_element_radius);

    // Balance the mesh
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);

    // Timeloop
    bool rebuild_neighbors = true;
    ResultViewType search_results;
    LocalResultViewType local_search_results;
    SearchSpheresViewType search_spheres;

    Kokkos::Timer overall_timer;
    Kokkos::Timer tps_timer;
    for (size_t time_step_index = 0; time_step_index < num_time_steps; ++time_step_index) {
      if (time_step_index % io_frequency == 0) {
        std::cout << "Time step: " << time_step_index << " | Total time: " << time_step_index * time_step_size
                  << std::endl;
        std::cout << "  Time elapsed: " << overall_timer.seconds() << " s" << std::endl;
        std::cout << "  Running avg tps: " << static_cast<double>(io_frequency) / tps_timer.seconds() << std::endl;
        tps_timer.reset();

        // // Comm fields to host
        // ngp_node_coordinates.sync_to_host();
        // ngp_node_force.sync_to_host();
        // ngp_node_velocity.sync_to_host();
        // ngp_element_radius.sync_to_host();

        // // Write to file using Paraview compatable naming
        // stk::io::write_mesh_with_fields("hertz_spheres.e-s." + std::to_string(time_step_index), bulk_data,
        // time_step_index + 1,
        //                                 time_step_index * time_step_size, stk::io::WRITE_RESULTS);
      }

      // Update the displacement since the last rebuild
      ngp_node_displacement_since_last_rebuild_field.sync_to_device();
      ngp_node_force.sync_to_device();
      stk::mesh::for_each_entity_run(
          ngp_mesh, stk::topology::NODE_RANK,
          stk::mesh::Selector(*ngp_node_displacement_since_last_rebuild_field.get_field_base()),
          KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
            ngp_node_displacement_since_last_rebuild_field(node_index, 0) +=
                time_step_size * ngp_node_force(node_index, 0);
            ngp_node_displacement_since_last_rebuild_field(node_index, 1) +=
                time_step_size * ngp_node_force(node_index, 1);
            ngp_node_displacement_since_last_rebuild_field(node_index, 2) +=
                time_step_size * ngp_node_force(node_index, 2);
          });
      ngp_node_displacement_since_last_rebuild_field.modify_on_device();
      double max_disp = get_max_speed(ngp_mesh, ngp_node_displacement_since_last_rebuild_field);
      if (max_disp > search_buffer) {
        rebuild_neighbors = true;
      }

      if (rebuild_neighbors) {
        std::cout << "Rebuilding neighbors." << std::endl;

        Kokkos::Timer create_search_timer;
        search_spheres = create_search_spheres(bulk_data, ngp_mesh, search_buffer, spheres_part, ngp_node_coordinates,
                                               ngp_element_radius);
        // std::cout << "Create search spheres time: " << create_search_timer.seconds() << std::endl;

        Kokkos::Timer search_timer;
        stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;

        stk::ngp::ExecSpace exec_space{};
        const bool results_parallel_symmetry = true;   // create source -> target and target -> source pairs
        const bool auto_swap_domain_and_range = true;  // swap source and target if target is owned and source is not
        const bool sort_search_results = true;         // sort the search results by source id
        stk::search::coarse_search(search_spheres, search_spheres, search_method, bulk_data.parallel(), search_results,
                                   exec_space, results_parallel_symmetry);
        // std::cout << "Search time: " << search_timer.seconds() << " with " << search_results.size() << " results."
        //           << std::endl;

        // Ghost the non-owned spheres
        Kokkos::Timer ghost_timer;
        ghost_neighbors(bulk_data, search_results);
        // std::cout << "Ghost time: " << ghost_timer.seconds() << std::endl;

        // Create local neighbor indices
        Kokkos::Timer local_index_conversion_timer;
        local_search_results = get_local_neighbor_indices(bulk_data, stk::topology::ELEMENT_RANK, search_results);
        // std::cout << "Local index conversion time: " << local_index_conversion_timer.seconds() << std::endl;

        // Reset the accumulated displacements and the rebuild flag
        ngp_node_displacement_since_last_rebuild_field.sync_to_device();
        ngp_node_displacement_since_last_rebuild_field.set_all(ngp_mesh, 0.0);
        ngp_node_displacement_since_last_rebuild_field.modify_on_device();
        rebuild_neighbors = false;
      }

      // Setup/reset the forces and velocities
      ngp_node_force.sync_to_device();
      ngp_node_velocity.sync_to_device();
      ngp_node_force.set_all(ngp_mesh, 0.0);
      ngp_node_velocity.set_all(ngp_mesh, 0.0);
      ngp_node_force.modify_on_device();
      ngp_node_velocity.modify_on_device();

      Kokkos::Timer contact_timer;
      apply_hertzian_contact_between_spheres(ngp_mesh, local_search_results, youngs_modulus, poisson_ratio,
                                             ngp_node_coordinates, ngp_node_force, ngp_element_radius);
      // std::cout << "Contact time: " << contact_timer.seconds() << std::endl;

      Kokkos::Timer flow_timer;
      apply_attractive_abc_flow(ngp_mesh, ngp_node_coordinates, ngp_node_force);
      // std::cout << "Flow time: " << flow_timer.seconds() << std::endl;

      // Non-dimensionalized viscous drag node_velocity = node_force
      Kokkos::Timer update_timer;
      update_sphere_positions(ngp_mesh, time_step_size, ngp_node_coordinates, ngp_node_force);
      // std::cout << "Update time: " << update_timer.seconds() << std::endl;
    }
  }

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}

#else

int main() {
  std::cout << "TEST DISABLED. Trilinos version must be at least 16.0.0." << std::endl;
  return 0;
}

#endif  // TRILINOS_MAJOR_MINOR_VERSION
