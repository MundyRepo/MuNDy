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

// STK
#include <stk_io/StkMeshIoBroker.hpp>
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
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_search/BoxIdent.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/Point.hpp>
#include <stk_search/SearchMethod.hpp>
#include <stk_search/Sphere.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

using ExecSpace = stk::ngp::ExecSpace;
using IdentProc = stk::search::IdentProc<stk::mesh::EntityId, int>;
using SphereIdentProc = stk::search::BoxIdentProc<stk::search::Sphere<double>, IdentProc>;
using Intersection = stk::search::IdentProcIntersection<IdentProc, IdentProc>;

using SearchSpheresViewType = Kokkos::View<SphereIdentProc *, ExecSpace>;
using ResultViewType = Kokkos::View<Intersection *, ExecSpace>;

using FastMeshIndicesViewType = Kokkos::View<stk::mesh::FastMeshIndex *, ExecSpace>;

void generate_particles(stk::mesh::BulkData &bulk_data, const size_t num_particles_global,
                        const stk::mesh::Part &particle_part) {
  // get the avenge number of particles per process
  size_t num_particles_local = num_particles_global / bulk_data.parallel_size();

  // num_particles_local isn't guaranteed to divide perfectly
  // add the extra workload to the first r ranks
  size_t remaining_particles = num_particles_global - num_particles_local * bulk_data.parallel_size();
  if (bulk_data.parallel_rank() < remaining_particles) {
    num_particles_local += 1;
  }

  bulk_data.modification_begin();

  std::vector<size_t> requests(metaData.entity_rank_count(), 0);
  const size_t num_nodes_requested = num_particles_local * particle_top.num_nodes();
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
  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, spheres, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::EntityId sphere_id = ngp_mesh.identifier(sphere);
        openrand::Philox rng(sphere_id, 0);

        // Random radius
        radius(sphere_index) = rng.rand<double>() * (radius_max - radius_min) + radius_min;

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

// Create local entities on host and copy to device
FastMeshIndicesViewType get_local_entity_indices(stk::mesh::EntityRank rank, stk::mesh::Selector selector) {
  std::vector<stk::mesh::Entity> local_entities;
  stk::mesh::get_entities(*m_bulk_data, rank, selector, local_entities);

  FastMeshIndicesViewType mesh_indices("mesh_indices", local_entities.size());
  FastMeshIndicesViewType::HostMirror host_mesh_indices =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, mesh_indices);

  for (size_t i = 0; i < local_entities.size(); ++i) {
    const stk::mesh::MeshIndex &mesh_index = m_bulk_data->mesh_index(local_entities[i]);
    host_mesh_indices(i) = stk::mesh::FastMeshIndex{mesh_index.bucket->bucket_id(), mesh_index.bucket_ordinal};
  }

  Kokkos::deep_copy(mesh_indices, host_mesh_indices);
  return mesh_indices;
}

SearchSpheresViewType create_search_spheres(const stk::mesh::BulkData &bulk_data, const stk::mesh::NgpMesh &ngp_mesh,
                                            const stk::mesh::Selector &spheres,
                                            const stk::mesh::NgpField<double> &node_coordinates,
                                            const stk::mesh::NgpField<double> &element_radius) {
  auto locally_owned_spheres = spheres & bulk_data.mesh_meta_data().locally_owned_part();
  const unsigned num_local_spheres =
      stk::mesh::count_entities(bulk_data, stk::topology::ELEMENT_RANK, locally_owned_spheres);
  SearchSpheresViewType search_spheres("search_spheres", num_local_spheres);

  // Slow host operation that is needed to get an index. There is plans to add this to the stk::mesh::NgpMesh.
  FastMeshIndicesViewType sphere_indices = get_local_entity_indices(stk::topology::ELEMENT_RANK, locally_owned_spheres);
  const int my_rank = bulk_data.parallel_rank();

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, num_local_spheres), KOKKOS_LAMBDA(const unsigned &i) {
        stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEMENT_RANK, sphere_indices(i));
        stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);

        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);
        stk::mesh::EntityFieldData<double> node_coords = node_coordinates(node_index);

        stk::search::Point<double> center(node_coords[0], node_coords[1], node_coords[2]);
        double radius = element_radius(sphere_index, 0);
        search_spheres(i) = SphereIdentProc{stk::search::Sphere<double>(center, radius),
                                            IdentProc(ngp_mesh.identifier(sphere), my_rank)};
      });

  return search_spheres;
}

void apply_hertzian_contact_between_spheres(stk::mesh::NgpMesh &ngp_mesh, const ResultViewType &search_results,
                                            const double youngs_modulus, const double poisson_ratio,
                                            const stk::mesh::NgpField<double> &node_coordinates,
                                            stk::mesh::NgpField<double> &node_forces,
                                            const stk::mesh::NgpField<double> &element_radius) {
  const double effective_youngs_modulus =
      (youngs_modulus * youngs_modulus) / (youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio +
                                           youngs_modulus - youngs_modulus * poisson_ratio * poisson_ratio);
  const double four_thirds = 4.0 / 3.0;

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, search_results.size()), KOKKOS_LAMBDA(const unsigned &i) {
        const auto search_result = search_results(i);

        stk::mesh::Entity source_entity =
            ngp_mesh.get_entity(stk::topology::ELEM_RANK, search_result.domainIdentProc.id());
        stk::mesh::Entity target_entity =
            ngp_mesh.get_entity(stk::topology::ELEM_RANK, search_result.rangeIdentProc.id());
        stk::mesh::Entity source_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_entity)[0];
        stk::mesh::Entity target_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_entity)[0];

        stk::mesh::EntityFieldData<double> source_coords = node_coordinates(source_node);
        stk::mesh::EntityFieldData<double> target_coords = node_coordinates(target_node);
        stk::mesh::EntityFieldData<double> source_radius = element_radius(source_entity);
        stk::mesh::EntityFieldData<double> target_radius = element_radius(target_entity);
        stk::mesh::EntityFieldData<double> source_forces = node_forces(source_node);
        stk::mesh::EntityFieldData<double> target_forces = node_forces(target_node);

        const double source_to_target_x = source_coords[0] - target_coords[0];
        const double source_to_target_y = source_coords[1] - target_coords[1];
        const double source_to_target_z = source_coords[2] - target_coords[2];

        const double distance_between_centers =
            Kokkos::sqrt(source_to_target_x * source_to_target_x + source_to_target_y * source_to_target_y +
                         source_to_target_z * source_to_target_z);

        const double signed_separation_distance = distance_between_centers - source_radius[0] - target_radius[0];

        if (signed_separation_distance < 0.0) {
          const double effective_radius = (source_radius[0] * target_radius[0]) / (source_radius[0] + target_radius[0]);
          const double normal_force_magnitude_scaled =
              four_thirds * effective_youngs_modulus * Kokkos::sqrt(effective_radius) *
              Kokkos::pow(-signed_separation_distance, 1.5) / distance_between_centers;
          source_forces[0] += normal_force_magnitude_scaled * source_to_target_x;
          source_forces[1] += normal_force_magnitude_scaled * source_to_target_y;
          source_forces[2] += normal_force_magnitude_scaled * source_to_target_z;
          target_forces[0] -= normal_force_magnitude_scaled * source_to_target_x;
          target_forces[1] -= normal_force_magnitude_scaled * source_to_target_y;
          target_forces[2] -= normal_force_magnitude_scaled * source_to_target_z;
        }
      });
  node_forces.modify_on_device();
}

void update_sphere_positions(stk::mesh::NgpMesh &ngp_mesh, const double time_step,
                             stk::mesh::NgpField<double> &node_coordinates,
                             stk::mesh::NgpField<double> &node_velocity) {
  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::NODE_RANK, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &node_index) {
        stk::mesh::Entity node = ngp_mesh.get_entity(stk::topology::NODE_RANK, node_index);
        stk::mesh::EntityFieldData<double> node_coords = node_coordinates(node_index);
        stk::mesh::EntityFieldData<double> node_vel = node_velocity(node_index);

        node_coords[0] += time_step * node_vel[0];
        node_coords[1] += time_step * node_vel[1];
        node_coords[2] += time_step * node_vel[2];
      });

  node_coordinates.modify_on_device();
}

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

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    // Simulation of N spheres in a cube
    const double viscosity = 1.0;
    const double youngs_modulus = 10000.0;
    const double poisson_ratio = 0.3;
    const double sphere_radius_min = 1.0;
    const double sphere_radius_max = 3.0;
    const double num_spheres = 10000;
    const Kokkos::Array<double, 3> unit_cell_bottom_left = {0.0, 0.0, 0.0};
    const Kokkos::Array<double, 3> unit_cell_top_right = {10.0, 10.0, 10.0};
    const double time_step = 0.00001;
    const size_t num_time_steps = 1000 / time_step;
    const size_t io_frequency = 0.001 / time_step;

    // Setup the STK mesh
    stk::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD)
        .set_spatial_dimension(3)
        .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
    meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition to
                                         // all fields are simple.
    meta_data_ptr->set_coordinate_field_name("COORDS");
    std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
    stk::mesh::MetaData &meta_data = *meta_data_ptr;
    stk::mesh::BulkData &bulk_data = *bulk_data_ptr;

    // Create the spheres
    auto &spheres_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
    stk::io::put_io_part_attribute(spheres_part);
    auto &node_coordinates = meta_data.declare_field<double>(stk::topology::NODE_RANK, "COORDS");
    auto &node_force = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
    auto &node_velocity = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");
    auto &element_radius = meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "RADIUS");

    // Create the sphere-sphere linkers
    auto &sphere_sphere_linkers_part = meta_data.declare_part("SPHERE_SPHERE_LINKERS", stk::topology::CONSTRAINT);
    auto &linked_entities_field = meta_data.declare_field<double>(stk::topology::CONSTRAINT_RANK, "LINKED_ENTITIES");
    auto &linked_entity_owners_field =
        meta_data.declare_field<int>(stk::topology::CONSTRAINT_RANK, "LINKED_ENTITY_OWNERS");

    // Assign fields to parts
    stk::mesh::put_field_on_mesh(node_coordinates, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_force, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(node_velocity, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(element_radius, spheres_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(linked_entities_field, sphere_sphere_linkers_part, 2, nullptr);
    stk::mesh::put_field_on_mesh(linked_entity_owners_field, sphere_sphere_linkers_part, 1, nullptr);

    // Concretize the mesh
    meta_data.commit();

    // Generate the particles
    generate_particles(bulk_data, num_spheres, spheres_part);

    // Balance the mesh
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);

    // Get the NGP stuff
    stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
    auto &ngp_node_coordinates = stk::mesh::get_updated_ngp_field<double>(node_coordinates);
    auto &ngp_node_force = stk::mesh::get_updated_ngp_field<double>(node_force);
    auto &ngp_node_velocity = stk::mesh::get_updated_ngp_field<double>(node_velocity);
    auto &ngp_element_radius = stk::mesh::get_updated_ngp_field<double>(element_radius);

    // Randomize the positions and radii
    randomize_positions_and_radii(ngp_mesh, spheres_part, sphere_radius_min, sphere_radius_max, unit_cell_bottom_left,
                                  unit_cell_top_right, ngp_node_coordinates, ngp_element_radius);

    // Timeloop
    bool rebuild_neighbors = true;
    ResultViewType search_results;
    for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
      if (time_step % io_frequency == 0) {
        std::cout << "Time step: " << time_step << std::endl;
        // Comm fields to host
        ngp_node_coordinates.sync_to_host();
        ngp_node_force.sync_to_host();
        ngp_node_velocity.sync_to_host();
        ngp_element_radius.sync_to_host();

        // Write to file
        stk::io::write_mesh("spheres_" + std::to_string(time_step) + ".e", bulk_data, stk::io::WRITE_RESULTS);
      }

      if (rebuild_neighbors) {
        SearchSpheresViewType search_spheres =
            create_search_spheres(bulk_data, ngp_mesh, spheres_part, ngp_node_coordinates, ngp_element_radius);

        stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;
        stk::ngp::ExecSpace exec_space = Kokkos::DefaultExecutionSpace{};
        const bool results_parallel_symmetry = false;  // Will create source -> target and target -> source pairs
        stk::search::coarse_search(search_spheres, search_spheres, search_method, bulk_data.parallel(), search_results,
                                   exec_space, results_parallel_symmetry);
      }

      apply_hertzian_contact_between_spheres(ngp_mesh, search_results, youngs_modulus, poisson_ratio,
                                             ngp_node_coordinates, ngp_node_force, ngp_element_radius);

      update_sphere_positions(ngp_mesh, time_step, ngp_node_coordinates, ngp_node_velocity);
    }
  }

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
