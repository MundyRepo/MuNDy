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
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <openrand/philox.h>

// C++ core
#include <algorithm>   // for std::transform
#include <filesystem>  // for std::filesystem::path
#include <fstream>     // for std::ofstream
#include <iostream>    // for std::cout, std::endl
#include <memory>      // for std::shared_ptr, std::unique_ptr
#include <numeric>     // for std::accumulate
#include <regex>       // for std::regex
#include <string>      // for std::string
#include <vector>      // for std::vector

// Kokkos and Kokkos-Kernels
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Teuchos
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList

// STK Mesh
#include <stk_mesh/base/Comm.hpp>              // for stk::mesh::comm_mesh_counts
#include <stk_mesh/base/DumpMeshInfo.hpp>      // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Entity.hpp>            // for stk::mesh::Entity
#include <stk_mesh/base/FieldParallel.hpp>     // for stk::parallel_sum
#include <stk_mesh/base/ForEachEntity.hpp>     // for mundy::mesh::for_each_entity_run
#include <stk_mesh/base/GetNgpField.hpp>       // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/GetNgpMesh.hpp>        // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpField.hpp>          // for stk::mesh::NgpField
#include <stk_mesh/base/NgpForEachEntity.hpp>  // for stk::mesh::for_each_entity_run
#include <stk_mesh/base/NgpMesh.hpp>           // for stk::mesh::NgpMesh
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Part.hpp>      // for stk::mesh::Part, stk::mesh::intersect
#include <stk_mesh/base/Selector.hpp>  // for stk::mesh::Selector

// STK Search
#include <stk_search/BoxIdent.hpp>
#include <stk_search/CoarseSearch.hpp>
#include <stk_search/IdentProc.hpp>
#include <stk_search/Point.hpp>
#include <stk_search/SearchMethod.hpp>
#include <stk_search/Sphere.hpp>

// STK Topology
#include <stk_topology/topology.hpp>  // for stk::topology

// STK Util
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// STK Balance
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh, stk::balance::BalanceSettings

// STK IO
#include <stk_io/IossBridge.hpp>  // for stk::io::set_field_role and stk::io::put_io_part_attribute
#include <stk_io/WriteMesh.hpp>   // for stk::io::write_mesh

// Mundy core
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator
#include <mundy_core/throw_assert.hpp>                         // for MUNDY_THROW_ASSERT

// Mundy math
#include <mundy_math/Hilbert.hpp>                      // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>                      // for mundy::math::Vector3
#include <mundy_math/distance/EllipsoidEllipsoid.hpp>  // for mundy::math::distance::ellipsoid_ellipsoid

// Mundy geom
#include <mundy_geom/distance.hpp>    // for mundy::geom::distance(primA, primB)
#include <mundy_geom/primitives.hpp>  // for all geometric primitives mundy::geom::Point, Line, Sphere, Ellipsoid...

// Mundy mesh
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>     // for mundy::mesh::field_fill, mundy::mesh::field_copy, etc
#include <mundy_mesh/fmt_stk_types.hpp>    // adds fmt::format for stk types
#include <mundy_mesh/utils/FillFieldWithValue.hpp>  // for mundy::mesh::utils::fill_field_with_value

// aLENS
#include <mundy_alens/periphery/Periphery.hpp>  // for gen_sphere_quadrature

using IdentProc = stk::search::IdentProc<stk::mesh::EntityId, int>;
using SphereIdentProc = stk::search::BoxIdentProc<stk::search::Sphere<double>, IdentProc>;
using Intersection = stk::search::IdentProcIntersection<IdentProc, IdentProc>;
using SearchSpheresViewType = Kokkos::View<SphereIdentProc *, stk::ngp::ExecSpace>;
using ResultViewType = Kokkos::View<Intersection *, stk::ngp::ExecSpace>;
using FastMeshIndicesViewType = Kokkos::View<stk::mesh::FastMeshIndex *, stk::ngp::ExecSpace>;

using LocalIdentProc = stk::search::IdentProc<stk::mesh::FastMeshIndex, int>;
using LocalIntersection = stk::search::IdentProcIntersection<LocalIdentProc, LocalIdentProc>;
using LocalResultViewType = Kokkos::View<LocalIntersection *, stk::ngp::ExecSpace>;

using Double1DView = Kokkos::View<double *, Kokkos::LayoutLeft, stk::ngp::MemSpace>;
using Double2DView = Kokkos::View<double **, Kokkos::LayoutLeft, stk::ngp::MemSpace>;
using DoubleMatDeviceView = Kokkos::View<double **, Kokkos::LayoutLeft, stk::ngp::MemSpace>;

namespace mundy {

namespace mesh {

void get_selected_entities(const stk::mesh::Selector &selector,              //
                           const stk::mesh::BucketVector &input_buckets,     //
                           stk::NgpVector<stk::mesh::Entity> &ngp_entities,  //
                           bool sort_by_global_id = true) {
  Kokkos::Profiling::pushRegion("mundy::mesh::get_selected_entities");
  if (input_buckets.empty()) {
    return;
  }

  // Fetch the entities on the host
  stk::mesh::EntityVector entity_vector;
  stk::mesh::get_selected_entities(selector, input_buckets, entity_vector, sort_by_global_id);

  // Fill the ngp vector on the host
  const size_t num_entities = entity_vector.size();
  ngp_entities.resize(num_entities, stk::mesh::Entity());
  for (unsigned i = 0; i < num_entities; i++) {
    ngp_entities[i] = entity_vector[i];
  }
  ngp_entities.copy_host_to_device();
  Kokkos::Profiling::popRegion();
}

// Create local entities on host and copy to device
FastMeshIndicesViewType get_local_entity_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                                 stk::mesh::Selector selector) {
  Kokkos::Profiling::pushRegion("mundy::mesh::get_local_entity_indices");
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
  Kokkos::Profiling::popRegion();
  return mesh_indices;
}

template <typename FieldDataType>
void print_field(const stk::mesh::Field<FieldDataType> &field) {
  stk::mesh::BulkData &bulk_data = field.get_mesh();
  stk::mesh::Selector selector = stk::mesh::Selector(field);

  stk::mesh::EntityVector entities;
  stk::mesh::get_selected_entities(selector, bulk_data.buckets(field.entity_rank()), entities);

  for (const stk::mesh::Entity &entity : entities) {
    const FieldDataType *field_data = stk::mesh::field_data(field, entity);
    std::cout << "Entity " << entity << " field data: ";
    const unsigned field_num_components =
        stk::mesh::field_scalars_per_entity(field, bulk_data.bucket(entity).bucket_id());
    for (size_t i = 0; i < field_num_components; ++i) {
      std::cout << field_data[i] << " ";
    }
    std::cout << std::endl;
  }
}

}  // namespace mesh

namespace mech {

//! \name Search
//@{

LocalResultViewType get_local_neighbor_indices(const stk::mesh::BulkData &bulk_data, stk::mesh::EntityRank rank,
                                               const ResultViewType &search_results) {
  Kokkos::Profiling::pushRegion("mundy::mech::get_local_neighbor_indices");

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
  Kokkos::Profiling::popRegion();
  return local_neighbor_indices;
}

SearchSpheresViewType create_search_spheres(const stk::mesh::BulkData &bulk_data,                  //
                                            const stk::mesh::NgpMesh &ngp_mesh,                    //
                                            const double search_buffer,                            //
                                            const stk::mesh::NgpField<double> &node_coords_field,  //
                                            const stk::mesh::NgpField<double> &elem_radius_field,  //
                                            const stk::mesh::Selector &spheres) {
  Kokkos::Profiling::pushRegion("mundy::mesh::create_search_spheres");
  auto locally_owned_spheres = spheres & bulk_data.mesh_meta_data().locally_owned_part();
  const unsigned num_local_spheres =
      stk::mesh::count_entities(bulk_data, stk::topology::ELEM_RANK, locally_owned_spheres);
  SearchSpheresViewType search_spheres("search_spheres", num_local_spheres);

  // Slow host operation that is needed to get an index. There is plans to add this to the stk::mesh::NgpMesh.
  FastMeshIndicesViewType sphere_indices =
      mundy::mesh::get_local_entity_indices(bulk_data, stk::topology::ELEM_RANK, locally_owned_spheres);
  const int my_rank = bulk_data.parallel_rank();

  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, num_local_spheres), KOKKOS_LAMBDA(const unsigned &i) {
        stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEM_RANK, sphere_indices(i));
        stk::mesh::FastMeshIndex sphere_index = ngp_mesh.fast_mesh_index(sphere);

        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(nodes[0]);
        stk::mesh::EntityFieldData<double> node_coords = node_coords_field(node_index);

        stk::search::Point<double> center(node_coords[0], node_coords[1], node_coords[2]);
        double radius = elem_radius_field(sphere_index, 0) + search_buffer;
        search_spheres(i) = SphereIdentProc{stk::search::Sphere<double>(center, radius),
                                            IdentProc(ngp_mesh.identifier(sphere), my_rank)};  // IDENTIFIER
      });

  Kokkos::Profiling::popRegion();
  return search_spheres;
}

void ghost_neighbors(stk::mesh::BulkData &bulk_data, const ResultViewType &search_results) {
  Kokkos::Profiling::pushRegion("mundy::mesh::ghost_neighbors");
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
  Kokkos::Profiling::popRegion();
}
//@}

//! \name Boundary Integral Equations
//@{

void check_max_overlap_with_periphery(stk::mesh::NgpMesh ngp_mesh,                         //
                                      const double &max_allowed_overlap,                   //
                                      const mundy::geom::Sphere<double> &periphery_shape,  //
                                      stk::mesh::NgpField<double> &node_coords_field,      //
                                      stk::mesh::NgpField<double> &elem_radius_field,      //
                                      const stk::mesh::Selector &selector) {
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();

  double shifted_periphery_hydro_radius = periphery_shape.radius() + max_allowed_overlap;

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::Entity node = nodes[0];
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
        const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node_index);
        const double sphere_radius = elem_radius_field(sphere_index, 0);
        const bool overlap_exceeds_threshold =
            mundy::math::norm(node_coords) + sphere_radius > shifted_periphery_hydro_radius;
        MUNDY_THROW_REQUIRE(!overlap_exceeds_threshold, std::runtime_error,
                            "Sphere overlaps with peruphery beyond max extent allowed.");
      });
}

void check_max_overlap_with_periphery(stk::mesh::NgpMesh &ngp_mesh,                           //
                                      const double &max_allowed_overlap,                      //
                                      const mundy::geom::Ellipsoid<double> &periphery_shape,  //
                                      stk::mesh::NgpField<double> &node_coords_field,         //
                                      stk::mesh::NgpField<double> &elem_radius_field,         //
                                      const stk::mesh::Selector &selector) {
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();

  double shifted_periphery_hydro_radius1 = periphery_shape.radius_1() + max_allowed_overlap;
  double shifted_periphery_hydro_radius2 = periphery_shape.radius_2() + max_allowed_overlap;
  double shifted_periphery_hydro_radius3 = periphery_shape.radius_3() + max_allowed_overlap;

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        // The following is an in-exact but cheap check.
        // If shrinks the periphery's level set by the max allowed overlap and the sphere radius and then checks
        // if the sphere's center is inside the shrunk periphery. Level sets don't follow the same rules as Euclidean
        // geometry, so this is a rough check and is not even guaranteed to be conservative.
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::Entity node = nodes[0];
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
        const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node_index);
        const double sphere_radius = elem_radius_field(sphere_index, 0);
        const double x = node_coords[0];
        const double y = node_coords[1];
        const double z = node_coords[2];
        const double x2 = x * x;
        const double y2 = y * y;
        const double z2 = z * z;
        const double a2 =
            (shifted_periphery_hydro_radius1 - sphere_radius) * (shifted_periphery_hydro_radius1 - sphere_radius);
        const double b2 =
            (shifted_periphery_hydro_radius2 - sphere_radius) * (shifted_periphery_hydro_radius2 - sphere_radius);
        const double c2 =
            (shifted_periphery_hydro_radius3 - sphere_radius) * (shifted_periphery_hydro_radius3 - sphere_radius);
        const double value = x2 / a2 + y2 / b2 + z2 / c2;
        MUNDY_THROW_REQUIRE(value <= 1.0, std::runtime_error,
                            "Sphere overlaps with periphery beyond max extent allowed.");
      });
}
//@}

//! \name Mobility
//@{

/// @brief An aggregate containing all the data needed to compute the effect of the no-slip periphery
/// \note We recommend using \ref NoSlipPeripheryBuilder to constructing this aggregate.
struct KokkosNoSlipPeripheryData {
  Double1DView surface_positions;
  Double1DView surface_normals;
  Double1DView surface_weights;
  Double1DView surface_radii;
  Double1DView surface_forces;
  Double1DView surface_velocities;
  DoubleMatDeviceView inv_self_interaction_matrix;
};

class NoSlipPeripheryBuilder {
 public:
  NoSlipPeripheryBuilder()
      : quadrature_setup_finished_(false),  //
        build_finished_(false),             //
        num_surface_nodes_(0),              //
        surface_positions_(                 //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_positions_view"), 3 * num_surface_nodes_),
        surface_normals_(  //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_normals_view"), 3 * num_surface_nodes_),
        surface_weights_(  //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_weights_view"), num_surface_nodes_),
        surface_radii_(  //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_radii_view"), num_surface_nodes_),
        surface_forces_(  //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_forces_view"), 3 * num_surface_nodes_),
        surface_velocities_(  //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_velocities_view"), 3 * num_surface_nodes_),
        inv_self_interaction_matrix_(  //
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "inv_self_interaction_matrix_view"), 3 * num_surface_nodes_,
            3 * num_surface_nodes_) {
    // Create the host mirrors
    host_surface_positions_ = Kokkos::create_mirror_view(surface_positions_);
    host_surface_normals_ = Kokkos::create_mirror_view(surface_normals_);
    host_surface_weights_ = Kokkos::create_mirror_view(surface_weights_);
    host_surface_radii_ = Kokkos::create_mirror_view(surface_radii_);
    host_surface_forces_ = Kokkos::create_mirror_view(surface_forces_);
    host_surface_velocities_ = Kokkos::create_mirror_view(surface_velocities_);
    host_inv_self_interaction_matrix_ = Kokkos::create_mirror_view(inv_self_interaction_matrix_);
  }

  NoSlipPeripheryBuilder &setup_quadrature_using_gauss_legendre(
      // const mundy::geom::Point<double> &center,  // For now, we center at zero
      const double radius, const unsigned spectral_order) {
    std::vector<double> points_vec;
    std::vector<double> weights_vec;
    std::vector<double> normals_vec;
    const bool invert = true;
    const bool include_poles = false;
    mundy::alens::periphery::gen_sphere_quadrature(spectral_order, radius, &points_vec, &weights_vec, &normals_vec,
                                                   include_poles, invert);

    // Allocate the views. Note, resizing does not automatically update the mirror views.
    num_surface_nodes_ = weights_vec.size();
    Kokkos::resize(surface_positions_, 3 * num_surface_nodes_);
    Kokkos::resize(surface_normals_, 3 * num_surface_nodes_);
    Kokkos::resize(surface_weights_, num_surface_nodes_);
    Kokkos::resize(surface_radii_, num_surface_nodes_);
    Kokkos::resize(surface_velocities_, 3 * num_surface_nodes_);
    Kokkos::resize(surface_forces_, 3 * num_surface_nodes_);
    host_surface_positions_ = Kokkos::create_mirror_view(surface_positions_);
    host_surface_normals_ = Kokkos::create_mirror_view(surface_normals_);
    host_surface_weights_ = Kokkos::create_mirror_view(surface_weights_);
    host_surface_radii_ = Kokkos::create_mirror_view(surface_radii_);
    host_surface_velocities_ = Kokkos::create_mirror_view(surface_velocities_);
    host_surface_forces_ = Kokkos::create_mirror_view(surface_forces_);

    // Copy the raw data into the views
    for (unsigned i = 0; i < num_surface_nodes_; i++) {
      surface_positions_host(3 * i + 0) = points_vec[3 * i + 0];
      surface_positions_host(3 * i + 1) = points_vec[3 * i + 1];
      surface_positions_host(3 * i + 2) = points_vec[3 * i + 2];
      surface_normals_host(3 * i + 0) = normals_vec[3 * i + 0];
      surface_normals_host(3 * i + 1) = normals_vec[3 * i + 1];
      surface_normals_host(3 * i + 2) = normals_vec[3 * i + 2];
      surface_velocities_host(3 * i + 0) = 0.0;
      surface_velocities_host(3 * i + 1) = 0.0;
      surface_velocities_host(3 * i + 2) = 0.0;
      surface_forces_host(3 * i + 0) = 0.0;
      surface_forces_host(3 * i + 1) = 0.0;
      surface_forces_host(3 * i + 2) = 0.0;
      surface_weights_host(i) = weights_vec[i];
      surface_radii_host(i) = 0.0;
    }

    // Copy the views to the device
    Kokkos::deep_copy(surface_positions, surface_positions_host);
    Kokkos::deep_copy(surface_normals, surface_normals_host);
    Kokkos::deep_copy(surface_weights, surface_weights_host);
    Kokkos::deep_copy(surface_radii, surface_radii_host);
    Kokkos::deep_copy(surface_velocities, surface_velocities_host);
    Kokkos::deep_copy(surface_forces, surface_forces_host);

    quadrature_setup_finished_ = true;
    return *this;
  }

  NoSlipPeripheryBuilder &setup_quadrature_from_file(const size_t num_surface_nodes_in_file,
                                                     const std::string &points_filename,
                                                     const std::string &normals_filename,
                                                     const std::string &weights_filename) {
    num_surface_nodes_ = num_surface_nodes_in_file;

    // Allocate the views. Note, resizing does not automatically update the mirror views.
    num_surface_nodes_ = weights_vec.size();
    Kokkos::resize(surface_positions_, 3 * num_surface_nodes_);
    Kokkos::resize(surface_normals_, 3 * num_surface_nodes_);
    Kokkos::resize(surface_weights_, num_surface_nodes_);
    Kokkos::resize(surface_radii_, num_surface_nodes_);
    Kokkos::resize(surface_velocities_, 3 * num_surface_nodes_);
    Kokkos::resize(surface_forces_, 3 * num_surface_nodes_);
    host_surface_positions_ = Kokkos::create_mirror_view(surface_positions_);
    host_surface_normals_ = Kokkos::create_mirror_view(surface_normals_);
    host_surface_weights_ = Kokkos::create_mirror_view(surface_weights_);
    host_surface_radii_ = Kokkos::create_mirror_view(surface_radii_);
    host_surface_velocities_ = Kokkos::create_mirror_view(surface_velocities_);
    host_surface_forces_ = Kokkos::create_mirror_view(surface_forces_);

    // Read the data from the files (to the host)
    mundy::alens::periphery::read_vector_from_file(weights_filename, num_surface_nodes_, host_surface_weights_);
    mundy::alens::periphery::read_vector_from_file(points_filename, 3 * num_surface_nodes_, host_surface_positions_);
    mundy::alens::periphery::read_vector_from_file(normals_filename, 3 * num_surface_nodes_, host_surface_normals_);

    // Copy the views to the device
    Kokkos::deep_copy(surface_positions_, host_surface_positions_);
    Kokkos::deep_copy(surface_normals_, host_surface_normals_);
    Kokkos::deep_copy(surface_weights_, host_surface_weights_);

    // Zero out the radii, forces, and velocities (on host and device)
    Kokkos::deep_copy(surface_radii_host, 0.0);
    Kokkos::deep_copy(surface_velocities_host, 0.0);
    Kokkos::deep_copy(surface_forces_host, 0.0);
    Kokkos::deep_copy(surface_radii, 0.0);
    Kokkos::deep_copy(surface_velocities, 0.0);
    Kokkos::deep_copy(surface_forces, 0.0);

    quadrature_setup_finished_ = true;
    return *this;
  }

  NoSlipPeripheryBuilder &precompute_self_interaction(const double viscosity, const bool write_to_file = false,
                                                      const bool use_values_from_file_if_present = false,
                                                      const std::string in_filename = "",
                                                      const std::string out_filename = "self_interaction_matrix.dat") {
    MUNDY_THROW_REQUIRE(quadrature_setup_finished_, std::runtime_error,
                        "Quadrature must be setup before precomputing the self-interaction matrix.");

    // Run the precomputation for the inverse self-interaction matrix
    Kokkos::resize(inv_self_interaction_matrix_, 3 * num_surface_nodes_, 3 * num_surface_nodes_);
    bool matrix_read_from_file = false;
    if (use_values_from_file_if_present) {
      auto does_file_exist = [](const std::string &filename) {
        std::ifstream f(filename.c_str());
        return f.good();
      };

      if (does_file_exist(in_filename)) {
        const size_t expected_num_rows_cols = 3 * num_surface_nodes_;
        mundy::alens::periphery::read_matrix_from_file(in_filename, expected_num_rows_cols, expected_num_rows_cols,
                                                       inv_self_interaction_matrix_);
        matrix_read_from_file = true;
      }
    }

    if (!matrix_read_from_file) {
      DoubleMatDeviceView self_interaction_matrix("self_interaction_matrix", 3 * num_surface_nodes_,
                                                  3 * num_surface_nodes_);
      mundy::alens::periphery::fill_skfie_matrix(stk::ngp::ExecSpace(), viscosity, num_surface_nodes_,
                                                 num_surface_nodes_, surface_positions_, surface_positions_,
                                                 surface_normals_, surface_weights_, self_interaction_matrix_);
      mundy::alens::periphery::invert_matrix(stk::ngp::ExecSpace(), self_interaction_matrix_,
                                             inv_self_interaction_matrix_);

      if (write_to_file) {
        Kokkos::deep_copy(host_inv_self_interaction_matrix_, inv_self_interaction_matrix_);
        mundy::alens::periphery::write_matrix_to_file(out_filename, host_inv_self_interaction_matrix_);
      }
    }

    build_finished_ = true;

    return *this;
  }

  KokkosNoSlipPeripheryData get_periphery_data(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Quadrature must be setup before accessing the periphery data.");
    return KokkosNoSlipPeripheryData{surface_positions_,
                                     surface_normals_,
                                     surface_weights_,
                                     surface_radii_,
                                     surface_forces_,
                                     surface_velocities_,
                                     inv_self_interaction_matrix_};
  }

  Double1DView &get_surface_positions(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Quadrature must be setup before accessing the surface positions.");
    return surface_positions_;
  }

  Double1DView &get_surface_normals(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Quadrature must be setup before accessing the surface normals.");
    return surface_normals_;
  }

  Double1DView &get_surface_weights(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Quadrature must be setup before accessing the surface weights.");
    return surface_weights_;
  }

  Double1DView &get_surface_radii(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Quadrature must be setup before accessing the surface radii.");
    return surface_radii_;
  }

  Double1DView &get_surface_forces(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Self-interaction matrix must be built before accessing the surface forces.");
    return surface_forces_;
  }

  Double1DView &get_surface_velocities(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && quadrature_setup_finished_, std::runtime_error,
                        "Self-interaction matrix must be built before accessing the surface velocities.");
    return surface_velocities_;
  }

  DoubleMatDeviceView &get_inv_self_interaction_matrix(bool throw_if_unfinished = true) {
    MUNDY_THROW_REQUIRE(throw_if_unfinished && build_finished_, std::runtime_error,
                        "Self-interaction matrix must be built before accessing the inverse self-interaction matrix.");
    return inv_self_interaction_matrix_;
  }

 private:
  bool quadrature_setup_finished_;
  bool build_finished_;
  size_t num_surface_nodes_;

  // Device views
  Double1DView surface_positions_;
  Double1DView surface_normals_;
  Double1DView surface_weights_;
  Double1DView surface_radii_;
  Double1DView surface_forces_;
  Double1DView surface_velocities_;
  DoubleMatDeviceView inv_self_interaction_matrix_;

  // Host mirrors
  Double1DView::HostMirror host_surface_positions_;
  Double1DView::HostMirror host_surface_normals_;
  Double1DView::HostMirror host_surface_weights_;
  Double1DView::HostMirror host_surface_radii_;
  Double1DView::HostMirror host_surface_forces_;
  Double1DView::HostMirror host_surface_velocities_;
  DoubleMatDeviceView::HostMirror host_inv_self_interaction_matrix_;
};

/// @brief An aggregate containing Kokkos data for a collection of spheres
struct KokkosSphereData {
  Double1DView positions;
  Double1DView radii;
};

/// \brief Create the Kokkos data for a collection of spheres
template <mundy::geom::ValidNgpSphereDataType NgpSphereDataType>
KokkosSphereData create_kokkos_sphere_data(stk::mesh::NgpMesh &ngp_mesh, NgpSphereDataType &ngp_sphere_data,
                                           stk::NgpVector<stk::mesh::Entity> &ngp_sphere_entities) {
  Kokkos::Profiling::pushRegion("mundy::mech::create_kokkos_sphere_data");
  const size_t num_spheres = ngp_sphere_entities.size();
  Double1DView positions("sphere_positions", 3 * num_spheres);
  Double1DView radii("sphere_radii", num_spheres);

  // Copy the sphere data to the device
  Kokkos::parallel_for(
      stk::ngp::DeviceRangePolicy(0, num_spheres), KOKKOS_LAMBDA(const unsigned &vector_index) {
        stk::mesh::Entity sphere = ngp_sphere_entities.device_get(vector_index);
        auto sphere_index = ngp_mesh.fast_mesh_index(sphere);
        auto &sphere_view = mundy::geom::create_ngp_sphere_entity_view(ngp_sphere_data, sphere_index);
        positions(vector_index * 3 + 0) = sphere_view.center()[0];
        positions(vector_index * 3 + 1) = sphere_view.center()[1];
        positions(vector_index * 3 + 2) = sphere_view.center()[2];

        radii(vector_index) = sphere_view.radius();
      });

  Kokkos::Profiling::popRegion();
  return KokkosSphereData{positions, radii};
}

/// @brief An aggregate containing Kokkos data for a collection of motile spheres
struct KokkosMotileSphereData {
  Double1DView positions;
  Double1DView radii;
  Double1DView forces;
  Double1DView velocities;
};

void compute_rpy_mobility_spheres(const double &viscosity, KokkosMotileSphereData &spheres) {
  Kokkos::Profiling::pushRegion("HP1::compute_rpy_mobility_spheres");
  auto &sphere_positions = spheres.positions;
  auto &sphere_radii = spheres.radii;
  auto &sphere_forces = spheres.forces;
  auto &sphere_velocities = spheres.velocities;

  const size_t num_spheres = sphere_radii.extent(0);
  MUNDY_THROW_ASSERT(sphere_positions.extent(0) == 3 * num_spheres, std::runtime_error,
                     "Sphere positions has the wrong size.");
  MUNDY_THROW_ASSERT(sphere_forces.extent(0) == 3 * num_spheres, std::runtime_error,
                     "Sphere forces has the wrong size.");
  MUNDY_THROW_ASSERT(sphere_velocities.extent(0) == 3 * num_spheres, std::runtime_error,
                     "Sphere velocities has the wrong size.");

  // Apply the RPY kernel from spheres to spheres
  mundy::alens::periphery::apply_rpy_kernel(stk::ngp::ExecSpace(), viscosity, sphere_positions, sphere_positions,
                                            sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

  // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
  mundy::alens::periphery::apply_local_drag(stk::ngp::ExecSpace(), viscosity, sphere_velocities, sphere_forces,
                                            sphere_radii);

  Kokkos::Profiling::popRegion();
}

void compute_confined_rpy_mobility_spheres(const double &viscosity,          //
                                           KokkosMotileSphereData &spheres,  //
                                           KokkosNoSlipPeripheryData &periphery_data) {
  Kokkos::Profiling::pushRegion("HP1::compute_confined_rpy_mobility_spheres");
  const size_t num_spheres = sphere_radii.extent(0);
  const size_t num_surface_nodes = surface_weights.extent(0);

  auto &surface_positions = periphery_data.surface_positions;
  auto &surface_normals = periphery_data.surface_normals;
  auto &surface_weights = periphery_data.surface_weights;
  auto &surface_radii = periphery_data.surface_radii;
  auto &surface_forces = periphery_data.surface_forces;
  auto &surface_velocities = periphery_data.surface_velocities;

  MUNDY_THROW_ASSERT(sphere_positions.extent(0) == 3 * num_spheres, std::runtime_error,
                     "Sphere positions has the wrong size.");
  MUNDY_THROW_ASSERT(sphere_forces.extent(0) == 3 * num_spheres, std::runtime_error,
                     "Sphere forces has the wrong size.");
  MUNDY_THROW_ASSERT(sphere_velocities.extent(0) == 3 * num_spheres, std::runtime_error,
                     "Sphere velocities has the wrong size.");
  MUNDY_THROW_ASSERT(surface_positions.extent(0) == 3 * num_surface_nodes, std::runtime_error,
                     "Surface positions has the wrong size.");
  MUNDY_THROW_ASSERT(surface_normals.extent(0) == 3 * num_surface_nodes, std::runtime_error,
                     "Surface normals has the wrong size.");
  MUNDY_THROW_ASSERT(surface_weights.extent(0) == num_surface_nodes, std::runtime_error,
                     "Surface weights has the wrong size.");
  MUNDY_THROW_ASSERT(surface_radii.extent(0) == num_surface_nodes, std::runtime_error,
                     "Surface radii has the wrong size.");
  MUNDY_THROW_ASSERT(surface_forces.extent(0) == 3 * num_surface_nodes, std::runtime_error,
                     "Surface forces has the wrong size.");
  MUNDY_THROW_ASSERT(surface_velocities.extent(0) == 3 * num_surface_nodes, std::runtime_error,
                     "Surface velocities has the wrong size.");
  MUNDY_THROW_ASSERT((inv_self_interaction_matrix.extent(0) == 3 * num_spheres &&
                      inv_self_interaction_matrix.extent(1) == 3 * num_spheres),
                     std::runtime_error, "Self interaction matrix has the wrong size.");

  // Apply the RPY kernel from spheres to spheres
  mundy::alens::periphery::apply_rpy_kernel(stk::ngp::ExecSpace(), viscosity, sphere_positions, sphere_positions,
                                            sphere_radii, sphere_radii, sphere_forces, sphere_velocities);

  /////////////////////////////////////////////////////////////
  // Apply the correction for the no-slip boundary condition //
  /////////////////////////////////////////////////////////////
  // Apply the RPY kernel from spheres to periphery
  mundy::alens::periphery::apply_rpy_kernel(stk::ngp::ExecSpace(), viscosity, sphere_positions, surface_positions,
                                            sphere_radii, surface_radii, sphere_forces, surface_velocities);

  // Map the slip velocities to the surface forces
  // The negative one in the gemv call accounts for the fact that our force should balance the u_slip
  KokkosBlas::gemv(stk::ngp::ExecSpace(), "N", -1.0, inv_self_interaction_matrix, surface_velocities, 1.0,
                   surface_forces);
  mundy::alens::periphery::apply_stokes_double_layer_kernel(
      stk::ngp::ExecSpace(), viscosity, num_surface_nodes, num_spheres, surface_positions, sphere_positions,
      surface_normals, surface_weights, surface_forces, sphere_velocities);

  //////////////////////
  // Self-interaction //
  //////////////////////
  // The RPY kernel is only long-range, it doesn't add on self-interaction for the spheres
  mundy::alens::periphery::apply_local_drag(stk::ngp::ExecSpace(), viscosity, sphere_velocities, sphere_forces,
                                            sphere_radii);

  Kokkos::Profiling::popRegion();
}

void compute_local_drag_mobility_spheres(stk::mesh::NgpMesh &ngp_mesh,                      //
                                         const double &viscosity,                           //
                                         stk::mesh::NgpField<double> &node_velocity_field,  //
                                         stk::mesh::NgpField<double> &node_force_field,     //
                                         stk::mesh::NgpField<double> &elem_radius_field,    //
                                         const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("HP1::compute_local_drag_mobility_spheres");

  node_velocity_field.sync_to_device();
  node_force_field.sync_to_device();
  elem_radius_field.sync_to_device();

  constexpr double pi = Kokkos::numbers::pi_v<double>;
  constexpr double one_over_6pi = 1.0 / (6.0 * pi);
  const double one_over_6pi_mu = one_over_6pi / viscosity;

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::Entity node = nodes[0];
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
        const double sphere_radius = elem_radius_field(sphere_index, 0);
        const auto node_force = mundy::mesh::vector3_field_data(node_force_field, node_index);

        const double inv_drag_coeff = one_over_6pi_mu / sphere_radius;
        auto node_velocity = mundy::mesh::vector3_field_data(node_velocity_field, node_index);
        node_velocity[0] += node_force[0] * inv_drag_coeff;
        node_velocity[1] += node_force[1] * inv_drag_coeff;
        node_velocity[2] += node_force[2] * inv_drag_coeff;
      });

  node_velocity_field.modify_on_device();

  Kokkos::Profiling::popRegion();
}

struct SphereLocalDragMobilityOp {
  SphereLocalDragMobilityOp(stk::mesh::NgpMesh &ngp_mesh,  //
                            const double viscosity,        //
                            stk::mesh::NgpField<double> &elem_radius_field, const stk::mesh::Selector &selector)
      : ngp_mesh_(ngp_mesh), viscosity_(viscosity), elem_radius_field_(elem_radius_field), selector_(selector) {
  }

  void update() {
  }

  void apply(stk::mesh::NgpField<double> &node_force_field, stk::mesh::NgpField<double> &node_velocity_field) {
    Kokkos::Profiling::pushRegion("mundy::mech::SphereLocalDragMobilityOp::apply");
    node_force_field.sync_to_device();
    node_velocity_field.sync_to_device();

    compute_local_drag_mobility_spheres(ngp_mesh_, viscosity_, elem_radius_field_, node_force_field,
                                        node_velocity_field, selector_);

    node_velocity_field.modify_on_device();
    Kokkos::Profiling::popRegion();
  }

  // Leaving private members public to make this class a composite
  stk::mesh::NgpMesh &ngp_mesh_;
  const double viscosity_;
  stk::mesh::NgpField<double> &elem_radius_field_;
  const stk::mesh::Selector &selector_;
};

struct SphereRpyMobilityOp {
  SphereRpyMobilityOp(stk::mesh::NgpMesh &ngp_mesh,                    //
                      const double viscosity,                          //
                      stk::mesh::NgpField<double> &node_coords_field,  //
                      stk::mesh::NgpField<double> &elem_radius_field,
                      const stk::NgpVector<stk::mesh::Entity> &ngp_sphere_entities)
      : ngp_mesh_(ngp_mesh),
        viscosity_(viscosity),
        node_coords_field_(node_coords_field),
        elem_radius_field_(elem_radius_field),
        ngp_sphere_entities_(ngp_sphere_entities),
        num_spheres_(ngp_sphere_entities.size()),
        sphere_positions_view_("sphere_positions_view", 3 * num_spheres_),
        sphere_radii_view_("sphere_radii_view", num_spheres_),
        sphere_forces_view_("sphere_forces_view", 3 * num_spheres_),
        sphere_velocities_view_("sphere_velocities_view", 3 * num_spheres_) {
  }

  void update() {
    node_coords_field_.sync_to_device();
    elem_radius_field_.sync_to_device();

    Kokkos::parallel_for(
        stk::ngp::RangePolicy<stk::ngp::ExecSpace>(0, num_spheres_), KOKKOS_LAMBDA(const int &vector_index) {
          stk::mesh::Entity sphere = ngp_sphere_entities_.device_get(vector_index);
          auto sphere_index = ngp_mesh_.fast_mesh_index(sphere);
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh_.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh_.fast_mesh_index(node);

          sphere_positions_view_(vector_index * 3 + 0) = node_coords_field_(node_index, 0);
          sphere_positions_view_(vector_index * 3 + 1) = node_coords_field_(node_index, 1);
          sphere_positions_view_(vector_index * 3 + 2) = node_coords_field_(node_index, 2);

          sphere_radii_view_(vector_index) = elem_radius_field_(sphere_index, 0);
        });
  }

  void apply(stk::mesh::NgpField<double> &node_force_field, stk::mesh::NgpField<double> &node_velocity_field) {
    Kokkos::Profiling::pushRegion("mundy::mech::SphereRpyMobilityOp::apply");

    node_force_field.sync_to_device();
    node_velocity_field.sync_to_device();

    // Copy the force to the view
    Kokkos::parallel_for(
        stk::ngp::RangePolicy<stk::ngp::ExecSpace>(0, num_spheres_), KOKKOS_LAMBDA(const int &vector_index) {
          stk::mesh::Entity sphere = ngp_sphere_entities_.device_get(vector_index);
          auto sphere_index = ngp_mesh_.fast_mesh_index(sphere);
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh_.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh_.fast_mesh_index(node);

          sphere_forces_view_(vector_index * 3 + 0) = node_force_field(node_index, 0);
          sphere_forces_view_(vector_index * 3 + 1) = node_force_field(node_index, 1);
          sphere_forces_view_(vector_index * 3 + 2) = node_force_field(node_index, 2);
        });
    Kokkos::deep_copy(sphere_velocities_view_, 0.0);

    // Apply the RPY kernel
    compute_rpy_mobility_spheres(viscosity_, sphere_positions_view_, sphere_radii_view_, sphere_forces_view_,
                                 sphere_velocities_view_);

    // Copy the velocities back to the field
    Kokkos::parallel_for(
        stk::ngp::RangePolicy<stk::ngp::ExecSpace>(0, num_spheres_), KOKKOS_LAMBDA(const int &vector_index) {
          stk::mesh::Entity sphere = ngp_sphere_entities_.device_get(vector_index);
          auto sphere_index = ngp_mesh_.fast_mesh_index(sphere);
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh_.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh_.fast_mesh_index(node);

          node_velocity_field(node_index, 0) += sphere_velocities_view_(vector_index * 3 + 0);
          node_velocity_field(node_index, 1) += sphere_velocities_view_(vector_index * 3 + 1);
          node_velocity_field(node_index, 2) += sphere_velocities_view_(vector_index * 3 + 2);
        });

    node_velocity_field.modify_on_device();
    Kokkos::Profiling::popRegion();
  }

 private:
  stk::mesh::NgpMesh &ngp_mesh_;
  const double viscosity_;
  stk::mesh::NgpField<double> &node_coords_field_;
  stk::mesh::NgpField<double> &elem_radius_field_;
  const stk::NgpVector<stk::mesh::Entity> &ngp_sphere_entities_;
  const size_t num_spheres_;
  Double1DView sphere_positions_view_;
  Double1DView sphere_radii_view_;
  Double1DView sphere_forces_view_;
  Double1DView sphere_velocities_view_;
};

struct SphereConfinedRpyMobilityOp {
  SphereConfinedRpyMobilityOp(stk::mesh::NgpMesh &ngp_mesh,                                  //
                              const double viscosity,                                        //
                              stk::mesh::NgpField<double> &node_coords_field,                //
                              stk::mesh::NgpField<double> &elem_radius_field,                //
                              const stk::NgpVector<stk::mesh::Entity> &ngp_sphere_entities,  //
                              KokkosSphereData &kokkos_spheres,                              //
                              KokkosNoSlipPeripheryData &kokkos_periphery_data)
      : ngp_mesh_(ngp_mesh),
        viscosity_(viscosity),
        node_coords_field_(node_coords_field),
        elem_radius_field_(elem_radius_field),
        ngp_sphere_entities_(ngp_sphere_entities),
        num_spheres_(ngp_sphere_entities.size()),
        kokkos_spheres_(kokkos_spheres),
        kokkos_periphery_data_(kokkos_periphery_data),
  {
  }

  void update() {
    node_coords_field_.sync_to_device();
    elem_radius_field_.sync_to_device();

    Kokkos::parallel_for(
        stk::ngp::RangePolicy<stk::ngp::ExecSpace>(0, num_spheres_), KOKKOS_LAMBDA(const int &vector_index) {
          stk::mesh::Entity sphere = ngp_sphere_entities_.device_get(vector_index);
          auto sphere_index = ngp_mesh_.fast_mesh_index(sphere);
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh_.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh_.fast_mesh_index(node);

          sphere_positions_view_(vector_index * 3 + 0) = node_coords_field_(node_index, 0);
          sphere_positions_view_(vector_index * 3 + 1) = node_coords_field_(node_index, 1);
          sphere_positions_view_(vector_index * 3 + 2) = node_coords_field_(node_index, 2);

          sphere_radii_view_(vector_index) = elem_radius_field_(sphere_index, 0);
        });
  }

  void apply(stk::mesh::NgpField<double> &node_force_field, stk::mesh::NgpField<double> &node_velocity_field) {
    Kokkos::Profiling::pushRegion("mundy::mech::SphereRpyMobilityOp::apply");

    node_force_field.sync_to_device();
    node_velocity_field.sync_to_device();

    // Copy the force to the view
    Kokkos::parallel_for(
        stk::ngp::RangePolicy<stk::ngp::ExecSpace>(0, num_spheres_), KOKKOS_LAMBDA(const int &vector_index) {
          stk::mesh::Entity sphere = ngp_sphere_entities_.device_get(vector_index);
          auto sphere_index = ngp_mesh_.fast_mesh_index(sphere);
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh_.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh_.fast_mesh_index(node);

          sphere_forces_view_(vector_index * 3 + 0) = node_force_field(node_index, 0);
          sphere_forces_view_(vector_index * 3 + 1) = node_force_field(node_index, 1);
          sphere_forces_view_(vector_index * 3 + 2) = node_force_field(node_index, 2);
        });
    Kokkos::deep_copy(sphere_velocities_view_, 0.0);

    // Apply the RPY kernel + boundary integral confinement
    compute_confined_rpy_mobility_spheres(viscosity_, kokkos_spheres_, periphery_data_);
    sphere_positions_view_, sphere_radii_view_, sphere_forces_view_,
                                          sphere_velocities_view_, surface_positions_view_, surface_normals_view_,
                                          surface_weights_view_, surface_radii_view_, surface_velocities_view_,
                                          surface_forces_view_, inv_self_interaction_matrix_);

    // Copy the velocities back to the field
    Kokkos::parallel_for(
        stk::ngp::RangePolicy<stk::ngp::ExecSpace>(0, num_spheres_), KOKKOS_LAMBDA(const int &vector_index) {
          stk::mesh::Entity sphere = ngp_sphere_entities_.device_get(vector_index);
          auto sphere_index = ngp_mesh_.fast_mesh_index(sphere);
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh_.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh_.fast_mesh_index(node);

          node_velocity_field(node_index, 0) += sphere_velocities_view_(vector_index * 3 + 0);
          node_velocity_field(node_index, 1) += sphere_velocities_view_(vector_index * 3 + 1);
          node_velocity_field(node_index, 2) += sphere_velocities_view_(vector_index * 3 + 2);
        });

    node_velocity_field.modify_on_device();
    Kokkos::Profiling::popRegion();
  }

 private:
  stk::mesh::NgpMesh &ngp_mesh_;
  const double viscosity_;

  stk::mesh::NgpField<double> &node_coords_field_;
  stk::mesh::NgpField<double> &elem_radius_field_;
  const stk::NgpVector<stk::mesh::Entity> &ngp_sphere_entities_;

  const size_t num_spheres_;
  KokkosSphereData spheres_;
  KokkosNoSlipPeripheryData &periphery_data_;
};
//@}

//! \name Spring forces
//@{

void compute_hookean_spring_forces(stk::mesh::NgpMesh &ngp_mesh,                           //
                                   stk::mesh::NgpField<double> &node_coords_field,         //
                                   stk::mesh::NgpField<double> &node_force_field,          //
                                   stk::mesh::NgpField<double> &spring_constant_field,     //
                                   stk::mesh::NgpField<double> &spring_rest_length_field,  //
                                   const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("mundy::mech::compute_hookean_spring_forces");

  node_coords_field.sync_to_device();
  node_force_field.sync_to_device();
  spring_constant_field.sync_to_device();
  spring_rest_length_field.sync_to_device();

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
        const stk::mesh::Entity node1 = nodes[0];
        const stk::mesh::Entity node2 = nodes[1];
        const stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(node1);
        const stk::mesh::FastMeshIndex node2_index = ngp_mesh.fast_mesh_index(node2);
        const auto node1_coords = node_coords_field(node1_index);
        const auto node2_coords = node_coords_field(node2_index);
        const double spring_constant = spring_constant_field(spring_index, 0);
        const double spring_rest_length = spring_rest_length_field(spring_index, 0);

        // Compute the spring force
        // Note the extract inv_spring_length due to not normalizing the direction vector.
        const double dx = node2_coords[0] - node1_coords[0];
        const double dy = node2_coords[1] - node1_coords[1];
        const double dz = node2_coords[2] - node1_coords[2];
        const double spring_length = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
        const double inv_spring_length = 1.0 / spring_length;
        const double force_magnitude = spring_constant * (spring_length - spring_rest_length) * inv_spring_length;
        const double fx = force_magnitude * dx;
        const double fy = force_magnitude * dy;
        const double fz = force_magnitude * dz;

        // Apply the force to the nodes. Use atomic add
        Kokkos::atomic_add(&node_force_field(node1_index, 0), fx);
        Kokkos::atomic_add(&node_force_field(node1_index, 1), fy);
        Kokkos::atomic_add(&node_force_field(node1_index, 2), fz);
        Kokkos::atomic_add(&node_force_field(node2_index, 0), -fx);
        Kokkos::atomic_add(&node_force_field(node2_index, 1), -fy);
        Kokkos::atomic_add(&node_force_field(node2_index, 2), -fz);
      });

  node_force_field.modify_on_device();
  Kokkos::Profiling::popRegion();
}

void compute_fene_spring_forces(stk::mesh::NgpMesh &ngp_mesh,                          //
                                stk::mesh::NgpField<double> &node_coords_field,        //
                                stk::mesh::NgpField<double> &node_force_field,         //
                                stk::mesh::NgpField<double> &spring_constant_field,    //
                                stk::mesh::NgpField<double> &spring_max_length_field,  //
                                const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("mundy::mech::compute_fene_spring_forces");

  node_coords_field.sync_to_device();
  node_force_field.sync_to_device();
  spring_constant_field.sync_to_device();
  spring_max_length_field.sync_to_device();

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &spring_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, spring_index);
        const stk::mesh::Entity node1 = nodes[0];
        const stk::mesh::Entity node2 = nodes[1];
        const stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(node1);
        const stk::mesh::FastMeshIndex node2_index = ngp_mesh.fast_mesh_index(node2);
        const auto node1_coords = node_coords_field(node1_index);
        const auto node2_coords = node_coords_field(node2_index);
        const double spring_constant = spring_constant_field(spring_index, 0);
        const double spring_max_length = spring_max_length_field(spring_index, 0);
        const double inv_spring_max_length = 1.0 / spring_max_length;

        // Compute the spring force
        // The fene spring force is F = -k * r / (1 - (r / r_max)^2)
        const double dx = node2_coords[0] - node1_coords[0];
        const double dy = node2_coords[1] - node1_coords[1];
        const double dz = node2_coords[2] - node1_coords[2];
        const double length = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
        const double inv_length = 1.0 / length;
        const double force_magnitude =
            spring_constant / (1.0 - (length * inv_spring_max_length) * (length * inv_spring_max_length));
        const double fx = force_magnitude * dx;
        const double fy = force_magnitude * dy;
        const double fz = force_magnitude * dz;

        // Apply the force to the nodes. Use atomic add
        Kokkos::atomic_add(&node_force_field(node1_index, 0), -fx);
        Kokkos::atomic_add(&node_force_field(node1_index, 1), -fy);
        Kokkos::atomic_add(&node_force_field(node1_index, 2), -fz);
        Kokkos::atomic_add(&node_force_field(node2_index, 0), fx);
        Kokkos::atomic_add(&node_force_field(node2_index, 1), fy);
        Kokkos::atomic_add(&node_force_field(node2_index, 2), fz);
      });

  node_force_field.modify_on_device();
  Kokkos::Profiling::popRegion();
}
//@}

//! \name Sphere-Sphere Collision Resolution
//@{

void compute_signed_separation_distance_and_contact_normal(stk::mesh::NgpMesh &ngp_mesh,                     //
                                                           const LocalResultViewType &local_search_results,  //
                                                           stk::mesh::NgpField<double> &node_coords_field,   //
                                                           stk::mesh::NgpField<double> &elem_radius_field,   //
                                                           const Double1DView &signed_sep_dist,              //
                                                           const Double2DView &con_normals_ij) {
  Kokkos::Profiling::pushRegion("mundy::mech::compute_signed_separation_distance_and_contact_normal");

  // Each neighbor pair will generate a constraint between the two spheres
  // Loop over each neighbor id pair, fetch each sphere's position, and compute the signed separation distance
  // defined by \|x_i - x_j\| - (r_i + r_j) where x_i and x_j are the sphere positions and r_i and r_j are the sphere
  // radii
  node_coords_field.sync_to_device();
  elem_radius_field.sync_to_device();

  // Loop over the neighbor pairs
  using range_policy = Kokkos::RangePolicy<stk::ngp::ExecSpace>;
  Kokkos::parallel_for(
      "GenerateCollisionConstraints", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_elem_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_elem_index = local_search_results(i).rangeIdentProc.id();
        if (source_elem_index == target_elem_index) {
          return;
        }

        const stk::mesh::Entity source_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_elem_index)[0];
        const stk::mesh::Entity target_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_elem_index)[0];
        const stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_node);
        const stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_node);

        // Fetch the sphere positions and radii
        const double x_i = node_coords_field(source_node_index, 0);
        const double y_i = node_coords_field(source_node_index, 1);
        const double z_i = node_coords_field(source_node_index, 2);
        const double x_j = node_coords_field(target_node_index, 0);
        const double y_j = node_coords_field(target_node_index, 1);
        const double z_j = node_coords_field(target_node_index, 2);
        const double radius_i = elem_radius_field(source_elem_index, 0);
        const double radius_j = elem_radius_field(target_elem_index, 0);

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

  Kokkos::Profiling::popRegion();
}

void compute_max_abs_projected_sep(const stk::ParallelMachine parallel,       //
                                   const Double1DView &lagrange_multipliers,  //
                                   const Double1DView &signed_sep_dist,       //
                                   const Double1DView &signed_sep_dot,        //
                                   const double dt,                           //
                                   double &max_abs_projected_sep) {
  Kokkos::Profiling::pushRegion("mundy::mech::compute_max_abs_projected_sep");

  // Perform parallel reduction over all linker indices
  double local_max_abs_projected_sep = -1.0;
  using range_policy = Kokkos::RangePolicy<stk::ngp::ExecSpace>;
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

        // update the max value
        if (abs_projected_sep > max_val) {
          max_val = abs_projected_sep;
        }
      },
      Kokkos::Max<double>(local_max_abs_projected_sep));

  // Global reduction
  max_abs_projected_sep = -1.0;
  stk::all_reduce_max(parallel, &local_max_abs_projected_sep, &max_abs_projected_sep, 1);

  Kokkos::Profiling::popRegion();
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

void compute_diff_dots(const stk::ParallelMachine parallel,           //
                       const Double1DView &lagrange_multipliers,      //
                       const Double1DView &lagrange_multipliers_tmp,  //
                       const Double1DView &signed_sep_dot,            //
                       const Double1DView &signed_sep_dot_tmp,        //
                       const double dt,                               //
                       double &dot_xkdiff_xkdiff,                     //
                       double &dot_xkdiff_gkdiff,                     //
                       double &dot_gkdiff_gkdiff) {
  Kokkos::Profiling::pushRegion("mundy::mech::compute_diff_dots");

  // Local variables to store dot products
  mundy::math::Vector3d local_xx_xg_gg_diff = {0.0, 0.0, 0.0};

  // Perform parallel reduction to compute the dot products
  using range_policy = Kokkos::RangePolicy<stk::ngp::ExecSpace>;
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
      DiffDotsReducer<stk::ngp::ExecSpace>(local_xx_xg_gg_diff));

  // Global reduction
  stk::all_reduce_sum(parallel, &local_xx_xg_gg_diff[0], &dot_xkdiff_xkdiff, 1);
  stk::all_reduce_sum(parallel, &local_xx_xg_gg_diff[1], &dot_xkdiff_gkdiff, 1);
  stk::all_reduce_sum(parallel, &local_xx_xg_gg_diff[2], &dot_gkdiff_gkdiff, 1);

  Kokkos::Profiling::popRegion();
}

void sum_collision_force(stk::mesh::NgpMesh &ngp_mesh,                     //
                         const LocalResultViewType &local_search_results,  //
                         const Double2DView &con_normal_ij,                //
                         const Double1DView &lagrange_multipliers,         //
                         stk::mesh::NgpField<double> &node_force_field) {
  Kokkos::Profiling::pushRegion("mundy::mech::sum_collision_force");

  node_force_field.sync_to_device();

  // Zero out the force first
  node_force_field.set_all(ngp_mesh, 0.0);

  // Loop over the neighbor pairs
  using range_policy = Kokkos::RangePolicy<stk::ngp::ExecSpace>;
  Kokkos::parallel_for(
      "SumCollisionForce", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_elem_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_elem_index = local_search_results(i).rangeIdentProc.id();
        if (source_elem_index == target_elem_index) {
          return;
        }

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
        // because the pair will not be swapped if both are owned, allowing us to always add into the source
        const stk::mesh::Entity node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_elem_index)[0];
        const stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(node);
        Kokkos::atomic_add(&node_force_field(source_node_index, 0), -lag_mult * normal_x);
        Kokkos::atomic_add(&node_force_field(source_node_index, 1), -lag_mult * normal_y);
        Kokkos::atomic_add(&node_force_field(source_node_index, 2), -lag_mult * normal_z);
      });

  node_force_field.modify_on_device();
}

void compute_rate_of_change_of_sep(stk::mesh::NgpMesh &ngp_mesh,                      //
                                   const LocalResultViewType &local_search_results,   //
                                   stk::mesh::NgpField<double> &node_velocity_field,  //
                                   const Double2DView &con_normal_ij,                 //
                                   const Double1DView &signed_sep_dot) {
  Kokkos::Profiling::pushRegion("mundy::mech::compute_rate_of_change_of_sep");

  node_velocity_field.sync_to_device();

  // Compute the (linearized) rate of change in sep
  using range_policy = Kokkos::RangePolicy<stk::ngp::ExecSpace>;
  Kokkos::parallel_for(
      "ComputeRateOfChangeOfSep", range_policy(0, local_search_results.extent(0)), KOKKOS_LAMBDA(const int i) {
        const stk::mesh::FastMeshIndex source_elem_index = local_search_results(i).domainIdentProc.id();
        const stk::mesh::FastMeshIndex target_elem_index = local_search_results(i).rangeIdentProc.id();
        if (source_elem_index == target_elem_index) {
          return;
        }

        const stk::mesh::Entity source_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, source_elem_index)[0];
        const stk::mesh::Entity target_node = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, target_elem_index)[0];
        const stk::mesh::FastMeshIndex source_node_index = ngp_mesh.fast_mesh_index(source_node);
        const stk::mesh::FastMeshIndex target_node_index = ngp_mesh.fast_mesh_index(target_node);

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

  Kokkos::Profiling::popRegion();
}

void update_con_gammas(const Double1DView &lagrange_multipliers,      //
                       const Double1DView &lagrange_multipliers_tmp,  //
                       const Double1DView &signed_sep_dist,           //
                       const Double1DView &signed_sep_dot,            //
                       const double dt,                               //
                       const double alpha) {
  Kokkos::Profiling::pushRegion("mundy::mech::update_con_gammas");

  using range_policy = Kokkos::RangePolicy<stk::ngp::ExecSpace>;
  Kokkos::parallel_for(
      "UpdateConGammas", range_policy(0, lagrange_multipliers.extent(0)), KOKKOS_LAMBDA(const int i) {
        // Fetch fields for the current linker
        const double sep_old = signed_sep_dist(i);
        const double sep_dot = signed_sep_dot(i);
        const double sep_new = sep_old + dt * sep_dot;

        // Update lagrange multipliers with bound projection
        lagrange_multipliers(i) = Kokkos::max(lagrange_multipliers_tmp(i) - alpha * sep_new, 0.0);
      });

  Kokkos::Profiling::popRegion();
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
double get_max_speed(Mesh &ngp_mesh, Field &vel_field,  //
                     const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("mundy::mech::get_max_speed");

  vel_field.sync_to_device();

  double local_max_speed_sq = 0.0;
  Kokkos::Max<double> max_reduction(local_max_speed_sq);
  FieldSpeedReductionFunctor<Field> functor(vel_field, max_reduction);
  stk::mesh::for_each_entity_reduce(ngp_mesh, vel_field.get_rank(), selector, max_reduction, functor);

  double global_max_speed_sq = 0.0;
  stk::all_reduce_max(vel_field.get_field_base()->get_mesh().parallel(), &local_max_speed_sq, &global_max_speed_sq, 1);

  Kokkos::Profiling::popRegion();
  return Kokkos::sqrt(global_max_speed_sq);
}

template <typename ApplyMobilityOp>
CollisionResult resolve_collisions(stk::mesh::NgpMesh &ngp_mesh,                                //
                                   const double viscosity,                                      //
                                   const double dt,                                             //
                                   const double max_allowable_overlap,                          //
                                   const int max_col_iterations,                                //
                                   ApplyMobilityOp &apply_mobility_op,                          //
                                   const LocalResultViewType &local_search_results,             //
                                   stk::mesh::NgpField<double> &node_force_field,               //
                                   stk::mesh::NgpField<double> &node_velocity_field,            //
                                   stk::mesh::NgpField<double> &node_collision_force_field,     //
                                   stk::mesh::NgpField<double> &node_collision_velocity_field,  //
                                   const Double1DView &signed_sep_dist,                         //
                                   const Double2DView &con_normal_ij,                           //
                                   const Double1DView &lagrange_multipliers,                    //
                                   const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("mundy::mech::resolve_collisions");

  // Matrix-free BBPGD
  int ite_count = 0;
  int num_collisions = local_search_results.extent(0);
  Double1DView lagrange_multipliers_tmp("lagrange_multipliers_tmp", num_collisions);
  Double1DView signed_sep_dot("signed_sep_dot", num_collisions);
  Double1DView signed_sep_dot_tmp("signed_sep_dot_tmp", num_collisions);
  const stk::ParallelMachine parallel = node_collision_force_field.get_field_base()->get_mesh().parallel();

  // Use the given lagrange_multipliers as the initial guess
  Kokkos::deep_copy(lagrange_multipliers_tmp, lagrange_multipliers);
  Kokkos::deep_copy(signed_sep_dot_tmp, 0.0);

  // To account for external forces and external velocities, use them to update the initial signed_sep_dist
  // phi_0_corrected = phi_0 + dt * D^T U_ext

  // U_ext += M F_ext
  apply_mobility_op.apply(
      node_force_field,
      node_velocity_field);  // No selector here, assuming that the mobility op allready accounts for the selector

  // D^T U_ext
  compute_rate_of_change_of_sep(ngp_mesh, local_search_results, node_velocity_field, con_normal_ij, signed_sep_dot);

  // signed_sep_dist += dt * signed_sep_dot
  Kokkos::parallel_for(
      "UpdateSignedSepDist", Kokkos::RangePolicy<stk::ngp::ExecSpace>(0, num_collisions),
      KOKKOS_LAMBDA(const int i) { signed_sep_dist(i) += dt * signed_sep_dot(i); });

  ///////////////////////
  Kokkos::deep_copy(signed_sep_dot, 0.0);

  // Compute gkm1 = D^T M D xkm1
  // Compute F = D xkm1
  sum_collision_force(ngp_mesh, local_search_results, con_normal_ij, lagrange_multipliers_tmp,
                      node_collision_force_field);

  // Compute U = M F
  apply_mobility_op.apply(node_collision_force_field, node_collision_velocity_field);

  // Compute gkm1 = dt D^T U
  compute_rate_of_change_of_sep(ngp_mesh, local_search_results, node_collision_velocity_field, con_normal_ij,
                                signed_sep_dot_tmp);

  ///////////////////////
  // Check convergence //
  ///////////////////////
  // res = max(abs(projectPhi(gkm1)));
  double max_abs_projected_sep = -1.0;
  compute_max_abs_projected_sep(parallel, lagrange_multipliers_tmp, signed_sep_dist, signed_sep_dot_tmp, dt,
                                max_abs_projected_sep);

  ///////////////////////
  // Loop if necessary //
  ///////////////////////
  if (max_abs_projected_sep < max_allowable_overlap) {
    // The initial guess was correct, nothing more is necessary
  } else {
    // Initial guess insufficient, iterate

    // First step, Dai&Fletcher2005 Section 5.
    double alpha = 1.0 / max_abs_projected_sep;
    while (ite_count < max_col_iterations) {
      ++ite_count;

      // Compute xk = xkm1 - alpha * gkm1 and perform the bound projection xk = boundProjection(xk)
      update_con_gammas(lagrange_multipliers, lagrange_multipliers_tmp, signed_sep_dist, signed_sep_dot, dt, alpha);

      // Compute new grad with xk: gk = dt D^T M D xk
      //   Compute F = D xk
      sum_collision_force(ngp_mesh, local_search_results, con_normal_ij, lagrange_multipliers,
                          node_collision_force_field);

      // Compute U = M F
      apply_mobility_op.apply(node_collision_force_field, node_collision_velocity_field);

      //   Compute gk = dt D^T U
      compute_rate_of_change_of_sep(ngp_mesh, local_search_results, node_collision_velocity_field, con_normal_ij,
                                    signed_sep_dot);

      // check convergence via res = max(abs(projectPhi(gk)));
      compute_max_abs_projected_sep(parallel, lagrange_multipliers, signed_sep_dist, signed_sep_dot, dt,
                                    max_abs_projected_sep);

      if (max_abs_projected_sep < max_allowable_overlap) {
        // con_gammas worked.
        std::cout << "Convergence reached: " << max_abs_projected_sep << " < " << max_allowable_overlap << std::endl;
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

  // Compute the max speed
  double max_speed = get_max_speed(ngp_mesh, node_collision_velocity_field, selector);
  CollisionResult result = {max_abs_projected_sep, ite_count, max_speed * dt};

  Kokkos::Profiling::popRegion();
  return result;
}

void check_overlap(const stk::mesh::BulkData &bulk_data,               //
                   const double max_allowable_overlap,                 //
                   const stk::mesh::Field<double> &node_coords_field,  //
                   const stk::mesh::Field<double> &elem_radius_field,  //
                   const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("mundy::mech::check_overlap");
  // Do the check on host for easier printing
  // Loop over all pairs of spheres via the element buckets
  bool no_overlap = true;

  const stk::mesh::BucketVector &all_sphere_buckets = bulk_data.get_buckets(stk::topology::ELEM_RANK, selector);
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
          const stk::mesh::Entity &target_sphere = target_bucket[target_bucket_ord];
          if (source_sphere == target_sphere) {
            continue;
          }
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
          const double ssd = distance_between_centers - 2.0 * source_radius;
          if (ssd < -max_allowable_overlap) {
            // The spheres are overlapping too much
            no_overlap = false;
            // std::cout << "Overlap detected between spheres " << t << " and " << s << std::endl;
            // std::cout << "Distance between centers: " << distance_between_centers << std::endl;
            // std::cout << "Overlap: " << ssd << std::endl;
            // std::cout << "Sphere positions: (" << x_i << ", " << y_i << ", " << z_i << ") and (" << x_j << ", " <<
            // y_j
            //           << ", " << z_j << ")" << std::endl;
          }
        }
      }
    }
  }

  if (no_overlap) {
    std::cout << "No overlap detected!" << std::endl;
  } else {
    std::cout << "Overlap detected!" << std::endl;
  }

  Kokkos::Profiling::popRegion();
}
//@}

}  // namespace mech

namespace alens {

//! \name Dynamic spring bind/unbind via Kinetic Monte Carlo
//@{

/// \brief Unbind a crosslinker from a node.
///
/// If the crosslinker isn't connected to the node on the given ordinal. This is a no-op if the crosslinker isn't
/// connected to the node on the given ordinal. We'll return a bool indicating whether the operation was successful.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
/// modification cycle.
///
/// \param bulk_data The bulk data object.
/// \param crosslinker The crosslinker entity.
/// \param conn_ordinal The ordinal of the connection to the crosslinker for which the node will be unbound.
inline bool unbind_crosslinker_from_node(mundy::mesh::BulkData &bulk_data, const stk::mesh::Entity &crosslinker,
                                         const int &conn_ordinal) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "unbind_crosslinker_from_node: The mesh must be in a modification cycle.");
  // If a node already exists at the ordinal, we'll destroy that relation.
  const int num_nodes = bulk_data.num_nodes(crosslinker);
  stk::mesh::Entity const *nodes = bulk_data.begin_nodes(crosslinker);
  stk::mesh::ConnectivityOrdinal const *node_ords = bulk_data.begin_node_ordinals(crosslinker);
  for (int i = 0; i < num_nodes; ++i) {
    if (node_ords[i] == conn_ordinal) {
      // We found the node in the ordinal that we're trying to bind to. We'll attempt to destroy this relation.
      // This doesn't mean that it was sucessfully destroyed. That's up to the bulk data object and will be returned by
      // destroy_relation.
      return bulk_data.destroy_relation(crosslinker, nodes[i], conn_ordinal);
    }
  }

  // If we didn't find the node, this is a no-op.
  return false;
}

/// \brief Connect a crosslinker to a new node.
///
/// If the crosslinker is already connected to to a node on the given ordinal, the operation will fail.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
/// modification cycle.
///
/// \param bulk_data The bulk data object.
/// \param crosslinker The crosslinker entity.
/// \param new_node The new node entity.
/// \param conn_ordinal The ordinal of the connection to the crosslinker for which the new node will be bound.
inline bool bind_crosslinker_to_node(mundy::mesh::BulkData &bulk_data,      //
                                     const stk::mesh::Entity &crosslinker,  //
                                     const stk::mesh::Entity &new_node,     //
                                     const int &conn_ordinal) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "bind_crosslinker_to_node: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(new_node) == stk::topology::NODE_RANK, std::logic_error,
                     "bind_crosslinker_to_node: The node must have NODE_RANK.");

  // Check a node already exists at the ordinal
  const int num_nodes = bulk_data.num_nodes(crosslinker);
  stk::mesh::Entity const *nodes = bulk_data.begin_nodes(crosslinker);
  stk::mesh::ConnectivityOrdinal const *node_ords = bulk_data.begin_node_ordinals(crosslinker);
  for (int i = 0; i < num_nodes; ++i) {
    if (node_ords[i] == conn_ordinal) {
      // We found the node in the ordinal that we're trying to bind to. Fail the operation.
      return false;
    }
  }

  // Declare the new relation.
  bulk_data.declare_relation(crosslinker, new_node, conn_ordinal);

  return true;
}

/// \brief Connect a crosslinker to a new node and unbind the existing node.
///
/// If the crosslinker is already connected to to a node on the given ordinal, we'll destroy that relation and replace
/// it with the new one. We'll return a bool indicating whether the operation was successful.
///
/// A parallel-local mesh modification operation.
///
/// Note, the relation-declarations must be symmetric across all sharers of the involved entities within a
/// modification cycle.
///
/// \param bulk_data The bulk data object.
/// \param crosslinker The crosslinker entity.
/// \param new_node The new node entity.
/// \param conn_ordinal The ordinal of the connection to the crosslinker for which the new node will be bound.
inline bool bind_crosslinker_to_node_unbind_existing(mundy::mesh::BulkData &bulk_data,      //
                                                     const stk::mesh::Entity &crosslinker,  //
                                                     const stk::mesh::Entity &new_node,     //
                                                     const int &conn_ordinal) {
  MUNDY_THROW_ASSERT(bulk_data.in_modifiable_state(), std::logic_error,
                     "bind_crosslinker_to_node: The mesh must be in a modification cycle.");
  MUNDY_THROW_ASSERT(bulk_data.entity_rank(new_node) == stk::topology::NODE_RANK, std::logic_error,
                     "bind_crosslinker_to_node: The node must have NODE_RANK.");

  // If a node already exists at the ordinal, we'll destroy that relation.
  unbind_crosslinker_from_node(bulk_data, crosslinker, conn_ordinal);

  // Declare the new relation.
  bulk_data.declare_relation(crosslinker, new_node, conn_ordinal);

  return true;
}

struct HookeanCrosslinkerBindRateHeterochromatin {
  HookeanCrosslinkerBindRateHeterochromatin(const mundy::mesh::BulkData &bulk_data,             //
                                            const double &kt,                                   //
                                            const double &crosslinker_binding_rate,             //
                                            const double &crosslinker_spring_constant,          //
                                            const double &crosslinker_rest_length,              //
                                            const stk::mesh::Field<double> &node_coords_field,  //
                                            const stk::mesh::Selector &heterochromatin_selector)
      : bulk_data_(bulk_data),
        inv_kt_(1.0 / kt),
        crosslinker_binding_rate_(crosslinker_binding_rate),
        crosslinker_spring_constant_(crosslinker_spring_constant),
        crosslinker_rest_length_(crosslinker_rest_length),
        node_coord_field_(node_coords_field) {
    heterochromatin_selector.get_parts(heterochromatin_parts_);
  }

  double operator()(const stk::mesh::Entity &crosslinker, const stk::mesh::Entity &bind_site) const {
    // Get the distance from the crosslinker's left node and the bind site
    const stk::mesh::Entity left_node = bulk_data_.begin_nodes(crosslinker)[0];
    const double *left_node_coords = stk::mesh::field_data(node_coord_field_, left_node);
    const double *bind_site_coords = stk::mesh::field_data(node_coord_field_, bind_site);

    const double dx = bind_site_coords[0] - left_node_coords[0];
    const double dy = bind_site_coords[1] - left_node_coords[1];
    const double dz = bind_site_coords[2] - left_node_coords[2];
    const double dr_mag = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

    // Z = A * exp(-0.5 * 1/kt * k * (dr - r0)^2)
    // A = crosslinker_binding_rates
    // k = crosslinker_spring_constant
    // r0 = crosslinker_spring_rest_length
    const double A = crosslinker_binding_rate_;
    const double k = crosslinker_spring_constant_;
    const double r0 = crosslinker_rest_length_;
    double Z = A * Kokkos::exp(-0.5 * inv_kt_ * k * (dr_mag - r0) * (dr_mag - r0));
    return Z;
  }

  const mundy::mesh::BulkData &bulk_data_;
  const double inv_kt_;
  const double crosslinker_binding_rate_;
  const double crosslinker_spring_constant_;
  const double crosslinker_rest_length_;
  const stk::mesh::Field<double> &node_coord_field_;
  stk::mesh::PartVector heterochromatin_parts_;
};

struct FeneCrosslinkerBindRateHeterochromatin {
  FeneCrosslinkerBindRateHeterochromatin(const mundy::mesh::BulkData &bulk_data,             //
                                         const double &kt,                                   //
                                         const double &crosslinker_binding_rate,             //
                                         const double &crosslinker_spring_constant,          //
                                         const double &crosslinker_rmax,                     //
                                         const stk::mesh::Field<double> &node_coords_field,  //
                                         const stk::mesh::Selector &heterochromatin_selector)
      : bulk_data_(bulk_data),
        inv_kt_(1.0 / kt),
        crosslinker_binding_rate_(crosslinker_binding_rate),
        crosslinker_spring_constant_(crosslinker_spring_constant),
        crosslinker_rmax_(crosslinker_rmax),
        node_coord_field_(node_coords_field) {
    heterochromatin_selector.get_parts(heterochromatin_parts_);
  }

  double operator()(const stk::mesh::Entity &crosslinker, const stk::mesh::Entity &bind_site) const {
    // Get the distance from the crosslinker's left node and the bind site
    const stk::mesh::Entity left_node = bulk_data_.begin_nodes(crosslinker)[0];
    const double *left_node_coords = stk::mesh::field_data(node_coord_field_, left_node);
    const double *bind_site_coords = stk::mesh::field_data(node_coord_field_, bind_site);

    const double dx = bind_site_coords[0] - left_node_coords[0];
    const double dy = bind_site_coords[1] - left_node_coords[1];
    const double dz = bind_site_coords[2] - left_node_coords[2];
    const double dr_mag = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

    // Z = A * (1 - (r/r0)^2)^(0.5 * 1/kt * k * rmax^2)
    // A = crosslinker_binding_rates
    // k = crosslinker_spring_constant
    // rmax = crosslinker_spring_max_length (FENE)
    // R = crosslinker_fene_max_distance
    const double A = crosslinker_binding_rate_;
    const double k = crosslinker_spring_constant_;
    const double rmax = crosslinker_rmax_;
    double Z = A * std::pow(1.0 - (dr_mag / rmax) * (dr_mag / rmax), 0.5 * inv_kt_ * k * rmax * rmax);
    return Z;
  }

  const mundy::mesh::BulkData &bulk_data_;
  const double inv_kt_;
  const double crosslinker_binding_rate_;
  const double crosslinker_spring_constant_;
  const double crosslinker_rmax_;
  const stk::mesh::Field<double> &node_coord_field_;
  stk::mesh::PartVector heterochromatin_parts_;
};

template <typename CrosslinkerBindRateHeterochromatin, typename CrosslinkerBindRatePeriphery>
struct CrosslinkerBindRateHeterochromatinOrPeriphery {
  CrosslinkerBindRateHeterochromatinOrPeriphery(
      const stk::mesh::BulkData &bulk_data,                                             //
      const CrosslinkerBindRateHeterochromatin &crosslinker_bind_rate_heterochromatin,  //
      const CrosslinkerBindRatePeriphery &crosslinker_bind_rate_periphery,              //
      const stk::mesh::Selector &heterochromatin_selector,                              //
      const stk::mesh::Selector &periphery_selector)
      : bulk_data_(bulk_data),
        crosslinker_bind_rate_heterochromatin_(crosslinker_bind_rate_heterochromatin),
        crosslinker_bind_rate_periphery_(crosslinker_bind_rate_periphery) {
    heterochromatin_selector.get_parts(heterochromatin_parts_);
    periphery_selector.get_parts(periphery_parts_);
  }

  double operator()(const stk::mesh::Entity &crosslinker, const stk::mesh::Entity &bind_site) const {
    // Determine if the bind site is in the heterochromatin or periphery
    bool is_heterochromatin = false;
    for (const stk::mesh::Part *part : heterochromatin_parts_) {
      if (bulk_data_.bucket(bind_site).member(*part)) {
        is_heterochromatin = true;
        break;
      }
    }
    bool is_periphery = false;
    for (const stk::mesh::Part *part : periphery_parts_) {
      if (bulk_data_.bucket(bind_site).member(*part)) {
        is_periphery = true;
        break;
      }
    }

    if (is_heterochromatin) {
      return crosslinker_bind_rate_heterochromatin_(crosslinker, bind_site);
    } else if (is_periphery) {
      return crosslinker_bind_rate_periphery_(crosslinker, bind_site);
    } else {
      MUNDY_THROW_ASSERT(false, std::logic_error, "Bind site is not in heterochromatin or periphery.");
    }
  }

  const mundy::mesh::BulkData &bulk_data_;
  const CrosslinkerBindRateHeterochromatin &crosslinker_bind_rate_heterochromatin_;
  const CrosslinkerBindRatePeriphery &crosslinker_bind_rate_periphery_;
  stk::mesh::PartVector heterochromatin_parts_;
  stk::mesh::PartVector periphery_parts_;
};

template <typename LeftToDoublyStateChangeRate>
void kmc_perform_state_change_left_bound(mundy::mesh::BulkData &bulk_data,                                            //
                                         const double timestep_size,                                                  //
                                         const LeftToDoublyStateChangeRate &left_to_doubly_state_change_rate_getter,  //
                                         const stk::mesh::Field<double> &neighboring_bind_sites_field,                //
                                         const stk::mesh::Field<double> &el_rng_field,                                //
                                         const stk::mesh::Selector &left_bound_springs_selector,                      //
                                         const stk::mesh::Selector &doubly_bound_springs_selector) {
  Kokkos::Profiling::pushRegion("mundy::alens::kmc_perform_state_change_left_bound");

  MUNDY_THROW_REQUIRE(bulk_data.in_modifiable_state(), std::logic_error, "Bulk data is not in a modification cycle.");

  // Get the vector of left/right bound parts in the selector
  stk::mesh::PartVector left_bound_spring_parts;
  stk::mesh::PartVector doubly_bound_spring_parts;
  left_bound_springs_selector.get_parts(left_bound_spring_parts);
  doubly_bound_springs_selector.get_parts(doubly_bound_spring_parts);

  // Get the vector of entities to modify
  stk::mesh::EntityVector left_bound_springs;
  stk::mesh::get_selected_entities(left_bound_springs_selector, bulk_data.buckets(stk::topology::ELEM_RANK),
                                   left_bound_springs);

  for (const stk::mesh::Entity &left_bound_spring : left_bound_springs) {
    const double *neighboring_bind_sites = stk::mesh::field_data(neighboring_bind_sites_field, left_bound_spring);
    const int num_neighboring_bind_sites = neighboring_bind_sites[0];

    double z_tot = 0.0;
    for (int s = 0; s < num_neighboring_bind_sites; ++s) {
      const auto &neighboring_bind_site_index = static_cast<stk::mesh::FastMeshIndex>(neighboring_bind_sites[s + 1]);
      const stk::mesh::Entity &bind_site = (*(bulk_data.buckets(
          stk::topology::NODE_RANK)[neighboring_bind_site_index.bucket_id]))[neighboring_bind_site_index.bucket_ord];
      const double z_i = timestep_size * left_to_doubly_state_change_rate_getter(left_bound_spring, bind_site);
      z_tot += z_i;
    }

    // Fetch the RNG state, get a random number out of it, and increment
    unsigned *rng_counter = stk::mesh::field_data(el_rng_field, left_bound_spring);
    const stk::mesh::EntityId spring_gid = bulk_data.identifier(left_bound_spring);
    openrand::Philox rng(spring_gid, rng_counter[0]);
    const double randu01 = rng.rand<double>();
    rng_counter[0]++;

    // Notice that the sum of all probabilities is 1.
    // The probability of nothing happening is
    //   std::exp(-z_tot)
    // The probability of an individual event happening is
    //   z_i / z_tot * (1 - std::exp(-z_tot))
    //
    // This is (by construction) true since
    //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
    //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
    //
    // This means that binding only happens if randu01 < (1 - std::exp(-z_tot))
    const double probability_of_no_state_change = 1.0 - std::exp(-z_tot);
    if (randu01 < probability_of_no_state_change) {
      // Binding occurs. Loop back over the neighbor bind sites to see which one we bind to.
      const double scale_factor = probability_of_no_state_change * timestep_size / z_tot;
      double cumsum = 0.0;
      for (int s = 0; s < num_neighboring_bind_sites; ++s) {
        const auto &neighboring_bind_site_index = static_cast<stk::mesh::FastMeshIndex>(neighboring_bind_sites[s + 1]);
        const stk::mesh::Entity &bind_site = (*(bulk_data.buckets(
            stk::topology::NODE_RANK)[neighboring_bind_site_index.bucket_id]))[neighboring_bind_site_index.bucket_ord];
        const double binding_probability =
            scale_factor * left_to_doubly_state_change_rate_getter(left_bound_spring, bind_site);
        cumsum += binding_probability;
        if (randu01 < cumsum) {
          // Bind to the given site
          const int right_node_index = 1;
          const bool bind_worked =
              bind_crosslinker_to_node_unbind_existing(bulk_data, left_bound_spring, bind_site, right_node_index);
          MUNDY_THROW_ASSERT(bind_worked, std::logic_error, "Failed to bind crosslinker to node.");

          std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Binding crosslinker "
                    << bulk_data.identifier(left_bound_spring) << " to node " << bulk_data.identifier(bind_site)
                    << std::endl;

          // Now change the part from left to doubly bound. Add to doubly bound, remove
          // from left bound
          const bool is_spring_locally_owned =
              bulk_data.parallel_owner_rank(left_bound_spring) == bulk_data.parallel_rank();
          if (is_spring_locally_owned) {
            bulk_data.change_entity_parts(left_bound_spring, doubly_bound_spring_parts, left_bound_spring_parts);
          }
        }
      }
    }
  }

  Kokkos::Profiling::popRegion();
}

template <typename LeftToDoublyStateChangeRate>
void kmc_perform_state_change_doubly_bound(
    mundy::mesh::BulkData &bulk_data,                                            //
    const double timestep_size,                                                  //
    const LeftToDoublyStateChangeRate &doubly_to_left_state_change_rate_getter,  //
    const stk::mesh::Field<double> &el_rng_field,                                //
    const stk::mesh::Selector &left_bound_springs_selector,                      //
    const stk::mesh::Selector &doubly_bound_springs_selector) {
  Kokkos::Profiling::pushRegion("mundy::alens::kmc_perform_state_change_doubly_bound");

  MUNDY_THROW_REQUIRE(bulk_data.in_modifiable_state(), std::logic_error, "Bulk data is not in a modification cycle.");

  // Get the vector of left/right bound parts in the selector
  stk::mesh::PartVector left_bound_spring_parts;
  stk::mesh::PartVector doubly_bound_spring_parts;
  left_bound_springs_selector.get_parts(left_bound_spring_parts);
  doubly_bound_springs_selector.get_parts(doubly_bound_spring_parts);

  // Get the vector of entities to modify
  stk::mesh::EntityVector doubly_bound_springs;
  stk::mesh::get_selected_entities(doubly_bound_springs_selector, bulk_data.buckets(stk::topology::ELEM_RANK),
                                   doubly_bound_springs);

  for (const stk::mesh::Entity &doubly_bound_spring : doubly_bound_springs) {
    double z_tot = timestep_size * doubly_to_left_state_change_rate_getter(doubly_bound_spring);

    // Fetch the RNG state, get a random number out of it, and increment
    unsigned *rng_counter = stk::mesh::field_data(el_rng_field, doubly_bound_spring);
    const stk::mesh::EntityId spring_gid = bulk_data.identifier(doubly_bound_spring);
    openrand::Philox rng(spring_gid, rng_counter[0]);
    const double randu01 = rng.rand<double>();
    rng_counter[0]++;

    // Notice that the sum of all probabilities is 1.
    // The probability of nothing happening is
    //   std::exp(-z_tot)
    // The probability of an individual event happening is
    //   z_i / z_tot * (1 - std::exp(-z_tot))
    //
    // This is (by construction) true since
    //  std::exp(-z_tot) + sum_i(z_i / z_tot * (1 - std::exp(-z_tot)))
    //    = std::exp(-z_tot) + (1 - std::exp(-z_tot)) = 1
    //
    // This means that unbinding only happens if randu01 < (1 - std::exp(-z_tot))
    // For now, its either transition to right bound or nothing
    const double probability_of_no_state_change = 1.0 - Kokkos::exp(-z_tot);
    if (randu01 < probability_of_no_state_change) {
      // Unbind the right side of the crosslinker from the current node and bind it to
      // the left crosslinker node
      const stk::mesh::Entity &left_node = bulk_data.begin_nodes(doubly_bound_spring)[0];
      const int right_node_index = 1;
      const bool unbind_worked =
          bind_crosslinker_to_node_unbind_existing(bulk_data, doubly_bound_spring, left_node, right_node_index);
      MUNDY_THROW_ASSERT(unbind_worked, std::logic_error, "Failed to unbind crosslinker from node.");

      std::cout << "Rank: " << stk::parallel_machine_rank(MPI_COMM_WORLD) << " Unbinding crosslinker "
                << bulk_data.identifier(doubly_bound_spring) << " from node "
                << bulk_data.identifier(bulk_data.begin_nodes(doubly_bound_spring)[1]) << std::endl;

      // Now change the part from doubly to left bound. Add to left bound, remove from
      // doubly bound
      const bool is_spring_locally_owned =
          bulk_data.parallel_owner_rank(doubly_bound_spring) == bulk_data.parallel_rank();
      if (is_spring_locally_owned) {
        bulk_data.change_entity_parts(doubly_bound_spring, left_bound_spring_parts, doubly_bound_spring_parts);
      }
    }
  }

  Kokkos::Profiling::popRegion();
}
//@}

//! \name Misc domain-specific physics
//@{

void compute_brownian_velocity(stk::mesh::NgpMesh &ngp_mesh,                      //
                               const double &timestep_size,                       //
                               const double &kt,                                  //
                               const double &viscosity,                           //
                               stk::mesh::NgpField<double> &node_velocity_field,  //
                               stk::mesh::NgpField<unsigned> &elem_rng_field,     //
                               stk::mesh::NgpField<double> &elem_radius_field,    //
                               const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

  node_velocity_field.sync_to_device();
  elem_rng_field.sync_to_device();
  elem_radius_field.sync_to_device();

  constexpr double pi = Kokkos::numbers::pi_v<double>;
  const double six_pi_mu = 6.0 * pi * viscosity;
  const double sqrt_6_pi_mu = Kokkos::sqrt(six_pi_mu);
  const double inv_six_pi_mu = 1.0 / six_pi_mu;
  const double inv_dt = 1.0 / timestep_size;
  const double sqrt_2_kt = Kokkos::sqrt(2.0 * kt);

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::Entity node = nodes[0];
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);

        // Setup the rng
        const stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::EntityId sphere_gid = ngp_mesh.identifier(sphere);
        auto rng_counter = elem_rng_field(sphere_index);
        openrand::Philox rng(sphere_gid, rng_counter[0]);

        // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
        // for drag coeff gamma = 6 * pi * mu * r
        const double sphere_radius = elem_radius_field(sphere_index, 0);
        auto node_velocity = node_velocity_field(node_index);
        const double coeff =
            sqrt_2_kt * Kokkos::sqrt(sqrt_6_pi_mu * sphere_radius * inv_dt) * inv_six_pi_mu / sphere_radius;
        node_velocity[0] += coeff * rng.randn<double>();
        node_velocity[1] += coeff * rng.randn<double>();
        node_velocity[2] += coeff * rng.randn<double>();
        rng_counter[0]++;
      });

  Kokkos::Profiling::popRegion();
}
//@}

}  // namespace alens

}  // namespace mundy

namespace mundy {

namespace chromalens {

void print_rank0(auto thing_to_print, int indent_level = 0) {
  if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
    std::string indent(indent_level * 2, ' ');
    std::cout << indent << thing_to_print << std::endl;
  }
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
    return std::string("NODE_COORDS");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings

//! \name Chromatin position generators
//@{

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_from_file(
    const std::string &file_name, const unsigned num_chromosomes) {
  // The file should be formatted as follows:
  // chromosome_id x y z
  // 0 x1 y1 z1
  // 0 x2 y2 z2
  // ...
  // 1 x1 y1 z1
  // 1 x2 y2 z2
  //
  // chromosome_id should start at 1 and increase by 1 for each new chromosome.
  //
  // And so on for each chromosome. The total number of chromosomes should match the expected total, lest we throw an
  // exception.
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  std::ifstream infile(file_name);
  MUNDY_THROW_REQUIRE(infile.is_open(), std::invalid_argument, fmt::format("Could not open file {}", file_name));

  // Read each line. While the chromosome_id is the same, keep adding nodes to the chromosome.
  size_t current_chromosome_id = 1;
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    int chromosome_id;
    double x, y, z;
    if (!(iss >> chromosome_id >> x >> y >> z)) {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Could not parse line {}", line));
    }
    if (chromosome_id != current_chromosome_id) {
      // We are starting a new chromosome
      MUNDY_THROW_REQUIRE(chromosome_id == current_chromosome_id + 1, std::invalid_argument,
                          "Chromosome IDs should be sequential.");
      MUNDY_THROW_REQUIRE(chromosome_id <= num_chromosomes, std::invalid_argument,
                          fmt::format("Chromosome ID {} is greater than the number of chromosomes.", chromosome_id));
      current_chromosome_id = chromosome_id;
    }
    // Add the node to the chromosome
    all_chromosome_positions[current_chromosome_id - 1].emplace_back(x, y, z);
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_grid(
    const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  const mundy::math::Vector3d alignment_dir{0.0, 0.0, 1.0};
  for (size_t j = 0; j < num_chromosomes; j++) {
    all_chromosome_positions[j].reserve(num_nodes_per_chromosome);
    openrand::Philox rng(j, 0);
    mundy::math::Vector3d start_pos(2.0 * static_cast<double>(j), 0.0, 0.0);
    for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
      const auto pos = start_pos + static_cast<double>(i) * segment_length * alignment_dir;
      all_chromosome_positions[j].emplace_back(pos);
    }
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_random_unit_cell(
    const unsigned num_chromosomes,           //
    const unsigned num_nodes_per_chromosome,  //
    const double segment_length,              //
    const double domain_low[3],               //
    const double domain_high[3]) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  for (size_t j = 0; j < num_chromosomes; j++) {
    all_chromosome_positions[j].reserve(num_nodes_per_chromosome);

    // Find a random place within the unit cell with a random orientation for the chain.
    openrand::Philox rng(j, 0);
    mundy::math::Vector3d pos_start{rng.uniform<double>(domain_low[0], domain_high[0]),
                                    rng.uniform<double>(domain_low[1], domain_high[1]),
                                    rng.uniform<double>(domain_low[2], domain_high[2])};

    // Find a random unit vector direction
    const double zrand = rng.rand<double>() - 1.0;
    const double wrand = std::sqrt(1.0 - zrand * zrand);
    const double trand = 2.0 * M_PI * rng.rand<double>();
    mundy::math::Vector3d u_hat{wrand * std::cos(trand), wrand * std::sin(trand), zrand};

    for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
      auto pos = pos_start + static_cast<double>(i) * segment_length * u_hat;
      all_chromosome_positions[j].emplace_back(pos);
    }
  }

  return all_chromosome_positions;
}

std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_hilbert_random_unit_cell(
    const unsigned num_chromosomes,           //
    const unsigned num_nodes_per_chromosome,  //
    const double segment_length,              //
    const double domain_low[3],               //
    const double domain_high[3]) {
  std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
  std::vector<mundy::geom::Point<double>> chromosome_centers_array(num_chromosomes);
  std::vector<double> chromosome_radii_array(num_chromosomes);
  for (size_t ichromosome = 0; ichromosome < num_chromosomes; ichromosome++) {
    // Generate a random unit vector (will be used for creating the location of the nodes, the random position in
    // the unit cell will be handled later).
    openrand::Philox rng(ichromosome, 0);
    const double zrand = rng.rand<double>() - 1.0;
    const double wrand = std::sqrt(1.0 - zrand * zrand);
    const double trand = 2.0 * M_PI * rng.rand<double>();
    mundy::math::Vector3d u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

    // Once we have the number of chromosome spheres we can get the hilbert curve set up. This will be at some
    // orientation and then have sides with a length of initial_chromosome_separation.
    auto [hilbert_position_array, hilbert_directors] =
        mundy::math::create_hilbert_positions_and_directors(num_nodes_per_chromosome, u_hat, segment_length);

    // Create the local positions of the spheres
    std::vector<mundy::math::Vector3d> sphere_position_array;
    for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
      sphere_position_array.push_back(hilbert_position_array[isphere]);
    }

    // Figure out where the center of the chromosome is, and its radius, in its own local space
    mundy::math::Vector3d r_chromosome_center_local(0.0, 0.0, 0.0);
    double r_max = 0.0;
    for (size_t i = 0; i < sphere_position_array.size(); i++) {
      r_chromosome_center_local += sphere_position_array[i];
    }
    r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
    for (size_t i = 0; i < sphere_position_array.size(); i++) {
      r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
    }

    // Do max_trials number of insertion attempts to get a random position and orientation within the unit cell that
    // doesn't overlap with exiting chromosomes.
    const size_t max_trials = 1000;
    size_t itrial = 0;
    bool chromosome_inserted = false;
    while (itrial <= max_trials) {
      // Generate a random position within the unit cell.
      mundy::math::Vector3d r_start(rng.uniform<double>(domain_low[0], domain_high[0]),
                                    rng.uniform<double>(domain_low[1], domain_high[1]),
                                    rng.uniform<double>(domain_low[2], domain_high[2]));

      // Check for overlaps with existing chromosomes
      bool found_overlap = false;
      for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
        double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
        if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
          found_overlap = true;
          break;
        }
      }
      if (found_overlap) {
        itrial++;
      } else {
        chromosome_inserted = true;
        chromosome_centers_array[ichromosome] = r_start;
        chromosome_radii_array[ichromosome] = r_max;
        break;
      }
    }
    MUNDY_THROW_REQUIRE(chromosome_inserted, std::runtime_error,
                        fmt::format("Failed to insert chromosome after {} trials.", max_trials));

    // Generate all the positions along the curve due to the placement in the global space
    const size_t num_nodes_per_chromosome = sphere_position_array.size();
    for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
      all_chromosome_positions[ichromosome].emplace_back(chromosome_centers_array.back() + r_chromosome_center_local -
                                                         sphere_position_array[i]);
    }
  }

  return all_chromosome_positions;
}
//@}

//! \name Periphery collision forces
//@{

void compute_periphery_collision_forces(stk::mesh::NgpMesh &ngp_mesh,                        //
                                        const double &periphery_collision_spring_constant,   //
                                        const mundy::geom::Sphere<double> &periphery_shape,  //
                                        stk::mesh::NgpField<double> &node_coords_field,      //
                                        stk::mesh::NgpField<double> &node_force_field,       //
                                        stk::mesh::NgpField<double> &elem_radius_field,      //
                                        const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("HP1::compute_spherical_periphery_collision_forces");

  node_coords_field.sync_to_device();
  node_force_field.sync_to_device();
  elem_radius_field.sync_to_device();

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::Entity node = nodes[0];
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
        const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node_index);
        const double sphere_radius = elem_radius_field(node_index, 0);
        const double node_coords_norm = mundy::math::two_norm(node_coords);
        const double shared_normal_ssd = periphery_shape.radius() - node_coords_norm - sphere_radius;
        if (shared_normal_ssd < 0.0) {
          auto node_force = mundy::mesh::vector3_field_data(node_force_field, node_index);
          auto inward_normal = -node_coords / node_coords_norm;
          node_force[0] -= periphery_collision_spring_constant * inward_normal[0] * shared_normal_ssd;
          node_force[1] -= periphery_collision_spring_constant * inward_normal[1] * shared_normal_ssd;
          node_force[2] -= periphery_collision_spring_constant * inward_normal[2] * shared_normal_ssd;
        }
      });

  node_force_field.modify_on_device();
  Kokkos::Profiling::popRegion();
}

void compute_periphery_collision_forces(stk::mesh::NgpMesh &ngp_mesh,                           //
                                        const double &periphery_collision_spring_constant,      //
                                        const mundy::geom::Ellipsoid<double> &periphery_shape,  //
                                        stk::mesh::NgpField<double> &node_coords_field,         //
                                        stk::mesh::NgpField<double> &node_force_field,          //
                                        stk::mesh::NgpField<double> &elem_radius_field,         //
                                        stk::mesh::NgpField<double> &element_aabb_field,        //
                                        const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("HP1::compute_ellipsoidal_periphery_collision_forces");

  // Setup the ellipsoid level set function
  const double a = 0.5 * periphery_shape.axis_length_1();
  const double b = 0.5 * periphery_shape.axis_length_2();
  const double c = 0.5 * periphery_shape.axis_length_3();
  const double inv_a2 = 1.0 / (a * a);
  const double inv_b2 = 1.0 / (b * b);
  const double inv_c2 = 1.0 / (c * c);
  auto level_set = [&inv_a2, &inv_b2, &inv_c2, &periphery_shape](const mundy::math::Vector3d &point) -> double {
    const auto body_frame_point =
        mundy::math::conjugate(periphery_shape.orientation()) * (point - periphery_shape.center());
    return (body_frame_point[0] * body_frame_point[0] * inv_a2 + body_frame_point[1] * body_frame_point[1] * inv_b2 +
            body_frame_point[2] * body_frame_point[2] * inv_c2) -
           1;
  };

  // Evaluate the potential
  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEMENT_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        // For our coarse search, we check if the coners of the sphere's aabb lie inside the ellipsoidal periphery
        // This can be done via the (body frame) inside outside unftion f(x, y, z) = 1 - (x^2/a^2 + y^2/b^2  z^2/c^2)
        // This is possible due to the convexity of the ellipsoid
        auto aabb = element_aabb_field(sphere_index);
        const double &x0 = aabb[0];
        const double &y0 = aabb[1];
        const double &z0 = aabb[2];
        const double &x1 = aabb[3];
        const double &y1 = aabb[4];
        const double &z1 = aabb[5];

        // Compute all 8 corners of the AABB
        const auto bottom_left_front = mundy::math::Vector3d(x0, y0, z0);
        const auto bottom_right_front = mundy::math::Vector3d(x1, y0, z0);
        const auto top_left_front = mundy::math::Vector3d(x0, y1, z0);
        const auto top_right_front = mundy::math::Vector3d(x1, y1, z0);
        const auto bottom_left_back = mundy::math::Vector3d(x0, y0, z1);
        const auto bottom_right_back = mundy::math::Vector3d(x1, y0, z1);
        const auto top_left_back = mundy::math::Vector3d(x0, y1, z1);
        const auto top_right_back = mundy::math::Vector3d(x1, y1, z1);
        const double all_points_inside_periphery =
            level_set(bottom_left_front) < 0.0 && level_set(bottom_right_front) < 0.0 &&
            level_set(top_left_front) < 0.0 && level_set(top_right_front) < 0.0 && level_set(bottom_left_back) < 0.0 &&
            level_set(bottom_right_back) < 0.0 && level_set(top_left_back) < 0.0 && level_set(top_right_back) < 0.0;

        if (!all_points_inside_periphery) {
          // We might have a collision, perform the more expensive check
          stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
          const stk::mesh::Entity node = nodes[0];
          const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);
          const auto node_coords = mundy::mesh::vector3_field_data(node_coords_field, node_index);
          const double sphere_radius = elem_radius_field(sphere_index, 0);

          // Note, the ellipsoid for the ssd calc has outward normal, whereas the periphery has inward normal.
          // Hence, the sign flip.
          mundy::math::Vector3d contact_point;
          mundy::math::Vector3d ellipsoid_nhat;
          const double shared_normal_ssd = -mundy::math::distance::shared_normal_ssd_between_ellipsoid_and_point(
                                               periphery_shape.center(), periphery_shape.orientation(), a, b, c,
                                               node_coords, contact_point, ellipsoid_nhat) -
                                           sphere_radius;

          if (shared_normal_ssd < 0.0) {
            // We have a collision, compute the force
            auto node_force = mundy::mesh::vector3_field_data(node_force_field, node_index);
            auto periphery_nhat = -ellipsoid_nhat;
            node_force[0] -= periphery_collision_spring_constant * periphery_nhat[0] * shared_normal_ssd;
            node_force[1] -= periphery_collision_spring_constant * periphery_nhat[1] * shared_normal_ssd;
            node_force[2] -= periphery_collision_spring_constant * periphery_nhat[2] * shared_normal_ssd;
          }
        }
      });

  node_force_field.modify_on_device();
  Kokkos::Profiling::popRegion();
}
//@}

//! \name Misc problem-specific physics
//@{

void compute_brownian_velocity(stk::mesh::NgpMesh &ngp_mesh,                      //
                               const double &timestep_size,                       //
                               const double &kt,                                  //
                               const double &viscosity,                           //
                               stk::mesh::NgpField<double> &node_velocity_field,  //
                               stk::mesh::NgpField<unsigned> &elem_rng_field,     //
                               stk::mesh::NgpField<double> &elem_radius_field,    //
                               const stk::mesh::Selector &selector) {
  Kokkos::Profiling::pushRegion("HP1::compute_brownian_velocity");

  node_velocity_field.sync_to_device();
  elem_rng_field.sync_to_device();
  elem_radius_field.sync_to_device();

  constexpr double pi = Kokkos::numbers::pi_v<double>;
  const double six_pi_mu = 6.0 * pi * viscosity;
  const double sqrt_6_pi_mu = Kokkos::sqrt(six_pi_mu);
  const double inv_six_pi_mu = 1.0 / six_pi_mu;
  const double inv_dt = 1.0 / timestep_size;
  const double sqrt_2_kt = Kokkos::sqrt(2.0 * kt);

  mundy::mesh::for_each_entity_run(
      ngp_mesh, stk::topology::ELEM_RANK, selector, KOKKOS_LAMBDA(const stk::mesh::FastMeshIndex &sphere_index) {
        stk::mesh::NgpMesh::ConnectedNodes nodes = ngp_mesh.get_nodes(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::Entity node = nodes[0];
        const stk::mesh::FastMeshIndex node_index = ngp_mesh.fast_mesh_index(node);

        // Setup the rng
        const stk::mesh::Entity sphere = ngp_mesh.get_entity(stk::topology::ELEM_RANK, sphere_index);
        const stk::mesh::EntityId sphere_gid = ngp_mesh.identifier(sphere);
        auto rng_counter = elem_rng_field(sphere_index);
        openrand::Philox rng(sphere_gid, rng_counter[0]);

        // U_brown = sqrt(2 * kt * gamma / dt) * randn / gamma
        // for drag coeff gamma = 6 * pi * mu * r
        const double sphere_radius = elem_radius_field(sphere_index, 0);
        auto node_velocity = node_velocity_field(node_index);
        const double coeff =
            sqrt_2_kt * Kokkos::sqrt(sqrt_6_pi_mu * sphere_radius * inv_dt) * inv_six_pi_mu / sphere_radius;
        node_velocity[0] += coeff * rng.randn<double>();
        node_velocity[1] += coeff * rng.randn<double>();
        node_velocity[2] += coeff * rng.randn<double>();
        rng_counter[0]++;
      });

  Kokkos::Profiling::popRegion();
}
//@}

//! \name Simulation setup/run
//@{

struct HP1ParamParser {
  void print_help_message() {
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "To run this code, please pass in --params=<input.yaml> as a command line argument." << std::endl;
    std::cout << std::endl;
    std::cout << "Note, all parameters and sublists in input.yaml must be contained in a single top-level list."
              << std::endl;
    std::cout << "Such as:" << std::endl;
    std::cout << std::endl;
    std::cout << "HP1:" << std::endl;
    std::cout << "  num_time_steps: 1000" << std::endl;
    std::cout << "  timestep_size: 1e-6" << std::endl;
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "The valid parameters that can be set in the input file are:" << std::endl;
    Teuchos::ParameterList valid_params = get_valid_params();

    auto print_options =
        Teuchos::ParameterList::PrintOptions().showTypes(false).showDoc(true).showDefault(true).showFlags(false).indent(
            1);
    valid_params.print(std::cout, print_options);
    std::cout << "#############################################################################################"
              << std::endl;
  }

  Teuchos::ParameterList parse(int argc, char **argv) {
    // Parse the command line options to find the input filename
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("params", &input_parameter_filename_, "The name of the input file.");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_result = cmdp.parse(argc, argv);
    if (parse_result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      print_help_message();

      // Safely exit the program
      // If we print the help message, we don't need to do anything else.
      exit(0);

    } else if (parse_result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      throw std::invalid_argument("Failed to parse the command line arguments.");
    }

    // Read, validate, and parse in the parameters from the parameter list.
    try {
      Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(input_parameter_filename_);
      return parse(param_list);
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Failed to read the input parameter file." << std::endl;
      std::cerr << "During read, the following error occurred: " << e.what() << std::endl;
      std::cerr << "NOTE: This can happen for any number of reasons. Check that the file exists and contains the "
                   "expected parameters."
                << std::endl;
      throw e;
    }

    return Teuchos::ParameterList();
  }

  Teuchos::ParameterList parse(const Teuchos::ParameterList &param_list) {
    // Validate the parameters and set the defaults.
    Teuchos::ParameterList valid_param_list = param_list;
    valid_param_list.validateParametersAndSetDefaults(get_valid_params());
    check_invariants(valid_param_list);
    dump_parameters(valid_param_list);
    return valid_param_list;
  }

  void check_invariants(const Teuchos::ParameterList &valid_param_list) {
    // Check the sim params
    const auto &sim_params = valid_param_list.sublist("sim");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("timestep_size") > 0, std::invalid_argument,
                        "timestep_size must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("viscosity") > 0, std::invalid_argument,
                        "viscosity must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("initial_chromosome_separation") >= 0, std::invalid_argument,
                        "initial_chromosome_separation must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<bool>("enable_periphery_hydrodynamics")
                            ? sim_params.get<bool>("enable_backbone_n_body_hydrodynamics")
                            : true,
                        std::invalid_argument,
                        "Periphery hydrodynamics requires backbone hydrodynamics to be enabled.");

    MUNDY_THROW_REQUIRE(sim_params.get<std::string>("initialization_type") == "GRID" ||
                            sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL" ||
                            sim_params.get<std::string>("initialization_type") == "OVERLAP_TEST" ||
                            sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL" ||
                            sim_params.get<std::string>("initialization_type") == "USHAPE_TEST" ||
                            sim_params.get<std::string>("initialization_type") == "FROM_EXO" ||
                            sim_params.get<std::string>("initialization_type") == "FROM_DAT",
                        std::invalid_argument,
                        fmt::format("Invalid initialization_type ({}). Valid options are GRID, RANDOM_UNIT_CELL, "
                                    "OVERLAP_TEST, HILBERT_RANDOM_UNIT_CELL, USHAPE_TEST, FROM_EXO, FROM_DAT.",
                                    sim_params.get<std::string>("initialization_type")));

    if (sim_params.get<bool>("enable_backbone_springs")) {
      const auto &backbone_spring_params = valid_param_list.sublist("backbone_springs");
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<std::string>("spring_type") == "HOOKEAN" ||
                              backbone_spring_params.get<std::string>("spring_type") == "FENE",
                          std::invalid_argument,
                          fmt::format("Invalid spring_type ({}). Valid options are HOOKEAN and FENE.",
                                      backbone_spring_params.get<std::string>("spring_type")));
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<double>("spring_constant") >= 0, std::invalid_argument,
                          "spring_constant must be non-negative.");
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<double>("spring_r0") >= 0, std::invalid_argument,
                          "max_spring_length must be non-negative.");
    }

    // Check the periphery_hydro params
    if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
      const auto &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");
      std::string periphery_hydro_shape = periphery_hydro_params.get<std::string>("shape");
      std::string periphery_hydro_quadrature = periphery_hydro_params.get<std::string>("quadrature");
      if (periphery_hydro_quadrature == "GAUSS_LEGENDRE") {
        double periphery_hydro_axis_radius1 = periphery_hydro_params.get<double>("axis_radius1");
        double periphery_hydro_axis_radius2 = periphery_hydro_params.get<double>("axis_radius2");
        double periphery_hydro_axis_radius3 = periphery_hydro_params.get<double>("axis_radius3");
        MUNDY_THROW_REQUIRE(
            (periphery_hydro_shape == "SPHERE") || ((periphery_hydro_shape == "ELLIPSOID") &&
                                                    (periphery_hydro_axis_radius1 == periphery_hydro_axis_radius2) &&
                                                    (periphery_hydro_axis_radius2 == periphery_hydro_axis_radius3) &&
                                                    (periphery_hydro_axis_radius3 == periphery_hydro_axis_radius1)),
            std::invalid_argument, "Gauss-Legendre quadrature is only valid for spherical peripheries.");
      }
    }
  }

  static Teuchos::ParameterList get_valid_params() {
    // Create a paramater entity validator for our large integers to allow for both int and long long.
    auto prefer_size_t = []() {
      if (std::is_same_v<size_t, unsigned short>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_SHORT;
      } else if (std::is_same_v<size_t, unsigned int>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      } else if (std::is_same_v<size_t, unsigned long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG;
      } else if (std::is_same_v<size_t, unsigned long long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG_LONG;
      } else {
        throw std::runtime_error("Unknown size_t type.");
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      }
    }();
    const bool allow_all_types_by_default = false;
    mundy::core::OurAnyNumberParameterEntryValidator::AcceptedTypes accept_int(allow_all_types_by_default);
    accept_int.allow_all_integer_types(true);
    auto make_new_validator = [](const auto &preferred_type, const auto &accepted_types) {
      return Teuchos::rcp(new mundy::core::OurAnyNumberParameterEntryValidator(preferred_type, accepted_types));
    };

    static Teuchos::ParameterList valid_parameter_list;

    valid_parameter_list.sublist("sim")
        .set("num_time_steps", 100, "Number of time steps.", make_new_validator(prefer_size_t, accept_int))
        .set("timestep_size", 0.001, "Time step size.")
        .set("viscosity", 1.0, "Viscosity.")
        // Initialization
        .set("num_chromosomes", static_cast<size_t>(1), "Number of chromosomes.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_hetero_euchromatin_blocks", static_cast<size_t>(2),
             "Number of heterochromatin/euchromatin blocks per chain.", make_new_validator(prefer_size_t, accept_int))
        .set("num_euchromatin_per_block", static_cast<size_t>(1), "Number of euchromatin beads per block.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_heterochromatin_per_block", static_cast<size_t>(1), "Number of heterochromatin beads per block.",
             make_new_validator(prefer_size_t, accept_int))
        .set("backbone_sphere_hydrodynamic_radius", 0.5,
             "Backbone sphere hydrodynamic radius. Even if n-body hydrodynamics is disabled, we still have "
             "self-interaction.")
        .set("check_max_speed_pre_position_update", false, "Check max speed before updating positions.")
        .set("max_allowable_speed", std::numeric_limits<double>::max(),
             "Maximum allowable speed (only used if "
             "check_max_speed_pre_position_update is true).")
        .set("initial_chromosome_separation", 1.0, "Initial chromosome separation.")
        .set("initialization_type", std::string("GRID"),
             "Initialization_type. Valid options are GRID, RANDOM_UNIT_CELL, "
             "OVERLAP_TEST, HILBERT_RANDOM_UNIT_CELL, USHAPE_TEST, "
             "FROM_EXO, FROM_DAT.")
        .set("initialize_from_exo_filename", std::string("HP1"),
             "Exo file to initialize from if initialization_type is FROM_EXO.")
        .set("initialize_from_dat_filename", std::string("HP1_pos.dat"),
             "Dat file to initialize from if initialization_type is FROM_DAT.")
        .set<Teuchos::Array<double>>(
            "domain_low", Teuchos::tuple<double>(0.0, 0.0, 0.0),
            "Lower left corner of the unit cell. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set<Teuchos::Array<double>>(
            "domain_high", Teuchos::tuple<double>(10.0, 10.0, 10.0),
            "Upper right corner of the unit cell. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set("loadbalance_post_initialization", false, "If we should load balance post-initialization or not.")
        // IO
        .set("io_frequency", static_cast<size_t>(10), "Number of timesteps between writing output.",
             make_new_validator(prefer_size_t, accept_int))
        .set("log_frequency", static_cast<size_t>(10), "Number of timesteps between logging.",
             make_new_validator(prefer_size_t, accept_int))
        .set("output_filename", std::string("HP1"), "Output filename.")
        .set("enable_continuation_if_available", true,
             "Enable continuing a previous simulation if an output file already exists.")
        // Control flags
        .set("enable_brownian_motion", true, "Enable chromatin Brownian motion.")
        .set("enable_backbone_springs", true, "Enable backbone springs.")
        .set("enable_backbone_collision", true, "Enable backbone collision.")
        .set("enable_backbone_n_body_hydrodynamics", true, "Enable backbone N-body hydrodynamics.")
        .set("enable_crosslinkers", true, "Enable crosslinkers.")
        .set("enable_periphery_collision", true, "Enable periphery collision.")
        .set("enable_periphery_hydrodynamics", true, "Enable periphery hydrodynamics.")
        .set("enable_periphery_binding", true, "Enable periphery binding.")
        .set("enable_active_euchromatin_forces", true, "Enable active euchromatin forces.");

    valid_parameter_list.sublist("brownian_motion").set("kt", 1.0, "Temperature kT for Brownian Motion.");

    valid_parameter_list.sublist("backbone_springs")
        .set("spring_type", std::string("HOOKEAN"), "Chromatin spring type. Valid options are HOOKEAN or FENE.")
        .set("spring_constant", 100.0, "Chromatin spring constant.")
        .set("spring_r0", 1.0, "Chromatin rest length (HOOKEAN) or rmax (FENE).");

    valid_parameter_list.sublist("backbone_collision")
        .set("backbone_sphere_collision_radius", 0.5, "Backbone sphere collision radius (as so aptly named).")
        .set("max_allowable_overlap", 1e-4, "Maximum allowable overlap between spheres post-collision resolution.")
        .set("max_collision_iterations", static_cast<size_t>(10000),
             "Maximum number of collision iterations. If this is reached, an error will be thrown.",
             make_new_validator(prefer_size_t, accept_int));

    valid_parameter_list.sublist("crosslinker")
        .set("spring_type", std::string("HOOKEAN"), "Crosslinker spring type. Valid options are HOOKEAN or FENE.")
        .set("kt", 1.0, "Temperature kT for crosslinkers.")
        .set("spring_constant", 10.0, "Crosslinker spring constant.")
        .set("spring_r0", 2.5, "Crosslinker rest length.")
        .set("left_binding_rate", 1.0, "Crosslinker left binding rate.")
        .set("right_binding_rate", 1.0, "Crosslinker right binding rate.")
        .set("left_unbinding_rate", 1.0, "Crosslinker left unbinding rate.")
        .set("right_unbinding_rate", 1.0, "Crosslinker right unbinding rate.");

    valid_parameter_list.sublist("periphery_hydro")
        .set("check_max_periphery_overlap", false, "Check max periphery overlap.")
        .set("max_allowed_periphery_overlap", 1e-6, "Maximum allowed periphery overlap.")
        .set("shape", std::string("SPHERE"), "Periphery hydrodynamic shape. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("quadrature", std::string("GAUSS_LEGENDRE"),
             "Periphery quadrature. Valid options are GAUSS_LEGENDRE or "
             "FROM_FILE.")
        .set("spectral_order", static_cast<size_t>(32),
             "Periphery spectral order (only used if periphery is spherical is Gauss-Legendre quadrature).",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_quadrature_points", static_cast<size_t>(1000),
             "Periphery number of quadrature points (only used if quadrature type is FROM_FILE). Number of points in "
             "the files must match this quantity.",
             make_new_validator(prefer_size_t, accept_int))
        .set("quadrature_points_filename", std::string("hp1_periphery_hydro_quadrature_points.dat"),
             "Periphery quadrature points filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_weights_filename", std::string("hp1_periphery_hydro_quadrature_weights.dat"),
             "Periphery quadrature weights filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_normals_filename", std::string("hp1_periphery_hydro_quadrature_normals.dat"),
             "Periphery quadrature normals filename (only used if quadrature type is FROM_FILE).");

    valid_parameter_list.sublist("periphery_collision")
        .set("shape", std::string("SPHERE"), "Periphery collision shape. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("collision_spring_constant", 1000.0, "Periphery collision spring constant.");

    valid_parameter_list.sublist("periphery_binding")
        .set("binding_rate", 1.0, "Periphery binding rate.")
        .set("unbinding_rate", 1.0, "Periphery unbinding rate.")
        .set("spring_constant", 1000.0, "Periphery spring constant.")
        .set("spring_r0", 1.0, "Periphery spring rest length.")
        .set("bind_sites_type", std::string("RANDOM"),
             "Periphery bind sites type. Valid options are RANDOM or FROM_FILE.")
        .set("shape", std::string("SPHERE"),
             "The shape of the binding site locations. Only used if bind_sites_type is RANDOM. Valid options are "
             "SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("num_bind_sites", static_cast<size_t>(1000),
             "Periphery number of binding sites (only used if periphery_binding_sites_type is RANDOM and periphery "
             "has spherical or ellipsoidal shape).",
             make_new_validator(prefer_size_t, accept_int))
        .set("bind_site_locations_filename", std::string("periphery_bind_sites.dat"),
             "Periphery binding sites filename (only used if periphery_binding_sites_type is FROM_FILE).");

    valid_parameter_list.sublist("active_euchromatin_forces")
        .set("force_sigma", 1.0, "Active euchromatin force sigma.")
        .set("kon", 1.0, "Active euchromatin force kon.")
        .set("koff", 1.0, "Active euchromatin force koff.");

    valid_parameter_list.sublist("neighbor_list").set("skin_distance", 1.0, "Neighbor list skin distance.");

    return valid_parameter_list;
  }

  void dump_parameters(const Teuchos::ParameterList &valid_param_list) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;

      std::cout << std::endl;
      const auto &sim_params = valid_param_list.sublist("sim");
      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  num_time_steps:  " << sim_params.get<size_t>("num_time_steps") << std::endl;
      std::cout << "  timestep_size:   " << sim_params.get<double>("timestep_size") << std::endl;
      std::cout << "  viscosity:       " << sim_params.get<double>("viscosity") << std::endl;
      std::cout << "  num_chromosomes: " << sim_params.get<size_t>("num_chromosomes") << std::endl;
      std::cout << "  num_hetero_euchromatin_blocks:      " << sim_params.get<size_t>("num_hetero_euchromatin_blocks")
                << std::endl;
      std::cout << "  num_euchromatin_per_block: " << sim_params.get<size_t>("num_euchromatin_per_block") << std::endl;
      std::cout << "  num_heterochromatin_per_block:  " << sim_params.get<size_t>("num_heterochromatin_per_block")
                << std::endl;
      std::cout << "  backbone_sphere_hydrodynamic_radius: "
                << sim_params.get<double>("backbone_sphere_hydrodynamic_radius") << std::endl;
      std::cout << "  initial_chromosome_separation:   " << sim_params.get<double>("initial_chromosome_separation")
                << std::endl;
      std::cout << "  initialization_type:             " << sim_params.get<std::string>("initialization_type")
                << std::endl;
      if (sim_params.get<std::string>("initialization_type") == "FROM_EXO") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_exo_filename")
                  << std::endl;
      }

      if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_dat_filename")
                  << std::endl;
      }

      if ((sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") ||
          (sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL")) {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        std::cout << "  domain_low: {" << domain_low[0] << ", " << domain_low[1] << ", " << domain_low[2] << "}"
                  << std::endl;
        std::cout << "  domain_high: {" << domain_high[0] << ", " << domain_high[1] << ", " << domain_high[2] << "}"
                  << std::endl;
      }

      std::cout << "  loadbalance_post_initialization: " << sim_params.get<bool>("loadbalance_post_initialization")
                << std::endl;
      std::cout << "  check_max_speed_pre_position_update: "
                << sim_params.get<bool>("check_max_speed_pre_position_update") << std::endl;
      if (sim_params.get<bool>("check_max_speed_pre_position_update")) {
        std::cout << "  max_allowable_speed: " << sim_params.get<double>("max_allowable_speed") << std::endl;
      }
      std::cout << std::endl;

      std::cout << "IO:" << std::endl;
      std::cout << "  io_frequency:    " << sim_params.get<size_t>("io_frequency") << std::endl;
      std::cout << "  log_frequency:   " << sim_params.get<size_t>("log_frequency") << std::endl;
      std::cout << "  output_filename: " << sim_params.get<std::string>("output_filename") << std::endl;
      std::cout << "  enable_continuation_if_available: " << sim_params.get<bool>("enable_continuation_if_available")
                << std::endl;
      std::cout << std::endl;

      std::cout << "CONTROL FLAGS:" << std::endl;
      std::cout << "  enable_brownian_motion: " << sim_params.get<bool>("enable_brownian_motion") << std::endl;
      std::cout << "  enable_backbone_springs:          " << sim_params.get<bool>("enable_backbone_springs")
                << std::endl;
      std::cout << "  enable_backbone_collision:        " << sim_params.get<bool>("enable_backbone_collision")
                << std::endl;
      std::cout << "  enable_backbone_n_body_hydrodynamics:    "
                << sim_params.get<bool>("enable_backbone_n_body_hydrodynamics") << std::endl;
      std::cout << "  enable_crosslinkers:              " << sim_params.get<bool>("enable_crosslinkers") << std::endl;
      std::cout << "  enable_periphery_hydrodynamics:   " << sim_params.get<bool>("enable_periphery_hydrodynamics")
                << std::endl;
      std::cout << "  enable_periphery_collision:       " << sim_params.get<bool>("enable_periphery_collision")
                << std::endl;
      std::cout << "  enable_periphery_binding:         " << sim_params.get<bool>("enable_periphery_binding")
                << std::endl;
      std::cout << "  enable_active_euchromatin_forces: " << sim_params.get<bool>("enable_active_euchromatin_forces")
                << std::endl;

      if (sim_params.get<bool>("enable_brownian_motion")) {
        const auto &brownian_motion_params = valid_param_list.sublist("brownian_motion");

        std::cout << std::endl;
        std::cout << "BROWNIAN MOTION:" << std::endl;
        std::cout << "  kt: " << brownian_motion_params.get<double>("kt") << std::endl;
      }

      if (sim_params.get<bool>("enable_backbone_springs")) {
        const auto &backbone_springs_params = valid_param_list.sublist("backbone_springs");

        std::cout << std::endl;
        std::cout << "BACKBONE SPRINGS:" << std::endl;
        std::cout << "  spring_type:      " << backbone_springs_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  spring_constant:  " << backbone_springs_params.get<double>("spring_constant") << std::endl;
        if (backbone_springs_params.get<std::string>("spring_type") == "HOOKEAN") {
          std::cout << "  spring_r0 (rest_length): " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        } else if (backbone_springs_params.get<std::string>("spring_type") == "FENE") {
          std::cout << "  spring_r0 (r_max):       " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_backbone_collision")) {
        const auto &backbone_collision_params = valid_param_list.sublist("backbone_collision");

        std::cout << std::endl;
        std::cout << "BACKBONE COLLISION:" << std::endl;
        std::cout << "  backbone_sphere_collision_radius: "
                  << backbone_collision_params.get<double>("backbone_sphere_collision_radius") << std::endl;
        std::cout << "  max_allowable_overlap: " << backbone_collision_params.get<double>("max_allowable_overlap")
                  << std::endl;
        std::cout << "  max_collision_iterations: " << backbone_collision_params.get<size_t>("max_collision_iterations")
                  << std::endl;
      }

      if (sim_params.get<bool>("enable_crosslinkers")) {
        const auto &crosslinker_params = valid_param_list.sublist("crosslinker");

        std::cout << std::endl;
        std::cout << "CROSSLINKERS:" << std::endl;
        std::cout << "  spring_type: " << crosslinker_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  kt: " << crosslinker_params.get<double>("kt") << std::endl;
        std::cout << "  spring_constant: " << crosslinker_params.get<double>("spring_constant") << std::endl;
        std::cout << "  spring_r0: " << crosslinker_params.get<double>("spring_r0") << std::endl;
        std::cout << "  left_binding_rate: " << crosslinker_params.get<double>("left_binding_rate") << std::endl;
        std::cout << "  right_binding_rate: " << crosslinker_params.get<double>("right_binding_rate") << std::endl;
        std::cout << "  left_unbinding_rate: " << crosslinker_params.get<double>("left_unbinding_rate") << std::endl;
        std::cout << "  right_unbinding_rate: " << crosslinker_params.get<double>("right_unbinding_rate") << std::endl;
      }

      if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
        const auto &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");

        std::cout << std::endl;
        std::cout << "PERIPHERY HYDRODYNAMICS:" << std::endl;
        std::cout << "  check_max_periphery_overlap: "
                  << periphery_hydro_params.get<bool>("check_max_periphery_overlap") << std::endl;
        if (periphery_hydro_params.get<bool>("check_max_periphery_overlap")) {
          std::cout << "  max_allowed_periphery_overlap: "
                    << periphery_hydro_params.get<double>("max_allowed_periphery_overlap") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_hydro_params.get<double>("radius") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_hydro_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_hydro_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_hydro_params.get<double>("axis_radius3") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("quadrature") == "GAUSS_LEGENDRE") {
          std::cout << "  quadrature: GAUSS_LEGENDRE" << std::endl;
          std::cout << "  spectral_order: " << periphery_hydro_params.get<size_t>("spectral_order") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("quadrature") == "FROM_FILE") {
          std::cout << "  quadrature: FROM_FILE" << std::endl;
          std::cout << "  num_quadrature_points: " << periphery_hydro_params.get<size_t>("num_quadrature_points")
                    << std::endl;
          std::cout << "  quadrature_points_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_points_filename") << std::endl;
          std::cout << "  quadrature_weights_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_weights_filename") << std::endl;
          std::cout << "  quadrature_normals_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_normals_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_periphery_collision")) {
        const auto &periphery_collision_params = valid_param_list.sublist("periphery_collision");

        std::cout << std::endl;
        std::cout << "PERIPHERY COLLISION:" << std::endl;
        if (periphery_collision_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_collision_params.get<double>("radius") << std::endl;
        } else if (periphery_collision_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_collision_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_collision_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_collision_params.get<double>("axis_radius3") << std::endl;
        }
        std::cout << "  collision_spring_constant: "
                  << periphery_collision_params.get<double>("collision_spring_constant") << std::endl;
      }

      if (sim_params.get<bool>("enable_periphery_binding")) {
        const auto &periphery_binding_params = valid_param_list.sublist("periphery_binding");

        std::cout << std::endl;
        std::cout << "PERIPHERY BINDING:" << std::endl;
        std::cout << "  binding_rate: " << periphery_binding_params.get<double>("binding_rate") << std::endl;
        std::cout << "  unbinding_rate: " << periphery_binding_params.get<double>("unbinding_rate") << std::endl;
        std::cout << "  spring_constant: " << periphery_binding_params.get<double>("spring_constant") << std::endl;
        std::cout << "  spring_r0: " << periphery_binding_params.get<double>("spring_r0") << std::endl;
        if (periphery_binding_params.get<std::string>("bind_sites_type") == "RANDOM") {
          std::cout << "  bind_sites_type: RANDOM" << std::endl;
          if (periphery_binding_params.get<std::string>("shape") == "SPHERE") {
            std::cout << "  shape: SPHERE" << std::endl;
            std::cout << "  radius: " << periphery_binding_params.get<double>("radius") << std::endl;
          } else if (periphery_binding_params.get<std::string>("shape") == "ELLIPSOID") {
            std::cout << "  shape: ELLIPSOID" << std::endl;
            std::cout << "  axis_radius1: " << periphery_binding_params.get<double>("axis_radius1") << std::endl;
            std::cout << "  axis_radius2: " << periphery_binding_params.get<double>("axis_radius2") << std::endl;
            std::cout << "  axis_radius3: " << periphery_binding_params.get<double>("axis_radius3") << std::endl;
          }

          std::cout << "  num_bind_sites: " << periphery_binding_params.get<size_t>("num_bind_sites") << std::endl;
        } else if (periphery_binding_params.get<std::string>("bind_sites_type") == "FROM_FILE") {
          std::cout << "  bind_sites_type: FROM_FILE" << std::endl;
          std::cout << "  bind_site_locations_filename: "
                    << periphery_binding_params.get<std::string>("bind_site_locations_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_active_euchromatin_forces")) {
        const auto &active_euchromatin_forces_params = valid_param_list.sublist("active_euchromatin_forces");

        std::cout << std::endl;
        std::cout << "ACTIVE EUCHROMATIN FORCES:" << std::endl;
        std::cout << "  force_sigma: " << active_euchromatin_forces_params.get<double>("force_sigma") << std::endl;
        std::cout << "  kon: " << active_euchromatin_forces_params.get<double>("kon") << std::endl;
        std::cout << "  koff: " << active_euchromatin_forces_params.get<double>("koff") << std::endl;
      }

      std::cout << std::endl;

      std::cout << "NEIGHBOR LIST:" << std::endl;
      const auto &neighbor_list_params = valid_param_list.sublist("neighbor_list");
      std::cout << "  skin_distance: " << neighbor_list_params.get<double>("skin_distance") << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

 private:
  /// \brief Default parameter filename if none is provided.
  std::string input_parameter_filename_ = "ngp_hp1.yaml";
};  // class HP1ParamParser

void run(int argc, char **argv) {
  // Preprocess
  Teuchos::ParameterList params = HP1ParamParser().parse(argc, argv);
  const auto &sim_params = params.sublist("sim");
  const auto &brownian_motion_params = params.sublist("brownian_motion");
  const auto &backbone_springs_params = params.sublist("backbone_springs");
  const auto &backbone_collision_params = params.sublist("backbone_collision");
  const auto &crosslinker_params = params.sublist("crosslinker");
  const auto &periphery_hydro_params = params.sublist("periphery_hydro");
  const auto &periphery_collision_params = params.sublist("periphery_collision");
  const auto &periphery_binding_params = params.sublist("periphery_binding");
  const auto &active_euchromatin_forces_params = params.sublist("active_euchromatin_forces");
  const auto &neighbor_list_params = params.sublist("neighbor_list");

  // Setup the STK mesh
  mundy::mesh::MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3).set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<mundy::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();
  meta_data_ptr->set_coordinate_field_name("COORDS");
  mundy::mesh::MetaData &meta_data = *meta_data_ptr;

  // Parts and their subsets
  auto particle_top = stk::topology::PARTICLE;
  auto beam2_top = stk::topology::BEAM_2;
  auto node_top = stk::topology::NODE;
  auto &spheres_part = meta_data.declare_part("SPHERES", stk::topology::ELEM_RANK);
  auto &e_spheres_part = meta_data.declare_part_with_topology("EUCHROMATIN_SPHERES", particle_top);
  auto &h_spheres_part = meta_data.declare_part_with_topology("HETEROCHROMATIN_SPHERES", particle_top);
  meta_data.declare_part_subset(spheres_part, e_spheres_part);
  meta_data.declare_part_subset(spheres_part, h_spheres_part);
  stk::io::put_assembly_io_part_attribute(spheres_part);
  stk::io::put_io_part_attribute(e_spheres_part);
  stk::io::put_io_part_attribute(h_spheres_part);

  auto &hp1_part = meta_data.declare_part("HP1", stk::topology::ELEM_RANK);
  auto &left_hp1_part = meta_data.declare_part_with_topology("LEFT_HP1", beam2_top);
  auto &doubly_hp1_h_part = meta_data.declare_part_with_topology("DOUBLY_HP1_H", beam2_top);
  auto &doubly_hp1_bs_part = meta_data.declare_part_with_topology("DOUBLY_HP1_BS", beam2_top);
  meta_data.declare_part_subset(hp1_part, left_hp1_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_h_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_bs_part);
  stk::io::put_assembly_io_part_attribute(hp1_part);
  stk::io::put_io_part_attribute(left_hp1_part);
  stk::io::put_io_part_attribute(doubly_hp1_h_part);
  stk::io::put_io_part_attribute(doubly_hp1_bs_part);

  auto &binding_sites_part = meta_data.declare_part_with_topology("BIND_SITES", node_top);
  stk::io::put_io_part_attribute(
      binding_sites_part);  // This is a node part and might not be compatible with IO unless we add special attributes.

  auto &backbone_segs_part = meta_data.declare_part("BACKBONE_SEGMENTS", stk::topology::ELEM_RANK);
  auto &ee_segs_part = meta_data.declare_part_with_topology("EE_SEGMENTS", beam2_top);
  auto &eh_segs_part = meta_data.declare_part_with_topology("EH_SEGMENTS", beam2_top);
  auto &hh_segs_part = meta_data.declare_part_with_topology("HH_SEGMENTS", beam2_top);
  meta_data.declare_part_subset(backbone_segs_part, ee_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, eh_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, hh_segs_part);
  stk::io::put_assembly_io_part_attribute(backbone_segs_part);
  stk::io::put_io_part_attribute(ee_segs_part);
  stk::io::put_io_part_attribute(eh_segs_part);
  stk::io::put_io_part_attribute(hh_segs_part);

  // Fields
  auto node_rank = stk::topology::NODE_RANK;
  auto element_rank = stk::topology::ELEMENT_RANK;
  auto &node_coords_field = meta_data.declare_field<double>(node_rank, "COORDS");
  auto &node_velocity_field = meta_data.declare_field<double>(node_rank, "VELOCITY");
  auto &node_force_field = meta_data.declare_field<double>(node_rank, "FORCE");
  auto &node_collision_velocity_field = meta_data.declare_field<double>(node_rank, "COLLISION_VELOCITY");
  auto &node_collision_force_field = meta_data.declare_field<double>(node_rank, "COLLISION_FORCE");
  auto &node_rng_field = meta_data.declare_field<unsigned>(node_rank, "RNG_COUNTER");
  auto &node_displacement_since_last_rebuild_field = meta_data.declare_field<double>(node_rank, "OUR_DISPLACEMENT");

  auto &elem_hydrodynamic_radius_field = meta_data.declare_field<double>(element_rank, "HYDRODYNAMIC_RADIUS");
  auto &elem_collision_radius_field = meta_data.declare_field<double>(element_rank, "COLLISION_RADIUS");
  auto &elem_binding_radius_field = meta_data.declare_field<double>(element_rank, "BINDING_RADIUS");

  auto &elem_spring_constant_field = meta_data.declare_field<double>(element_rank, "SPRING_CONSTANT");
  auto &elem_spring_r0_field = meta_data.declare_field<double>(element_rank, "SPRING_R0");

  auto &elem_binding_rates_field = meta_data.declare_field<double>(element_rank, "BINDING_RATES");
  auto &elem_unbinding_rates_field = meta_data.declare_field<double>(element_rank, "UNBINDING_RATES");
  auto &elem_rng_field = meta_data.declare_field<unsigned>(element_rank, "RNG_COUNTER");
  auto &elem_chain_id_field = meta_data.declare_field<unsigned>(element_rank, "CHAIN_ID");

  auto &elem_e_state_field = meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE");
  auto &elem_e_state_change_next_time_field =
      meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE_CHANGE_NEXT_TIME");
  auto &elem_e_state_time_field = meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE_CHANGE_TIME");

  auto transient_role = Ioss::Field::TRANSIENT;
  stk::io::set_field_role(node_velocity_field, transient_role);
  stk::io::set_field_role(node_force_field, transient_role);
  stk::io::set_field_role(node_collision_velocity_field, transient_role);
  stk::io::set_field_role(node_collision_force_field, transient_role);
  stk::io::set_field_role(node_rng_field, transient_role);
  stk::io::set_field_role(node_displacement_since_last_rebuild_field, transient_role);
  stk::io::set_field_role(elem_hydrodynamic_radius_field, transient_role);
  stk::io::set_field_role(elem_collision_radius_field, transient_role);
  stk::io::set_field_role(elem_binding_radius_field, transient_role);
  stk::io::set_field_role(elem_spring_constant_field, transient_role);
  stk::io::set_field_role(elem_spring_r0_field, transient_role);
  stk::io::set_field_role(elem_binding_rates_field, transient_role);
  stk::io::set_field_role(elem_unbinding_rates_field, transient_role);
  stk::io::set_field_role(elem_rng_field, transient_role);
  stk::io::set_field_role(elem_chain_id_field, transient_role);
  stk::io::set_field_role(elem_e_state_field, transient_role);
  stk::io::set_field_role(elem_e_state_change_next_time_field, transient_role);
  stk::io::set_field_role(elem_e_state_time_field, transient_role);

  auto scalar_io_type = stk::io::FieldOutputType::SCALAR;
  auto vector_3d_io_type = stk::io::FieldOutputType::VECTOR_3D;
  stk::io::set_field_output_type(node_velocity_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_force_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_collision_velocity_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_collision_force_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_rng_field, scalar_io_type);
  stk::io::set_field_output_type(elem_hydrodynamic_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_collision_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_binding_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_spring_constant_field, scalar_io_type);
  stk::io::set_field_output_type(elem_spring_r0_field, scalar_io_type);
  // stk::io::set_field_output_type(elem_binding_rates_field, stk::io::FieldOutputType::VECTOR_2D);  // These aren't
  // really Vector2Ds. stk::io::set_field_output_type(elem_unbinding_rates_field, stk::io::FieldOutputType::VECTOR_2D);
  stk::io::set_field_output_type(elem_rng_field, scalar_io_type);
  stk::io::set_field_output_type(elem_chain_id_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_change_next_time_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_time_field, scalar_io_type);

  // Sew it all together. Start off fields as uninitialized.
  // Give all nodes and elements a random number generator counter.
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rng_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rng_field, meta_data.universal_part(), 1, nullptr);

  // Heterochromatin and euchromatin spheres are used for hydrodynamics and collision.
  // They move and have forces applied to them. If brownian motion is enabled, they will have a
  // stocastic velocity. Heterochromatin spheres are considered for hp1 binding.
  stk::mesh::put_field_on_mesh(node_velocity_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_collision_velocity_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_collision_force_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_displacement_since_last_rebuild_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_hydrodynamic_radius_field, spheres_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_collision_radius_field, spheres_part, 1, nullptr);

  // Backbone segs apply spring forces to their nodes.
  // The difference between ee, eh, and hh segs is that ee segs can exert an active dipole.
  stk::mesh::put_field_on_mesh(node_force_field, backbone_segs_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_constant_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_r0_field, backbone_segs_part, 1, nullptr);

  // HP1 crosslinkers are used for binding/unbinding and apply forces to their nodes.
  stk::mesh::put_field_on_mesh(node_force_field, hp1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_binding_rates_field, hp1_part, 2, nullptr);
  stk::mesh::put_field_on_mesh(elem_unbinding_rates_field, hp1_part, 2, nullptr);
  stk::mesh::put_field_on_mesh(elem_binding_radius_field, hp1_part, 1, nullptr);

  // Bind sites are use for binding/unbinding. They are merely a point in space and have no
  //   inherent field besides node_coords.

  // That's it for the mesh. Commit it's structure and create the bulk data.
  meta_data.commit();
  std::shared_ptr<mundy::mesh::BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  mundy::mesh::BulkData &bulk_data = *bulk_data_ptr;

  // Perform restart (optional)
  bool restart_performed = false;
  if (!restart_performed) {
    /* Declare the chromatin and HP1
    //  E : euchromatin spheres
    //  H : heterochromatin spheres
    //  | : crosslinkers
    // ---: backbone springs/backbone segments
    //
    //  |   |                           |   |
    //  H---H---E---E---E---E---E---E---H---H
    //
    // The actual connectivity looks like this:
    //  n : node, s : segment and or spring, c : crosslinker
    //
    // c1_      c3_       c5_       c7_
    // | /      | /       | /       | /
    // n1       n3        n5        n7
    //  \      /  \      /  \      /
    //   s1   s2   s3   s4   s5   s6
    //    \  /      \  /      \  /
    //     n2        n4        n6
    //     | \       | \       | \
    //     c2⎻       c4⎻       c6⎻
    //
    // If you look at this long enough, the pattern is clear.
    //  - One less segment than nodes.
    //  - Same number of crosslinkers as heterochromatin nodes.
    //  - Segment i connects to nodes i and i+1.
    //  - Crosslinker i connects to nodes i and i.
    //
    // We need to use this information to populate the node and element info vectors.
    // Mundy will handle passing off this information to the bulk data. Just make sure that all
    // MPI ranks contain the same node and element info. This way, we can determine which nodes
    // should become shared.
    //
    // Rules (non-exhaustive):
    //  - Neither nodes nor elements need to have parts or fields.
    //  - The rank and type of the fields must be consistant. You can't pass an element field to a node,
    //    nor can you set the value of a field to a different type or size than it was declared as.
    //  - The owner of a node must be the same as one of the elements that connects to it.
    //  - A node connected to an element not on the same rank as the node will be shared with the owner of the
    element.
    //  - Field/Part names are case-sensitive but don't attempt to declare "field_1" and "Field_1" as if
    //    that will give two different fields since STKIO will not be able to distinguish between them.
    //  - A (non-zero) negative node id in the element connection list can be used to indicate that a node should be
    left unassigned.
    //  - All parts need to be able to contain an element of the given topology.
    */

    // Fill the declare entities helper
    mundy::mesh::DeclareEntitiesHelper dec_helper;
    size_t node_count = 0;
    size_t element_count = 0;

    // Setup the periphery bind sites
    {
      const std::string bind_sites_type = periphery_binding_params.get<std::string>("bind_sites_type");
      if (bind_sites_type == "RANDOM") {
        const size_t num_bind_sites = periphery_binding_params.get<size_t>("num_bind_sites");
        const std::string periphery_shape = periphery_binding_params.get<std::string>("shape");
        openrand::Philox rng(0, 0);
        if (periphery_shape == "SPHERE") {
          const double radius = periphery_binding_params.get<double>("radius");

          for (size_t i = 0; i < num_bind_sites; i++) {
            // Generate a random point on the unit sphere
            const double u1 = rng.rand<double>();
            const double u2 = rng.rand<double>();
            const double theta = 2.0 * M_PI * u1;
            const double phi = std::acos(2.0 * u2 - 1.0);
            double node_coords[3] = {radius * std::sin(phi) * std::cos(theta),  //
                                     radius * std::sin(phi) * std::sin(theta),  //
                                     radius * std::cos(phi)};

            // Declare the node
            dec_helper.create_node()
                .owning_proc(0)                 //
                .id(node_count + 1)             //
                .add_part(&binding_sites_part)  //
                .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
            node_count++;
          }
        } else if (periphery_shape == "ELLIPSOID") {
          const double a = periphery_binding_params.get<double>("axis_radius1");
          const double b = periphery_binding_params.get<double>("axis_radius2");
          const double c = periphery_binding_params.get<double>("axis_radius3");
          const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
          openrand::Philox rng(0, 0);
          auto keep = [&a, &b, &c, &inv_mu_max, &rng](double x, double y, double z) {
            const double mu_xyz =
                std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
            return inv_mu_max * mu_xyz > rng.rand<double>();
          };

          for (size_t i = 0; i < num_bind_sites; i++) {
            // Rejection sampling to place the periphery binding sites
            double node_coords[3];
            while (true) {
              // Generate a random point on the unit sphere
              const double u1 = rng.rand<double>();
              const double u2 = rng.rand<double>();
              const double theta = 2.0 * M_PI * u1;
              const double phi = std::acos(2.0 * u2 - 1.0);
              node_coords[0] = std::sin(phi) * std::cos(theta);
              node_coords[1] = std::sin(phi) * std::sin(theta);
              node_coords[2] = std::cos(phi);

              // Keep this point with probability proportional to the surface area element
              if (keep(node_coords[0], node_coords[1], node_coords[2])) {
                // Pushforward the point to the ellipsoid
                node_coords[0] *= a;
                node_coords[1] *= b;
                node_coords[2] *= c;
                break;
              }
            }

            // Declare the node
            dec_helper.create_node()
                .owning_proc(0)                 //
                .id(node_count + 1)             //
                .add_part(&binding_sites_part)  //
                .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
            node_count++;
          }
        }
      }
    }

    // Setup the chromatin fibers
    {
      const size_t num_chromosomes = sim_params.get<size_t>("num_chromosomes");
      const size_t num_he_blocks = sim_params.get<size_t>("num_hetero_euchromatin_blocks");
      const size_t num_h_per_block = sim_params.get<size_t>("num_heterochromatin_per_block");
      const size_t num_e_per_block = sim_params.get<size_t>("num_euchromatin_per_block");
      const size_t num_nodes_per_chromosome = num_he_blocks * (num_h_per_block + num_e_per_block);
      const double segment_length = sim_params.get<double>("initial_chromosome_separation");

      std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions;
      if (sim_params.get<std::string>("initialization_type") == "GRID") {
        all_chromosome_positions =
            get_chromosome_positions_grid(num_chromosomes, num_nodes_per_chromosome, segment_length);
      } else if (sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        all_chromosome_positions = get_chromosome_positions_random_unit_cell(
            num_chromosomes, num_nodes_per_chromosome, segment_length, domain_low.getRawPtr(), domain_high.getRawPtr());
      } else if (sim_params.get<std::string>("initialization_type") == "HIILBERT_RANDOM_UNIT_CELL") {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        all_chromosome_positions = get_chromosome_positions_hilbert_random_unit_cell(
            num_chromosomes, num_nodes_per_chromosome, segment_length, domain_low.getRawPtr(), domain_high.getRawPtr());
      } else if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        all_chromosome_positions = get_chromosome_positions_from_file(
            sim_params.get<std::string>("initialize_from_dat_filename"), num_chromosomes);
      } else {
        MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Invalid initialization type.");
      }

      for (size_t f = 0; f < num_chromosomes; f++) {
        // Declare the nodes, segments, and heterochromatin/euchromatin
        for (size_t r = 0; r < num_he_blocks; ++r) {
          // Heterochromatin
          for (size_t h = 0; h < num_h_per_block; ++h) {
            const size_t node_index = r * (num_h_per_block + num_e_per_block) + h;

            dec_helper.create_node()
                .owning_proc(0)                                                                            //
                .id(node_count + 1)                                                                        //
                .add_field_data<unsigned>(&node_rng_field, 0u)                                             //
                .add_field_data<double>(&node_coords_field, {all_chromosome_positions[f][node_index][0],   //
                                                             all_chromosome_positions[f][node_index][1],   //
                                                             all_chromosome_positions[f][node_index][2]})  //
                .add_field_data<double>(&node_displacement_since_last_rebuild_field, {0.0, 0.0, 0.0})      //
                .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})                             //
                .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                                //
                .add_field_data<double>(&node_collision_velocity_field, {0.0, 0.0, 0.0})                   //
                .add_field_data<double>(&node_collision_force_field, {0.0, 0.0, 0.0});
            node_count++;

            // Only create the segment if the node is not the last node in this fiber
            if (node_index < num_nodes_per_chromosome - 1) {
              auto segment = dec_helper.create_element();
              segment
                  .owning_proc(0)                       //
                  .id(element_count + 1)                //
                  .topology(stk::topology::BEAM_2)      //
                  .add_part(&backbone_segs_part)        //
                  .nodes({node_count, node_count + 1})  //
                  .add_field_data<unsigned>(&elem_rng_field, 0u);
              element_count++;

              if (sim_params.get<bool>("enable_backbone_springs")) {
                segment
                    .add_field_data<double>(&elem_spring_constant_field,
                                            backbone_springs_params.get<double>("spring_constant"))  //
                    .add_field_data<double>(&elem_spring_r0_field, backbone_springs_params.get<double>("spring_r0"));
              }

              // Determine if the segment is hh or eh
              const bool left_and_right_node_in_heterochromatin =
                  (h != 0 || r == 0) && (h != num_h_per_block - 1 || r == num_he_blocks - 1);
              if (left_and_right_node_in_heterochromatin) {
                segment.add_part(&hh_segs_part);
              } else {
                segment.add_part(&eh_segs_part);
              }
            }

            // Declare the heterochromatin sphere
            auto h_sphere = dec_helper.create_element();
            h_sphere
                .owning_proc(0)                     //
                .id(element_count + 1)              //
                .topology(stk::topology::PARTICLE)  //
                .add_part(&h_spheres_part)          //
                .nodes({node_count})                //
                .add_field_data<double>(&elem_hydrodynamic_radius_field,
                                        sim_params.get<double>("backbone_sphere_hydrodynamic_radius"));

            if (sim_params.get<bool>("enable_backbone_collision")) {
              h_sphere.add_field_data<double>(&elem_collision_radius_field, backbone_collision_params.get<double>(
                                                                                "backbone_sphere_collision_radius"));
            }
            element_count++;
          }

          for (size_t e = 0; e < num_e_per_block; ++e) {
            const size_t node_index = r * (num_h_per_block + num_e_per_block) + num_h_per_block + e;
            dec_helper.create_node()
                .owning_proc(0)                                                                            //
                .id(node_count + 1)                                                                        //
                .add_field_data<unsigned>(&node_rng_field, 0u)                                             //
                .add_field_data<double>(&node_coords_field, {all_chromosome_positions[f][node_index][0],   //
                                                             all_chromosome_positions[f][node_index][1],   //
                                                             all_chromosome_positions[f][node_index][2]})  //
                .add_field_data<double>(&node_displacement_since_last_rebuild_field, {0.0, 0.0, 0.0})      //
                .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})                             //
                .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                                //
                .add_field_data<double>(&node_collision_velocity_field, {0.0, 0.0, 0.0})                   //
                .add_field_data<double>(&node_collision_force_field, {0.0, 0.0, 0.0});
            node_count++;

            // Only create the segment if the node is not the last node in this fiber
            if (node_index < num_nodes_per_chromosome - 1) {
              auto segment = dec_helper.create_element();
              segment
                  .owning_proc(0)                       //
                  .id(element_count + 1)                //
                  .topology(stk::topology::BEAM_2)      //
                  .add_part(&backbone_segs_part)        //
                  .nodes({node_count, node_count + 1})  //
                  .add_field_data<unsigned>(&elem_rng_field, 0u);
              element_count++;

              if (sim_params.get<bool>("enable_backbone_springs")) {
                segment
                    .add_field_data<double>(&elem_spring_constant_field,
                                            backbone_springs_params.get<double>("spring_constant"))  //
                    .add_field_data<double>(&elem_spring_r0_field, backbone_springs_params.get<double>("spring_r0"));
              }

              // Determine if the segement is ee or eh
              const bool left_and_right_node_in_euchromatin =
                  (e != 0 || r == 0) && (e != num_e_per_block - 1 || r == num_he_blocks - 1);
              if (left_and_right_node_in_euchromatin) {
                segment.add_part(&ee_segs_part);
              } else {
                segment.add_part(&eh_segs_part);
              }
            }

            // Declare the euchromatin sphere
            auto e_sphere = dec_helper.create_element();
            e_sphere
                .owning_proc(0)                     //
                .id(element_count + 1)              //
                .topology(stk::topology::PARTICLE)  //
                .add_part(&e_spheres_part)          //
                .nodes({node_count})                //
                .add_field_data<double>(&elem_hydrodynamic_radius_field,
                                        sim_params.get<double>("backbone_sphere_hydrodynamic_radius"));

            if (sim_params.get<bool>("enable_backbone_collision")) {
              e_sphere.add_field_data<double>(
                  &elem_collision_radius_field,
                  {backbone_collision_params.get<double>("backbone_sphere_collision_radius")});
            }
            element_count++;
          }
        }
      }
    }

    dec_helper.check_consistency(bulk_data);

    // Declare the entities
    bulk_data.modification_begin();
    dec_helper.declare_entities(bulk_data);
    bulk_data.modification_end();

    // Write the mesh to file
    size_t step = 1;  // Step = 0 doesn't write out fields...
    stk::io::write_mesh_with_fields("ngp_hp1.exo", bulk_data, step);
  }

  // Post-setup but pre-run
  if (sim_params.get<bool>("loadbalance_post_initialization")) {
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);
  }

  // Get the NGP stuff
  stk::mesh::NgpMesh ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  auto &ngp_node_coords_field = stk::mesh::get_updated_ngp_field<double>(node_coords_field);
  auto &ngp_node_displacement_since_last_rebuild_field =
      stk::mesh::get_updated_ngp_field<double>(node_displacement_since_last_rebuild_field);
  auto &ngp_node_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_velocity_field);
  auto &ngp_node_force_field = stk::mesh::get_updated_ngp_field<double>(node_force_field);
  auto &ngp_node_collision_velocity_field = stk::mesh::get_updated_ngp_field<double>(node_collision_velocity_field);
  auto &ngp_node_collision_force_field = stk::mesh::get_updated_ngp_field<double>(node_collision_force_field);
  auto &ngp_node_rng_field = stk::mesh::get_updated_ngp_field<unsigned>(node_rng_field);
  auto &ngp_elem_hydrodynamic_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_hydrodynamic_radius_field);
  auto &ngp_elem_binding_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_binding_radius_field);
  auto &ngp_elem_collision_radius_field = stk::mesh::get_updated_ngp_field<double>(elem_collision_radius_field);
  auto &ngp_elem_spring_constant_field = stk::mesh::get_updated_ngp_field<double>(elem_spring_constant_field);
  auto &ngp_elem_spring_r0_field = stk::mesh::get_updated_ngp_field<double>(elem_spring_r0_field);
  auto &ngp_elem_binding_rates_field = stk::mesh::get_updated_ngp_field<double>(elem_binding_rates_field);
  auto &ngp_elem_unbinding_rates_field = stk::mesh::get_updated_ngp_field<double>(elem_unbinding_rates_field);
  auto &ngp_elem_rng_field = stk::mesh::get_updated_ngp_field<unsigned>(elem_rng_field);

  // Time loop
  print_rank0(std::string("Running the simulation for ") + std::to_string(sim_params.get<size_t>("num_time_steps")) +
              " timesteps.");

  // Allocate the neighbor search vectors/views
  ResultViewType search_results;
  LocalResultViewType local_search_results;
  SearchSpheresViewType search_spheres;

  // Collision constraint memory
  size_t num_neighbor_pairs;
  Double1DView signed_sep_dist("signed_sep_dist", 0);
  Double2DView con_normal_ij("con_normal_ij", 0, 3);
  Double1DView lagrange_multipliers("lagrange_multipliers", 0);

  // Unpack the parameters simulation params
  const size_t num_time_steps = sim_params.get<size_t>("num_time_steps");
  const double timestep_size = sim_params.get<double>("timestep_size");
  const double search_buffer = neighbor_list_params.get<double>("skin_distance");
  const double viscosity = sim_params.get<double>("viscosity");
  const size_t io_frequency = sim_params.get<size_t>("io_frequency");

  const bool enable_brownian_motion = sim_params.get<bool>("enable_brownian_motion");
  const bool enable_backbone_collision = sim_params.get<bool>("enable_backbone_collision");
  const bool enable_backbone_springs = sim_params.get<bool>("enable_backbone_springs");
  const bool enable_crosslinkers = sim_params.get<bool>("enable_crosslinkers");
  const bool enable_periphery_collision = sim_params.get<bool>("enable_periphery_collision");
  const bool enable_periphery_hydrodynamics = sim_params.get<bool>("enable_periphery_hydrodynamics");

  ////////////////////////////////
  // Setup the mobility problem //
  ////////////////////////////////
  mundy::mech::SphereLocalDragMobilityOp sphere_mobility_op(ngp_mesh, viscosity, ngp_elem_hydrodynamic_radius_field,
                                                            spheres_part);

  // Allocate the hydro vectors/matrices
  unsigned num_surface_nodes = 0;
  DoubleMatDeviceView inv_self_interaction_matrix(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "inv_self_interaction_matrix"), 0, 0);
  Double1DView surface_positions(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_positions"), 0);
  Double1DView surface_normals(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_normals"), 0);
  Double1DView surface_weights(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_weights"), 0);
  Double1DView surface_radii(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_radii"), 0);
  Double1DView surface_velocities(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_velocities"), 0);
  Double1DView surface_forces(Kokkos::view_alloc(Kokkos::WithoutInitializing, "surface_forces"), 0);
  Double1DView::HostMirror surface_positions_host = Kokkos::create_mirror_view(surface_positions);
  Double1DView::HostMirror surface_normals_host = Kokkos::create_mirror_view(surface_normals);
  Double1DView::HostMirror surface_weights_host = Kokkos::create_mirror_view(surface_weights);
  Double1DView::HostMirror surface_radii_host = Kokkos::create_mirror_view(surface_radii);
  Double1DView::HostMirror surface_velocities_host = Kokkos::create_mirror_view(surface_velocities);
  Double1DView::HostMirror surface_forces_host = Kokkos::create_mirror_view(surface_forces);

  if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
    // Initialize the periphery points, weights, normals, and radii
    std::string quadrature_type = periphery_hydro_params.get<std::string>("quadrature");

    if (quadrature_type == "GAUSS_LEGENDRE") {
      std::string hydro_shape = periphery_hydro_params.get<std::string>("shape");
      const bool shape_is_sphere = (hydro_shape == "SPHERE");
      const bool shape_is_ellipsoid_with_equal_radii =
          (hydro_shape == "ELLIPSOID") &&
          (periphery_hydro_params.get<double>("axis_radius1") == periphery_hydro_params.get<double>("axis_radius2")) &&
          (periphery_hydro_params.get<double>("axis_radius2") == periphery_hydro_params.get<double>("axis_radius3")) &&
          (periphery_hydro_params.get<double>("axis_radius3") == periphery_hydro_params.get<double>("axis_radius1"));
      MUNDY_THROW_REQUIRE(shape_is_sphere || shape_is_ellipsoid_with_equal_radii, std::invalid_argument,
                          "We only support GAUSS_LEGENDRE quadrature for spheres or ellipsoids with equal radii.");

      // Generate the quadrature points and weights for the sphere using GL quadrature
      const size_t spectral_order = periphery_hydro_params.get<size_t>("spectral_order");
      const double radius = shape_is_sphere ? periphery_hydro_params.get<double>("radius")
                                            : periphery_hydro_params.get<double>("axis_radius1");
      std::vector<double> points_vec;
      std::vector<double> weights_vec;
      std::vector<double> normals_vec;
      const bool invert = true;
      const bool include_poles = false;
      mundy::alens::periphery::gen_sphere_quadrature(spectral_order, radius, &points_vec, &weights_vec, &normals_vec,
                                                     include_poles, invert);

      // Allocate the views. Note, resizing does not automatically update the mirror views.
      num_surface_nodes = weights_vec.size();
      Kokkos::resize(surface_positions, 3 * num_surface_nodes);
      Kokkos::resize(surface_normals, 3 * num_surface_nodes);
      Kokkos::resize(surface_weights, num_surface_nodes);
      Kokkos::resize(surface_radii, num_surface_nodes);
      Kokkos::resize(surface_velocities, 3 * num_surface_nodes);
      Kokkos::resize(surface_forces, 3 * num_surface_nodes);
      surface_positions_host = Kokkos::create_mirror_view(surface_positions);
      surface_normals_host = Kokkos::create_mirror_view(surface_normals);
      surface_weights_host = Kokkos::create_mirror_view(surface_weights);
      surface_radii_host = Kokkos::create_mirror_view(surface_radii);
      surface_velocities_host = Kokkos::create_mirror_view(surface_velocities);
      surface_forces_host = Kokkos::create_mirror_view(surface_forces);

      // Copy the raw data into the views
      for (unsigned i = 0; i < num_surface_nodes; i++) {
        surface_positions_host(3 * i + 0) = points_vec[3 * i + 0];
        surface_positions_host(3 * i + 1) = points_vec[3 * i + 1];
        surface_positions_host(3 * i + 2) = points_vec[3 * i + 2];
        surface_normals_host(3 * i + 0) = normals_vec[3 * i + 0];
        surface_normals_host(3 * i + 1) = normals_vec[3 * i + 1];
        surface_normals_host(3 * i + 2) = normals_vec[3 * i + 2];
        surface_velocities_host(3 * i + 0) = 0.0;
        surface_velocities_host(3 * i + 1) = 0.0;
        surface_velocities_host(3 * i + 2) = 0.0;
        surface_forces_host(3 * i + 0) = 0.0;
        surface_forces_host(3 * i + 1) = 0.0;
        surface_forces_host(3 * i + 2) = 0.0;
        surface_weights_host(i) = weights_vec[i];
        surface_radii_host(i) = 0.0;
      }

      // Copy the views to the device
      Kokkos::deep_copy(surface_positions, surface_positions_host);
      Kokkos::deep_copy(surface_normals, surface_normals_host);
      Kokkos::deep_copy(surface_weights, surface_weights_host);
      Kokkos::deep_copy(surface_radii, surface_radii_host);
      Kokkos::deep_copy(surface_velocities, surface_velocities_host);
      Kokkos::deep_copy(surface_forces, surface_forces_host);
    } else if (quadrature_type == "FROM_FILE") {
      const std::string quadrature_points_filename =
          periphery_hydro_params.get<std::string>("quadrature_points_filename");
      const std::string quadrature_normals_filename =
          periphery_hydro_params.get<std::string>("quadrature_normals_filename");
      const std::string quadrature_weights_filename =
          periphery_hydro_params.get<std::string>("quadrature_weights_filename");
      num_surface_nodes = periphery_hydro_params.get<size_t>("num_surface_nodes");
      mundy::alens::periphery::read_vector_from_file(quadrature_weights_filename, num_surface_nodes,
                                                     surface_weights_host);
      mundy::alens::periphery::read_vector_from_file(quadrature_points_filename, 3 * num_surface_nodes,
                                                     surface_positions_host);
      mundy::alens::periphery::read_vector_from_file(quadrature_normals_filename, 3 * num_surface_nodes,
                                                     surface_normals_host);
      Kokkos::deep_copy(surface_positions, surface_positions_host);
      Kokkos::deep_copy(surface_normals, surface_normals_host);
      Kokkos::deep_copy(surface_weights, surface_weights_host);

      // Zero out the radii, forces, and velocities (on host and device)
      Kokkos::deep_copy(surface_radii_host, 0.0);
      Kokkos::deep_copy(surface_velocities_host, 0.0);
      Kokkos::deep_copy(surface_forces_host, 0.0);
      Kokkos::deep_copy(surface_radii, 0.0);
      Kokkos::deep_copy(surface_velocities, 0.0);
      Kokkos::deep_copy(surface_forces, 0.0);
    } else {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Invalid quadrature type: {}", quadrature_type));
    }

    // Run the precomputation for the inverse self-interaction matrix
    const bool write_to_file = false;
    const bool use_values_from_file_if_present = true;
    const std::string inverse_self_interaction_matrix_filename = "inverse_self_interaction_matrix.dat";
    Kokkos::resize(inv_self_interaction_matrix, 3 * num_surface_nodes, 3 * num_surface_nodes);

    bool matrix_read_from_file = false;
    if (use_values_from_file_if_present) {
      auto does_file_exist = [](const std::string &filename) {
        std::ifstream f(filename.c_str());
        return f.good();
      };

      if (does_file_exist(inverse_self_interaction_matrix_filename)) {
        const size_t expected_num_rows_cols = 3 * num_surface_nodes;
        mundy::alens::periphery::read_matrix_from_file(inverse_self_interaction_matrix_filename, expected_num_rows_cols,
                                                       expected_num_rows_cols, inv_self_interaction_matrix);
        matrix_read_from_file = true;
      }
    }

    if (!matrix_read_from_file) {
      const double viscosity = sim_params.get<double>("viscosity");
      DoubleMatDeviceView self_interaction_matrix("self_interaction_matrix", 3 * num_surface_nodes,
                                                  3 * num_surface_nodes);
      mundy::alens::periphery::fill_skfie_matrix(stk::ngp::ExecSpace(), viscosity, num_surface_nodes, num_surface_nodes,
                                                 surface_positions, surface_positions, surface_normals, surface_weights,
                                                 self_interaction_matrix);
      mundy::alens::periphery::invert_matrix(stk::ngp::ExecSpace(), self_interaction_matrix,
                                             inv_self_interaction_matrix);

      if (write_to_file) {
        mundy::alens::periphery::write_matrix_to_file(inverse_self_interaction_matrix_filename,
                                                      inv_self_interaction_matrix);
      }
    }
  }

  bool rebuild_neighbors = true;
  Kokkos::Timer tps_timer;
  for (size_t timestep_idx = 0; timestep_idx < num_time_steps; timestep_idx++) {
    if (timestep_idx % io_frequency == 0) {
      std::cout << "Time step: " << timestep_idx << " running at "
                << static_cast<double>(io_frequency) / tps_timer.seconds() << " tps "
                << " | " << tps_timer.seconds() / static_cast<double>(io_frequency) << " spt" << std::endl;
      tps_timer.reset();
      // Comm fields to host
      ngp_node_coords_field.sync_to_host();
      ngp_node_velocity_field.sync_to_host();
      ngp_node_force_field.sync_to_host();
      ngp_node_collision_velocity_field.sync_to_host();
      ngp_node_collision_force_field.sync_to_host();
      ngp_node_rng_field.sync_to_host();
      ngp_elem_hydrodynamic_radius_field.sync_to_host();
      ngp_elem_binding_radius_field.sync_to_host();
      ngp_elem_collision_radius_field.sync_to_host();
      ngp_elem_spring_constant_field.sync_to_host();
      ngp_elem_spring_r0_field.sync_to_host();
      ngp_elem_binding_rates_field.sync_to_host();
      ngp_elem_unbinding_rates_field.sync_to_host();
      ngp_elem_rng_field.sync_to_host();

      // Write to file using Paraview compatable naming
      stk::io::write_mesh_with_fields("ngp_hp1.e-s." + std::to_string(timestep_idx), bulk_data, timestep_idx + 1,
                                      timestep_idx * timestep_size, stk::io::WRITE_RESULTS);
    }

    // Prepare the current configuration.
    mundy::mesh::field_fill(0.0, node_velocity_field, stk::ngp::ExecSpace());
    mundy::mesh::field_fill(0.0, node_force_field, stk::ngp::ExecSpace());
    mundy::mesh::field_fill(0.0, node_collision_velocity_field, stk::ngp::ExecSpace());
    mundy::mesh::field_fill(0.0, node_collision_force_field, stk::ngp::ExecSpace());
    mundy::mesh::field_fill(0.0, elem_binding_rates_field, stk::ngp::ExecSpace());
    mundy::mesh::field_fill(0.0, elem_unbinding_rates_field, stk::ngp::ExecSpace());

    //////////////////////
    // Detect neighbors //
    //////////////////////
    const double max_component_wise_displacement = mundy::mesh::field_max<double>(
        node_displacement_since_last_rebuild_field, backbone_segs_part, stk::ngp::ExecSpace());
    if (max_component_wise_displacement > search_buffer) {
      rebuild_neighbors = true;
    }

    if (rebuild_neighbors) {
      std::cout << "Rebuilding neighbors..." << std::endl;
      Kokkos::Timer search_timer;
      search_spheres = mundy::mech::create_search_spheres(bulk_data, ngp_mesh, search_buffer, ngp_node_coords_field,
                                                          ngp_elem_collision_radius_field, spheres_part);

      // Perform the backbone sphere to backbone sphere search
      const stk::search::SearchMethod search_method = stk::search::MORTON_LBVH;
      const bool results_parallel_symmetry = true;   // Create source -> target and target -> source pairs
      const bool auto_swap_domain_and_range = true;  // Swap source and target if target is owned and source is not
                                                     // This must be true to avoid double counting forces.
      stk::search::coarse_search(search_spheres, search_spheres, search_method, bulk_data.parallel(), search_results,
                                 stk::ngp::ExecSpace{}, results_parallel_symmetry, auto_swap_domain_and_range);
      num_neighbor_pairs = search_results.extent(0);

      // Ghost the non-owned spheres
      mundy::mech::ghost_neighbors(bulk_data, search_results);

      // Create local neighbor indices
      local_search_results =
          mundy::mech::get_local_neighbor_indices(bulk_data, stk::topology::ELEM_RANK, search_results);

      // TODO(palmerb4): Store the local neighbors within a field to allow the fetching of all neighbors of a given
      //  element. We need to validate if it is better for us to use the same neighbor list for both the sphere-sphere
      //  collision and the binding of HP1 to heterochromatin or if we should have separate lists due to the difference
      //  in their search radii.

      if (enable_backbone_collision) {
        // Resize the collision views if the neighbor pairs are regenerated.
        signed_sep_dist = Double1DView("signed_sep_dist", num_neighbor_pairs);
        con_normal_ij = Double2DView("con_normal_ij", num_neighbor_pairs, 3);
        lagrange_multipliers = Double1DView("lagrange_multipliers", num_neighbor_pairs);
      }

      // Reset the accumulated displacements and the rebuild flag
      mundy::mesh::field_fill(0.0, node_displacement_since_last_rebuild_field, stk::ngp::ExecSpace());
      std::cout << "Search time: " << search_timer.seconds() << " with " << num_neighbor_pairs << " results."
                << std::endl;
      rebuild_neighbors = false;
    }

    // /////////
    // // KMC //
    // /////////
    // if (enable_crosslinkers) {
    //   bulk_data.modification_begin();

    //   HookeanCrosslinkerBindRateHeterochromatin left_to_h_realized_rate_getter(
    //       bulk_data, binding_kt, left_to_h_binding_rate, crosslinker_spring_constant, crosslinker_spring_r0,
    //       node_coords_field, h_spheres);

    //   kmc_perform_state_change_left_bound(bulk_data, timestep_size, left_to_h_realized_rate_getter,
    //                                       neighboring_bind_sites_field, elem_rng_field, left_hp1_part,
    //                                       doubly_hp1_h_part);
    //   kmc_perform_state_change_doubly_bound(bulk_data, timestep_size, left_to_h_realized_rate_getter, elem_rng_field,
    //                                         left_hp1_part, doubly_hp1_h_part);

    //   bulk_data.modification_end();
    // }

    /////////////////////////////
    // Evaluate forces f(x(t)) //
    /////////////////////////////
    if (enable_backbone_springs) {
      const std::string spring_type = backbone_springs_params.get<std::string>("spring_type");
      if (spring_type == "HOOKEAN") {
        mundy::mech::compute_hookean_spring_forces(ngp_mesh, ngp_node_coords_field, ngp_node_force_field,
                                                   ngp_elem_spring_constant_field, ngp_elem_spring_r0_field,
                                                   backbone_segs_part);
      } else if (spring_type == "FENE") {
        mundy::mech::compute_fene_spring_forces(ngp_mesh, ngp_node_coords_field, ngp_node_force_field,
                                                ngp_elem_spring_constant_field, ngp_elem_spring_r0_field,
                                                backbone_segs_part);
      } else {
        MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid spring type.");
      }
    }

    //   if (enable_crosslinkers) {
    //     compute_hookean_spring_forces();
    //     compute_fene_spring_forces();
    //   }

    //   if (enable_periphery_collision) {
    //     Kokkos::Profiling::pushRegion("HP1::compute_periphery_collision_forces");
    //     if (periphery_collision_shape_ == PERIPHERY_SHAPE::SPHERE) {
    //       compute_spherical_periphery_collision_forces();
    //     } else if (periphery_collision_shape_ == PERIPHERY_SHAPE::ELLIPSOID) {
    //       compute_ellipsoidal_periphery_collision_forces();
    //     } else {
    //       MUNDY_THROW_REQUIRE(false, std::logic_error, "Invalid periphery type.");
    //     }
    //     Kokkos::Profiling::popRegion();
    //   }

    ////////////////////////
    // Compute velocities //
    ////////////////////////
    if (enable_brownian_motion) {
      const double brownian_kt = brownian_motion_params.get<double>("kt");
      compute_brownian_velocity(ngp_mesh, timestep_size, brownian_kt, viscosity, ngp_node_velocity_field,
                                ngp_elem_rng_field, ngp_elem_hydrodynamic_radius_field, spheres_part);
    }

    if (enable_backbone_collision) {
      //////////////////////////////////////////
      // Initialize and solve the constraints //
      //////////////////////////////////////////
      Kokkos::Timer collision_timer;
      mundy::mech::compute_signed_separation_distance_and_contact_normal(
          ngp_mesh, local_search_results, ngp_node_coords_field, ngp_elem_collision_radius_field, signed_sep_dist,
          con_normal_ij);

      sphere_mobility_op.update();  // Update the mobility to the current configuration
      const double max_allowable_overlap = backbone_collision_params.get<double>("max_allowable_overlap");
      const size_t max_collision_iterations = backbone_collision_params.get<size_t>("max_collision_iterations");
      Kokkos::deep_copy(lagrange_multipliers, 0.0);                                                    // initial guess
      mundy::mech::CollisionResult result = mundy::mech::resolve_collisions(ngp_mesh,                  //
                                                                            viscosity,                 //
                                                                            timestep_size,             //
                                                                            max_allowable_overlap,     //
                                                                            max_collision_iterations,  //
                                                                            sphere_mobility_op,        //
                                                                            local_search_results,      //
                                                                            ngp_node_force_field,      //
                                                                            ngp_node_velocity_field,   //
                                                                            ngp_node_collision_force_field,     //
                                                                            ngp_node_collision_velocity_field,  //
                                                                            signed_sep_dist,                    //
                                                                            con_normal_ij,                      //
                                                                            lagrange_multipliers,               //
                                                                            spheres_part);

      // Sum the collision force/velocity into the total force/velocity
      mundy::mesh::field_axpby(1.0, node_collision_force_field, 1.0, node_force_field, spheres_part,
                               stk::ngp::ExecSpace());
      mundy::mesh::field_axpby(1.0, node_collision_velocity_field, 1.0, node_velocity_field, spheres_part,
                               stk::ngp::ExecSpace());

      std::cout << "Result: " << std::endl;
      std::cout << "  Max abs projected sep: " << result.max_abs_projected_sep << std::endl;
      std::cout << "  Number of iterations: " << result.ite_count << std::endl;
      std::cout << "  Max displacement: " << result.max_displacement << std::endl;
      std::cout << "  Time: " << collision_timer.seconds() << std::endl;
    }

    // Take an Euler step
    mundy::mesh::field_axpby(timestep_size, node_velocity_field, 1.0, node_displacement_since_last_rebuild_field,
                             stk::ngp::ExecSpace());
    mundy::mesh::field_axpby(timestep_size, node_velocity_field, 1.0, node_coords_field, stk::ngp::ExecSpace());

    // Check for overlap
    // if (enable_backbone_collision) {
    //   node_coords_field.sync_to_host();
    //   elem_collision_radius_field.sync_to_host();
    //   const double max_allowable_overlap = backbone_collision_params.get<double>("max_allowable_overlap");
    //   mundy::mech::check_overlap(bulk_data, max_allowable_overlap, node_coords_field, elem_collision_radius_field,
    //                              spheres_part);
    // }
  }
}
//@}

}  // namespace chromalens

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  // Run the simulation!
  mundy::chromalens::run(argc, argv);

  // Before exiting, sleep for some amount of time to force Kokkos to print better at the end.
  std::this_thread::sleep_for(std::chrono::milliseconds(stk::parallel_machine_rank(MPI_COMM_WORLD)));

  Kokkos::finalize();
  stk::parallel_machine_finalize();
  return 0;
}
