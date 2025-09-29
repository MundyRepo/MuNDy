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

// #if TRILINOS_MAJOR_MINOR_VERSION >= 160000

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/ForEachEntity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_core/throw_assert.hpp>             // for MUNDY_THROW_ASSERT
#include <mundy_mesh/Aggregate.hpp>                // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>                 // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>          // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/GenerateNeighborLinkers.hpp>  // for mundy::mesh::GenerateNeighborLinkers
#include <mundy_mesh/LinkData.hpp>                 // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>              // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>                 // for mundy::mesh::MetaData

//! \name Physics
//@{

template <typename Center, typename Radius, typename YoungsModulus, typename PoissonRatio, typename LinkedEntities,
          typename SphereAgg>
class sphere_sphere_hertzian_contact {
 public:
  sphere_sphere_hertzian_contact(const SphereAgg &sphere_agg) : ngp_sphere_agg_(get_updated_ngp_data(sphere_agg)) {
  }

  KOKKOS_INLINE_FUNCTION
  double comp_effective_radius(const double r1, const double r2) const {
    return (r1 * r2) / (r1 + r2);
  }

  KOKKOS_INLINE_FUNCTION
  double comp_effective_youngs_modulus(const double ym1, const double ym2, const double pr1, const double pr2) const {
    return (ym1 * ym2) / (ym1 - ym1 * pr1 * pr1 + ym2 - ym2 * pr2 * pr2);
  }

  KOKKOS_INLINE_FUNCTION
  double comp_normal_force_magnitude(const double effective_radius, const double effective_ym, const double signed_sep,
                                     const double center_distance) const {
    return 4.0 / 3.0 * effective_ym * Kokkos::sqrt(effective_radius) * Kokkos::pow(-signed_sep, 1.5) / center_distance;
  }

  KOKKOS_INLINE_FUNCTION
  double comp_normal_force_magnitude(const double r1, const double r2, const double ym1, const double ym2,
                                     const double pr1, const double pr2, const double signed_sep,
                                     const double center_distance) const {
    const auto effective_radius = comp_effective_radius(r1, r2);
    const auto effective_ym = comp_effective_youngs_modulus(ym1, ym2, pr1, pr2);
    return comp_normal_force_magnitude(effective_radius, effective_ym, signed_sep, center_distance);
  }

  /// \brief Apply this functor to a single sphere-sphere linker
  KOKKOS_INLINE_FUNCTION
  void operator()(const auto &link_view) {
    stk::mesh::Entity src = get<LinkedEntities>(link_view)[0];
    stk::mesh::Entity trg = get<LinkedEntities>(link_view)[1];
    auto src_view = ngp_sphere_agg_.get_view(link_view.ngp_mesh().fast_mesh_index(src));
    auto trg_view = ngp_sphere_agg_.get_view(link_view.ngp_mesh().fast_mesh_index(trg));

    const auto src_center = get<Center>(src_view);
    const auto trg_center = get<Center>(trg_view);
    const double src_radius = get<Radius>(src_view)[0];
    const double trg_radius = get<Radius>(trg_view)[0];
    const double src_ym = get<YoungsModulus>(src_view)[0];
    const double trg_ym = get<YoungsModulus>(trg_view)[0];
    const double src_pr = get<PoissonRatio>(src_view)[0];
    const double trg_pr = get<PoissonRatio>(trg_view)[0];

    const auto src_to_trg = trg_center - src_center;
    const auto center_distance = src_to_trg.norm();
    const auto signed_sep = center_distance - src_radius - trg_radius;

    if (signed_sep < 0.0) {
      const double force_mag_scaled = comp_normal_force_magnitude(src_radius, trg_radius, src_ym, trg_ym, src_pr,
                                                                  trg_pr, signed_sep, center_distance);

      // Atomic add/sub the force one component at a time
      auto src_force = get<Force>(src_view);
      auto trg_force = get<Force>(trg_view);
      Kokkos::atomic_add(&src_force[0], -force_mag_scaled * src_to_trg[0]);
      Kokkos::atomic_add(&src_force[1], -force_mag_scaled * src_to_trg[1]);
      Kokkos::atomic_add(&src_force[2], -force_mag_scaled * src_to_trg[2]);
      Kokkos::atomic_add(&trg_force[0], force_mag_scaled * src_to_trg[0]);
      Kokkos::atomic_add(&trg_force[1], force_mag_scaled * src_to_trg[1]);
      Kokkos::atomic_add(&trg_force[2], force_mag_scaled * src_to_trg[2]);
    }
  }

  /// \brief Apply this functor to some sphere-sphere linkers in an linker agg
  void apply_to(auto &sphere_sphere_link_agg, const stk::mesh::Selector &subset_selector) {
    ngp_sphere_agg_.sync_to_device<Center, Radius, YoungsModulus, PoissonRatio, Force>();

    auto ngp_sphere_sphere_link_agg = get_updated_ngp_data(sphere_sphere_link_agg);
    ngp_sphere_sphere_link_agg.sync_to_device<LinkedEntities>();
    ngp_sphere_sphere_link_agg.template for_each((*this));

    ngp_sphere_agg_.modified_on_device<Force>();
  }

  /// \brief Apply this functor to all sphere-sphere linkers in an linker agg
  void apply_to(auto &sphere_sphere_link_agg) {
    ngp_sphere_agg_.sync_to_device<Center, Radius, YoungsModulus, PoissonRatio, Force>();

    auto ngp_sphere_sphere_link_agg = get_updated_ngp_data(sphere_sphere_link_agg);
    ngp_sphere_sphere_link_agg.sync_to_device<LinkedEntities>();
    ngp_sphere_sphere_link_agg.template for_each((*this), subset_selector);

    ngp_sphere_agg_.modified_on_device<Force>();
  }

 private:
  using NgpSphereAgg = decltype(get_updated_ngp_data(std::declval<SphereAgg>()));
  const NgpSphereAgg &ngp_sphere_agg_;
};

template <typename Position, typename Force>
class attractive_abc_flow {
 public:
  attractive_abc_flow(const double a, const double b, const double c, const double attraction_coeff)
      : a_(a), b_(b), c_(c), attraction_coeff_(attraction_coeff) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto &node_view) {
    const auto coords = get<Position>(node_view);
    auto force = get<Force>(node_view);

    const double r = math::norm(coords);
    const double inv_r = 1.0 / r;

    force[0] += a_ * Kokkos::sin(coords[2]) + c_ * Kokkos::cos(coords[1]) - attraction_coeff_ * coords[0] * inv_r;
    force[1] += b_ * Kokkos::sin(coords[0]) + a_ * Kokkos::cos(coords[2]) - attraction_coeff_ * coords[1] * inv_r;
    force[2] += c_ * Kokkos::sin(coords[1]) + b_ * Kokkos::cos(coords[0]) - attraction_coeff_ * coords[2] * inv_r;
  }

  void apply_to(auto &node_agg, const stk::mesh::Selector &subset_selector) {
    static_assert(node_agg.rank == stk::topology::NODE_RANK, "This functor only works on nodes");

    node_agg.sync_to_device<Position, Force>();
    node_agg.template for_each((*this), subset_selector);
    node_agg.modified_on_device<Force>();
  }

 private:
  const double a_;
  const double b_;
  const double c_;
  const double attraction_coeff_;
};

template <typename Position, typename Velocity>
class node_euler_step {
 public:
  node_euler_step(const double time_step_size) : time_step_size_(time_step_size) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto &node_view) {
    auto coords = get<Position>(node_view);
    auto vel = get<Velocity>(node_view);
    coords += time_step_size_ * vel;
  }

  void apply_to(auto &node_agg, const stk::mesh::Selector &subset_selector) {
    static_assert(node_agg.rank == stk::topology::NODE_RANK, "This functor only works on nodes");

    node_agg.sync_to_device<Position, Velocity>();
    node_agg.template for_each((*this), subset_selector);
    node_agg.modified_on_device<Position>();
  }

 private:
  const double time_step_size_;
};
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
    return std::string("NODE_COORDS");
  }
  virtual bool shouldPrintMetrics() const {
    return false;
  }
};  // RcbSettings
//@}

struct OUR_DISP {};

int main(int argc, char **argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::Selector;
  using stk::topology::ELEM_RANK;
  using stk::topology::NODE_RANK;

  {
    // Simulation of N spheres in a cube
    const double youngs_modulus = 100.0;
    const double poisson_ratio = 0.3;
    const double sphere_radius_min = 1.0;
    const double sphere_radius_max = 1.0;
    const double num_spheres = 100000;
    const double viscosity = 1.0 / (6.0 * Kokkos::numbers::pi * sphere_radius_max);

    const double force_a = 1.0;
    const double force_b = Kokkos::sqrt(2.0);
    const double force_c = Kokkos::sqrt(3.0);
    const double force_attraction_coeff = 2.0;

    const math::Vector3d unit_cell_bottom_left = {-50.0, -50.0, -50.0};
    const math::Vector3d unit_cell_top_right = {50.0, 50.0, 50.0};
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
    std::cout << "  Max volume fraction: " << volume_fraction << std::endl;
    std::cout << "  IO frequency: " << io_frequency << std::endl;

    // Setup the STK mesh
    MeshBuilder mesh_builder(MPI_COMM_WORLD);
    mesh_builder
        .set_spatial_dimension(3)  //
        .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
    std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = mesh_builder.create_meta_data();
    meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition
                                         // to all fields are simple.
    meta_data_ptr->set_coordinate_field_name("NODE_COORDS");
    std::shared_ptr<BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
    MetaData &meta_data = *meta_data_ptr;
    BulkData &bulk_data = *bulk_data_ptr;

    // Create the spheres
    auto &spheres_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
    stk::io::put_io_part_attribute(spheres_part);

    auto &node_coords = meta_data.declare_field<double>(NODE_RANK, "NODE_COORDS");
    auto &node_disp_since_last_rebuild_field = meta_data.declare_field<double>(NODE_RANK, "OUR_DISP");
    auto &node_force = meta_data.declare_field<double>(NODE_RANK, "FORCE");
    auto &node_velocity = meta_data.declare_field<double>(NODE_RANK, "VELOCITY");
    auto &elem_radius = meta_data.declare_field<double>(ELEM_RANK, "RADIUS");
    auto &elem_rng_counter = meta_data.declare_field<size_t>(ELEM_RANK, "RNG_COUNTER");

    stk::mesh::put_field_on_mesh(node_coords, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(node_disp_since_last_rebuild_field, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(node_force, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(node_velocity, spheres_part, 3, nullptr);
    stk::mesh::put_field_on_mesh(elem_radius, spheres_part, 1, nullptr);
    stk::mesh::put_field_on_mesh(elem_rng_counter, spheres_part, 1, nullptr);

    // Create the sphere-sphere links
    LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", NODE_RANK);
    LinkData link_data = declare_link_data(bulk_data, link_meta_data);
    Part &neighbor_link_part = link_meta_data.declare_link_part("NEIGHBOR_LINKS", 2 /* our dimensionality */);
    stk::io::put_io_part_attribute(neighbor_link_part);

    // Aggregate the mesh into logical building blocks
    auto center_accessor = Vector3FieldComponent(node_coords);
    auto disp_since_last_rebuild_accessor = Vector3FieldComponent(node_disp_since_last_rebuild_field);
    auto force_accessor = Vector3FieldComponent(node_force);
    auto velocity_accessor = Vector3FieldComponent(node_velocity);
    auto radius_accessor = ScalarFieldComponent(elem_radius);
    auto rng_counter_accessor = ScalarFieldComponent(elem_rng_counter);
    auto linked_entities_accessor = FieldComponent(link_meta_data.get_linked_entities_field());

    auto sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, spheres_part)
                           .add_component<CENTER, NODE_RANK>(center_accessor)
                           .add_component<OUR_DISP, NODE_RANK>(disp_since_last_rebuild_accessor)
                           .add_component<FORCE, NODE_RANK>(force_accessor)
                           .add_component<VELOCITY, NODE_RANK>(velocity_accessor)
                           .add_component<RADIUS, ELEM_RANK>(radius_accessor)
                           .add_component<RNG_COUNTER, ELEM_RANK>(rng_counter_accessor);

    auto node_data = make_ranked_aggregate<NODE_RANK>(bulk_data, spheres_part)
                         .add_component<CENTER, NODE_RANK>(center_accessor)
                         .add_component<OUR_DISP, NODE_RANK>(disp_since_last_rebuild_accessor)
                         .add_component<FORCE, NODE_RANK>(force_accessor)
                         .add_component<VELOCITY, NODE_RANK>(velocity_accessor);

    auto sphere_sphere_link_data = make_ranked_aggregate<NODE_RANK>(bulk_data, neighbor_link_part)
                                       .add_component<LINKED_ENTITIES, NODE_RANK>(linked_entities_accessor);

    // Concretize the mesh
    meta_data.commit();

    // Use the DeclareEntitiesHelper to declare the spheres
    // Each processor will declare a subset of the spheres
    DeclareEntitiesHelper dec_helper;
    size_t num_spheres_per_proc = num_spheres / bulk_data.parallel_size();
    size_t remainder = num_spheres % bulk_data.parallel_size();
    size_t num_spheres_this_proc = num_spheres_per_proc + (bulk_data.parallel_rank() < remainder ? 1 : 0);
    size_t starting_node_id =
        num_spheres_per_proc * bulk_data.parallel_rank() + std::min(bulk_data.parallel_rank(), remainder) + 1;
    size_t starting_elem_id = starting_node_id;

    for (size_t i = starting_node_id; i < starting_node_id + num_spheres_this_proc; ++i) {
      openrand::Philox rng(i, 0);
      double rand_radius = rng.rand<double>() * (sphere_radius_max - sphere_radius_min) + sphere_radius_min;
      double rand_x =
          rng.rand<double>() * (unit_cell_top_right[0] - unit_cell_bottom_left[0]) + unit_cell_bottom_left[0];
      double rand_y =
          rng.rand<double>() * (unit_cell_top_right[1] - unit_cell_bottom_left[1]) + unit_cell_bottom_left[1];
      double rand_z =
          rng.rand<double>() * (unit_cell_top_right[2] - unit_cell_bottom_left[2]) + unit_cell_bottom_left[2];

      dec_helper.create_node()
          .owning_proc(0)  //
          .id(i)           //
          .add_field_data<double>(&node_coords, {rand_x, rand_y, rand_z})
          .add_field_data<double>(&node_disp_since_last_rebuild_field, {0.0, 0.0, 0.0})
          .add_field_data<double>(&node_force, {0.0, 0.0, 0.0})
          .add_field_data<double>(&node_velocity, {0.0, 0.0, 0.0});

      dec_helper.create_element()
          .owning_proc(0)                     //
          .id(i)                              //
          .topology(stk::topology::PARTICLE)  //
          .nodes({i})                         //
          .add_part(&spheres_part)
          .add_field_data<double>(&elem_radius, {rand_radius})
          .add_field_data<size_t>(&elem_rng_counter, {0});
    }

    bulk_data.modification_begin();
    dec_helper.declare_entities(bulk_data);
    bulk_data.modification_end();

    // Balance the mesh
    stk::balance::balanceStkMesh(RcbSettings{}, bulk_data);

    // Setup our neighbor search manager
    auto gen_neighbors = create_gen_neighbor_links(link_data, stk::ngp::ExecSpace{});
    auto search_sphere_gen = [&sphere_data](autp &sphere_view) {
      return geom::Sphere<double>(get<CENTER>(sphere_view), get<RADIUS>(sphere_view)[0]);
    };
    gen_neighbors
        .set_source_target_rank(ELEM_RANK, ELEM_RANK)                                                 //
        .acts_on(sphere_data, sphere_data, search_sphere_gen, search_sphere_gen, neighbor_link_part)  //
        .get_enforce_source_target_symmetry(true)                                                     //
        .set_search_method(stk::search::MORTON_LBVH)                                                  //
        .set_search_buffer(search_buffer)                                                             //
        .concretize();

    // Timeloop
    Kokkos::Timer overall_timer;
    Kokkos::Timer tps_timer;
    for (size_t time_step_index = 0; time_step_index < num_time_steps; ++time_step_index) {
      if (time_step_index % io_frequency == 0) {
        std::cout << "Time step: " << time_step_index << " | Total time: " << time_step_index * time_step_size
                  << std::endl;
        std::cout << "  Time elapsed: " << overall_timer.seconds() << " s" << std::endl;
        std::cout << "  Running avg tps: " << static_cast<double>(io_frequency) / tps_timer.seconds() << std::endl;
        tps_timer.reset();

        // Comm io fields to host
        node_coords.sync_to_host();
        node_force.sync_to_host();
        node_velocity.sync_to_host();
        elem_radius.sync_to_host();

        // Write to file using Paraview compatable naming
        stk::io::write_mesh_with_fields("hertz_spheres.e-s." + std::to_string(time_step_index), bulk_data,
                                        time_step_index + 1, time_step_index * time_step_size, stk::io::WRITE_RESULTS);
      }

      // Reset the forces and velocities
      field_fill<double>(0.0, node_force, stk::ngp::ExecSpace{});
      field_fill<double>(0.0, node_velocity, stk::ngp::ExecSpace{});

      Kokkos::Timer neighbor_timer;
      bool rebuild_performed = gen_neighbors.generate();
      if (rebuild_performed) {
        std::cout << "Neighbor rebuild time: " << neighbor_timer.seconds() << std::endl;
      } else {
        std::cout << "Neighbor no rebuild: " << neighbor_timer.seconds() << std::endl;
      }

      Kokkos::Timer contact_timer;
      sphere_sphere_hertzian_contact(sphere_data).apply_to(sphere_sphere_link_data);
      std::cout << "Contact time: " << contact_timer.seconds() << std::endl;

      Kokkos::Timer flow_timer;
      attractive_abc_flow(force_a, force_b, force_c, force_attraction_coeff).apply_to(node_data);
      std::cout << "Flow time: " << flow_timer.seconds() << std::endl;

      Kokkos::Timer update_timer;
      node_euler_step(time_step_size).apply_to(node_data);
      std::cout << "Update time: " << update_timer.seconds() << std::endl;
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
