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

// C++ core
#include <memory>  // for std::unique_ptr
#include <vector>  // for std::vector

// PLY reader
#include "happly.h"

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, etc

// STK
#include <stk_io/IossBridge.hpp>       // for stk::io::set_field_role and stk::io::put_io_part_attribute
#include <stk_io/WriteMesh.hpp>        // for stk::io::write_mesh
#include <stk_mesh/base/BulkData.hpp>  // for stk::mesh::BulkData
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/MetaData.hpp>  // for stk::mesh::MetaData
#include <stk_topology/topology.hpp>

// Mundy
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_REQUIRE
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper

void neuron() {
  // A neuron read in from a ply file
  std::cout << "Reading in a neuron from a ply file" << std::endl;

  // Setup a mesh
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  builder.set_aura_option(stk::mesh::BulkData::AUTO_AURA);

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  auto &meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("coordinates");

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  auto &bulk_data = *bulk_data_ptr;

  // Declare a triangle part with a triangle topology
  stk::mesh::Part &triangle_part = meta_data.declare_part_with_topology("triangle_part", stk::topology::SHELL_TRI_3);
  stk::io::put_io_part_attribute(triangle_part);

  // Add a node and element-rank color field
  stk::mesh::Field<int> &node_rgb_field = meta_data.declare_field<int>(stk::topology::NODE_RANK, "rgb");
  stk::mesh::Field<int> &element_rgb_field = meta_data.declare_field<int>(stk::topology::ELEMENT_RANK, "rgb");
  stk::io::set_field_role(node_rgb_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(element_rgb_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_output_type(node_rgb_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(element_rgb_field, stk::io::FieldOutputType::VECTOR_3D);

  // Add the node coordinates field
  stk::mesh::Field<double> &node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");

  // Put the fields on the mesh
  stk::mesh::put_field_on_mesh(node_rgb_field, triangle_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(element_rgb_field, triangle_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Commit the meta data
  meta_data.commit();

  // Fill the declare entities helper from a ply file
  mundy::mesh::DeclareEntitiesHelper dec_helper;
  std::string filename = "2013_03_06_cell08_876_H41_05_Cell2_cell-axon.ply";
  happly::PLYData ply_in(filename);

  // Setup the nodes
  std::vector<double> x = ply_in.getElement("vertex").getProperty<double>("x");
  std::vector<double> y = ply_in.getElement("vertex").getProperty<double>("y");
  std::vector<double> z = ply_in.getElement("vertex").getProperty<double>("z");

  std::vector<unsigned char> vert_red = ply_in.getElement("vertex").getProperty<unsigned char>("red");
  std::vector<unsigned char> vert_green = ply_in.getElement("vertex").getProperty<unsigned char>("green");
  std::vector<unsigned char> vert_blue = ply_in.getElement("vertex").getProperty<unsigned char>("blue");

  size_t num_nodes = x.size();
  for (size_t i = 0; i < num_nodes; ++i) {
    dec_helper.create_node()
        .owning_proc(0)                                                  //
        .id(i + 1)                                                       //
        .add_field_data<double>(&node_coords_field, {x[i], y[i], z[i]})  //
        .add_field_data<int>(&node_rgb_field, {vert_red[i], vert_green[i], vert_blue[i]});
  }

  // Setup the elements
  // Note, happily's node indices are 0-based, but STK's are 1-based
  std::vector<std::vector<size_t>> face_ind = ply_in.getFaceIndices<size_t>();
  std::vector<unsigned char> face_red = ply_in.getElement("face").getProperty<unsigned char>("red");
  std::vector<unsigned char> face_green = ply_in.getElement("face").getProperty<unsigned char>("green");
  std::vector<unsigned char> face_blue = ply_in.getElement("face").getProperty<unsigned char>("blue");

  size_t num_faces = face_ind.size();
  for (size_t i = 0; i < num_faces; ++i) {
    MUNDY_THROW_REQUIRE(face_ind[i].size() == 3, std::runtime_error, "Triangle face must have 3 vertices");
    dec_helper.create_element()
        .owning_proc(0)                        //
        .id(i + 1)                             //
        .topology(stk::topology::SHELL_TRI_3)  //
        .add_part(&triangle_part)              //
        .nodes({face_ind[i][0] + 1, face_ind[i][1] + 1, face_ind[i][2] + 1})
        .add_field_data<int>(&element_rgb_field, {face_red[i], face_green[i], face_blue[i]});
  }

  // Declare the entities
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // Write the mesh to file
  size_t step = 1;  // Step = 0 doesn't write out fields...
  stk::io::write_mesh_with_fields("neuron.exo", bulk_data, step);
}

class Helix {
 public:
  /// Constructor
  Helix(const size_t &num_nodes, const double &radius, const double &pitch, const double &distance_between_nodes,
        const double &start_x, const double &start_y, const double &start_z, const double &axis_x, const double &axis_y,
        const double &axis_z)
      : num_nodes_(num_nodes),
        num_edges_(num_nodes - 1),
        radius_(radius),
        pitch_(pitch),
        b_(pitch / (2.0 * M_PI)),
        distance_between_nodes_(distance_between_nodes),
        delta_t_(distance_between_nodes_ / std::sqrt(radius_ * radius_ + b_ * b_)),
        start_{start_x, start_y, start_z},
        axis_{axis_x, axis_y, axis_z} {
    // A note about the calculation of delta_t given the fixed euclidean distance between points on the helix.
    //
    // The helix is parameterized by the parametric variable t, the radius a, and the pitch p.
    // To help, we will define b := p / (2 * pi).
    // The helix is then given by the parametric equations:
    // x = a * cos(t)
    // y = a * sin(t)
    // z = b * t
    //
    // The angle between the tangent vector and the plane perpendicular to the axis is given by:
    // phi = atan( b / a )
    //
    // This angle is invariant under projections of the tangent vector onto a plane that intersects the axis.
    //
    // Now, we are interested in discretizing the helix into a set of nodes that are a fixed distance apart. If we take
    // one node at t and the next at t + delta_t, then the euclidean distance between these nodes is fixed at some
    // prescribed amount d.
    //
    // Notice that, phi is also the angle between the vector from the current node at t to the next at t + delta_t and
    // the plane perpendicular to the helix axis. Thankfully, because of this projection relation, we can form two
    // similar triangles: one with base a, height b, and hypotenuse sqrt(a^2 + b^2), and the other with an unimportant
    // base, height b delta t, and hypotenuse d. This gives us
    // d sin(phi) = b * delta_t -> delta_t = d * sin(phi) / b = d * sin( atan( b / a ) ) / b = d / sqrt( a^2 + b^2 )

    // Not that we don't trust you or anything, but we need to make sure the axis is normalized.
    axis_ /= mundy::math::norm(axis_);

    // We need to find two orthonormal vectors to the axis of the helix.
    // We can do this by finding an arbitrary vector that is not parallel to the axis, taking the cross product
    // with the normal, and normalizing the result. This gives us a vector that is orthogonal to the axis.
    // By taking the cross product of the axis and this vector, we get a second vector that is orthogonal to both.
    const mundy::math::Vector3d ihat(1.0, 0.0, 0.0);
    const mundy::math::Vector3d jhat(0.0, 1.0, 0.0);
    basis_vector0_ = mundy::math::norm(mundy::math::cross(axis_, ihat)) > 1.0e-12 ? ihat : jhat;
    basis_vector0_ /= mundy::math::norm(basis_vector0_);
    basis_vector1_ = mundy::math::cross(axis_, basis_vector0_);
    basis_vector1_ /= mundy::math::norm(basis_vector1_);
  }

  /// \brief Get the grid coordinate corresponding to a given grid index.
  /// \param archlength_index The archlength index in [0 to num_nodes-1].
  /// \return The corresponding coordinate.
  std::array<double, 3> get_grid_coordinate(const size_t &archlength_index) const {
    // t = delta_t * archlength_index
    // x_ref = a * cos(t)
    // y_ref = a * sin(t)
    // z_ref = b * t
    //
    // pos = start + x_ref * basis_vector0 + y_ref * basis_vector1 + z_ref * axis
    const double t = delta_t_ * static_cast<double>(archlength_index);
    const double x_ref = radius_ * std::cos(t);
    const double y_ref = radius_ * std::sin(t);
    const double z_ref = b_ * t;
    const auto pos = start_ + x_ref * basis_vector0_ + y_ref * basis_vector1_ + z_ref * axis_;
    return {pos[0], pos[1], pos[2]};
  }

 private:
  size_t num_nodes_;
  size_t num_edges_;
  double radius_;
  double pitch_;
  double b_;
  double distance_between_nodes_;
  double delta_t_;
  mundy::math::Vector3d start_;
  mundy::math::Vector3d axis_;
  mundy::math::Vector3d basis_vector0_;
  mundy::math::Vector3d basis_vector1_;
};  // class Helix

void ciliated_sphere() {
  // A generated ciliated sphere
  // The sphere comes from sphere.ply
  std::cout << "Generating a ciliated sphere" << std::endl;

  // Setup a mesh
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  builder.set_aura_option(stk::mesh::BulkData::AUTO_AURA);

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  auto &meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("coordinates");

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  auto &bulk_data = *bulk_data_ptr;

  // Declare a triangle part with a triangle topology
  stk::mesh::Part &fixed_nodes_part = meta_data.declare_part_with_topology("fixed_nodes_part", stk::topology::NODE);
  stk::mesh::Part &triangle_part = meta_data.declare_part_with_topology("triangle_part", stk::topology::SHELL_TRI_3);
  stk::mesh::Part &segment_part = meta_data.declare_part_with_topology("segment_part", stk::topology::BEAM_2);
  stk::mesh::Part &angular_spring_part =
      meta_data.declare_part_with_topology("angular_spring_part", stk::topology::SHELL_TRI_3);
  stk::io::put_io_part_attribute(fixed_nodes_part);
  stk::io::put_io_part_attribute(triangle_part);
  stk::io::put_io_part_attribute(segment_part);
  stk::io::put_io_part_attribute(angular_spring_part);

  // Add a node and element-rank color field
  stk::mesh::Field<double> &node_mass_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "mass");
  stk::mesh::Field<double> &node_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "velocity");
  stk::mesh::Field<double> &node_acceleration_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "acceleration");
  stk::mesh::Field<double> &node_force_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "force");

  stk::mesh::Field<double> &element_radius =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
  stk::mesh::Field<double> &element_spring_constant =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "spring_constant");
  stk::mesh::Field<double> &element_angular_spring_constant =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "angular_spring_constant");

  stk::io::set_field_role(node_mass_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_acceleration_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_role(element_radius, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(element_spring_constant, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(element_angular_spring_constant, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(node_mass_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_acceleration_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);

  stk::io::set_field_output_type(element_radius, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(element_spring_constant, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(element_angular_spring_constant, stk::io::FieldOutputType::SCALAR);

  // Add the node coordinates field
  stk::mesh::Field<double> &node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");

  // Put the fields on the mesh
  stk::mesh::put_field_on_mesh(node_mass_field, segment_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, segment_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_acceleration_field, segment_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, segment_part, 3, nullptr);

  stk::mesh::put_field_on_mesh(element_radius, segment_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(element_spring_constant, segment_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(element_angular_spring_constant, angular_spring_part, 1, nullptr);

  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Commit the meta data
  meta_data.commit();

  // Fill the triangulated sphere from the ply file
  std::string filename = "sphere.ply";
  happly::PLYData ply_in(filename);
  mundy::mesh::DeclareEntitiesHelper dec_helper;
  size_t node_count = 0;
  size_t element_count = 0;

  // Setup the cilia
  //   Each vertex is the start of a new cilium
  //   Each cilia consists of a series of connected segments. Else then the very first and very last node, each node has
  //   an angular spring that goes between the previous and next node.
  const int num_segs_per_cilia = 500;
  const int num_nodes_per_cilia = num_segs_per_cilia + 1;
  const double segment_length = 0.001;
  const double helix_radius = 10 * segment_length;
  const double helix_pitch = 50 * segment_length;

  std::vector<double> x = ply_in.getElement("vertex").getProperty<double>("x");
  std::vector<double> y = ply_in.getElement("vertex").getProperty<double>("y");
  std::vector<double> z = ply_in.getElement("vertex").getProperty<double>("z");

  size_t num_verts = x.size();
  std::vector<size_t> vert_node_ids(num_verts);
  for (size_t i = 0; i < num_verts; ++i) {
    std::array<double, 3> start_coords = {x[i], y[i], z[i]};
    const double len = std::sqrt(start_coords[0] * start_coords[0] + start_coords[1] * start_coords[1] +
                                 start_coords[2] * start_coords[2]);
    std::array<double, 3> start_dir = {start_coords[0] / len, start_coords[1] / len, start_coords[2] / len};
    Helix helix(num_nodes_per_cilia, helix_radius, helix_pitch, segment_length, x[i], y[i], z[i], start_dir[0],
                start_dir[1], start_dir[2]);

    // Only the first node is fixed
    dec_helper.create_node()
        .owning_proc(0)               //
        .id(node_count + 1)           //
        .add_part(&fixed_nodes_part)  //
        .add_field_data<double>(&node_coords_field, {x[i], y[i], z[i]});
    vert_node_ids[i] = node_count + 1;
    node_count++;

    // The rest of the nodes
    for (int j = 1; j < num_nodes_per_cilia; ++j) {
      // The current node
      std::array<double, 3> coords = helix.get_grid_coordinate(j);
      dec_helper.create_node()
          .owning_proc(0)      //
          .id(node_count + 1)  //
          .add_field_data<double>(&node_coords_field, {coords[0], coords[1], coords[2]});
      node_count++;

      // The segment, connecting the left and current nodes
      dec_helper.create_element()
          .owning_proc(0)                   //
          .id(element_count + 1)            //
          .topology(stk::topology::BEAM_2)  //
          .add_part(&segment_part)          //
          .nodes({node_count - 1, node_count});
      element_count++;

      // The angular spring, connecting the left node and right nodes to the current
      if (j < num_nodes_per_cilia - 1) {
        dec_helper.create_element()
            .owning_proc(0)                                        //
            .id(element_count + 1)                                 //
            .topology(stk::topology::SHELL_TRI_3)                  //
            .add_part(&angular_spring_part)                        //
            .nodes({node_count - 1, node_count + 1, node_count});  // left, right, middle
        element_count++;
      }
    }
  }

  // Add the triangles for the sphere
  std::vector<std::vector<size_t>> face_ind = ply_in.getFaceIndices<size_t>();
  size_t num_faces = face_ind.size();
  for (size_t i = 0; i < num_faces; ++i) {
    MUNDY_THROW_REQUIRE(face_ind[i].size() == 3, std::runtime_error, "Triangle face must have 3 vertices");
    dec_helper.create_element()
        .owning_proc(0)                        //
        .id(element_count + 1)                 //
        .topology(stk::topology::SHELL_TRI_3)  //
        .add_part(&triangle_part)              //
        .nodes({vert_node_ids[face_ind[i][0]], vert_node_ids[face_ind[i][1]], vert_node_ids[face_ind[i][2]]});
    element_count++;
  }

  // Declare the entities
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // Write the mesh to file
  size_t step = 1;  // Step = 0 doesn't write out fields...
  stk::io::write_mesh_with_fields("ciliated_sphere.exo", bulk_data, step);
}

double distance_sq_from_point_to_line_segment(const mundy::math::Vector3d &x, const mundy::math::Vector3d &p1,
                                              const mundy::math::Vector3d &p2,
                                              mundy::math::Vector3d *const closest_point = nullptr,
                                              double *const t = nullptr) {
  // Define some temporary variables
  mundy::math::Vector3d closest_point_tmp;
  double t_tmp;

  // Determine appropriate vectors
  const auto p21 = p2 - p1;

  // Get parametric location
  const double num = mundy::math::dot(p21, x - p1);
  if ((num < mundy::math::get_zero_tolerance<double>()) & (num > -mundy::math::get_zero_tolerance<double>())) {
    // CASE 1: The vector from p1 to x is orthogonal to the line.
    // In this case, the closest point is p1 and the parametric coordinate is 0.
    t_tmp = 0.0;
    closest_point_tmp = p1;
  } else {
    const double denom = mundy::math::dot(p21, p21);

    if (denom < mundy::math::get_zero_tolerance<double>()) {
      // CASE 2: The line is degenerate (i.e., p1 and p2 are numerically the same point).
      // In this case, either point could really be the closest point. We'll arbitrarily pick p1 and set t to 0.
      closest_point_tmp = p1;
      t_tmp = 0.0;
    } else {
      // CASE 3: The line is well-defined and we can compute the closest point.
      t_tmp = num / denom;

      if (t_tmp < 0.0) {
        // CASE 3.1: The parameter for the infinite line is less than 0. Therefore, the closest point is p1.
        t_tmp = 0.0;
        closest_point_tmp = p1;
      } else if (t_tmp > 1.0) {
        // CASE 3.2: The parameter for the infinite line is greater than 1. Therefore, the closest point is p2.
        t_tmp = 1.0;
        closest_point_tmp = p2;
      } else {
        // CASE 3.3: The closest point is falls within the line segment.
        closest_point_tmp = p1 + t_tmp * p21;
      }
    }
  }

  const double distance_sq = mundy::math::two_norm_squared(closest_point_tmp - x);
  if (t != nullptr) {
    *t = t_tmp;
  }
  if (closest_point != nullptr) {
    *closest_point = closest_point_tmp;
  }
  return distance_sq;
}

void bacteria_in_a_porous_media() {
  // N bacteria (rods) in a porous media made of M overlapping, randomly placed spheres
  // The bacteria are placed randomly in the domain with a random orientation
  // If a bacteria is placed inside a sphere, we redraw its position and orientation

  std::cout << "Generating bacteria in a porous media" << std::endl;

  // Setup a mesh
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  builder.set_aura_option(stk::mesh::BulkData::AUTO_AURA);

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  auto &meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("coordinates");

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  auto &bulk_data = *bulk_data_ptr;

  // Declare the rod and sphere parts
  stk::mesh::Part &rod_part = meta_data.declare_part_with_topology("rod_part", stk::topology::BEAM_2);
  stk::mesh::Part &sphere_part = meta_data.declare_part_with_topology("sphere_part", stk::topology::PARTICLE);
  stk::io::put_io_part_attribute(rod_part);
  stk::io::put_io_part_attribute(sphere_part);

  // The rods have element radius and node velocity
  stk::mesh::Field<double> &rod_radius_field =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
  stk::mesh::Field<double> &node_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "velocity");

  // The spheres have element radius
  stk::mesh::Field<double> &sphere_radius_field =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");

  // The nodes have coordinates
  stk::mesh::Field<double> &node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");

  // Declare the field io roles and output types
  stk::io::set_field_role(rod_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(sphere_radius_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(rod_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(sphere_radius_field, stk::io::FieldOutputType::SCALAR);

  // Put the fields on the mesh
  stk::mesh::put_field_on_mesh(rod_radius_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(sphere_radius_field, sphere_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Commit the meta data
  meta_data.commit();

  // Fill the declare entities helper
  mundy::mesh::DeclareEntitiesHelper dec_helper;
  size_t node_count = 0;
  size_t element_count = 0;

  std::array<double, 3> domain_min = {0.0, 0.0, 0.0};
  std::array<double, 3> domain_max = {2.0, 2.0, 2.0};

  size_t num_rods = 100000;
  double rod_radius = 0.001;
  double min_rod_length = 4 * rod_radius;
  double max_rod_length = 8 * rod_radius;

  size_t num_spheres = 10000;
  double min_sphere_radius = 0.01;
  double max_sphere_radius = 0.1;

  // Place the spheres
  std::vector<mundy::math::Vector3d> sphere_centers(num_spheres);
  std::vector<double> sphere_radii(num_spheres);
  for (size_t i = 0; i < num_spheres; ++i) {
    double sphere_radius = min_sphere_radius + (max_sphere_radius - min_sphere_radius) * rand() / RAND_MAX;
    mundy::math::Vector3d sphere_center = {domain_min[0] + (domain_max[0] - domain_min[0]) * rand() / RAND_MAX,
                                           domain_min[1] + (domain_max[1] - domain_min[1]) * rand() / RAND_MAX,
                                           domain_min[2] + (domain_max[2] - domain_min[2]) * rand() / RAND_MAX};
    sphere_centers[i] = sphere_center;
    sphere_radii[i] = sphere_radius;

    dec_helper.create_node()
        .owning_proc(0)          //
        .id(node_count + 1)      //
        .add_part(&sphere_part)  //
        .add_field_data<double>(&node_coords_field, {sphere_center[0], sphere_center[1], sphere_center[2]});
    dec_helper.create_element()
        .owning_proc(0)                     //
        .id(element_count + 1)              //
        .topology(stk::topology::PARTICLE)  //
        .nodes({node_count + 1})            //
        .add_part(&sphere_part)             //
        .add_field_data<double>(&sphere_radius_field, {sphere_radius});
    node_count++;
    element_count++;
  }

  // Place the rods
  const size_t max_num_attempts_per_rod = 100000;
  for (size_t i = 0; i < num_rods; ++i) {
    for (size_t a = 0; a < max_num_attempts_per_rod; a++) {
      // Generate a random rod length
      double rod_length = min_rod_length + (max_rod_length - min_rod_length) * rand() / RAND_MAX;

      // Generate a random left node
      mundy::math::Vector3d left_node_coord = {domain_min[0] + (domain_max[0] - domain_min[0]) * rand() / RAND_MAX,
                                               domain_min[1] + (domain_max[1] - domain_min[1]) * rand() / RAND_MAX,
                                               domain_min[2] + (domain_max[2] - domain_min[2]) * rand() / RAND_MAX};

      // Generate a random orientation
      const double u1 = static_cast<double>(rand()) / RAND_MAX;
      const double u2 = static_cast<double>(rand()) / RAND_MAX;
      const double theta = 2.0 * M_PI * u1;
      const double phi = std::acos(2.0 * u2 - 1.0);
      mundy::math::Vector3d right_node_coord = {rod_length * std::sin(phi) * std::cos(theta) + left_node_coord[0],
                                                rod_length * std::sin(phi) * std::sin(theta) + left_node_coord[1],
                                                rod_length * std::cos(phi) + left_node_coord[2]};

      // Check if the rod intersects any spheres
      bool intersects = false;
      for (size_t i = 0; i < num_spheres; ++i) {
        const double distance_sq =
            distance_sq_from_point_to_line_segment(sphere_centers[i], left_node_coord, right_node_coord);
        if (std::sqrt(distance_sq) < sphere_radii[i] + rod_radius) {
          intersects = true;
          break;
        }
      }

      if (intersects) {
        MUNDY_THROW_REQUIRE(a < max_num_attempts_per_rod - 1, std::runtime_error, "Failed to place all rods");
        continue;  // Random sample failed. Try again.
      }

      // Declare the left and right nodes
      dec_helper.create_node()
          .owning_proc(0)      //
          .id(node_count + 1)  //
          .add_field_data<double>(&node_coords_field, {left_node_coord[0], left_node_coord[1], left_node_coord[2]});
      node_count++;

      dec_helper.create_node()
          .owning_proc(0)      //
          .id(node_count + 1)  //
          .add_field_data<double>(&node_coords_field, {right_node_coord[0], right_node_coord[1], right_node_coord[2]});
      node_count++;

      // Declare the rod
      dec_helper.create_element()
          .owning_proc(0)                   //
          .id(element_count + 1)            //
          .topology(stk::topology::BEAM_2)  //
          .add_part(&rod_part)              //
          .nodes({node_count - 1, node_count});
      element_count++;

      break;  // Successfully placed the rod
    }
  }

  // Declare the entities
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // Write the mesh to file
  size_t step = 1;  // Step = 0 doesn't write out fields...
  stk::io::write_mesh_with_fields("bacteria_in_a_porous_media.exo", bulk_data, step);
}

bool is_point_inside_ellipsoid(const mundy::math::Vector3d &point, const mundy::math::Vector3d &center,
                               const mundy::math::Vector3d &radii) {
  const double x = (point[0] - center[0]) / radii[0];
  const double y = (point[1] - center[1]) / radii[1];
  const double z = (point[2] - center[2]) / radii[2];
  return x * x + y * y + z * z <= 1.0;
}

mundy::math::Vector3d random_point_inside_ellipsoid(const mundy::math::Vector3d &center,
                                                    const mundy::math::Vector3d &radii) {
  // Generate random points within the bounding box of the ellipsoid
  // And reject points that are not within the ellipsoid until we get a winner
  const size_t max_num_attempts = 100000;
  for (size_t i = 0; i < max_num_attempts; ++i) {
    const mundy::math::Vector3d point = {
        center[0] - radii[0] + 2.0 * radii[0] * static_cast<double>(rand()) / RAND_MAX,
        center[1] - radii[1] + 2.0 * radii[1] * static_cast<double>(rand()) / RAND_MAX,
        center[2] - radii[2] + 2.0 * radii[2] * static_cast<double>(rand()) / RAND_MAX};
    const bool inside = is_point_inside_ellipsoid(point, center, radii);
    if (inside) {
      return point;
    }
  }
  return center;  // Failed to find a point. Return the center.
}

std::vector<mundy::math::Vector3d> random_walk_inside_ellipsoid(const mundy::math::Vector3d &start,
                                                                const mundy::math::Vector3d &center,
                                                                const mundy::math::Vector3d &radii,
                                                                const size_t num_steps, const double step_length) {
  std::vector<mundy::math::Vector3d> walk(num_steps);
  walk[0] = start;
  const size_t max_num_attempts = 100000;
  for (size_t i = 1; i < num_steps; ++i) {
    for (size_t a = 0; a < max_num_attempts; ++a) {
      const double u1 = static_cast<double>(rand()) / RAND_MAX;
      const double u2 = static_cast<double>(rand()) / RAND_MAX;
      const double theta = 2.0 * M_PI * u1;
      const double phi = std::acos(2.0 * u2 - 1.0);
      walk[i] = walk[i - 1] + step_length * mundy::math::Vector3d{std::sin(phi) * std::cos(theta),  //
                                                                  std::sin(phi) * std::sin(theta),  //
                                                                  std::cos(phi)};
      if (is_point_inside_ellipsoid(walk[i], center, radii)) {
        break;
      }
    }
  }

  return walk;
}

void chromatin() {
  // Initialize N chromatin fibers in an ellipsoidal nucleus each with repeating chunks of euchromatin and
  // heterochromatin. The chromatin will be modeled as a series of connected segments with spheres at the nodes and
  // springs along the segments. The spheres will be marked as either heterochromatin or euchromatin.
  // Initialization will be done by allowing the fibers to take a random walk in the nucleus without consideration for
  // fiber overlaps.
  //
  // Periphery binding sites will be placed randomly on the surface of the nucleus.

  std::cout << "Generating chromatin" << std::endl;

  // Setup a mesh
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  builder.set_aura_option(stk::mesh::BulkData::AUTO_AURA);

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  auto &meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("coordinates");

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  auto &bulk_data = *bulk_data_ptr;

  // Declare te parts
  stk::mesh::Part &nucleus_part = meta_data.declare_part_with_topology("nucleus", stk::topology::SHELL_TRI_3);
  stk::mesh::Part &chromatin_segment_part =
      meta_data.declare_part_with_topology("chromatin_segment", stk::topology::BEAM_2);
  stk::mesh::Part &euchromatin_part = meta_data.declare_part_with_topology("euchromatin", stk::topology::PARTICLE);
  stk::mesh::Part &heterochromatin_part =
      meta_data.declare_part_with_topology("heterochromatin", stk::topology::PARTICLE);
  stk::mesh::Part &nucleus_binding_site_part = meta_data.declare_part_with_topology(
      "nucleus_binding_site",
      stk::topology::PARTICLE);  // It looks like these need to be particles for the exodus output
  stk::io::put_io_part_attribute(nucleus_part);
  stk::io::put_io_part_attribute(chromatin_segment_part);
  stk::io::put_io_part_attribute(euchromatin_part);
  stk::io::put_io_part_attribute(heterochromatin_part);
  stk::io::put_io_part_attribute(nucleus_binding_site_part);

  // The segments will store a radius and spring constant
  stk::mesh::Field<double> &segment_radius_field =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
  stk::mesh::Field<double> &segment_spring_constant_field =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "spring_constant");

  // The euchromatin/heterochromatin will store a hydrodynamic radius
  stk::mesh::Field<double> &hydrodynamic_radius_field =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "hydrodynamic");

  // The nodes within the chromatin segments will store a velocity and a force
  stk::mesh::Field<double> &node_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "velocity");
  stk::mesh::Field<double> &node_force_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "force");

  // All nodes have a coordinate field
  stk::mesh::Field<double> &node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");

  // Assign the field roles
  stk::io::set_field_role(segment_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(segment_spring_constant_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(hydrodynamic_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);

  // Assign the field output types
  stk::io::set_field_output_type(segment_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(segment_spring_constant_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(hydrodynamic_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);

  // Put the fields on the mesh
  stk::mesh::put_field_on_mesh(segment_radius_field, chromatin_segment_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(segment_spring_constant_field, chromatin_segment_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(hydrodynamic_radius_field, euchromatin_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(hydrodynamic_radius_field, heterochromatin_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, chromatin_segment_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, chromatin_segment_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Commit the meta data
  meta_data.commit();

  // Fill the declare entities helper
  mundy::mesh::DeclareEntitiesHelper dec_helper;
  size_t node_count = 0;
  size_t element_count = 0;

  // Setup in the ellipsoidal nucleus
  std::string filename = "ellipsoid.ply";
  happly::PLYData ply_in(filename);

  std::vector<double> x = ply_in.getElement("vertex").getProperty<double>("x");
  std::vector<double> y = ply_in.getElement("vertex").getProperty<double>("y");
  std::vector<double> z = ply_in.getElement("vertex").getProperty<double>("z");
  std::vector<std::vector<size_t>> face_ind = ply_in.getFaceIndices<size_t>();

  const size_t num_verts = x.size();
  const size_t num_faces = face_ind.size();
  for (size_t i = 0; i < num_verts; ++i) {
    dec_helper.create_node()
        .owning_proc(0)      //
        .id(node_count + 1)  //
        .add_field_data<double>(&node_coords_field, {x[i], y[i], z[i]});
    node_count++;
  }

  for (size_t i = 0; i < num_faces; ++i) {
    MUNDY_THROW_REQUIRE(face_ind[i].size() == 3, std::runtime_error, "Triangle face must have 3 vertices");
    dec_helper.create_element()
        .owning_proc(0)                        //
        .id(element_count + 1)                 //
        .topology(stk::topology::SHELL_TRI_3)  //
        .add_part(&nucleus_part)               //
        .nodes({face_ind[i][0] + 1, face_ind[i][1] + 1,
                face_ind[i][2] + 1});  // only works because we declared the nucleus first
    element_count++;
  }

  // Setup the nucleus bind sites
  const size_t num_bind_sites = 10000;
  const double nucleus_axis_radius1 = 1.0;
  const double nucleus_axis_radius2 = 2.0;
  const double nucleus_axis_radius3 = 1.5;

  const double a = nucleus_axis_radius1;
  const double b = nucleus_axis_radius2;
  const double c = nucleus_axis_radius3;
  const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
  auto keep = [&a, &b, &c, &inv_mu_max](double x, double y, double z) {
    const double mu_xyz = std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
    return inv_mu_max * mu_xyz > (static_cast<double>(rand()) / RAND_MAX);
  };

  for (size_t i = 0; i < num_bind_sites; i++) {
    // Rejection sampling to place the periphery binding sites
    double node_coords[3];
    while (true) {
      // Generate a random point on the unit sphere
      const double u1 = (static_cast<double>(rand()) / RAND_MAX);
      const double u2 = (static_cast<double>(rand()) / RAND_MAX);
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
        .owning_proc(0)      //
        .id(node_count + 1)  //
        .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
    node_count++;

    // Declare the element
    dec_helper.create_element()
        .owning_proc(0)                        //
        .id(element_count + 1)                 //
        .topology(stk::topology::PARTICLE)     //
        .add_part(&nucleus_binding_site_part)  //
        .nodes({node_count});
    element_count++;
  }

  // Setup the chromatin fibers
  const mundy::math::Vector3d nucleus_center = {0.0, 0.0, 0.0};
  const mundy::math::Vector3d nucleus_radii = {a, b, c};
  const size_t num_fibers = 100;
  const size_t num_hetero_euchromatin_repeats = 7;
  const size_t num_heterochromatin_per_repeat = 90;
  const size_t num_euchromatin_per_repeat = 315;
  const size_t num_nodes_per_heterochromatin =
      num_hetero_euchromatin_repeats * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat);
  const double fiber_radius = 0.01;
  const double segment_length = 0.02;
  const double segment_spring_constant = 1.0;
  const double euchromatin_hydrodynamic_radius = 0.01;
  const double heterochromatin_hydrodynamic_radius = 0.01;

  for (size_t f = 0; f < num_fibers; f++) {
    // Generate a random walk for the fiber starting from a random location in the nucleus
    mundy::math::Vector3d start = random_point_inside_ellipsoid(nucleus_center, nucleus_radii);
    std::vector<mundy::math::Vector3d> fiber_walk = random_walk_inside_ellipsoid(
        start, nucleus_center, nucleus_radii, num_nodes_per_heterochromatin, segment_length);

    // Declare the nodes, segments, and heterochromatin/euchromatin
    for (size_t r = 0; r < num_hetero_euchromatin_repeats; ++r) {
      // Heterochromatin
      for (size_t h = 0; h < num_heterochromatin_per_repeat; ++h) {
        const size_t node_index = r * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat) + h;

        dec_helper.create_node()
            .owning_proc(0)      //
            .id(node_count + 1)  //
            .add_field_data<double>(&node_coords_field,
                                    {fiber_walk[node_index][0], fiber_walk[node_index][1], fiber_walk[node_index][2]});
        node_count++;

        // Only create the segment if the node is not the last node in this fiber
        if (node_index < num_nodes_per_heterochromatin - 1) {
          dec_helper.create_element()
              .owning_proc(0)                     //
              .id(element_count + 1)              //
              .topology(stk::topology::BEAM_2)    //
              .add_part(&chromatin_segment_part)  //
              .nodes({node_count, node_count + 1})
              .add_field_data<double>(&segment_radius_field, {fiber_radius})
              .add_field_data<double>(&segment_spring_constant_field, {segment_spring_constant});

          element_count++;
        }

        // Declare the heterochromatin
        dec_helper.create_element()
            .owning_proc(0)                     //
            .id(element_count + 1)              //
            .topology(stk::topology::PARTICLE)  //
            .add_part(&heterochromatin_part)    //
            .nodes({node_count})
            .add_field_data<double>(&hydrodynamic_radius_field, {heterochromatin_hydrodynamic_radius});
        element_count++;
      }

      for (size_t e = 0; e < num_euchromatin_per_repeat; ++e) {
        const size_t node_index =
            r * (num_heterochromatin_per_repeat + num_euchromatin_per_repeat) + num_heterochromatin_per_repeat + e;

        dec_helper.create_node()
            .owning_proc(0)      //
            .id(node_count + 1)  //
            .add_field_data<double>(&node_coords_field,
                                    {fiber_walk[node_index][0], fiber_walk[node_index][1], fiber_walk[node_index][2]});
        node_count++;

        // Only create the segment if the node is not the last node in this fiber
        if (node_index < num_nodes_per_heterochromatin - 1) {
          dec_helper.create_element()
              .owning_proc(0)                     //
              .id(element_count + 1)              //
              .topology(stk::topology::BEAM_2)    //
              .add_part(&chromatin_segment_part)  //
              .nodes({node_count, node_count + 1})
              .add_field_data<double>(&segment_radius_field, {fiber_radius})
              .add_field_data<double>(&segment_spring_constant_field, {segment_spring_constant});

          element_count++;
        }

        // Declare the euchromatin
        dec_helper.create_element()
            .owning_proc(0)                     //
            .id(element_count + 1)              //
            .topology(stk::topology::PARTICLE)  //
            .add_part(&euchromatin_part)        //
            .nodes({node_count})
            .add_field_data<double>(&hydrodynamic_radius_field, {euchromatin_hydrodynamic_radius});
        element_count++;
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
  stk::io::write_mesh_with_fields("chromatin.exo", bulk_data, step);
}

bool line_segment_intersects_triangle(const mundy::math::Vector3d &p1, const mundy::math::Vector3d &p2,
                                      const mundy::math::Vector3d &v1, const mundy::math::Vector3d &v2,
                                      const mundy::math::Vector3d &v3) {
  // Tolerance for floating-point comparisons
  constexpr double epsilon = 1e-12;

  // Calculate the normal of the plane
  const auto edge1 = v2 - v1;
  const auto edge2 = v3 - v1;
  const auto n = mundy::math::cross(edge1, edge2);

  // Check if line segment endpoints are on opposite sides of the plane
  const auto s1 = p1 - v1;
  const auto s2 = p2 - v1;
  double d1 = mundy::math::dot(n, s1);
  double d2 = mundy::math::dot(n, s2);

  if (std::abs(d1) < epsilon || std::abs(d2) < epsilon) return false;  // Edge case: endpoint on plane
  if (d1 * d2 > 0.0) return false;                                     // Same side of the plane

  // Direction vector for the ray (p1 to p2)
  const auto dir = p2 - p1;
  const auto h = mundy::math::cross(dir, edge2);
  double a = mundy::math::dot(edge1, h);

  // Check if the ray is parallel to the triangle plane
  if (std::abs(a) < epsilon) return false;  // Parallel: no intersection

  double inv_a = 1.0 / a;
  double u = inv_a * mundy::math::dot(s1, h);

  // Check if the intersection is outside the triangle
  if (u < 0.0 || u > 1.0) return false;

  const auto q = mundy::math::cross(s1, edge1);
  double v = inv_a * mundy::math::dot(dir, q);

  // Check if the intersection is outside the triangle
  if (v < 0.0 || (u + v) > 1.0) return false;

  // At this stage, we know there is an intersection with the triangle plane
  double t = inv_a * mundy::math::dot(edge2, q);

  // Check if the intersection lies on the segment (0 <= t <= 1)
  return (t >= 0.0 && t <= 1.0);
}

double line_segment_signed_distance_to_triangle(const mundy::math::Vector3d &p1, const mundy::math::Vector3d &p2,
                                                const mundy::math::Vector3d &v1, const mundy::math::Vector3d &v2,
                                                const mundy::math::Vector3d &v3) {
  // Tolerance for floating-point comparisons
  constexpr double epsilon = 1e-12;

  // Calculate the normal of the plane
  const auto edge1 = v2 - v1;
  const auto edge2 = v3 - v1;
  auto n = mundy::math::cross(edge1, edge2);
  n /= mundy::math::norm(n);

  // Signed distances from p1 and p2 to the plane of the triangle
  const auto s1 = p1 - v1;
  const auto s2 = p2 - v1;
  double d1 = mundy::math::dot(n, s1);
  double d2 = mundy::math::dot(n, s2);

  // Check if the endpoints are on opposite sides of the plane
  if (d1 * d2 <= 0.0) {
    // The line segment intersects the plane
    const auto dir = p2 - p1;
    const auto h = mundy::math::cross(dir, edge2);
    double a = mundy::math::dot(edge1, h);

    if (std::abs(a) < epsilon) {
      return 0.0;  // Segment lies in plane (rare edge case)
    }

    double inv_a = 1.0 / a;
    double u = inv_a * mundy::math::dot(s1, h);
    if (u < 0.0 || u > 1.0) return std::min(d1, d2);  // Closest distance to plane if outside triangle bounds

    const auto q = mundy::math::cross(s1, edge1);
    double v = inv_a * mundy::math::dot(dir, q);
    if (v < 0.0 || (u + v) > 1.0) return std::min(d1, d2);  // Closest distance to plane if outside triangle bounds

    double t = inv_a * mundy::math::dot(edge2, q);
    if (t >= 0.0 && t <= 1.0) {
      return -std::min(std::abs(d1), std::abs(d2));  // Negative to indicate overlap
    }
  }

  // No intersection: return the signed distance to the closest endpoint
  return d1 < d2 ? d1 : d2;
}

void bee_hive() {
  // A cellular lattice of extruded hexagonal cells
  // On each triangular face of the hexagonal cells, we want to randomly place binding sites
  // Within each cell, we will randomly place N fibers that do not overlap with
  // any of the faces.

  std::cout << "Generating bee hive" << std::endl;

  // Setup a mesh
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  builder.set_aura_option(stk::mesh::BulkData::AUTO_AURA);

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  auto &meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  meta_data.set_coordinate_field_name("coordinates");

  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  auto &bulk_data = *bulk_data_ptr;

  // Declare the rod and sphere parts
  stk::mesh::Part &rod_part = meta_data.declare_part_with_topology("rod_part", stk::topology::BEAM_2);
  stk::mesh::Part &triangle_part = meta_data.declare_part_with_topology("triangle_part", stk::topology::SHELL_TRI_3);
  stk::mesh::Part &binding_site_part =
      meta_data.declare_part_with_topology("binding_site_part", stk::topology::PARTICLE);
  stk::io::put_io_part_attribute(rod_part);
  stk::io::put_io_part_attribute(triangle_part);
  stk::io::put_io_part_attribute(binding_site_part);

  // The rods have element radius and node velocity
  stk::mesh::Field<double> &rod_radius_field =  //
      meta_data.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
  stk::mesh::Field<double> &node_velocity_field =  //
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "velocity");

  // The nodes have coordinates
  stk::mesh::Field<double> &node_coords_field =
      meta_data.declare_field<double>(stk::topology::NODE_RANK, "coordinates");

  // Declare the field io roles and output types
  stk::io::set_field_role(rod_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_velocity_field, Ioss::Field::TRANSIENT);

  stk::io::set_field_output_type(rod_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(node_velocity_field, stk::io::FieldOutputType::VECTOR_3D);

  // Put the fields on the mesh
  stk::mesh::put_field_on_mesh(rod_radius_field, rod_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, rod_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);

  // Commit the meta data
  meta_data.commit();

  // Fill the declare entities helper
  mundy::mesh::DeclareEntitiesHelper dec_helper;
  size_t node_count = 0;
  size_t element_count = 0;

  // Read in the bee hive mesh
  std::string filename = "bee_hive.ply";
  happly::PLYData ply_in(filename);

  std::vector<double> x = ply_in.getElement("vertex").getProperty<double>("x");
  std::vector<double> y = ply_in.getElement("vertex").getProperty<double>("y");
  std::vector<double> z = ply_in.getElement("vertex").getProperty<double>("z");
  std::vector<std::vector<size_t>> face_ind = ply_in.getFaceIndices<size_t>();

  const size_t num_verts = x.size();
  const size_t num_faces = face_ind.size();
  for (size_t i = 0; i < num_verts; ++i) {
    dec_helper.create_node()
        .owning_proc(0)      //
        .id(node_count + 1)  //
        .add_field_data<double>(&node_coords_field, {x[i], y[i], z[i]});
    node_count++;
  }

  for (size_t i = 0; i < num_faces; ++i) {
    MUNDY_THROW_REQUIRE(face_ind[i].size() == 3, std::runtime_error, "Triangle face must have 3 vertices");
    dec_helper.create_element()
        .owning_proc(0)                        //
        .id(element_count + 1)                 //
        .topology(stk::topology::SHELL_TRI_3)  //
        .add_part(&triangle_part)              //
        .nodes({face_ind[i][0] + 1, face_ind[i][1] + 1,
                face_ind[i][2] + 1});  // only works because we declared the hive first
    element_count++;
  }

  // Setup the binding sites
  const size_t num_bind_sites_per_face = 100;
  for (size_t i = 0; i < num_faces; i++) {
    const mundy::math::Vector3d coord1 = {x[face_ind[i][0]], y[face_ind[i][0]], z[face_ind[i][0]]};
    const mundy::math::Vector3d coord2 = {x[face_ind[i][1]], y[face_ind[i][1]], z[face_ind[i][1]]};
    const mundy::math::Vector3d coord3 = {x[face_ind[i][2]], y[face_ind[i][2]], z[face_ind[i][2]]};

    for (size_t j = 0; j < num_bind_sites_per_face; j++) {
      // Generate a random point on the triangle
      double u1 = static_cast<double>(rand()) / RAND_MAX;
      double u2 = static_cast<double>(rand()) / RAND_MAX;
      if (u1 + u2 > 1.0) {
        u1 = 1.0 - u1;
        u2 = 1.0 - u2;
      }

      // Calculate the random point using barycentric coordinates
      const auto coord = coord1 * (1.0 - u1 - u2) + coord2 * u1 + coord3 * u2;

      dec_helper.create_node()
          .owning_proc(0)      //
          .id(node_count + 1)  //
          .add_field_data<double>(&node_coords_field, {coord[0], coord[1], coord[2]});
      node_count++;

      dec_helper.create_element()
          .owning_proc(0)                     //
          .id(element_count + 1)              //
          .topology(stk::topology::PARTICLE)  //
          .add_part(&binding_site_part)       //
          .nodes({node_count});
      element_count++;
    }
  }

  // Generate the rods
  const size_t num_rods = 100000;
  const size_t max_num_attempts_per_rod = 100;
  const double rod_radius = 0.01;
  const double min_rod_length = 4 * rod_radius;
  const double max_rod_length = 8 * rod_radius;

  // Fill the domain min and max as the min and max of the vertices
  mundy::math::Vector3d domain_min = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max()};
  mundy::math::Vector3d domain_max = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(),
                                      -std::numeric_limits<double>::max()};
  for (size_t i = 0; i < num_verts; ++i) {
    domain_min[0] = std::min(domain_min[0], x[i]);
    domain_min[1] = std::min(domain_min[1], y[i]);
    domain_min[2] = std::min(domain_min[2], z[i]);
    domain_max[0] = std::max(domain_max[0], x[i]);
    domain_max[1] = std::max(domain_max[1], y[i]);
    domain_max[2] = std::max(domain_max[2], z[i]);
  }

  // Start by just creating random rods in the domain
  for (size_t i = 0; i < num_rods; ++i) {
    for (size_t a = 0; a < max_num_attempts_per_rod; ++a) {
      MUNDY_THROW_REQUIRE(a < max_num_attempts_per_rod - 1, std::runtime_error, "Failed to place all rods");

      // Generate a random rod length
      double rod_length = min_rod_length + (max_rod_length - min_rod_length) * rand() / RAND_MAX;

      // Generate a random left node
      const mundy::math::Vector3d left_node_coord = {
          domain_min[0] + (domain_max[0] - domain_min[0]) * rand() / RAND_MAX,
          domain_min[1] + (domain_max[1] - domain_min[1]) * rand() / RAND_MAX,
          domain_min[2] + (domain_max[2] - domain_min[2]) * rand() / RAND_MAX};

      // Generate a random orientation
      const double u1 = static_cast<double>(rand()) / RAND_MAX;
      const double u2 = static_cast<double>(rand()) / RAND_MAX;
      const double theta = 2.0 * M_PI * u1;
      const double phi = std::acos(2.0 * u2 - 1.0);
      mundy::math::Vector3d right_node_coord = {rod_length * std::sin(phi) * std::cos(theta) + left_node_coord[0],
                                                rod_length * std::sin(phi) * std::sin(theta) + left_node_coord[1],
                                                rod_length * std::cos(phi) + left_node_coord[2]};

      // Check if the rod intersects any triangles TODO: This currently doesn't work!!
      bool intersects = false;
      // for (size_t j = 0; j < num_faces; ++j) {
      //   const mundy::math::Vector3d coord1 = {x[face_ind[j][0]], y[face_ind[j][0]], z[face_ind[j][0]]};
      //   const mundy::math::Vector3d coord2 = {x[face_ind[j][1]], y[face_ind[j][1]], z[face_ind[j][1]]};
      //   const mundy::math::Vector3d coord3 = {x[face_ind[j][2]], y[face_ind[j][2]], z[face_ind[j][2]]};

      //   const double signed_sep = line_segment_signed_distance_to_triangle(left_node_coord, right_node_coord, coord1,
      //   coord2, coord3) - rod_radius; if (signed_sep < 0.0) {
      //     intersects = true;
      //     break;
      //   }
      // }

      if (!intersects) {
        // Declare the left and right nodes
        dec_helper.create_node()
            .owning_proc(0)      //
            .id(node_count + 1)  //
            .add_field_data<double>(&node_coords_field, {left_node_coord[0], left_node_coord[1], left_node_coord[2]});
        node_count++;

        dec_helper.create_node()
            .owning_proc(0)      //
            .id(node_count + 1)  //
            .add_field_data<double>(&node_coords_field,
                                    {right_node_coord[0], right_node_coord[1], right_node_coord[2]});
        node_count++;

        // Declare the rod
        dec_helper.create_element()
            .owning_proc(0)                   //
            .id(element_count + 1)            //
            .topology(stk::topology::BEAM_2)  //
            .add_part(&rod_part)              //
            .nodes({node_count - 1, node_count})
            .add_field_data<double>(&rod_radius_field, {rod_radius});
        element_count++;

        break;  // Successfully placed the rod
      }
    }
  }

  // Declare the entities
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // Write the mesh to file
  size_t step = 1;  // Step = 0 doesn't write out fields...
  stk::io::write_mesh_with_fields("bee_hive.exo", bulk_data, step);
}

int main(int argc, char **argv) {
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  neuron();
  ciliated_sphere();
  bacteria_in_a_porous_media();
  chromatin();
  bee_hive();

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
