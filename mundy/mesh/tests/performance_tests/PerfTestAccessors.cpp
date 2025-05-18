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

// C++ core libs
#include <algorithm>    // for std::max
#include <map>          // for std::map
#include <memory>       // for std::shared_ptr, std::unique_ptr
#include <stdexcept>    // for std::logic_error, std::invalid_argument
#include <string>       // for std::string
#include <type_traits>  // for std::enable_if, std::is_base_of, std::conjunction, std::is_convertible
#include <utility>      // for std::move, std::pair, std::make_pair
#include <vector>       // for std::vector

// Trilinos libs
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Part.hpp>  // for stk::mesh::Part
#include <stk_mesh/base/Selector.hpp>

// Mundy libs
#include <mundy_math/Vector3.hpp>      // for mundy::math::Vector3
#include <mundy_mesh/Aggregate.hpp>    // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>     // for mundy::mesh::BulkData
#include <mundy_mesh/MeshBuilder.hpp>  // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>     // for mundy::mesh::MetaData

namespace mundy {

namespace mesh {

namespace {

void run_test() {
  // For each of the accessor types,
  //   - create a field of the necessary size
  //   - populate it with data
  //   - create an accessor
  //   - create a vector of entities
  //   - loop over all entities and fetch the data directly, perform some operation.
  //   - loop over all entities and fetch the data via the accessor and perform the same operation.

  size_t num_trials = 100;

  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  using DoubleField = stk::mesh::Field<double>;
  DoubleField& scalar_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "SCALAR");
  DoubleField& vector3_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VECTOR3");
  DoubleField& matrix3_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "MATRIX3");
  DoubleField& quaternion_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "QUATERNION");
  DoubleField& aabb_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "AABB");

  double expected_scalar_data[1] = {1.0};
  double expected_vector3_data[3] = {1.0, 2.0, 3.0};
  double expected_matrix3_data[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  double expected_quaternion_data[4] = {1.0, 2.0, 3.0, 4.0};
  double expected_aabb_data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  stk::mesh::put_field_on_mesh(scalar_field, meta_data.universal_part(), 1, expected_scalar_data);
  stk::mesh::put_field_on_mesh(vector3_field, meta_data.universal_part(), 3, expected_vector3_data);
  stk::mesh::put_field_on_mesh(matrix3_field, meta_data.universal_part(), 9, expected_matrix3_data);
  stk::mesh::put_field_on_mesh(quaternion_field, meta_data.universal_part(), 4, expected_quaternion_data);
  stk::mesh::put_field_on_mesh(aabb_field, meta_data.universal_part(), 6, expected_aabb_data);
  meta_data.commit();

  // Create the nodes and populate the fields
  bulk_data.modification_begin();
  size_t num_nodes = 100000;
  std::vector<stk::mesh::Entity> nodes(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    stk::mesh::Entity node = bulk_data.declare_node(i + 1);  // 1-based indexing
    nodes[i] = node;
    stk::mesh::field_data(scalar_field, node)[0] = expected_scalar_data[0];
    for (size_t j = 0; j < 3; ++j) {
      stk::mesh::field_data(vector3_field, node)[j] = expected_vector3_data[j];
    }
    for (size_t j = 0; j < 9; ++j) {
      stk::mesh::field_data(matrix3_field, node)[j] = expected_matrix3_data[j];
    }
    for (size_t j = 0; j < 4; ++j) {
      stk::mesh::field_data(quaternion_field, node)[j] = expected_quaternion_data[j];
    }
    for (size_t j = 0; j < 6; ++j) {
      stk::mesh::field_data(aabb_field, node)[j] = expected_aabb_data[j];
    }
  }
  bulk_data.modification_end();

  // Create the accessors
  auto scalar_accessor = ScalarFieldComponent(scalar_field);
  auto vector3_accessor = Vector3FieldComponent(vector3_field);
  auto matrix3_accessor = Matrix3FieldComponent(matrix3_field);
  auto quaternion_accessor = QuaternionFieldComponent(quaternion_field);
  auto aabb_accessor = AABBFieldComponent(aabb_field);

  ////////////
  // Scalar //
  ////////////
  Kokkos::Timer scalar_field_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      double& scalar = stk::mesh::field_data(scalar_field, node)[0];
      scalar += i;
    }
  }
  double scalar_field_time = scalar_field_timer.seconds() / num_trials;

  Kokkos::Timer scalar_accessor_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      auto scalar = scalar_accessor(node);
      scalar[0] += i;
    }
  }
  double scalar_accessor_time = scalar_accessor_timer.seconds() / num_trials;
  std::cout << "Scalar field time: " << scalar_field_time << " vs Scalar accessor time: " << scalar_accessor_time
            << std::endl;
  std::cout << " Ratio (accessor/field): " << scalar_accessor_time / scalar_field_time << std::endl;

  /////////////
  // Vector3 //
  /////////////
  Kokkos::Timer vector3_field_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      for (size_t j = 0; j < 3; ++j) {
        double& vector3 = stk::mesh::field_data(vector3_field, node)[j];
        vector3 += i * j;
      }
    }
  }
  double vector3_field_time = vector3_field_timer.seconds() / num_trials;

  Kokkos::Timer vector3_accessor_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      auto vector3 = vector3_accessor(node);
      for (size_t j = 0; j < 3; ++j) {
        vector3[j] += i * j;
      }
    }
  }
  double vector3_accessor_time = vector3_accessor_timer.seconds() / num_trials;
  std::cout << "Vector3 field time: " << vector3_field_time << " vs Vector3 accessor time: " << vector3_accessor_time
            << std::endl;
  std::cout << " Ratio (accessor/field): " << vector3_accessor_time / vector3_field_time << std::endl;
  /////////////
  // Matrix3 //
  /////////////
  Kokkos::Timer matrix3_field_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      for (size_t j = 0; j < 9; ++j) {
        double& matrix3 = stk::mesh::field_data(matrix3_field, node)[j];
        matrix3 += i * j;
      }
    }
  }
  double matrix3_field_time = matrix3_field_timer.seconds() / num_trials;

  Kokkos::Timer matrix3_accessor_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      auto matrix3 = matrix3_accessor(node);
      for (size_t j = 0; j < 9; ++j) {
        matrix3[j] += i * j;
      }
    }
  }
  double matrix3_accessor_time = matrix3_accessor_timer.seconds() / num_trials;
  std::cout << "Matrix3 field time: " << matrix3_field_time << " vs Matrix3 accessor time: " << matrix3_accessor_time
            << std::endl;
  std::cout << " Ratio (accessor/field): " << matrix3_accessor_time / matrix3_field_time << std::endl;

  ////////////////
  // Quaternion //
  ////////////////
  Kokkos::Timer quaternion_field_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      for (size_t j = 0; j < 4; ++j) {
        double& quaternion = stk::mesh::field_data(quaternion_field, node)[j];
        quaternion += i * j;
      }
    }
  }
  double quaternion_field_time = quaternion_field_timer.seconds() / num_trials;

  Kokkos::Timer quaternion_accessor_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      auto quaternion = quaternion_accessor(node);
      for (size_t j = 0; j < 4; ++j) {
        quaternion[j] += i * j;
      }
    }
  }
  double quaternion_accessor_time = quaternion_accessor_timer.seconds() / num_trials;
  std::cout << "Quaternion field time: " << quaternion_field_time
            << " vs Quaternion accessor time: " << quaternion_accessor_time << std::endl;
  std::cout << " Ratio (accessor/field): " << quaternion_accessor_time / quaternion_field_time << std::endl;

  //////////
  // AABB //
  //////////
  Kokkos::Timer aabb_field_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      for (size_t j = 0; j < 6; ++j) {
        double& aabb = stk::mesh::field_data(aabb_field, node)[j];
        aabb += i * j;
      }
    }
  }
  double aabb_field_time = aabb_field_timer.seconds() / num_trials;

  Kokkos::Timer aabb_accessor_timer;
  for (size_t t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < num_nodes; ++i) {
      stk::mesh::Entity node = nodes[i];
      auto aabb = aabb_accessor(node);
      for (size_t j = 0; j < 6; ++j) {
        aabb[j] += i * j;
      }
    }
  }
  double aabb_accessor_time = aabb_accessor_timer.seconds() / num_trials;
  std::cout << "AABB field time: " << aabb_field_time << " vs AABB accessor time: " << aabb_accessor_time << std::endl;
  std::cout << " Ratio (accessor/field): " << aabb_accessor_time / aabb_field_time << std::endl;
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