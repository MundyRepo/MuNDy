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

// External libs
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

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

// template <typename Radius, typename Center>
// struct move_spheres {
//   /// \brief Apply this functor to a single sphere object
//   void operator()(auto& sphere_view) const {
//     // We'll fetch the index
//     std::size_t i = sphere_view.entity();

//     // We'll fetch the center
//     double& c = sphere_view.template get<Center>(0);

//     // We'll fetch the radius (from radii[i])
//     float& r = sphere_view.template get<Radius>();

//     std::cout << "Entity " << i << ": center = " << c << ", radius = " << r << "\n";

//     // Maybe we update them
//     c += 0.1 * i;
//     r *= 1.01f;
//   }

//   /// \brief Apply this functor to all entities in a sphere aggregate
//   /// Syncs all components to the owning space and marks marks modified components.
//   void apply_to(const auto& sphere_data, const stk::mesh::Selector& subset_selector) {
//     sphere_data.sync_to_device<Radius, Center>();
//     sphere_data.template for_each((*this), subset_selector);
//     sphere_data.modified_on_device<Center>();
//   }
// };

struct SCALAR_DATA {};
struct VECTOR3_DATA {};
struct MATRIX3_DATA {};
struct QUATERNION_DATA {};
struct AABB_DATA {};

TEST(UnitTestAggregate, Accessors) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

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

  bulk_data.modification_begin();
  size_t num_nodes = 10;
  for (size_t i = 0; i < num_nodes; ++i) {
    stk::mesh::Entity node = bulk_data.declare_node(i + 1);  // 1-based indexing
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

  // Fetch the data for the entity via the accessor's operator()
  for (size_t i = 0; i < num_nodes; ++i) {
    stk::mesh::Entity node = bulk_data.get_entity(stk::topology::NODE_RANK, i + 1);
    auto scalar = scalar_accessor(node);
    auto vector3 = vector3_accessor(node);
    auto matrix3 = matrix3_accessor(node);
    auto quaternion = quaternion_accessor(node);
    auto aabb = aabb_accessor(node);
    EXPECT_DOUBLE_EQ(scalar[0], expected_scalar_data[0]);
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(vector3[j], expected_vector3_data[j]);
    }
    for (size_t j = 0; j < 9; ++j) {
      EXPECT_DOUBLE_EQ(matrix3[j], expected_matrix3_data[j]);
    }
    for (size_t j = 0; j < 4; ++j) {
      EXPECT_DOUBLE_EQ(quaternion[j], expected_quaternion_data[j]);
    }
    for (size_t j = 0; j < 6; ++j) {
      EXPECT_DOUBLE_EQ(aabb[j], expected_aabb_data[j]);
    }
  }
}

TEST(UnitTestAggregate, BasicUsage) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& sphere_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
  stk::mesh::Field<double>& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "CENTER");
  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "RADIUS");

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, sphere_part, 1, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  stk::mesh::Entity node1 = bulk_data.declare_node(1);
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&sphere_part});
  bulk_data.declare_relation(elem1, node1, 0);
  bulk_data.modification_end();

  // Populate the center and radius
  math::Vector3<double> expected_center{1.0, 2.0, 3.0};
  double expected_radius = 0.5;
  stk::mesh::field_data(node_center_field, node1)[0] = expected_center[0];
  stk::mesh::field_data(node_center_field, node1)[1] = expected_center[1];
  stk::mesh::field_data(node_center_field, node1)[2] = expected_center[2];
  stk::mesh::field_data(elem_radius_field, elem1)[0] = expected_radius;

  // Create the accessors
  auto center_accessor = Vector3FieldComponent(node_center_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);

  // Fetch the data for the entity via the accessor's operator()
  auto center = center_accessor(node1);
  auto radius = radius_accessor(elem1);
  EXPECT_DOUBLE_EQ(center[0], expected_center[0]);
  EXPECT_DOUBLE_EQ(center[1], expected_center[1]);
  EXPECT_DOUBLE_EQ(center[2], expected_center[2]);
  EXPECT_DOUBLE_EQ(radius[0], expected_radius);

  // Create an aggregate for the spheres
  const auto collision_sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
                                         .add_component<CENTER, stk::topology::NODE_RANK>(center_accessor)
                                         .add_component<COLLISION_RADIUS, stk::topology::ELEM_RANK>(radius_accessor);

  // Validate our get_component() method
  EXPECT_EQ(collision_sphere_data.get_component<CENTER>().component().field().mesh_meta_data_ordinal(),
            node_center_field.mesh_meta_data_ordinal());
  EXPECT_EQ(collision_sphere_data.get_component<COLLISION_RADIUS>().component().field().mesh_meta_data_ordinal(),
            elem_radius_field.mesh_meta_data_ordinal());

  // Validate our selector
  EXPECT_EQ(collision_sphere_data.selector(), stk::mesh::Selector(sphere_part));

  // Get an accessor-independent view of the entity's data
  auto sphere_view = collision_sphere_data.get_view(elem1);
  EXPECT_EQ(sphere_view.entity(), elem1);
  EXPECT_EQ(sphere_view.rank(), stk::topology::ELEM_RANK);
  EXPECT_EQ(sphere_view.topology(), stk::topology::PARTICLE);
  unsigned center_node_con_ordinal = 0;
  auto also_center = sphere_view.get<CENTER>(center_node_con_ordinal);
  auto also_radius = sphere_view.get<COLLISION_RADIUS>();
  EXPECT_DOUBLE_EQ(also_center[0], expected_center[0]);
  EXPECT_DOUBLE_EQ(also_center[1], expected_center[1]);
  EXPECT_DOUBLE_EQ(also_center[2], expected_center[2]);
  EXPECT_DOUBLE_EQ(also_radius[0], expected_radius);

  collision_sphere_data.for_each([&expected_center, &expected_radius, &elem1](auto& other_sphere_view) {
    // To avoid having users worry about the return type of get<TAG>() and if it returns a reference or a view,
    // we switched to always returning a view even if the return type is a scalar. This means that you should always
    // use auto to capture the return value of get<TAG>(). ScalarViews are just VectorViews of size 1, so they should
    // feel the same to the user and have all the same operators/operations.
    auto c = other_sphere_view.template get<CENTER>(0);
    auto r = other_sphere_view.template get<COLLISION_RADIUS>();

    // Because calling .template get<TAG>() is syntactically awkward, we offer a get<TAG>(view) method
    auto c2 = get<CENTER>(other_sphere_view, 0);
    auto r2 = get<COLLISION_RADIUS>(other_sphere_view);

    // There is only one sphere, so we can perform the same checks as above
    EXPECT_DOUBLE_EQ(c[0], expected_center[0]);
    EXPECT_DOUBLE_EQ(c[1], expected_center[1]);
    EXPECT_DOUBLE_EQ(c[2], expected_center[2]);
    EXPECT_DOUBLE_EQ(r[0], expected_radius);

    EXPECT_DOUBLE_EQ(c2[0], expected_center[0]);
    EXPECT_DOUBLE_EQ(c2[1], expected_center[1]);
    EXPECT_DOUBLE_EQ(c2[2], expected_center[2]);
    EXPECT_DOUBLE_EQ(r2[0], expected_radius);

    EXPECT_EQ(other_sphere_view.entity(), elem1);
    EXPECT_EQ(other_sphere_view.rank(), stk::topology::ELEM_RANK);
    EXPECT_EQ(other_sphere_view.topology(), stk::topology::PARTICLE);
  });
}

struct a_non_lambda_functor {
  KOKKOS_INLINE_FUNCTION
  int operator()(auto& sphere_view) const {
    auto c = sphere_view.template get<CENTER>(0);
    auto r = sphere_view.template get<COLLISION_RADIUS>();
    c[0] += 1.0;
    r += 1.0;
    return 2;
  }
};

void run_canonical_test() {
  // Setup
  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  stk::mesh::Part& sphere_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
  stk::mesh::Field<double>& node_center_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "CENTER");
  stk::mesh::Field<double>& node_force_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "FORCE");
  stk::mesh::Field<double>& node_velocity_field = meta_data.declare_field<double>(stk::topology::NODE_RANK, "VELOCITY");

  stk::mesh::Field<double>& elem_radius_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "RADIUS");
  stk::mesh::Field<double>& elem_mass_field = meta_data.declare_field<double>(stk::topology::ELEM_RANK, "MASS");

  stk::mesh::put_field_on_mesh(node_center_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, sphere_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_velocity_field, sphere_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_radius_field, sphere_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_mass_field, sphere_part, 1, nullptr);
  meta_data.commit();

  bulk_data.modification_begin();
  unsigned num_spheres = 10;
  for (unsigned i = 0; i < num_spheres; ++i) {
    // STK is 1 indexed for EntityIds
    stk::mesh::Entity node = bulk_data.declare_node(i + 1);
    stk::mesh::Entity elem = bulk_data.declare_element(i + 1, stk::mesh::PartVector{&sphere_part});
    bulk_data.declare_relation(elem, node, 0);

    // Populate the fields
    vector3_field_data(node_center_field, node).set(1.1 * i, 2.2 * i, 3.3);
    vector3_field_data(node_force_field, node).set(5.0, 6.0, 7.0);
    vector3_field_data(node_velocity_field, node).set(1.0, 2.0, 3.0);
    scalar_field_data(elem_radius_field, elem).set(0.5);
    scalar_field_data(elem_mass_field, elem).set(1.0);
  }
  bulk_data.modification_end();

  // Create the accessors
  auto center_accessor = Vector3FieldComponent(node_center_field);
  auto force_accessor = Vector3FieldComponent(node_force_field);
  auto velocity_accessor = Vector3FieldComponent(node_velocity_field);
  auto radius_accessor = ScalarFieldComponent(elem_radius_field);
  auto mass_accessor = ScalarFieldComponent(elem_mass_field);

  // Create an aggregate for the spheres
  const auto sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
                               .add_component<CENTER, stk::topology::NODE_RANK>(center_accessor)
                               .add_component<FORCE, stk::topology::NODE_RANK>(force_accessor)
                               .add_component<VELOCITY, stk::topology::NODE_RANK>(velocity_accessor)
                               .add_component<COLLISION_RADIUS, stk::topology::ELEM_RANK>(radius_accessor)
                               .add_component<MASS, stk::topology::ELEM_RANK>(mass_accessor);

  // Move the spheres on the CPU
  // Note, data that is not fetched via .get is neither accessed nor modified
  double dt = 1e-5;
  sphere_data.for_each([dt](auto& sphere_view) {
    // Note: The zero is the center node connectivity ordinal.
    //  If you had, say, a BEAM_2 you could use .get<VELOCITY>(0) and .get<VELOCITY>(1)
    //  For the velocity of the left and right nodes, respectively.
    auto c = get<CENTER>(sphere_view, 0);
    auto v = get<VELOCITY>(sphere_view, 0);
    c += dt * v;
  });

  // Same but on GPU
  auto ngp_mesh = get_updated_ngp_mesh(bulk_data);
  auto ngp_sphere_data = get_updated_ngp_aggregate(sphere_data);

  EXPECT_TRUE(ngp_sphere_data.ngp_mesh().is_up_to_date());
  EXPECT_TRUE(ngp_sphere_data.ngp_mesh().get_spatial_dimension() == 3)
      << "If this works, we know that the NgpMesh was not default constructed";

  ngp_sphere_data.template sync_to_device<CENTER, COLLISION_RADIUS>();
  ngp_sphere_data.template for_each(a_non_lambda_functor{});
  ngp_sphere_data.template modify_on_device<CENTER, COLLISION_RADIUS>();
}

TEST(UnitTestAggregate, CanonicalExample) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  run_canonical_test();
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
