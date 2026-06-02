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
#include <gtest/gtest.h>

// C++ core
#include <type_traits>

// STK mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_topology/topology.hpp>

// Mundy libs
#include <mundy_geom/primitives/AABB.hpp>
#include <mundy_math/Matrix3.hpp>
#include <mundy_math/Quaternion.hpp>
#include <mundy_math/Vector.hpp>
#include <mundy_math/Vector3.hpp>
#include <mundy_mesh/Aggregate.hpp>
#include <mundy_mesh/ComponentAccess.hpp>
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/DeclareComponent.hpp>
#include <mundy_mesh/DeclarePart.hpp>
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/FieldViews.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/SharedComponent.hpp>

namespace mundy {

namespace mesh {

namespace {

struct DECLARED_COORDS;
struct DECLARED_SPEED;
struct ORDERED_FIELD_1;
struct ORDERED_FIELD_2;
struct ORDERED_FIELD_3;
struct ORDERED_FIELD_4;
struct ORDERED_FIELD_5;
struct VELOCITY;
struct ANGULAR_VELOCITY;
struct DRAG_COEFFICIENT;

struct UnknownAccessLike;
struct CANONICAL_SCALAR;
struct CANONICAL_VECTOR;
struct CANONICAL_MATRIX3;
struct CANONICAL_QUATERNION;
struct CANONICAL_AABB;

template <typename AccessLike, typename ExpectedAccess>
constexpr bool canonical_component_access_is_v =
    std::is_same_v<canonical_component_access_t<AccessLike>, ExpectedAccess>;

template <typename AccessLike, typename ExpectedFieldScalar, typename ExpectedSharedValue, bool ExpectedFixedScalars,
          unsigned ExpectedFieldScalars = 0>
constexpr bool component_access_shape_matches_v = []() {
  using shape = component_access_shape<canonical_component_access_t<AccessLike>>;
  if constexpr (ExpectedFixedScalars) {
    return std::is_same_v<typename shape::field_value_typeype, ExpectedFieldScalar> &&
           std::is_same_v<typename shape::shared_value_type, ExpectedSharedValue> && shape::has_fixed_field_scalars &&
           shape::field_scalars == ExpectedFieldScalars;
  } else {
    return std::is_same_v<typename shape::field_value_typeype, ExpectedFieldScalar> &&
           std::is_same_v<typename shape::shared_value_type, ExpectedSharedValue> && !shape::has_fixed_field_scalars;
  }
}();

static_assert(canonical_component_access_is_v<UnknownAccessLike, access::raw<UnknownAccessLike>>);
static_assert(canonical_component_access_is_v<const UnknownAccessLike&, access::raw<UnknownAccessLike>>);
static_assert(canonical_component_access_is_v<double, access::scalar<double>>);
static_assert(canonical_component_access_is_v<const double&, access::scalar<double>>);
static_assert(canonical_component_access_is_v<access::raw<const double&>, access::raw<double>>);
static_assert(canonical_component_access_is_v<access::scalar<const double&>, access::scalar<double>>);
static_assert(canonical_component_access_is_v<access::vector<const float&, 5>, access::vector<float, 5>>);
static_assert(canonical_component_access_is_v<access::vector1<int>, access::vector<int, 1>>);
static_assert(canonical_component_access_is_v<access::vector2f, access::vector<float, 2>>);
static_assert(canonical_component_access_is_v<access::vector3d, access::vector<double, 3>>);
static_assert(canonical_component_access_is_v<access::vector4i, access::vector<int, 4>>);
static_assert(canonical_component_access_is_v<access::vector6<double>, access::vector<double, 6>>);
static_assert(canonical_component_access_is_v<access::matrix<const double, 2, 3>, access::matrix<double, 2, 3>>);
static_assert(canonical_component_access_is_v<access::matrix23f, access::matrix<float, 2, 3>>);
static_assert(canonical_component_access_is_v<access::matrix36d, access::matrix<double, 3, 6>>);
static_assert(canonical_component_access_is_v<access::matrix4i, access::matrix<int, 4, 4>>);
static_assert(canonical_component_access_is_v<access::matrix6<double>, access::matrix<double, 6, 6>>);
static_assert(canonical_component_access_is_v<access::matrix3<const double>, access::matrix3<double>>);
static_assert(canonical_component_access_is_v<access::quaternion<volatile float>, access::quaternion<float>>);
static_assert(canonical_component_access_is_v<access::aabb<const double>, access::aabb<double>>);
static_assert(canonical_component_access_is_v<Vector<double, 5>, access::vector<double, 5>>);
static_assert(canonical_component_access_is_v<const Vector3d&, access::vector<double, 3>>);
static_assert(canonical_component_access_is_v<Matrix3<double>, access::matrix3<double>>);
static_assert(canonical_component_access_is_v<Matrix<double, 2, 3>, access::matrix<double, 2, 3>>);
static_assert(canonical_component_access_is_v<Matrix6i, access::matrix<int, 6, 6>>);
static_assert(canonical_component_access_is_v<const Quaternion<double>&, access::quaternion<double>>);
static_assert(canonical_component_access_is_v<AABB<double>, access::aabb<double>>);

static_assert(component_access_shape_matches_v<UnknownAccessLike, UnknownAccessLike, UnknownAccessLike, false>);
static_assert(component_access_shape_matches_v<double, double, double, true, 1>);
static_assert(component_access_shape_matches_v<Vector<double, 5>, double, Vector<double, 5>, true, 5>);
static_assert(component_access_shape_matches_v<Matrix3<double>, double, Matrix3<double>, true, 9>);
static_assert(component_access_shape_matches_v<Matrix<double, 2, 3>, double, Matrix<double, 2, 3>, true, 6>);
static_assert(component_access_shape_matches_v<access::matrix45f, float, Matrix<float, 4, 5>, true, 20>);
static_assert(component_access_shape_matches_v<Quaternion<double>, double, Quaternion<double>, true, 4>);
static_assert(component_access_shape_matches_v<AABB<double>, double, AABB<double>, true, 6>);
static_assert(std::is_same_v<
              typename impl::field_component_for<canonical_component_access_t<Vector3d>>::template type<double>,
              Vector3FieldComponent<double>>);
static_assert(std::is_same_v<
              typename impl::field_component_for<canonical_component_access_t<Matrix<double, 2, 3>>>::template type<double>,
              MatrixFieldComponent<double, 2, 3>>);
static_assert(std::is_same_v<typename impl::shared_component_for<canonical_component_access_t<Matrix3<double>>>::type,
                             SharedMatrix3Component<double>>);
static_assert(std::is_same_v<typename impl::shared_component_for<canonical_component_access_t<Matrix<double, 2, 3>>>::type,
                             SharedMatrixComponent<double, 2, 3>>);

}  // namespace

namespace {

TEST(UnitTestComponentDeclaration, CanonicalComponentAccessDrivesConcreteDeclarations) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper decl(meta_data);

  auto scalar =
      decl.rank(ELEM_RANK).name("CANONICAL_SCALAR").field<const double&>().tag<CANONICAL_SCALAR>().declare();
  auto vector = decl.rank(ELEM_RANK)
                    .name("CANONICAL_VECTOR")
                    .field<Vector<double, 5>>()
                    .tag<CANONICAL_VECTOR>()
                    .declare();
  auto matrix3 = decl.rank(ELEM_RANK)
                     .name("CANONICAL_MATRIX3")
                     .field<Matrix3<double>>()
                     .tag<CANONICAL_MATRIX3>()
                     .declare();
  auto matrix23 = decl.rank(ELEM_RANK).name("CANONICAL_MATRIX23").field<access::matrix23d>().declare();
  auto quaternion = decl.rank(ELEM_RANK)
                        .name("CANONICAL_QUATERNION")
                        .field<Quaternion<double>>()
                        .tag<CANONICAL_QUATERNION>()
                        .declare();
  auto aabb =
      decl.rank(ELEM_RANK).name("CANONICAL_AABB").field<AABB<double>>().tag<CANONICAL_AABB>().declare();

  static_assert(std::is_same_v<decltype(scalar), TaggedComponent<CANONICAL_SCALAR, ScalarFieldComponent<double>>>);
  static_assert(std::is_same_v<decltype(vector), TaggedComponent<CANONICAL_VECTOR, VectorFieldComponent<double, 5>>>);
  static_assert(std::is_same_v<decltype(matrix3), TaggedComponent<CANONICAL_MATRIX3, Matrix3FieldComponent<double>>>);
  static_assert(std::is_same_v<decltype(matrix23), MatrixFieldComponent<double, 2, 3>>);
  static_assert(
      std::is_same_v<decltype(quaternion), TaggedComponent<CANONICAL_QUATERNION, QuaternionFieldComponent<double>>>);
  static_assert(std::is_same_v<decltype(aabb), TaggedComponent<CANONICAL_AABB, AABBFieldComponent<double>>>);

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& elem_part = part_decl.name("CANONICAL_COMPONENT_ACCESS_PART")
                                   .topology(stk::topology::PARTICLE)
                                   .put_component(scalar, nullptr)
                                   .put_component(vector, nullptr)
                                   .put_component(matrix3, nullptr)
                                   .put_component(matrix23, nullptr)
                                   .put_component(quaternion, nullptr)
                                   .put_component(aabb, nullptr)
                                   .declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;
  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&elem_part});
  bulk_data.modification_end();

  EXPECT_EQ(stk::mesh::field_scalars_per_entity(scalar.component().field(), elem1), 1u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(vector.component().field(), elem1), 5u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(matrix3.component().field(), elem1), 9u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(matrix23.field(), elem1), 6u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(quaternion.component().field(), elem1), 4u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(aabb.component().field(), elem1), 6u);

  auto shared_scalar =
      ComponentDeclarationHelper().shared<const double&>(2.0).rank(ELEM_RANK).tag<CANONICAL_SCALAR>().declare();
  auto shared_matrix3 = ComponentDeclarationHelper()
                            .shared<Matrix3<double>>(Matrix3<double>{})
                            .rank(ELEM_RANK)
                            .tag<CANONICAL_MATRIX3>()
                            .declare();
  auto shared_matrix23 =
      ComponentDeclarationHelper().shared<access::matrix23d>(Matrix<double, 2, 3>{}).rank(ELEM_RANK).declare();

  static_assert(
      std::is_same_v<decltype(shared_scalar), TaggedComponent<CANONICAL_SCALAR, SharedScalarComponent<double>>>);
  static_assert(std::is_same_v<decltype(shared_matrix3),
                               TaggedComponent<CANONICAL_MATRIX3, SharedMatrix3Component<double>>>);
  static_assert(std::is_same_v<decltype(shared_matrix23), SharedMatrixComponent<double, 2, 3>>);
}

TEST(UnitTestComponentDeclaration, CanonicalUse) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using Ioss::Field::TRANSIENT;
  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper component_decl(meta_data);

  auto coords = component_decl.type<double>()
                    .role(TRANSIENT)
                    .rank(ELEM_RANK)
                    .name("DECLARED_COORDS")
                    .field<Vector3d>()
                    .tag<DECLARED_COORDS>()
                    .declare();

  auto speed = ComponentDeclarationHelper().shared<double>(2.5).rank(ELEM_RANK).tag<DECLARED_SPEED>().declare();

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& particle_part =
      part_decl.name("DECLARED_PARTICLES").topology(stk::topology::PARTICLE).put_component(coords, nullptr).declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&particle_part});
  bulk_data.modification_end();

  vector3_field_data(coords.component().field(), elem1).set(1.0, 2.0, 3.0);
  coords.modify_on_host();

  auto agg = Aggregate(bulk_data, particle_part).add_component(coords).add_component(speed);

  EXPECT_EQ(agg.get_component<DECLARED_COORDS>().component().field().mesh_meta_data_ordinal(),
            coords.component().field().mesh_meta_data_ordinal());
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(coords.component().field(), elem1), 3u);

  auto coords_view = agg.get<DECLARED_COORDS>(elem1);
  auto speed_view = agg.get<DECLARED_SPEED>(elem1);

  EXPECT_DOUBLE_EQ(coords_view[0], 1.0);
  EXPECT_DOUBLE_EQ(coords_view[1], 2.0);
  EXPECT_DOUBLE_EQ(coords_view[2], 3.0);
  EXPECT_DOUBLE_EQ(speed_view[0], 2.5);
}

TEST(UnitTestComponentDeclaration, FieldComponentDeclarationIsInvariantToFluentCallOrder) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using Ioss::Field::TRANSIENT;
  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper component_decl(meta_data);
  auto field1 = component_decl.type<double>()
                    .role(TRANSIENT)
                    .rank(ELEM_RANK)
                    .name("ORDERED_FIELD_1")
                    .field<Vector3d>()
                    .tag<ORDERED_FIELD_1>()
                    .declare();

  auto field2 = component_decl.tag<ORDERED_FIELD_2>()
                    .field<Vector3d>()
                    .name("ORDERED_FIELD_2")
                    .role(TRANSIENT)
                    .type<double>()
                    .rank(ELEM_RANK)
                    .declare();

  auto field3 = component_decl.name("ORDERED_FIELD_3")
                    .rank(ELEM_RANK)
                    .type<double>()
                    .tag<ORDERED_FIELD_3>()
                    .role(TRANSIENT)
                    .field<Vector3d>()
                    .declare();

  auto field4 = component_decl.field<Vector3d>()
                    .tag<ORDERED_FIELD_4>()
                    .rank(ELEM_RANK)
                    .role(TRANSIENT)
                    .name("ORDERED_FIELD_4")
                    .type<double>()
                    .declare();

  auto field5 = component_decl.rank(ELEM_RANK)
                    .name("ORDERED_FIELD_5")
                    .field<Vector3d>()
                    .type<double>()
                    .tag<ORDERED_FIELD_5>()
                    .role(TRANSIENT)
                    .declare();

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& particle_part = part_decl.name("ORDERED_PARTICLES")
                                       .topology(stk::topology::PARTICLE)
                                       .put_component(field1, nullptr)
                                       .put_component(field2, nullptr)
                                       .put_component(field3, nullptr)
                                       .put_component(field4, nullptr)
                                       .put_component(field5, nullptr)
                                       .declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;

  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity elem1 = bulk_data.declare_element(1, stk::mesh::PartVector{&particle_part});
  bulk_data.modification_end();

  EXPECT_EQ(field1.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field2.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field3.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field4.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(field5.component().field().entity_rank(), ELEM_RANK);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field1.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field2.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field3.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field4.component().field(), elem1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(field5.component().field(), elem1), 3u);

  vector3_field_data(field1.component().field(), elem1).set(1.0, 2.0, 3.0);
  vector3_field_data(field2.component().field(), elem1).set(4.0, 5.0, 6.0);
  vector3_field_data(field3.component().field(), elem1).set(7.0, 8.0, 9.0);
  vector3_field_data(field4.component().field(), elem1).set(10.0, 11.0, 12.0);
  vector3_field_data(field5.component().field(), elem1).set(13.0, 14.0, 15.0);

  auto field1_view = field1(elem1);
  auto field2_view = field2(elem1);
  auto field3_view = field3(elem1);
  auto field4_view = field4(elem1);
  auto field5_view = field5(elem1);

  EXPECT_DOUBLE_EQ(field1_view[0], 1.0);
  EXPECT_DOUBLE_EQ(field1_view[1], 2.0);
  EXPECT_DOUBLE_EQ(field1_view[2], 3.0);

  EXPECT_DOUBLE_EQ(field2_view[0], 4.0);
  EXPECT_DOUBLE_EQ(field2_view[1], 5.0);
  EXPECT_DOUBLE_EQ(field2_view[2], 6.0);

  EXPECT_DOUBLE_EQ(field3_view[0], 7.0);
  EXPECT_DOUBLE_EQ(field3_view[1], 8.0);
  EXPECT_DOUBLE_EQ(field3_view[2], 9.0);

  EXPECT_DOUBLE_EQ(field4_view[0], 10.0);
  EXPECT_DOUBLE_EQ(field4_view[1], 11.0);
  EXPECT_DOUBLE_EQ(field4_view[2], 12.0);

  EXPECT_DOUBLE_EQ(field5_view[0], 13.0);
  EXPECT_DOUBLE_EQ(field5_view[1], 14.0);
  EXPECT_DOUBLE_EQ(field5_view[2], 15.0);
}

TEST(UnitTestComponentDeclaration, SharedComponentDeclarationViaExplicitAccess) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  ComponentDeclarationHelper component_decl;
  auto speed = component_decl.shared<double>(3.5).tag<DECLARED_SPEED>().rank(stk::topology::ELEM_RANK).declare();

  stk::mesh::Entity entity = stk::mesh::Entity();
  auto speed_view = speed(entity);
  EXPECT_DOUBLE_EQ(speed_view[0], 3.5);
}

TEST(UnitTestComponentDeclaration, FieldComponentDeclarationRejectsOutputTypeMismatch) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper component_decl(meta_data);

  EXPECT_THROW((void)component_decl.type<double>()
                   .output_type(stk::io::FieldOutputType::SCALAR)
                   .rank(stk::topology::ELEM_RANK)
                   .name("BAD_VECTOR_OUTPUT")
                   .field<Vector3d>()
                   .declare(),
               std::invalid_argument);
}

TEST(UnitTestComponentDeclaration, ExpectedFailureModes) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper component_decl(meta_data);
  ComponentDeclarationHelper shared_component_decl;

  // Missing name: rank and backend set, name omitted.
  EXPECT_THROW((void)component_decl.type<double>().rank(ELEM_RANK).field<double>().declare(), std::logic_error);
  // Missing rank: name and backend set, rank omitted.
  EXPECT_THROW((void)component_decl.type<double>().name("MISSING_FIELD_COMPONENT_RANK").field<double>().declare(),
               std::logic_error);
  // Missing rank on shared: calling .declare() without .rank() on a shared builder throws.
  EXPECT_THROW((void)shared_component_decl.shared<double>(3.5).tag<DECLARED_SPEED>().declare(), std::logic_error);
  // NOTE: calling .declare() without first calling .field<A>() or .shared<A>(source) is a compile error.
  // TaggedFieldDeclarationHelperT (.tag() without backend selection) has no .declare() member.
}

// Tests for the unified ComponentDeclarationHelper(meta_data) constructor.

TEST(UnitTestComponentDeclaration, ComponentDeclarationHelperFieldPath) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using Ioss::Field::TRANSIENT;
  using stk::topology::NODE_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper decl(meta_data);

  // Name-first field path: rank + name set before backend selection.
  auto vel = decl.role(TRANSIENT).rank(NODE_RANK).name("VELOCITY").field<Vector3d>().tag<VELOCITY>().declare();

  // Backend-first field path: field access chosen first, remaining metadata set after.
  auto ang = decl.field<Vector3d>()
                 .tag<ANGULAR_VELOCITY>()
                 .role(TRANSIENT)
                 .rank(NODE_RANK)
                 .name("ANGULAR_VELOCITY")
                 .declare();

  PartDeclarationHelper part_decl(meta_data);
  stk::mesh::Part& node_part = part_decl.name("SPHERE_NODES")
                                   .rank(NODE_RANK)
                                   .put_component(vel, nullptr)
                                   .put_component(ang, nullptr)
                                   .declare();

  auto bulk_data_ptr = builder.create(meta_data_ptr);
  stk::mesh::BulkData& bulk_data = *bulk_data_ptr;
  meta_data.commit();
  bulk_data.modification_begin();
  stk::mesh::Entity node1 = bulk_data.declare_node(1, stk::mesh::PartVector{&node_part});
  bulk_data.modification_end();

  EXPECT_EQ(vel.component().field().entity_rank(), NODE_RANK);
  EXPECT_EQ(ang.component().field().entity_rank(), NODE_RANK);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(vel.component().field(), node1), 3u);
  EXPECT_EQ(stk::mesh::field_scalars_per_entity(ang.component().field(), node1), 3u);
}

TEST(UnitTestComponentDeclaration, ComponentDeclarationHelperSharedPath) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using stk::topology::ELEM_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper decl(meta_data);

  // Backend-first shared path: .shared<A>(v).rank(r).name(n).tag<T>().declare()
  auto drag =
      decl.shared<double>(0.47).rank(ELEM_RANK).name("drag_coeff").tag<DRAG_COEFFICIENT>().declare();

  // Name-first shared path: .rank(r).name(n).shared<A>(v).declare()
  auto radius = decl.rank(ELEM_RANK).name("sphere_radius").shared<double>(1.5).declare();

  stk::mesh::Entity entity = stk::mesh::Entity();
  EXPECT_DOUBLE_EQ(drag(entity)[0], 0.47);
  EXPECT_DOUBLE_EQ(radius(entity)[0], 1.5);

  // Verify name and rank metadata is preserved through the tag wrapper.
  EXPECT_EQ(drag.component().name(), "drag_coeff");
  EXPECT_EQ(drag.component().entity_rank(), ELEM_RANK);

  // Verify name and rank on untagged shared component.
  EXPECT_EQ(radius.name(), "sphere_radius");
  EXPECT_EQ(radius.entity_rank(), ELEM_RANK);
}

TEST(UnitTestComponentDeclaration, CompatibleNonDefaultOutputTypeIsAcceptedForFixedSizeAccess) {
  if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
    GTEST_SKIP();
  }

  using stk::topology::NODE_RANK;

  stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
  builder.set_spatial_dimension(3);
  builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

  auto meta_data_ptr = builder.create_meta_data();
  stk::mesh::MetaData& meta_data = *meta_data_ptr;
  meta_data.use_simple_fields();

  ComponentDeclarationHelper decl(meta_data);

  // FULL_TENSOR_12 has 3 scalars — same as Vector3d — so our scalar-count check accepts it even though
  // it is not the default output type for this access shape. SCALAR (1 scalar) must still be rejected.
  EXPECT_NO_THROW((void)decl.rank(NODE_RANK)
                             .name("TENSOR12_OUTPUT_FIELD")
                             .field<Vector3d>()
                             .output_type(stk::io::FieldOutputType::FULL_TENSOR_12)
                             .declare());

  EXPECT_THROW((void)decl.rank(NODE_RANK)
                          .name("BAD_SCALAR_OUTPUT_FIELD")
                          .field<Vector3d>()
                          .output_type(stk::io::FieldOutputType::SCALAR)
                          .declare(),
               std::invalid_argument);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
