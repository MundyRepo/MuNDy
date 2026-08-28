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
#include <memory>
#include <stdexcept>
#include <type_traits>

// Kokkos
#include <Kokkos_Core.hpp>

// STK mesh
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

// Mundy libs
#include <mundy_math/Vector3.hpp>
#include <mundy_mesh/BulkData.hpp>
#include <mundy_mesh/Component.hpp>
#include <mundy_mesh/FieldComponent.hpp>
#include <mundy_mesh/FieldViews.hpp>
#include <mundy_mesh/MeshBuilder.hpp>
#include <mundy_mesh/MetaData.hpp>
#include <mundy_mesh/SharedComponent.hpp>

namespace mundy {

namespace mesh {

namespace {

class UnitTestComponentFixture : public ::testing::Test {
 protected:
  using DoubleField = stk::mesh::Field<double>;

  void SetUp() override {
    if (stk::parallel_machine_size(MPI_COMM_WORLD) != 1) {
      GTEST_SKIP();
    }

    stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
    builder.set_spatial_dimension(3);
    builder.set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});

    meta_data_ptr_ = builder.create_meta_data();
    stk::mesh::MetaData& meta_data = *meta_data_ptr_;
    meta_data.use_simple_fields();
    bulk_data_ptr_ = builder.create(meta_data_ptr_);

    scalar_field_ptr_ = &meta_data.declare_field<double>(stk::topology::NODE_RANK, "SCALAR");
    scalar_alt_field_ptr_ = &meta_data.declare_field<double>(stk::topology::NODE_RANK, "SCALAR_ALT");
    vector3_field_ptr_ = &meta_data.declare_field<double>(stk::topology::NODE_RANK, "VECTOR3");
    matrix3_field_ptr_ = &meta_data.declare_field<double>(stk::topology::NODE_RANK, "MATRIX3");
    quaternion_field_ptr_ = &meta_data.declare_field<double>(stk::topology::NODE_RANK, "QUATERNION");
    aabb_field_ptr_ = &meta_data.declare_field<double>(stk::topology::NODE_RANK, "AABB");

    stk::mesh::put_field_on_mesh(*scalar_field_ptr_, meta_data.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*scalar_alt_field_ptr_, meta_data.universal_part(), 1, nullptr);
    stk::mesh::put_field_on_mesh(*vector3_field_ptr_, meta_data.universal_part(), 3, nullptr);
    stk::mesh::put_field_on_mesh(*matrix3_field_ptr_, meta_data.universal_part(), 9, nullptr);
    stk::mesh::put_field_on_mesh(*quaternion_field_ptr_, meta_data.universal_part(), 4, nullptr);
    stk::mesh::put_field_on_mesh(*aabb_field_ptr_, meta_data.universal_part(), 6, nullptr);
    meta_data.commit();

    stk::mesh::BulkData& bulk_data = *bulk_data_ptr_;
    bulk_data.modification_begin();
    node1_ = bulk_data.declare_node(1);
    node2_ = bulk_data.declare_node(2);
    bulk_data.modification_end();

    set_node_values(node1_, 1.0);
    set_node_values(node2_, 101.0);
    mark_all_fields_modified_on_host();
  }

  void TearDown() override {
    bulk_data_ptr_.reset();
    meta_data_ptr_.reset();
  }

  stk::mesh::BulkData& bulk_data() {
    return *bulk_data_ptr_;
  }

  void set_node_values(stk::mesh::Entity node, double base) {
    scalar_field_data(*scalar_field_ptr_, node).set(base + 0.0);
    scalar_field_data(*scalar_alt_field_ptr_, node).set(base + 200.0);
    vector3_field_data(*vector3_field_ptr_, node).set(base + 10.0, base + 11.0, base + 12.0);
    matrix3_field_data(*matrix3_field_ptr_, node)
        .set(base + 20.0, base + 21.0, base + 22.0, base + 23.0, base + 24.0, base + 25.0, base + 26.0, base + 27.0,
             base + 28.0);
    quaternion_field_data(*quaternion_field_ptr_, node).set(base + 30.0, base + 31.0, base + 32.0, base + 33.0);

    auto aabb = aabb_field_data(*aabb_field_ptr_, node);
    aabb.x_min() = base + 40.0;
    aabb.y_min() = base + 41.0;
    aabb.z_min() = base + 42.0;
    aabb.x_max() = base + 43.0;
    aabb.y_max() = base + 44.0;
    aabb.z_max() = base + 45.0;
  }

  void mark_all_fields_modified_on_host() {
    scalar_field_ptr_->modify_on_host();
    scalar_alt_field_ptr_->modify_on_host();
    vector3_field_ptr_->modify_on_host();
    matrix3_field_ptr_->modify_on_host();
    quaternion_field_ptr_->modify_on_host();
    aabb_field_ptr_->modify_on_host();
  }

  std::shared_ptr<stk::mesh::MetaData> meta_data_ptr_;
  std::shared_ptr<stk::mesh::BulkData> bulk_data_ptr_;
  DoubleField* scalar_field_ptr_ = nullptr;
  DoubleField* scalar_alt_field_ptr_ = nullptr;
  DoubleField* vector3_field_ptr_ = nullptr;
  DoubleField* matrix3_field_ptr_ = nullptr;
  DoubleField* quaternion_field_ptr_ = nullptr;
  DoubleField* aabb_field_ptr_ = nullptr;
  stk::mesh::Entity node1_ = stk::mesh::Entity();
  stk::mesh::Entity node2_ = stk::mesh::Entity();
};

template <typename ComponentType>
constexpr bool has_default_copy_move_assign_v =
    std::is_default_constructible_v<ComponentType> && std::is_copy_constructible_v<ComponentType> &&
    std::is_move_constructible_v<ComponentType> && std::is_copy_assignable_v<ComponentType> &&
    std::is_move_assignable_v<ComponentType>;

using TaggedScalarFieldComponent = TaggedComponent<POSITION, ScalarFieldComponent<double>>;
using TaggedScalarSharedComponent = TaggedComponent<POSITION, SharedScalarComponent<double>>;
using NgpDoubleField = stk::mesh::NgpField<double>;

static_assert(has_default_copy_move_assign_v<FieldComponentBase>);
static_assert(has_default_copy_move_assign_v<impl::FieldComponent<double, impl::FieldDataAccessPolicy>>);
static_assert(has_default_copy_move_assign_v<FieldComponent<double>>);
static_assert(has_default_copy_move_assign_v<ScalarFieldComponent<double>>);
static_assert(has_default_copy_move_assign_v<Vector3FieldComponent<double>>);
static_assert(has_default_copy_move_assign_v<Matrix3FieldComponent<double>>);
static_assert(has_default_copy_move_assign_v<QuaternionFieldComponent<double>>);
static_assert(has_default_copy_move_assign_v<AABBFieldComponent<double>>);
static_assert(has_default_copy_move_assign_v<NgpFieldComponentBase>);
static_assert(has_default_copy_move_assign_v<impl::NgpFieldComponent<NgpDoubleField, impl::FieldDataAccessPolicy>>);
static_assert(has_default_copy_move_assign_v<NgpFieldComponent<NgpDoubleField>>);
static_assert(has_default_copy_move_assign_v<NgpScalarFieldComponent<NgpDoubleField>>);
static_assert(has_default_copy_move_assign_v<NgpVector3FieldComponent<NgpDoubleField>>);
static_assert(has_default_copy_move_assign_v<NgpMatrix3FieldComponent<NgpDoubleField>>);
static_assert(has_default_copy_move_assign_v<NgpQuaternionFieldComponent<NgpDoubleField>>);
static_assert(has_default_copy_move_assign_v<NgpAABBFieldComponent<NgpDoubleField>>);
static_assert(has_default_copy_move_assign_v<SharedComponent<double>>);
static_assert(has_default_copy_move_assign_v<SharedScalarComponent<double>>);
static_assert(has_default_copy_move_assign_v<SharedVector3Component<double>>);
static_assert(has_default_copy_move_assign_v<SharedMatrix3Component<double>>);
static_assert(has_default_copy_move_assign_v<SharedQuaternionComponent<double>>);
static_assert(has_default_copy_move_assign_v<SharedAABBComponent<double>>);
static_assert(has_default_copy_move_assign_v<NgpSharedComponent<double, stk::ngp::MemSpace>>);
static_assert(has_default_copy_move_assign_v<NgpSharedScalarComponent<double>>);
static_assert(has_default_copy_move_assign_v<NgpSharedVector3Component<double>>);
static_assert(has_default_copy_move_assign_v<NgpSharedMatrix3Component<double>>);
static_assert(has_default_copy_move_assign_v<NgpSharedQuaternionComponent<double>>);
static_assert(has_default_copy_move_assign_v<NgpSharedAABBComponent<double>>);
static_assert(has_default_copy_move_assign_v<TaggedScalarFieldComponent>);
static_assert(has_default_copy_move_assign_v<TaggedScalarSharedComponent>);
static_assert(has_default_copy_move_assign_v<NgpTaggedComponent<POSITION, NgpSharedScalarComponent<double>>>);

TEST_F(UnitTestComponentFixture, FieldComponentExposeTypedViewsAndMutations) {
  FieldComponent<double> raw_accessor(*quaternion_field_ptr_);
  ScalarFieldComponent<double> scalar_accessor(*scalar_field_ptr_);
  Vector3FieldComponent<double> vector3_accessor(*vector3_field_ptr_);
  Matrix3FieldComponent<double> matrix3_accessor(*matrix3_field_ptr_);
  QuaternionFieldComponent<double> quaternion_accessor(*quaternion_field_ptr_);
  AABBFieldComponent<double> aabb_accessor(*aabb_field_ptr_);

  auto scalar1 = scalar_accessor(node1_);
  auto scalar2 = scalar_accessor(node2_);
  EXPECT_DOUBLE_EQ(scalar1[0], 1.0);
  EXPECT_DOUBLE_EQ(scalar2[0], 101.0);
  scalar1[0] = -2.5;
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*scalar_field_ptr_, node1_)[0], -2.5);
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*scalar_field_ptr_, node2_)[0], 101.0);

  auto vector1 = vector3_accessor(node1_);
  EXPECT_DOUBLE_EQ(vector1[0], 11.0);
  EXPECT_DOUBLE_EQ(vector1[1], 12.0);
  EXPECT_DOUBLE_EQ(vector1[2], 13.0);
  vector1.set(-1.0, -2.0, -3.0);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node1_)[0], -1.0);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node1_)[1], -2.0);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node1_)[2], -3.0);

  auto matrix1 = matrix3_accessor(node1_);
  EXPECT_DOUBLE_EQ(matrix1(0, 0), 21.0);
  EXPECT_DOUBLE_EQ(matrix1(2, 2), 29.0);
  matrix1(1, 2) = 123.0;
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*matrix3_field_ptr_, node1_)[5], 123.0);

  auto raw_quaternion = raw_accessor(node1_);
  EXPECT_DOUBLE_EQ(raw_quaternion[0], 32.0);
  EXPECT_DOUBLE_EQ(raw_quaternion[1], 33.0);
  EXPECT_DOUBLE_EQ(raw_quaternion[2], 34.0);
  EXPECT_DOUBLE_EQ(raw_quaternion[3], 31.0);
  raw_quaternion[3] = 44.0;
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*quaternion_field_ptr_, node1_)[3], 44.0);

  auto quaternion1 = quaternion_accessor(node1_);
  EXPECT_DOUBLE_EQ(quaternion1.w(), 44.0);
  EXPECT_DOUBLE_EQ(quaternion1.z(), 34.0);
  quaternion1.set(0.1, 0.2, 0.3, 0.4);
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*quaternion_field_ptr_, node1_)[0], 0.2);
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*quaternion_field_ptr_, node1_)[1], 0.3);
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*quaternion_field_ptr_, node1_)[2], 0.4);
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*quaternion_field_ptr_, node1_)[3], 0.1);

  auto aabb1 = aabb_accessor(node1_);
  EXPECT_DOUBLE_EQ(aabb1.x_min(), 41.0);
  EXPECT_DOUBLE_EQ(aabb1.z_max(), 46.0);
  aabb1.y_max() = 77.0;
  EXPECT_DOUBLE_EQ(stk::mesh::field_data(*aabb_field_ptr_, node1_)[4], 77.0);
}

TEST_F(UnitTestComponentFixture, ShallowCopyAssignment) {
  ScalarFieldComponent<double> lhs_accessor(*scalar_field_ptr_);
  ScalarFieldComponent<double> rhs_accessor(*scalar_alt_field_ptr_);

  lhs_accessor = rhs_accessor;

  EXPECT_EQ(&lhs_accessor.field(), scalar_alt_field_ptr_);
  EXPECT_EQ(&rhs_accessor.field(), scalar_alt_field_ptr_);
  EXPECT_DOUBLE_EQ(lhs_accessor(node1_)[0], 201.0);
  EXPECT_DOUBLE_EQ(lhs_accessor(node2_)[0], 301.0);

  lhs_accessor(node1_)[0] = -17.25;

  EXPECT_DOUBLE_EQ(scalar_field_data(*scalar_alt_field_ptr_, node1_)[0], -17.25);
  EXPECT_DOUBLE_EQ(scalar_field_data(*scalar_field_ptr_, node1_)[0], 1.0);
}

void mutate_components_on_device(
    const NgpScalarFieldComponent<stk::mesh::NgpField<double>>& ngp_scalar_accessor,
    const NgpVector3FieldComponent<stk::mesh::NgpField<double>>& ngp_vector3_accessor,
    const NgpMatrix3FieldComponent<stk::mesh::NgpField<double>>& ngp_matrix3_accessor,
    const NgpQuaternionFieldComponent<stk::mesh::NgpField<double>>& ngp_quaternion_accessor,
    const NgpAABBFieldComponent<stk::mesh::NgpField<double>>& ngp_aabb_accessor, stk::mesh::FastMeshIndex node1_index) {
  Kokkos::parallel_for(
      "mutate_components_on_device", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        ngp_scalar_accessor(node1_index)[0] += 2.5;

        auto vector3 = ngp_vector3_accessor(node1_index);
        vector3.set(7.0, 8.0, 9.0);

        auto matrix3 = ngp_matrix3_accessor(node1_index);
        matrix3(0, 1) = 222.0;
        matrix3(2, 2) = 333.0;

        auto quaternion = ngp_quaternion_accessor(node1_index);
        quaternion.set(4.0, 5.0, 6.0, 7.0);

        auto aabb = ngp_aabb_accessor(node1_index);
        aabb.x_min() = -1.0;
        aabb.z_max() = 99.0;
      });
}

TEST_F(UnitTestComponentFixture, NgpFieldComponentRoundTripDeviceMutations) {
  ScalarFieldComponent<double> scalar_accessor(*scalar_field_ptr_);
  Vector3FieldComponent<double> vector3_accessor(*vector3_field_ptr_);
  Matrix3FieldComponent<double> matrix3_accessor(*matrix3_field_ptr_);
  QuaternionFieldComponent<double> quaternion_accessor(*quaternion_field_ptr_);
  AABBFieldComponent<double> aabb_accessor(*aabb_field_ptr_);

  auto ngp_scalar_accessor = get_updated_ngp_component(scalar_accessor);
  auto ngp_vector3_accessor = get_updated_ngp_component(vector3_accessor);
  auto ngp_matrix3_accessor = get_updated_ngp_component(matrix3_accessor);
  auto ngp_quaternion_accessor = get_updated_ngp_component(quaternion_accessor);
  auto ngp_aabb_accessor = get_updated_ngp_component(aabb_accessor);

  static_assert(std::is_same_v<decltype(ngp_scalar_accessor), NgpScalarFieldComponent<stk::mesh::NgpField<double>>>);
  static_assert(std::is_same_v<decltype(ngp_vector3_accessor), NgpVector3FieldComponent<stk::mesh::NgpField<double>>>);
  static_assert(std::is_same_v<decltype(ngp_matrix3_accessor), NgpMatrix3FieldComponent<stk::mesh::NgpField<double>>>);
  static_assert(
      std::is_same_v<decltype(ngp_quaternion_accessor), NgpQuaternionFieldComponent<stk::mesh::NgpField<double>>>);
  static_assert(std::is_same_v<decltype(ngp_aabb_accessor), NgpAABBFieldComponent<stk::mesh::NgpField<double>>>);

  ngp_scalar_accessor.sync_to_device();
  ngp_vector3_accessor.sync_to_device();
  ngp_matrix3_accessor.sync_to_device();
  ngp_quaternion_accessor.sync_to_device();
  ngp_aabb_accessor.sync_to_device();

  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
  stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(node1_);

  // KOKKOS_LAMBDA cannot be called in a GTEST test body due to CUDA's rule about:
  // "The enclosing parent function ("TestBody") for an extended __host__ __device__ lambda cannot have private or
  // protected access within its class"
  mutate_components_on_device(ngp_scalar_accessor, ngp_vector3_accessor, ngp_matrix3_accessor, ngp_quaternion_accessor,
                              ngp_aabb_accessor, node1_index);
  Kokkos::fence();

  ngp_scalar_accessor.modify_on_device();
  ngp_vector3_accessor.modify_on_device();
  ngp_matrix3_accessor.modify_on_device();
  ngp_quaternion_accessor.modify_on_device();
  ngp_aabb_accessor.modify_on_device();

  scalar_accessor.sync_to_host();
  vector3_accessor.sync_to_host();
  matrix3_accessor.sync_to_host();
  quaternion_accessor.sync_to_host();
  aabb_accessor.sync_to_host();

  EXPECT_DOUBLE_EQ(scalar_field_data(*scalar_field_ptr_, node1_)[0], 3.5);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node1_)[0], 7.0);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node1_)[1], 8.0);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node1_)[2], 9.0);
  EXPECT_DOUBLE_EQ(matrix3_field_data(*matrix3_field_ptr_, node1_)(0, 1), 222.0);
  EXPECT_DOUBLE_EQ(matrix3_field_data(*matrix3_field_ptr_, node1_)(2, 2), 333.0);
  EXPECT_DOUBLE_EQ(quaternion_field_data(*quaternion_field_ptr_, node1_).w(), 4.0);
  EXPECT_DOUBLE_EQ(quaternion_field_data(*quaternion_field_ptr_, node1_).z(), 7.0);
  EXPECT_DOUBLE_EQ(aabb_field_data(*aabb_field_ptr_, node1_).x_min(), -1.0);
  EXPECT_DOUBLE_EQ(aabb_field_data(*aabb_field_ptr_, node1_).z_max(), 99.0);

  EXPECT_DOUBLE_EQ(scalar_field_data(*scalar_field_ptr_, node2_)[0], 101.0);
  EXPECT_DOUBLE_EQ(vector3_field_data(*vector3_field_ptr_, node2_)[0], 111.0);
  EXPECT_DOUBLE_EQ(matrix3_field_data(*matrix3_field_ptr_, node2_)(0, 1), 122.0);
  EXPECT_DOUBLE_EQ(quaternion_field_data(*quaternion_field_ptr_, node2_).w(), 131.0);
  EXPECT_DOUBLE_EQ(aabb_field_data(*aabb_field_ptr_, node2_).z_max(), 146.0);
}

static_assert(
    std::is_same_v<decltype(SharedComponent(Kokkos::View<double*, Kokkos::HostSpace>("", 1))), SharedComponent<double>>,
    "SharedComponent CTAD failed for host view.");

static_assert(
    std::is_same_v<decltype(SharedComponent(Kokkos::View<double*, Kokkos::HostSpace>("", 1)))::view_t, double&>,
    "SharedComponent type CTAD failed for host view.");

static_assert(std::is_same_v<decltype(SharedScalarComponent(Kokkos::View<double*, Kokkos::HostSpace>("", 1))),
                             SharedScalarComponent<double>>,
              "SharedScalarComponent CTAD failed for host view.");

static_assert(std::is_same_v<decltype(SharedScalarComponent(Kokkos::View<double*, Kokkos::HostSpace>("", 1)))::view_t,
                             decltype(get_scalar<double>(std::declval<double*>()))>,
              "SharedScalarComponent type CTAD failed for host view.");

TEST_F(UnitTestComponentFixture, SharedComponentSupportsOwnedAndAliasedConstruction) {
  Kokkos::View<double*, Kokkos::HostSpace> managed_view("managed_shared_value", 1);
  managed_view(0) = 0.5;

  SharedScalarComponent<double> managed_component(managed_view);
  EXPECT_EQ(&managed_component.shared_value(), managed_view.data());
  managed_component(node1_)[0] = 1.25;
  EXPECT_DOUBLE_EQ(managed_view(0), 1.25);

  using unmanaged_host_view_t = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  unmanaged_host_view_t unmanaged_view(managed_view.data(), 1);
  auto unmanaged_component = SharedScalarComponent(unmanaged_view);
  EXPECT_EQ(&unmanaged_component.shared_value(), managed_view.data());
  unmanaged_component(node2_)[0] = 2.25;
  EXPECT_DOUBLE_EQ(managed_view(0), 2.25);

  double raw_value = 3.5;
  auto owned_component = SharedScalarComponent(raw_value);
  EXPECT_NE(&owned_component.shared_value(), &raw_value);
  EXPECT_DOUBLE_EQ(owned_component(node1_)[0], 3.5);
  raw_value = -1.0;
  EXPECT_DOUBLE_EQ(owned_component(node2_)[0], 3.5);
  owned_component(node1_)[0] = 4.5;
  EXPECT_DOUBLE_EQ(owned_component.shared_value(), 4.5);
  EXPECT_DOUBLE_EQ(raw_value, -1.0);
}

TEST_F(UnitTestComponentFixture, SharedComponentRejectsBadViewsAndEnforcesSyncProtocol) {
  Kokkos::View<double*, Kokkos::HostSpace> bad_view("bad_shared_value", 2);
  EXPECT_THROW((void)SharedScalarComponent(bad_view), std::invalid_argument);

  SharedScalarComponent<double> shared_component(2.0);
  auto ngp_shared_component = get_updated_ngp_component(shared_component);

  shared_component.modify_on_host();
  EXPECT_THROW(shared_component.modify_on_device(), std::invalid_argument);

  shared_component.clear_host_sync_state();
  ngp_shared_component.modify_on_device();
  EXPECT_THROW(shared_component.modify_on_host(), std::invalid_argument);
}

TEST_F(UnitTestComponentFixture, SharedComponentAreShallowCopiesAndCacheNgpInstances) {
  Kokkos::View<double*, Kokkos::HostSpace> managed_view("cached_shared_value", 1);
  managed_view(0) = 0.75;

  SharedScalarComponent<double> shared_component(managed_view);
  auto copied_component = shared_component;

  copied_component(node2_)[0] = 1.75;
  EXPECT_DOUBLE_EQ(shared_component(node1_)[0], 1.75);
  EXPECT_EQ(&shared_component.shared_value(), &copied_component.shared_value());

  auto ngp_shared_component0 = get_updated_ngp_component(shared_component);
  auto ngp_shared_component1 = get_updated_ngp_component(copied_component);
  EXPECT_EQ(ngp_shared_component0.ngp_view().data(), ngp_shared_component1.ngp_view().data());

  if constexpr (Kokkos::SpaceAccessibility<stk::ngp::MemSpace, Kokkos::HostSpace>::accessible) {
    EXPECT_EQ(ngp_shared_component0.ngp_view().data(), managed_view.data());
  }
}

TEST_F(UnitTestComponentFixture, SharedComponentAssignmentRebindsToRhsState) {
  Kokkos::View<double*, Kokkos::HostSpace> rhs_view("assigned_shared_value", 1);
  rhs_view(0) = 2.5;

  SharedScalarComponent<double> lhs_component(1.0);
  SharedScalarComponent<double> rhs_component(rhs_view);

  lhs_component = rhs_component;

  EXPECT_EQ(&lhs_component.shared_value(), &rhs_component.shared_value());
  EXPECT_EQ(&lhs_component.shared_value(), rhs_view.data());
  EXPECT_DOUBLE_EQ(lhs_component(node1_)[0], 2.5);

  rhs_component(node2_)[0] = 4.75;

  EXPECT_DOUBLE_EQ(lhs_component(node1_)[0], 4.75);
  EXPECT_DOUBLE_EQ(rhs_view(0), 4.75);
}

void mutate_shared_component_on_device(const NgpSharedScalarComponent<double>& ngp_shared_component,
                                       stk::mesh::FastMeshIndex node1_index) {
  Kokkos::parallel_for(
      "mutate_shared_component_on_device", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(int) { ngp_shared_component(node1_index) += 0.75; });
}

TEST_F(UnitTestComponentFixture, NgpSharedComponentRoundTripHostAndDeviceMutations) {
  Kokkos::View<double*, Kokkos::HostSpace> managed_view("roundtrip_shared_value", 1);
  managed_view(0) = 0.5;

  SharedScalarComponent<double> shared_component(managed_view);
  auto ngp_shared_component = get_updated_ngp_component(shared_component);

  shared_component(node1_)[0] = 1.25;
  shared_component.modify_on_host();
  shared_component.sync_to_device();

  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data());
  stk::mesh::FastMeshIndex node1_index = ngp_mesh.fast_mesh_index(node1_);

  mutate_shared_component_on_device(ngp_shared_component, node1_index);
  Kokkos::fence();

  ngp_shared_component.modify_on_device();
  shared_component.sync_to_host();

  EXPECT_DOUBLE_EQ(shared_component(node2_)[0], 2.0);
  EXPECT_DOUBLE_EQ(managed_view(0), 2.0);
}

}  // namespace

}  // namespace mesh

}  // namespace mundy
