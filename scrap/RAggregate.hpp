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

#ifndef MUNDY_MESH_RAGGREGATES_HPP_
#define MUNDY_MESH_RAGGREGATES_HPP_

// C++ core
#include <tuple>
#include <type_traits>  // for std::conditional_t, std::false_type, std::true_type

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize, Kokkos::Timer

// Trilinos
#include <Trilinos_version.h>  // for TRILINOS_MAJOR_MINOR_VERSION

// STK mesh
#include <stk_mesh/base/Entity.hpp>       // for stk::mesh::Entity
#include <stk_mesh/base/GetNgpField.hpp>  // for stk::mesh::get_updated_ngp_field
#include <stk_mesh/base/NgpField.hpp>     // for stk::mesh::NgpField
#include <stk_mesh/base/NgpMesh.hpp>      // for stk::mesh::NgpMesh
#include <stk_topology/topology.hpp>      // for stk::topology::topology_t

// Mundy
#include <mundy_core/throw_assert.hpp>   // for MUNDY_THROW_ASSERT
#include <mundy_core/tuple.hpp>          // for mundy::core::tuple
#include <mundy_mesh/BulkData.hpp>       // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>     // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/fmt_stk_types.hpp>  // for STK-compatible fmt::format

namespace mundy {

namespace mesh {

/// \brief Runtime aggregates
// Runtime aggregates perform pack and unpack. 

struct FieldAccessor {
 std::string name = "FieldAccessor";
 };
struct ScalarAccessor {
 std::string name = "ScalarAccessor";
 };
struct AABBAccessor {
 std::string name = "AABBAccessor";
 };
template<size_t N> 
struct VectorAccessor {
 std::string name = "VectorAccessorN";
 };


namespace accessor_t {
  struct FIELD;
  
  struct SCALAR;
  
  template<size_t N> 
  struct VECTOR;
  
  using VECTOR3 = VECTOR<3>;
  
  template<size_t N, size_t M> 
  struct MATRIX;
  
  using MATRIX33 = MATRIX<3, 3>;

  struct AABB;
}

template <typename T>
struct to_accessor_type;

template <typename T>
using to_accessor_type_t = to_accessor_type<T>::type;

template <>
struct to_accessor_type<accessor_t::FIELD> {
  using type = FieldAccessor;
};
template <>
struct to_accessor_type<accessor_t::SCALAR> {
  using type = ScalarAccessor;
};
template<size_t N> 
struct to_accessor_type<accessor_t::VECTOR<N>> {
  using type = VectorAccessor<N>;
};
template <>
struct to_accessor_type<accessor_t::AABB> {
  using type = AABBAccessor;
};

// ragg.get_accessor<accessor_t::ENTITY_FIELD_DATA<T>>(rank, name) -> FieldAccessor
// ragg.get_accessor<accessor_t::SCALAR<T>>(rank, name) -> ScalarAccessor
// ragg.get_accessor<accessor_t::AABB<T>>(rank, name) -> AABBAccessor
// ragg.get_accessor<accessor_t::USER_TYPE, T>(rank, name) -> TheirCustomAccessor


// every accessor has at most 4 types:
// - shared single value
// - singe value per part
// - field
// - one field per part

// auto agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
//             .add_component<CENTER>(center_accessor)
//             .add_component<COLLISION_RADIUS>(radius_accessor);


RuntimeAggregate ragg = make_runtime_aggregate(bulk_data, selector, stk::topology::PARTICLE)
    .add_accessor(NODE_RANK, "OUR_CENTER", center_accessor)
    .add_accessor(NODE_RANK, "OUR_RADIUS", radius_accessor);

std::map<std::string, std::string> rename_map{
  {"CENTER", "OUR_CENTER"},
  {"RADIUS", "OUR_RADIUS"}
};

// Option 1: Compile-time aggregate with no concept of connectivity.
auto agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
        .add_accessor<CENTER>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]))
        .add_accessor<RADIUS>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["RADIUS"]));
stk::mesh::for_each_entity_run(ngp_mesh, 
  KOKKOS_LAMBDA(stk::mesh::FastMeshIndex sphere_index)
    stk::mesh::FastMeshIndex center_node_index = ngp_mesh.fast_mesh_index(ngp_mesh.nodes(sphere_index)[0]);
    auto center = agg.get<CENTER>(center_node_index);
    auto radius = agg.get<RADIUS>(sphere_index);
    center += radius[0];
  );

// Option 2: Compile-time aggregate with connectivity helper
auto agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
        .add_accessor<CENTER>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]))
        .add_accessor<RADIUS>(ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["RADIUS"]));
agg.for_each(
  KOKKOS_LAMBDA(auto& sphere)
    auto center = sphere.get<CENTER>(0 /* node ord */);
    auto radius = sphere.get<RADIUS>();
    center += radius[0];
  );

// Option 3: No aggregates within kernels
auto sphere_centers = ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["CENTER"]);
auto sphere_radii = ragg.get_accessor<accessor_t::VECTOR3<double>>(NODE_RANK, rename_map["RADIUS"]);

stk::mesh::for_each_entity_run(ngp_mesh, 
  KOKKOS_LAMBDA(stk::mesh::FastMeshIndex sphere_index)
    stk::mesh::FastMeshIndex center_node_index = ngp_mesh.fast_mesh_index(ngp_mesh.nodes(sphere_index)[0]);
    auto center = sphere_centers(center_node_index);
    auto radius = sphere_radii(sphere_index);
    center += radius[0];
  );


class RuntimeAggregate {
 public:
  RuntimeAggregate() = default;
  RuntimeAggregate(stk::topology top) : rank_(top.rank()), topology_(top) {}
  RuntimeAggregate(stk::mesh::EntityRank rank) : rank_(rank), topology_(stk::topology::INVALID_TOPOLOGY) {}

  RuntimeAggregate& add_accessor(const stk::mesh::EntityRank rank, std::string name, AccessorBase accessor) {
    auto result = ranked_accessor_maps_[rank].insert({name, accessor});
    MUNDY_THROW_REQUIRE(result.second, std::logic_error, 
      fmt::format("Accessor with rank {} and name '{}' already exists. No duplicates allowed.", rank, name));

    return this;
  }

  template<accessor_t A>
  auto get_accessor(const stk::mesh::EntityRank rank, std::string name) -> to_accessor_type_t<A> {
    // Check if an aggregate of the given rank/name exists
    MUNDY_THROW_REQUIRE(ranked_accessor_maps_[rank].contains(name), std::logic_error, 
      fmt::format("Failed to find aggregate of rank {} with name '{}'", rank, name));

    return dynamic_cast< to_accessor_type_t<A> >(ranked_accessor_maps_[rank][name]);
  }
 
 private:
  stk::mesh::EntitRank rank_;
  stk::topology topology_;

  using AggregateMap = std::map<std::string, AggregateBase>;
  AggregateMap ranked_accessor_maps_[stk::topology::NUM_RANKS];
};



}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_RAGGREGATES_HPP_
