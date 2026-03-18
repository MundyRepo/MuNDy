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

#ifndef MUNDY_MESH_AGGREGATES_HPP_
#define MUNDY_MESH_AGGREGATES_HPP_

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
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/FieldViews.hpp>       // for mundy::mesh::vector3_field_data, mundy::mesh::quaternion_field_data
#include <mundy_mesh/ForEachEntity.hpp>    // for mundy::mesh::for_each_entity_run
#include <mundy_mesh/NgpAccessorExpr.hpp>  // for mundy::mesh::AccessorExpr and EntityExprBase
#include <mundy_mesh/fmt_stk_types.hpp>    // for STK-compatible fmt::format
#include <mundy_utils/aggregate.hpp>  // for mundy::all_have_tags_v, mundy::all_tags_unique_v, mundy::contains_tag_v
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>       // for MUNDY_THROW_ASSERT
#include <mundy_utils/tuple.hpp>              // for mundy::tuple
#include <mundy_mesh/Components.hpp>         // for mundy::mesh::FieldComponent, mundy::mesh::Vector3FieldComponent, etc.

namespace mundy {

namespace mesh {

namespace impl {

/* What all do we offer
  - has_static_rank<T>: check if a tagged component defines a static rank member of type stk::topology::rank_t
  - all_have_static_rank<Components...>: check if every tagged component defines a static rank member of type
  stk::topology::rank_t

*/

// **********************************************************************************************************************
/// \brief Check if a type defines a static rank member of type stk::topology::rank_t
template <typename T, typename = void>
struct has_static_rank : std::false_type {};
//
template <typename T>
struct has_static_rank<T, std::void_t<decltype(T::rank)>>
    : std::bool_constant<std::is_same_v<std::remove_cv_t<decltype(T::rank)>, stk::topology::rank_t> &&
                         !std::is_member_object_pointer_v<decltype(&T::rank)>> {};
//
template <typename T>
static constexpr bool has_static_rank_v = has_static_rank<T>::value;
//
template <typename T>
concept has_static_rank_c = has_static_rank_v<T>;

// **********************************************************************************************************************
/// \brief Check if all types in a pack define a static rank member of type stk::topology::rank_t
template <typename... Components>
struct all_have_static_rank {
  static constexpr bool value = (has_static_rank_v<Components> && ...);
};

template <typename... Components>
static constexpr bool all_have_static_rank_v = all_have_static_rank<Components...>::value;

template <typename... Components>
concept all_have_static_rank_c = all_have_static_rank_v<Components...>;

/// \brief A helper class for wrapping a functor(view) with an operator()(FastMeshIndex)
template <typename NgpAggregateType, typename FunctorType>
class NgpFunctorWrapper {
 public:
  NgpFunctorWrapper(NgpAggregateType agg, const FunctorType& functor) : agg_(agg), functor_{functor} {
  }

  // Ignore the fact that functor may be a host function and yet operator is device compatable.
  // So long as operator() is evaluated on the host, all should be well.
  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH

  KOKKOS_FUNCTION
  void operator()(stk::mesh::FastMeshIndex entity_index) const {
    auto view = agg_.get_view(entity_index);
    functor_(view);
  }

  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_POP

 private:
  NgpAggregateType agg_;
  FunctorType functor_;
};  // NgpFunctorWrapper

}  // namespace impl

/// Forward declarations:
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
class EntityView;

template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... NgpComponents>
class NgpEntityView;

/// \brief An aggregator of components
///
/// # ECS Overview
/// This class is the main entry point for the user to interact with that we refer to as components, in accordance with
/// the Entity Component System (ECS). If you aren't familiar with this pattern, ECS is an architectural pattern
/// designed to decouple data from behavior, enabling flexibility, performance, and scalability in software systems.
/// ECS's has increased in popularity since the late 2000s and is now gaining widespread use within the gaming industry,
/// with addoption by Minecraft (via the elegant EnTT library) and engines like Unity ECS and Unreal ECS.
///
/// Unity states: "ECS (Entity Component System) is a data-oriented framework [that] scales processing performance,
/// enabling experienced creators to build more ambitious games with an unprecedented level of control and determinism."
///
///
/// # ECS Core Concepts
///   *Entities* are simple, unique identifiers—purely conceptual representations of "things" in your application.
///   They have no data or behavior themselves and are rather lightweight, often consisting of nothing more than a
///   unique ID. This makes getting and manipulating lists of entities far faster then getting and manipulating lists
///   of objects.
///
///   *Components* are data-only structures that can be assigned to entities. They contain no behavior, only data.
///   They typically represent a single aspect of an entity's state, such as position, velocity, mass, acceleration.
///   As with most ECS designs, components may be added or removed from entities at runtime.
///
///   Importantly, ECS typically discourages hard-coding collections of component at compile-time. Unlike
///   polymorphism, we do not offer a RigidBody class or Sphere class; rather, any entity that ~looks~ like a sphere
///   (i.e. has a component that ~looks~ like a radius and a component that ~looks~ like a center) ~is~ a sphere. This
///   does not require an explicit Sphere class and attempting to create one would be counter to the ECS design. This
///   gives a level of flexibility that is difficult to achieve with traditional object-oriented programming and allows
///   for more optimized memory access patterns and cache coherency.
///
///   *Systems* are functions or processes that operate on entities possessing specific sets of components. They are
///   typically "free" functions that are outside of any class hierarchy or inheritance chain. Unlike functions in a
///   class, users can add "free" functions meant to operate on entities with specific components without needing to
///   modify some hard-coded class definition. This is more flexible to extension, as users can add new free functions
///   without needing to modify existing classes. In many ways, we like to think of ECS as a runtime-extensible
///   deconstructed class hierarchy.
///
///
/// # STK's flavor of ECS
/// STK's domain model can be seen as an extension of ECS, adding to it the concept of connections between entities of
/// different *ranks* and the ability for entities to possess a graph *topology*, statically defining its connectivity.
/// This concept of rank and topology is common in mesh-based or molecular dynamics simulations, where we have nodes,
/// edges, faces, and elements that connect to each other in a hierarchical manner. Unlike simplistic ECS systems, like
/// EnTT, adding ranks and topologies complicates the design, necessitating additional features and care.
///
/// STK introduces the concept of Parts, Fields, and Selectors.
/// data.
///
///   *Parts* are collections of entities that share the same properties. They may possess a topology, requiring that
///   all entities in that part of the same rank as the topology, have said topology. They may instead possess only a
///   rank, allowing them to hold any entities of that rank. Or they may possess neither, allowing them to be used as
///   Assemblies of any entities. Importantly, Parts may be subsets of other Parts, allowing for hierarchical
///   organization at runtime. Parts (by default) have inherited part membership, meaning that if an entity within the
///   part of its primary rank connects to an entity of lower rank, that entity is also considered to be within the
///   part.
///
///   *Fields* are collections of ranked data that can be assigned to any number of Parts. These Fields can be seen as
///   one type of component. They have a rank, a name, and a type. If a Part has a Field, then all entities in that
///   Part of the same rank as the Field pickup that Field.
///
///   *Selectors* (also called groups or views in other ECS systems) are a means of identifying a subset of entities. In
///   STK, Selectors are formed using set arithmetic applied to the Parts and Fields. For example, a Selector might
///   abstractly represent "all entities of Part A that have Field B but are not in Part C". Selectors are used to
///   define the scope of Systems such as "for all entities in Selector X, do Y". In practice, the smallest unit of work
///   in STK is defined by a Selector (Parts themselves may act as Selectors). If there is ever a need to iterate over a
///   specific subset set of entities that cannot be fetched by set arithmetic applied to your current Parts and Fields,
///   you likely need to create a new Part. That said, sometimes it's more efficient to iterate over a larger set of
///   entities and use a conditional to filter out the entities you don't want. If the subset of entities you wish to
///   iterate over is large, then using a new Part is likely more efficient.
///
/// # Accessors
/// We extend STK's domain model to include the concept of Aggregates and Accessors, as an organizational layer above
/// Parts, Fields, and Selectors, meant to abstract away access patterns into an entity's data while reducing
/// boilerplate code. Similar to Field and Parts, we ~assemble~ Aggregates at runtime.
///
/// Notably, Accessors overcome the following limitation of STK's domain model:
///   It's common to have a collection of entities within some set of Parts for which we want to store a shared value.
///   This might include a collection of spheres with the same radius and material properties. Similarly, one might want
///   to have a single shared material per part. Simply because a collection of spheres share a radius rather than
///   having a radius stored within a Field shouldn't impact the design of systems meant to operate on spheres, and yet
///   STK offers no means to abstract away shared vs non-shared data. This is a direct consequence of the lack of
///   separation of concerns with regard to data storage and access patterns.
///
/// Accessors provide an interface through which data beyond just Fields may be treated as components and accessed in a
/// unified manner. If a user starts with an aggregate for spheres that have a shared radius and then decides later to
/// switch to using a Field of radii, they need only update the aggregate's definition. The systems that act on the
/// aggregate will remain unchanged, as how the data is accesses does not concern them. Notably, accessors are ~views~
/// into data, i.e. they should not ~hold~ the data they access. They are meant to be cheap to construct and trivial to
/// copy (just like Kokkos::Views). Interface-wise, Accessors must provide a sync_to_host, sync_to_device,
/// modify_on_host, and modify_on_device method the same as STK's NgpField. When called, these methods should
/// synchronize the data to the appropriate space/mark the data as modified. Synchronization should be a no-op if the
/// data is up-to-date on the requested space.
///
/// Our Accessors, the data they access, and the return type of the Accessor's get_view method are as follows:
///          Component Name                :              Data it accesses             ->         Return Type
///   ScalarFieldComponent                 :  Field<scalar_t>                          ->  ScalarView<scalar_t>
///   VectorNFieldComponent                :  Field<scalar_t>                          ->  VectorView<scalar_t, N>
///   Matrix3FieldComponent                :  Field<scalar_t>                          ->  Matrix3View<scalar_t>
///   QuaternionFieldComponent             :  Field<scalar_t>                          ->  QuaternionView<scalar_t>
///   AABBFieldComponent                   :  Field<scalar_t>                          ->  AABBView<scalar_t>
///   SharedViewComponent (size 1)         :  mundy::NgpVector<SharedType>             ->  SharedType&
///   PartMappedComponent<OtherComponent>  :  Kokkos::Map<PartOrdinal, OtherComponent> -> OtherComponent's return type
///
/// \note mundy::NgpVector is just a wrapper for a Kokkos::View and its host mirror with the necessary sync/modify
/// methods. Just like Kokkos::Views, it takes ownership of the data it accesses. We cannot simply store a reference to
/// the given SharedType, as we have no guarantee that the data will remain valid for the lifetime of the Accessor nor
/// are we able to copy a reference to host memory to the device.
///
///
/// # Aggregates
/// Aggregates are just collections of accessors(components) that can be accessed via a unified EntityView. The
/// components are "tagged". That is, they are associated with some type that is used to fetch the data. That type is
/// often nothing more than an emtpy struct, but it can be used to differentiate between components that have the same
/// underlying type. This is opposed to creating a SphereAggregate which stores a radius_field and a center_field.
/// Instead, we can access the radius and center accessors via their tags.
///
/// The Aggregate class can be constructed directly "Aggregate<topology, rank>(bulk_data, selector)", but we also offer
/// some non-member helper functions to streamline this process.
///   - Use "make_aggregate<topology>(bulk_data, selector)" to avoid the need to specify the rank redundantly.
///   - Use "make_ranked_aggregate<rank>(bulk_data, selector)" if the entities don't have a topology.
///
/// Adding components to an aggregate is done via the fluent interface "add_component<Tag>(accessor)",
/// which returns a new aggregate with the added component. This allows for easy chaining of components,
/// as seen in the below example.
///
/// Aggregates can then be used to fetch an EntityView for a given entity, which can be used to access the entity's
/// components. It offers basic information like the rank and topology of the entity, as well as the entity itself
/// and access to its connections. To then fetch the components, use the get<Tag>() method for components of the
/// same rank as the entity, or get<Tag>(connectivity_ordinal) for components of a lower rank then the entity. To then
/// apply functors to all entities in the aggregate, use the for_each method. This method runs the given functor on the
/// EntityView of each entity in the aggregate.
///
/// \note Accessors and aggregates are not a replacement for STK's Field and Part system. They are an abstraction layer
///   that sits above and beside it. We choose to use Aggregates to organize our data and Accessors to access it, but
///   you may also directly act on accessors or directly on fields and parts. The choice is yours.
///
///
/// # Example Usage
/// \code {.cpp}
///    // We'll assume that there exists an elem1 within a Spheres part of PARTICLE topology connected to a node1
///    //   with a NODE_RANK center_field and a shared ELEM_RANK radius. Both have double type.
///
///    // Create the accessors
///    auto center_accessor = make_scalar_field_accessor(center_field);
///    auto radius_accessor = make_shared_view_accessor(mundy::NgpVector<float>{radius});  // Radius is copied
///
///    // Fetch the data for the entity via the accessor's operator()
///    Vector3View<double> center = center_accessor(elem1);
///    double& radius = radius_accessor(node1);
///
///    // Create an aggregate for the spheres
///    auto collision_sphere_data = make_aggregate<stk::topology::PARTICLE>(bulk_data, selector)
///            .add_component<CENTER>(center_accessor)
///            .add_component<COLLISION_RADIUS>(radius_accessor);
///
///    // Sync the data to the device and mark it as modified
///    collision_sphere_data.sync_to_device<CENTER, COLLISION_RADIUS>();
///    collision_sphere_data.modify_on_device<CENTER, COLLISION_RADIUS>();
///
///    // Do the same directly via the accessors
///    center_accessor.sync_to_device();
///    radius_accessor.modify_on_device();
///    collision_sphere_data.get_component<CENTER>().sync_to_device();
///    collision_sphere_data.get_component<COLLISION_RADIUS>().modify_on_device();
///
///    // Get an accessor-independent view of the entity's data
///    auto sphere_view = collision_sphere_data.get_view(entity1);
///    sphere_view.entity();    // Returns entity1
///    sphere_view.rank();      // Returns ELEM_RANK
///    sphere_view.topology();  // Returns PARTICLE
///    unsigned center_node_con_ordinal = 0;
///    Vector3View<double> also_center = sphere_view.get<CENTER>(center_node_con_ordinal);
///    double& also_radius = sphere_view.get<COLLISION_RADIUS>();
///
///    // Apply a functor to all entities in the aggregate
///    collision_sphere_data.for_each([](auto& other_sphere_view) {
///        Vector3View<double> c = other_sphere_view.get<CENTER>();
///        double& r = other_sphere_view.get<COLLISION_RADIUS>();
///        std::cout << "Center = " << c << ", Radius = " << r << std::endl;
///    });
///
///    // Run a functor move_spheres<CenterTag, RadiusTag> on all entities in the aggregate
///    collision_sphere_data.for_each(move_spheres<CENTER, COLLISION_RADIUS>{});
///
///   // Use a functor to move all spheres
///   move_spheres<CENTER, COLLISION_RADIUS>{}.apply_to(collision_sphere_data);
///
///    // Directly use accessors without an aggregate
///    stk::mesh::for_each_entity_run(bulk_data, stk::topology::ELEM_RANK, selector,
///       [center_accessor, radius_accessor](const stk::mesh::BulkData &bulk_data, const stk::mesh::Entity &entity) {
///           Vector3View<double> c = center_accessor(entity);
///           double& r = radius_accessor(entity);
///           std::cout << "Center = " << c << ", Radius = " << r << std::endl;
///       });
///
///   // Directly pass accessors to a free-function move_spheres2 templated by the accessor types
///   move_spheres2(center_accessor, radius_accessor);
/// \endcode
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
  requires(all_have_tags_v<Components...>                 /* All of the given components must have tags. */
           && impl::all_have_static_rank_v<Components...> /* All components must have a static rank. */
           && all_tags_unique_v<Components...> /* All tags in an Aggregate must be unique. */)
class Aggregate {
 public:
  using ComponentsTuple = tuple<Components...>;

  //! \name Constructors
  //@{

  /// \brief Construct an Aggregate that has no components
  Aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector)
    requires(sizeof...(Components) == 0)
      : bulk_data_(bulk_data), selector_(std::move(selector)), components_{} {
  }

  /// \brief Construct an Aggregate that has the given components
  Aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector, ComponentsTuple components)
      : bulk_data_(bulk_data), selector_(std::move(selector)), components_(std::move(components)) {
  }

  /// \brief Default copy/move/assign constructors
  Aggregate(const Aggregate&) = default;
  Aggregate(Aggregate&&) = default;
  Aggregate& operator=(const Aggregate&) = default;
  Aggregate& operator=(Aggregate&&) = default;
  //@}

  //! \name Accessors
  //@{

  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }
  const stk::mesh::BulkData& bulk_data() const {
    return bulk_data_;
  }
  const stk::mesh::MetaData& mesh_meta_data() const {
    return bulk_data_.mesh_meta_data();
  }
  const stk::mesh::Selector& selector() const {
    return selector_;
  }
  //@}

  /// \brief Add a component (fluent interface):
  template <typename Tag, stk::topology::rank_t component_rank, typename NewComponent>
    requires(
        all_tags_unique_v<TaggedComponent<Tag, component_rank, NewComponent>,
                          Components...> /* The new component's tag must be unique from all existing component tags */)
  auto add_component(NewComponent new_component) const {
    auto new_tagged_comp = TaggedComponent<Tag, component_rank, NewComponent>{std::move(new_component)};
    auto new_tuple = tuple_cat(components_, make_tuple(new_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = Aggregate<OurTopology, OurRank, Components..., decltype(new_tagged_comp)>;
    return NewType(bulk_data_, selector_, new_tuple);
  }

  template <typename Tag, stk::topology::rank_t component_rank, typename NewComponent>
    requires(!all_tags_unique_v<TaggedComponent<Tag, component_rank, NewComponent>, Components...>)
  void add_component(NewComponent /*new_component*/) const {
    static_assert(all_tags_unique_v<TaggedComponent<Tag, component_rank, NewComponent>, Components...>,
                  "The new component's tag must be unique from all existing component tags.");
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
    requires(contains_tag_v<Tag, Components...>)
  const auto& get_component() const {
    return ::mundy::impl::find_component<Tag>(components_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, Components...>)
  const void get_component() const {
    static_assert(contains_tag_v<Tag, Components...>,
                  "Attempting to get a component that does not exist in the aggregate");
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
    requires(contains_tag_v<Tag, Components...>)
  auto& get_component() {
    return ::mundy::impl::find_component<Tag>(components_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, Components...>)
  void get_component() {
    static_assert(contains_tag_v<Tag, Components...>,
                  "Attempting to get a component that does not exist in the aggregate");
  }

  /// \brief Synchronize the components marked by the given tags to the device
  template <typename... TagsToSync>
  void sync_to_device() {
    (get_component<TagsToSync>().sync_to_device(), ...);
  }

  /// \brief Synchronize the components marked by the given tags to the host
  template <typename... TagsToSync>
  void sync_to_host() {
    (get_component<TagsToSync>().sync_to_host(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the device
  template <typename... TagsToModify>
  void modify_on_device() {
    (get_component<TagsToModify>().modify_on_device(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the host
  template <typename... TagsToModify>
  void modify_on_host() {
    (get_component<TagsToModify>().modify_on_host(), ...);
  }

  /// \brief Get an EntityView for the given entity
  EntityView<OurTopology, OurRank, Components...> get_view(stk::mesh::Entity entity) const {
    return EntityView<OurTopology, OurRank, Components...>(*this, entity);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate that is also in the given
  /// subset selector
  template <typename Functor>
  void for_each(const stk::mesh::Selector& subset_selector, const Functor& f) const {
    // Only loop over the set intersection of the agg's selector and the subset selector
    auto agg = *this;
    stk::mesh::Selector sel = agg.selector() & subset_selector;

    auto wrapped_functor = [&agg, &f](const stk::mesh::BulkData& bulk_data, const stk::mesh::Entity& entity) {
      auto view = agg.get_view(entity);
      f(view);
    };

    ::mundy::mesh::for_each_entity_run(agg.bulk_data(), agg.rank(), sel, wrapped_functor);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate
  template <typename Functor>
  void for_each(const Functor& f) const {
    auto agg = *this;

    auto wrapped_functor = [&agg, &f]([[maybe_unused]] const stk::mesh::BulkData& bulk_data,
                                      const stk::mesh::Entity& entity) {
      auto view = agg.get_view(entity);
      f(view);
    };

    ::mundy::mesh::for_each_entity_run(agg.bulk_data(), agg.rank(), agg.selector(), wrapped_functor);
  }

 private:
  //! \name Private members
  //@{

  const stk::mesh::BulkData& bulk_data_;
  stk::mesh::Selector selector_;
  ComponentsTuple components_;
  //@}
};  // Aggregate

template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... NgpComponents>
  requires(::mundy::all_have_tags_v<NgpComponents...> /* All of the given components must have tags */
           && ::mundy::all_tags_unique_v<NgpComponents...> /* All tags in an NgpAggregate must be unique */)
class NgpAggregate {
 public:
  using NgpComponentsTuple = tuple<NgpComponents...>;

  //! \name Constructors
  //@{

  /// \brief Default constructor
  NgpAggregate() : ngp_mesh_{}, host_selector_{}, ngp_components_{} {
  }

  /// \brief Construct an Aggregate that has no components
  NgpAggregate(stk::mesh::NgpMesh ngp_mesh, stk::mesh::Selector selector)
    requires(sizeof...(NgpComponents) == 0)
      : ngp_mesh_(ngp_mesh), host_selector_(std::move(selector)), ngp_components_{} {
  }

  /// \brief Construct an Aggregate that has the given components
  NgpAggregate(stk::mesh::NgpMesh ngp_mesh, stk::mesh::Selector selector, NgpComponentsTuple ngp_components)
      : ngp_mesh_(ngp_mesh), host_selector_(std::move(selector)), ngp_components_(std::move(ngp_components)) {
  }

  /// \brief Default move/copy/assign constructors
  NgpAggregate(NgpAggregate&& other)
      : ngp_mesh_(other.ngp_mesh_), host_selector_(other.host_selector_), ngp_components_(other.ngp_components_) {
  }
  NgpAggregate(const NgpAggregate& other)
      : ngp_mesh_(other.ngp_mesh_), host_selector_(other.host_selector_), ngp_components_(other.ngp_components_) {
  }
  NgpAggregate& operator=(NgpAggregate&& other) {
    ngp_mesh_ = other.ngp_mesh_;
    host_selector_ = other.host_selector_;
    ngp_components_ = other.ngp_components_;
    return *this;
  }
  NgpAggregate& operator=(const NgpAggregate& other) {
    ngp_mesh_ = other.ngp_mesh_;
    host_selector_ = other.host_selector_;
    ngp_components_ = other.ngp_components_;
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }

  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  const stk::mesh::BulkData& bulk_data() const {
    return ngp_mesh_.get_bulk_on_host();
  }

  const stk::mesh::MetaData& mesh_meta_data() const {
    return ngp_mesh_.get_bulk_on_host().mesh_meta_data();
  }

  const stk::mesh::Selector& selector() const {
    return host_selector_;
  }
  //@}

  /// \brief Add a component (fluent interface):
  /// TODO(palmerb4): If we do decide to use get_updated_ngp_aggregate with references, then this function will
  ///   need removed, as Aggregates managing the lifetime of NGP components means users should construct them
  template <typename Tag, stk::topology::rank_t component_rank, typename NewNgpComponent>
    requires(all_tags_unique_v<NgpTaggedComponent<Tag, component_rank, NewNgpComponent>, NgpComponents...>)
  auto add_component(NewNgpComponent new_ngp_component) const {
    auto new_ngp_tagged_comp = NgpTaggedComponent<Tag, component_rank, NewNgpComponent>{std::move(new_ngp_component)};
    auto new_tuple = tuple_cat(ngp_components_, make_tuple(new_ngp_tagged_comp));

    // Form the new type that has the old components plus the new appended
    // one.
    using NewType = NgpAggregate<OurTopology, OurRank, NgpComponents..., decltype(new_ngp_tagged_comp)>;
    return NewType(ngp_mesh_, host_selector_, new_tuple);
  }
  template <typename Tag, stk::topology::rank_t component_rank, typename NewNgpComponent>
    requires(!all_tags_unique_v<NgpTaggedComponent<Tag, component_rank, NewNgpComponent>, NgpComponents...>)
  void add_component(NewNgpComponent /*new_ngp_component*/) const {
    static_assert(all_tags_unique_v<NgpTaggedComponent<Tag, component_rank, NewNgpComponent>, NgpComponents...>,
                  "The new component's tag must be unique from all existing component tags.");
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
    requires(contains_tag_v<Tag, NgpComponents...>)
  KOKKOS_INLINE_FUNCTION const auto& get_component() const {
    return ::mundy::impl::find_component<Tag>(ngp_components_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, NgpComponents...>)
  KOKKOS_INLINE_FUNCTION const void get_component() const {
    static_assert(contains_tag_v<Tag, NgpComponents...>,
                  "Attempting to get a component that does not exist in the NGP aggregate");
  }

  /// \brief Fetch the component corresponding to the given Tag
  template <typename Tag>
    requires(contains_tag_v<Tag, NgpComponents...>)
  KOKKOS_INLINE_FUNCTION auto& get_component() {
    return ::mundy::impl::find_component<Tag>(ngp_components_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, NgpComponents...>)
  KOKKOS_INLINE_FUNCTION void get_component() {
    static_assert(contains_tag_v<Tag, NgpComponents...>,
                  "Attempting to get a component that does not exist in the NGP aggregate");
  }

  /// \brief Synchronize the components marked by the given tags to the device
  template <typename... TagsToSync>
  void sync_to_device() {
    (get_component<TagsToSync>().sync_to_device(), ...);
  }

  /// \brief Synchronize the components marked by the given tags to the host
  template <typename... TagsToSync>
  void sync_to_host() {
    (get_component<TagsToSync>().sync_to_host(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the device
  template <typename... TagsToModify>
  void modify_on_device() {
    (get_component<TagsToModify>().modify_on_device(), ...);
  }

  /// \brief Mark the components marked by the given tags as modified on the host
  template <typename... TagsToModify>
  void modify_on_host() {
    (get_component<TagsToModify>().modify_on_host(), ...);
  }

  /// \brief Get an EntityView for the given entity
  KOKKOS_INLINE_FUNCTION
  NgpEntityView<OurTopology, OurRank, NgpComponents...> get_view(stk::mesh::FastMeshIndex entity_index) const {
    return NgpEntityView<OurTopology, OurRank, NgpComponents...>(ngp_mesh_, ngp_components_, entity_index);
  }

  /// \brief Get an EntityView for the given entity
  KOKKOS_INLINE_FUNCTION
  NgpEntityView<OurTopology, OurRank, NgpComponents...> get_view(stk::mesh::Entity entity) const {
    stk::mesh::FastMeshIndex entity_index = ngp_mesh_.fast_mesh_index(entity);
    return NgpEntityView<OurTopology, OurRank, NgpComponents...>(ngp_mesh_, ngp_components_, entity_index);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate that is also in the given
  /// subset selector
  template <typename Functor>
  inline void for_each(const stk::mesh::Selector& subset_selector, const Functor& f) const {
    // Only loop over the set intersection of the agg's selector and the subset selector
    using our_t = NgpAggregate<OurTopology, OurRank, NgpComponents...>;
    our_t agg = *this;

    auto local_ngp_mesh = agg.ngp_mesh();
    impl::NgpFunctorWrapper<our_t, Functor> wrapper(agg, f);
    stk::mesh::Selector sel = agg.selector() & subset_selector;
    ::mundy::mesh::for_each_entity_run(local_ngp_mesh, agg.rank(), sel, wrapper);
  }

  /// \brief Apply a functor on the EntityView of each entity in the current data aggregate
  template <typename Functor>
  inline void for_each(const Functor& f) const {
    using our_t = NgpAggregate<OurTopology, OurRank, NgpComponents...>;
    our_t agg = *this;

    auto local_ngp_mesh = agg.ngp_mesh();
    impl::NgpFunctorWrapper<our_t, Functor> wrapper(agg, f);
    ::mundy::mesh::for_each_entity_run(local_ngp_mesh, agg.rank(), agg.selector(), wrapper);
  }

 private:
  //! \name Private members
  //@{

  stk::mesh::NgpMesh ngp_mesh_;
  stk::mesh::Selector host_selector_;
  NgpComponentsTuple ngp_components_;
  //@}
};  // NgpAggregate

/// \brief A view into the components, connections, and meta-information for an entity
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
class EntityView {
 public:
  /// \brief Construct an EntityView for the given entity
  /// TODO(palmerb4) Optimize for reuse of connectivity.
  EntityView(const Aggregate<OurTopology, OurRank, Components...>& parent, stk::mesh::Entity entity)
      : parent_(parent), entity_(entity) {
    MUNDY_THROW_ASSERT(parent.bulk_data().is_valid(entity), std::runtime_error,
                       "EntityView created with invalid entity");
    MUNDY_THROW_ASSERT(parent.bulk_data().entity_rank(entity) == OurRank, std::runtime_error,
                       fmt::format("EntityView created with entity of rank {} but aggregate has rank {}",
                                   parent.bulk_data().entity_rank(entity), OurRank));
  }

 private:
  const Aggregate<OurTopology, OurRank, Components...>& parent_;
  stk::mesh::Entity entity_;

 public:
  /// \brief Fetch the entity that we view
  stk::mesh::Entity entity() const {
    return entity_;
  }

  /// \brief Fetch the rank of the entity that we view
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }

  /// \brief Fetch the topology of the entity that we view
  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, Components...> &&
             (type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  decltype(auto) get() {
    auto& comp = parent_.template get_component<Tag>();
    return comp(entity_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, Components...> ||
             !(type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  void get() {
    static_assert(contains_tag_v<Tag, Components...>, "EntityView::get() called with a tag that does not exist");
    static_assert((type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank),
                  "EntityView::get() called with a tag that does not correspond to a component of the same rank as "
                  "the entity");
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, Components...> &&
             (type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  decltype(auto) get() const {
    auto& comp = parent_.template get_component<Tag>();
    return comp(entity_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, Components...> ||
             !(type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  void get() const {
    static_assert(contains_tag_v<Tag, Components...>, "EntityView::get() called with a tag that does not exist");
    static_assert((type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank),
                  "EntityView::get() called with a tag that does not correspond to a component of the same rank as "
                  "the entity");
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, Components...> &&
             !(type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  decltype(auto) get(unsigned connectivity_ordinal) {
    using TaggedComponentType = std::decay_t<decltype(parent_.template get_component<Tag>())>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = parent_.template get_component<Tag>();

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().num_connectivity(entity_, comp_rank) > connectivity_ordinal, std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but entity has only {} "
                    "connectivities of rank {}",
                    connectivity_ordinal, parent_.bulk_data().num_connectivity(entity_, comp_rank), comp_rank));

    // TODO: The following assumes that the entity is connected to all of its possible lower rank entities
    // such that the list of connected entities can be indexed by the connectivity_ordinal.
    const stk::mesh::Entity& connected_entity = parent_.bulk_data().begin(entity_, comp_rank)[connectivity_ordinal];

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().is_valid(connected_entity), std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but connected entity is invalid",
                    connectivity_ordinal));

    return comp(connected_entity);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, Components...> ||
             (type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  void get(unsigned /*connectivity_ordinal*/) {
    static_assert(contains_tag_v<Tag, Components...>, "EntityView::get(ordinal) called with a tag that does not exist");
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, Components...> &&
             !(type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  decltype(auto) get(unsigned connectivity_ordinal) const {
    using TaggedComponentType = std::decay_t<decltype(parent_.template get_component<Tag>())>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = parent_.template get_component<Tag>();

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().num_connectivity(entity_, comp_rank) > connectivity_ordinal, std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but entity has only {} "
                    "connectivities of rank {}",
                    connectivity_ordinal, parent_.bulk_data().num_connectivity(entity_, comp_rank), comp_rank));

    const stk::mesh::Entity& connected_entity = parent_.bulk_data().begin(entity_, comp_rank)[connectivity_ordinal];

    MUNDY_THROW_ASSERT(
        parent_.bulk_data().is_valid(connected_entity), std::runtime_error,
        fmt::format("EntityView::get() called with connectivity_ordinal {} but connected entity is invalid",
                    connectivity_ordinal));

    return comp(connected_entity);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, Components...> ||
             (type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank))
  void get(unsigned /*connectivity_ordinal*/) const {
    static_assert(contains_tag_v<Tag, Components...>, "EntityView::get(ordinal) called with a tag that does not exist");
    static_assert(
        (type_at_index_t<index_of_tag_v<Tag, Components...>, Components...>::rank == OurRank),
        "EntityView::get(ordinal) called with a tag that does not correspond to a component of a different rank "
        "than the entity");
  }
};  // EntityView

/// \brief A view into the components, connections, and meta-information for an entity
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... NgpComponents>
class NgpEntityView {
 public:
  using NgpComponentsTuple = tuple<NgpComponents...>;

  /// \brief Construct an EntityView for the given entity
  /// TODO(palmerb4) Optimize for reuse of connectivity.
  KOKKOS_INLINE_FUNCTION
  NgpEntityView(const stk::mesh::NgpMesh& ngp_mesh, const NgpComponentsTuple& components,
                stk::mesh::FastMeshIndex entity_index)
      : ngp_mesh_(ngp_mesh), ngp_components_(components), entity_index_(entity_index) {
  }

 private:
  const stk::mesh::NgpMesh& ngp_mesh_;
  const NgpComponentsTuple& ngp_components_;
  stk::mesh::FastMeshIndex entity_index_;

 public:
  /// \brief Fetch the entity that we view
  KOKKOS_INLINE_FUNCTION
  stk::mesh::FastMeshIndex entity_index() const {
    return entity_index_;
  }

  /// \brief Fetch the identifier of the entity that we view
  KOKKOS_INLINE_FUNCTION
  stk::mesh::EntityId entity_id() const {
    return ngp_mesh_.identifier(ngp_mesh_.get_entity(OurRank, entity_index_));
  }

  /// \brief Fetch the rank of the entity that we view
  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::rank_t rank() {
    return OurRank;
  }

  /// \brief Fetch the topology of the entity that we view
  KOKKOS_INLINE_FUNCTION
  static constexpr stk::topology::topology_t topology() {
    return OurTopology;
  }

  /// \brief Fetch the ngp mesh
  KOKKOS_INLINE_FUNCTION
  const stk::mesh::NgpMesh& ngp_mesh() const {
    return ngp_mesh_;
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, NgpComponents...> &&
             (type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION decltype(auto) get() {
    auto& comp = ::mundy::impl::find_component<Tag>(ngp_components_);
    return comp(entity_index_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, NgpComponents...> ||
             !(type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION void get() {
    static_assert(contains_tag_v<Tag, NgpComponents...>, "NgpEntityView::get() called with a tag that does not exist");
    static_assert((type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank),
                  "NgpEntityView::get() called with a tag that does not correspond to a component of the same rank "
                  "as the entity");
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of the same rank as the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, NgpComponents...> &&
             (type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION decltype(auto) get() const {
    auto& comp = ::mundy::impl::find_component<Tag>(ngp_components_);
    return comp(entity_index_);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, NgpComponents...> ||
             !(type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION void get() const {
    static_assert(contains_tag_v<Tag, NgpComponents...>, "NgpEntityView::get() called with a tag that does not exist");
    static_assert((type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank),
                  "NgpEntityView::get() called with a tag that does not correspond to a component of the same rank "
                  "as the entity");
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, NgpComponents...> &&
             !(type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION decltype(auto) get(unsigned connectivity_ordinal) {
    using TaggedComponentType = std::decay_t<decltype(::mundy::impl::find_component<Tag>(ngp_components_))>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = ::mundy::impl::find_component<Tag>(ngp_components_);

    // TODO: The following assumes that the entity is connected to all of its possible lower rank entities
    // such that the list of connected entities can be indexed by the connectivity_ordinal.
    const auto connected_entities = ngp_mesh_.get_connected_entities(OurRank, entity_index_, comp_rank);

    MUNDY_THROW_ASSERT(connected_entities.size() > connectivity_ordinal, std::runtime_error,
                       "EntityView::get() called with a connectivity ordinal that exceeds the number of connected "
                       "entities of the tag rank");

    const stk::mesh::FastMeshIndex connected_entity_index =
        ngp_mesh_.fast_mesh_index(connected_entities[connectivity_ordinal]);
    return comp(connected_entity_index);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, NgpComponents...> ||
             (type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION void get(unsigned /*connectivity_ordinal*/) {
    static_assert(contains_tag_v<Tag, NgpComponents...>,
                  "NgpEntityView::get(ordinal) called with a tag that does not exist");
    static_assert(
        !(type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank),
        "NgpEntityView::get(ordinal) called with a tag that corresponds to a component of the same rank as the entity");
  }

  /// \brief Get the data marked by the given tag and fetched using the corresponding accessor
  /// Only works for components of a different rank then the entity
  template <typename Tag>
    requires(contains_tag_v<Tag, NgpComponents...> &&
             !(type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION decltype(auto) get(unsigned connectivity_ordinal) const {
    using TaggedComponentType = std::decay_t<decltype(::mundy::impl::find_component<Tag>(ngp_components_))>;
    static constexpr auto comp_rank = TaggedComponentType::rank;
    auto& comp = ::mundy::impl::find_component<Tag>(ngp_components_);
    const auto connected_entities = ngp_mesh_.get_connected_entities(OurRank, entity_index_, comp_rank);

    MUNDY_THROW_ASSERT(connected_entities.size() > connectivity_ordinal, std::runtime_error,
                       "EntityView::get() called with a connectivity ordinal that exceeds the number of connected "
                       "entities of the tag rank");

    const stk::mesh::FastMeshIndex connected_entity_index =
        ngp_mesh_.fast_mesh_index(connected_entities[connectivity_ordinal]);
    return comp(connected_entity_index);
  }
  template <typename Tag>
    requires(!contains_tag_v<Tag, NgpComponents...> ||
             (type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank))
  KOKKOS_INLINE_FUNCTION void get(unsigned /*connectivity_ordinal*/) const {
    static_assert(contains_tag_v<Tag, NgpComponents...>,
                  "NgpEntityView::get(ordinal) called with a tag that does not exist");
    static_assert(
        !(type_at_index_t<index_of_tag_v<Tag, NgpComponents...>, NgpComponents...>::rank == OurRank),
        "NgpEntityView::get(ordinal) called with a tag that corresponds to a component of the same rank as the entity");
  }
};  // NgpEntityView

/// \brief Make an aggregate for the given topology
template <stk::topology::topology_t OurTopology>
auto make_aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector) {
  static constexpr stk::topology::rank_t rank = stk::topology_detail::topology_data<OurTopology>::rank;
  return Aggregate<OurTopology, rank>(bulk_data, selector);
}

/// \brief Make an aggregate for the given rank
template <stk::topology::rank_t OurRank>
auto make_ranked_aggregate(const stk::mesh::BulkData& bulk_data, stk::mesh::Selector selector) {
  return Aggregate<stk::topology::INVALID_TOPOLOGY, OurRank>(bulk_data, selector);
}

/// \brief Get a component of the given aggregate (const)
/// This simply calls the get_component method of the given aggregate and solely exists so you don't need to write
///  "aggregate. template get_component<Tag>()" every time you want to fetch a component. Instead,
/// you use "get_component<Tag>(aggregate)". Same concept as std::get<N>(tuple).
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
const auto& get_component(const Aggregate<OurTopology, OurRank, Components...>& aggregate) {
  return aggregate.template get_component<Tag>();
}

/// \brief Get a component of the given aggregate
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
auto& get_component(Aggregate<OurTopology, OurRank, Components...>& aggregate) {
  return aggregate.template get_component<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view (const)
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(const EntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(EntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(const EntityView<OurTopology, OurRank, Components...>& entity_view, unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
decltype(auto) get(EntityView<OurTopology, OurRank, Components...>& entity_view, unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief Get the data tagged by the given tag from the given entity view (const)
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(const NgpEntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(NgpEntityView<OurTopology, OurRank, Components...>& entity_view) {
  return entity_view.template get<Tag>();
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(const NgpEntityView<OurTopology, OurRank, Components...>& entity_view,
                                          unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief Get the data tagged by the given tag from the given entity view
template <typename Tag, stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... Components>
KOKKOS_INLINE_FUNCTION decltype(auto) get(NgpEntityView<OurTopology, OurRank, Components...>& entity_view,
                                          unsigned connectivity_ordinal) {
  return entity_view.template get<Tag>(connectivity_ordinal);
}

/// \brief A helper function for getting the NGP aggregate from a regular aggregate
template <stk::topology::topology_t OurTopology, stk::topology::rank_t OurRank, typename... TaggedComponents>
auto get_updated_ngp_aggregate(const Aggregate<OurTopology, OurRank, TaggedComponents...>& aggregate) {
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(aggregate.bulk_data());

  auto ngp_components =
      make_tuple(get_updated_ngp_component(aggregate.template get_component<typename TaggedComponents::tag_type>())...);

  return NgpAggregate<OurTopology, OurRank,
                      std::decay_t<decltype(get_updated_ngp_component(
                          aggregate.template get_component<typename TaggedComponents::tag_type>()))>...>(
      ngp_mesh, aggregate.selector(), ngp_components);
}

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_AGGREGATES_HPP_
