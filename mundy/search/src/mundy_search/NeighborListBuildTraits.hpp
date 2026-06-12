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

#ifndef MUNDY_SEARCH_NEIGHBORLISTBUILDTRAITS_HPP_
#define MUNDY_SEARCH_NEIGHBORLISTBUILDTRAITS_HPP_

/// \file NeighborListBuildTraits.hpp
/// \brief NeighborListInputType concept and NeighborListBuildTraits primary template.
///
/// Include this header when specializing NeighborListBuildTraits for a new concrete list type.

// C++ core
#include <concepts>     // for std::same_as
#include <type_traits>  // for std::remove_cvref_t

// Trilinos
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>  // for stk::mesh::EntityRank

// Mundy
#include <mundy_geom/primitives/AABB.hpp>  // for mundy::is_aabb_v

namespace mundy {

namespace search {

/// \concept NeighborListInputType
/// \brief A component-backed selected target/source chunk used to build a neighbor list (`SearchInput` /
/// `PeriodicSearchInput`).
///
/// Exposes the selector (the entity chunk), the geometry-yielding component that maps each entity to its geometry,
/// and the entity rank of the chunk. The pair `(selector, component)` fully determines which entities are searched,
/// their ordering, and how to (re)read their geometry — including for entities ghosted during a multi-rank build,
/// which a raw box view cannot supply. Every neighbor-list build consumes a component input, so this is the single
/// input contract: raw box views are internal build buffers, never public inputs.
template <typename T>
concept NeighborListInputType = requires(const T& input) {
  { input.selector() } -> std::same_as<const stk::mesh::Selector&>;
  input.component();
  { input.rank() } -> std::same_as<stk::mesh::EntityRank>;
};

/// \concept AABBSearchInputType
/// \brief A `NeighborListInputType` whose component yields an AABB — the broad-phase volume expected by the
/// AABB-based neighbor lists (STK and ArborX).
///
/// Each neighbor list expresses, as a concept, the `component::view_t` it expects: the build packs that volume
/// directly into the backend's search structure with no primitive conversion. The STK and ArborX lists currently
/// expect an `AABB`; a future search-sphere mode would add a sibling concept expecting a `Sphere`.
template <typename T>
concept AABBSearchInputType =
    NeighborListInputType<T> && mundy::is_aabb_v<std::remove_cvref_t<typename T::component_type::view_t>>;

/// \concept AABBSearchInputTypeFor
/// \brief An `AABBSearchInputType` whose component yields an AABB in exactly `Scalar` precision.
///
/// Refines `AABBSearchInputType` by fixing the broad-phase volume's scalar type, so a consumer with a fixed working
/// precision can require its input to match and reject a mismatched-scalar component at compile time.
template <typename T, typename Scalar>
concept AABBSearchInputTypeFor =
    AABBSearchInputType<T> &&
    std::same_as<typename std::remove_cvref_t<typename T::component_type::view_t>::value_type, Scalar>;

/// \concept PeriodicAABBSearchInputType
/// \brief An `AABBSearchInputType` that also carries a periodicity metric (`PeriodicSearchInput`).
///
/// Periodic neighbor-list builds use the metric to wrap each owner's geometry into the primary cell and to stamp
/// lattice-shifted source images. The component still yields the broad-phase volume (an AABB), translated per image.
template <typename T>
concept PeriodicAABBSearchInputType = AABBSearchInputType<T> && requires(const T& input) {
  input.periodic_metric();
  typename T::metric_type;
};

/// \struct NeighborListBuildTraits
/// \brief Traits that couple a concrete neighbor-list type to its build logic and type-specific parameters.
///
/// Specialize this struct for each concrete neighbor-list type. A valid specialization must provide:
///   - `target_input_type` — expected `TargetInput` type for the builder.
///   - `source_input_type` — expected `SourceInput` type for the builder.
///   - `args_type` — a struct holding any build-specific parameters (empty struct if none are needed).
///   - `static list_type build(const Builder&, const stk::mesh::BulkData&, const args_type&)` — a static
///     function template that builds the list from a complete `NeighborListBuilder`, `BulkData`, and args.
///
/// The primary template leaves `build()` undefined so that accessing it on an unspecialized type is a
/// compile error. The `struct args_type {}` in the primary allows function signatures in `NeighborListBuilder`
/// to compile even before a specialization is visible.
/// \tparam ListType Concrete neighbor-list type being described.
template <typename ListType>
struct NeighborListBuildTraits {
  /// \brief Default empty args; specializations override this with build-specific parameters.
  struct args_type {};
};

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_NEIGHBORLISTBUILDTRAITS_HPP_
