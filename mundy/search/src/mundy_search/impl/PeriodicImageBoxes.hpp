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

#ifndef MUNDY_SEARCH_IMPL_PERIODICIMAGEBOXES_HPP_
#define MUNDY_SEARCH_IMPL_PERIODICIMAGEBOXES_HPP_

/// \file impl/PeriodicImageBoxes.hpp
/// \brief Backend-neutral periodic image generation: produces imaged geometry (AABBs) + per-object image shifts.

// C++ core
#include <cstddef>  // for size_t
#include <limits>   // for std::numeric_limits

// Trilinos
#include <Kokkos_Core.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>  // for stk::mesh::get_updated_ngp_mesh
#include <stk_mesh/base/NgpMesh.hpp>     // for stk::mesh::NgpMesh::fast_mesh_index
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>  // for stk::mesh::EntityRank

// Mundy
#include <mundy_geom/periodicity.hpp>       // for mundy::image_index, lattice_displacement, reference_point
#include <mundy_geom/primitives/AABB.hpp>   // for mundy::AABB + AABB::cast
#include <mundy_geom/primitives/Point.hpp>  // for mundy::Point
#include <mundy_geom/transform.hpp>         // for mundy::translate
#include <mundy_math/Vector3.hpp>           // for mundy::Vector3
#include <mundy_mesh/EntityIndices.hpp>     // for mundy::mesh::get_local_entities
#include <mundy_mesh/FieldComponent.hpp>    // for mundy::mesh::get_updated_ngp_component

namespace mundy {

namespace search {

namespace impl {

// ---------------------------------------------------------------------------
// Periodic image generation from a component input (backend-neutral)
// ---------------------------------------------------------------------------
//
// A periodic image is the owner geometry rigidly translated so its reference point (the AABB's min corner) is wrapped
// into the primary cell (targets) and optionally moved to a lattice-neighbour cell (sources).  The translation is the
// stored image shift: the displacement from the owner's *original* reference point to its imaged reference point.
// Targets get exactly one image (the wrapped owner); sources get one per periodic lattice neighbour (≤3^d for d
// periodic axes), each pruned against the union target bounding box via mundy::intersects on the geom AABB.  Source
// generation uses a device count → scan → fill so only the surviving images are allocated (never the full 3^d·N).
//
// This producer is independent of any search backend: it yields imaged geom AABBs plus per-object metadata.  A backend
// (ArborX, STK) packs `aabbs` into its own box type.  The image shift is `Σ (nbᵢ - kᵢ)·aᵢ`, where `aᵢ` are the lattice
// vectors, `kᵢ` is the owner's wrap image (mundy::image_index), and `nbᵢ ∈ {-1,0,1}` is the lattice-neighbour offset (0
// on a non-periodic axis); it is reconstructed exactly through the metric's fractional transforms (mundy::
// lattice_displacement), correct for orthorhombic and tilted cells alike.  All metric math runs in the metric's scalar;
// only the stored shift is cast to the image-shift scalar.

/// \brief Per-axis lattice-neighbour bound: 1 on a periodic axis (offsets {-1,0,1}), 0 otherwise (offset {0}).
template <typename Metric>
KOKKOS_INLINE_FUNCTION mundy::Vector3<int> lattice_neighbour_bound(const Metric& metric) {
  return mundy::Vector3<int>{metric.is_periodic(0) ? 1 : 0,  //
                             metric.is_periodic(1) ? 1 : 0,  //
                             metric.is_periodic(2) ? 1 : 0};
}

/// \struct PeriodicImages
/// \brief A set of periodic images as imaged geom AABBs plus per-object metadata, ready for a backend to pack.
template <typename MScalar, typename ShiftScalar, typename MemSpace>
struct PeriodicImages {
  //! Imaged geom AABB per image (owner geometry translated by its image shift).
  Kokkos::View<mundy::AABB<MScalar>*, MemSpace> aabbs;
  //! Owner entities indexed by dense owner ordinal.
  Kokkos::View<stk::mesh::Entity*, MemSpace> owner_entities;
  //! Owner ordinal for each image.
  Kokkos::View<size_t*, MemSpace> owner_indices;
  //! Image shift (original → imaged reference point) for each image.
  Kokkos::View<mundy::Vector3<ShiftScalar>*, MemSpace> shifts;
};

/// \brief Build periodic *target* images: one per owner, the owner rigidly wrapped into the primary cell.
template <typename ShiftScalar, typename ExecSpace, typename Component, typename Metric>
PeriodicImages<typename Metric::value_type, ShiftScalar, typename ExecSpace::memory_space> make_periodic_target_images(
    const stk::mesh::BulkData& bulk_data, const ExecSpace& exec, stk::mesh::EntityRank rank,
    const stk::mesh::Selector& selector, Component& component, const Metric& metric) {
  using memory_space = typename ExecSpace::memory_space;
  using mscalar = typename Metric::value_type;
  using shift_type = mundy::Vector3<ShiftScalar>;

  auto entities_ngp = mundy::mesh::get_local_entities(bulk_data, rank, selector, exec);
  entities_ngp.sync_to_device();
  component.sync_to_device();
  Kokkos::View<stk::mesh::Entity*, memory_space> owners = entities_ngp.view_device();
  auto ngp_component = mundy::mesh::get_updated_ngp_component(component);
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  const size_t num_owners = owners.extent(0);

  Kokkos::View<mundy::AABB<mscalar>*, memory_space> aabbs(  //
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_tgt_aabbs"), num_owners);
  Kokkos::View<size_t*, memory_space> owner_indices(  //
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_tgt_oi"), num_owners);
  Kokkos::View<shift_type*, memory_space> shifts(  //
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_tgt_shifts"), num_owners);
  Kokkos::parallel_for(
      "mundy_make_periodic_target_images", Kokkos::RangePolicy<ExecSpace>(exec, 0, num_owners),
      KOKKOS_LAMBDA(const size_t i) {
        const auto aabb = ngp_component(ngp_mesh.fast_mesh_index(owners(i))).template cast<mscalar>();
        const auto k = mundy::image_index(mundy::reference_point(aabb), metric);
        const auto shift = mundy::lattice_displacement(mundy::Vector3<int>{-k[0], -k[1], -k[2]}, metric);
        aabbs(i) = mundy::translate(aabb, shift);
        owner_indices(i) = i;
        shifts(i) = shift_type{static_cast<ShiftScalar>(shift[0]), static_cast<ShiftScalar>(shift[1]),
                               static_cast<ShiftScalar>(shift[2])};
      });

  return {aabbs, owners, owner_indices, shifts};
}

/// \brief Union AABB (in float) of a set of periodic images; the prune target for source generation.
///
/// Over an empty image set the result is a degenerate (inside-out) AABB that intersects nothing.
template <typename ExecSpace, typename MScalar, typename ShiftScalar, typename MemSpace>
mundy::AABB<float> periodic_images_bounding_box(const ExecSpace& exec,
                                                const PeriodicImages<MScalar, ShiftScalar, MemSpace>& images) {
  const auto aabbs = images.aabbs;
  const size_t n = aabbs.extent(0);
  constexpr float kPosInf = std::numeric_limits<float>::max();
  float lo0 = kPosInf, lo1 = kPosInf, lo2 = kPosInf;
  float hi0 = -kPosInf, hi1 = -kPosInf, hi2 = -kPosInf;
  Kokkos::parallel_reduce(
      "mundy_periodic_images_bbox", Kokkos::RangePolicy<ExecSpace>(exec, 0, n),
      KOKKOS_LAMBDA(const size_t i, float& l0, float& l1, float& l2, float& h0, float& h1, float& h2) {
        const auto& lo = aabbs(i).min_corner();
        const auto& hi = aabbs(i).max_corner();
        l0 = Kokkos::min(l0, static_cast<float>(lo[0]));
        l1 = Kokkos::min(l1, static_cast<float>(lo[1]));
        l2 = Kokkos::min(l2, static_cast<float>(lo[2]));
        h0 = Kokkos::max(h0, static_cast<float>(hi[0]));
        h1 = Kokkos::max(h1, static_cast<float>(hi[1]));
        h2 = Kokkos::max(h2, static_cast<float>(hi[2]));
      },
      Kokkos::Min<float>(lo0), Kokkos::Min<float>(lo1), Kokkos::Min<float>(lo2), Kokkos::Max<float>(hi0),
      Kokkos::Max<float>(hi1), Kokkos::Max<float>(hi2));
  return mundy::AABB<float>{mundy::Point<float>{lo0, lo1, lo2}, mundy::Point<float>{hi0, hi1, hi2}};
}

/// \brief Build periodic *source* images: one per periodic lattice neighbour per owner, pruned by `target_bbox`.
///
/// A device count → exclusive scan → fill allocates only the survivors (never the full 3^d·N).
template <typename ShiftScalar, typename ExecSpace, typename Component, typename Metric, typename TargetBBox>
PeriodicImages<typename Metric::value_type, ShiftScalar, typename ExecSpace::memory_space> make_periodic_source_images(
    const stk::mesh::BulkData& bulk_data, const ExecSpace& exec, stk::mesh::EntityRank rank,
    const stk::mesh::Selector& selector, Component& component, const Metric& metric, const TargetBBox& target_bbox) {
  using memory_space = typename ExecSpace::memory_space;
  using mscalar = typename Metric::value_type;
  using shift_type = mundy::Vector3<ShiftScalar>;

  auto entities_ngp = mundy::mesh::get_local_entities(bulk_data, rank, selector, exec);
  entities_ngp.sync_to_device();
  component.sync_to_device();
  Kokkos::View<stk::mesh::Entity*, memory_space> owners = entities_ngp.view_device();
  auto ngp_component = mundy::mesh::get_updated_ngp_component(component);
  auto ngp_mesh = stk::mesh::get_updated_ngp_mesh(bulk_data);
  const size_t num_owners = owners.extent(0);
  const mundy::Vector3<int> nb_bound = lattice_neighbour_bound(metric);

  // Count pass: surviving images per owner.
  Kokkos::View<size_t*, memory_space> counts(Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_src_counts"),
                                             num_owners);
  Kokkos::parallel_for(
      "mundy_make_periodic_source_count", Kokkos::RangePolicy<ExecSpace>(exec, 0, num_owners),
      KOKKOS_LAMBDA(const size_t i) {
        const auto aabb = ngp_component(ngp_mesh.fast_mesh_index(owners(i))).template cast<mscalar>();
        const auto k = mundy::image_index(mundy::reference_point(aabb), metric);
        size_t survivors = 0;
        for (int nx = -nb_bound[0]; nx <= nb_bound[0]; ++nx) {
          for (int ny = -nb_bound[1]; ny <= nb_bound[1]; ++ny) {
            for (int nz = -nb_bound[2]; nz <= nb_bound[2]; ++nz) {
              const auto shift =
                  mundy::lattice_displacement(mundy::Vector3<int>{nx - k[0], ny - k[1], nz - k[2]}, metric);
              if (mundy::intersects(mundy::translate(aabb, shift), target_bbox)) {
                ++survivors;
              }
            }
          }
        }
        counts(i) = survivors;
      });

  // Exclusive prefix scan → per-owner image offsets; offsets(num_owners) = total surviving images.
  Kokkos::View<size_t*, memory_space> offsets(Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_src_offsets"),
                                              num_owners + 1);
  Kokkos::parallel_scan(
      "mundy_make_periodic_source_scan", Kokkos::RangePolicy<ExecSpace>(exec, 0, num_owners + 1),
      KOKKOS_LAMBDA(const size_t i, size_t& update, const bool final_pass) {
        if (final_pass) offsets(i) = update;
        update += (i < num_owners) ? counts(i) : size_t(0);
      });
  size_t num_images = 0;
  Kokkos::deep_copy(num_images, Kokkos::subview(offsets, num_owners));

  // Fill pass: each owner writes its survivors into its contiguous slice [offsets(i), offsets(i+1)).
  Kokkos::View<mundy::AABB<mscalar>*, memory_space> aabbs(  //
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_src_aabbs"), num_images);
  Kokkos::View<size_t*, memory_space> owner_indices(  //
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_src_oi"), num_images);
  Kokkos::View<shift_type*, memory_space> shifts(  //
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "per_src_shifts"), num_images);
  Kokkos::parallel_for(
      "mundy_make_periodic_source_fill", Kokkos::RangePolicy<ExecSpace>(exec, 0, num_owners),
      KOKKOS_LAMBDA(const size_t i) {
        const auto aabb = ngp_component(ngp_mesh.fast_mesh_index(owners(i))).template cast<mscalar>();
        const auto k = mundy::image_index(mundy::reference_point(aabb), metric);
        size_t w = offsets(i);
        for (int nx = -nb_bound[0]; nx <= nb_bound[0]; ++nx) {
          for (int ny = -nb_bound[1]; ny <= nb_bound[1]; ++ny) {
            for (int nz = -nb_bound[2]; nz <= nb_bound[2]; ++nz) {
              const auto shift =
                  mundy::lattice_displacement(mundy::Vector3<int>{nx - k[0], ny - k[1], nz - k[2]}, metric);
              const auto image_aabb = mundy::translate(aabb, shift);
              if (mundy::intersects(image_aabb, target_bbox)) {
                aabbs(w) = image_aabb;
                owner_indices(w) = i;
                shifts(w) = shift_type{static_cast<ShiftScalar>(shift[0]), static_cast<ShiftScalar>(shift[1]),
                                       static_cast<ShiftScalar>(shift[2])};
                ++w;
              }
            }
          }
        }
      });

  return {aabbs, owners, owner_indices, shifts};
}

}  // namespace impl

}  // namespace search

}  // namespace mundy

#endif  // MUNDY_SEARCH_IMPL_PERIODICIMAGEBOXES_HPP_
