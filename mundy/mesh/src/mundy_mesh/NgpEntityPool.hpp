// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
// 
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is empty software: you can redistribute it and/or modify it under the terms of the GNU General Public License
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

#ifndef MUNDY_MESH_NGPENTITYPOOL_HPP_
#define MUNDY_MESH_NGPENTITYPOOL_HPP_

// C++ core
#include <stdexcept>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_core/NgpPool.hpp>
#include <mundy_core/NgpView.hpp>
#include <mundy_core/throw_assert.hpp>

namespace mundy {

namespace mesh {

/// \brief NgpEntityPoolT is a Kokkos-compatible pool of stk::mesh::Entity objects.
///
/// Unlike a regular NgpPool, we offer a reserve and declare method to both reserve space within the pool
/// and fill that space with entities from a given mesh.
template <typename MemorySpace, typename SizeType = long int>
class NgpEntityPoolT : public core::NgpPoolT<stk::mesh::Entity, MemorySpace, SizeType> {
 public:
  // Type aliases
  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using entity_vector_t = std::vector<stk::mesh::Entity>;
  using entity_view_t = core::NgpViewT<stk::mesh::Entity*, MemorySpace>;
  using our_size_t = SizeType;
  using base_t = core::NgpPoolT<stk::mesh::Entity, MemorySpace, SizeType>;

  // No default constructor. How would you set the rank?
  NgpEntityPoolT() = delete;

  // Constructor for empty pool
  NgpEntityPoolT(stk::mesh::BulkData& bulk_data, stk::mesh::EntityRank rank) : base_t(0), bulk_data_(bulk_data), rank_(rank) {
  }

  NgpEntityPoolT(stk::mesh::BulkData& bulk_data, stk::mesh::EntityRank rank, size_t capacity)
      : base_t(capacity), bulk_data_(bulk_data), rank_(rank) {
  }

  ~NgpEntityPoolT() = default;

  // Default copy
  NgpEntityPoolT(const NgpEntityPoolT&) = default;
  NgpEntityPoolT& operator=(const NgpEntityPoolT&) = default;

  // Default move
  NgpEntityPoolT(NgpEntityPoolT&&) = default;
  NgpEntityPoolT& operator=(NgpEntityPoolT&&) = default;

  void reserve_and_declare(our_size_t requested_capacity) {
    MUNDY_THROW_ASSERT(
        bulk_data_.in_modifiable_state(), std::runtime_error,
        "Cannot reserve and declare entities in NgpEntityPoolT when the mesh is not in a modifiable state.");
    base_t::reserve(requested_capacity);

    if (requested_capacity > this->size()) {
      const our_size_t num_new_entities = requested_capacity - this->size();
      std::vector<size_t> requests(bulk_data_.mesh_meta_data().entity_rank_count(), 0);
      requests[rank_] = num_new_entities;

      std::vector<stk::mesh::Entity> requested_entities;
      bulk_data_.generate_new_entities(requests, requested_entities);

      base_t::batch_add_host(requested_entities);
      base_t::modify_on_host();
      base_t::sync_to_device();
    }
  }

 private:
  stk::mesh::BulkData& bulk_data_;
  const stk::mesh::EntityRank rank_;
};  // NgpEntityPoolT

/// \brief Our default NgpEntityPool type for use in Mundy.
///
/// Unlike NgpNgpEntityPoolT, we follow stk::ngp conventions by using stk::ngp::ExecSpace as our
/// chosen device space.
///
/// We chose to hard-code the default size type here to avoid users writing NgpEntityPool<>.
using NgpEntityPool = NgpEntityPoolT<typename stk::ngp::ExecSpace::memory_space, long int>;

}  // namespace mesh

}  // namespace mundy

#endif  // MUNDY_MESH_NGPENTITYPOOL_HPP_
