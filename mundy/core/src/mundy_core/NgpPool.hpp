// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
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

#ifndef MUNDY_CORE_NGPPOOL_HPP_
#define MUNDY_CORE_NGPPOOL_HPP_

// C++ core
#include <stdexcept>
#include <string>
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>

// STK
#include <stk_util/ngp/NgpSpaces.hpp>

// Mundy
#include <mundy_core/NgpView.hpp>
#include <mundy_core/throw_assert.hpp>

namespace mundy {

namespace core {

/// \brief NgpPoolT is a Kokkos-compatible pool of default constructable objects.
template <class DataType, typename MemorySpace, typename SizeType = long int>
class NgpPoolT {
 public:
  // Type aliases
  using memory_space = MemorySpace;
  using execution_space = typename MemorySpace::execution_space;
  using pool_vector_t = std::vector<DataType>;
  using pool_view_t = NgpViewT<DataType*, MemorySpace>;
  using value_type = DataType;
  using our_size_t = SizeType;

 private:
  // Both capacity and size are stored as views so that they are consistent between copies of this pool.
  using ngp_size_t = NgpViewT<our_size_t, MemorySpace>;
  mutable ngp_size_t ngp_size_;
  mutable ngp_size_t ngp_capacity_;
  mutable pool_view_t pool_;

 public:
  NgpPoolT() : NgpPoolT(0) {
  }

  NgpPoolT(size_t capacity)
      : ngp_size_("NgpPoolT::ngp_size"), ngp_capacity_("NgpPoolT::ngp_capacity"), pool_("NgpPoolT::pool", capacity) {
    // Initialize size/capacity
    ngp_size_.view_host()() = 0;
    ngp_capacity_.view_host()() = capacity;
    Kokkos::deep_copy(ngp_size_.view_device(), ngp_size_.view_host());
    Kokkos::deep_copy(ngp_capacity_.view_device(), ngp_capacity_.view_host());

    // Initialize the host and device pools to default constructed objects
    Kokkos::deep_copy(pool_.view_device(), DataType());
    Kokkos::deep_copy(pool_.view_host(), DataType());
  }

  ~NgpPoolT() = default;

  // Default copy
  NgpPoolT(const NgpPoolT&) = default;
  NgpPoolT& operator=(const NgpPoolT&) = default;

  // Default move
  NgpPoolT(NgpPoolT&&) = default;
  NgpPoolT& operator=(NgpPoolT&&) = default;

  //! \name Accessors
  //@{

  KOKKOS_FUNCTION
  our_size_t capacity() const {
    return ngp_capacity_.view_device()();
  }

  our_size_t capacity_host() const {
    return ngp_capacity_.view_host()();
  }

  KOKKOS_FUNCTION
  size_t size() const {
    return ngp_size_.view_device()();
  }

  size_t size_host() const {
    return ngp_size_.view_host()();
  }

  void reserve(our_size_t requested_capacity) {
    if (requested_capacity > ngp_capacity_.view_host()()) {
      pool_.sync_to_device();
      pool_.resize(requested_capacity);  // Increase capacity to requested_capacity
      ngp_capacity_.view_host()() = requested_capacity;
      Kokkos::deep_copy(ngp_capacity_.view_device(), ngp_capacity_.view_host());

      pool_.sync_to_host();  // Ensure the host view is updated
    }
  }

  /// \brief Acquire a value from the pool. Returns by move. Modifies on device but does not mark as modified.
  /// It's up to you to mark the pool as modified on the device after using this.
  ///
  /// Something to be aware of: This function is const because a KOKKOS_LAMBDA will const capture this class making
  /// acquire on the device impossible otherwise. This is why we mark size, capacity, and our internal pool as mutable.
  KOKKOS_INLINE_FUNCTION
  value_type acquire() const {
    // For those not familiar with atomic_fetch_sub, it returns the value before the subtraction.
    our_size_t old_size = Kokkos::atomic_fetch_sub(&ngp_size_.view_device()(), 1);
    if (old_size == 0) {
      MUNDY_THROW_ASSERT(false, std::runtime_error, "Attempting to acquire an object from an empty pool.");
      return value_type();
    }

    // Eat the last value in the pool and replace it with a default constructed value.
    value_type v = std::move(pool_.view_device()(old_size - 1));
    pool_.view_device()(old_size - 1) = value_type();
    return v;
  }

  /// \brief Acquire a value from the pool. Returns by move. Modifies on host but does not mark as modified.
  /// It's up to you to mark the pool as modified on the host after using this.
  value_type acquire_host() const {
    // For those not familiar with atomic_fetch_sub, it returns the value before the subtraction.
    our_size_t old_size = Kokkos::atomic_fetch_sub(&ngp_size_.view_host()(), 1);

    if (old_size - 1 < 0) {
      MUNDY_THROW_ASSERT(false, std::runtime_error, "Attempting to acquire an object from an empty pool.");
      return value_type();
    }

    // Eat the last value in the pool and replace it with a default constructed value.
    value_type v = std::move(pool_.view_host()(old_size - 1));
    pool_.view_host()(old_size - 1) = value_type();

    return v;
  }

  /// \brief Add an object into the pool. Modifies on device but does not mark as modified.
  KOKKOS_INLINE_FUNCTION
  void add(value_type p) const {
    our_size_t old_size = Kokkos::atomic_fetch_add(&ngp_size_.view_device()(), 1);
    if (old_size + 1 > ngp_capacity_.view_device()()) {
      MUNDY_THROW_ASSERT(false, std::runtime_error, "Attempting to add an object into a full pool.");
      return;
    }

    pool_.view_device()(old_size) = p;
  }

  /// \brief Add an object into the pool. Modifies on host but does not mark as modified.
  void add_host(value_type p) const {
    our_size_t old_size = Kokkos::atomic_fetch_add(&ngp_size_.view_host()(), 1);
    if (old_size + 1 > ngp_capacity_.view_host()()) {
      MUNDY_THROW_ASSERT(false, std::runtime_error, "Attempting to add an object into a full pool.");
      return;
    }

    pool_.view_host()(old_size) = p;
  }

  /// \brief Acquire N objects from the pool. Returns by move. Modifies on device and marks as modified.
  /// Results are returned in an NgpViewT<ValueType, MemorySpace>.
  pool_view_t batch_acquire(our_size_t n) {
    sync_to_device();
    size_t old_size = ngp_size_.view_host()();
    ngp_size_.view_host()() -= n;
    Kokkos::deep_copy(ngp_size_.view_device(), ngp_size_.view_host());

    MUNDY_THROW_ASSERT(old_size >= n, std::runtime_error,
                       "Attempting to acquire more objects than are available in the pool.");

    // Eat the last n values in the pool and replace them with a default constructed value.
    pool_view_t out("NgpPoolT::acquire", n);
    auto local_pool = pool_;

    using range_policy = stk::ngp::RangePolicy<execution_space>;
    Kokkos::parallel_for(
        range_policy(0, n), KOKKOS_LAMBDA(const our_size_t i) {
          out.view_device()(i) = local_pool.view_device()(old_size - i - 1);
          local_pool.view_device()(old_size - i - 1) = value_type();
        });
    Kokkos::fence();

    out.modify_on_device();
    modify_on_device();

    return out;
  }

  /// \brief Acquire N objects from the pool. Returns by move. Modifies on host and marks as modified.
  /// Results are returned in an std::vector<ValueType>.
  pool_vector_t batch_acquire_host(our_size_t n) {
    our_size_t old_size = Kokkos::atomic_fetch_sub(&ngp_size_.view_host()(), n);
    MUNDY_THROW_ASSERT(old_size - n >= 0, std::runtime_error,
                       "Attempting to acquire more objects than are available in the pool.");

    // Eat the last n values in the pool and replace them with a default constructed value.
    pool_vector_t out(n);
    for (our_size_t i = 0; i < n; ++i) {
      out[i] = std::move(pool_.view_host()(old_size - i - 1));
      pool_.view_host()(old_size - i - 1) = value_type();
    }

    modify_on_host();

    return out;
  }

  /// \brief Add N objects into the pool. Modifies on device and marks as modified.
  void batch_add(pool_view_t p) {
    sync_to_host();
    size_t old_size = ngp_size_.view_host()();
    ngp_size_.view_host()() += p.extent(0);
    Kokkos::deep_copy(ngp_size_.view_device(), ngp_size_.view_host());
    MUNDY_THROW_ASSERT(old_size + p.extent(0) <= ngp_capacity_.view_host()(), std::runtime_error,
                       "Released objects would exceed pool capacity.");

    auto local_pool = pool_;
    p.sync_to_device();  // Ensure the device view is up to date

    using range_policy = stk::ngp::RangePolicy<execution_space>;
    Kokkos::parallel_for(
        range_policy(0, p.extent(0)),
        KOKKOS_LAMBDA(const our_size_t i) { local_pool.view_device()(old_size + i) = p.view_device()(i); });

    modify_on_device();
  }

  /// \brief Add N objects into the pool. Modifies on host and marks as modified.
  void batch_add_host(pool_vector_t p) {
    our_size_t old_size = Kokkos::atomic_fetch_add(&ngp_size_.view_host()(), p.size());
    MUNDY_THROW_ASSERT(old_size + p.size() <= ngp_capacity_.view_host()(), std::runtime_error,
                       "Released objects would exceed pool capacity.");

    auto local_pool = pool_;

    using range_policy = stk::ngp::HostRangePolicy;
    Kokkos::parallel_for(
        range_policy(0, p.size()), KOKKOS_LAMBDA(const our_size_t i) { local_pool.view_host()(old_size + i) = p[i]; });

    modify_on_host();
  }
  //@}

  //! \name Ngp interface
  //@{

  /// \brief Mark the host pool as modified.
  ///
  /// Call this method after updating the host pool so that the NgpPoolT is aware
  /// that the device pool may now be out of date.
  inline void modify_on_host() {
    pool_.modify_on_host();
    ngp_size_.modify_on_host();
    ngp_capacity_.modify_on_host();
  }

  /// \brief Mark the device pool as modified.
  ///
  /// Call this method after updating the device pool so that the NgpPoolT is aware
  /// that the host pool may now be out of date.
  inline void modify_on_device() {
    pool_.modify_on_device();
    ngp_size_.modify_on_device();
    ngp_capacity_.modify_on_device();
  }

  /// \brief Abstract method for marking the pool as modified.
  template <typename Space>
  inline void modify_on() {
    pool_.template modify_on<Space>();
    ngp_size_.template modify_on<Space>();
    ngp_capacity_.template modify_on<Space>();
  }

  /// \brief Synchronize the host pool to the device pool if needed.
  ///
  /// If the device pool has been modified more recently than the host pool,
  /// this function performs a deep copy from the device pool to the host pool.
  inline void sync_to_host() {
    pool_.sync_to_host();
    ngp_size_.sync_to_host();
    ngp_capacity_.sync_to_host();
  }

  /// \brief Synchronize the device pool to the host pool if needed.
  ///
  /// If the host pool has been modified more recently than the device pool,
  /// this function performs a deep copy from the host pool to the device pool.
  inline void sync_to_device() {
    pool_.sync_to_device();
    ngp_size_.sync_to_device();
    ngp_capacity_.sync_to_device();
  }

  /// \brief Abstract method for synchronizing the pool.
  template <typename Space>
  inline void sync_to() {
    pool_.template sync_to<Space>();
    ngp_size_.template sync_to<Space>();
    ngp_capacity_.template sync_to<Space>();
  }

  /// \brief Return if we need to sync to the host.
  inline bool need_sync_to_host() const {
    return pool_.need_sync_to_host() || ngp_size_.need_sync_to_host() || ngp_capacity_.need_sync_to_host();
  }

  /// \brief Return if we need to sync to the device.
  inline bool need_sync_to_device() const {
    return pool_.need_sync_to_device() || ngp_size_.need_sync_to_device() || ngp_capacity_.need_sync_to_device();
  }

  /// \brief Abstract method for checking if we need to sync.
  template <typename Space>
  inline bool need_sync_to() const {
    return pool_.template need_sync_to<Space>() || ngp_size_.template need_sync_to<Space>() ||
           ngp_capacity_.template need_sync_to<Space>();
  }
  //@}
};  // NgpPoolT

/// \brief Our default NgpPool type for use in Mundy.
///
/// Unlike NgpPoolT, we follow stk::ngp conventions by using stk::ngp::ExecSpace as our
/// chosen device space.
template <class DataType, typename SizeType = long int>
using NgpPool = NgpPoolT<DataType, typename stk::ngp::ExecSpace::memory_space, SizeType>;

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_NGPPOOL_HPP_
