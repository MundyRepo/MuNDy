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
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <iostream>
#include <stdexcept>  // for logic_error, invalid_argument, etc
#include <vector>

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize

// Mundy libs
#include <mundy_core/NgpPool.hpp>  // for NgpPool
#include <mundy_core/NgpView.hpp>  // for NgpView

namespace mundy {

namespace core {

namespace {

TEST(NgpPoolTest, DefaultConstruction) {
  // A default constructed pool is empty
  NgpPool<int> pool;
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity_host(), 0);
}

void run_simple_test() {
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity_host(), capacity);

  // Insert some values into the pool
  int value = 42;
  for (size_t i = 0; i < capacity; ++i) {
    pool.add_host(value);
    EXPECT_EQ(pool.size_host(), i + 1) << "Failed on iteration " << i;
    EXPECT_EQ(pool.capacity_host(), capacity);
  }
  pool.modify_on_host();

  // Acquire all the values from the pool
  // In the end, the pool should be empty, but it's capacity should be unchanged.
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire_host(), value) << "Failed on iteration " << i;
    EXPECT_EQ(pool.size_host(), capacity - i - 1) << "Size should decrement by 1";
    EXPECT_EQ(pool.capacity_host(), capacity) << "Capacity should be unchanged";
  }
}

TEST(NgpPoolTest, Simple) {
  run_simple_test();
}

void run_simple_host_device_test() {
  size_t capacity = 10;
  NgpPool<int> pool(capacity);

  constexpr bool is_pool_host_accessible =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, NgpPool<int>::memory_space>::accessible;

  if constexpr (is_pool_host_accessible) {
    // Host can access the device memory
    // Typically, we'll modify on host or device and then perform syncs. The following just affirms that the two are the
    // same.
    EXPECT_EQ(pool.size(), 0);
    EXPECT_EQ(pool.size_host(), 0);
    EXPECT_EQ(pool.capacity(), capacity);

    // Insert some values into the pool
    int value = 42;
    for (size_t i = 0; i < capacity; ++i) {
      pool.add(value);
      EXPECT_EQ(pool.size(), i + 1);
      EXPECT_EQ(pool.size_host(), i + 1);
    }

    // Acquire all the values from the pool
    // In the end, the pool should be empty, but it's capacity should be unchanged.
    for (size_t i = 0; i < capacity; ++i) {
      EXPECT_EQ(pool.acquire_host(), value) << "Failed on iteration " << i;
    }
    EXPECT_EQ(pool.size(), 0);
    EXPECT_EQ(pool.size_host(), 0);
    EXPECT_EQ(pool.capacity(), capacity);

    // If we push back a value, the size should increase.
    pool.add_host(value);
    EXPECT_EQ(pool.size(), 1);
    EXPECT_EQ(pool.size_host(), 1);
    EXPECT_EQ(pool.capacity(), capacity);
    EXPECT_EQ(pool.acquire(), value);
  } else {
    // Host and device differ
    EXPECT_EQ(pool.size_host(), 0);
    EXPECT_EQ(pool.capacity_host(), capacity);

    // Insert some values into the pool
    int value = 42;
    Kokkos::parallel_for("Fill pool", capacity, KOKKOS_LAMBDA(const size_t i) { pool.add(value); });
    Kokkos::fence();

    // Because we haven't synced the device to the host, the host data should be untouched.
    EXPECT_EQ(pool.size_host(), 0);

    pool.modify_on_device();
    pool.sync_to_host();
    EXPECT_EQ(pool.size_host(), capacity);

    // Acquire all the values from the pool on the host
    // In the end, the pool should be empty, but it's capacity should be unchanged.
    for (size_t i = 0; i < capacity; ++i) {
      EXPECT_EQ(pool.acquire_host(), value);
    }
    EXPECT_EQ(pool.size_host(), 0);
    EXPECT_EQ(pool.capacity_host(), capacity);

    // Perform the mod on the host
    for (size_t i = 0; i < capacity; ++i) {
      pool.add_host(value);
      EXPECT_EQ(pool.size_host(), i + 1);
    }

    // Acquire all the values from the pool
    for (size_t i = 0; i < capacity; ++i) {
      EXPECT_EQ(pool.acquire_host(), value);
    }
    EXPECT_EQ(pool.size_host(), 0);
    EXPECT_EQ(pool.capacity_host(), capacity);
    pool.modify_on_host();
    pool.sync_to_device();

    // If we push back a value, the size should increase.
    int value_to_push_back = 42;
    Kokkos::parallel_for("Fill pool", 1, KOKKOS_LAMBDA(const size_t i) { pool.add(value_to_push_back); });
    Kokkos::fence();
    EXPECT_EQ(pool.size_host(), 0) << "Host should be untouched";

    Kokkos::parallel_for(
        "Check pool", 1, KOKKOS_LAMBDA(const size_t i) {
          MUNDY_THROW_REQUIRE(pool.size() == 1, std::runtime_error, "Size should be 1");
          MUNDY_THROW_REQUIRE(pool.capacity() == capacity, std::runtime_error, "Capacity should be 10");
          MUNDY_THROW_REQUIRE(pool.acquire() == 42, std::runtime_error, "Acquire should be 42");
        });
    Kokkos::fence();

    pool.add_host(value_to_push_back);
    EXPECT_EQ(pool.size_host(), 1);
    EXPECT_EQ(pool.acquire_host(), 42);

    Kokkos::parallel_for(
        "Device should be untouched", 1, KOKKOS_LAMBDA(const size_t i) {
          MUNDY_THROW_REQUIRE(pool.size() == 0, std::runtime_error, "Size should be 0");
          MUNDY_THROW_REQUIRE(pool.capacity() == capacity, std::runtime_error, "Capacity should be 10");
        });

    // Perform a mod on host and mark/sync to device
    // Acquire has no guarantee of order, but we can check that the values are correct.
    pool.add_host(33);
    pool.add_host(44);
    pool.modify_on_host();
    pool.sync_to_device();
    EXPECT_EQ(pool.size_host(), 2);
    Kokkos::parallel_for(
        "Check pool", 1, KOKKOS_LAMBDA(const size_t i) {
          MUNDY_THROW_REQUIRE(pool.size() == 2, std::runtime_error, "Size should be 2");
          MUNDY_THROW_REQUIRE(pool.capacity() == capacity, std::runtime_error, "Capacity should be 10");
          auto first_value = pool.acquire();
          auto second_value = pool.acquire();
          MUNDY_THROW_REQUIRE((first_value == 33 && second_value == 44) || (first_value == 44 && second_value == 33),
                              std::runtime_error, "Acquire should be 33 and 44");
        });
    Kokkos::fence();
    EXPECT_EQ(pool.size_host(), 2);
    pool.modify_on_device();
    pool.sync_to_host();
    EXPECT_EQ(pool.size_host(), 0);
  }
}

TEST(NgpPoolTest, SimpleHostDevice) {
  run_simple_host_device_test();
}

TEST(NgpPoolTest, Reserve) {
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity_host(), capacity);

  // Insert some values into the pool
  int value = 42;
  for (size_t i = 0; i < capacity; ++i) {
    pool.add_host(value);
    EXPECT_EQ(pool.size_host(), i + 1);
  }
  pool.modify_on_host();

  // Reserve a smaller size
  pool.reserve(capacity - 2);
  EXPECT_EQ(pool.size_host(), capacity);
  EXPECT_EQ(pool.capacity_host(), capacity);

  // Reserve a larger size
  size_t new_capacity = 20;
  pool.reserve(new_capacity);
  EXPECT_EQ(pool.size_host(), capacity);
  EXPECT_EQ(pool.capacity_host(), new_capacity);

  // Acquire all the values from the pool
  for (size_t i = 0; i < capacity; ++i) {
    EXPECT_EQ(pool.acquire_host(), value) << "Failed on iteration " << i;
  }
  EXPECT_EQ(pool.size_host(), 0);
  EXPECT_EQ(pool.capacity_host(), new_capacity);
}

TEST(NgpPoolTest, OverUnderflow) {
  NgpPool<int> pool1;
  NgpPool<int> pool2;
  EXPECT_EQ(pool1.size_host(), 0);
  EXPECT_EQ(pool1.capacity_host(), 0);

#ifndef NDEBUG
  // Only throws in debug
  EXPECT_THROW(pool1.acquire_host(), std::runtime_error) << "Overfetching from the pool should throw in debug";
  EXPECT_THROW(pool2.add_host(42), std::runtime_error) << "Overfilling the pool should throw in debug";
#else
  // Should not throw in release allowing us to enter undefined behavior
  EXPECT_NO_THROW(pool1.acquire_host()) << "Overfetching from the pool should not throw in release";
  EXPECT_NO_THROW(pool2.add_host(42)) << "Overfilling the pool should not throw in release";
#endif
}

TEST(NgpPoolTest, BatchedAdditionsFetches) {
  int value = 42;
  size_t capacity = 10;
  NgpPool<int> pool(capacity);
  for (size_t i = 0; i < capacity; ++i) {
    pool.add_host(value);
  }
  pool.modify_on_host();

  // Acquire all values
  auto values = pool.batch_acquire(capacity);
  values.sync_to_host();
  EXPECT_EQ(values.extent(0), capacity);
  EXPECT_EQ(pool.size_host(), 0);
  for (size_t i = 0; i < values.extent(0); ++i) {
    EXPECT_EQ(values.view_host()(i), value) << "Failed on iteration " << i;
    values.view_host()(i) = 33;
  }
  values.modify_on_host();
  pool.modify_on_host();

  // Add the values back
  pool.batch_add(values);
  pool.sync_to_host();
  EXPECT_EQ(pool.size_host(), 10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(pool.acquire_host(), 33) << "Failed on iteration " << i;
  }
  EXPECT_EQ(pool.size_host(), 0);

  // Acquire a subset of the values
  pool.batch_add(values);
  pool.sync_to_host();
  auto values2 = pool.batch_acquire(capacity - 4);
  values2.sync_to_host();
  EXPECT_EQ(values2.extent(0), capacity - 4);
  for (size_t i = 0; i < values2.extent(0); ++i) {
    EXPECT_EQ(values2.view_host()(i), 33);
  }
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(pool.acquire_host(), 33);
  }
}

}  // namespace

}  // namespace core

}  // namespace mundy
