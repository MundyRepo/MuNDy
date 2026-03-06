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

// External
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// C++ core
#include <type_traits>

// Mundy
#include <mundy_core/reference_wrapper.hpp>
#include <mundy_core/storage.hpp>

namespace mundy {

namespace core {

namespace {

template <class T>
concept can_store = requires(T&& t) { ::mundy::core::store(static_cast<T&&>(t)); };

struct LifetimeTracked {
  static inline int live_count = 0;

  int value = 0;

  explicit LifetimeTracked(int value_in) : value(value_in) {
    ++live_count;
  }

  LifetimeTracked(const LifetimeTracked& other) : value(other.value) {
    ++live_count;
  }

  LifetimeTracked(LifetimeTracked&& other) noexcept : value(other.value) {
    ++live_count;
    other.value = -1;
  }

  ~LifetimeTracked() {
    --live_count;
  }
};

auto make_owned_tracked_storage(int value) {
  LifetimeTracked local(value);
  return ::mundy::core::store(std::move(local));
}

TEST(StorageTest, StoreLvalueUsesReferenceWrapperStorage) {
  int value = 7;
  auto stash = ::mundy::core::store(value);

  static_assert(std::is_same_v<decltype(stash), ::mundy::core::storage<int&>>,
                "store(lvalue) should produce storage<T&>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, ::mundy::core::reference_wrapper<int>>,
                "storage<int&> should store reference_wrapper<int>");
  static_assert(std::is_same_v<decltype(stash.get()), int&>, "get() should return int& for lvalue storage");

  stash.get() = 19;
  EXPECT_EQ(value, 19);
}

TEST(StorageTest, StoreConstLvalueKeepsConstReferenceSemantics) {
  const int value = 11;
  auto stash = ::mundy::core::store(value);

  static_assert(std::is_same_v<decltype(stash), ::mundy::core::storage<const int&>>,
                "store(const lvalue) should produce storage<const T&>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, ::mundy::core::reference_wrapper<const int>>,
                "const lvalue should store reference_wrapper<const T>");
  static_assert(std::is_same_v<decltype(stash.get()), const int&>, "get() should return const int&");

  EXPECT_EQ(stash.get(), 11);
}

TEST(StorageTest, StoreRvalueOwnsValue) {
  auto stash = ::mundy::core::store(42);

  static_assert(std::is_same_v<decltype(stash), ::mundy::core::storage<int>>,
                "store(rvalue) should produce storage<T>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, int>, "storage<int> should store int");
  static_assert(std::is_same_v<decltype(stash.get()), int&>, "mutable get() should return int&");

  EXPECT_EQ(stash.get(), 42);
  stash.get() = 55;
  EXPECT_EQ(stash.get(), 55);
}

TEST(StorageTest, StorePointerStoresPointerAndGetReturnsPointer) {
  int value = 9;
  int* pointer = &value;
  const int* const_pointer = &value;

  auto stash = ::mundy::core::store(pointer);
  auto const_stash = ::mundy::core::store(const_pointer);

  static_assert(std::is_same_v<decltype(stash), ::mundy::core::storage<int*&>>,
                "store(pointer lvalue) should produce storage<T*&>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, int*>, "pointer storage should store raw pointer");
  static_assert(std::is_same_v<decltype(stash.get()), int*>, "get() should return int* for pointer storage");

  static_assert(std::is_same_v<typename decltype(const_stash)::stored_type, const int*>,
                "const pointee pointer should preserve pointee const");
  static_assert(std::is_same_v<decltype(const_stash.get()), const int*>, "get() should return const int*");

  EXPECT_EQ(stash.get(), &value);
  EXPECT_EQ(const_stash.get(), &value);
}

TEST(StorageTest, StoreReferenceWrapperKeepsWrapperTypeAndReference) {
  int value = 3;
  auto wrapped = ::mundy::core::ref(value);
  auto stash = ::mundy::core::store(wrapped);

  static_assert(std::is_same_v<decltype(stash), ::mundy::core::storage<::mundy::core::reference_wrapper<int>&>>,
                "store(reference_wrapper lvalue) should preserve wrapper semantics");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, ::mundy::core::reference_wrapper<int>>,
                "wrapper storage should directly store reference_wrapper<T>");
  static_assert(std::is_same_v<decltype(stash.get()), int&>, "get() should unwrap to T&");

  stash.get() = 14;
  EXPECT_EQ(value, 14);
}

TEST(StorageTest, StoreStorageReturnsSameType) {
  int value = 100;
  auto stash = ::mundy::core::store(value);
  auto again = ::mundy::core::store(stash);

  static_assert(std::is_same_v<decltype(again), decltype(stash)>, "store(storage<T>) should return storage<T>");

  again.get() = 101;
  EXPECT_EQ(value, 101);
}

TEST(StorageTest, ConstructibilityContracts) {
  static_assert(can_store<int&>, "store(lvalue) should be callable");
  static_assert(can_store<int>, "store(rvalue) should be callable");
  static_assert(can_store<int*&>, "store(pointer lvalue) should be callable");
}

TEST(StorageTest, RvalueStorageOwnsObjectAcrossSourceScopeExit) {
  EXPECT_EQ(LifetimeTracked::live_count, 0);

  {
    auto stash = make_owned_tracked_storage(33);

    static_assert(std::is_same_v<decltype(stash), ::mundy::core::storage<LifetimeTracked>>,
                  "store(std::move(local)) should own LifetimeTracked by value");
    EXPECT_EQ(LifetimeTracked::live_count, 1);
    EXPECT_EQ(stash.get().value, 33);

    stash.get().value = 44;
    EXPECT_EQ(stash.get().value, 44);
  }

  EXPECT_EQ(LifetimeTracked::live_count, 0);
}

TEST(StorageTest, ReferenceStorageIsSafeWhenReferentOutlivesStorage) {
  int owner = 10;
  ::mundy::core::storage<int&> stash = ::mundy::core::store(owner);

  {
    int& alias = stash.get();
    alias = 17;
  }

  EXPECT_EQ(owner, 17);
  owner = 22;
  EXPECT_EQ(stash.get(), 22);
}

}  // namespace

}  // namespace core

}  // namespace mundy