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
#include <mundy_utils/reference_wrapper.hpp>
#include <mundy_utils/storage.hpp>

namespace mundy {

namespace {

template <class T>
concept can_store = requires(T&& t) { ::mundy::store(static_cast<T&&>(t)); };

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
  return ::mundy::store(std::move(local));
}

TEST(StorageTest, StoreLvalueUsesReferenceWrapperStorage) {
  int value = 7;
  auto stash = ::mundy::store(value);

  static_assert(std::is_same_v<decltype(stash), ::mundy::storage<int&>>, "store(lvalue) should produce storage<T&>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, ::mundy::reference_wrapper<int>>,
                "storage<int&> should store reference_wrapper<int>");
  static_assert(std::is_same_v<decltype(stash.get()), int&>, "get() should return int& for lvalue storage");

  stash.get() = 19;
  EXPECT_EQ(value, 19);
}

TEST(StorageTest, StoreConstLvalueKeepsConstReferenceSemantics) {
  const int value = 11;
  auto stash = ::mundy::store(value);

  static_assert(std::is_same_v<decltype(stash), ::mundy::storage<const int&>>,
                "store(const lvalue) should produce storage<const T&>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, ::mundy::reference_wrapper<const int>>,
                "const lvalue should store reference_wrapper<const T>");
  static_assert(std::is_same_v<decltype(stash.get()), const int&>, "get() should return const int&");

  EXPECT_EQ(stash.get(), 11);
}

TEST(StorageTest, StoreRvalueOwnsValue) {
  auto stash = ::mundy::store(42);

  static_assert(std::is_same_v<decltype(stash), ::mundy::storage<int>>, "store(rvalue) should produce storage<T>");
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

  auto stash = ::mundy::store(pointer);
  auto const_stash = ::mundy::store(const_pointer);

  static_assert(std::is_same_v<decltype(stash), ::mundy::storage<int*&>>,
                "store(pointer lvalue) should produce storage<T*&>");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, int*>,
                "pointer storage should store raw pointer");
  static_assert(std::is_same_v<decltype(stash.get()), int*>, "get() should return int* for pointer storage");

  static_assert(std::is_same_v<typename decltype(const_stash)::stored_type, const int*>,
                "const pointee pointer should preserve pointee const");
  static_assert(std::is_same_v<decltype(const_stash.get()), const int*>, "get() should return const int*");

  EXPECT_EQ(stash.get(), &value);
  EXPECT_EQ(const_stash.get(), &value);
}

TEST(StorageTest, StoreReferenceWrapperKeepsWrapperTypeAndReference) {
  int value = 3;
  auto wrapped = ::mundy::ref(value);
  auto stash = ::mundy::store(wrapped);

  static_assert(std::is_same_v<decltype(stash), ::mundy::storage<::mundy::reference_wrapper<int>&>>,
                "store(reference_wrapper lvalue) should preserve wrapper semantics");
  static_assert(std::is_same_v<typename decltype(stash)::stored_type, ::mundy::reference_wrapper<int>>,
                "wrapper storage should directly store reference_wrapper<T>");
  static_assert(std::is_same_v<decltype(stash.get()), int&>, "get() should unwrap to T&");

  stash.get() = 14;
  EXPECT_EQ(value, 14);
}

TEST(StorageTest, StoreStorageReturnsSameType) {
  int value = 100;
  auto stash = ::mundy::store(value);
  auto again = ::mundy::store(stash);

  using stash_t = decltype(stash);
  using again_t = decltype(again);
  static_assert(std::is_same_v<stash_t, again_t>, "store(storage<T>) should return the same storage type");
  static_assert(std::is_same_v<typename stash_t::stored_type, typename again_t::stored_type>,
                "storage type should have the same stored_type");

  again.get() = 101;
  EXPECT_EQ(value, 101);
}

TEST(StorageTest, StorageTypeNormalizationRules) {
  using wrapper_t = ::mundy::reference_wrapper<int>;
  using storage_ref_t = ::mundy::storage<int&>;

  static_assert(std::is_same_v<::mundy::impl::storage_type_t<int>, int>, "value type should store as value");
  static_assert(std::is_same_v<::mundy::impl::storage_type_t<int&>, wrapper_t>,
                "lvalue reference should store as reference_wrapper<T>");
  static_assert(std::is_same_v<::mundy::impl::storage_type_t<int*>, int*>, "pointer should store as pointer");
  static_assert(std::is_same_v<::mundy::impl::storage_type_t<const int* const&>, const int*>,
                "pointer cv/ref should normalize to pointer with pointee cv preserved");
  static_assert(std::is_same_v<::mundy::impl::storage_type_t<wrapper_t&>, wrapper_t>,
                "reference_wrapper should store as the wrapper type itself");
  static_assert(std::is_same_v<::mundy::impl::storage_type_t<storage_ref_t&>, typename storage_ref_t::stored_type>,
                "storage<T> input should normalize to storage<T>::stored_type");
}

TEST(StorageTest, ValueTypeMatchesGetResultStrippedOfCvref) {
  using wrapper_t = ::mundy::reference_wrapper<int>;

  static_assert(std::is_same_v<::mundy::storage<int>::value_type, int>, "owned storage value_type should be int");
  static_assert(std::is_same_v<::mundy::storage<int&>::value_type, int>,
                "reference storage value_type should be int, not int&");
  static_assert(std::is_same_v<::mundy::storage<const int&>::value_type, int>,
                "const-reference storage value_type should strip const");
  static_assert(std::is_same_v<::mundy::storage<int*>::value_type, int*>,
                "pointer storage value_type should be the pointer type itself");
  static_assert(std::is_same_v<::mundy::storage<wrapper_t>::value_type, int>,
                "reference_wrapper storage value_type should unwrap to int");
}

TEST(StorageTest, DirectCTADAgreesWithStore) {
  int value = 5;

  auto via_store = ::mundy::store(value);
  ::mundy::storage via_ctad(value);
  static_assert(std::is_same_v<decltype(via_store), decltype(via_ctad)>,
                "storage(value) should deduce the same type as store(value)");

  auto owned_via_store = ::mundy::store(42);
  ::mundy::storage owned_via_ctad(42);
  static_assert(std::is_same_v<decltype(owned_via_store), decltype(owned_via_ctad)>,
                "storage(rvalue) should deduce the same type as store(rvalue)");

  auto restashed_via_store = ::mundy::store(via_store);
  ::mundy::storage restashed_via_ctad(via_ctad);
  static_assert(std::is_same_v<decltype(via_store), decltype(restashed_via_store)>,
                "storage(storage<T>) should deduce storage<T> itself, matching store()");
  static_assert(std::is_same_v<decltype(via_ctad), decltype(restashed_via_ctad)>,
                "storage(storage<T>) via direct CTAD should also deduce storage<T> itself");

  via_ctad.get() = 8;
  EXPECT_EQ(value, 8);
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

    static_assert(std::is_same_v<decltype(stash), ::mundy::storage<LifetimeTracked>>,
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
  ::mundy::storage<int&> stash = ::mundy::store(owner);

  {
    int& alias = stash.get();
    alias = 17;
  }

  EXPECT_EQ(owner, 17);
  owner = 22;
  EXPECT_EQ(stash.get(), 22);
}

}  // namespace

}  // namespace mundy