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

/// \file UnitTestStorage.cpp
/// \brief Unit tests for mundy::storage and mundy::store.
///
/// Test structure:
///   Group 0 — Compile-time contracts. The normalization policy, stored_type/value_type, get() return types
///     including const propagation, store()/CTAD agreement, and constructibility. These are type-level facts, so
///     they are file-scope static_asserts rather than tests.
///   Group 1 — Aliasing: every storage that refers to something must write through to the referent.
///   Group 2 — Ownership: a by-value storage must keep its value alive past the source's scope.
///   Group 3 — Construction cost: a by-value storage moves from an rvalue and copies from an lvalue or const rvalue.
///   Group 4 — Move-only values: owned from an rvalue, referenced from an lvalue.

// External
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// C++ core
#include <concepts>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_utils/reference_wrapper.hpp>
#include <mundy_utils/storage.hpp>

namespace mundy {

namespace {

// =============================================================================
// Test types
// =============================================================================

/// \brief Counts construction traffic and live instances so cost and lifetime are both observable.
struct Tracked {
  static inline int live = 0;
  static inline int copies = 0;
  static inline int moves = 0;

  int value = 0;

  explicit Tracked(int value_in) : value(value_in) {
    ++live;
  }

  Tracked(const Tracked& other) : value(other.value) {
    ++live;
    ++copies;
  }

  Tracked(Tracked&& other) noexcept : value(other.value) {
    ++live;
    ++moves;
    other.value = -1;
  }

  ~Tracked() {
    --live;
  }

  static void reset() {
    copies = 0;
    moves = 0;
  }
};

struct MoveOnly {
  int value = 0;

  explicit MoveOnly(int value_in) : value(value_in) {
  }

  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
};

template <class T>
concept can_store = requires(T&& t) { ::mundy::store(static_cast<T&&>(t)); };

template <class T>
concept can_own = requires(T&& t) { ::mundy::own(static_cast<T&&>(t)); };

using wrapper_t = ::mundy::reference_wrapper<int>;

// =============================================================================
// Compile-time tests
// =============================================================================

// Normalization policy: what an input type is stored as.
static_assert(std::is_same_v<impl::storage_type_t<int>, int>, "A value should be stored by value.");
static_assert(std::is_same_v<impl::storage_type_t<int&>, wrapper_t>,
              "An lvalue reference should be stored as reference_wrapper<T>.");
static_assert(std::is_same_v<impl::storage_type_t<int*>, int*>, "A pointer should be stored as a pointer.");
static_assert(std::is_same_v<impl::storage_type_t<int*&>, int*>,
              "A pointer lvalue should be stored as a pointer, not as a reference to one.");
static_assert(std::is_same_v<impl::storage_type_t<const int* const&>, const int*>,
              "Pointer cv/ref should normalize away while pointee const survives.");
static_assert(std::is_same_v<impl::storage_type_t<int[4]>, int*>,
              "An array cannot be stored by value, so it should decay to a pointer.");
static_assert(std::is_same_v<impl::storage_type_t<int (&)[4]>, int*>,
              "An array lvalue should decay rather than become a reference to an array.");
static_assert(std::is_same_v<impl::storage_type_t<const int (&)[4]>, const int*>,
              "Array decay should keep element const.");
static_assert(std::is_same_v<impl::storage_type_t<wrapper_t&>, wrapper_t>,
              "A reference_wrapper should be stored as the wrapper itself.");
static_assert(std::is_same_v<impl::storage_type_t<storage<int&>&>, typename storage<int&>::stored_type>,
              "A storage input should collapse to that storage's stored_type rather than nesting.");

// The normalization is idempotent, so the stored types above are exactly its fixed points.
static_assert(std::is_same_v<impl::storage_type_t<wrapper_t>, wrapper_t>, "reference_wrapper<T> is a fixed point.");
static_assert(std::is_same_v<impl::storage_type_t<int*>, int*>, "T* is a fixed point.");
static_assert(std::is_same_v<impl::storage_type_t<int>, int>, "A by-value type is a fixed point.");

// value_type names the element behind the storage, with cv/ref stripped.
static_assert(std::is_same_v<storage<int>::value_type, int>, "An owned int should have value_type int.");
static_assert(std::is_same_v<storage<int&>::value_type, int>, "A referenced int should have value_type int, not int&.");
static_assert(std::is_same_v<storage<const int&>::value_type, int>, "value_type should strip const.");
static_assert(std::is_same_v<storage<int*>::value_type, int*>, "A pointer's value_type is the pointer itself.");
static_assert(std::is_same_v<storage<wrapper_t>::value_type, int>, "A wrapper's value_type should unwrap to int.");

// get() hands back the referent, and const propagates only into owned values -- a view stays shallow-const.
static_assert(std::is_same_v<decltype(std::declval<storage<int>&>().get()), int&>, "Owned get() should be int&.");
static_assert(std::is_same_v<decltype(std::declval<const storage<int>&>().get()), const int&>,
              "Owned const get() should be const int&.");
static_assert(std::is_same_v<decltype(std::declval<storage<int&>&>().get()), int&>, "Referenced get() should be int&.");
static_assert(std::is_same_v<decltype(std::declval<const storage<int&>&>().get()), int&>,
              "A const reference storage is shallow-const, so get() stays int&.");
static_assert(std::is_same_v<decltype(std::declval<const storage<const int&>&>().get()), const int&>,
              "A reference to const must stay const.");
static_assert(std::is_same_v<decltype(std::declval<storage<int*>&>().get()), int*>,
              "A pointer storage should hand back the pointer by value.");
static_assert(std::is_same_v<decltype(std::declval<const storage<int*>&>().get()), int*>,
              "A const pointer storage is shallow-const, so get() stays int*.");
static_assert(std::is_same_v<decltype(std::declval<const storage<const int*>&>().get()), const int*>,
              "Pointee const must survive.");

// store() and CTAD must agree, including on a storage input.
static_assert(std::is_same_v<decltype(store(std::declval<int&>())), storage<int&>>, "store(lvalue) -> storage<T&>.");
static_assert(std::is_same_v<decltype(store(std::declval<int>())), storage<int>>, "store(rvalue) -> storage<T>.");
static_assert(std::is_same_v<decltype(storage(std::declval<int&>())), decltype(store(std::declval<int&>()))>,
              "CTAD should deduce what store() deduces for an lvalue.");
static_assert(std::is_same_v<decltype(storage(std::declval<int>())), decltype(store(std::declval<int>()))>,
              "CTAD should deduce what store() deduces for an rvalue.");
static_assert(std::is_same_v<decltype(store(std::declval<storage<int&>&>())), storage<int&>>,
              "store(storage<T>) should return that same storage type.");
static_assert(std::is_same_v<decltype(storage(std::declval<storage<int&>&>())), storage<int&>>,
              "CTAD on a storage input should not nest.");

// Constructibility. A by-value storage needs a copy from an lvalue but only a move from an rvalue, so a move-only
// value is storable by value only from an rvalue -- and is always referenceable.
static_assert(can_store<int&> && can_store<int> && can_store<int*&>, "store() should accept these.");
static_assert(std::constructible_from<storage<Tracked>, Tracked&>, "A copyable value may be stored from an lvalue.");
static_assert(std::constructible_from<storage<MoveOnly>, MoveOnly&&>,
              "A move-only value should be storable by value from an rvalue.");
static_assert(!std::constructible_from<storage<MoveOnly>, MoveOnly&>,
              "A move-only value must not be storable by value from an lvalue, which would need a copy.");
static_assert(std::is_same_v<typename decltype(store(std::declval<MoveOnly&>()))::stored_type,
                             ::mundy::reference_wrapper<MoveOnly>>,
              "A move-only lvalue should be referenced rather than copied.");
static_assert(!std::constructible_from<storage<int&>, int&&>,
              "A reference storage must not bind a temporary.");

// own() decays its argument to a prvalue of the value type, whatever the argument's reference-ness or constness.
static_assert(std::is_same_v<decltype(own(std::declval<int&>())), int>, "own(lvalue) should yield a prvalue.");
static_assert(std::is_same_v<decltype(own(std::declval<const int&>())), int>, "own should strip const.");
static_assert(std::is_same_v<decltype(own(std::declval<int>())), int>, "own of an rvalue is still a value.");
static_assert(std::is_same_v<decltype(own(std::declval<Tracked&>())), Tracked>,
              "own of a class lvalue should yield that class by value.");
static_assert(std::is_same_v<decltype(own(std::declval<const Tracked&>())), Tracked>,
              "own of a const class lvalue should yield a mutable value.");
static_assert(can_own<Tracked&> && can_own<Tracked>, "own should accept a copyable class either way.");
static_assert(can_own<MoveOnly>, "own of a move-only rvalue should move.");
static_assert(!can_own<MoveOnly&>, "own must reject a move-only lvalue rather than fail inside its body.");

// Consequently a by-value storage sees own(lvalue) exactly as it sees an rvalue.
static_assert(std::is_same_v<decltype(store(own(std::declval<Tracked&>()))), storage<Tracked>>,
              "store(own(lvalue)) should own by value, exactly as store(rvalue) does.");

// =============================================================================
// Runtime tests
// =============================================================================

TEST(Storage, Aliasing) {
  int value = 7;

  auto from_lvalue = store(value);
  from_lvalue.get() = 19;
  EXPECT_EQ(value, 19) << "Storage of an lvalue should write through.";

  auto from_wrapper = store(ref(value));
  from_wrapper.get() = 14;
  EXPECT_EQ(value, 14) << "Storage of a reference_wrapper should write through.";

  auto from_pointer = store(&value);
  EXPECT_EQ(from_pointer.get(), &value) << "Storage of a pointer should hand back that pointer.";
  *from_pointer.get() = 21;
  EXPECT_EQ(value, 21);

  auto nested = store(store(value));
  nested.get() = 33;
  EXPECT_EQ(value, 33) << "Storage of a storage should alias the original referent, not a copy.";

  const int const_value = 11;
  auto from_const_lvalue = store(const_value);
  EXPECT_EQ(&from_const_lvalue.get(), &const_value) << "Storage of a const lvalue should refer to it.";

  // Mutations of the referent must be visible after the fact, which is what makes this a view.
  value = 44;
  EXPECT_EQ(from_lvalue.get(), 44);
  EXPECT_EQ(nested.get(), 44);
}

TEST(Storage, Ownership) {
  const int live_before = Tracked::live;

  auto make_owned = [](int value) {
    Tracked local(value);
    return store(std::move(local));
  };

  {
    auto owned = make_owned(33);
    static_assert(std::is_same_v<decltype(owned), storage<Tracked>>, "store(std::move(local)) should own by value.");

    EXPECT_EQ(Tracked::live, live_before + 1) << "The owned value must outlive the local it came from.";
    EXPECT_EQ(owned.get().value, 33);

    owned.get().value = 44;
    EXPECT_EQ(owned.get().value, 44);
  }

  EXPECT_EQ(Tracked::live, live_before) << "Destroying the storage must destroy the owned value.";
}

TEST(Storage, ConstructionCost) {
  Tracked::reset();
  Tracked lvalue_source(7);
  storage<Tracked> from_lvalue(lvalue_source);
  EXPECT_EQ(Tracked::copies, 1) << "An lvalue can only be copied.";
  EXPECT_EQ(Tracked::moves, 0) << "An lvalue must never be moved out of.";
  EXPECT_EQ(lvalue_source.value, 7) << "The source must be left intact.";
  EXPECT_EQ(from_lvalue.get().value, 7);

  Tracked::reset();
  storage<Tracked> from_rvalue(Tracked(8));
  EXPECT_EQ(Tracked::moves, 1) << "An rvalue should be moved.";
  EXPECT_EQ(Tracked::copies, 0) << "An rvalue should not be copied.";
  EXPECT_EQ(from_rvalue.get().value, 8);

  Tracked::reset();
  const Tracked const_source(9);
  storage<Tracked> from_const_rvalue(std::move(const_source));
  EXPECT_EQ(Tracked::copies, 1) << "A const rvalue is not move eligible, so it should be copied.";
  EXPECT_EQ(Tracked::moves, 0) << "A const rvalue must not be moved out of.";
  EXPECT_EQ(from_const_rvalue.get().value, 9);

  Tracked::reset();
  Tracked store_source(5);
  auto via_store = store(std::move(store_source));
  EXPECT_EQ(Tracked::moves, 1) << "store(rvalue) should move exactly once.";
  EXPECT_EQ(Tracked::copies, 0) << "store(rvalue) should not copy.";
  EXPECT_EQ(via_store.get().value, 5);

  Tracked::reset();
  Tracked ref_source(6);
  auto referenced = store(ref_source);
  EXPECT_EQ(Tracked::copies, 0) << "Referencing an lvalue should not construct anything.";
  EXPECT_EQ(Tracked::moves, 0);
  EXPECT_EQ(&referenced.get(), &ref_source);

  // own() buys a by-value storage from an lvalue without spending the source: one copy out to the prvalue, then one
  // move into the storage.
  Tracked::reset();
  Tracked own_source(11);
  auto from_own = store(own(own_source));
  EXPECT_EQ(Tracked::copies, 1) << "own(lvalue) should copy exactly once.";
  EXPECT_EQ(Tracked::moves, 1) << "The resulting prvalue should be moved into the storage.";
  EXPECT_EQ(own_source.value, 11) << "own must leave the source intact.";
  EXPECT_EQ(from_own.get().value, 11);

  Tracked::reset();
  Tracked own_move_source(12);
  auto from_own_move = store(own(std::move(own_move_source)));
  EXPECT_EQ(Tracked::copies, 0) << "own(rvalue) should not copy.";
  EXPECT_EQ(Tracked::moves, 2) << "own(rvalue) should move out and then into the storage.";
  EXPECT_EQ(from_own_move.get().value, 12);
}

TEST(Storage, MoveOnly) {
  auto owned = store(MoveOnly(21));
  static_assert(std::is_same_v<typename decltype(owned)::stored_type, MoveOnly>,
                "A move-only rvalue should be owned by value.");
  EXPECT_EQ(owned.get().value, 21);

  MoveOnly source(33);
  auto referenced = store(source);
  EXPECT_EQ(&referenced.get(), &source) << "A move-only lvalue should be referenced.";
  EXPECT_EQ(referenced.get().value, 33);

  referenced.get().value = 34;
  EXPECT_EQ(source.value, 34) << "The reference should write through.";
}

}  // namespace

}  // namespace mundy
