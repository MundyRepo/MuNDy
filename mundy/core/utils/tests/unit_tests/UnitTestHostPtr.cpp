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
#include <gtest/gtest.h>

// C++ core
#include <memory>         // for std::unique_ptr, std::make_unique
#include <set>            // for std::set
#include <unordered_set>  // for std::unordered_set
#include <utility>        // for std::in_place, std::move, std::pair

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/host_ptr.hpp>

namespace mundy {

namespace {

//! A type that counts its live instances, to verify host_ptr constructs/destroys the value exactly once.
struct LifetimeTracked {
  static inline int live_count = 0;
  int value = 0;

  explicit LifetimeTracked(int value_in) : value(value_in) {
    ++live_count;
  }
  LifetimeTracked(const LifetimeTracked& other) : value(other.value) {
    ++live_count;
  }
  ~LifetimeTracked() {
    --live_count;
  }
};

struct Base {
  virtual ~Base() = default;
  int b = 0;
};
struct Derived : Base {
  int d = 0;
};

// --- Construction & basic access ------------------------------------------------------------------------------------

TEST(HostPtr, EmptyByDefault) {
  host_ptr<int> p;
  EXPECT_FALSE(static_cast<bool>(p));
  EXPECT_EQ(p.get(), nullptr);
  EXPECT_EQ(p.use_count(), 0);
}

TEST(HostPtr, HoldsValue) {
  host_ptr<int> p(42);
  ASSERT_TRUE(static_cast<bool>(p));
  EXPECT_EQ(*p, 42);
  EXPECT_EQ(*p.get(), 42);  // get() returns a pointer, like std::shared_ptr
  *p = 7;
  EXPECT_EQ(*p, 7);
}

TEST(HostPtr, NullptrIsEmpty) {
  host_ptr<int> p = nullptr;
  EXPECT_FALSE(static_cast<bool>(p));
  EXPECT_TRUE(p == nullptr);
  EXPECT_TRUE(nullptr == p);
}

TEST(HostPtr, MakeHostPtr) {
  auto p = make_host_ptr<std::pair<int, int>>(3, 4);
  ASSERT_TRUE(static_cast<bool>(p));
  EXPECT_EQ(p->first, 3);
  EXPECT_EQ(p->second, 4);
}

TEST(HostPtr, InPlaceConstruction) {
  host_ptr<std::pair<int, int>> p(std::in_place, 3, 4);
  EXPECT_EQ(p->first, 3);
  EXPECT_EQ(p->second, 4);
}

// --- Ownership / lifetime --------------------------------------------------------------------------------------------

TEST(HostPtr, CopySharesOwnership) {
  host_ptr<int> a(10);
  host_ptr<int> b = a;
  EXPECT_EQ(&*a, &*b);  // identical storage, not a deep copy
  *a = 99;
  EXPECT_EQ(*b, 99);
  EXPECT_EQ(a.use_count(), 2);
}

TEST(HostPtr, SingleDestructionOnLastReference) {
  LifetimeTracked::live_count = 0;
  {
    host_ptr<LifetimeTracked> a(std::in_place, 1);
    EXPECT_EQ(LifetimeTracked::live_count, 1);
    {
      host_ptr<LifetimeTracked> b = a;
      EXPECT_EQ(LifetimeTracked::live_count, 1);  // shared, not duplicated
    }
    EXPECT_EQ(LifetimeTracked::live_count, 1);  // b dropped, a still owns
  }
  EXPECT_EQ(LifetimeTracked::live_count, 0);  // destroyed exactly once
}

TEST(HostPtr, MoveTransfersOwnership) {
  LifetimeTracked::live_count = 0;
  {
    host_ptr<LifetimeTracked> a(std::in_place, 5);
    host_ptr<LifetimeTracked> b = std::move(a);
    EXPECT_EQ(LifetimeTracked::live_count, 1);  // moved, not duplicated
    EXPECT_EQ(b->value, 5);
  }
  EXPECT_EQ(LifetimeTracked::live_count, 0);
}

// Regression: assigning over the last reference must destroy the prior value (a defaulted assignment would leak it).
TEST(HostPtr, AssignmentReleasesOldValue) {
  LifetimeTracked::live_count = 0;
  {
    host_ptr<LifetimeTracked> a(std::in_place, 1);
    EXPECT_EQ(LifetimeTracked::live_count, 1);
    a = make_host_ptr<LifetimeTracked>(2);
    EXPECT_EQ(LifetimeTracked::live_count, 1);  // would be 2 if the old value leaked
    EXPECT_EQ(a->value, 2);
  }
  EXPECT_EQ(LifetimeTracked::live_count, 0);
}

// Assigning over a non-last reference leaves the shared value alive for the other owner.
TEST(HostPtr, AssignmentKeepsSharedValueAlive) {
  LifetimeTracked::live_count = 0;
  {
    host_ptr<LifetimeTracked> a(std::in_place, 1);
    host_ptr<LifetimeTracked> b = a;
    EXPECT_EQ(LifetimeTracked::live_count, 1);
    a = make_host_ptr<LifetimeTracked>(2);
    EXPECT_EQ(LifetimeTracked::live_count, 2);  // b's 1 + a's 2
    EXPECT_EQ(b->value, 1);
    EXPECT_EQ(a->value, 2);
  }
  EXPECT_EQ(LifetimeTracked::live_count, 0);
}

// --- shared_ptr-parity construction ---------------------------------------------------------------------------------

TEST(HostPtr, RawPointerAdoption) {
  LifetimeTracked::live_count = 0;
  {
    host_ptr<LifetimeTracked> p(new LifetimeTracked(7));
    EXPECT_EQ(LifetimeTracked::live_count, 1);
    EXPECT_EQ(p->value, 7);
  }
  EXPECT_EQ(LifetimeTracked::live_count, 0);  // deleted via default delete on last reference
}

TEST(HostPtr, CustomDeleter) {
  bool deleted = false;
  int sentinel = 0;
  {
    host_ptr<int> p(&sentinel, [&deleted](int*) { deleted = true; });
    EXPECT_TRUE(static_cast<bool>(p));
    EXPECT_FALSE(deleted);
  }
  EXPECT_TRUE(deleted);  // custom deleter ran on last-reference drop
}

TEST(HostPtr, UniquePtrAdoption) {
  LifetimeTracked::live_count = 0;
  {
    std::unique_ptr<LifetimeTracked> owner = std::make_unique<LifetimeTracked>(9);
    host_ptr<LifetimeTracked> p = std::move(owner);
    EXPECT_EQ(p->value, 9);
    EXPECT_EQ(LifetimeTracked::live_count, 1);
  }
  EXPECT_EQ(LifetimeTracked::live_count, 0);
}

// --- Polymorphism, casts, aliasing ----------------------------------------------------------------------------------

TEST(HostPtr, PolymorphicConversionAndCasts) {
  host_ptr<Derived> derived = make_host_ptr<Derived>();
  derived->b = 1;
  derived->d = 2;

  host_ptr<Base> base = derived;  // converting ctor (Derived -> Base), shares ownership
  EXPECT_EQ(base.use_count(), 2);
  EXPECT_EQ(base->b, 1);

  host_ptr<Derived> back = static_pointer_cast<Derived>(base);
  EXPECT_EQ(back->d, 2);
  EXPECT_EQ(back.use_count(), 3);  // all three share one control block

  host_ptr<Derived> dyn = dynamic_pointer_cast<Derived>(base);
  ASSERT_TRUE(static_cast<bool>(dyn));
  EXPECT_EQ(dyn->d, 2);
}

TEST(HostPtr, AliasingConstructor) {
  host_ptr<std::pair<int, int>> owner = make_host_ptr<std::pair<int, int>>(3, 4);
  host_ptr<int> alias(owner, &owner->second);  // shares owner's lifetime, points at the member
  EXPECT_EQ(*alias, 4);
  EXPECT_EQ(alias.use_count(), 2);
  owner.reset();          // owner released, but the pair stays alive via the alias
  EXPECT_EQ(*alias, 4);
}

// --- Modifiers, ordering, containers --------------------------------------------------------------------------------

TEST(HostPtr, ResetAndSwap) {
  LifetimeTracked::live_count = 0;
  host_ptr<LifetimeTracked> a(std::in_place, 1);
  host_ptr<LifetimeTracked> b = a;
  EXPECT_EQ(LifetimeTracked::live_count, 1);
  a.reset();
  EXPECT_FALSE(static_cast<bool>(a));
  EXPECT_EQ(LifetimeTracked::live_count, 1);  // b still owns
  b.reset();
  EXPECT_EQ(LifetimeTracked::live_count, 0);

  host_ptr<int> x(1);
  host_ptr<int> y(2);
  swap(x, y);
  EXPECT_EQ(*x, 2);
  EXPECT_EQ(*y, 1);
}

TEST(HostPtr, OwnerBeforeAndContainers) {
  auto a = make_host_ptr<int>(1);
  auto b = make_host_ptr<int>(2);
  EXPECT_TRUE(a.owner_before(b) || b.owner_before(a));  // distinct owners are strictly ordered
  auto a2 = a;
  EXPECT_FALSE(a.owner_before(a2));
  EXPECT_FALSE(a2.owner_before(a));  // shared control block -> equivalent

  std::set<host_ptr<int>> ordered;  // exercises operator<=>
  ordered.insert(a);
  ordered.insert(b);
  ordered.insert(a2);
  EXPECT_EQ(ordered.size(), 2u);  // a and a2 share get() -> one key

  std::unordered_set<host_ptr<int>> hashed;  // exercises std::hash + operator==
  hashed.insert(a);
  hashed.insert(b);
  hashed.insert(a2);
  EXPECT_EQ(hashed.size(), 2u);
}

}  // namespace

}  // namespace mundy
