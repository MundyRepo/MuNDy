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
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core
#include <type_traits>

// Mundy
#include <mundy_utils/reference_wrapper.hpp>  // for mundy::utils::reference_wrapper

namespace mundy {

namespace utils {

namespace {

template <class T>
concept can_ref = requires(T&& t) { ::mundy::utils::ref(static_cast<T&&>(t)); };

template <class T>
concept can_cref = requires(T&& t) { ::mundy::utils::cref(static_cast<T&&>(t)); };

template <class T>
concept can_construct_wrapper =
    requires(T&& t) { ::mundy::utils::reference_wrapper<std::remove_reference_t<T>>(static_cast<T&&>(t)); };

KOKKOS_FUNCTION constexpr bool constexpr_get_and_conversion_work() {
  int value = 2;
  ::mundy::utils::reference_wrapper<int> wrapper(value);
  int& as_ref = wrapper;
  as_ref += 5;
  return wrapper.get() == 7 && value == 7;
}

KOKKOS_FUNCTION constexpr bool constexpr_ref_and_cref_work() {
  int value = 10;
  auto wrapped = ::mundy::utils::ref(value);
  wrapped.get() += 3;

  const int const_value = 9;
  auto const_wrapped = ::mundy::utils::cref(const_value);
  return value == 13 && wrapped.get() == 13 && const_wrapped.get() == 9;
}

struct Callable {
  int bias;
  KOKKOS_FUNCTION constexpr int operator()(int x) const {
    return x + bias;
  }
};

KOKKOS_FUNCTION constexpr bool constexpr_callable_forwarding_works() {
  Callable callable{6};
  auto wrapped = ::mundy::utils::ref(callable);
  return wrapped(4) == 10;
}

KOKKOS_FUNCTION constexpr ::mundy::utils::reference_wrapper<int> make_wrapper_from_scope(int& value) {
  return ::mundy::utils::ref(value);
}

KOKKOS_FUNCTION constexpr ::mundy::utils::reference_wrapper<const int> make_const_wrapper_from_scope(const int& value) {
  return ::mundy::utils::cref(value);
}

TEST(ReferenceWrapperTest, BasicGetAndConversionMutatesOriginal) {
  int value = 3;
  ::mundy::utils::reference_wrapper<int> wrapper(value);

  EXPECT_EQ(wrapper.get(), 3);

  wrapper.get() = 7;
  EXPECT_EQ(value, 7);

  int& value_ref = wrapper;
  value_ref = 11;
  EXPECT_EQ(value, 11);
}

TEST(ReferenceWrapperTest, RefAndCrefHelpersWrapLvalues) {
  int value = 5;
  auto mutable_wrapper = ::mundy::utils::ref(value);
  static_assert(std::is_same_v<decltype(mutable_wrapper), ::mundy::utils::reference_wrapper<int>>,
                "ref(int&) should produce reference_wrapper<int>");

  mutable_wrapper.get() = 9;
  EXPECT_EQ(value, 9);

  auto const_wrapper = ::mundy::utils::cref(value);
  static_assert(std::is_same_v<decltype(const_wrapper), ::mundy::utils::reference_wrapper<const int>>,
                "cref(int&) should produce reference_wrapper<const int>");
  EXPECT_EQ(const_wrapper.get(), 9);
}

TEST(ReferenceWrapperTest, RefAndCrefFromExistingWrappersPreserveReference) {
  int value = 4;
  auto wrapper = ::mundy::utils::ref(value);

  auto wrapper_again = ::mundy::utils::ref(wrapper);
  wrapper_again.get() = 12;
  EXPECT_EQ(value, 12);

  auto const_from_wrapper = ::mundy::utils::cref(wrapper);
  EXPECT_EQ(const_from_wrapper.get(), 12);

  auto const_from_const_wrapper = ::mundy::utils::cref(const_from_wrapper);
  EXPECT_EQ(const_from_const_wrapper.get(), 12);
}

TEST(ReferenceWrapperTest, WrapperCopySemanticsStillReferenceSameObject) {
  int value = 21;
  ::mundy::utils::reference_wrapper<int> first(value);
  ::mundy::utils::reference_wrapper<int> second(first);

  second.get() = 42;
  EXPECT_EQ(value, 42);

  int other = -1;
  ::mundy::utils::reference_wrapper<int> third(other);
  third = first;
  third.get() = 77;
  EXPECT_EQ(value, 77);
  EXPECT_EQ(other, -1);
}

TEST(ReferenceWrapperTest, CallableForwardingWorksForMutableCallable) {
  struct Accumulator {
    int sum = 0;

    int operator()(int a, int b) {
      sum += a + b;
      return sum;
    }
  };

  Accumulator accumulator{};
  auto wrapped = ::mundy::utils::ref(accumulator);

  EXPECT_EQ(wrapped(1, 2), 3);
  EXPECT_EQ(wrapped(5, 4), 12);
  EXPECT_EQ(accumulator.sum, 12);
}

TEST(ReferenceWrapperTest, TypeTraitsAndDeductionGuide) {
  static_assert(::mundy::utils::is_reference_wrapper_v<::mundy::utils::reference_wrapper<int>>,
                "reference_wrapper<T> should be detected as a wrapper");
  static_assert(!::mundy::utils::is_reference_wrapper_v<int>, "non-wrapper types should not be detected as wrappers");

  int value = 8;
  ::mundy::utils::reference_wrapper deduced(value);
  static_assert(std::is_same_v<decltype(deduced), ::mundy::utils::reference_wrapper<int>>,
                "CTAD should deduce reference_wrapper<int>");

  EXPECT_EQ(deduced.get(), 8);
}

TEST(ReferenceWrapperTest, ConversionAndConstructibilityContracts) {
  static_assert(std::is_convertible_v<::mundy::utils::reference_wrapper<int>, int&>,
                "reference_wrapper<int> should be implicitly convertible to int&");
  static_assert(std::is_convertible_v<::mundy::utils::reference_wrapper<const int>, const int&>,
                "reference_wrapper<const int> should be implicitly convertible to const int&");
  static_assert(!std::is_convertible_v<::mundy::utils::reference_wrapper<const int>, int&>,
                "reference_wrapper<const int> should not convert to mutable reference");

  static_assert(can_construct_wrapper<int&>, "wrapper should construct from lvalue reference");
  static_assert(!can_construct_wrapper<int>, "wrapper should not construct from rvalues");
}

TEST(ReferenceWrapperTest, DeletedRvalueRefAndCrefOverloadsAreNotCallable) {
  static_assert(can_ref<int&>, "ref(lvalue) should be callable");
  static_assert(!can_ref<int>, "ref(rvalue) should be ill-formed");

  static_assert(can_cref<int&>, "cref(lvalue) should be callable");
  static_assert(!can_cref<int>, "cref(rvalue) should be ill-formed");
}

TEST(ReferenceWrapperTest, ConstWrapperAllowsReadOnlyAccess) {
  int value = 13;
  const ::mundy::utils::reference_wrapper<int> wrapper(value);

  EXPECT_EQ(wrapper.get(), 13);
  wrapper.get() = 14;
  EXPECT_EQ(value, 14);

  const auto const_wrapper = ::mundy::utils::cref(value);
  EXPECT_EQ(const_wrapper.get(), 14);
}

TEST(ReferenceWrapperTest, LifetimeAndAliasingAcrossScopes) {
  int value = 31;

  const auto wrapped_from_scope = make_wrapper_from_scope(value);
  const auto wrapped_const_from_scope = make_const_wrapper_from_scope(value);

  wrapped_from_scope.get() = 44;
  EXPECT_EQ(value, 44);
  EXPECT_EQ(wrapped_const_from_scope.get(), 44);

  int& as_ref = wrapped_from_scope;
  as_ref = 58;
  EXPECT_EQ(value, 58);
  EXPECT_EQ(wrapped_const_from_scope.get(), 58);
}

TEST(ReferenceWrapperTest, CallableForwardingSupportsReferenceReturnAndConstCallable) {
  struct RefReturningCallable {
    mutable int stored = 0;

    int& operator()(int add) {
      stored += add;
      return stored;
    }

    const int& operator()(int add) const {
      stored += add;
      return stored;
    }
  };

  RefReturningCallable callable{};
  auto wrapped = ::mundy::utils::ref(callable);

  int& mutable_result = wrapped(3);
  mutable_result += 7;
  EXPECT_EQ(callable.stored, 10);

  const auto const_wrapped = ::mundy::utils::cref(callable);
  const int& const_result = const_wrapped(5);
  EXPECT_EQ(const_result, 15);
  EXPECT_EQ(callable.stored, 15);
}

TEST(ReferenceWrapperTest, ConstexprCoreOperations) {
  static_assert(constexpr_get_and_conversion_work(),
                "constexpr reference_wrapper construction/get/conversion should work");
  static_assert(constexpr_ref_and_cref_work(), "constexpr ref/cref should work for lvalues");
  static_assert(constexpr_callable_forwarding_works(), "constexpr callable forwarding should work");

  EXPECT_TRUE(constexpr_get_and_conversion_work());
  EXPECT_TRUE(constexpr_ref_and_cref_work());
  EXPECT_TRUE(constexpr_callable_forwarding_works());
}

}  // namespace

}  // namespace utils

}  // namespace mundy
