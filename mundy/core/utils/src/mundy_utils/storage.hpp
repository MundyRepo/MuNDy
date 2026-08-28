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

#ifndef MUNDY_UTILS_STORAGE_HPP_
#define MUNDY_UTILS_STORAGE_HPP_

// C++ core
#include <concepts>
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/reference_wrapper.hpp>  // for mundy::reference_wrapper, ref, is_reference_wrapper_v
#include <mundy_utils/requires.hpp>

namespace mundy {

template <class T>
class storage;

namespace impl {

template <class T>
struct is_storage : std::false_type {};

template <class T>
struct is_storage<storage<T>> : std::true_type {};

template <class T>
static constexpr bool is_storage_v = is_storage<std::remove_cvref_t<T>>::value;

template <class T>
struct storage_underlying_type {
  using type = T;
};

template <class T>
MUNDY_REQUIRES(is_storage_v<T>)
struct storage_underlying_type<T> {
  using type = typename T::stored_type;
};

template <class T>
using storage_underlying_type_t = typename storage_underlying_type<T>::type;

template <class T>
struct storage_type {
 private:
  using no_ref_t = std::remove_reference_t<T>;
  using bare_t = std::remove_cv_t<no_ref_t>;
  using no_cvref_t = std::remove_cvref_t<T>;
  using storage_unwrapped_t = storage_underlying_type_t<no_cvref_t>;
  // An array cannot be stored by value, so it decays to a pointer to its first element, element cv intact.
  using decayed_array_t = std::remove_extent_t<no_ref_t>*;

 public:
  using type = std::conditional_t<
      is_storage_v<no_cvref_t>, storage_unwrapped_t,
      std::conditional_t<
          is_reference_wrapper_v<no_cvref_t>, bare_t,
          std::conditional_t<std::is_pointer_v<no_ref_t>, bare_t,
                             std::conditional_t<std::is_array_v<no_ref_t>, decayed_array_t,
                                                std::conditional_t<std::is_lvalue_reference_v<T>,
                                                                   reference_wrapper<no_ref_t>, no_cvref_t>>>>>;
};

template <class T>
struct store_input_type {
  using type = T;
};

template <class T>
MUNDY_REQUIRES(is_storage_v<std::remove_cvref_t<T>>)
struct store_input_type<T> {
  using type = typename std::remove_cvref_t<T>::input_type;
};

template <class T>
using storage_type_t = typename storage_type<T>::type;

template <class T>
using store_input_type_t = typename store_input_type<T>::type;

/// \brief Reach the value behind a stored object: unwrap a nested storage or reference_wrapper, hand back a pointer by
/// value, and otherwise yield the object itself.
///
/// The argument's value category is preserved, so an rvalue yields an rvalue and the stored value can be move
/// constructed rather than copied.
template <class Stored>
KOKKOS_FUNCTION constexpr decltype(auto) storage_get(Stored&& value) noexcept {
  using bare_t = std::remove_cvref_t<Stored>;
  if constexpr (is_storage_v<bare_t>) {
    return value.get();
  } else if constexpr (is_reference_wrapper_v<bare_t>) {
    return value.get();
  } else if constexpr (std::is_pointer_v<bare_t>) {
    return static_cast<std::remove_cv_t<bare_t>>(value);
  } else {
    return std::forward<Stored>(value);
  }
}

template <class T>
using storage_value_type_t = std::remove_cvref_t<decltype(storage_get(std::declval<storage_type_t<T>&>()))>;

}  // namespace impl

/// \brief Own or view a value using a simple normalized storage policy.
///
/// Storage policy for `T`:
///  - if `T` is (or refers to) `reference_wrapper<U>`, store `reference_wrapper<U>`
///  - else if `T` is (or refers to) a pointer type `U*`, store `U*`
///  - else if `T` is (or refers to) an array `U[N]`, store `U*`
///  - else if `T` is an lvalue reference `U&`, store `reference_wrapper<U>`
///  - otherwise, store `std::remove_cvref_t<T>` by value
template <class T>
class storage {
 public:
  using input_type = T;
  using stored_type = impl::storage_type_t<T>;
  using value_type = impl::storage_value_type_t<T>;

  KOKKOS_DEFAULTED_FUNCTION constexpr storage() MUNDY_REQUIRES(std::default_initializable<stored_type>) = default;

  KOKKOS_DEFAULTED_FUNCTION constexpr storage(const storage&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr storage(storage&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr storage& operator=(const storage&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr storage& operator=(storage&&) = default;

  template <class U>
  MUNDY_REQUIRES(std::constructible_from<stored_type, decltype(impl::storage_get(std::declval<U&&>()))>)
  KOKKOS_FUNCTION constexpr explicit storage(U&& value) noexcept(
      std::is_nothrow_constructible_v<stored_type, decltype(impl::storage_get(std::declval<U&&>()))>)
      : m_storage(impl::storage_get(std::forward<U>(value))) {
  }

  KOKKOS_FUNCTION constexpr decltype(auto) get() noexcept {
    return impl::storage_get(m_storage);
  }

  KOKKOS_FUNCTION constexpr decltype(auto) get() const noexcept {
    return impl::storage_get(m_storage);
  }

 private:
  stored_type m_storage;
};

/// \brief The type storage<T> normalizes an input of type T to before deciding its stored_type/value_type. Useful for
/// a class's own deduction guide when its constructor forwards straight into a storage<...> member, so the guide's
/// deduced template argument matches what storage(value)/store(value) would produce for that same input.
template <class T>
using store_input_type_t = impl::store_input_type_t<T>;

/// \brief CTAD applies the same input normalization as store(), so direct construction (storage(value)) and store()
/// always agree, including for a storage<U> input, which both resolve to storage<U> itself rather than nesting.
template <class T>
storage(T&&) -> storage<store_input_type_t<T>>;

/// \brief Create a storage object from a forwarding reference. Equivalent to direct construction: storage(value).
template <class T>
KOKKOS_FUNCTION constexpr auto store(T&& value) noexcept(noexcept(storage(std::forward<T>(value))))
    -> storage<store_input_type_t<T>> {
  return storage(std::forward<T>(value));
}

/// \brief Copy a value into a prvalue, leaving the source intact.
///
/// Hands a value off by value without declaring the source spent, which is what `std::move` would do. Reach for this
/// when an interface takes ownership of whatever it is given but the caller still needs its own copy.
template <class T>
KOKKOS_FUNCTION constexpr auto own(T&& value) noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<T>, T&&>)
    -> std::remove_cvref_t<T> MUNDY_REQUIRES(std::constructible_from<std::remove_cvref_t<T>, T&&>) {
  return std::forward<T>(value);
}

}  // namespace mundy

#endif  // MUNDY_UTILS_STORAGE_HPP_