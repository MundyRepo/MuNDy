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

#ifndef MUNDY_CORE_STORAGE_HPP_
#define MUNDY_CORE_STORAGE_HPP_

// C++ core
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_core/reference_wrapper.hpp>  // for mundy::core::reference_wrapper, ref, is_reference_wrapper_v

namespace mundy {

namespace core {

namespace impl {

template <class T>
struct storage_type {
 private:
  using no_ref_t = std::remove_reference_t<T>;

 public:
  using type =
      std::conditional_t<is_reference_wrapper_v<std::remove_cvref_t<T>>, std::remove_cv_t<no_ref_t>,
                         std::conditional_t<std::is_pointer_v<no_ref_t>, std::remove_cv_t<no_ref_t>,
                                            std::conditional_t<std::is_lvalue_reference_v<T>,
                                                               reference_wrapper<std::remove_reference_t<T>>,
                                                               std::remove_cvref_t<T>>>>;
};

template <class T>
using storage_type_t = typename storage_type<T>::type;

template <class Stored>
KOKKOS_FUNCTION constexpr decltype(auto) storage_get(Stored& value) noexcept {
  if constexpr (is_reference_wrapper_v<Stored>) {
    return value.get();
  } else if constexpr (std::is_pointer_v<Stored>) {
    return static_cast<std::remove_cv_t<Stored>>(value);
  } else {
    return (value);
  }
}

template <class Stored>
KOKKOS_FUNCTION constexpr decltype(auto) storage_get(const Stored& value) noexcept {
  if constexpr (is_reference_wrapper_v<Stored>) {
    return value.get();
  } else if constexpr (std::is_pointer_v<Stored>) {
    return static_cast<std::remove_cv_t<Stored>>(value);
  } else {
    return (value);
  }
}

}  // namespace impl

/// \brief Own or view a value using a simple normalized storage policy.
///
/// Storage policy for `T`:
///  - if `T` is (or refers to) `core::reference_wrapper<U>`, store `core::reference_wrapper<U>`
///  - else if `T` is (or refers to) a pointer type `U*`, store `U*`
///  - else if `T` is an lvalue reference `U&`, store `core::reference_wrapper<U>`
///  - otherwise, store `std::remove_cvref_t<T>` by value
template <class T>
class storage {
 public:
  using input_type = T;
  using stored_type = impl::storage_type_t<T>;

  KOKKOS_DEFAULTED_FUNCTION constexpr storage(const storage&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr storage(storage&&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr storage& operator=(const storage&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr storage& operator=(storage&&) = default;

  template <class U>
    requires std::constructible_from<stored_type, U&&>
  KOKKOS_FUNCTION constexpr explicit storage(U&& value) noexcept(std::is_nothrow_constructible_v<stored_type, U&&>)
      : m_storage(std::forward<U>(value)) {
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

template <class T>
storage(T&&) -> storage<T>;

/// \brief Create a storage object from a forwarding reference.
template <class T>
KOKKOS_FUNCTION constexpr auto store(T&& value) noexcept(noexcept(storage<T>(std::forward<T>(value)))) -> storage<T> {
  return storage<T>(std::forward<T>(value));
}

template <class T>
KOKKOS_FUNCTION constexpr storage<T> store(storage<T> value) noexcept {
  return value;
}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_STORAGE_HPP_