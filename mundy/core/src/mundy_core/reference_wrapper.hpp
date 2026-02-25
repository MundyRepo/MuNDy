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

#ifndef MUNDY_CORE_REFERENCE_WRAPPER_HPP_
#define MUNDY_CORE_REFERENCE_WRAPPER_HPP_

// C++ core
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

namespace mundy {

namespace core {

namespace impl {

template <class T>
struct is_reference_wrapper : std::false_type {
};

template <class T>
KOKKOS_FUNCTION constexpr T* addressof(T& value) noexcept {
  return __builtin_addressof(value);
}

template <class F, class... Args>
KOKKOS_FUNCTION constexpr decltype(auto) invoke(F&& f, Args&&... args)
  noexcept(noexcept(static_cast<F&&>(f)(static_cast<Args&&>(args)...))) {
  return static_cast<F&&>(f)(static_cast<Args&&>(args)...);
}

}  // namespace impl

/// \brief A Kokkos-compatible wrapper around a reference.
///
/// Provides the same core behavior as std::reference_wrapper:
///  - stores a reference as a pointer,
///  - implicit conversion back to T&,
///  - get(),
///  - callable forwarding with operator() when T is invocable.
template <class T>
class reference_wrapper {
 public:
  static_assert(!std::is_reference_v<T>, "reference_wrapper<T> requires T to be a non-reference type");

  using type = T;

  KOKKOS_FUNCTION
  constexpr reference_wrapper(T& ref) noexcept : m_ptr(impl::addressof(ref)) {
  }

  KOKKOS_FUNCTION
  reference_wrapper(T&&) = delete;

  KOKKOS_DEFAULTED_FUNCTION constexpr reference_wrapper(const reference_wrapper&) = default;
  KOKKOS_DEFAULTED_FUNCTION constexpr reference_wrapper& operator=(const reference_wrapper&) = default;

  KOKKOS_FUNCTION
  constexpr operator T&() const noexcept {
    return *m_ptr;
  }

  KOKKOS_FUNCTION
  constexpr T& get() const noexcept {
    return *m_ptr;
  }

  template <class... Args>
  KOKKOS_FUNCTION constexpr decltype(auto) operator()(Args&&... args) const
    requires requires(T& callable, Args&&... invoke_args) {
      impl::invoke(callable, std::forward<Args>(invoke_args)...);
    }
  {
    return impl::invoke(*m_ptr, std::forward<Args>(args)...);
  }

 private:
  T* m_ptr;
};

template <class T>
reference_wrapper(T&) -> reference_wrapper<T>;

namespace impl {

template <class T>
struct is_reference_wrapper<reference_wrapper<T>> : std::true_type {
};

}  // namespace impl

/// \brief Detect if a type is mundy::core::reference_wrapper<...>.
template <class T>
struct is_reference_wrapper : impl::is_reference_wrapper<std::remove_cv_t<T>> {
};

template <class T>
static constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;

/// \brief Make a mutable reference wrapper.
template <class T>
KOKKOS_FUNCTION constexpr auto ref(T&& t) noexcept
  -> reference_wrapper<std::remove_reference_t<T>>
  requires(std::is_lvalue_reference_v<T&&> && !is_reference_wrapper_v<std::remove_cvref_t<T>>)
{
  return reference_wrapper<std::remove_reference_t<T>>(t);
}

template <class T>
KOKKOS_FUNCTION constexpr reference_wrapper<T> ref(reference_wrapper<T> t) noexcept {
  return t;
}

/// \brief Make a const reference wrapper.
template <class T>
KOKKOS_FUNCTION constexpr auto cref(T&& t) noexcept
  -> reference_wrapper<const std::remove_reference_t<T>>
  requires(std::is_lvalue_reference_v<T&&> && !is_reference_wrapper_v<std::remove_cvref_t<T>>)
{
  return reference_wrapper<const std::remove_reference_t<T>>(t);
}

template <class T>
KOKKOS_FUNCTION constexpr reference_wrapper<const T> cref(reference_wrapper<T> t) noexcept {
  return reference_wrapper<const T>(t.get());
}

template <class T>
KOKKOS_FUNCTION constexpr reference_wrapper<const T> cref(reference_wrapper<const T> t) noexcept {
  return t;
}

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_REFERENCE_WRAPPER_HPP_
