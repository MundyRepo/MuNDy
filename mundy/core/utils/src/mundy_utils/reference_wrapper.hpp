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

#ifndef MUNDY_UTILS_REFERENCE_WRAPPER_HPP_
#define MUNDY_UTILS_REFERENCE_WRAPPER_HPP_

// C++ core
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/requires.hpp>
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP

namespace mundy {

namespace impl {

template <class T>
struct is_reference_wrapper : std::false_type {};

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T* addressof(T& value) noexcept {
  return __builtin_addressof(value);
}

/// \brief Is invocable concept
template <class F, class... Args>
concept invocable = requires(F&& f, Args&&... args) { std::forward<F>(f)(std::forward<Args>(args)...); };

/// \brief Is subscriptable concept
template <class T, class Index>
concept subscriptable = requires(T&& t, Index&& idx) { std::forward<T>(t)[std::forward<Index>(idx)]; };

MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH

template <class F, class... Args>
MUNDY_REQUIRES(invocable<F, Args...>)
KOKKOS_INLINE_FUNCTION constexpr decltype(auto)
    invoke(F&& f, Args&&... args) noexcept(noexcept(std::forward<F>(f)(std::forward<Args>(args)...))) {
  return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <class T, class Index>
MUNDY_REQUIRES(subscriptable<T, Index>)
KOKKOS_INLINE_FUNCTION constexpr decltype(auto)
    subscript(T&& t, Index&& idx) noexcept(noexcept(std::forward<T>(t)[std::forward<Index>(idx)])) {
  return std::forward<T>(t)[std::forward<Index>(idx)];
}

MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_POP

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
      MUNDY_REQUIRES(impl::invocable<T&, Args&&...>) {
    return impl::invoke(*m_ptr, std::forward<Args>(args)...);
  }

  template <class Index>
  KOKKOS_FUNCTION constexpr decltype(auto) operator[](Index&& idx) const
      MUNDY_REQUIRES(impl::subscriptable<T&, Index&&>) {
    return impl::subscript(*m_ptr, std::forward<Index>(idx));
  }

  template <class... Args>
  KOKKOS_FUNCTION constexpr decltype(auto) operator()(Args&&... args) MUNDY_REQUIRES(impl::invocable<T&, Args&&...>) {
    return impl::invoke(*m_ptr, std::forward<Args>(args)...);
  }

  template <class Index>
  KOKKOS_FUNCTION constexpr decltype(auto) operator[](Index&& idx) MUNDY_REQUIRES(impl::subscriptable<T&, Index&&>) {
    return impl::subscript(*m_ptr, std::forward<Index>(idx));
  }

 private:
  T* m_ptr;
};

template <class T>
reference_wrapper(T&) -> reference_wrapper<T>;

namespace impl {

template <class T>
struct is_reference_wrapper<reference_wrapper<T>> : std::true_type {};

}  // namespace impl

/// \brief Detect if a type is mundy::reference_wrapper<...>.
template <class T>
struct is_reference_wrapper : impl::is_reference_wrapper<std::remove_cv_t<T>> {};

template <class T>
static constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;

/// \brief Make a mutable reference wrapper.
template <class T>
KOKKOS_FUNCTION constexpr auto ref(T&& t) noexcept
    -> reference_wrapper<std::remove_reference_t<T>> MUNDY_REQUIRES(std::is_lvalue_reference_v<T&&> &&
                                                                    !is_reference_wrapper_v<std::remove_cvref_t<T>>) {
  return reference_wrapper<std::remove_reference_t<T>>(t);
}

template <class T>
KOKKOS_FUNCTION constexpr reference_wrapper<T> ref(reference_wrapper<T> t) noexcept {
  return t;
}

/// \brief Make a const reference wrapper.
template <class T>
KOKKOS_FUNCTION constexpr auto cref(T&& t) noexcept
    -> reference_wrapper<const std::remove_reference_t<T>> MUNDY_REQUIRES(
        std::is_lvalue_reference_v<T&&> && !is_reference_wrapper_v<std::remove_cvref_t<T>>) {
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

}  // namespace mundy

#endif  // MUNDY_UTILS_REFERENCE_WRAPPER_HPP_
