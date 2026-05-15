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

#ifndef MUNDY_MATH_ACCESSOR_HPP_
#define MUNDY_MATH_ACCESSOR_HPP_

// C++ core libs
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/impl/AccessorImpl.hpp>
#include <mundy_utils/storage.hpp>
#include <mundy_utils/throw_assert.hpp>

namespace mundy {

namespace impl {

template <typename Accessor>
struct is_stored_accessor : std::false_type {};

template <typename T>
struct is_stored_accessor<storage<T>> : std::true_type {};

template <typename Accessor>
inline constexpr bool is_stored_accessor_v = is_stored_accessor<std::remove_cvref_t<Accessor>>::value;

template <typename Accessor>
struct accessor_underlying_type {
  using type = std::remove_cvref_t<Accessor>;
};

template <typename T>
struct accessor_underlying_type<storage<T>> {
  using type = std::remove_cvref_t<decltype(std::declval<storage<T>&>().get())>;
};

template <typename Accessor>
using accessor_underlying_type_t = typename accessor_underlying_type<std::remove_cvref_t<Accessor>>::type;

template <typename Accessor>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) unwrap_accessor(Accessor&& accessor) {
  if constexpr (is_stored_accessor_v<Accessor>) {
    return std::forward<Accessor>(accessor).get();
  } else {
    return std::forward<Accessor>(accessor);
  }
}

}  // namespace impl

// Separation of Concerns: Vectors, Matrices, and Quaternions shouldn't care about memory access patterns.
// They should be able to work with any type of memory access pattern, whether it is contiguous or strided, owned or
// unowned. This is especially important for GPU-compatable code.
//
// To achieve this, Vectors, Matrices, and Quaternions will be templated by an Accessor class. In each case, an Accessor
// needs to be copyable and provide a const [] operator. If the Accessor is able to be modified, it should also provide
// a non-const [] operator. The signatures of these operators are as follows:
//   KOKKOS_INLINE_FUNCTION const T& operator[](size_t idx) const;
//   KOKKOS_INLINE_FUNCTION T& operator[](size_t idx); // Optional
// For Vector3, idx is 0, 1, or 2.
// For Matrix3, it is 0, 1, 2, 3, 4, 5, 6, 7, or 8.
// For Quaternion, it is 0, 1, 2, or 3.
// The underlying type of the accessor can be fetched with std::remove_reference_t<decltype(accessor[0])>. Or,
// alternatively, Vector3, Matrix3, and Quaternion can be templated by T and AccessorType. This approach allows us to
// define default Accessors while still allowing templating by T.
//
// Accessors may be owning or non-owning, that is irrelevant to the Vector3, Matrix3, and Quaternion classes; however,
// these accessors should be lightweight such that they can be copied around without much overhead. As a result, the
// lifetime of the data underlying the accessor should be as long as the Vector3, Matrix3, or Quaternion that use it.
//
// For efficiency reasons, accessors may consider using a stride between elements. Again, this is irrelevant to the
// Vector3, Matrix3, and Quaternion classes.
//
// Good examples of accessors include the following:
//   - A simple pointer to the data (non-owning)
//   - A Kokkos::View (non-owning)
//   - A class that wraps a pointer and a stride (non-owning)
//   - A class that owns a T array with or without a stride (owning)
//
// Bad examples of accessors include the following:
//   - A std::vector (owning) due to its HEAVY copy constructor
//   - A std::shared_ptr (owning) due to its lack of thread safety
//
// By default, a Vector3, Matrix3, or Quaternion will use a non-strided owning accessor consisting of a T[3], T[9], or
// T[4] and the necessary const and non-const [] operators.
//
// To give our accessors names, we will use the following naming convention:
//   - Vector3Data, Matrix3Data, or QuaternionData: contain non-strided arrays of the correct sizes templated by type
//   - Vector3, Matrix3View, or Quaternion: contain Kokkos::Views templated by type and layout
// Note that there are no inherent requirements on the types within the accessors, simply requirements on lengths. It's
// up to Vector3, Matrix3, and Quaternion to enforce their type requirements. As a result, the Data and View classes can
// all be consistently named as Arrays and Views, respectively and templated by their size.

/// \brief A concept that checks if Accessor has a const [] operator
template <typename Accessor, typename T>
concept HasConstAccessOperator = requires(Accessor a, size_t idx) {
  { impl::unwrap_accessor(a)[idx] } -> std::convertible_to<const T&>;
} || requires(Accessor a, size_t idx) {
  { impl::unwrap_accessor(a)(idx) } -> std::convertible_to<const T&>;
};

/// \brief A concept that checks if Accessor has a non-const [] operator
template <typename Accessor, typename T>
concept HasNonConstAccessOperator = requires(Accessor a, size_t idx) {
  { impl::unwrap_accessor(a)[idx] } -> std::convertible_to<T&>;
} || requires(Accessor a, size_t idx) {
  { impl::unwrap_accessor(a)(idx) } -> std::convertible_to<T&>;
};

/// \brief A concept that checks if Accessor has a [] operator regardless of constness
template <typename Accessor>
concept HasSubscriptOperator = requires(Accessor a, size_t idx) { impl::unwrap_accessor(a)[idx]; };

/// \brief A concept that checks if Accessor has a () operator regardless of constness
template <typename Accessor>
concept HasCallOperator = requires(Accessor a, size_t idx) { impl::unwrap_accessor(a)(idx); };

/// \brief A concept that checks if Accessor is wrapped in storage
template <typename Accessor>
concept StoredAccessor = impl::is_stored_accessor_v<Accessor>;

/// \brief A type alias for the underlying accessor type for raw or stored accessors
template <typename Accessor>
using accessor_underlying_type_t = impl::accessor_underlying_type_t<Accessor>;

/// \brief A concept that checks if a stored accessor provides const access to T
template <typename Accessor, typename T>
concept HasConstStoredAccessOperator = StoredAccessor<Accessor> && HasConstAccessOperator<Accessor, T>;

/// \brief A concept that checks if a stored accessor provides non-const access to T
template <typename Accessor, typename T>
concept HasNonConstStoredAccessOperator = StoredAccessor<Accessor> && HasNonConstAccessOperator<Accessor, T>;

/// \brief A concept that checks if Accessor has a copy constructor
template <typename Accessor>
concept HasCopyConstructor = requires(Accessor a) { Accessor{a}; };

/// \brief A concept that checks if Accessor has a move constructor
template <typename Accessor>
concept HasMoveConstructor = requires(Accessor a) { Accessor{std::move(a)}; };

/// \brief A concept that checks if an type is a valid accessor, aka it has a const [] operator or a non-const []
/// operator
template <typename Accessor, typename T>
concept ValidAccessor = (HasConstAccessOperator<Accessor, T> || HasNonConstAccessOperator<Accessor, T>);

/// \brief A concept that checks if an Accessor is default constructible
template <typename Accessor>
concept HasDefaultConstructor = requires { Accessor{}; };

/// \brief A concept that checks if an Accessor is constructible from N arguments of type T
template <typename Accessor, typename T, size_t N>
concept HasNArgConstructor =
    impl::can_construct_from_unpacked_tuple<Accessor, decltype(impl::generate_tuple_with_t_repeated_n_times<T, N>())>();

/// \brief A concept that checks if an Accessor is constructible from an initializer list of type T
template <typename Accessor, typename T>
concept HasInitializerListConstructor = requires(std::initializer_list<T> list) { Accessor{list}; };

/// @brief  Literal class enums! These are clearer and easier to work with than bools or explicit enums.
namespace Ownership {
struct Owns {};
struct Views {};
struct Mixed {};
struct Invalid {};
}  // namespace Ownership

namespace impl {

struct SizeTOnly {
  constexpr operator std::size_t() const {
    return 0;
  }
  constexpr operator unsigned() const = delete;
  constexpr operator long int() const = delete;
  constexpr operator int() const = delete;
};

struct UnsignedOnly {
  constexpr operator unsigned() const {
    return 0u;
  }
  constexpr operator std::size_t() const = delete;
  constexpr operator long int() const = delete;
  constexpr operator int() const = delete;
};

struct LongIntOnly {
  constexpr operator long int() const {
    return 0l;
  }
  constexpr operator std::size_t() const = delete;
  constexpr operator unsigned() const = delete;
  constexpr operator int() const = delete;
};

struct IntOnly {
  constexpr operator int() const {
    return 0;
  }
  constexpr operator std::size_t() const = delete;
  constexpr operator unsigned() const = delete;
  constexpr operator long int() const = delete;
};

template <typename Accessor>
concept SubscriptTakesSizeT = requires { unwrap_accessor(std::declval<Accessor>())[SizeTOnly{}]; };

template <typename Accessor>
concept SubscriptTakesUnsigned = requires { unwrap_accessor(std::declval<Accessor>())[UnsignedOnly{}]; };

template <typename Accessor>
concept SubscriptTakesLongInt = requires { unwrap_accessor(std::declval<Accessor>())[LongIntOnly{}]; };

template <typename Accessor>
concept SubscriptTakesInt = requires { unwrap_accessor(std::declval<Accessor>())[IntOnly{}]; };

template <typename Accessor>
concept CallTakesSizeT = requires { unwrap_accessor(std::declval<Accessor>())(SizeTOnly{}); };

template <typename Accessor>
concept CallTakesUnsigned = requires { unwrap_accessor(std::declval<Accessor>())(UnsignedOnly{}); };

template <typename Accessor>
concept CallTakesLongInt = requires { unwrap_accessor(std::declval<Accessor>())(LongIntOnly{}); };

template <typename Accessor>
concept CallTakesInt = requires { unwrap_accessor(std::declval<Accessor>())(IntOnly{}); };

template <typename T>
inline constexpr bool dependent_false_v = false;

template <typename IndexType>
KOKKOS_INLINE_FUNCTION constexpr IndexType checked_index_cast(const size_t idx) {
  MUNDY_THROW_ASSERT(idx <= static_cast<size_t>(Kokkos::Experimental::finite_max_v<IndexType>), std::out_of_range,
                     "Accessor index exceeds maximum representable index type.");
  return static_cast<IndexType>(idx);
}

/// \brief Unified index accessor with [] preferred over () if both are available.
template <typename Accessor>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) access_at(Accessor&& accessor, size_t idx) {
  auto&& unwrapped_accessor = unwrap_accessor(std::forward<Accessor>(accessor));
  using unwrapped_accessor_t = decltype(unwrapped_accessor);

  if constexpr (HasSubscriptOperator<unwrapped_accessor_t>) {
    if constexpr (SubscriptTakesSizeT<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)[checked_index_cast<size_t>(idx)];
    } else if constexpr (SubscriptTakesUnsigned<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)[checked_index_cast<unsigned>(idx)];
    } else if constexpr (SubscriptTakesLongInt<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)[checked_index_cast<long int>(idx)];
    } else if constexpr (SubscriptTakesInt<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)[checked_index_cast<int>(idx)];
    } else {
      static_assert(dependent_false_v<Accessor>,
                    "Accessor operator[] must accept one of: size_t, unsigned, long int, int.");
    }
  } else {
    static_assert(HasCallOperator<unwrapped_accessor_t>, "Accessor must support either operator[] or operator().");
    if constexpr (CallTakesSizeT<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)(checked_index_cast<size_t>(idx));
    } else if constexpr (CallTakesUnsigned<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)(checked_index_cast<unsigned>(idx));
    } else if constexpr (CallTakesLongInt<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)(checked_index_cast<long int>(idx));
    } else if constexpr (CallTakesInt<unwrapped_accessor_t>) {
      return std::forward<unwrapped_accessor_t>(unwrapped_accessor)(checked_index_cast<int>(idx));
    } else {
      static_assert(dependent_false_v<Accessor>,
                    "Accessor operator() must accept one of: size_t, unsigned, long int, int.");
    }
  }
}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_ACCESSOR_HPP_
