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

#ifndef MUNDY_UTILS_STRINGSINK_HPP_
#define MUNDY_UTILS_STRINGSINK_HPP_

/// \file StringSink.hpp
/// \brief Helpers for building message strings with `<<` syntax.
///
/// `mundy::sink()` starts a message-building pipeline. Literal-only pipelines stay in a
/// `StringLiteralSink`, so they remain usable in `constexpr` code and in device-only paths that require compile-time
/// strings. Appending any non-literal streamable object promotes the pipeline to a `StringSink`, which stores the
/// streamed chunks lazily and materializes a `std::string` only when needed.
///
/// Typical usage:
/// \code{.cpp}
/// constexpr auto device_message = mundy::sink() << "entity " << "was not found";
/// auto host_message = mundy::sink() << "Failure for a = " << a;
/// \endcode

// C++ core
#include <concepts>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

// Kokkos
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_utils/StringLiteral.hpp>
#include <mundy_utils/tuple.hpp>
#include <mundy_utils/requires.hpp>

namespace mundy {

template <size_t N>
struct StringLiteralSink;

template <typename... Chunks>
struct StringSink;

struct SinkStart {};

namespace impl {

template <typename T>
struct sink_stored_type {
  using type = std::remove_cvref_t<T>;
};

template <size_t N>
struct sink_stored_type<char[N]> {
  using type = StringLiteral<N>;
};

template <size_t N>
struct sink_stored_type<const char[N]> {
  using type = StringLiteral<N>;
};

template <size_t N>
struct sink_stored_type<StringLiteral<N>> {
  using type = StringLiteral<N>;
};

template <typename T>
using sink_stored_t = typename sink_stored_type<std::remove_cvref_t<T>>::type;

template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr StringLiteral<N> make_sink_piece(const char (&value)[N]) {
  return make_string_literal(value);
}

template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr StringLiteral<N> make_sink_piece(const StringLiteral<N>& value) {
  return value;
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr std::remove_cvref_t<T> make_sink_piece(T&& value) {
  return std::forward<T>(value);
}

template <typename T>
concept LiteralSinkChunk = is_char_array_v<T> || is_our_string_literal_v<T>;

template <typename T>
concept SinkChunk = requires { typename sink_stored_t<T>; } &&
                    std::constructible_from<sink_stored_t<T>, decltype(make_sink_piece(std::declval<T>()))>;

template <typename T>
concept RuntimeSinkChunk = SinkChunk<T> && !LiteralSinkChunk<T>;

template <typename... Chunks>
  MUNDY_REQUIRES(SinkChunk<Chunks> && ...)
KOKKOS_INLINE_FUNCTION constexpr auto make_string_sink(Chunks&&... chunks) {
  using SinkType = StringSink<sink_stored_t<Chunks>...>;
  return SinkType(::mundy::make_tuple(make_sink_piece(std::forward<Chunks>(chunks))...));
}

}  // namespace impl

/// \brief Sink that still represents a compile-time string.
///
/// This sink is produced when every chunk streamed into `mundy::sink()` is either a string literal or a
/// `mundy::StringLiteral`. It supports further literal concatenation at compile time and promotes to `StringSink`
/// automatically when a runtime value is appended.
template <size_t N>
struct StringLiteralSink {
  static constexpr bool is_compile_time = true;

  StringLiteral<N> value;

  KOKKOS_INLINE_FUNCTION constexpr explicit StringLiteralSink(const StringLiteral<N>& input) : value(input) {
  }

  std::string to_string() const {
    return value.to_string();
  }

  KOKKOS_INLINE_FUNCTION constexpr StringLiteral<N> to_string_literal() const {
    return value;
  }

  //! \name Stream operators
  //@{

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr auto operator<<(const char (&rhs)[OtherSize]) const {
    return StringLiteralSink<N + OtherSize - 1>(value + rhs);
  }

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr auto operator<<(const StringLiteral<OtherSize>& rhs) const {
    return StringLiteralSink<N + OtherSize - 1>(value + rhs);
  }

  template <impl::RuntimeSinkChunk T>
  KOKKOS_INLINE_FUNCTION constexpr auto operator<<(T&& rhs) const;
  //@}

  //! \name Equality operators
  //@{

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const char (&other)[OtherSize]) const {
    if constexpr (N != OtherSize) {
      return false;
    } else {
      return value == make_string_literal(other);
    }
  }

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const StringLiteralSink<OtherSize>& other) const {
    if constexpr (N != OtherSize) {
      return false;
    } else {
      return value == other.value;
    }
  }

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const StringLiteral<OtherSize>& other) const {
    if constexpr (N != OtherSize) {
      return false;
    } else {
      return value == other;
    }
  }
  //@}
};

/// \brief Sink that stores streamed chunks until a string is actually needed.
///
/// `StringSink` is the chunked counterpart to `StringLiteralSink`. It stores each chunk by value and only joins them
/// into a single `std::string` when `to_string()` or `operator<<` is used. If every stored chunk is a
/// `mundy::StringLiteral`, it can also be collapsed to a `StringLiteral` at compile time.
template <typename... Chunks>
struct StringSink {
  static constexpr bool is_compile_time = (is_our_string_literal_v<Chunks> && ...);

  ::mundy::tuple<Chunks...> chunks;

  KOKKOS_INLINE_FUNCTION constexpr explicit StringSink(::mundy::tuple<Chunks...> input_chunks)
      : chunks(std::move(input_chunks)) {
  }

  template <impl::SinkChunk T>
  KOKKOS_INLINE_FUNCTION constexpr auto operator<<(T&& rhs) const;

  std::string to_string() const {
    std::ostringstream os;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      ((os << ::mundy::get<Is>(chunks)), ...);
    }(std::make_index_sequence<sizeof...(Chunks)>{});
    return os.str();
  }

  KOKKOS_INLINE_FUNCTION constexpr auto to_string_literal() const
    MUNDY_REQUIRES(is_compile_time)
  {
    if constexpr (sizeof...(Chunks) == 0) {
      return make_string_literal("");
    } else {
      return to_string_literal_impl(std::make_index_sequence<sizeof...(Chunks)>{});
    }
  }

  //! \name Equality operators
  //@{

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const char (&other)[OtherSize]) const
    MUNDY_REQUIRES(is_compile_time)
  {
    return *this == make_string_literal(other);
  }

  template <size_t OtherSize>
  bool operator==(const char (&other)[OtherSize]) const
    MUNDY_REQUIRES(!is_compile_time)
  {
    return to_string() == other;
  }

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const StringLiteralSink<OtherSize>& other) const
    MUNDY_REQUIRES(is_compile_time)
  {
    return *this == other.value;
  }

  template <size_t OtherSize>
  bool operator==(const StringLiteralSink<OtherSize>& other) const
    MUNDY_REQUIRES(!is_compile_time)
  {
    return to_string() == other.to_string();
  }

  template <size_t OtherSize>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const StringLiteral<OtherSize>& other) const
    MUNDY_REQUIRES(is_compile_time)
  {
    constexpr size_t our_size = decltype(to_string_literal())::size;
    if constexpr (our_size != OtherSize) {
      return false;
    } else {
      return to_string_literal() == other;
    }
  }

  template <size_t OtherSize>
  bool operator==(const StringLiteral<OtherSize>& other) const
    MUNDY_REQUIRES(!is_compile_time)
  {
    return to_string() == other.to_string();
  }

  template <typename... OtherChunks>
  KOKKOS_INLINE_FUNCTION constexpr bool operator==(const StringSink<OtherChunks...>& other) const
    MUNDY_REQUIRES(is_compile_time && StringSink<OtherChunks...>::is_compile_time)
  {
    constexpr size_t our_size = decltype(to_string_literal())::size;
    constexpr size_t other_size = decltype(other.to_string_literal())::size;
    if constexpr (our_size != other_size) {
      return false;
    } else {
      return to_string_literal() == other.to_string_literal();
    }
  }

  template <typename... OtherChunks>
  bool operator==(const StringSink<OtherChunks...>& other) const
    MUNDY_REQUIRES(!(is_compile_time && StringSink<OtherChunks...>::is_compile_time))
  {
    return to_string() == other.to_string();
  }

  bool operator==(const std::string& other) const {
    return to_string() == other;
  }
  //@}

 private:
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION constexpr auto to_string_literal_impl(std::index_sequence<Is...>) const {
    return (::mundy::get<Is>(chunks) + ...);
  }
};

template <typename T>
struct is_string_literal_sink : std::false_type {};

template <size_t N>
struct is_string_literal_sink<StringLiteralSink<N>> : std::true_type {};

template <typename T>
inline constexpr bool is_string_literal_sink_v = is_string_literal_sink<std::remove_cvref_t<T>>::value;

template <typename T>
concept LiteralStringSink = is_string_literal_sink_v<T>;

template <typename T>
struct is_string_sink : std::false_type {};

template <typename... Chunks>
struct is_string_sink<StringSink<Chunks...>> : std::true_type {};

template <typename T>
inline constexpr bool is_string_sink_v = is_string_sink<std::remove_cvref_t<T>>::value;

template <typename T>
concept ChunkedStringSink = is_string_sink_v<T>;

template <typename T>
concept AnyStringSink = LiteralStringSink<T> || ChunkedStringSink<T>;

template <typename T>
concept CompileTimeStringSink = AnyStringSink<T> && requires { std::remove_cvref_t<T>::is_compile_time; } &&
                                std::remove_cvref_t<T>::is_compile_time;

/// \brief Start a new sink pipeline.
KOKKOS_INLINE_FUNCTION constexpr SinkStart sink() {
  return {};
}

template <impl::RuntimeSinkChunk T>
KOKKOS_INLINE_FUNCTION constexpr auto operator<<(SinkStart, T&& rhs) {
  return impl::make_string_sink(std::forward<T>(rhs));
}

template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr auto operator<<(SinkStart, const char (&rhs)[N]) {
  return StringLiteralSink<N>(make_string_literal(rhs));
}

template <size_t N>
KOKKOS_INLINE_FUNCTION constexpr auto operator<<(SinkStart, const StringLiteral<N>& rhs) {
  return StringLiteralSink<N>(rhs);
}

template <size_t N>
template <impl::RuntimeSinkChunk T>
KOKKOS_INLINE_FUNCTION constexpr auto StringLiteralSink<N>::operator<<(T&& rhs) const {
  return impl::make_string_sink(value, std::forward<T>(rhs));
}

template <typename... Chunks>
template <impl::SinkChunk T>
KOKKOS_INLINE_FUNCTION constexpr auto StringSink<Chunks...>::operator<<(T&& rhs) const {
  return StringSink<Chunks..., impl::sink_stored_t<T>>(
      ::mundy::tuple_cat(chunks, ::mundy::make_tuple(impl::make_sink_piece(std::forward<T>(rhs)))));
}

template <size_t N>
std::ostream& operator<<(std::ostream& os, const StringLiteralSink<N>& sink_to_print) {
  os << sink_to_print.value;
  return os;
}

template <typename... Chunks>
std::ostream& operator<<(std::ostream& os, const StringSink<Chunks...>& sink_to_print) {
  [&]<size_t... Is>(std::index_sequence<Is...>) {
    ((os << ::mundy::get<Is>(sink_to_print.chunks)), ...);
  }(std::make_index_sequence<sizeof...(Chunks)>{});
  return os;
}

}  // namespace mundy

#endif  // MUNDY_UTILS_STRINGSINK_HPP_
