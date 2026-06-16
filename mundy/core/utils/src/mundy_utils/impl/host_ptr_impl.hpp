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

#ifndef MUNDY_UTILS_IMPL_HOST_PTR_IMPL_HPP_
#define MUNDY_UTILS_IMPL_HOST_PTR_IMPL_HPP_

/// \file host_ptr_impl.hpp
/// \brief A `std::shared_ptr`-equivalent whose handle is trivially copyable onto a device (host-resident value).

// C++ core
#include <compare>      // for std::strong_ordering, std::compare_three_way
#include <cstddef>      // for std::nullptr_t, std::size_t, std::max_align_t
#include <functional>   // for std::hash
#include <memory>       // for std::unique_ptr, std::default_delete
#include <new>          // for placement new, std::launder
#include <type_traits>  // for std::is_convertible_v
#include <utility>      // for std::move, std::forward, std::in_place_t

// Kokkos
#include <Kokkos_Core.hpp>

namespace mundy {

namespace impl {

/// \struct host_control_header
/// \brief Type-erased header placed at offset 0 of every `host_ptr` control block.
///
/// `dispose` destroys the managed object and then the control block's own members (e.g. an owned deleter). Because it
/// is the first member of the standard-layout control-block types below, a control-block pointer and a
/// `host_control_header` pointer are pointer-interconvertible — so the disposer can recover the concrete block.
struct host_control_header {
  void (*dispose)(host_control_header*) noexcept = nullptr;
};

/// \struct host_control_inplace
/// \brief Control block that owns the object inline (single allocation, `make_shared`-style).
///
/// The object lives in aligned storage rather than as a live member, so `dispose` destroys it exactly once (the
/// block's own destructor leaves the storage bytes alone).
template <typename Y>
struct host_control_inplace {
  host_control_header header;
  alignas(Y) unsigned char storage[sizeof(Y)];

  template <typename... Args>
  explicit host_control_inplace(Args&&... args) {
    header.dispose = &dispose;
    ::new (static_cast<void*>(&storage)) Y(std::forward<Args>(args)...);
  }

  Y* object() noexcept {
    return std::launder(reinterpret_cast<Y*>(&storage));
  }

  static void dispose(host_control_header* h) noexcept {
    auto* self = reinterpret_cast<host_control_inplace*>(h);  // h is `header`, the first member → interconvertible
    self->object()->~Y();
    self->~host_control_inplace();
  }
};

/// \struct host_control_ptr
/// \brief Control block that owns an external pointer plus a (possibly stateful) deleter.
template <typename Y, typename Deleter>
struct host_control_ptr {
  host_control_header header;
  Y* pointer;
  Deleter deleter;

  host_control_ptr(Y* pointer_in, Deleter deleter_in) : pointer(pointer_in), deleter(std::move(deleter_in)) {
    header.dispose = &dispose;
  }

  static void dispose(host_control_header* h) noexcept {
    auto* self = reinterpret_cast<host_control_ptr*>(h);
    self->deleter(self->pointer);
    self->~host_control_ptr();
  }
};

}  // namespace impl

} // namespace mundy

#endif  // MUNDY_UTILS_IMPL_HOST_PTR_IMPL_HPP_
