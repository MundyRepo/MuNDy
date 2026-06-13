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

#ifndef MUNDY_UTILS_HOST_PTR_HPP_
#define MUNDY_UTILS_HOST_PTR_HPP_

/// \file host_ptr.hpp
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

// Mundy
#include <mundy_utils/impl/host_ptr_impl.hpp>

namespace mundy {

/// \class host_ptr
/// \brief A reference-counted shared handle to a host-resident `T` whose *handle* is trivially copyable onto a device.
///
/// `host_ptr<T>` is, semantically, a `std::shared_ptr<T>` that survives being captured by value into a device kernel:
/// were device support not a concern, a plain `std::shared_ptr<T>` would be used instead. It mirrors `std::shared_ptr`
/// in ownership model and interface — owning raw-pointer construction, custom deleters, `unique_ptr` adoption, the
/// aliasing constructor, `Derived`→`Base` conversions, the pointer casts, ordering/hashing/`owner_before`, and
/// `reset` — with the single deliberate omission of `weak_ptr`/`enable_shared_from_this`.
///
/// \par How the device-copyability works
/// A `host_ptr` is a `Kokkos::View` (the device-copyable, reference-counted allocation) holding a small type-erased
/// control block, plus a typed `T* ptr_`. The `View` supplies the reference count — atomic on the host, with tracking
/// disabled inside kernels, so a copy captured into a kernel is a cheap non-owning handle and the owner must outlive
/// the kernel. The hand-rolled control block supplies only type-erased *destruction* (the disposer), so deleters,
/// adoption, aliasing, and polymorphism all work without `std::shared_ptr`. The value lives in host memory and is
/// accessible **only on the host**: the accessors are not `KOKKOS_FUNCTION`, so dereferencing on the device is a
/// compile error.
///
/// \tparam T Host-resident pointed-to type.
template <typename T>
class host_ptr {
 public:
  //! \name Type aliases
  //@{
  using element_type = T;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Construct an empty handle that owns and points to nothing.
  host_ptr() = default;

  /// \brief Construct an empty handle from `nullptr` (mirrors `std::shared_ptr`).
  host_ptr(std::nullptr_t) noexcept {  // NOLINT(runtime/explicit) — implicit, like std::shared_ptr
  }

  /// \brief Take ownership of `ptr`, deleting it with `delete` when the last reference drops.
  template <typename U>
    requires std::is_convertible_v<U*, T*>
  explicit host_ptr(U* ptr) {
    make_control<impl::host_control_ptr<U, std::default_delete<U>>>(ptr, std::default_delete<U>{});
    ptr_ = ptr;
  }

  /// \brief Take ownership of `ptr`, invoking `deleter(ptr)` when the last reference drops.
  template <typename U, typename Deleter>
    requires std::is_convertible_v<U*, T*>
  host_ptr(U* ptr, Deleter deleter) {
    make_control<impl::host_control_ptr<U, Deleter>>(ptr, std::move(deleter));
    ptr_ = ptr;
  }

  /// \brief Own no object but carry a deleter to invoke with `nullptr` (mirrors `std::shared_ptr`).
  template <typename Deleter>
  host_ptr(std::nullptr_t, Deleter deleter) {
    make_control<impl::host_control_ptr<T, Deleter>>(nullptr, std::move(deleter));
  }

  /// \brief Adopt the object owned by a `std::unique_ptr`.
  template <typename U, typename Deleter>
    requires std::is_convertible_v<U*, T*>
  host_ptr(std::unique_ptr<U, Deleter>&& owner) {  // NOLINT(runtime/explicit) — implicit, like std::shared_ptr
    if (owner) {
      U* ptr = owner.get();
      make_control<impl::host_control_ptr<U, Deleter>>(ptr, std::move(owner.get_deleter()));
      owner.release();
      ptr_ = ptr;
    }
  }

  /// \brief Construct a fresh `T` by copying `value` (convenience beyond `std::shared_ptr`).
  explicit host_ptr(const T& value) {
    ptr_ = make_control<impl::host_control_inplace<T>>(value)->object();
  }

  /// \brief Construct a fresh `T` by moving `value` (convenience beyond `std::shared_ptr`).
  explicit host_ptr(T&& value) {
    ptr_ = make_control<impl::host_control_inplace<T>>(std::move(value))->object();
  }

  /// \brief Construct a fresh `T` in place, forwarding `args` to its constructor.
  template <typename... Args>
  explicit host_ptr(std::in_place_t, Args&&... args) {
    ptr_ = make_control<impl::host_control_inplace<T>>(std::forward<Args>(args)...)->object();
  }

  /// \brief Shallow copy: shares ownership. Trivially device-copyable.
  KOKKOS_DEFAULTED_FUNCTION host_ptr(const host_ptr&) = default;
  KOKKOS_DEFAULTED_FUNCTION host_ptr(host_ptr&&) = default;

  /// \brief Converting copy: `host_ptr<Base>` from `host_ptr<Derived>` (shares ownership, retypes the pointer).
  template <typename U>
    requires std::is_convertible_v<U*, T*>
  host_ptr(const host_ptr<U>& other)  // NOLINT(runtime/explicit) — implicit, like std::shared_ptr
      : control_(other.control_), header_(other.header_), ptr_(other.ptr_) {
  }

  /// \brief Converting move.
  template <typename U>
    requires std::is_convertible_v<U*, T*>
  host_ptr(host_ptr<U>&& other)  // NOLINT(runtime/explicit) — implicit, like std::shared_ptr
      : control_(std::move(other.control_)), header_(other.header_), ptr_(other.ptr_) {
    other.header_ = nullptr;
    other.ptr_ = nullptr;
  }

  /// \brief Aliasing constructor: shares `other`'s ownership but points at `ptr` (which need not be `other`'s object).
  template <typename U>
  host_ptr(const host_ptr<U>& other, T* ptr) : control_(other.control_), header_(other.header_), ptr_(ptr) {
  }

  /// \brief Release this handle's ownership; runs the disposer if this was the last reference.
  KOKKOS_FUNCTION ~host_ptr() {
    KOKKOS_IF_ON_HOST((release();))
  }
  //@}

  //! \name Assignment (host-only; copy-and-swap, so the prior value is released exactly once)
  //@{
  host_ptr& operator=(const host_ptr& other) {
    host_ptr(other).swap(*this);
    return *this;
  }
  host_ptr& operator=(host_ptr&& other) noexcept {
    host_ptr(std::move(other)).swap(*this);
    return *this;
  }
  template <typename U>
    requires std::is_convertible_v<U*, T*>
  host_ptr& operator=(const host_ptr<U>& other) {
    host_ptr(other).swap(*this);
    return *this;
  }
  template <typename U>
    requires std::is_convertible_v<U*, T*>
  host_ptr& operator=(host_ptr<U>&& other) {
    host_ptr(std::move(other)).swap(*this);
    return *this;
  }
  host_ptr& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }
  //@}

  //! \name Modifiers (host-only)
  //@{

  /// \brief Become empty, releasing ownership (runs the disposer if this was the last reference).
  void reset() noexcept {
    host_ptr().swap(*this);
  }

  /// \brief Replace the managed object with ownership of `ptr` (deleted with `delete`).
  template <typename U>
    requires std::is_convertible_v<U*, T*>
  void reset(U* ptr) {
    host_ptr(ptr).swap(*this);
  }

  /// \brief Replace the managed object with ownership of `ptr` and a custom `deleter`.
  template <typename U, typename Deleter>
    requires std::is_convertible_v<U*, T*>
  void reset(U* ptr, Deleter deleter) {
    host_ptr(ptr, std::move(deleter)).swap(*this);
  }

  /// \brief Replace the managed object with a fresh `T` constructed from `args` (convenience).
  template <typename... Args>
  void emplace(Args&&... args) {
    host_ptr(std::in_place, std::forward<Args>(args)...).swap(*this);
  }

  /// \brief Swap ownership and pointer with another handle.
  void swap(host_ptr& other) noexcept {
    Kokkos::View<std::max_align_t*, Kokkos::HostSpace> tmp_control = control_;
    control_ = other.control_;
    other.control_ = tmp_control;
    impl::host_control_header* tmp_header = header_;
    header_ = other.header_;
    other.header_ = tmp_header;
    T* tmp_ptr = ptr_;
    ptr_ = other.ptr_;
    other.ptr_ = tmp_ptr;
  }
  //@}

  //! \name Observers (host-only)
  //@{

  /// \brief Reference to the pointed-to value. Undefined if this handle points to nothing.
  T& operator*() const noexcept {
    return *ptr_;
  }

  /// \brief Pointer to the pointed-to value (null if this handle points to nothing).
  T* operator->() const noexcept {
    return ptr_;
  }

  /// \brief Pointer to the pointed-to value (null if this handle points to nothing). Mirrors `std::shared_ptr::get`.
  T* get() const noexcept {
    return ptr_;
  }

  /// \brief Number of `host_ptr` instances sharing ownership (0 if empty).
  long use_count() const noexcept {
    return static_cast<long>(control_.use_count());
  }

  /// \brief True if this handle points to an object.
  explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  /// \brief Ownership-based ordering (by control block, like `std::shared_ptr::owner_before`).
  template <typename U>
  bool owner_before(const host_ptr<U>& other) const noexcept {
    return control_.data() < other.control_.data();
  }
  //@}

 private:
  //! Fetch (creating) the typed control block in a fresh allocation, recording its header. Returns the block.
  template <typename Control, typename... Args>
  Control* make_control(Args&&... args) {
    static_assert(alignof(Control) <= alignof(std::max_align_t), "host_ptr control block is over-aligned");
    const std::size_t units = (sizeof(Control) + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
    control_ =
        Kokkos::View<std::max_align_t*, Kokkos::HostSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mundy_host_ptr"), units);
    Control* block = ::new (static_cast<void*>(control_.data())) Control(std::forward<Args>(args)...);
    header_ = &block->header;
    return block;
  }

  void release() {
    if (control_.extent(0) > 0 && control_.use_count() == 1) {
      header_->dispose(header_);
    }
  }

  template <typename U>
  friend class host_ptr;

  //! Device-copyable, reference-counted allocation holding the type-erased control block (empty when this is empty).
  Kokkos::View<std::max_align_t*, Kokkos::HostSpace> control_;
  //! Typed access into the control block's disposer (null when empty). A host pointer.
  impl::host_control_header* header_ = nullptr;
  //! Typed pointer for access (may alias a subobject of the owned object). A host pointer — host-only to dereference.
  T* ptr_ = nullptr;
};

/// \brief Construct a `host_ptr<T>` owning a fresh `T`, forwarding `args` to its constructor. Mirrors `make_shared`.
template <typename T, typename... Args>
host_ptr<T> make_host_ptr(Args&&... args) {
  return host_ptr<T>(std::in_place, std::forward<Args>(args)...);
}

/// \brief Non-member swap.
template <typename T>
void swap(host_ptr<T>& lhs, host_ptr<T>& rhs) noexcept {
  lhs.swap(rhs);
}

//! \name Pointer casts (share ownership, retype the pointer — mirror std::shared_ptr's casts)
//@{
template <typename U, typename T>
host_ptr<U> static_pointer_cast(const host_ptr<T>& ptr) noexcept {
  return host_ptr<U>(ptr, static_cast<U*>(ptr.get()));
}
template <typename U, typename T>
host_ptr<U> dynamic_pointer_cast(const host_ptr<T>& ptr) noexcept {
  U* casted = dynamic_cast<U*>(ptr.get());
  return casted != nullptr ? host_ptr<U>(ptr, casted) : host_ptr<U>();
}
template <typename U, typename T>
host_ptr<U> const_pointer_cast(const host_ptr<T>& ptr) noexcept {
  return host_ptr<U>(ptr, const_cast<U*>(ptr.get()));
}
template <typename U, typename T>
host_ptr<U> reinterpret_pointer_cast(const host_ptr<T>& ptr) noexcept {
  return host_ptr<U>(ptr, reinterpret_cast<U*>(ptr.get()));
}
//@}

//! \name Comparisons (by the pointed-to address, like std::shared_ptr)
//@{
template <typename T>
bool operator==(const host_ptr<T>& lhs, const host_ptr<T>& rhs) noexcept {
  return lhs.get() == rhs.get();
}
template <typename T>
std::strong_ordering operator<=>(const host_ptr<T>& lhs, const host_ptr<T>& rhs) noexcept {
  return std::compare_three_way{}(lhs.get(), rhs.get());
}
template <typename T>
bool operator==(const host_ptr<T>& ptr, std::nullptr_t) noexcept {
  return ptr.get() == nullptr;
}
template <typename T>
std::strong_ordering operator<=>(const host_ptr<T>& ptr, std::nullptr_t) noexcept {
  return std::compare_three_way{}(ptr.get(), static_cast<T*>(nullptr));
}
//@}

}  // namespace mundy

//! \brief Hash a `host_ptr` by its pointed-to address, like `std::hash<std::shared_ptr<T>>`.
template <typename T>
struct std::hash<mundy::host_ptr<T>> {
  std::size_t operator()(const mundy::host_ptr<T>& ptr) const noexcept {
    return std::hash<T*>{}(ptr.get());
  }
};

#endif  // MUNDY_UTILS_HOST_PTR_HPP_
