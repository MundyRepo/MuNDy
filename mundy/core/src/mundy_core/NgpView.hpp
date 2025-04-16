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

#ifndef MUNDY_CORE_NGPVIEW_HPP_
#define MUNDY_CORE_NGPVIEW_HPP_

// C++ core
#include <string>

// Kokkos
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

// STK
#include <stk_util/ngp/NgpSpaces.hpp>

namespace mundy {

namespace core {

/// \brief NgpViewT is an enhanced DualView with a slightly expanded interface.
///
/// NgpViewT inherits from Kokkos::DualView, exposing all its functionality while
/// adding convenience methods for marking and synchronizing modifications between
/// host and device. This class replicates the constructors of DualView and provides
/// detailed documentation for users who may be new to Kokkos views.
///
template <class DataType, class... Properties>
class NgpViewT : public Kokkos::DualView<DataType, Properties...> {
 public:
  // Type aliases for brevity.
  using t_dv   = Kokkos::DualView<DataType, Properties...>;
  using t_dev  = typename t_dv::t_dev;
  using t_host = typename t_dv::t_host;

  //! \name Constructors
  //@{

  /// \brief Default constructor.
  ///
  /// Constructs an empty NgpViewT object. Both the device and host views
  /// are constructed using their default constructors, meaning no memory is
  /// allocated until you assign or allocate data later. The internal "modified"
  /// flags are initialized to indicate that neither view is marked as modified.
  NgpViewT() = default;

  /// \brief Allocates device and host views with the specified dimensions.
  ///
  /// This constructor behaves similarly to the corresponding constructor of
  /// Kokkos::DualView. The first argument is a user-defined label that can help
  /// with debugging or profiling. The remaining arguments specify the dimensions
  /// of the view. For a multi-dimensional view, only specify the dimensions that
  /// are nonzero; additional dimensions default to an implementation-defined value.
  ///
  /// \param label A string used to label the view.
  /// \param n0    The size of the first dimension.
  /// \param n1    (Optional) The size of the second dimension.
  /// \param n2    (Optional) The size of the third dimension.
  /// \param n3    (Optional) The size of the fourth dimension.
  /// \param n4    (Optional) The size of the fifth dimension.
  /// \param n5    (Optional) The size of the sixth dimension.
  /// \param n6    (Optional) The size of the seventh dimension.
  /// \param n7    (Optional) The size of the eighth dimension.
  NgpViewT(const std::string& label,
          const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : t_dv(label, n0, n1, n2, n3, n4, n5, n6, n7)
  {}

  /// \brief Allocates device and host views using a property object.
  ///
  /// This constructor lets you wrap up various construction properties (such as
  /// memory space, label, or initialization behavior) in a ViewCtorProp object.
  /// The subsequent integer arguments specify the dimensions of the view, similar
  /// to the previous constructor.
  ///
  /// \tparam P A parameter pack representing properties wrapped in the ViewCtorProp.
  /// \param arg_prop A ViewCtorProp object encapsulating construction properties.
  /// \param n0       The size of the first dimension.
  /// \param n1       (Optional) The size of the second dimension.
  /// \param n2       (Optional) The size of the third dimension.
  /// \param n3       (Optional) The size of the fourth dimension.
  /// \param n4       (Optional) The size of the fifth dimension.
  /// \param n5       (Optional) The size of the sixth dimension.
  /// \param n6       (Optional) The size of the seventh dimension.
  /// \param n7       (Optional) The size of the eighth dimension.
  template <class... P>
  NgpViewT(const Kokkos::Impl::ViewCtorProp<P...>& arg_prop,
          std::enable_if_t<!Kokkos::Impl::ViewCtorProp<P...>::has_pointer, size_t> const n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
          const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
      : t_dv(arg_prop, n0, n1, n2, n3, n4, n5, n6, n7)
  {}

  /// \brief Shallow copy constructor.
  ///
  /// Constructs a new NgpViewT as a shallow copy of the source view. The underlying
  /// host and device views will refer to the same memory as those in the source.
  /// Both the data pointers and the "modified" flags are copied.
  ///
  /// \tparam DT The data type of the source view.
  /// \tparam DP The properties of the source view.
  /// \param src The NgpViewT from which to create a copy.
  template <typename DT, typename... DP>
  NgpViewT(const NgpViewT<DT, DP...>& src)
      : t_dv(src)
  {}

  /// \brief Constructs a subview of an existing NgpViewT.
  ///
  /// This constructor allows you to create a new NgpViewT that is a subview of an
  /// existing one. This is especially useful for extracting lower-dimensional
  /// slices from higher-dimensional views. The additional arguments specify the
  /// indices or ranges to select.
  ///
  /// \tparam DT   The data type of the source view.
  /// \tparam DP   The properties of the source view.
  /// \tparam Arg0 The type of the first argument for slicing.
  /// \tparam Args The types of additional slicing arguments.
  /// \param src   The source NgpViewT from which the subview is created.
  /// \param arg0  The first argument for defining the subview.
  /// \param args  Additional arguments for defining the subview.
  template <class DT, class... DP, class Arg0, class... Args>
  NgpViewT(const NgpViewT<DT, DP...>& src, const Arg0& arg0, Args... args)
      : t_dv(src, arg0, args...)
  {}

  /// \brief Constructs an NgpViewT from existing device and host views.
  ///
  /// This constructor creates an NgpViewT using already allocated device and host
  /// views. It assumes that the two views are synchronized (i.e., contain identical
  /// data) at the time of construction. After constructing the NgpViewT, you can use
  /// the sync() and modify() methods to manage synchronization.
  ///
  /// \param d_view_ A pre-existing device view.
  /// \param h_view_ A pre-existing host view. This must be the host mirror of the device view.
  NgpViewT(const t_dev& d_view_, const t_host& h_view_)
      : t_dv(d_view_, h_view_)
  {}
  //@}

  /// \brief Mark the host view as modified.
  ///
  /// Call this method after updating the host view so that the DualView is aware
  /// that the device view may now be out of date.
  inline void modify_on_host() {
    t_dv::modify_host();
  }

  /// \brief Mark the device view as modified.
  ///
  /// Call this method after updating the device view so that the DualView is aware
  /// that the host view may now be out of date.
  inline void modify_on_device() {
    t_dv::modify_device();
  }

  /// \brief Abstract method for marking the view as modified.
  template <typename Space>
  inline void modify_on() {
    t_dv::template modify<Space>();
  }

  /// \brief Synchronize the host view to the device view if needed.
  ///
  /// If the device view has been modified more recently than the host view,
  /// this function performs a deep copy from the device view to the host view.
  inline void sync_to_host() {
    t_dv::sync_host();
  }

  /// \brief Synchronize the device view to the host view if needed.
  ///
  /// If the host view has been modified more recently than the device view,
  /// this function performs a deep copy from the host view to the device view.
  inline void sync_to_device() {
    t_dv::sync_device();
  }

  /// \brief Abstract method for synchronizing the view.
  template <typename Space>
  inline void sync_to() {
    t_dv::template sync<Space>();
  }

  /// \brief Return if we need to sync to the host.
  inline bool need_sync_to_host() const {
    return t_dv::need_sync_host();
  }

  /// \brief Return if we need to sync to the device.
  inline bool need_sync_to_device() const {
    return t_dv::need_sync_device();
  }

  /// \brief Abstract method for checking if we need to sync.
  template <typename Space>
  inline bool need_sync_to() const {
    return t_dv::template need_sync<Space>();
  }
};  // NgpViewT

/// \brief Our default NgpView type for use in Mundy.
///
/// Unlike NgpViewT, we follow stk::ngp conventions by using stk::ngp::ExecSpace as our
/// chosen device space.
template <class DataType, class... Properties>
using NgpView = NgpViewT<DataType, Properties..., typename stk::ngp::ExecSpace::memory_space>;

}  // namespace core

}  // namespace mundy

#endif  // MUNDY_CORE_NGPVIEW_HPP_
