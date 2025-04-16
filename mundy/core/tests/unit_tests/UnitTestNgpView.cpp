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

// External libs
#include <gmock/gmock.h>  // for EXPECT_THAT, HasSubstr, etc
#include <gtest/gtest.h>  // for TEST, ASSERT_NO_THROW, etc

// C++ core libs
#include <iostream>
#include <stdexcept>  // for logic_error, invalid_argument, etc

// Kokkos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize

// Mundy libs
#include <mundy_core/NgpView.hpp>  // for NgpView

namespace mundy {

namespace core {

namespace {

template <typename SomeNgpView>
void run_some_device_func(SomeNgpView ngp_view) {
  ngp_view.sync_to_device();

  auto d_view = ngp_view.view_device();
  Kokkos::parallel_for("SomeKernel", d_view.extent(0), KOKKOS_LAMBDA(const int i) { d_view(i) = i % 2; });

  ngp_view.modify_on_device();
}

template <typename SomeNgpView>
void run_some_host_func(SomeNgpView ngp_view) {
  ngp_view.sync_to_host();

  auto h_view = ngp_view.view_host();
  for (size_t i = 0; i < h_view.extent(0); ++i) {
    h_view(i) = std::sqrt(h_view(i));  // Some host-bound operation
  }

  ngp_view.modify_on_host();  // Of course, if you don't modify the data, you don't need to call this.
}

void run_some_kernel(NgpView<int*> ngp_view) {
  ngp_view.sync_to_device();

  auto d_view = ngp_view.view_device();
  Kokkos::parallel_for("SomeKernel", d_view.extent(0), KOKKOS_LAMBDA(const int i) { d_view(i) = i % 2; });

  ngp_view.modify_on_device();
}

TEST(NgpViewTest, SimpleUsage) {
  // Because NgpView inherits from Kokkos::DualView, we need not test the DualView functionality here.
  // Instead, the focus here is on showing how NgpView can be used in a typical Mundy application
  // while adding in some EXPECT calls to demonstrate that the NgpView is behaving as expected.

  // If you don't provide a layout or a memory space, the default will be used. The default layout depends on the
  // default memory space and the default memory space is Kokkos::DefaultExecutionSpace::memory_space.
  NgpView<int*> ngp_view("ngp_view", 10);
  constexpr bool is_device_host_accessible =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, NgpView<int*>::t_dev::memory_space>::accessible;

  // Modify the host view
  auto h_view = ngp_view.view_host();
  for (size_t i = 0; i < 10; ++i) {
    h_view(i) = i;
  }

  // Mark the host view as modified
  ngp_view.modify_on_host();

  // Sync the host view to the device view
  // Because we marked the data modified on the host, a call to sync_to_device() will copy the data from the host to the
  // device, and reset the modified flags.

  // If we are using a space that can access host memory, then the data will not be copied and the need_sync_to_device()
  // flag will not be set.
  if constexpr (is_device_host_accessible) {
    EXPECT_EQ(ngp_view.need_sync_to_device(), false);
  } else {
    EXPECT_EQ(ngp_view.need_sync_to_device(), true);
  }

  ngp_view.sync_to_device();  // No op for host space
  EXPECT_EQ(ngp_view.need_sync_to_device(), false) << "Any subsequent calls to sync_to_device() will not copy the data again, because the data is already in sync.";


  // Modify on device, mark the device view as modified, sync to the host, and check that the host data has been updated.
  run_some_kernel(ngp_view);  // Don't use Kokkos lambdas in gtest macros.
  ngp_view.sync_to_host();
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(h_view(i), i % 2);
  }

  // From a practical perspective, the reason we use this pattern is because it removes the hassle of memory management
  // between the host and device. If you add in a function that needs the data to be up to date on the device, you just
  // call sync_to_device() at the start of the function and mark any changes with modify_on_device(). This is
  // demonstrated in run_some_func/ run_some_host_func above.
  //
  // You should always attempt to minimize the interleaving of host and device code, but so long as you encapsulate your
  // functions with syncs/modified, you are free to add or delete new code without worrying about memory management.
  run_some_device_func(ngp_view);
  run_some_host_func(ngp_view);
}

TEST(NgpViewTest, Functionality) {
  // Allocate without initializing
  NgpView<int*> ngp_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "ngp_view"), 10);

  // Initialize the view
  auto h_view = ngp_view.view_host();
  for (size_t i = 0; i < 10; ++i) {
    h_view(i) = i;
  }

  // Resize the view
  ngp_view.resize(20);
  EXPECT_EQ(ngp_view.extent(0), 20);

  // Specify the layout of a 2D view. The fact that this is 2D is expressed by the type**
  // Use a 2D view
  NgpView<int**, Kokkos::LayoutLeft> ngp_view2("ngp_view2", 10, 3);
  auto h_view2 = ngp_view2.view_host();
  for (size_t i = 0; i < 10; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      h_view2(i, j) = i + j;
    }
  }

  // Construct from existing device and host views
  auto an_existing_device_view = ngp_view2.view_device();
  auto an_existing_host_view = ngp_view2.view_host();
  NgpView<int**, Kokkos::LayoutLeft> ngp_view3(an_existing_device_view, an_existing_host_view);

  // For power users, if you really want to use our NgpView with a different memory space, you can do so.
  // Just use NgpViewT directly.
  NgpViewT<int**, Kokkos::LayoutLeft, stk::ngp::ExecSpace> ngp_view4("ngp_view4", 10, 3);
}

}  // namespace

}  // namespace core

}  // namespace mundy
