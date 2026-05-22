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

#include <Kokkos_Core.hpp>
#include <iostream>
#include <mundy_utils/NgpView.hpp>  // for mundy::NgpView
#include <vector>

//---------------------------------------------------------------------------------------------------------------------//
// NgpView basics.
//---------------------------------------------------------------------------------------------------------------------//
void ngp_view_basics_example() {
  std::cout << "\n--- NgpView Basics ---\n" << std::endl;

  /*
    Particle simulations need data on both the CPU (host) and the GPU
    (device).  The naive approach -- two separate arrays -- requires manual
    bookkeeping: you must remember which copy is current, copy in the right
    direction before every kernel, and not forget to copy back when reading
    results on the CPU.  It is easy to get wrong and the bugs are subtle
    (stale data that looks valid).

    mundy::NgpView<T*> solves this with one logical array backed by two
    physical copies.  The sync/modify protocol makes the bookkeeping explicit
    and auditable:

      The two rules:
        1.  SYNC before reading in a memory space.
        2.  MARK MODIFIED after writing in a memory space.

      The four operations:
        view_host()       / view_device()       -- access the view for reading or writing
        modify_on_host()  / modify_on_device()  -- declare that this side was written
        sync_to_host()    / sync_to_device()    -- copy stale data from the other side; already
                                                   a no-op when the target is up to date
        need_sync_to_host() / need_sync_to_device() -- query sync state for diagnostics and
                                                       assertions; NOT a guard for sync calls

    NgpView<T*> inherits from Kokkos::DualView<T*>.  The "*" in the type
    parameter indicates a 1D dynamically-sized array, following Kokkos
    notation.  A 2D array would be NgpView<T**>.

    You never read from the stale side.  You never write without marking
    modified.  If you follow those two rules, the dual view is always
    consistent.
  */

  const int n = 8;
  mundy::NgpView<double*> data("particle_data", n);

  /*
    Step 1: initialize on the host.
    Get the host view, write, then mark the host side as modified.
  */
  {
    auto h = data.view_host();
    for (int i = 0; i < n; ++i) {
      h(i) = static_cast<double>(i);
    }
    data.modify_on_host();
  }

  std::cout << "Initialized on host: [";
  {
    auto h = data.view_host();
    for (int i = 0; i < n; ++i) {
      std::cout << h(i);
      if (i < n - 1) std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  /*
    Step 2: run a Kokkos kernel on the device.
    Sync first (copies host -> device since host was last modified), then
    get the device view, launch the kernel, and mark the device modified.
  */
  data.sync_to_device();
  {
    auto d = data.view_device();
    Kokkos::parallel_for("scale_by_2", n, KOKKOS_LAMBDA(const int i) { d(i) *= 2.0; });
    Kokkos::fence();
    data.modify_on_device();
  }

  /*
    Step 3: read results back on the host.
    Sync device -> host, then read.
  */
  data.sync_to_host();
  std::cout << "After device kernel (x2): [";
  {
    auto h = data.view_host();
    for (int i = 0; i < n; ++i) {
      std::cout << h(i);
      if (i < n - 1) std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Sync state inspection with need_sync_to_*.
//---------------------------------------------------------------------------------------------------------------------//
void sync_state_inspection_example() {
  std::cout << "\n--- Sync State Inspection ---\n" << std::endl;

  /*
    sync_to_device() and sync_to_host() already check internally whether a
    copy is necessary.  If the target side is current they return immediately
    without touching memory.  You should never guard a sync call:

      if (v.need_sync_to_device()) v.sync_to_device();  // WRONG -- redundant
      v.sync_to_device();                                // RIGHT -- call unconditionally

    need_sync_to_host() and need_sync_to_device() exist for diagnostics and
    assertions.  Use them to verify that your protocol is correct, not to
    decide whether to call sync.
  */

  const int n = 4;
  mundy::NgpView<double*> v("v", n);

  auto h = v.view_host();
  for (int i = 0; i < n; ++i) h(i) = static_cast<double>(i + 1);
  v.modify_on_host();

  // Diagnostic: confirm the expected state before syncing.
  std::cout << "After modifying host:" << std::endl;
  std::cout << "  need_sync_to_device? " << (v.need_sync_to_device() ? "yes" : "no") << "  (expected: yes)" << std::endl;
  std::cout << "  need_sync_to_host?   " << (v.need_sync_to_host() ? "yes" : "no") << "  (expected: no)" << std::endl;

  // Sync unconditionally -- sync_to_device already skips the copy when not needed.
  v.sync_to_device();

  // Diagnostic: confirm the flag was cleared.
  std::cout << "After sync_to_device:" << std::endl;
  std::cout << "  need_sync_to_device? " << (v.need_sync_to_device() ? "yes" : "no") << "  (expected: no)" << std::endl;

  // Calling sync_to_device a second time is safe: it sees no work and returns immediately.
  v.sync_to_device();
  std::cout << "After redundant sync_to_device (no-op):" << std::endl;
  std::cout << "  need_sync_to_device? " << (v.need_sync_to_device() ? "yes" : "no") << "  (expected: no)" << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// 2D NgpView for particle positions.
//---------------------------------------------------------------------------------------------------------------------//
void two_dim_ngp_view_example() {
  std::cout << "\n--- 2D NgpView for Particle Positions ---\n" << std::endl;

  /*
    A common layout for particle positions is a 2D array: rows are particles,
    columns are the three spatial components (x, y, z).

    NgpView<double**> stores this as a 2D Kokkos view.  Access uses two
    indices: view(particle_index, component).

    This is more cache-friendly for kernels that loop over particles and
    access all three coordinates together than a separate view per component.
  */

  const int num_particles = 5;
  const int num_dims = 3;

  mundy::NgpView<double**> positions("positions", num_particles, num_dims);

  // Initialize positions on the host.
  {
    auto h = positions.view_host();
    for (int p = 0; p < num_particles; ++p) {
      h(p, 0) = static_cast<double>(p);  // x = particle index
      h(p, 1) = 0.0;
      h(p, 2) = 0.0;
    }
    positions.modify_on_host();
  }

  // On the device, shift all x-positions by +1.
  positions.sync_to_device();
  {
    auto d = positions.view_device();
    Kokkos::parallel_for("shift_x", num_particles, KOKKOS_LAMBDA(const int p) { d(p, 0) += 1.0; });
    Kokkos::fence();
    positions.modify_on_device();
  }

  // Read results back.
  positions.sync_to_host();
  std::cout << "Positions after +1 shift in x:" << std::endl;
  {
    auto h = positions.view_host();
    for (int p = 0; p < num_particles; ++p) {
      std::cout << "  particle " << p << ": (" << h(p, 0) << ", " << h(p, 1) << ", " << h(p, 2) << ")" << std::endl;
    }
  }
}

//---------------------------------------------------------------------------------------------------------------------//
// Putting it together: parallel norm computation.
//---------------------------------------------------------------------------------------------------------------------//
void parallel_norm_example() {
  std::cout << "\n--- Parallel Norm Computation ---\n" << std::endl;

  /*
    A realistic use case: given N particle positions, compute the Euclidean
    norm of each position vector in parallel on the device, then read the
    norms back on the host.

    This pattern will appear in production Mundy code: fill data on the
    host, move to device, run a kernel that reads positions and writes a
    derived quantity, then optionally pull the result back for output or
    further host-side processing.
  */

  const int N = 6;

  mundy::NgpView<double**> pos("pos", N, 3);
  mundy::NgpView<double*> norms("norms", N);

  // Fill positions on the host.
  {
    auto h = pos.view_host();
    // clang-format off
    h(0, 0) = 1.0; h(0, 1) = 0.0; h(0, 2) = 0.0;  // norm = 1
    h(1, 0) = 3.0; h(1, 1) = 4.0; h(1, 2) = 0.0;  // norm = 5
    h(2, 0) = 1.0; h(2, 1) = 1.0; h(2, 2) = 1.0;  // norm = sqrt(3)
    h(3, 0) = 0.0; h(3, 1) = 0.0; h(3, 2) = 0.0;  // norm = 0
    h(4, 0) = 2.0; h(4, 1) = 0.0; h(4, 2) = 0.0;  // norm = 2
    h(5, 0) = 1.0; h(5, 1) = 2.0; h(5, 2) = 2.0;  // norm = 3
    // clang-format on
    pos.modify_on_host();
  }

  // Compute norms on the device.
  pos.sync_to_device();
  {
    auto d_pos = pos.view_device();
    auto d_norms = norms.view_device();
    Kokkos::parallel_for(
        "compute_norms", N, KOKKOS_LAMBDA(const int i) {
          double x = d_pos(i, 0);
          double y = d_pos(i, 1);
          double z = d_pos(i, 2);
          d_norms(i) = Kokkos::sqrt(x * x + y * y + z * z);
        });
    Kokkos::fence();
    norms.modify_on_device();
  }

  // Read norms on the host.
  norms.sync_to_host();
  {
    auto h = norms.view_host();
    std::cout << "Particle norms:" << std::endl;
    for (int i = 0; i < N; ++i) {
      std::cout << "  particle " << i << ": " << h(i) << std::endl;
    }
  }
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  ngp_view_basics_example();
  sync_state_inspection_example();
  two_dim_ngp_view_example();
  parallel_norm_example();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
