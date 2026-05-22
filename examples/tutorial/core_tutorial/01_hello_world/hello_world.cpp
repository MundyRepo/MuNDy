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

#include <Kokkos_Core.hpp>  // for Kokkos::ScopeGuard
#include <iostream>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_REQUIRE

//---------------------------------------------------------------------------------------------------------------------//
// Hello World example.
//---------------------------------------------------------------------------------------------------------------------//
void hello_world_example() {
  /*
    Welcome to Mundy.

    Mundy (Multi-body Nonlocal Dynamics) is a portability-first C++ library
    for geometric computation in soft-matter and biophysics simulations.  It
    is built on Kokkos and provides three layered packages:

      mundy_utils  -- portable error checking, compile-time strings, type
                      utilities, and host/device data structures
      mundy_math   -- fixed-size vectors, matrices, and quaternions with full
                      arithmetic, plus non-owning views into existing storage
      mundy_geom   -- geometric primitives (spheres, rods, ellipsoids, …),
                      distance queries, bounding geometry, rigid transforms,
                      and periodic boundary conditions

    This tutorial series introduces each package in turn.  By the end you
    will know how to represent physical bodies, query their geometry, and
    manage host/device data with Kokkos -- all the building blocks for a
    particle simulation.

    The one rule you must follow to use any Kokkos-based library: initialize
    Kokkos before any Kokkos code runs, and finalize it when you are done.
    Kokkos::ScopeGuard handles both automatically.  It initializes Kokkos
    when constructed (parsing argc/argv for runtime options) and calls
    Kokkos::finalize() when it goes out of scope, even if the program exits
    early via an exception.  Any code that touches Kokkos must live inside
    the scope of this guard.
  */

  std::cout << "Hello from Mundy!" << std::endl;

  /*
    Mundy's error system lives in mundy_utils and is the subject of the next
    tutorial.  Here is the smallest possible preview.

    MUNDY_THROW_REQUIRE checks an assertion unconditionally.  On the host it
    throws the requested exception type if the assertion is false.  On the
    device it aborts (you cannot throw across a GPU kernel boundary).

    The message can be a plain string literal or a chain of values built with
    the sink() helper you will meet in Tutorial 02.
  */
  MUNDY_THROW_REQUIRE(1 + 1 == 2, std::logic_error, "math is broken");

  std::cout << "Assertion passed. Mundy is ready." << std::endl;
}

//---------------------------------------------------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------------------------------------------------//
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  hello_world_example();

  return 0;
}

//---------------------------------------------------------------------------------------------------------------------//
