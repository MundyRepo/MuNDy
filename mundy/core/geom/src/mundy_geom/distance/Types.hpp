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

#ifndef MUNDY_GEOM_DISTANCE_TYPES_HPP_
#define MUNDY_GEOM_DISTANCE_TYPES_HPP_

// External libs
#include <Kokkos_Core.hpp>

namespace mundy {

/// \addtogroup MundyGeomDistance
/// @{

/// \brief The distance types
///
/// These types are uses for function overloading of our distance functions.
/// This allows for a consistant distance(distance_type, object1, object2) interface
/// with overloads for distance types and object types as necessary.
///
/// All of our distance functions default to SharedNormalSigned distance.
struct Euclidean {};
struct SharedNormalSigned {};

/// \brief Tag selecting the finite-difference-gradient variant of the shared-normal signed distance.
///
/// Uses central-difference approximations for the L-BFGS gradient instead of the analytical
/// gradient employed by the default \c SharedNormalSigned implementation.  Retained for
/// benchmarking and regression testing only — prefer \c SharedNormalSigned for all production use.
struct SharedNormalSignedFiniteDiff {};

/// @}

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_TYPES_HPP_
