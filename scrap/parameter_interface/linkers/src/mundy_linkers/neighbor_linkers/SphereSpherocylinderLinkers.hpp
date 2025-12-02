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

#ifndef MUNDY_LINKERS_NEIGHBOR_LINKERS_SPHERESPHEROCYLINDERLINKERS_HPP_
#define MUNDY_LINKERS_NEIGHBOR_LINKERS_SPHERESPHEROCYLINDERLINKERS_HPP_

/// \file SphereSpherocylinderLinkers.hpp
/// \brief Declaration of the SphereSpherocylinderLinkers part class

// C++ core libs
#include <memory>  // for std::shared_ptr, std::unique_ptr
#include <string>  // for std::string
#include <vector>  // for std::vector

// Trilinos libs
#include <stk_topology/topology.hpp>  // for stk::topology

// Mundy includes
#include <mundy_linkers/NeighborLinkers.hpp>  // for mundy::linkers::NeighborLinkers
#include <mundy_meta/FieldReqs.hpp>           // for mundy::meta::FieldReqs
#include <mundy_meta/MeshReqs.hpp>            // for mundy::meta::MeshReqs
#include <mundy_meta/PartReqs.hpp>            // for mundy::meta::PartReqs

namespace mundy {

namespace linkers {

namespace neighbor_linkers {

/// \class SphereSpherocylinderLinkers
/// \brief The static interface for all of Mundy's SphereSpherocylinderLinkers neighbor linkers.
///
/// The design of this class is in accordance with the static interface requirements of mundy::agents::AgentFactory.
///
/// \note This class is a constraint rank assembly part containing neighbor linkers between spheres and spherocylinders.
/// It is a subset of the NeighborLinkers agent.
class SphereSpherocylinderLinkers
    : public mundy::agents::RankedAssembly<mundy::core::make_string_literal("SPHERE_SPHEROCYLINDER_LINKERS"),
                                           stk::topology::CONSTRAINT_RANK, mundy::linkers::NeighborLinkers> {
};  // SphereSpherocylinderLinkers

}  // namespace neighbor_linkers

}  // namespace linkers

}  // namespace mundy

#endif  // MUNDY_LINKERS_NEIGHBOR_LINKERS_SPHERESPHEROCYLINDERLINKERS_HPP_
