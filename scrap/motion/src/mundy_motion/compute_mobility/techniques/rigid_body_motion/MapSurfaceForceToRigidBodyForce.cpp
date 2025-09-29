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

/// \file MapSurfaceForceToRigidBodyForce.cpp
/// \brief Definition of the MapSurfaceForceToRigidBodyForce class

// Trilinos libs
#include <Teuchos_ParameterList.hpp>  // for Teuchos::ParameterList

// Mundy libs
#include <mundy_mesh/BulkData.hpp>              // for mundy::mesh::BulkData
#include <mundy_meta/MetaKernelDispatcher.hpp>  // for mundy::meta::MetaKernelDispatcher
#include <mundy_motion/compute_mobility/techniques/rigid_body_motion/MapSurfaceForceToRigidBodyForce.hpp>  // for mundy::motion::...::MapSurfaceForceToRigidBodyForce

namespace mundy {

namespace motion {

namespace compute_mobility {

namespace techniques {

namespace rigid_body_motion {

// \name Constructors and destructor
//{

// TODO(palmerb4): The following won't function properly if the center body nodes are connected to surface nodes.
// Currently, we map surface nodes to body nodes. What if we stored COM force on the actual elements themselves,
// mapped surface nodes to elements, then for each node fetch their connected elements and sum their forces/torques.
// In doing so, race conditions are impossible because each element only has one linker and nodes perform the
// reduction. This design does not work in the case where the linker connects to a body node, as this will double
// count forces at that point.
//
// Honest question, why should I map my rigid body force to my nodes? Just use the rigid body force on the element to
// compute mobility, then propagate rigid body velocity to the dynamic nodes and static nodes.
MapSurfaceForceToRigidBodyForce::MapSurfaceForceToRigidBodyForce(mundy::mesh::BulkData *const bulk_data_ptr,
                                                                 const Teuchos::ParameterList &fixed_params)
    : mundy::meta::MetaKernelDispatcher<MapSurfaceForceToRigidBodyForce>(bulk_data_ptr, fixed_params) {
}
//}

}  // namespace rigid_body_motion

}  // namespace techniques

}  // namespace compute_mobility

}  // namespace motion

}  // namespace mundy
