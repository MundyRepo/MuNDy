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

#ifndef MUNDY_GEOM_AGGREGATES_HPP_
#define MUNDY_GEOM_AGGREGATES_HPP_

/// \brief All of our headers for aggregates of primitive geometric objects
///
/// These aggregates follow the data oriented design pattern with view objects that provide object-like access to
/// object data stored within bucketed structure of arrays while maintaining some independence from how the data is
/// stored. This is best explained by example.
///
/// Consider a collection of sphere-like objects stored as PARTICLE topology stk::mesh::Entity objects with element-rank
/// radius and node-rank center coordinates. We store the stk::mesh::NgpField for the radius and center coordinates
/// within the SphereData aggregate. This struct has no methods, only data members, and is meant to organize the data
/// for the spheres to simplify interfaces and to allow for functional dispatch based on the data type. Our current
/// SphereData struct allows users to either provide a field for the radius and center or to provide a single value for
/// each, meant to be shared by all spheres. Use create_sphere_data to aid in creating the SphereData struct.
///
/// If you wish to augment the SphereData struct with additional data, you can create a new struct that meets the
/// ValidDefaultSphereDataType concept; aka, it has the same data members as SphereData. Alternatively, if you don't
/// want to name your center and radius data members the same as SphereData or if you want to use a different access
/// pattern, you can create a new struct and provide a specialization of SphereDataTraits for your new struct. Doing so
/// allows you to seamlessly use our SphereEntityView view object with your new struct. Note, we suggest that
/// augmentations only add data members and not methods to maintain the data oriented design pattern. Aggregates hold
/// data and non-member functions operate on that data, making for more extensible and maintainable code.
///
/// The SphereEntityView struct is a view object that provides object-like access to the sphere data stored in the mesh.
/// It is constructed to give an identical interface to the Sphere primitive object, but it is constructed from the
/// entity and the mesh. Most importantly, it will not access the field data until the user requests it. This way, if
/// you don't use the node coordinates of the sphere, we won't fetch them from the mesh. This is important for
/// performance when using complex aggregates with many fields and is a key benefit of the data oriented design pattern.
/// While our view pattern requires repeated object construction, we have optimized our objects to have minimal overhead
/// with less than a 0.1% overhead compared to direct field access.
///
/// The above is general to all of our primitive types. All you need to do is replace Sphere with the primitive type you
/// are interested in. They all implement the same pattern.
#include <mundy_geom/aggregates/GeomAABB.hpp>
#include <mundy_geom/aggregates/GeomEllipsoid.hpp>
#include <mundy_geom/aggregates/GeomLine.hpp>
#include <mundy_geom/aggregates/GeomLineSegment.hpp>
#include <mundy_geom/aggregates/GeomPoint.hpp>
#include <mundy_geom/aggregates/GeomSphere.hpp>
#include <mundy_geom/aggregates/GeomSpherocylinder.hpp>
#include <mundy_geom/aggregates/GeomSpherocylinderSegment.hpp>

#endif  // MUNDY_GEOM_AGGREGATES_HPP_