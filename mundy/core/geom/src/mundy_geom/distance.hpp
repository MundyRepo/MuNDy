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

#ifndef MUNDY_GEOM_DISTANCE_HPP_
#define MUNDY_GEOM_DISTANCE_HPP_

// All of our headers for computing separation distances between shape primitives

// Point to <shape> distance headers
#include <mundy_geom/distance/PointEllipsoid.hpp>
#include <mundy_geom/distance/PointLine.hpp>
#include <mundy_geom/distance/PointLineSegment.hpp>
#include <mundy_geom/distance/PointPoint.hpp>
#include <mundy_geom/distance/PointSphere.hpp>

// Line to <shape> distance headers
#include <mundy_geom/distance/LineEllipsoid.hpp>
#include <mundy_geom/distance/LineLine.hpp>
// #include <mundy_geom/distance/LineLineSegment.hpp>  // Not implemented
#include <mundy_geom/distance/LineSphere.hpp>

// Line segment to <shape> distance headers
#include <mundy_geom/distance/LineSegmentEllipsoid.hpp>
#include <mundy_geom/distance/LineSegmentLineSegment.hpp>
#include <mundy_geom/distance/LineSegmentSphere.hpp>

// Sphere to <shape> distance headers
#include <mundy_geom/distance/SphereEllipsoid.hpp>
#include <mundy_geom/distance/SphereSphere.hpp>

// Ellipsoid to <shape> distance headers
#include <mundy_geom/distance/EllipsoidEllipsoid.hpp>

// Circle3D to <shape> distance headers
#include <mundy_geom/distance/Circle3DCircle3D.hpp>

#endif  // MUNDY_GEOM_DISTANCE_HPP_