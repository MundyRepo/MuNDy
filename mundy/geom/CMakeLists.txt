#
# A) Define the subpackage
#
tribits_subpackage(Geom)

# By default we enable all optional packages. They can be disabled by the user if they are not needed.
GLOBAL_SET(${PACKAGE_NAME}_ENABLE_ALL_OPTIONAL_PACKAGES ON)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use 
# find src/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort
SET(NOINSTALLHEADERS 
  src/mundy_geom/distance/DistanceMetrics.hpp
  src/mundy_geom/distance/EllipsoidEllipsoid.hpp
  src/mundy_geom/distance.hpp
  src/mundy_geom/distance/LineEllipsoid.hpp
  src/mundy_geom/distance/LineLine.hpp
  src/mundy_geom/distance/LinePlane.hpp
  src/mundy_geom/distance/LineSegmentEllipsoid.hpp
  src/mundy_geom/distance/LineSegmentLineSegment.hpp
  src/mundy_geom/distance/LineSegmentPlane.hpp
  src/mundy_geom/distance/LineSegmentSphere.hpp
  src/mundy_geom/distance/LineSphere.hpp
  src/mundy_geom/distance/PlaneEllipsoid.hpp
  src/mundy_geom/distance/PlanePlane.hpp
  src/mundy_geom/distance/PlaneSphere.hpp
  src/mundy_geom/distance/PointEllipsoid.hpp
  src/mundy_geom/distance/PointLine.hpp
  src/mundy_geom/distance/PointLineSegment.hpp
  src/mundy_geom/distance/PointPlane.hpp
  src/mundy_geom/distance/PointPoint.hpp
  src/mundy_geom/distance/PointSphere.hpp
  src/mundy_geom/distance/SphereEllipsoid.hpp
  src/mundy_geom/distance/SphereSphere.hpp
  src/mundy_geom/distance/Types.hpp
  src/mundy_geom/primitives/AABB.hpp
  src/mundy_geom/primitives/Ellipsoid.hpp
  src/mundy_geom/primitives.hpp
  src/mundy_geom/primitives/Line.hpp
  src/mundy_geom/primitives/LineSegment.hpp
  src/mundy_geom/primitives/Point.hpp
  src/mundy_geom/primitives/Sphere.hpp
  src/mundy_geom/primitives/Spherocylinder.hpp
  src/mundy_geom/primitives/SpherocylinderSegment.hpp
  src/mundy_geom/primitives/VSegment.hpp
  src/mundy_geom/compute_aabb.hpp
  src/mundy_geom/compute_bounding_radius.hpp
  )

SET(SOURCES
  src/mundy_geom/a_necessary_empty_file.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_geom
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_geom
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()