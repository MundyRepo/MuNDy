#
# A) Define the subpackage
#
tribits_subpackage(Shape)

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
  src/mundy_shapes/ComputeAABB.hpp
  src/mundy_shapes/compute_aabb/kernels/Sphere.hpp
  src/mundy_shapes/compute_aabb/kernels/Spherocylinder.hpp
  src/mundy_shapes/compute_aabb/kernels/SpherocylinderSegment.hpp
  src/mundy_shapes/ComputeBoundingRadius.hpp
  src/mundy_shapes/compute_bounding_radius/kernels/Sphere.hpp
  src/mundy_shapes/compute_bounding_radius/kernels/Spherocylinder.hpp
  src/mundy_shapes/compute_bounding_radius/kernels/SpherocylinderSegment.hpp
  src/mundy_shapes/ComputeOBB.hpp
  src/mundy_shapes/compute_obb/kernels/Sphere.hpp
  src/mundy_shapes/compute_obb/kernels/Spherocylinder.hpp
  src/mundy_shapes/compute_obb/kernels/SpherocylinderSegment.hpp
  src/mundy_shapes/declare_and_initialize_shapes/techniques/GridCoordinateMapping.hpp
  src/mundy_shapes/declare_and_initialize_shapes/techniques/GridOfSpheres.hpp
  src/mundy_shapes/DeclareAndInitShapes.hpp
  src/mundy_shapes/Shapes.hpp
  src/mundy_shapes/Spheres.hpp
  src/mundy_shapes/SpherocylinderSegments.hpp
  src/mundy_shapes/Spherocylinders.hpp
  )

SET(SOURCES 
  src/mundy_shapes/compute_aabb/kernels/Sphere.cpp
  src/mundy_shapes/compute_aabb/kernels/Spherocylinder.cpp
  src/mundy_shapes/compute_aabb/kernels/SpherocylinderSegment.cpp
  src/mundy_shapes/compute_bounding_radius/kernels/Sphere.cpp
  src/mundy_shapes/compute_bounding_radius/kernels/Spherocylinder.cpp
  src/mundy_shapes/compute_bounding_radius/kernels/SpherocylinderSegment.cpp
  src/mundy_shapes/compute_obb/kernels/Sphere.cpp
  src/mundy_shapes/compute_obb/kernels/Spherocylinder.cpp
  src/mundy_shapes/compute_obb/kernels/SpherocylinderSegment.cpp
  src/mundy_shapes/declare_and_initialize_shapes/techniques/GridOfSpheres.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_shapes
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_shapes
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()