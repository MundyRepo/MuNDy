#
# A) Define the subpackage
#
tribits_subpackage(Math)

# By default we enable all optional packages. They can be disabled by the user if they are not needed.
global_set(${PACKAGE_NAME}_ENABLE_ALL_OPTIONAL_PACKAGES ON)

#
# B) Define the header and source files
#

# Headers are relative to the tribits_include_directory
# We don't recommend typing these by hand. Use
# find src/ -type f -name "*.hpp" | sed 's/^.\/\///' | sort
# find src/ -type f -name "*.cpp" | sed 's/^.\/\///' | sort
set(NOINSTALLHEADERS
  src/mundy_math/Accessor.hpp
  src/mundy_math/Array.hpp
  src/mundy_math/AutoDiffScalar.hpp
  src/mundy_math/belos_solver.hpp
  src/mundy_math/convex_spaces.hpp
  src/mundy_math/cqpp.hpp
  src/mundy_math/Hilbert.hpp
  src/mundy_math/impl/AccessorImpl.hpp
  src/mundy_math/impl/ArrayImpl.hpp
  src/mundy_math/impl/belos_solver_impl.hpp
  src/mundy_math/impl/linear_ops_impl.hpp
  src/mundy_math/impl/MatrixImpl.hpp
  src/mundy_math/impl/minimize_impl.hpp
  src/mundy_math/impl/QuaternionImpl.hpp
  src/mundy_math/impl/solver_backends_impl.hpp
  src/mundy_math/impl/VectorImpl.hpp
  src/mundy_math/lcp.hpp
  src/mundy_math/linear_ops.hpp
  src/mundy_math/linear_system.hpp
  src/mundy_math/MaskedAccessor.hpp
  src/mundy_math/Matrix3.hpp
  src/mundy_math/Matrix.hpp
  src/mundy_math/minimize.hpp
  src/mundy_math/NumTraits.hpp
  src/mundy_math/Quaternion.hpp
  src/mundy_math/residuals.hpp
  src/mundy_math/ScalarBinaryOpTraits.hpp
  src/mundy_math/ScalarWrapper.hpp
  src/mundy_math/ShiftedAccessor.hpp
  src/mundy_math/solver_backends.hpp
  src/mundy_math/StridedAccessor.hpp
  src/mundy_math/Tolerance.hpp
  src/mundy_math/TransposedAccessor.hpp
  src/mundy_math/Vector3.hpp
  src/mundy_math/Vector.hpp
  src/mundy_math/zmort.hpp
)

set(SOURCES src/mundy_math/a_necessary_empty_file.cpp)

tribits_configure_file(MundyMath_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_math
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/MundyMath_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_math
  FILES_MATCHING
  PATTERN "*.hpp"
  SOURCES ${SOURCES}
)

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()
