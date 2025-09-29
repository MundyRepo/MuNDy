#
# A) Define the subpackage
#
tribits_subpackage(Mesh)

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
  src/mundy_mesh/Aggregate.hpp
  src/mundy_mesh/BulkData.hpp
  src/mundy_mesh/DeclareEntities.hpp
  src/mundy_mesh/FieldViews.hpp
  src/mundy_mesh/fmt_stk_types.hpp
  src/mundy_mesh/ForEachEntity.hpp
  src/mundy_mesh/GetNgpLinkData.hpp
  src/mundy_mesh/impl/HostDeviceCOOSynchronizer.hpp
  src/mundy_mesh/impl/HostDeviceCRSSynchronizer.hpp
  src/mundy_mesh/impl/HostDeviceSynchronizer.hpp
  src/mundy_mesh/impl/LinkedBucketConn.hpp
  src/mundy_mesh/impl/NgpCOOToCRSSynchronizer.hpp
  src/mundy_mesh/impl/NgpFieldBLASImpl.hpp
  src/mundy_mesh/impl/NgpLinkMetaData.hpp
  src/mundy_mesh/LinkCOOData.hpp
  src/mundy_mesh/LinkCRSBucketConn.hpp
  src/mundy_mesh/LinkCRSData.hpp
  src/mundy_mesh/LinkCRSPartition.hpp
  src/mundy_mesh/LinkData.hpp
  src/mundy_mesh/LinkDeclarationRequests.hpp
  src/mundy_mesh/LinkDestructionRequests.hpp
  src/mundy_mesh/LinkMetaData.hpp
  src/mundy_mesh/MeshBuilder.hpp
  src/mundy_mesh/MetaData.hpp
  src/mundy_mesh/NgpEntityPool.hpp
  src/mundy_mesh/NgpFieldBLAS.hpp
  src/mundy_mesh/NgpForEachLink.hpp
  src/mundy_mesh/NgpLinkData.hpp
  src/mundy_mesh/NgpUtils.hpp
  src/mundy_mesh/StringToSelector.hpp
  src/mundy_mesh/StringToTopology.hpp
  src/mundy_mesh/Types.hpp
  src/mundy_mesh/utils/DestroyFlaggedEntities.hpp
  src/mundy_mesh/utils/FillFieldWithValue.hpp
  src/mundy_mesh/utils/SelectorEval.hpp
  src/mundy_mesh/utils/SelectorLexem.hpp
  src/mundy_mesh/utils/SelectorNode.hpp
  src/mundy_mesh/utils/SelectorParser.hpp
  )

SET(SOURCES 
  src/mundy_mesh/DeclareEntities.cpp
  src/mundy_mesh/MeshBuilder.cpp
  src/mundy_mesh/MetaData.cpp
  src/mundy_mesh/StringToSelector.cpp
  src/mundy_mesh/StringToTopology.cpp
  src/mundy_mesh/utils/DestroyFlaggedEntities.cpp
  src/mundy_mesh/utils/SelectorEval.cpp
  src/mundy_mesh/utils/SelectorLexem.cpp
  src/mundy_mesh/utils/SelectorNode.cpp
  src/mundy_mesh/utils/SelectorParser.cpp
  )

tribits_configure_file(${PACKAGE_NAME}_config.hpp)
tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# C) Add the libraries, tests, and examples
#

# Add a standard library
mundy_tribits_add_library(
  mundy_mesh
  HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  NOINSTALLHEADERS ${NOINSTALLHEADERS}
  DIRECTORIES src/mundy_mesh
  FILES_MATCHING PATTERN "*.hpp"
  SOURCES ${SOURCES}
  )

tribits_add_test_directories(tests)

#
# D) Do standard post processing
#
tribits_subpackage_postprocess()