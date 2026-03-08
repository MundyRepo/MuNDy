# Must set the project name at very beginning before including anything else
# This name must be unique not only within its defined TriBITS repository but 
# also across all packages in all TriBITS repositories that ever might be 
# cobbled together into a single TriBITS (meta) project! 
set(PROJECT_NAME Mundy)

# Turn on by default the generation of the export files
set(${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES_DEFAULT ON)
