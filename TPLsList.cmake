# As a reminder, every TriBITS Package is assigned a test group. 
# These test groups are for Primary Tested (PT) code, Secondary Tested (ST) code, and Experimental (EX) code.
# The test group defines what package get selected (or are excluded from being selected) to include in a 
# given build for testing-related purposes.

# - Primary Tested (PT) Code is of the highest priority to keep working for the current development effort.
# - Secondary Tested (ST) Code is still very important code for the project and represents important capability 
#    to maintain but is excluded from the PT set of code for one of a few different reasons. 
# - Experimental (EX) Code is usually too unstable, buggy, or non-portable to be maintained as part of the 
# project’s automated testing processes.

tribits_repository_define_tpls(
  MPI            "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/"  PT
  CUDA           "${${PROJECT_NAME}_TRIBITS_DIR}/core/std_tpls/"  ST
  Kokkos         "cmake/TPLs/"                                    PT
  KokkosKernels  "cmake/TPLs/"                                    PT
  Teuchos        "cmake/TPLs/"                                    PT
  STK            "cmake/TPLs/"                                    PT
  ArborX         "cmake/TPLs/"                                    PT
  ParMETIS       "cmake/TPLs/"                                    PT
  GTest          "cmake/TPLs/"                                    PT
  OpenRAND       "cmake/TPLs/"                                    PT
  fmt            "cmake/TPLs/"                                    PT
  nanobench      "cmake/TPLs/"                                    PT
  PVFMM          "cmake/TPLs/"                                    EX
  STKFMM         "cmake/TPLs/"                                    EX
  )
