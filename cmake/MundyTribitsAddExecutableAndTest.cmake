# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2024 Bryce Palmer
#
# **********************************************************************************************************************
# @HEADER

include(TribitsAddExecutableAndTest)
include(CMakeParseArguments)


function(_mundy_tribits_is_positive_integer VALUE RESULT_OUT)
  if ("${VALUE}" MATCHES "^[1-9][0-9]*$")
    set(${RESULT_OUT} TRUE PARENT_SCOPE)
  else()
    set(${RESULT_OUT} FALSE PARENT_SCOPE)
  endif()
endfunction()


function(_mundy_tribits_expand_mpi_proc_range MPI_PROC_RANGE_OUT)
  set(mpiProcRangeIn ${ARGN})
  list(LENGTH mpiProcRangeIn numMpiProcRangeValues)

  if (numMpiProcRangeValues LESS 1)
    message(FATAL_ERROR "MPI_PROC_RANGE must contain one range or one or more positive integer values.")
  endif()

  set(mpiProcRangeOut "")

  if (numMpiProcRangeValues EQUAL 1 AND "${mpiProcRangeIn}" MATCHES "^[1-9][0-9]*-[1-9][0-9]*$")
    string(REGEX REPLACE "^([1-9][0-9]*)-([1-9][0-9]*)$" "\\1" minMpiProc "${mpiProcRangeIn}")
    string(REGEX REPLACE "^([1-9][0-9]*)-([1-9][0-9]*)$" "\\2" maxMpiProc "${mpiProcRangeIn}")

    if (minMpiProc GREATER maxMpiProc)
      message(FATAL_ERROR "MPI_PROC_RANGE range '${mpiProcRangeIn}' has a lower bound greater than its upper bound.")
    endif()

    foreach(mpiProc RANGE ${minMpiProc} ${maxMpiProc})
      list(APPEND mpiProcRangeOut ${mpiProc})
    endforeach()
  else()
    foreach(mpiProc ${mpiProcRangeIn})
      _mundy_tribits_is_positive_integer("${mpiProc}" isPositiveInteger)
      if (NOT isPositiveInteger)
        message(FATAL_ERROR
          "MPI_PROC_RANGE values must be positive integers or one range like '1-4'. Invalid value: '${mpiProc}'.")
      endif()
      if (mpiProc IN_LIST mpiProcRangeOut)
        message(FATAL_ERROR "MPI_PROC_RANGE contains duplicate value '${mpiProc}'.")
      endif()
      list(APPEND mpiProcRangeOut ${mpiProc})
    endforeach()
  endif()

  set(${MPI_PROC_RANGE_OUT} ${mpiProcRangeOut} PARENT_SCOPE)
endfunction()


# @FUNCTION: mundy_tribits_add_executable_and_test()
#
# Add an executable and one or more tests. Without ``MPI_PROC_RANGE``, this
# delegates directly to ``tribits_add_executable_and_test()``.
#
# Usage::
#
#   mundy_tribits_add_executable_and_test(
#     <exeRootName>
#     [arguments accepted by tribits_add_executable_and_test()]
#     [MPI_PROC_RANGE <n0> <n1> ... | MPI_PROC_RANGE <first>-<last>]
#     )
#
# ``MPI_PROC_RANGE`` is mutually exclusive with ``NUM_MPI_PROCS``. When
# provided, the executable is registered once and a separate TriBITS test is
# registered for each requested MPI process count. TriBITS still applies
# ``MPI_EXEC_MAX_NUMPROCS`` to each generated test. A range like ``1-4`` is
# inclusive. A list like ``1 2 4`` preserves order; duplicate values are an
# error because they would create duplicate test names.
#
function(mundy_tribits_add_executable_and_test EXE_NAME)

  cmake_parse_arguments(
     #prefix
     PARSE
     #options
     "RUN_SERIAL;STANDARD_PASS_OUTPUT;WILL_FAIL;ADD_DIR_TO_NAME;INSTALLABLE;NOEXEPREFIX;NOEXESUFFIX"
     #one_value_keywords
     "DISABLED"
     #multi_value_keywords
     "SOURCES;DEPLIBS;TESTONLYLIBS;IMPORTEDLIBS;NAME;NAME_POSTFIX;NUM_MPI_PROCS;MPI_PROC_RANGE;DIRECTORY;KEYWORDS;COMM;ARGS;PASS_REGULAR_EXPRESSION;FAIL_REGULAR_EXPRESSION;ENVIRONMENT;TIMEOUT;LIST_SEPARATOR;CATEGORIES;HOST;XHOST;XHOST_TEST;HOSTTYPE;XHOSTTYPE;EXCLUDE_IF_NOT_TRUE;XHOSTTYPE_TEST;LINKER_LANGUAGE;TARGET_DEFINES;DEFINES;ADDED_EXE_TARGET_NAME_OUT;ADDED_TESTS_NAMES_OUT"
     ${ARGN}
     )

  if ("MPI_PROC_RANGE" IN_LIST PARSE_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR "MPI_PROC_RANGE must contain one range or one or more positive integer values.")
  endif()

  if (NOT PARSE_MPI_PROC_RANGE)
    tribits_add_executable_and_test(${EXE_NAME} ${ARGN})
    return()
  endif()

  tribits_check_for_unparsed_arguments()

  if (PARSE_NUM_MPI_PROCS)
    message(FATAL_ERROR "MPI_PROC_RANGE and NUM_MPI_PROCS are mutually exclusive.")
  endif()

  _mundy_tribits_expand_mpi_proc_range(mpiProcRange ${PARSE_MPI_PROC_RANGE})

  if(${PROJECT_NAME}_VERBOSE_CONFIGURE)
    message("")
    message("MUNDY_TRIBITS_ADD_EXECUTABLE_AND_TEST: ${EXE_NAME} ${ARGN}")
  endif()

  if(PARSE_ADDED_EXE_TARGET_NAME_OUT)
    set(${PARSE_ADDED_EXE_TARGET_NAME_OUT} "" PARENT_SCOPE)
  endif()
  if(PARSE_ADDED_TESTS_NAMES_OUT)
    set(${PARSE_ADDED_TESTS_NAMES_OUT} "" PARENT_SCOPE)
  endif()

  set(COMMON_CALL_ARGS "")
  tribits_fwd_parse_arg(COMMON_CALL_ARGS COMM)
  tribits_fwd_parse_arg(COMMON_CALL_ARGS CATEGORIES)
  tribits_fwd_parse_arg(COMMON_CALL_ARGS HOST)
  tribits_fwd_parse_arg(COMMON_CALL_ARGS XHOST)
  tribits_fwd_parse_arg(COMMON_CALL_ARGS HOSTTYPE)
  tribits_fwd_parse_arg(COMMON_CALL_ARGS XHOSTTYPE)
  tribits_fwd_parse_arg(COMMON_CALL_ARGS EXCLUDE_IF_NOT_TRUE)
  tribits_fwd_parse_opt(COMMON_CALL_ARGS NOEXEPREFIX)
  tribits_fwd_parse_opt(COMMON_CALL_ARGS NOEXESUFFIX)

  set(CALL_ARGS "")
  tribits_fwd_parse_arg(CALL_ARGS SOURCES)
  tribits_fwd_parse_arg(CALL_ARGS DEPLIBS)
  tribits_fwd_parse_arg(CALL_ARGS TESTONLYLIBS)
  tribits_fwd_parse_arg(CALL_ARGS IMPORTEDLIBS)
  tribits_fwd_parse_arg(CALL_ARGS DIRECTORY)
  tribits_fwd_parse_opt(CALL_ARGS ADD_DIR_TO_NAME)
  tribits_fwd_parse_arg(CALL_ARGS LINKER_LANGUAGE)
  tribits_fwd_parse_arg(CALL_ARGS TARGET_DEFINES)
  tribits_fwd_parse_arg(CALL_ARGS DEFINES)
  tribits_fwd_parse_opt(CALL_ARGS INSTALLABLE)
  if (PARSE_ADDED_EXE_TARGET_NAME_OUT)
    list(APPEND CALL_ARGS ADDED_EXE_TARGET_NAME_OUT ADDED_EXE_TARGET_NAME)
  endif()

  if (PARSE_DEPLIBS)
    tribits_deprecated("DEPLIBS argument of mundy_tribits_add_executable_and_test() is deprecated.")
  endif()

  tribits_add_executable_wrapper(${EXE_NAME} ${COMMON_CALL_ARGS} ${CALL_ARGS})

  if(PARSE_ADDED_EXE_TARGET_NAME_OUT)
    set(${PARSE_ADDED_EXE_TARGET_NAME_OUT} "${ADDED_EXE_TARGET_NAME}" PARENT_SCOPE)
  endif()

  set(TEST_CALL_ARGS "")
  tribits_fwd_parse_arg(TEST_CALL_ARGS NAME)
  tribits_fwd_parse_arg(TEST_CALL_ARGS NAME_POSTFIX)
  tribits_fwd_parse_arg(TEST_CALL_ARGS DIRECTORY)
  tribits_fwd_parse_arg(TEST_CALL_ARGS KEYWORDS)
  tribits_fwd_parse_arg(TEST_CALL_ARGS ARGS)
  tribits_fwd_parse_arg(TEST_CALL_ARGS PASS_REGULAR_EXPRESSION)
  tribits_fwd_parse_arg(TEST_CALL_ARGS FAIL_REGULAR_EXPRESSION)
  tribits_fwd_parse_arg(TEST_CALL_ARGS ENVIRONMENT)
  tribits_fwd_parse_arg(TEST_CALL_ARGS DISABLED)
  tribits_fwd_parse_opt(TEST_CALL_ARGS RUN_SERIAL)
  tribits_fwd_parse_opt(TEST_CALL_ARGS STANDARD_PASS_OUTPUT)
  tribits_fwd_parse_opt(TEST_CALL_ARGS WILL_FAIL)
  tribits_fwd_parse_arg(TEST_CALL_ARGS TIMEOUT)
  tribits_fwd_parse_arg(TEST_CALL_ARGS LIST_SEPARATOR)
  tribits_fwd_parse_opt(TEST_CALL_ARGS ADD_DIR_TO_NAME)
  if (PARSE_XHOST_TEST)
    list(APPEND TEST_CALL_ARGS XHOST ${PARSE_XHOST_TEST})
  endif()
  if (PARSE_XHOSTTYPE_TEST)
    list(APPEND TEST_CALL_ARGS XHOSTTYPE ${PARSE_XHOSTTYPE_TEST})
  endif()

  set(ALL_ADDED_TESTS_NAMES "")
  foreach(mpiProc ${mpiProcRange})
    set(ADDED_TESTS_NAMES "")
    set(TEST_CALL_ARGS_WITH_MPI_PROCS ${TEST_CALL_ARGS} NUM_MPI_PROCS ${mpiProc})
    if (PARSE_ADDED_TESTS_NAMES_OUT)
      list(APPEND TEST_CALL_ARGS_WITH_MPI_PROCS ADDED_TESTS_NAMES_OUT ADDED_TESTS_NAMES)
    endif()

    tribits_add_test_wrapper(${EXE_NAME} ${COMMON_CALL_ARGS} ${TEST_CALL_ARGS_WITH_MPI_PROCS})
    list(APPEND ALL_ADDED_TESTS_NAMES ${ADDED_TESTS_NAMES})
  endforeach()

  if(PARSE_ADDED_TESTS_NAMES_OUT)
    set(${PARSE_ADDED_TESTS_NAMES_OUT} "${ALL_ADDED_TESTS_NAMES}" PARENT_SCOPE)
  endif()

endfunction()
