# https://cmake-format.readthedocs.io/en/latest/

# TRIBITS commands
_exec_kwargs = {"SOURCES": "*",
                "CATEGORIES": "*",
                "HOST": "*",
                "XHOST": "*",
                "HOSTTYPE": "*",
                "XHOSTTYPE": "*",
                "EXCLUDE_IF_NOT_TRUE": "*",
                "DIRECTORY": "*",
                "TESTONLYLIBS": "*",
                "IMPORTEDLIBS": "*",
                "COMM": "*",
                "LINKER_LANGUAGE": "*",
                "TARGET_DEFINES": "*",
                "INSTALLABLE": "*",
                "ADDED_EXE_TARGET_NAME_OUT": "*"}

_cmdline = {"pargs": {"nargs": "+", "tags": ["cmdline"]}}
_filelist = {"pargs": {"nargs": "+", "sortable": True}}

_test_kwargs = {"NAME": "*",
                "NAME_POSTFIX": "*",
                "XHOST_TEST": "*",
                "XHOSTTYPE_TEST": "*",
                "DISABLED": "*",
                "NUM_MPI_PROCS": "*",
                "COMM": "*",
                "ARGS": _cmdline,
                "POSTFIX_AND_ARGS_0": _cmdline,
                "POSTFIX_AND_ARGS_1": _cmdline,
                "POSTFIX_AND_ARGS_2": _cmdline,
                "POSTFIX_AND_ARGS_3": _cmdline,
                "POSTFIX_AND_ARGS_4": _cmdline,
                "POSTFIX_AND_ARGS_5": _cmdline,
                "POSTFIX_AND_ARGS_6": _cmdline,
                "POSTFIX_AND_ARGS_7": _cmdline,
                "POSTFIX_AND_ARGS_8": _cmdline,
                "POSTFIX_AND_ARGS_9": _cmdline,
                "CATEGORIES": "*",
                "RUN_SERIAL": "*",
                "STANDARD_PASS_OUTPUT": "*",
                "PASS_REGULAR_EXPRESSION": "*",
                "FAIL_REGULAR_EXPRESSION": "*",
                "ENVIRONMENT": "*",
                "WILL_FAIL": "*",
                "TIMEOUT": "*",
                "LIST_SEPARATOR": "*",
                "ADDED_TESTS_NAMES_OUT": "*",
                "EXEC": "*",
                "CMND": "*",
                "OVERALL_NUM_MPI_PROCS": "*",
                "TEST_0": "*",
                "TEST_1": "*",
                "TEST_2": "*",
                "TEST_3": "*"}

_lib_kwargs = {"HEADERS": "*",
               "HEADERS_INSTALL_SUBDIR": "*",
               "NOINSTALLHEADERS": "*",
               "SOURCES": "*",
               "DEPLIBS": "*",
               "IMPORTEDLIBS": "*",
               "ADDED_LIB_TARGET_NAME_OUT": "*"
               }

_mundy_lib_kwargs = {"HEADERS": "*",
                     "HEADERS_INSTALL_SUBDIR": "*",
                     "NOINSTALLHEADERS": "*",
                     "DIRECTORIES": "*",
                     "NOINSTALLDIRECTORIES": "*",
                     "FILES_MATCHING": "*",
                     "PATTERN": "*",
                     "REGEX": "*",
                     "PERMISSIONS": "*",
                     "SOURCES": "*",
                     "DEPLIBS": "*",
                     "IMPORTEDLIBS": "*",
                     "ADDED_LIB_TARGET_NAME_OUT": "*"
                     }

# -----------------------------
# Options affecting parsing.
# -----------------------------
with section("parse"):
    additional_commands = {"tribits_add_test": {"kwargs": _test_kwargs,
                                                "flags": ["NOEXESUFFIX"]},
                           "tribits_add_advanced_test": {"kwargs": _test_kwargs,
                                                      "flags": ["NOEXEPREFIX", "NOEXESUFFIX"]},
                           "tribits_add_executable_and_test": {
                               "kwargs": {**_test_kwargs, **_exec_kwargs},
                               "flags": ["NOEXEPREFIX", "NOEXESUFFIX", "ADD_DIR_TO_NAME",
                                         "RUN_SERIAL", "STANDARD_PASS_OUTPUT", "WILL_FAIL"]
                           },
                           "muelu_add_serial_and_mpi_test": {"kwargs": _test_kwargs,
                                                      "flags": ["NOEXEPREFIX", "NOEXESUFFIX"]},
                           "tribits_add_executable": {
                               "kwargs": _exec_kwargs,
                               "flags": ["NOEXEPREFIX", "NOEXESUFFIX", "ADD_DIR_TO_NAME"]
                           },
                           "mundy_tribits_add_library": {
                               "kwargs": _mundy_lib_kwargs,
                               "flags": ["STATIC", "SHARED", "TESTONLY", "NO_INSTALL_LIB_OR_HEADERS",
                                         "CUDALIBRARY", "EXCLUDE"]
                           },
                           "tribits_add_library": {
                               "kwargs": _lib_kwargs,
                               "flags": ["STATIC", "SHARED", "TESTONLY", "NO_INSTALL_LIB_OR_HEADERS",
                                         "CUDALIBRARY", "HEADERONLY"]
                           },
                           "tribits_copy_files_to_binary_dir": {"kwargs": {"SOURCE_FILES": _filelist,
                                                                           "SOURCE_DIR": "*",
                                                                           "DEST_FILES": _filelist,
                                                                           "DEST_DIR": "*",
                                                                           "TARGETDEPS": "*",
                                                                           "EXEDEPS": "*",
                                                                           "CATEGORIES": "*",
                                                                           },
                                                                "flags": ["NOEXEPREFIX", "NOEXESUFFIX"]},
                           "tribits_add_option_and_define": {"pargs": {"nargs": 4}},
                           }

# -----------------------------
# Options affecting formatting.
# -----------------------------
with section("format"):

    # How wide to allow formatted cmake files
    line_width = 120

    # If an argument group contains more than this many sub-groups (parg or kwarg
    # groups) then force it to a vertical layout.
    max_subgroups_hwrap = 2

    # If a positional argument group contains more than this many arguments, then
    # force it to a vertical layout.
    max_pargs_hwrap = 3

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True

    # If the trailing parenthesis must be 'dangled' on its on line, then align it
    # to this reference: `prefix`: the start of the statement,  `prefix-indent`:
    # the start of the statement, plus one indentation  level, `child`: align to
    # the column of the arguments
    dangle_align = 'prefix'

    # Format command names consistently as 'lower' or 'upper' case
    command_case = 'lower'

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = False

    # If true, the argument lists which are known to be sortable will be sorted lexicographically.
    enable_sort = True

    # If true, the parsers may infer whether or not an argument list is sortable (without annotation).
    autosort = True


# ------------------------------------------------
# Options affecting comment reflow and formatting.
# ------------------------------------------------
with section("markup"):
    # enable comment markup parsing and reflow
    enable_markup = False