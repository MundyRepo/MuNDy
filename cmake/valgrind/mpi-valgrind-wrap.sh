#!/usr/bin/env bash
# mpi-valgrind-wrap.sh — CTest memcheck wrapper for MPI tests.
#
# Problem: CTest's -T memcheck prepends valgrind to the entire test command,
# producing:
#   valgrind [opts] mpiexec -np 1 ./test.exe [test_args]
#
# When valgrind wraps mpiexec and follows children (--trace-children=yes),
# it JIT-compiles AVX-512 instructions from spack-built shared libraries,
# generating SIGILL.  The test binary never actually runs.
#
# Solution: this script is set as MEMORYCHECK_COMMAND.  CTest calls it as:
#   mpi-valgrind-wrap.sh [valgrind_opts] mpiexec -np N ./test.exe [test_args]
#
# The script splits the argument list at mpiexec, then reassembles as:
#   mpiexec -np N [mpi_opts] valgrind [valgrind_opts] ./test.exe [test_args]
#
# valgrind now instruments the test binary directly (no mpiexec fork/exec
# trickery), and mpiexec is not under valgrind at all.
#
# For non-MPI tests (cmake -P scripts, etc.) that contain no mpiexec in their
# command, the script falls through and runs valgrind directly.
#
# The --log-file=... argument CTest injects into valgrind_opts is forwarded
# unchanged, so CTest can still parse the results from the correct log path.

set -euo pipefail

REAL_VALGRIND="$(command -v valgrind)"

ARGS=("$@")
VALGRIND_ARGS=()
MPIEXEC_CMD=""
MPIEXEC_OPTS=()
EXE_ARGS=()

# Find the first argument that looks like mpiexec.
MPIEXEC_IDX=-1
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
        *mpiexec* | *orterun* | *mpirun*)
            MPIEXEC_IDX=$i
            break
            ;;
    esac
done

if [ "$MPIEXEC_IDX" -eq -1 ]; then
    # No mpiexec in args — non-MPI test.  Run valgrind normally.
    exec "$REAL_VALGRIND" "${ARGS[@]}"
fi

# Split: everything before mpiexec is valgrind args.
VALGRIND_ARGS=("${ARGS[@]:0:$MPIEXEC_IDX}")
MPIEXEC_CMD="${ARGS[$MPIEXEC_IDX]}"
REST=("${ARGS[@]:$((MPIEXEC_IDX + 1))}")

# Parse REST: collect -np/-n and other mpi flags until we hit the executable.
in_exe=false
i=0
while [ $i -lt "${#REST[@]}" ]; do
    arg="${REST[$i]}"
    if ! $in_exe; then
        case "$arg" in
            -np | -n)
                MPIEXEC_OPTS+=("$arg")
                i=$((i + 1))
                MPIEXEC_OPTS+=("${REST[$i]}")
                ;;
            -*)
                MPIEXEC_OPTS+=("$arg")
                ;;
            *)
                # First non-flag argument is the executable.
                in_exe=true
                EXE_ARGS+=("$arg")
                ;;
        esac
    else
        EXE_ARGS+=("$arg")
    fi
    i=$((i + 1))
done

# Reassemble: mpiexec [mpi_opts] valgrind [valgrind_opts] exe [exe_args]
exec "$MPIEXEC_CMD" "${MPIEXEC_OPTS[@]}" \
     "$REAL_VALGRIND" "${VALGRIND_ARGS[@]}" \
     "${EXE_ARGS[@]}"
