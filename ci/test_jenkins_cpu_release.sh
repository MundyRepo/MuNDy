#!/bin/bash
set -euo pipefail
. ${SPACK_ROOT}/share/spack/setup-env.sh
spack env activate ${SPACK_TRILINOS}
cd build/
ctest --output-on-failure \
    --timeout 1200 \
    --output-junit ctest_results.xml
