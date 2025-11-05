cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE:-RELEASE} \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_CXX_FLAGS="-O3 -march=native -fPIC -flto -Wall -Wextra" \
${ccache_args} \
${compiler_flags} \
${install_dir} \
${extra_args} \
../
