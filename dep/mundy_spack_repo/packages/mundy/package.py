# @HEADER
# **********************************************************************************************************************
#
#                                          Mundy: Multi-body Nonlocal Dynamics
#                                              Copyright 2024 Bryce Palmer
#
# Developed under support from the NSF Graduate Research Fellowship Program.
#
# Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mundy. If not, see
# <https://www.gnu.org/licenses/>.
#
# **********************************************************************************************************************
# @HEADER

from spack.package import *
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage

class Mundy(CMakePackage, CudaPackage):
    """MuNDy: Multibody Nonlocal Dynamics.

    A TriBITS-based C++ library organized as a hierarchy of subpackages:
    MundyUtils, MundyMath, MundyGeom, MundyMesh, and MundyMech.
    """

    homepage = "https://github.com/MundyRepo/MuNDy"
    git      = "https://github.com/MundyRepo/MuNDy.git"
    url      = "https://github.com/MundyRepo/MuNDy/archive/refs/tags/v#.#.#.tar.gz"  # Doesn't yet exist

    maintainers("palmerb4")
    license("GPL-3.0-or-later")
    tags = ["???"]

    version("main", branch="main", submodules=True)
    version("dev", branch="streamlining", submodules=True)
    # version("#.#.#", sha256="<fill-me-in>", submodules=True)

    #
    # Common build variants
    #
    variant("shared", default=True, description="Build shared libraries")
    variant("debug", default=False, description="Build with debug flags")
    variant("tests", default=False, description="Enable both unit and performance tests")
    variant("unit_tests", default=False, description="Enable unit/regression tests")
    variant(
        "performance_tests",
        default=False,
        description="Enable PERFORMANCE tests",
    )
    variant("examples", default=False, description="Enable examples")
    variant(
        "cxxstd",
        default="20",
        values=("20", "23"),
        multi=False,
        description="C++ standard",
    )

    #
    # Internal MuNDy package variants
    #
    variant("core", default=True, description="Enable all of Mundy's core functionality (MundyUtils, MundyMath, MundyGeom, MundyMech)")
    variant("utils", default=True, description="Enable MundyUtils")
    variant("math", default=True, description="Enable MundyMath")
    variant("geom", default=True, description="Enable MundyGeom")
    variant("mech", default=True, description="Enable MundyMech")
    variant("mesh", default=False, description="Enable MundyMesh")

    #
    # Internal package hierarchy
    #
    requires("+utils", when="+core", msg="Enabling +core requires MundyUtils")
    requires("+math", when="+core", msg="Enabling +core requires MundyMath")
    requires("+geom", when="+core", msg="Enabling +core requires MundyGeom")
    requires("+mech", when="+core", msg="Enabling +core requires MundyMech")

    requires("+utils", when="+math", msg="MundyMath requires MundyUtils")
    requires("+math", when="+geom", msg="MundyGeom requires MundyMath")
    requires("+geom", when="+mesh", msg="MundyMesh requires MundyGeom")
    requires("+geom", when="+mech", msg="MundyMech requires MundyGeom")

    #
    # Optional TPL / feature variants
    #
    variant("mpi", default=False, description="Enable MPI support")
    variant("teuchos", default=False, description="Enable Teuchos support")
    variant("stk", default=False, description="Enable STK support")
    variant("kokkos-kernels", default=False, description="Enable KokkosKernels support")
    variant("openrand", default=True, description="Enable OpenRAND support")

    #
    # Build tools
    #
    depends_on("cmake@3.21:", type="build")
    depends_on("ninja", type="build", when="generator=ninja")
    depends_on("c", type="build")    # generated-style compiler deps
    depends_on("cxx", type="build")

    #
    # Always-required TPLs from your dependency tables
    #
    depends_on("kokkos@4.3.1:")

    #
    # Optional TPLs
    #
    depends_on("mpi", when="+mpi")
    depends_on("trilinos@16.1.0", when="+teuchos")
    depends_on("trilinos@16.1.0+stk", when="+stk")
    depends_on("kokkos-kernels", when="+kokkos-kernels")
    depends_on("openrand", when="+openrand")
    depends_on("nanobench", when="+performance_tests")
    depends_on("googletest@1.16.0:", when="+unit_tests")

    #
    # Internal package implications for optional TPLs
    #
    requires("+openrand", when="+utils", msg="MuNDy packages currently require OpenRAND")
    requires("+teuchos", when="+mesh", msg="MundyMesh requires Teuchos support")
    requires("+stk", when="+mesh", msg="MundyMesh requires STK support")
    requires("+teuchos", when="+stk", msg="STK support is expected to come with Teuchos")

    conflicts("+tests", when="~performance_tests ~unit_tests", msg="Enabling 'tests' requires at least one of +unit_tests or +performance_tests")

    #
    # CUDA-related dependency propagation
    #
    _cuda_arch_values = CudaPackage.cuda_arch_values

    if _cuda_arch_values is None:
        _cuda_arch_values = []

    for arch in _cuda_arch_values:
        depends_on(
            "kokkos+cuda cuda_arch={0}".format(arch),
            when="+cuda cuda_arch={0}".format(arch),
        )
        depends_on(
            "kokkos-kernels+cuda cuda_arch={0}".format(arch),
            when="+kokkos-kernels +cuda cuda_arch={0}".format(arch),
        )
        depends_on(
            "trilinos@16.1.0+cuda cuda_arch={0}".format(arch),
            when="+teuchos +cuda cuda_arch={0}".format(arch),
        )
        depends_on(
            "trilinos@16.1.0+stk+cuda cuda_arch={0}".format(arch),
            when="+stk +cuda cuda_arch={0}".format(arch),
        )

    conflicts("+cuda", when="~utils", msg="CUDA support requires at least MundyUtils")

    def cmake_args(self):
        spec = self.spec
        args = []

        #
        # Generic CMake build options
        #
        args.append(self.define("BUILD_SHARED_LIBS", "+shared" in spec))
        args.append(self.define("CMAKE_BUILD_TYPE", "Debug" if "+debug" in spec else "Release"))
        args.append(self.define("CMAKE_CXX_STANDARD", spec.variants["cxxstd"].value))

        #
        # Project-wide feature toggles
        #
        enable_unit_tests = "+tests" in spec or "+unit_tests" in spec
        enable_performance_tests = "+tests" in spec or "+performance_tests" in spec
        enable_any_tests = enable_unit_tests or enable_performance_tests

        args.append(self.define("Mundy_ENABLE_TESTS", enable_any_tests))
        args.append(self.define("Mundy_ENABLE_EXAMPLES", "+examples" in spec))
        if enable_any_tests:
            categories = []
            if enable_unit_tests:
                categories.extend(["BASIC", "CONTINUOUS", "NIGHTLY", "HEAVY"])
            if enable_performance_tests:
                categories.append("PERFORMANCE")
            args.append(self.define("Mundy_TEST_CATEGORIES", ";".join(categories)))

        #
        # Internal MuNDy packages
        # Adjust "Mundy_ENABLE_*" if your TriBITS project uses a different prefix.
        #
        args.append(self.define("Mundy_ENABLE_MundyUtils", "+utils" in spec))
        args.append(self.define("Mundy_ENABLE_MundyMath", "+math" in spec))
        args.append(self.define("Mundy_ENABLE_MundyGeom", "+geom" in spec))
        args.append(self.define("Mundy_ENABLE_MundyMesh", "+mesh" in spec))
        args.append(self.define("Mundy_ENABLE_MundyMech", "+mech" in spec))

        #
        # Optional third-party integrations
        # These names may need to match your project exactly.
        #
        args.append(self.define("TPL_ENABLE_MPI", "+mpi" in spec))
        args.append(self.define("TPL_ENABLE_Teuchos", "+teuchos" in spec))
        args.append(self.define("TPL_ENABLE_STK", "+stk" in spec))
        args.append(self.define("TPL_ENABLE_KokkosKernels", "+kokkos-kernels" in spec))
        args.append(self.define("TPL_ENABLE_OpenRAND", "+openrand" in spec))
        args.append(self.define("TPL_ENABLE_nanobench", enable_performance_tests))
        args.append(self.define("TPL_ENABLE_GTest", enable_unit_tests))

        #
        # CUDA support
        #
        args.append(self.define("TPL_ENABLE_CUDA", "+cuda" in spec))
        if "+cuda" in spec:
            # Works with modern CMake / Spack CUDA handling
            args.append(self.define("CMAKE_CUDA_ARCHITECTURES", spec.variants["cuda_arch"].value))
            args.append(self.define("CUDAToolkit_ROOT", spec["cuda"].prefix))

        return args
