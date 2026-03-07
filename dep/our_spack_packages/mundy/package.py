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

import os
import sys

from spack.build_environment import dso_suffix
from spack.error import NoHeadersError
from spack.operating_systems.mac_os import macos_version
from spack.package import *
from spack.pkg.builtin.kokkos import Kokkos

class Mundy(CMakePackage, CudaPackage):
    """MuNDy: Multibody Nonlocal Dynamics.

    A TriBITS-based C++ library organized as a hierarchy of subpackages:
    MundyCore, MundyMath, MundyGeom, MundyMesh, and MundyMech.
    """

    homepage = "https://github.com/MundyRepo/MuNDy"
    git      = "https://github.com/MundyRepo/MuNDy.git"
    url      = "https://github.com/MundyRepo/MuNDy/archive/refs/tags/v#.#.#.tar.gz"  # Doesn't yet exist

    maintainers("palmerb4")
    license("GPL-3.0-or-later")
    tags = ["???"]

    version("main", branch="main", submodules=True)
    version("dev", branch="polishing", submodules=True)
    # version("#.#.#", sha256="<fill-me-in>", submodules=True)

    #
    # Common build variants
    #
    variant("shared", default=True, description="Build shared libraries")
    variant("debug", default=False, description="Build with debug flags")
    variant("tests", default=False, description="Enable tests")
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
    variant("core", default=True, description="Enable MundyCore")
    variant("math", default=True, description="Enable MundyMath")
    variant("geom", default=True, description="Enable MundyGeom")
    variant("mesh", default=True, description="Enable MundyMesh")
    variant("mech", default=True, description="Enable MundyMech")

    #
    # Internal package hierarchy
    #
    with when("~core"):
        conflicts("+math", msg="MundyMath requires MundyCore")
        conflicts("+geom", msg="MundyGeom requires MundyCore")
        conflicts("+mesh", msg="MundyMesh requires MundyCore")
        conflicts("+mech", msg="MundyMech requires MundyCore")

    with when("~math"):
        conflicts("+geom", msg="MundyGeom requires MundyMath")
        conflicts("+mesh", msg="MundyMesh requires MundyMath")
        conflicts("+mech", msg="MundyMech requires MundyMath")

    with when("~geom"):
        conflicts("+mesh", msg="MundyMesh requires MundyGeom")
        conflicts("+mech", msg="MundyMech requires MundyGeom")

    with when("~mesh"):
        conflicts("+mech", msg="MundyMech requires MundyMesh")

    #
    # Optional TPL / feature variants
    #
    variant("mpi", default=False, description="Enable MPI support")
    variant("teuchos", default=False, description="Enable Teuchos support")
    variant("stk", default=False, description="Enable STK support")
    variant("kokkos-kernels", default=False, description="Enable KokkosKernels support")
    variant("openrand", default=False, description="Enable OpenRAND support")
    variant("nanobench", default=False, description="Enable nanobench support")

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
    depends_on("fmt")

    #
    # Optional TPLs
    #
    depends_on("mpi", when="+mpi")
    depends_on("trilinos@16.0.0", when="+teuchos")
    depends_on("trilinos@16.0.0+stk", when="+stk")
    depends_on("kokkos-kernels", when="+kokkos-kernels")
    depends_on("openrand", when="+openrand")
    depends_on("nanobench", when="+nanobench")
    depends_on("googletest@1.16.0:", when="+tests")

    #
    # Optional TPL requirements induced by selected MuNDy packages
    #
    with when("~stk"):
        conflicts("+mesh", msg="MundyMesh requires STK support")

    with when("~teuchos"):
        conflicts("+mesh", msg="MundyMesh requires Teuchos support")

    with when("~openrand"):
        conflicts("+mech", msg="MundyMech requires OpenRAND support")

    #
    # Test dependency requirements
    #
    with when("+tests"):
        with when("~nanobench"):
            conflicts("+math", msg="MundyMath tests require nanobench")
            conflicts("+geom", msg="MundyGeom tests require nanobench")
            conflicts("+mesh", msg="MundyMesh tests require nanobench")

        with when("~openrand"):
            conflicts("+math", msg="MundyMath tests require OpenRAND")
            conflicts("+mech", msg="MundyMech requires OpenRAND support")

    #
    # Package-specific implication dependencies from the MuNDy hierarchy
    #
    conflicts("+math", when="~core", msg="MundyMath requires MundyCore")
    conflicts("+geom", when="~math", msg="MundyGeom requires MundyMath")
    conflicts("+mesh", when="~geom", msg="MundyMesh requires MundyGeom")
    conflicts("+mesh", when="~math", msg="MundyMesh requires MundyMath")
    conflicts("+mesh", when="~core", msg="MundyMesh requires MundyCore")
    conflicts("+mech", when="~mesh", msg="MundyMech requires MundyMesh")
    conflicts("+mech", when="~geom", msg="MundyMech requires MundyGeom")
    conflicts("+mech", when="~math", msg="MundyMech requires MundyMath")
    conflicts("+mech", when="~core", msg="MundyMech requires MundyCore")

    #
    # TPL requirements implied by selected internal packages
    #
    # conflicts("+core", when="~kokkos", msg="MundyCore requires Kokkos")  # defensive
    # conflicts("+core", when="~fmt",    msg="MundyCore requires fmt")     # defensive
    conflicts("+math", when="~core",   msg="MundyMath requires MundyCore")
    conflicts("+geom", when="~core",   msg="MundyGeom requires MundyCore")
    conflicts("+mesh", when="~stk",    msg="MundyMesh requires STK")
    conflicts("+mesh", when="~teuchos", msg="MundyMesh requires Teuchos")
    conflicts("+mech", when="~openrand", msg="MundyMech requires OpenRAND")

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
            "trilinos@16.0.0+cuda cuda_arch={0}".format(arch),
            when="+teuchos +cuda cuda_arch={0}".format(arch),
        )
        depends_on(
            "trilinos@16.0.0+stk+cuda cuda_arch={0}".format(arch),
            when="+stk +cuda cuda_arch={0}".format(arch),
        )

    conflicts("+cuda", when="~core", msg="CUDA support requires at least MundyCore")
    conflicts("+stk", when="~teuchos", msg="STK support is expected to come with Teuchos")

    def cmake_args(self):
        spec = self.spec
        args = []

        def onoff(cond):
            return "ON" if cond else "OFF"

        #
        # Generic CMake build options
        #
        args.append(self.define("BUILD_SHARED_LIBS", "+shared" in spec))
        args.append(self.define("CMAKE_BUILD_TYPE", "Debug" if "+debug" in spec else "Release"))
        args.append(self.define("CMAKE_CXX_STANDARD", spec.variants["cxxstd"].value))

        #
        # Project-wide feature toggles
        #
        args.append(self.define("Mundy_ENABLE_TESTS", "+tests" in spec))
        args.append(self.define("Mundy_ENABLE_EXAMPLES", "+examples" in spec))

        #
        # Internal MuNDy packages
        # Adjust "Mundy_ENABLE_*" if your TriBITS project uses a different prefix.
        #
        args.append(self.define("Mundy_ENABLE_MundyCore", "+core" in spec))
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
        args.append(self.define("TPL_ENABLE_nanobench", "+nanobench" in spec))
        args.append(self.define("TPL_ENABLE_GTest", "+tests" in spec))

        #
        # CUDA support
        #
        args.append(self.define("TPL_ENABLE_CUDA", "+cuda" in spec))
        if "+cuda" in spec:
            # Works with modern CMake / Spack CUDA handling
            args.append(self.define("CMAKE_CUDA_ARCHITECTURES", spec.variants["cuda_arch"].value))
            args.append(self.define("CUDAToolkit_ROOT", spec["cuda"].prefix))

        return args