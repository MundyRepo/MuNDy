from spack.package import *
from spack_repo.builtin.build_systems.cmake import CMakePackage

class Nanobench(CMakePackage):
    """nanobench: a small benchmarking library."""

    homepage = "https://github.com/martinus/nanobench"
    git      = "https://github.com/martinus/nanobench.git"

    # Fill this in correctly if you know it
    license("MIT")

    version("master", branch="master")

    variant("cxxstd", default="17", values=("11", "14", "17", "20", "23"),
            multi=False, description="C++ standard for standalone nanobench build")

    depends_on("cmake@3.8:", type="build")
    depends_on("cxx", type="build")

    def cmake_args(self):
        args = []

        # Upstream only exposes NB_cxx_standard in standalone mode.
        args.append(self.define("NB_cxx_standard", self.spec.variants["cxxstd"].value))

        return args
