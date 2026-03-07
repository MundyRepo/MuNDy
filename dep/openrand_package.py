from spack.package import *

class Openrand(CMakePackage):
    """OpenRAND: reproducible random number generation for parallel computations."""

    homepage = "https://github.com/msu-sparta/OpenRAND"
    git      = "https://github.com/msu-sparta/OpenRAND.git"

    license("MIT")

    version("main", branch="main")
    version("develop", branch="develop")

    variant("tests", default=False, description="Build tests")
    variant("examples", default=False, description="Build examples")
    variant("benchmarks", default=False, description="Build benchmarks")

    depends_on("cmake@3.15:", type="build")
    depends_on("cxx", type="build")

    # Only needed if you actually enable tests/benchmarks and upstream requires them.
    # depends_on("testu01", when="+tests")

    def cmake_args(self):
        return [
            self.define_from_variant("OpenRAND_ENABLE_TESTS", "tests"),
            self.define_from_variant("OpenRAND_ENABLE_EXAMPLES", "examples"),
            self.define_from_variant("OpenRAND_ENABLE_BENCHMARKS", "benchmarks"),
        ]