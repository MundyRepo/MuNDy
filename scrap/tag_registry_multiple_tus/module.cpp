#include "myclass.hpp"
#include "tag_registry.hpp"
#include "tag_A.hpp"
#include "tag_B.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for std::vector<int> conversion

namespace sam { namespace py = ::pybind11; }

PYBIND11_MODULE(sam_demo, m) {
  using namespace sam;

  // Bind MyClass
  py::class_<MyClass>(m, "MyClass")
    .def(py::init<>())
    .def("dump", &MyClass::dump)
    .def_readonly("log", &MyClass::log);

  // Iterate the *runtime* registry and bind each tag + its two append overloads.
  for (const auto& e : TagRegistry::entries()) {
    e.binder(m, e.name);
  }
}
