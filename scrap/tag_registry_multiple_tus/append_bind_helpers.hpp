#pragma once
#include "myclass.hpp"
#include "tag_registry.hpp"
#include <vector>
#include <string>

namespace sam {
namespace py = ::pybind11;

// Generic binder that *already knows* the two append overloads via NTTPs.
// No name lookup for `append` happens in the binding TU.
template<
  class Tag,
  void (*AppendInt)(MyClass&, Tag, int),
  void (*AppendVec)(MyClass&, Tag, const std::vector<int>&)
>
void bind_tag_with_overloads(py::module_& m, const char* py_name_for_tag) {
  // Expose the empty tag type so Python can construct/pass it.
  py::class_<Tag>(m, py_name_for_tag).def(py::init<>());

  // Bind the two non-templated overloads that were captured as function pointers.
  m.def("append", AppendInt, "Append (int) for this tag");
  m.def("append", AppendVec, "Append (vector[int]) for this tag");
}

} // namespace sam
