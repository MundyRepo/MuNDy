#pragma once
#include <vector>
#include <string>
#include <mutex>

// We include pybind11 here for simplicity in the demo.
// In a production layout you could decouple this, but you're asking for proof.
#include <pybind11/pybind11.h>

namespace sam {
namespace py = ::pybind11;

// Forward declare your class so binders can mention it.
struct MyClass;

// A registry entry captures (name, binder)
// The binder is a uniform function pointer that knows
// how to bind *one tag's* type + its two append overloads.
struct TagEntry {
  const char* name;  // canonical Python-visible name
  void (*binder)(py::module_& m, const char* name); // uniform callback
};

class TagRegistry {
public:
  // Add an entry exactly once (idempotent); safe for static init
  static bool add(const char* name, void (*binder)(py::module_&, const char*)) {
    auto& vec = entries_();
    // Extremely simple dedup by pointer equality on the name and binder.
    // Good enough for the demo, and stable under inline ODR rules.
    for (const auto& e : vec) {
      if (e.name == name && e.binder == binder) return false;
    }
    vec.push_back(TagEntry{name, binder});
    return true;
  }

  static const std::vector<TagEntry>& entries() { return entries_(); }

private:
  static std::vector<TagEntry>& entries_() {
    static std::vector<TagEntry> v;
    return v;
  }
};

// A helper template the macro will reference. It’s declared here;
// each tag will specialize it with an inline variable.
template<class Tag> struct RegisterNewTag;

} // namespace sam
