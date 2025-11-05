#pragma once
#include "myclass.hpp"
#include "append_bind_helpers.hpp"
#include "tag_registry.hpp"
#include <vector>
#include <string>

#define MUNDY_DECLARE_TAG(TagName)                                                \
  namespace sam {                                                                 \
    struct TagName {};                                                            \
                                                                                  \
    /* Two inline non-templated overloads, keyed by TagName */                    \
    inline void append(MyClass& obj, TagName, int x) {                            \
      obj.log.push_back(std::string(#TagName) + ": int=" + std::to_string(x));    \
    }                                                                             \
    inline void append(MyClass& obj, TagName, const std::vector<int>& xs) {       \
      obj.log.push_back(std::string(#TagName) + ": vec[size=" +                   \
                        std::to_string(xs.size()) + "]");                         \
    }                                                                             \
                                                                                  \
    /* Resolve the two exact overloads to function pointers (no ambiguity). */    \
    static constexpr auto TagName##_append_int =                                  \
      static_cast<void(*)(MyClass&, TagName, int)>(&append);                      \
    static constexpr auto TagName##_append_vec =                                  \
      static_cast<void(*)(MyClass&, TagName, const std::vector<int>&)>(&append);  \
                                                                                  \
    /* A concrete binder type that bakes in Tag and both overload pointers. */    \
    template<> struct RegisterNewTag<TagName> {                                   \
      static inline const bool is_registered = TagRegistry::add(                  \
        #TagName,                                                                 \
        &bind_tag_with_overloads<                                                 \
          TagName,                                                                \
          TagName##_append_int,                                                   \
          TagName##_append_vec                                                    \
        >                                                                         \
      );                                                                          \
    };                                                                            \
                                                                                  \
    /* Force ODR-use so static init runs in every TU that sees the tag. */        \
    [[maybe_unused]] static const bool TagName##_registration_anchor =            \
        RegisterNewTag<TagName>::is_registered;                                   \
  } /* namespace sam */
