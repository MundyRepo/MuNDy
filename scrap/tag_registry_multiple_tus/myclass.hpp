#pragma once
#include <vector>
#include <string>
#include <iostream>

namespace sam {

struct MyClass {
  std::vector<std::string> log;

  void dump() const {
    std::cout << "MyClass.log: [";
    for (size_t i = 0; i < log.size(); ++i) {
      std::cout << (i ? ", " : "") << log[i];
    }
    std::cout << "]\n";
  }
};

} // namespace sam
