#pragma once
#include <cstdlib>
#include <string>

namespace aphrodite {

// aphrodite_is_batch_invariant(); returns true
// if env APHRODITE_BATCH_INVARIANT=1
inline bool aphrodite_is_batch_invariant() {
  static bool cached = []() {
    std::string env_key = "APHRODITE_BATCH_INVARIANT";
    const char* val = std::getenv(env_key.c_str());
    return (val && std::atoi(val) != 0) ? 1 : 0;
  }();
  return cached;
}

}  // namespace aphrodite
