#pragma once

#include <string>
#include <torch/all.h>

namespace sonar {
namespace vmm {

static inline torch::Dtype torch_dtype_from_size(size_t dtype_size) {
  switch (dtype_size) {
    case 1:
      return torch::kInt8;
    case 2:
      return torch::kInt16;
    case 4:
      return torch::kInt32;
    case 8:
      return torch::kInt64;
    default:
      throw std::runtime_error("Unsupported dtype size: " +
                               std::to_string(dtype_size));
  }
}

}  // namespace vmm
}  // namespace sonar
