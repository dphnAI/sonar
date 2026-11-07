#pragma once

#include <memory>

#include <cuda.h>
#include <torch/all.h>

#include "constants.hpp"

namespace sonar {
namespace vmm {

class Page {
 public:
  virtual ~Page() = default;
  virtual bool map(void* vaddr, bool set_access = true) = 0;
};

class GPUPage : public Page {
 public:
  GPUPage(page_id_t page_id, int dev_idx);
  ~GPUPage();

  bool map(void* vaddr, bool set_access = true);

 private:
  page_id_t page_id_;
  CUdevice dev_;
  CUmemGenericAllocationHandle handle_;
};

class CPUPage : public Page {
 public:
  CPUPage(page_id_t page_id);
  ~CPUPage();

  bool map(void* vaddr, bool set_access = true);

 private:
  page_id_t page_id_;
  generic_ptr_t mapped_addr_;
};

}  // namespace vmm
}  // namespace sonar
