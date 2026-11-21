#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

inline void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}
