#pragma once

#include <vulkan/vulkan.h>

#include <format>
#include <stdexcept>

inline void CheckVk(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(std::format("{} failed (VkResult {})", operation,
                                         static_cast<int>(result)));
  }
}
