#pragma once

#include <vulkan/vulkan.h>

#include <filesystem>
#include <memory>

namespace deep_infer {

class GltfRenderer {
 public:
  GltfRenderer(VkPhysicalDevice physical, VkDevice device, VkQueue queue,
               std::uint32_t family, VkExtent2D extent, VkFormat output_format,
               const std::filesystem::path& path);
  ~GltfRenderer();

  void Record(VkCommandBuffer cmd, VkImageView output, float time) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace deep_infer
