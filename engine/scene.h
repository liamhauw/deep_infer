#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <memory>

namespace deep_infer {

class Scene {
 public:
  Scene(VkPhysicalDevice physical, VkDevice device, std::uint32_t queue_family,
        const std::filesystem::path& engine_path);
  ~Scene();

  VkExtent2D InputExtent() const;
  VkExtent2D OutputExtent() const;
  VkImageView InputView() const;
  VkImageView OutputView() const;
  VkSemaphore InputReady() const;
  VkSemaphore OutputReady() const;

  void RecordScene(VkCommandBuffer cmd, float time, bool upscale);
  void EnqueueUpscale();
  void RecordOutput(VkCommandBuffer cmd);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace deep_infer
