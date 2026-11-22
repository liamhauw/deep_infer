#pragma once

#include <cstdint>
#include <span>

#include "vk_check.h"

namespace deep_infer {

inline constexpr VkImageSubresourceRange kColorRange{
    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
};

struct Shader {
  Shader(VkDevice device, std::span<const std::uint32_t> code)
      : device(device) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size_bytes();
    info.pCode = code.data();
    CheckVk(vkCreateShaderModule(device, &info, nullptr, &module),
            "Create shader module");
  }
  Shader(const Shader&) = delete;
  Shader& operator=(const Shader&) = delete;
  ~Shader() { vkDestroyShaderModule(device, module, nullptr); }

  VkPipelineShaderStageCreateInfo Stage(VkShaderStageFlagBits stage) const {
    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = stage,
        .module = module,
        .pName = "main",
    };
  }

  VkDevice device;
  VkShaderModule module{};
};

inline void TransitionImage(VkCommandBuffer cmd, VkImage image,
                            VkImageLayout old_layout, VkImageLayout new_layout,
                            VkPipelineStageFlags2 src_stage,
                            VkAccessFlags2 src_access,
                            VkPipelineStageFlags2 dst_stage,
                            VkAccessFlags2 dst_access,
                            VkImageSubresourceRange range = kColorRange) {
  VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcStageMask = src_stage;
  barrier.srcAccessMask = src_access;
  barrier.dstStageMask = dst_stage;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange = range;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

}  // namespace deep_infer
