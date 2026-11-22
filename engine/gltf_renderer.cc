#include "gltf_renderer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "gltf_model.h"
#include "vk_render.h"

namespace deep_infer {
namespace {

constexpr auto kVertexCode = std::to_array<std::uint32_t>(
#include "gltf.vert.inc"
);
constexpr auto kFragmentCode = std::to_array<std::uint32_t>(
#include "gltf.frag.inc"
);
constexpr auto kShadowVertexCode = std::to_array<std::uint32_t>(
#include "shadow.vert.inc"
);
constexpr auto kShadowFragmentCode = std::to_array<std::uint32_t>(
#include "shadow.frag.inc"
);
constexpr auto kToneVertexCode = std::to_array<std::uint32_t>(
#include "tonemap.vert.inc"
);
constexpr auto kToneFragmentCode = std::to_array<std::uint32_t>(
#include "tonemap.frag.inc"
);

constexpr VkFormat kHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkExtent2D kShadowExtent{1536, 1536};
constexpr VkImageSubresourceRange kDepthRange{
    VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1,
};
constexpr auto kDepthStages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                              VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
constexpr auto kDepthAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
constexpr auto kCombinedImageSampler =
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

struct Buffer {
  VkBuffer handle{};
  VkDeviceMemory memory{};
};

struct Image {
  VkImage handle{};
  VkDeviceMemory memory{};
  VkImageView view{};
  VkSampler sampler{};
};

struct Frame {
  Mat4 view_projection;
  Mat4 light_projection;
  std::array<float, 4> eye;
  std::array<float, 4> light;
};

static_assert(sizeof(Frame) == 160 && sizeof(MaterialParams) == 52);
static_assert(sizeof(Vec3) == 12);

VkSamplerAddressMode AddressMode(int wrap) {
  switch (wrap) {
    case 33071:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 33648:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  }
}

void SetViewport(VkCommandBuffer cmd, VkExtent2D extent) {
  const VkViewport kViewport{
      0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height),
      0, 1,
  };
  const VkRect2D kScissor{{0, 0}, extent};
  vkCmdSetViewport(cmd, 0, 1, &kViewport);
  vkCmdSetScissor(cmd, 0, 1, &kScissor);
}

VkRenderingAttachmentInfo DepthAttachment(VkImageView view,
                                          VkAttachmentStoreOp store) {
  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depth.imageView = view;
  depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth.storeOp = store;
  depth.clearValue.depthStencil.depth = 1.0F;
  return depth;
}

constexpr float kHalfFov = 0.39F;
constexpr float kElevation = 0.30F;

float FitOrbit(Vec3 half_extent, float aspect) {
  // Enclose the model at every orbit angle using its vertical cylinder.
  const float kRadius = std::hypot(half_extent.x, half_extent.z);
  const float kS = std::sin(kElevation);
  const float kC = std::cos(kElevation);
  const float kY = 1.0F / (0.91F * std::tan(kHalfFov));
  const float kX = kY / aspect;
  const float kHorizontal = kRadius * std::hypot(kC, kX) + half_extent.y * kS;
  const float kTop =
      kRadius * std::abs(kC - kS * kY) + half_extent.y * std::abs(kS + kC * kY);
  const float kBottom =
      kRadius * std::abs(kC + kS * kY) + half_extent.y * std::abs(kS - kC * kY);
  return std::max({kHorizontal, kTop, kBottom}) + 0.1F;
}

Frame MakeFrame(float aspect, float distance, float time) {
  const float kOrbit = 0.28F + time * 0.24F;
  const float kHorizontal = distance * std::cos(kElevation);
  const Vec3 kEye{
      std::sin(kOrbit) * kHorizontal,
      distance * std::sin(kElevation),
      std::cos(kOrbit) * kHorizontal,
  };
  const float kF = 1.0F / std::tan(kHalfFov);
  const float kFarPlane = distance * 2.0F + 4.0F;
  constexpr float kNear = 0.1F;
  const Mat4 kProjection{
      {
          {kF / aspect, 0, 0, 0},
          {0, -kF, 0, 0},
          {0, 0, kFarPlane / (kNear - kFarPlane), -1},
          {0, 0, kNear * kFarPlane / (kNear - kFarPlane), 0},
      },
  };
  const Vec3 kLight = Normalize({-0.48F, 0.82F, 0.56F});
  constexpr Mat4 kLightProjection{
      {
          {1.0F / 1.8F, 0, 0, 0},
          {0, -1.0F / 1.8F, 0, 0},
          {0, 0, -1.0F / 6.0F, 0},
          {0, 0, 0, 1},
      },
  };
  return {
      Multiply(kProjection, LookAt(kEye, {})),
      Multiply(kLightProjection, LookAt(kLight * 3.0F, {})),
      {kEye.x, kEye.y, kEye.z, 0},
      {kLight.x, kLight.y, kLight.z, 0},
  };
}

}  // namespace

struct GltfRenderer::Impl {
  Impl(VkPhysicalDevice physical, VkDevice device, VkExtent2D extent)
      : physical(physical),
        device(device),
        extent(extent),
        aspect(static_cast<float>(extent.width) /
               static_cast<float>(extent.height)) {}

  ~Impl() {
    ReleaseStaging();
    for (VkPipeline pipeline :
         {shadow_pipeline, opaque_pipeline, blend_pipeline, tone_pipeline}) {
      vkDestroyPipeline(device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(device, geometry_layout, nullptr);
    vkDestroyPipelineLayout(device, tone_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptors, nullptr);
    for (VkDescriptorSetLayout layout :
         {frame_layout, material_layout, hdr_layout}) {
      vkDestroyDescriptorSetLayout(device, layout, nullptr);
    }
    for (const auto& texture : textures) {
      Destroy(texture);
    }
    Destroy(hdr);
    Destroy(depth);
    Destroy(shadow);
    Destroy(vertices);
    Destroy(indices);
    Destroy(frame_buffer);
  }

  void Destroy(const Buffer& buffer) const {
    vkDestroyBuffer(device, buffer.handle, nullptr);
    vkFreeMemory(device, buffer.memory, nullptr);
  }

  void Destroy(const Image& image) const {
    vkDestroySampler(device, image.sampler, nullptr);
    vkDestroyImageView(device, image.view, nullptr);
    vkDestroyImage(device, image.handle, nullptr);
    vkFreeMemory(device, image.memory, nullptr);
  }

  void ReleaseStaging() {
    vkDestroyCommandPool(device, std::exchange(upload_pool, VK_NULL_HANDLE),
                         nullptr);
    for (const auto& buffer : staging) {
      Destroy(buffer);
    }
    staging.clear();
  }

  VkFormatFeatureFlags2 FormatFeatures(VkFormat format) const {
    VkFormatProperties3 features{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 properties{
        VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        &features,
    };
    vkGetPhysicalDeviceFormatProperties2(physical, format, &properties);
    return features.optimalTilingFeatures;
  }

  void Allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags,
                VkDeviceMemory& memory) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & (1U << i)) &&
          (properties.memoryTypes[i].propertyFlags & flags) == flags) {
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = requirements.size;
        info.memoryTypeIndex = i;
        CheckVk(vkAllocateMemory(device, &info, nullptr, &memory),
                "Allocate glTF resource");
        return;
      }
    }
    throw std::runtime_error(
        "No suitable Vulkan memory type for glTF resource");
  }

  void CreateBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memory) const {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    CheckVk(vkCreateBuffer(device, &info, nullptr, &buffer.handle),
            "Create glTF buffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
    Allocate(requirements, memory, buffer.memory);
    CheckVk(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0),
            "Bind glTF buffer");
  }

  void CreateImage(Image& image, VkExtent2D size, VkFormat format,
                   VkImageUsageFlags usage,
                   VkImageSubresourceRange range) const {
    VkImageCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .samples = VK_SAMPLE_COUNT_1_BIT,
    };
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {size.width, size.height, 1};
    info.mipLevels = range.levelCount;
    info.arrayLayers = 1;
    info.usage = usage;
    CheckVk(vkCreateImage(device, &info, nullptr, &image.handle),
            "Create glTF image");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image.handle, &requirements);
    Allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image.memory);
    CheckVk(vkBindImageMemory(device, image.handle, image.memory, 0),
            "Bind glTF image");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image.handle;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange = range;
    CheckVk(vkCreateImageView(device, &view, nullptr, &image.view),
            "Create glTF image view");
  }

  void CreateSampler(Image& image, const VkSamplerCreateInfo& info) const {
    CheckVk(vkCreateSampler(device, &info, nullptr, &image.sampler),
            "Create glTF sampler");
  }

  VkBuffer Stage(std::span<const std::byte> bytes) {
    auto& buffer = staging.emplace_back();
    CreateBuffer(buffer, bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = nullptr;
    CheckVk(vkMapMemory(device, buffer.memory, 0, bytes.size(), 0, &mapped),
            "Map glTF staging buffer");
    std::memcpy(mapped, bytes.data(), bytes.size());
    vkUnmapMemory(device, buffer.memory);
    return buffer.handle;
  }

  void UploadBuffer(VkCommandBuffer cmd, Buffer& buffer,
                    VkBufferUsageFlags usage,
                    std::span<const std::byte> bytes) {
    CreateBuffer(buffer, bytes.size(), usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const VkBufferCopy kCopy{0, 0, bytes.size()};
    vkCmdCopyBuffer(cmd, Stage(bytes), buffer.handle, 1, &kCopy);
  }

  void UploadTexture(VkCommandBuffer cmd, const ModelTexture& texture,
                     Image& image) {
    const VkFormat kFormat =
        texture.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    constexpr auto kMipFeatures =
        VK_FORMAT_FEATURE_2_BLIT_SRC_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    const bool kMipmapped =
        texture.min_filter >= 9984 &&
        (FormatFeatures(kFormat) & kMipFeatures) == kMipFeatures;
    auto width = static_cast<std::uint32_t>(texture.width);
    auto height = static_cast<std::uint32_t>(texture.height);
    auto range = kColorRange;
    range.levelCount = kMipmapped ? std::bit_width(std::max(width, height)) : 1;
    VkImageUsageFlags usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (kMipmapped) {
      usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    CreateImage(image, {width, height}, kFormat, usage, range);

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter =
        texture.mag_filter == 9728 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    const int kMin = texture.min_filter;
    sampler.minFilter = kMin == 9728 || kMin == 9984 || kMin == 9986
                            ? VK_FILTER_NEAREST
                            : VK_FILTER_LINEAR;
    sampler.mipmapMode = kMin >= 9986 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                      : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = AddressMode(texture.wrap_s);
    sampler.addressModeV = AddressMode(texture.wrap_t);
    sampler.maxLod = static_cast<float>(range.levelCount - 1);
    CreateSampler(image, sampler);

    TransitionImage(cmd, image.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT, range);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, Stage(std::as_bytes(std::span(texture.pixels))),
                           image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &copy);

    for (std::uint32_t i = 1; i < range.levelCount; ++i) {
      auto previous = kColorRange;
      previous.baseMipLevel = i - 1;
      TransitionImage(cmd, image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_BLIT_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT, previous);
      VkImageBlit blit{};
      blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
      blit.srcOffsets[1] = {
          static_cast<int>(width),
          static_cast<int>(height),
          1,
      };
      width = std::max(width / 2, 1U);
      height = std::max(height / 2, 1U);
      blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
      blit.dstOffsets[1] = {
          static_cast<int>(width),
          static_cast<int>(height),
          1,
      };
      vkCmdBlitImage(cmd, image.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                     &blit, VK_FILTER_LINEAR);
    }
    if (range.levelCount > 1) {
      auto generated = range;
      --generated.levelCount;
      TransitionImage(cmd, image.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_BLIT_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, generated);
    }
    range.baseMipLevel = range.levelCount - 1;
    range.levelCount = 1;
    TransitionImage(cmd, image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                    VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, range);
  }

  void Initialize(VkQueue queue, std::uint32_t family, VkFormat output_format,
                  const GltfModel& model) {
    camera_distance = FitOrbit(model.half_extent, aspect);
    CreateTargets();
    CreateBuffer(
        frame_buffer, sizeof(Frame),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    textures.resize(model.textures.size());
    materials = model.materials;
    for (const auto& draw : model.draws) {
      auto& list = (materials[draw.material].params.flags & kAlphaBlend)
                       ? transparent
                       : opaque;
      list.push_back(draw);
    }

    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool.queueFamilyIndex = family;
    CheckVk(vkCreateCommandPool(device, &pool, nullptr, &upload_pool),
            "Create glTF upload pool");
    VkCommandBufferAllocateInfo allocate{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    };
    allocate.commandPool = upload_pool;
    allocate.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    CheckVk(vkAllocateCommandBuffers(device, &allocate, &cmd),
            "Allocate glTF upload commands");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CheckVk(vkBeginCommandBuffer(cmd, &begin), "Begin glTF upload");
    UploadBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 std::as_bytes(std::span(model.vertices)));
    UploadBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 std::as_bytes(std::span(model.indices)));
    for (std::size_t i = 0; i < textures.size(); ++i) {
      UploadTexture(cmd, model.textures[i], textures[i]);
    }
    VkMemoryBarrier2 geometry{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    geometry.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    geometry.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    geometry.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    geometry.dstAccessMask =
        VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &geometry;
    vkCmdPipelineBarrier2(cmd, &dependency);

    CreateDescriptors();
    CreatePipelines(output_format);
    UpdateFrame(cmd, MakeFrame(aspect, camera_distance, 0));
    RecordShadow(cmd);
    CheckVk(vkEndCommandBuffer(cmd), "Finish glTF upload");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    CheckVk(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE),
            "Submit glTF upload");
    CheckVk(vkQueueWaitIdle(queue), "Wait for glTF upload");
    ReleaseStaging();
  }

  void CreateTargets() {
    for (const auto kFormat : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM}) {
      constexpr auto kRequired =
          VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
          VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
          VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
      if ((FormatFeatures(kFormat) & kRequired) == kRequired) {
        depth_format = kFormat;
        break;
      }
    }
    if (depth_format == VK_FORMAT_UNDEFINED) {
      throw std::runtime_error(
          "No Vulkan depth format supports shadow comparison");
    }
    CreateImage(depth, extent, depth_format,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, kDepthRange);
    CreateImage(shadow, kShadowExtent, depth_format,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT,
                kDepthRange);
    VkSamplerCreateInfo shadow_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    shadow_sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadow_sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadow_sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    shadow_sampler.compareEnable = VK_TRUE;
    shadow_sampler.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    CreateSampler(shadow, shadow_sampler);

    constexpr auto kHdrFeatures =
        VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT |
        VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
    if ((FormatFeatures(kHdrFormat) & kHdrFeatures) != kHdrFeatures) {
      throw std::runtime_error(
          "Vulkan device cannot blend or sample the HDR color format");
    }
    CreateImage(
        hdr, extent, kHdrFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        kColorRange);
    const VkSamplerCreateInfo kSampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    CreateSampler(hdr, kSampler);
  }

  void CreateSetLayout(std::span<const VkDescriptorSetLayoutBinding> bindings,
                       VkDescriptorSetLayout& layout) const {
    VkDescriptorSetLayoutCreateInfo info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    };
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    CheckVk(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout),
            "Create glTF descriptor set layout");
  }

  VkDescriptorSet AllocateSet(VkDescriptorSetLayout layout) const {
    VkDescriptorSetAllocateInfo info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    };
    info.descriptorPool = descriptors;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    VkDescriptorSet set{};
    CheckVk(vkAllocateDescriptorSets(device, &info, &set),
            "Allocate glTF descriptor set");
    return set;
  }

  void WriteImage(
      VkDescriptorSet set, std::uint32_t binding, const Image& image,
      VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) const {
    const VkDescriptorImageInfo kSampled{image.sampler, image.view, layout};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = kCombinedImageSampler;
    write.pImageInfo = &kSampled;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }

  void CreateDescriptors() {
    constexpr VkDescriptorSetLayoutBinding kSamplerBinding{
        0,
        kCombinedImageSampler,
        1,
        VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const std::array kFrameBindings{
        VkDescriptorSetLayoutBinding{
            0,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            1,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        VkDescriptorSetLayoutBinding{
            1,
            kCombinedImageSampler,
            1,
            VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };
    CreateSetLayout(kFrameBindings, frame_layout);
    CreateSetLayout(std::span(&kSamplerBinding, 1), hdr_layout);

    std::array<VkDescriptorSetLayoutBinding, kTextureSlots> material_bindings{};
    for (std::uint32_t i = 0; i < material_bindings.size(); ++i) {
      material_bindings[i] = kSamplerBinding;
      material_bindings[i].binding = i;
    }
    CreateSetLayout(material_bindings, material_layout);
    if (materials.size() > (UINT32_MAX - 2) / kTextureSlots) {
      throw std::runtime_error("Too many glTF materials");
    }
    const auto kCount = static_cast<std::uint32_t>(materials.size());
    const std::array kSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        VkDescriptorPoolSize{
            kCombinedImageSampler,
            2U + kCount * static_cast<std::uint32_t>(kTextureSlots),
        },
    };
    VkDescriptorPoolCreateInfo pool{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    };
    pool.maxSets = 2 + kCount;
    pool.poolSizeCount = static_cast<std::uint32_t>(kSizes.size());
    pool.pPoolSizes = kSizes.data();
    CheckVk(vkCreateDescriptorPool(device, &pool, nullptr, &descriptors),
            "Create glTF descriptor pool");

    frame_set = AllocateSet(frame_layout);
    const VkDescriptorBufferInfo kBuffer{frame_buffer.handle, 0, sizeof(Frame)};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = frame_set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &kBuffer;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    WriteImage(frame_set, 1, shadow,
               VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    hdr_set = AllocateSet(hdr_layout);
    WriteImage(hdr_set, 0, hdr);
    for (const auto& material : materials) {
      const VkDescriptorSet kSet =
          material_sets.emplace_back(AllocateSet(material_layout));
      for (std::uint32_t i = 0; i < material.textures.size(); ++i) {
        WriteImage(kSet, i, textures[material.textures[i]]);
      }
    }
  }

  void CreatePipelines(VkFormat output_format) {
    const std::array kSets{frame_layout, material_layout};
    const VkPushConstantRange kPush{
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(MaterialParams),
    };
    VkPipelineLayoutCreateInfo layout{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    layout.setLayoutCount = static_cast<std::uint32_t>(kSets.size());
    layout.pSetLayouts = kSets.data();
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &kPush;
    CheckVk(vkCreatePipelineLayout(device, &layout, nullptr, &geometry_layout),
            "Create glTF geometry pipeline layout");
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &hdr_layout;
    layout.pushConstantRangeCount = 0;
    CheckVk(vkCreatePipelineLayout(device, &layout, nullptr, &tone_layout),
            "Create glTF tone mapping pipeline layout");

    const Shader kVertex(device, kVertexCode);
    const Shader kFragment(device, kFragmentCode);
    const Shader kShadowVertex(device, kShadowVertexCode);
    const Shader kShadowFragment(device, kShadowFragmentCode);
    const Shader kToneVertex(device, kToneVertexCode);
    const Shader kToneFragment(device, kToneFragmentCode);
    CreatePipeline(kShadowVertex, kShadowFragment, VK_FORMAT_UNDEFINED,
                   depth_format, false, shadow_pipeline);
    CreatePipeline(kVertex, kFragment, kHdrFormat, depth_format, false,
                   opaque_pipeline);
    CreatePipeline(kVertex, kFragment, kHdrFormat, depth_format, true,
                   blend_pipeline);
    CreatePipeline(kToneVertex, kToneFragment, output_format,
                   VK_FORMAT_UNDEFINED, false, tone_pipeline);
  }

  void CreatePipeline(const Shader& vertex, const Shader& fragment,
                      VkFormat color_format, VkFormat depth_attachment,
                      bool transparent_pass, VkPipeline& pipeline) const {
    const bool kGeometry = depth_attachment != VK_FORMAT_UNDEFINED;
    const bool kShadowPass = color_format == VK_FORMAT_UNDEFINED;
    const std::array kStages{
        vertex.Stage(VK_SHADER_STAGE_VERTEX_BIT),
        fragment.Stage(VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    const VkVertexInputBindingDescription kBinding{
        0,
        sizeof(ModelVertex),
        VK_VERTEX_INPUT_RATE_VERTEX,
    };
    constexpr auto kUv = offsetof(ModelVertex, uv);
    constexpr auto kUvSize = sizeof(ModelVertex::uv[0]);
    constexpr std::array kAttributes{
        VkVertexInputAttributeDescription{
            0,
            0,
            VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(ModelVertex, position),
        },
        VkVertexInputAttributeDescription{
            1,
            0,
            VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(ModelVertex, normal),
        },
        VkVertexInputAttributeDescription{
            2,
            0,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            offsetof(ModelVertex, tangent),
        },
        VkVertexInputAttributeDescription{
            3,
            0,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            offsetof(ModelVertex, color),
        },
        VkVertexInputAttributeDescription{4, 0, VK_FORMAT_R32G32_SFLOAT, kUv},
        VkVertexInputAttributeDescription{
            5,
            0,
            VK_FORMAT_R32G32_SFLOAT,
            kUv + kUvSize,
        },
        VkVertexInputAttributeDescription{
            6,
            0,
            VK_FORMAT_R32G32_SFLOAT,
            kUv + 2 * kUvSize,
        },
        VkVertexInputAttributeDescription{
            7,
            0,
            VK_FORMAT_R32G32_SFLOAT,
            kUv + 3 * kUvSize,
        },
        VkVertexInputAttributeDescription{
            8,
            0,
            VK_FORMAT_R32G32_SFLOAT,
            kUv + 4 * kUvSize,
        },
    };
    constexpr std::array kShadowAttributes{
        kAttributes[0],
        kAttributes[3],
        kAttributes[4],
    };
    VkPipelineVertexInputStateCreateInfo input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    if (kGeometry) {
      input.vertexBindingDescriptionCount = 1;
      input.pVertexBindingDescriptions = &kBinding;
      const auto kActive =
          kShadowPass
              ? std::span<const VkVertexInputAttributeDescription>(
                    kShadowAttributes)
              : std::span<const VkVertexInputAttributeDescription>(kAttributes);
      input.vertexAttributeDescriptionCount =
          static_cast<std::uint32_t>(kActive.size());
      input.pVertexAttributeDescriptions = kActive.data();
    }
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    };
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
    };
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    };
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0F;
    const VkPipelineMultisampleStateCreateInfo kSamples{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineDepthStencilStateCreateInfo depth_state{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
    };
    depth_state.depthTestEnable = kGeometry;
    depth_state.depthWriteEnable = kGeometry && !transparent_pass;
    depth_state.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    attachment.blendEnable = transparent_pass;
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    };
    blend.attachmentCount = kShadowPass ? 0 : 1;
    blend.pAttachments = &attachment;
    constexpr std::array kStates{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    };
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(kStates.size());
    dynamic.pDynamicStates = kStates.data();
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
    };
    rendering.colorAttachmentCount = blend.attachmentCount;
    rendering.pColorAttachmentFormats = &color_format;
    rendering.depthAttachmentFormat = depth_attachment;
    VkGraphicsPipelineCreateInfo info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        &rendering,
    };
    info.stageCount = static_cast<std::uint32_t>(kStages.size());
    info.pStages = kStages.data();
    info.pVertexInputState = &input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &kSamples;
    info.pDepthStencilState = &depth_state;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = kGeometry ? geometry_layout : tone_layout;
    CheckVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                      &pipeline),
            "Create glTF graphics pipeline");
  }

  void UpdateFrame(VkCommandBuffer cmd, const Frame& frame) const {
    // Scene submissions share one frame buffer and wait for the preceding
    // frame.
    vkCmdUpdateBuffer(cmd, frame_buffer.handle, 0, sizeof(frame), &frame);
    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = frame_buffer.handle;
    barrier.size = sizeof(frame);
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);
  }

  void BindGeometry(VkCommandBuffer cmd) const {
    constexpr VkDeviceSize kOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.handle, &kOffset);
    vkCmdBindIndexBuffer(cmd, indices.handle, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            geometry_layout, 0, 1, &frame_set, 0, nullptr);
  }

  void Draw(VkCommandBuffer cmd, VkPipeline pipeline,
            std::span<const ModelDraw> draws) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    for (const auto& draw : draws) {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              geometry_layout, 1, 1,
                              &material_sets[draw.material], 0, nullptr);
      vkCmdPushConstants(cmd, geometry_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(MaterialParams),
                         &materials[draw.material].params);
      vkCmdDrawIndexed(cmd, draw.index_count, 1, draw.first_index, 0, 0);
    }
  }

  void RecordShadow(VkCommandBuffer cmd) const {
    TransitionImage(cmd, shadow.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, kDepthStages,
                    kDepthAccess, kDepthRange);
    const auto kAttachment =
        DepthAttachment(shadow.view, VK_ATTACHMENT_STORE_OP_STORE);
    VkRenderingInfo pass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    pass.renderArea.extent = kShadowExtent;
    pass.layerCount = 1;
    pass.pDepthAttachment = &kAttachment;
    vkCmdBeginRendering(cmd, &pass);
    SetViewport(cmd, kShadowExtent);
    BindGeometry(cmd);
    Draw(cmd, shadow_pipeline, opaque);
    vkCmdEndRendering(cmd);
    TransitionImage(
        cmd, shadow.handle, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, kDepthStages,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, kDepthRange);
  }

  void Record(VkCommandBuffer cmd, VkImageView output, float time) const {
    const Frame kFrame = MakeFrame(aspect, camera_distance, time);
    UpdateFrame(cmd, kFrame);
    TransitionImage(cmd, depth.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, kDepthStages,
                    kDepthAccess, kDepthRange);
    TransitionImage(cmd, hdr.handle, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
    };
    color.imageView = hdr.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.018F, 0.024F, 0.039F, 1.0F}};
    const auto kDepthAttachment =
        DepthAttachment(depth.view, VK_ATTACHMENT_STORE_OP_DONT_CARE);
    VkRenderingInfo pass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    pass.renderArea.extent = extent;
    pass.layerCount = 1;
    pass.colorAttachmentCount = 1;
    pass.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &pass);
    SetViewport(cmd, extent);
    BindGeometry(cmd);
    Draw(cmd, opaque_pipeline, opaque);
    if (!transparent.empty()) {
      auto sorted = transparent;
      const Vec3 kEye{kFrame.eye[0], kFrame.eye[1], kFrame.eye[2]};
      std::stable_sort(sorted.begin(), sorted.end(),
                       [kEye](const ModelDraw& a, const ModelDraw& b) {
                         return Dot(a.center, kEye) < Dot(b.center, kEye);
                       });
      Draw(cmd, blend_pipeline, sorted);
    }
    vkCmdEndRendering(cmd);

    TransitionImage(cmd, hdr.handle, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    color.imageView = output;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    pass.pDepthAttachment = nullptr;
    vkCmdBeginRendering(cmd, &pass);
    SetViewport(cmd, extent);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tone_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tone_layout,
                            0, 1, &hdr_set, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
  }

  VkPhysicalDevice physical;
  VkDevice device;
  VkExtent2D extent;
  float aspect, camera_distance{};
  VkFormat depth_format{VK_FORMAT_UNDEFINED};
  Image depth, shadow, hdr;
  Buffer vertices, indices, frame_buffer;
  VkCommandPool upload_pool{};
  std::vector<Buffer> staging;
  std::vector<Image> textures;
  std::vector<ModelMaterial> materials;
  std::vector<ModelDraw> opaque, transparent;
  VkDescriptorPool descriptors{};
  VkDescriptorSetLayout frame_layout{}, material_layout{}, hdr_layout{};
  VkDescriptorSet frame_set{}, hdr_set{};
  std::vector<VkDescriptorSet> material_sets;
  VkPipelineLayout geometry_layout{}, tone_layout{};
  VkPipeline shadow_pipeline{}, opaque_pipeline{}, blend_pipeline{},
      tone_pipeline{};
};

GltfRenderer::GltfRenderer(VkPhysicalDevice physical, VkDevice device,
                           VkQueue queue, std::uint32_t family,
                           VkExtent2D extent, VkFormat output_format,
                           const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(physical, device, extent)) {
  try {
    impl_->Initialize(queue, family, output_format, LoadGltf(path));
  } catch (const std::exception& error) {
    const auto kName = path.u8string();
    throw std::runtime_error(
        std::string(reinterpret_cast<const char*>(kName.c_str())) + ": " +
        error.what());
  }
}

GltfRenderer::~GltfRenderer() = default;

void GltfRenderer::Record(VkCommandBuffer cmd, VkImageView output,
                          float time) const {
  impl_->Record(cmd, output, time);
}

}  // namespace deep_infer
