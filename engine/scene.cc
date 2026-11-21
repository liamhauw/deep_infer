#include "scene.h"

#include <NvInferRuntime.h>
#include <windows.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <span>
#include <utility>
#include <vector>

#include "cuda_check.h"
#include "vk_check.h"

namespace deep_infer {
namespace {

constexpr VkFormat kImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr auto kMemoryHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr auto kSemaphoreHandle =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr auto kTensorUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
constexpr VkImageSubresourceRange kColorRange{
    VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1,
};

constexpr auto kVertexCode = std::to_array<std::uint32_t>(
#include "scene.vert.inc"
);
constexpr auto kFragmentCode = std::to_array<std::uint32_t>(
#include "scene.frag.inc"
);
constexpr auto kPackCode = std::to_array<std::uint32_t>(
#include "pack.comp.inc"
);
constexpr auto kUnpackCode = std::to_array<std::uint32_t>(
#include "unpack.comp.inc"
);

struct Image {
  VkImage image{};
  VkDeviceMemory memory{};
  VkImageView view{};
};

struct SharedBuffer {
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  cudaExternalMemory_t cuda_memory{};
  void* address{};
};

struct SharedSemaphore {
  VkSemaphore semaphore{};
  cudaExternalSemaphore_t cuda_semaphore{};
};

class Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TensorRT-RTX] %s\n", message);
    }
  }
};

struct Tensor {
  const char* name{};
  VkExtent2D extent{};

  VkDeviceSize Bytes() const {
    return static_cast<VkDeviceSize>(extent.width) * extent.height * 3 *
           sizeof(float);
  }
};

Tensor GetTensor(const nvinfer1::ICudaEngine& engine,
                 nvinfer1::TensorIOMode mode,
                 const VkPhysicalDeviceLimits& limits) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char* name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) != mode) {
      continue;
    }
    const auto kShape = engine.getTensorShape(name);
    if (kShape.nbDims != 4 || kShape.d[0] != 1 || kShape.d[1] != 3 ||
        kShape.d[2] <= 0 || kShape.d[3] <= 0 ||
        std::cmp_greater(kShape.d[2], limits.maxImageDimension2D) ||
        std::cmp_greater(kShape.d[3], limits.maxImageDimension2D) ||
        engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
        engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
        engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
      throw std::runtime_error(
          "Requires static, linear FP32 1x3xHxW GPU tensors");
    }
    const Tensor kTensor{
        name,
        {
            static_cast<std::uint32_t>(kShape.d[3]),
            static_cast<std::uint32_t>(kShape.d[2]),
        },
    };
    if (kTensor.Bytes() > limits.maxStorageBufferRange) {
      throw std::runtime_error(
          "Engine tensor exceeds Vulkan's storage buffer range");
    }
    return kTensor;
  }
  throw std::runtime_error("Engine is missing an input or output tensor");
}

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

void TransitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout old_layout, VkImageLayout new_layout,
                     VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                     VkPipelineStageFlags2 dst_stage,
                     VkAccessFlags2 dst_access) {
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
  barrier.subresourceRange = kColorRange;
  VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  vkCmdPipelineBarrier2(cmd, &dependency);
}

}  // namespace

struct Scene::Impl {
  Impl(VkPhysicalDevice physical, VkDevice device, std::uint32_t queue_family)
      : physical(physical), device(device), queue_family(queue_family) {}

  ~Impl() {
    if (stream) {
      cudaStreamSynchronize(stream);
    }
    context.reset();
    if (stream) {
      cudaStreamDestroy(stream);
    }
    DestroySemaphore(output_ready);
    DestroySemaphore(input_ready);
    vkDestroyPipeline(device, unpack, nullptr);
    vkDestroyPipeline(device, pack, nullptr);
    vkDestroyPipeline(device, graphics, nullptr);
    vkDestroyPipelineLayout(device, compute_layout, nullptr);
    vkDestroyPipelineLayout(device, graphics_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptors, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    DestroyBuffer(input);
    DestroyBuffer(output);
    DestroyImage(native_image);
    DestroyImage(upscaled_image);
  }

  void Initialize(const std::filesystem::path& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    file.seekg(0, std::ios::end);
    const auto kBytes = static_cast<std::streamoff>(file.tellg());
    if (kBytes <= 0) {
      throw std::runtime_error("Cannot open engine: " + engine_path.string());
    }
    std::vector<char> plan(static_cast<std::size_t>(kBytes));
    file.seekg(0);
    if (!file.read(plan.data(), kBytes)) {
      throw std::runtime_error("Cannot read engine: " + engine_path.string());
    }
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
      throw std::runtime_error("Cannot create TensorRT-RTX runtime");
    }
    engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine || engine->getNbIOTensors() != 2) {
      throw std::runtime_error(
          "Requires an engine with one input and one output");
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    input_tensor =
        GetTensor(*engine, nvinfer1::TensorIOMode::kINPUT, properties.limits);
    output_tensor =
        GetTensor(*engine, nvinfer1::TensorIOMode::kOUTPUT, properties.limits);
    CheckExternalSupport();
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "Create inference stream");

    CreateBuffer(input_tensor.Bytes(), input);
    CreateBuffer(output_tensor.Bytes(), output);
    CreateSemaphore(input_ready);
    CreateSemaphore(output_ready);
    CreateImage(input_tensor.extent, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                native_image);
    CreateImage(output_tensor.extent, 0, upscaled_image);
    CreatePipelines();

    context.reset(engine->createExecutionContext());
    if (!context ||
        !context->setInputTensorAddress(input_tensor.name, input.address) ||
        !context->setOutputTensorAddress(output_tensor.name, output.address)) {
      throw std::runtime_error(
          "Cannot bind exported Vulkan buffers to TensorRT-RTX tensors");
    }
  }

  void CheckExternalSupport() const {
    const VkPhysicalDeviceExternalBufferInfo kBufferInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .usage = kTensorUsage,
        .handleType = kMemoryHandle,
    };
    VkExternalBufferProperties buffer_properties{
        VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
    };
    vkGetPhysicalDeviceExternalBufferProperties(physical, &kBufferInfo,
                                                &buffer_properties);
    const VkPhysicalDeviceExternalSemaphoreInfo kSemaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
        .handleType = kSemaphoreHandle,
    };
    VkExternalSemaphoreProperties semaphore_properties{
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
    };
    vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &kSemaphoreInfo,
                                                   &semaphore_properties);
    constexpr VkExternalMemoryFeatureFlags kExportableMemory =
        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
    constexpr VkExternalSemaphoreFeatureFlags kExportableSemaphore =
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT;
    if (!(buffer_properties.externalMemoryProperties.externalMemoryFeatures &
          kExportableMemory) ||
        !(semaphore_properties.externalSemaphoreFeatures &
          kExportableSemaphore)) {
      throw std::runtime_error(
          "Vulkan device cannot export Win32 storage buffers and semaphores");
    }
  }

  std::uint32_t DeviceMemoryIndex(std::uint32_t bits) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((bits & (1U << i)) != 0 &&
          (properties.memoryTypes[i].propertyFlags &
           VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}) != 0) {
        return i;
      }
    }
    throw std::runtime_error("No compatible device-local Vulkan memory type");
  }

  void CreateImage(VkExtent2D extent, VkImageUsageFlags extra_usage,
                   Image& target) const {
    const VkImageCreateInfo kInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = kImageFormat,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VkImageUsageFlags{VK_IMAGE_USAGE_STORAGE_BIT} |
                 VkImageUsageFlags{VK_IMAGE_USAGE_SAMPLED_BIT} | extra_usage,
    };
    CheckVk(vkCreateImage(device, &kInfo, nullptr, &target.image),
            "Create scene image");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, target.image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = DeviceMemoryIndex(requirements.memoryTypeBits);
    CheckVk(vkAllocateMemory(device, &allocation, nullptr, &target.memory),
            "Allocate scene image");
    CheckVk(vkBindImageMemory(device, target.image, target.memory, 0),
            "Bind scene image");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = kImageFormat;
    view.subresourceRange = kColorRange;
    CheckVk(vkCreateImageView(device, &view, nullptr, &target.view),
            "Create scene image view");
  }

  void CreateBuffer(VkDeviceSize bytes, SharedBuffer& target) const {
    VkExternalMemoryBufferCreateInfo external{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    };
    external.handleTypes = kMemoryHandle;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    info.size = bytes;
    info.usage = kTensorUsage;
    CheckVk(vkCreateBuffer(device, &info, nullptr, &target.buffer),
            "Create shared tensor buffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, target.buffer, &requirements);
    VkMemoryDedicatedAllocateInfo dedicated{
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
    };
    dedicated.buffer = target.buffer;
    VkExportMemoryAllocateInfo export_info{
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        &dedicated,
    };
    export_info.handleTypes = kMemoryHandle;
    VkMemoryAllocateInfo allocation{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        &export_info,
    };
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = DeviceMemoryIndex(requirements.memoryTypeBits);
    CheckVk(vkAllocateMemory(device, &allocation, nullptr, &target.memory),
            "Allocate shared tensor buffer");
    CheckVk(vkBindBufferMemory(device, target.buffer, target.memory, 0),
            "Bind shared tensor buffer");

    const auto kGetHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
    if (!kGetHandle) {
      throw std::runtime_error("Vulkan Win32 memory export is unavailable");
    }
    const VkMemoryGetWin32HandleInfoKHR kHandleInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory = target.memory,
        .handleType = kMemoryHandle,
    };
    HANDLE handle{};
    CheckVk(kGetHandle(device, &kHandleInfo, &handle), "Export tensor buffer");
    const std::unique_ptr<void, decltype(&CloseHandle)> kHandle(handle,
                                                                CloseHandle);
    const cudaExternalMemoryHandleDesc kMemory{
        .type = cudaExternalMemoryHandleTypeOpaqueWin32,
        .handle = {.win32 = {.handle = kHandle.get()}},
        .size = requirements.size,
        .flags = cudaExternalMemoryDedicated,
    };
    CheckCuda(cudaImportExternalMemory(&target.cuda_memory, &kMemory),
              "Import Vulkan memory into CUDA");
    cudaExternalMemoryBufferDesc mapping{};
    mapping.size = bytes;
    CheckCuda(cudaExternalMemoryGetMappedBuffer(&target.address,
                                                target.cuda_memory, &mapping),
              "Map shared tensor buffer");
  }

  void CreateSemaphore(SharedSemaphore& target) const {
    VkExportSemaphoreCreateInfo export_info{
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
    };
    export_info.handleTypes = kSemaphoreHandle;
    VkSemaphoreCreateInfo info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        &export_info,
    };
    CheckVk(vkCreateSemaphore(device, &info, nullptr, &target.semaphore),
            "Create shared semaphore");
    const auto kGetHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    if (!kGetHandle) {
      throw std::runtime_error("Vulkan Win32 semaphore export is unavailable");
    }
    const VkSemaphoreGetWin32HandleInfoKHR kHandleInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
        .semaphore = target.semaphore,
        .handleType = kSemaphoreHandle,
    };
    HANDLE handle{};
    CheckVk(kGetHandle(device, &kHandleInfo, &handle),
            "Export Vulkan semaphore");
    const std::unique_ptr<void, decltype(&CloseHandle)> kHandle(handle,
                                                                CloseHandle);
    const cudaExternalSemaphoreHandleDesc kSemaphore{
        .type = cudaExternalSemaphoreHandleTypeOpaqueWin32,
        .handle = {.win32 = {.handle = kHandle.get()}},
    };
    CheckCuda(cudaImportExternalSemaphore(&target.cuda_semaphore, &kSemaphore),
              "Import Vulkan semaphore into CUDA");
  }

  void DestroyImage(const Image& image) const {
    vkDestroyImageView(device, image.view, nullptr);
    vkDestroyImage(device, image.image, nullptr);
    vkFreeMemory(device, image.memory, nullptr);
  }

  void DestroyBuffer(const SharedBuffer& buffer) const {
    if (buffer.address) {
      cudaFree(buffer.address);
    }
    if (buffer.cuda_memory) {
      cudaDestroyExternalMemory(buffer.cuda_memory);
    }
    vkDestroyBuffer(device, buffer.buffer, nullptr);
    vkFreeMemory(device, buffer.memory, nullptr);
  }

  void DestroySemaphore(const SharedSemaphore& semaphore) const {
    if (semaphore.cuda_semaphore) {
      cudaDestroyExternalSemaphore(semaphore.cuda_semaphore);
    }
    vkDestroySemaphore(device, semaphore.semaphore, nullptr);
  }

  void CreatePipelines() {
    VkPushConstantRange time{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};
    VkPipelineLayoutCreateInfo graphics_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    graphics_info.pushConstantRangeCount = 1;
    graphics_info.pPushConstantRanges = &time;
    CheckVk(vkCreatePipelineLayout(device, &graphics_info, nullptr,
                                   &graphics_layout),
            "Create scene pipeline layout");
    CreateGraphicsPipeline();

    const std::array kBindings{
        VkDescriptorSetLayoutBinding{
            0,
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        },
        VkDescriptorSetLayoutBinding{
            1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            1,
            VK_SHADER_STAGE_COMPUTE_BIT,
            nullptr,
        },
    };
    VkDescriptorSetLayoutCreateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    };
    set_info.bindingCount = static_cast<std::uint32_t>(kBindings.size());
    set_info.pBindings = kBindings.data();
    CheckVk(
        vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout),
        "Create tensor descriptor layout");
    VkPipelineLayoutCreateInfo compute_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };
    compute_info.setLayoutCount = 1;
    compute_info.pSetLayouts = &set_layout;
    CheckVk(
        vkCreatePipelineLayout(device, &compute_info, nullptr, &compute_layout),
        "Create tensor pipeline layout");
    CreateComputePipeline(kPackCode, pack);
    CreateComputePipeline(kUnpackCode, unpack);

    const std::array kSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
    };
    VkDescriptorPoolCreateInfo pool{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    };
    pool.maxSets = 2;
    pool.poolSizeCount = static_cast<std::uint32_t>(kSizes.size());
    pool.pPoolSizes = kSizes.data();
    CheckVk(vkCreateDescriptorPool(device, &pool, nullptr, &descriptors),
            "Create tensor descriptor pool");
    const std::array kLayouts{set_layout, set_layout};
    VkDescriptorSetAllocateInfo sets{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    };
    sets.descriptorPool = descriptors;
    sets.descriptorSetCount = static_cast<std::uint32_t>(kLayouts.size());
    sets.pSetLayouts = kLayouts.data();
    CheckVk(vkAllocateDescriptorSets(device, &sets, tensor_sets.data()),
            "Allocate tensor descriptors");
    WriteDescriptors(tensor_sets[0], native_image.view, input.buffer,
                     input_tensor.Bytes());
    WriteDescriptors(tensor_sets[1], upscaled_image.view, output.buffer,
                     output_tensor.Bytes());
  }

  void CreateGraphicsPipeline() {
    const Shader kVertex(device, kVertexCode);
    const Shader kFragment(device, kFragmentCode);
    const std::array kStages{
        kVertex.Stage(VK_SHADER_STAGE_VERTEX_BIT),
        kFragment.Stage(VK_SHADER_STAGE_FRAGMENT_BIT),
    };
    VkPipelineVertexInputStateCreateInfo vertex{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
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
    raster.lineWidth = 1.0F;
    const VkPipelineMultisampleStateCreateInfo kSamples{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask =
        VkColorComponentFlags{VK_COLOR_COMPONENT_R_BIT} |
        VkColorComponentFlags{VK_COLOR_COMPONENT_G_BIT} |
        VkColorComponentFlags{VK_COLOR_COMPONENT_B_BIT} |
        VkColorComponentFlags{VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
    };
    blend.attachmentCount = 1;
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
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &kImageFormat;
    VkGraphicsPipelineCreateInfo info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        &rendering,
    };
    info.stageCount = static_cast<std::uint32_t>(kStages.size());
    info.pStages = kStages.data();
    info.pVertexInputState = &vertex;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &kSamples;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = graphics_layout;
    CheckVk(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr,
                                      &graphics),
            "Create scene graphics pipeline");
  }

  void CreateComputePipeline(std::span<const std::uint32_t> code,
                             VkPipeline& pipeline) const {
    const Shader kShader(device, code);
    const VkComputePipelineCreateInfo kInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = kShader.Stage(VK_SHADER_STAGE_COMPUTE_BIT),
        .layout = compute_layout,
    };
    CheckVk(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &kInfo, nullptr,
                                     &pipeline),
            "Create tensor compute pipeline");
  }

  void WriteDescriptors(VkDescriptorSet set, VkImageView view, VkBuffer buffer,
                        VkDeviceSize bytes) const {
    const VkDescriptorImageInfo kImage{
        VK_NULL_HANDLE,
        view,
        VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorBufferInfo kBuffer{buffer, 0, bytes};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::uint32_t i = 0; i < writes.size(); ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &kImage;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &kBuffer;
    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
  }

  void Dispatch(VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet set,
                VkExtent2D extent) const {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_layout,
                            0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);
  }

  void TransferBuffers(VkCommandBuffer cmd, bool release) const {
    std::array<VkBufferMemoryBarrier2, 2> barriers{};
    for (auto& barrier : barriers) {
      barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
      barrier.srcQueueFamilyIndex =
          release ? queue_family : VK_QUEUE_FAMILY_EXTERNAL;
      barrier.dstQueueFamilyIndex =
          release ? VK_QUEUE_FAMILY_EXTERNAL : queue_family;
      barrier.size = VK_WHOLE_SIZE;
    }
    barriers[0].buffer = input.buffer;
    barriers[1].buffer = output.buffer;
    if (release) {
      barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    } else {
      barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    }
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount =
        static_cast<std::uint32_t>(barriers.size());
    dependency.pBufferMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dependency);
  }

  void RecordScene(VkCommandBuffer cmd, float time, bool upscale) const {
    TransitionImage(cmd, native_image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo color{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
    };
    color.imageView = native_image.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    const VkRect2D kRect{{0, 0}, input_tensor.extent};
    const VkViewport kViewport{
        0,
        0,
        static_cast<float>(kRect.extent.width),
        static_cast<float>(kRect.extent.height),
        0,
        1,
    };
    VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
    rendering.renderArea = kRect;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color;
    vkCmdBeginRendering(cmd, &rendering);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics);
    vkCmdSetViewport(cmd, 0, 1, &kViewport);
    vkCmdSetScissor(cmd, 0, 1, &kRect);
    vkCmdPushConstants(cmd, graphics_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(time), &time);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    TransitionImage(cmd, native_image.image,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    upscale ? VK_IMAGE_LAYOUT_GENERAL
                            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    upscale ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                            : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    upscale ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                            : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    if (upscale) {
      Dispatch(cmd, pack, tensor_sets[0], input_tensor.extent);
      TransferBuffers(cmd, true);
    }
  }

  void RecordOutput(VkCommandBuffer cmd) const {
    TransferBuffers(cmd, false);
    TransitionImage(cmd, upscaled_image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
                    VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    Dispatch(cmd, unpack, tensor_sets[1], output_tensor.extent);
    TransitionImage(cmd, upscaled_image.image, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  }

  VkPhysicalDevice physical;
  VkDevice device;
  std::uint32_t queue_family;
  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  Tensor input_tensor;
  Tensor output_tensor;
  Image native_image;
  Image upscaled_image;
  SharedBuffer input;
  SharedBuffer output;
  SharedSemaphore input_ready;
  SharedSemaphore output_ready;
  cudaStream_t stream{};
  VkPipelineLayout graphics_layout{};
  VkPipelineLayout compute_layout{};
  VkPipeline graphics{};
  VkPipeline pack{};
  VkPipeline unpack{};
  VkDescriptorSetLayout set_layout{};
  VkDescriptorPool descriptors{};
  std::array<VkDescriptorSet, 2> tensor_sets{};
};

Scene::Scene(VkPhysicalDevice physical, VkDevice device,
             std::uint32_t queue_family,
             const std::filesystem::path& engine_path)
    : impl_(std::make_unique<Impl>(physical, device, queue_family)) {
  impl_->Initialize(engine_path);
}

Scene::~Scene() = default;
VkExtent2D Scene::InputExtent() const { return impl_->input_tensor.extent; }
VkExtent2D Scene::OutputExtent() const { return impl_->output_tensor.extent; }
VkImageView Scene::InputView() const { return impl_->native_image.view; }
VkImageView Scene::OutputView() const { return impl_->upscaled_image.view; }
VkSemaphore Scene::InputReady() const { return impl_->input_ready.semaphore; }
VkSemaphore Scene::OutputReady() const { return impl_->output_ready.semaphore; }

void Scene::RecordScene(VkCommandBuffer cmd, float time, bool upscale) {
  impl_->RecordScene(cmd, time, upscale);
}

void Scene::EnqueueUpscale() {
  const cudaExternalSemaphoreWaitParams kWait{};
  CheckCuda(cudaWaitExternalSemaphoresAsync(&impl_->input_ready.cuda_semaphore,
                                            &kWait, 1, impl_->stream),
            "Wait for Vulkan input");
  if (!impl_->context->enqueueV3(impl_->stream)) {
    throw std::runtime_error("TensorRT-RTX inference enqueue failed");
  }
  const cudaExternalSemaphoreSignalParams kSignal{};
  CheckCuda(
      cudaSignalExternalSemaphoresAsync(&impl_->output_ready.cuda_semaphore,
                                        &kSignal, 1, impl_->stream),
      "Signal Vulkan output");
}

void Scene::RecordOutput(VkCommandBuffer cmd) { impl_->RecordOutput(cmd); }

}  // namespace deep_infer
