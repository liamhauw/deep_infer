#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_win32.h>
#include <imgui.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cuda_check.h"
#include "scene.h"
#include "vk_check.h"

// This declaration uses the symbol name exported by the Dear ImGui backend.
// NOLINTNEXTLINE(readability-identifier-naming)
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM,
                                                             LPARAM);

namespace {

constexpr const wchar_t* kWindowClass = L"DeepInfer";
constexpr std::uint32_t kMinImages = 2;
constexpr auto kForever = std::numeric_limits<std::uint64_t>::max();
constexpr auto kDefaultEnginePath =
    L"resource/model/cnnps_feat16_conv8_scale2_1x3x320x640_static_fp32_op20.trt";

std::filesystem::path DefaultScenePath() {
  std::filesystem::path resource_path(L"resource/scene/studio.glb");
  if (std::filesystem::is_regular_file(resource_path)) {
    return resource_path;
  }
  std::wstring exe(32768, L'\0');
  const DWORD kLength =
      GetModuleFileNameW(nullptr, exe.data(), static_cast<DWORD>(exe.size()));
  if (kLength == 0 || kLength >= exe.size()) {
    throw std::runtime_error("Cannot locate the bundled glTF scene");
  }
  exe.resize(kLength);
  return std::filesystem::path(exe).parent_path() / L"studio.glb";
}

LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wparam,
                            LPARAM lparam) {
  if (ImGui_ImplWin32_WndProcHandler(window, message, wparam, lparam)) {
    return 1;
  }
  if (message == WM_CLOSE) {
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProcW(window, message, wparam, lparam);
}

void ReportVkError(VkResult result) {
  if (result < 0) {
    CheckVk(result, "Dear ImGui Vulkan backend");
  }
}

VkSemaphoreSubmitInfo SemaphoreInfo(VkSemaphore semaphore,
                                    VkPipelineStageFlags2 stage) {
  VkSemaphoreSubmitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
  info.semaphore = semaphore;
  info.stageMask = stage;
  return info;
}

void Submit(VkQueue queue, VkCommandBuffer cmd,
            std::span<const VkSemaphoreSubmitInfo> waits,
            std::span<const VkSemaphoreSubmitInfo> signals, VkFence fence) {
  VkCommandBufferSubmitInfo command{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
  };
  command.commandBuffer = cmd;
  command.deviceMask = 1;
  VkSubmitInfo2 info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
  info.waitSemaphoreInfoCount = static_cast<std::uint32_t>(waits.size());
  info.pWaitSemaphoreInfos = waits.data();
  info.commandBufferInfoCount = 1;
  info.pCommandBufferInfos = &command;
  info.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signals.size());
  info.pSignalSemaphoreInfos = signals.data();
  CheckVk(vkQueueSubmit2(queue, 1, &info, fence), "Submit Vulkan commands");
}

void BeginCommands(VkDevice device, VkCommandPool pool, VkCommandBuffer cmd) {
  CheckVk(vkResetCommandPool(device, pool, 0), "Reset command pool");
  VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CheckVk(vkBeginCommandBuffer(cmd, &info), "Begin command buffer");
}

class App {
 public:
  App() = default;
  App(const App&) = delete;
  App& operator=(const App&) = delete;
  ~App() {
    if (device_) {
      vkDeviceWaitIdle(device_);
    }
    if (imgui_vulkan_) {
      if (upscaled_texture_) {
        ImGui_ImplVulkan_RemoveTexture(upscaled_texture_);
      }
      if (native_texture_) {
        ImGui_ImplVulkan_RemoveTexture(native_texture_);
      }
      ImGui_ImplVulkan_Shutdown();
    }
    if (imgui_win32_) {
      ImGui_ImplWin32_Shutdown();
    }
    if (ImGui::GetCurrentContext()) {
      ImGui::DestroyContext();
    }
    scene_.reset();
    if (device_) {
      vkDestroyCommandPool(device_, scene_pool_, nullptr);
      if (swapchain_.Swapchain) {
        ImGui_ImplVulkanH_DestroyWindow(instance_, device_, &swapchain_,
                                        nullptr);
      }
      vkDestroyDevice(device_, nullptr);
    }
    if (surface_) {
      vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (instance_) {
      vkDestroyInstance(instance_, nullptr);
    }
    if (window_) {
      DestroyWindow(window_);
    }
    if (window_class_) {
      UnregisterClassW(kWindowClass, module_);
    }
  }

  void Initialize(const std::filesystem::path& engine_path,
                  const std::filesystem::path& scene_path) {
    CreateWindowHandle();
    CreateInstance();
    SelectDevice();
    CreateDevice();
    CreateSceneCommands();
    scene_ = std::make_unique<deep_infer::Scene>(
        physical_, device_, queue_, queue_family_, engine_path, scene_path);
    CreateSwapchain();
    InitializeUi();
    ShowWindow(window_, SW_SHOWNORMAL);
  }

  void Run() {
    const auto kStart = std::chrono::steady_clock::now();
    for (;;) {
      MSG message{};
      while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) {
          return;
        }
        TranslateMessage(&message);
        DispatchMessageW(&message);
      }
      if (IsIconic(window_)) {
        WaitMessage();
        continue;
      }
      RECT rect{};
      GetClientRect(window_, &rect);
      if (rect.right <= 0 || rect.bottom <= 0) {
        WaitMessage();
        continue;
      }
      if (resize_ || rect.right != swapchain_.Width ||
          rect.bottom != swapchain_.Height) {
        CreateSwapchain();
      }
      if (last_fence_) {
        CheckVk(vkWaitForFences(device_, 1, &last_fence_, VK_TRUE, kForever),
                "Wait for previous frame");
      }
      DrawUi();
      const auto kElapsed = std::chrono::duration<float>(
          std::chrono::steady_clock::now() - kStart);
      DrawFrame(kElapsed.count());
    }
  }

 private:
  void CreateWindowHandle() {
    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSW window_class{};
    window_class.lpfnWndProc = WindowProc;
    window_class.hInstance = module_;
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    window_class.lpszClassName = kWindowClass;
    window_class_ = RegisterClassW(&window_class);
    if (!window_class_) {
      throw std::runtime_error("Cannot register window class");
    }
    window_ = CreateWindowExW(0, kWindowClass, L"Deep infer",
                              WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              1120, 660, nullptr, nullptr, module_, nullptr);
    if (!window_) {
      throw std::runtime_error("Cannot create window");
    }
  }

  void CreateInstance() {
    const VkApplicationInfo kApp{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Deep infer",
        0,
        nullptr,
        0,
        VK_API_VERSION_1_3,
    };
    constexpr std::array kExtensions{
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    };
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &kApp;
    info.enabledExtensionCount = static_cast<std::uint32_t>(kExtensions.size());
    info.ppEnabledExtensionNames = kExtensions.data();
    CheckVk(vkCreateInstance(&info, nullptr, &instance_),
            "Create Vulkan 1.3 instance");
    VkWin32SurfaceCreateInfoKHR surface{
        VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
    };
    surface.hinstance = module_;
    surface.hwnd = window_;
    CheckVk(vkCreateWin32SurfaceKHR(instance_, &surface, nullptr, &surface_),
            "Create Win32 surface");
  }

  void SelectDevice() {
    int cuda_count = 0;
    CheckCuda(cudaGetDeviceCount(&cuda_count), "Enumerate CUDA devices");
    std::vector<cudaDeviceProp> cuda_devices(
        static_cast<std::size_t>(cuda_count));
    for (int i = 0; i < cuda_count; ++i) {
      CheckCuda(cudaGetDeviceProperties(&cuda_devices[i], i),
                "Query CUDA device");
    }
    std::uint32_t count = 0;
    CheckVk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
            "Enumerate Vulkan devices");
    std::vector<VkPhysicalDevice> devices(count);
    CheckVk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
            "Read Vulkan devices");
    for (VkPhysicalDevice physical : devices) {
      VkPhysicalDeviceIDProperties id{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
      };
      VkPhysicalDeviceProperties2 properties{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
          &id,
      };
      vkGetPhysicalDeviceProperties2(physical, &properties);
      if (properties.properties.apiVersion < VK_API_VERSION_1_3) {
        continue;
      }
      const auto kMatch =
          std::ranges::find_if(cuda_devices, [&id](const cudaDeviceProp& cuda) {
            return std::memcmp(id.deviceUUID, cuda.uuid.bytes, VK_UUID_SIZE) ==
                   0;
          });
      if (kMatch == cuda_devices.end() || !FindQueue(physical)) {
        continue;
      }
      physical_ = physical;
      CheckCuda(cudaSetDevice(static_cast<int>(kMatch - cuda_devices.begin())),
                "Match Vulkan and CUDA device");
      std::printf("Vulkan and CUDA device: %s\n",
                  properties.properties.deviceName);
      return;
    }
    throw std::runtime_error(
        "No matching Vulkan 1.3 and CUDA GPU with graphics, compute and "
        "presentation");
  }

  bool FindQueue(VkPhysicalDevice physical) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
    constexpr auto kRequired = VkQueueFlags{VK_QUEUE_GRAPHICS_BIT} |
                               VkQueueFlags{VK_QUEUE_COMPUTE_BIT};
    for (std::uint32_t i = 0; i < count; ++i) {
      VkBool32 present = VK_FALSE;
      CheckVk(
          vkGetPhysicalDeviceSurfaceSupportKHR(physical, i, surface_, &present),
          "Check presentation support");
      if (present && (queues[i].queueFlags & kRequired) == kRequired) {
        queue_family_ = i;
        return true;
      }
    }
    return false;
  }

  void CreateDevice() {
    VkPhysicalDeviceVulkan13Features features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceFeatures2 available{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &features,
    };
    vkGetPhysicalDeviceFeatures2(physical_, &available);
    if (!features.synchronization2 || !features.dynamicRendering) {
      throw std::runtime_error(
          "The Vulkan GPU must support synchronization2 and dynamic rendering");
    }
    features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features.synchronization2 = VK_TRUE;
    features.dynamicRendering = VK_TRUE;
    constexpr std::array kExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    };
    constexpr float kPriority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    };
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &kPriority;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue_info;
    info.enabledExtensionCount = static_cast<std::uint32_t>(kExtensions.size());
    info.ppEnabledExtensionNames = kExtensions.data();
    CheckVk(vkCreateDevice(physical_, &info, nullptr, &device_),
            "Create Vulkan external-memory device");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
  }

  void CreateSceneCommands() {
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.queueFamilyIndex = queue_family_;
    CheckVk(vkCreateCommandPool(device_, &info, nullptr, &scene_pool_),
            "Create scene command pool");
    VkCommandBufferAllocateInfo allocate{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    };
    allocate.commandPool = scene_pool_;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    CheckVk(vkAllocateCommandBuffers(device_, &allocate, &scene_cmd_),
            "Allocate scene command buffer");
  }

  void CreateSwapchain() {
    RECT rect{};
    GetClientRect(window_, &rect);
    CheckVk(vkDeviceWaitIdle(device_), "Wait before creating swapchain");
    last_fence_ = VK_NULL_HANDLE;
    swapchain_.Surface = surface_;
    constexpr std::array kFormats{
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
    };
    swapchain_.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        physical_, surface_, kFormats.data(), static_cast<int>(kFormats.size()),
        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
    swapchain_.PresentMode = VK_PRESENT_MODE_FIFO_KHR;
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        instance_, physical_, device_, &swapchain_, queue_family_, nullptr,
        rect.right, rect.bottom, kMinImages, 0);
    if (!swapchain_.Swapchain || !swapchain_.RenderPass ||
        swapchain_.ImageCount < kMinImages) {
      throw std::runtime_error("Cannot create swapchain");
    }
    swapchain_.SemaphoreIndex = 0;
    resize_ = false;
  }

  void InitializeUi() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowPadding = {18, 16};
    style.FramePadding = {12, 8};
    style.FrameRounding = 5;
    style.Colors[ImGuiCol_WindowBg] = {0.055F, 0.065F, 0.090F, 1.0F};
    style.Colors[ImGuiCol_Button] = {0.10F, 0.32F, 0.42F, 1.0F};
    imgui_win32_ = ImGui_ImplWin32_Init(window_);
    if (!imgui_win32_) {
      throw std::runtime_error("Cannot initialize ImGui Win32 backend");
    }

    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = instance_;
    info.PhysicalDevice = physical_;
    info.Device = device_;
    info.QueueFamily = queue_family_;
    info.Queue = queue_;
    info.DescriptorPoolSize =
        IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + 2;
    info.MinImageCount = kMinImages;
    info.ImageCount = swapchain_.ImageCount;
    info.PipelineInfoMain.RenderPass = swapchain_.RenderPass;
    info.CheckVkResultFn = ReportVkError;
    imgui_vulkan_ = ImGui_ImplVulkan_Init(&info);
    if (!imgui_vulkan_) {
      throw std::runtime_error("Cannot initialize ImGui Vulkan backend");
    }
    native_texture_ = ImGui_ImplVulkan_AddTexture(
        scene_->InputView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    upscaled_texture_ = ImGui_ImplVulkan_AddTexture(
        scene_->OutputView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void DrawUi() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    const auto* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    constexpr auto kWindowFlags = unsigned{ImGuiWindowFlags_NoDecoration} |
                                  unsigned{ImGuiWindowFlags_NoMove} |
                                  unsigned{ImGuiWindowFlags_NoSavedSettings} |
                                  unsigned{ImGuiWindowFlags_NoScrollbar};
    ImGui::Begin("Scene", nullptr, static_cast<ImGuiWindowFlags>(kWindowFlags));
    if (ImGui::Button(upscale_ ? "Super-resolution: ON###Upscale"
                               : "Super-resolution: OFF###Upscale")) {
      upscale_ = !upscale_;
    }
    ImGui::SameLine();
    const auto kExtent =
        upscale_ ? scene_->OutputExtent() : scene_->InputExtent();
    ImGui::Text("%u x %u   |   %.1f FPS", kExtent.width, kExtent.height,
                ImGui::GetIO().Framerate);
    const auto kAvailable = ImGui::GetContentRegionAvail();
    if (kAvailable.x > 0 && kAvailable.y > 0) {
      const float kAspect = static_cast<float>(kExtent.width) /
                            static_cast<float>(kExtent.height);
      const float kWidth = std::min(kAvailable.x, kAvailable.y * kAspect);
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                           (kAvailable.x - kWidth) * 0.5F);
      const VkDescriptorSet kTexture =
          upscale_ ? upscaled_texture_ : native_texture_;
      ImGui::Image(
          static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(kTexture)),
          {kWidth, kWidth / kAspect});
    }
    ImGui::End();
    ImGui::Render();
  }

  void DrawFrame(float time) {
    const int kSemaphoreIndex = static_cast<int>(swapchain_.SemaphoreIndex);
    const VkSemaphore kAcquired =
        swapchain_.FrameSemaphores[kSemaphoreIndex].ImageAcquiredSemaphore;
    const VkResult kAcquireResult = vkAcquireNextImageKHR(
        device_, swapchain_.Swapchain, kForever, kAcquired, VK_NULL_HANDLE,
        &swapchain_.FrameIndex);
    if (kAcquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
      resize_ = true;
      return;
    }
    if (kAcquireResult != VK_SUBOPTIMAL_KHR) {
      CheckVk(kAcquireResult, "Acquire swapchain image");
    }
    const int kFrameIndex = static_cast<int>(swapchain_.FrameIndex);
    auto& frame = swapchain_.Frames[kFrameIndex];
    const VkSemaphore kPresented =
        swapchain_.FrameSemaphores[kFrameIndex].RenderCompleteSemaphore;

    BeginCommands(device_, frame.CommandPool, frame.CommandBuffer);
    if (upscale_) {
      BeginCommands(device_, scene_pool_, scene_cmd_);
      scene_->RecordScene(scene_cmd_, time, true);
      CheckVk(vkEndCommandBuffer(scene_cmd_), "Finish scene commands");
      scene_->RecordOutput(frame.CommandBuffer);
    } else {
      scene_->RecordScene(frame.CommandBuffer, time, false);
    }
    RecordPresentation(frame);
    CheckVk(vkEndCommandBuffer(frame.CommandBuffer),
            "Finish presentation commands");
    CheckVk(vkResetFences(device_, 1, &frame.Fence), "Reset frame fence");

    std::array waits{
        SemaphoreInfo(kAcquired,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT),
        SemaphoreInfo(scene_->OutputReady(),
                      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
    };
    if (upscale_) {
      const std::array kSignals{
          SemaphoreInfo(scene_->InputReady(),
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
      };
      Submit(queue_, scene_cmd_, {}, kSignals, VK_NULL_HANDLE);
      scene_->EnqueueUpscale();
    }
    const std::array kSignals{
        SemaphoreInfo(kPresented, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
    };
    Submit(queue_, frame.CommandBuffer,
           std::span(waits).first(upscale_ ? 2 : 1), kSignals, frame.Fence);
    last_fence_ = frame.Fence;

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &kPresented;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_.Swapchain;
    present.pImageIndices = &swapchain_.FrameIndex;
    const VkResult kPresentResult = vkQueuePresentKHR(queue_, &present);
    if (kPresentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        kPresentResult == VK_SUBOPTIMAL_KHR) {
      resize_ = true;
    } else {
      CheckVk(kPresentResult, "Present");
      if (!presented_mode_ || *presented_mode_ != upscale_) {
        const auto kExtent =
            upscale_ ? scene_->OutputExtent() : scene_->InputExtent();
        std::printf("Presented %s frame (%u x %u)\n",
                    upscale_ ? "TensorRT-RTX" : "original Vulkan",
                    kExtent.width, kExtent.height);
        std::fflush(stdout);
        presented_mode_ = upscale_;
      }
    }
    if (kAcquireResult == VK_SUBOPTIMAL_KHR) {
      resize_ = true;
    }
    swapchain_.SemaphoreIndex =
        (swapchain_.SemaphoreIndex + 1) % swapchain_.SemaphoreCount;
  }

  void RecordPresentation(const ImGui_ImplVulkanH_Frame& frame) const {
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = swapchain_.RenderPass;
    pass.framebuffer = frame.Framebuffer;
    pass.renderArea.extent = {
        static_cast<std::uint32_t>(swapchain_.Width),
        static_cast<std::uint32_t>(swapchain_.Height),
    };
    pass.clearValueCount = 1;
    pass.pClearValues = &swapchain_.ClearValue;
    vkCmdBeginRenderPass(frame.CommandBuffer, &pass,
                         VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.CommandBuffer);
    vkCmdEndRenderPass(frame.CommandBuffer);
  }

  HINSTANCE module_{GetModuleHandleW(nullptr)};
  ATOM window_class_{};
  HWND window_{};
  VkInstance instance_{};
  VkSurfaceKHR surface_{};
  VkPhysicalDevice physical_{};
  VkDevice device_{};
  VkQueue queue_{};
  std::uint32_t queue_family_{};
  ImGui_ImplVulkanH_Window swapchain_;
  VkCommandPool scene_pool_{};
  VkCommandBuffer scene_cmd_{};
  VkFence last_fence_{};
  std::unique_ptr<deep_infer::Scene> scene_;
  VkDescriptorSet native_texture_{};
  VkDescriptorSet upscaled_texture_{};
  bool imgui_win32_{};
  bool imgui_vulkan_{};
  bool upscale_{true};
  std::optional<bool> presented_mode_;
  bool resize_{};
};

}  // namespace

// The Windows CRT expects this entry-point name.
// NOLINTNEXTLINE(readability-identifier-naming)
int wmain(int argc, wchar_t** argv) try {
  if (argc > 3 || (argc == 2 && std::wstring_view(argv[1]) == L"--help")) {
    std::fputs(
        "Usage: deep_infer_engine [engine.trt "
        "[scene.gltf|scene.glb|--procedural]]\n",
        argc > 3 ? stderr : stdout);
    return argc > 3 ? EXIT_FAILURE : EXIT_SUCCESS;
  }
  const bool kProcedural =
      argc == 3 && std::wstring_view(argv[2]) == L"--procedural";
  if (argc == 3 && !kProcedural && std::wstring_view(argv[2]).empty()) {
    std::fputs("Expected a glTF scene path\n", stderr);
    return EXIT_FAILURE;
  }
  std::filesystem::path scene_path;
  if (!kProcedural) {
    scene_path =
        argc == 3 ? std::filesystem::path(argv[2]) : DefaultScenePath();
  }
  App app;
  app.Initialize(argc >= 2 ? argv[1] : kDefaultEnginePath, scene_path);
  app.Run();
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fputs("Unexpected error\n", stderr);
  return EXIT_FAILURE;
}
