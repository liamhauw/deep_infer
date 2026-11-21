#include "png_io.h"

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cwctype>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace png_io {
namespace {

constexpr int kChannels = 3;
constexpr float kMaxByte = 255.0F;

std::size_t GetPixelCount(ImageSize size) {
  if (size.width <= 0 || size.height <= 0 ||
      size.width > std::numeric_limits<int>::max() / kChannels / size.height) {
    throw std::runtime_error("PNG dimensions exceed the codec buffer limit");
  }
  return static_cast<std::size_t>(size.width) * size.height;
}

void CheckImageSize(ImageSize expected, int width, int height) {
  if (width != expected.width || height != expected.height) {
    throw std::runtime_error(std::format(
        "Incorrect PNG size: {}x{}; engine requires {}x{} (width x height)",
        width, height, expected.width, expected.height));
  }
}

void WriteEncodedBytes(void* context, void* data, int size) {
  auto& file = *static_cast<std::ofstream*>(context);
  file.write(static_cast<const char*>(data), size);
}

}  // namespace

bool IsPngPath(const std::filesystem::path& path) {
  return std::ranges::equal(path.extension().wstring(),
                            std::wstring_view(L".png"),
                            [](wchar_t actual, wchar_t expected) {
                              return std::towlower(actual) == expected;
                            });
}

std::vector<float> ReadPng(const std::filesystem::path& path, ImageSize size) {
  const auto kPixels = GetPixelCount(size);
  const auto kFileName = path.u8string();
  const auto* bytes = reinterpret_cast<const char*>(kFileName.c_str());
  int width = 0;
  int height = 0;
  if (!stbi_info(bytes, &width, &height, nullptr)) {
    throw std::runtime_error("Cannot read PNG header: " + path.string());
  }
  CheckImageSize(size, width, height);
  const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> kPixelsRgb(
      stbi_load(bytes, &width, &height, nullptr, kChannels), stbi_image_free);
  if (!kPixelsRgb) {
    throw std::runtime_error("Cannot decode PNG: " + path.string());
  }
  CheckImageSize(size, width, height);

  const auto* pixels = kPixelsRgb.get();
  std::vector<float> rgb(kChannels * kPixels);
  for (std::size_t i = 0; i < kPixels; ++i) {
    rgb[i] = static_cast<float>(pixels[kChannels * i]) / kMaxByte;
    rgb[kPixels + i] = static_cast<float>(pixels[kChannels * i + 1]) / kMaxByte;
    rgb[2 * kPixels + i] =
        static_cast<float>(pixels[kChannels * i + 2]) / kMaxByte;
  }
  return rgb;
}

void WritePng(const std::filesystem::path& path, ImageSize size,
              std::span<const float> rgb) {
  const auto kPixels = GetPixelCount(size);
  if (rgb.size() != kChannels * kPixels) {
    throw std::runtime_error("Invalid RGB tensor size for PNG output");
  }
  const auto kToByte = [](float value) {
    return static_cast<stbi_uc>(
        std::lround(std::clamp(value, 0.0F, 1.0F) * kMaxByte));
  };
  std::vector<stbi_uc> pixels(rgb.size());
  for (std::size_t i = 0; i < kPixels; ++i) {
    pixels[kChannels * i] = kToByte(rgb[i]);
    pixels[kChannels * i + 1] = kToByte(rgb[kPixels + i]);
    pixels[kChannels * i + 2] = kToByte(rgb[2 * kPixels + i]);
  }

  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open output PNG: " + path.string());
  }
  const int kWritten = stbi_write_png_to_func(
      WriteEncodedBytes, &file, size.width, size.height, kChannels,
      pixels.data(), static_cast<std::ptrdiff_t>(size.width) * kChannels);
  file.close();
  if (!kWritten || !file) {
    throw std::runtime_error("Cannot write output PNG: " + path.string());
  }
}

}  // namespace png_io
