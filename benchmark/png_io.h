#pragma once

#include <filesystem>
#include <span>
#include <vector>

namespace png_io {

struct ImageSize {
  int width;
  int height;
};

bool IsPngPath(const std::filesystem::path& path);
std::vector<float> ReadPng(const std::filesystem::path& path, ImageSize size);
void WritePng(const std::filesystem::path& path, ImageSize size,
              std::span<const float> rgb);

}  // namespace png_io
