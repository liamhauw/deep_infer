#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "scene_math.h"

namespace deep_infer {

constexpr std::size_t kTextureSlots = 5;

enum MaterialFlag : std::uint8_t {
  kAlphaMask = 1,
  kAlphaBlend = 2,
  kDoubleSided = 4,
  kUnlit = 8,
  kNormalMap = 16,
};

struct ModelVertex {
  Vec3 position{};
  Vec3 normal{};
  std::array<float, 4> tangent{};
  std::array<float, 4> color{1, 1, 1, 1};
  std::array<std::array<float, 2>, kTextureSlots> uv{};
};

struct ModelTexture {
  int width = 1;
  int height = 1;
  int mag_filter = 9729;
  int min_filter = 9987;
  int wrap_s = 10497;
  int wrap_t = 10497;
  bool srgb = false;
  std::vector<unsigned char> pixels{255, 255, 255, 255};
};

struct MaterialParams {
  std::array<float, 4> base_color{1, 1, 1, 1};
  std::array<float, 4> emissive_cutoff{0, 0, 0, 0.5F};
  std::array<float, 4> pbr{1, 1, 1, 1};
  std::uint32_t flags{};
};

struct ModelMaterial {
  MaterialParams params;
  std::array<std::uint32_t, kTextureSlots> textures{};
};

struct ModelDraw {
  std::uint32_t first_index, index_count, material;
  Vec3 center;
};

struct GltfModel {
  std::vector<ModelVertex> vertices;
  std::vector<std::uint32_t> indices;
  std::vector<ModelTexture> textures{ModelTexture{}};
  std::vector<ModelMaterial> materials;
  std::vector<ModelDraw> draws;
  Vec3 half_extent{};
};

GltfModel LoadGltf(const std::filesystem::path& path);

}  // namespace deep_infer
