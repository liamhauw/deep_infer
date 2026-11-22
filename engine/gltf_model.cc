#include "gltf_model.h"

#include <cgltf.h>
#include <stb_image.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace deep_infer {
namespace {

[[noreturn]] void Fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

cgltf_result ReadFile(const cgltf_memory_options* memory,
                      const cgltf_file_options* /*file_options*/,
                      const char* path, cgltf_size* size, void** data) {
  try {
    std::ifstream file(
        std::filesystem::path(reinterpret_cast<const char8_t*>(path)),
        std::ios::binary | std::ios::ate);
    if (!file) {
      return cgltf_result_file_not_found;
    }
    const auto kLength = static_cast<std::streamoff>(file.tellg());
    if (kLength <= 0 || std::cmp_greater(*size, kLength)) {
      return cgltf_result_data_too_short;
    }
    if (*size == 0) {
      *size = static_cast<cgltf_size>(kLength);
    }
    void* bytes = memory->alloc_func
                      ? memory->alloc_func(memory->user_data, *size)
                      : std::malloc(*size);
    if (!bytes) {
      return cgltf_result_out_of_memory;
    }
    if (!file.seekg(0).read(static_cast<char*>(bytes),
                            static_cast<std::streamsize>(*size))) {
      if (memory->free_func) {
        memory->free_func(memory->user_data, bytes);
      } else {
        std::free(bytes);
      }
      return cgltf_result_io_error;
    }
    *data = bytes;
    return cgltf_result_success;
  } catch (...) {
    return cgltf_result_io_error;
  }
}

void Check(cgltf_result result, const char* action) {
  constexpr std::array kErrors{
      "success",        "truncated file",
      "unknown format", "invalid JSON",
      "invalid glTF",   "invalid options",
      "file not found", "I/O error",
      "out of memory",  "glTF 1.0 is unsupported",
  };
  if (result != cgltf_result_success) {
    Fail(std::string(action) + ": " + kErrors.at(result));
  }
}

std::vector<unsigned char> ReadImage(const cgltf_image& image,
                                     const std::filesystem::path& directory) {
  if (image.buffer_view) {
    const auto* bytes = cgltf_buffer_view_data(image.buffer_view);
    if (!bytes) {
      Fail("Missing embedded glTF image buffer");
    }
    return {bytes, bytes + image.buffer_view->size};
  }
  if (!image.uri) {
    Fail("glTF texture has no PNG or JPEG image");
  }
  std::string uri(image.uri);
  if (uri.starts_with("data:")) {
    const auto kComma = uri.find(',');
    if (kComma == uri.npos ||
        !std::string_view(uri).substr(0, kComma).ends_with(";base64")) {
      Fail("Expected a base64 glTF image URI");
    }
    const std::string_view kEncoded(uri.data() + kComma + 1,
                                    uri.size() - kComma - 1);
    if (kEncoded.empty() || kEncoded.size() % 4 != 0) {
      Fail("Invalid base64 glTF image URI");
    }
    std::size_t padding = 0;
    if (kEncoded.ends_with('=')) {
      padding = kEncoded.ends_with("==") ? 2 : 1;
    }
    const auto kSize = kEncoded.size() / 4 * 3 - padding;
    void* raw = nullptr;
    const cgltf_options kOptions{};
    Check(cgltf_load_buffer_base64(&kOptions, kSize, uri.c_str() + kComma + 1,
                                   &raw),
          "Decode glTF image URI");
    const std::unique_ptr<unsigned char, decltype(&std::free)> kBytes(
        static_cast<unsigned char*>(raw), std::free);
    return {kBytes.get(), kBytes.get() + kSize};
  }

  uri.resize(cgltf_decode_uri(uri.data()));
  const auto kPath =
      directory /
      std::filesystem::path(reinterpret_cast<const char8_t*>(uri.c_str()));
  std::ifstream file(kPath, std::ios::binary | std::ios::ate);
  const auto kLength = static_cast<std::streamoff>(file.tellg());
  if (kLength <= 0 || kLength > INT_MAX) {
    Fail("Cannot open glTF texture: " + uri);
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(kLength));
  if (!file.seekg(0).read(reinterpret_cast<char*>(bytes.data()), kLength)) {
    Fail("Cannot read glTF texture: " + uri);
  }
  return bytes;
}

ModelTexture DecodeTexture(const cgltf_texture& texture, bool srgb,
                           const std::filesystem::path& directory) {
  if (!texture.image) {
    Fail("Texture requires an unsupported image extension");
  }
  const auto kBytes = ReadImage(*texture.image, directory);
  if (kBytes.size() > INT_MAX) {
    Fail("glTF texture is too large");
  }
  ModelTexture result;
  int channels = 0;
  std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
      stbi_load_from_memory(kBytes.data(), static_cast<int>(kBytes.size()),
                            &result.width, &result.height, &channels, 4),
      stbi_image_free);
  if (!pixels) {
    Fail(std::string("Cannot decode glTF PNG/JPEG: ") + stbi_failure_reason());
  }
  result.pixels.assign(
      pixels.get(),
      pixels.get() + std::size_t{4} * result.width * result.height);
  result.srgb = srgb;
  if (const auto* sampler = texture.sampler) {
    if (sampler->mag_filter) {
      result.mag_filter = sampler->mag_filter;
    }
    if (sampler->min_filter) {
      result.min_filter = sampler->min_filter;
    }
    result.wrap_s = sampler->wrap_s;
    result.wrap_t = sampler->wrap_t;
  }
  return result;
}

using TextureViews = std::array<const cgltf_texture_view*, kTextureSlots>;

TextureViews GetTextureViews(const cgltf_material* material) {
  if (!material) {
    return {};
  }
  return {
      &material->pbr_metallic_roughness.base_color_texture,
      &material->pbr_metallic_roughness.metallic_roughness_texture,
      &material->normal_texture,
      &material->occlusion_texture,
      &material->emissive_texture,
  };
}

void Unpack(const cgltf_accessor& source, std::vector<float>& values) {
  const auto kComponents = cgltf_num_components(source.type);
  values.resize(source.count * kComponents);
  auto base = source;
  base.is_sparse = false;
  if (cgltf_accessor_unpack_floats(&base, values.data(), values.size()) !=
      values.size()) {
    Fail("Cannot unpack glTF vertex attribute");
  }
  if (!source.is_sparse) {
    return;
  }

  const auto& sparse = source.sparse;
  auto sparse_values = base;
  sparse_values.count = sparse.count;
  sparse_values.buffer_view = sparse.values_buffer_view;
  sparse_values.offset = sparse.values_byte_offset;
  sparse_values.stride = cgltf_calc_size(base.type, base.component_type);
  cgltf_accessor locations{};
  locations.type = cgltf_type_scalar;
  locations.count = sparse.count;
  locations.buffer_view = sparse.indices_buffer_view;
  locations.offset = sparse.indices_byte_offset;
  locations.component_type = sparse.indices_component_type;
  locations.stride = cgltf_component_size(locations.component_type);
  for (std::size_t i = 0; i < sparse.count; ++i) {
    const auto kIndex = cgltf_accessor_read_index(&locations, i);
    if (kIndex >= source.count ||
        !cgltf_accessor_read_float(&sparse_values, i,
                                   values.data() + kIndex * kComponents,
                                   kComponents)) {
      Fail("Invalid sparse glTF vertex attribute");
    }
  }
}

template <typename Apply>
void ReadAttribute(const cgltf_primitive& primitive, cgltf_attribute_type type,
                   int set, std::size_t count, std::size_t min_components,
                   std::size_t max_components, Apply apply) {
  const auto* accessor = cgltf_find_accessor(&primitive, type, set);
  if (!accessor) {
    return;
  }
  const auto kComponents = cgltf_num_components(accessor->type);
  if (accessor->count != count || kComponents < min_components ||
      kComponents > max_components) {
    Fail("Invalid glTF vertex attribute dimensions");
  }
  std::vector<float> values;
  Unpack(*accessor, values);
  for (std::size_t i = 0; i < count; ++i) {
    const auto* value = values.data() + i * kComponents;
    if (!std::all_of(value, value + kComponents,
                     [](float element) { return std::isfinite(element); })) {
      Fail("Non-finite glTF vertex attribute");
    }
    apply(i, value, kComponents);
  }
}

std::vector<std::uint32_t> ReadIndices(const cgltf_primitive& primitive,
                                       std::size_t vertex_count) {
  const auto* accessor = primitive.indices;
  const std::size_t kCount = accessor ? accessor->count : vertex_count;
  if (kCount > UINT32_MAX) {
    Fail("Too many glTF indices");
  }
  std::vector<std::uint32_t> indices(kCount);
  if (!accessor) {
    std::iota(indices.begin(), indices.end(), 0U);
  } else {
    auto base = *accessor;
    base.is_sparse = false;
    if (base.buffer_view && cgltf_accessor_unpack_indices(
                                &base, indices.data(), 4, kCount) != kCount) {
      Fail("Cannot unpack glTF indices");
    }
    if (accessor->is_sparse) {
      const auto& sparse = accessor->sparse;
      auto values = base;
      values.buffer_view = sparse.values_buffer_view;
      values.offset = sparse.values_byte_offset;
      values.stride = cgltf_component_size(values.component_type);
      auto locations = base;
      locations.buffer_view = sparse.indices_buffer_view;
      locations.offset = sparse.indices_byte_offset;
      locations.component_type = sparse.indices_component_type;
      locations.stride = cgltf_component_size(locations.component_type);
      for (std::size_t i = 0; i < sparse.count; ++i) {
        const auto kIndex = cgltf_accessor_read_index(&locations, i);
        if (kIndex >= kCount) {
          Fail("Invalid sparse glTF index location");
        }
        indices[kIndex] =
            static_cast<std::uint32_t>(cgltf_accessor_read_index(&values, i));
      }
    }
  }
  if (std::any_of(indices.begin(), indices.end(), [vertex_count](auto index) {
        return index >= vertex_count;
      })) {
    Fail("glTF index exceeds its vertex count");
  }
  return indices;
}

struct Bounds {
  Vec3 lower{INFINITY, INFINITY, INFINITY};
  Vec3 upper{-INFINITY, -INFINITY, -INFINITY};

  void Include(Vec3 point) {
    lower = {
        std::min(lower.x, point.x),
        std::min(lower.y, point.y),
        std::min(lower.z, point.z),
    };
    upper = {
        std::max(upper.x, point.x),
        std::max(upper.y, point.y),
        std::max(upper.z, point.z),
    };
  }

  Vec3 Center() const { return lower * 0.5F + upper * 0.5F; }
};

void AppendPrimitive(GltfModel& model, const cgltf_primitive& primitive,
                     const Mat4& world, std::uint32_t material) {
  if (primitive.type != cgltf_primitive_type_triangles &&
      primitive.type != cgltf_primitive_type_triangle_strip &&
      primitive.type != cgltf_primitive_type_triangle_fan) {
    Fail("glTF scene requires triangle primitives");
  }
  const auto* position =
      cgltf_find_accessor(&primitive, cgltf_attribute_type_position, 0);
  if (!position || position->count == 0 ||
      position->count > UINT32_MAX - model.vertices.size()) {
    Fail("Invalid glTF position accessor");
  }
  const std::size_t kCount = position->count;
  const auto kFirst = static_cast<std::uint32_t>(model.vertices.size());
  model.vertices.resize(kFirst + kCount);
  auto* vertices = model.vertices.data() + kFirst;

  const Vec3 kX{world[0][0], world[0][1], world[0][2]};
  const Vec3 kY{world[1][0], world[1][1], world[1][2]};
  const Vec3 kZ{world[2][0], world[2][1], world[2][2]};
  const Vec3 kTranslation{world[3][0], world[3][1], world[3][2]};
  const Vec3 kNormalX = Cross(kY, kZ);
  const Vec3 kNormalY = Cross(kZ, kX);
  const Vec3 kNormalZ = Cross(kX, kY);
  const bool kMirrored = Dot(kX, kNormalX) < 0;
  const float kSign = kMirrored ? -1.0F : 1.0F;
  Bounds bounds;

  ReadAttribute(
      primitive, cgltf_attribute_type_position, 0, kCount, 3, 3,
      [&](std::size_t i, const float* v, std::size_t) {
        auto p = kX * v[0] + kY * v[1] + kZ * v[2] + kTranslation;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
          Fail("Invalid glTF node transform");
        }
        bounds.Include(p);
        vertices[i].position = p;
      });
  ReadAttribute(
      primitive, cgltf_attribute_type_normal, 0, kCount, 3, 3,
      [&](std::size_t i, const float* v, std::size_t) {
        vertices[i].normal = Normalize(
            (kNormalX * v[0] + kNormalY * v[1] + kNormalZ * v[2]) * kSign);
      });
  ReadAttribute(primitive, cgltf_attribute_type_color, 0, kCount, 3, 4,
                [&](std::size_t i, const float* v, std::size_t components) {
                  std::copy_n(v, components, vertices[i].color.begin());
                });

  const auto kViews = GetTextureViews(primitive.material);
  for (std::size_t slot = 0; slot < kViews.size(); ++slot) {
    const auto* view = kViews[slot];
    if (!view || !view->texture) {
      continue;
    }
    const auto& transform = view->transform;
    const int kSet = view->has_transform && transform.has_texcoord
                         ? transform.texcoord
                         : view->texcoord;
    const float kC = std::cos(transform.rotation);
    const float kS = std::sin(transform.rotation);
    if (!cgltf_find_accessor(&primitive, cgltf_attribute_type_texcoord, kSet)) {
      Fail("glTF material references missing texture coordinates");
    }
    ReadAttribute(primitive, cgltf_attribute_type_texcoord, kSet, kCount, 2, 2,
                  [&](std::size_t i, const float* v, std::size_t) {
                    const float kU = v[0] * transform.scale[0];
                    const float kW = v[1] * transform.scale[1];
                    vertices[i].uv[slot] =
                        view->has_transform
                            ? std::array{
                                  transform.offset[0] + kC * kU - kS * kW,
                                  transform.offset[1] + kS * kU + kC * kW,
                              }
                            : std::array{v[0], v[1]};
                  });
  }
  if (kViews[2] && kViews[2]->texture && !kViews[2]->has_transform &&
      cgltf_find_accessor(&primitive, cgltf_attribute_type_normal, 0)) {
    ReadAttribute(primitive, cgltf_attribute_type_tangent, 0, kCount, 4, 4,
                  [&](std::size_t i, const float* v, std::size_t) {
                    const Vec3 kTangent =
                        Normalize(kX * v[0] + kY * v[1] + kZ * v[2]);
                    vertices[i].tangent = {
                        kTangent.x,
                        kTangent.y,
                        kTangent.z,
                        v[3] * kSign,
                    };
                  });
  }

  const auto kIndices = ReadIndices(primitive, kCount);
  const bool kTriangles = primitive.type == cgltf_primitive_type_triangles;
  if (kIndices.size() < 3 || (kTriangles && kIndices.size() % 3 != 0)) {
    Fail("Invalid glTF triangle count");
  }
  const std::size_t kTriangleCount =
      kTriangles ? kIndices.size() / 3 : kIndices.size() - 2;
  if (kTriangleCount > (UINT32_MAX - model.indices.size()) / 3) {
    Fail("Too many glTF triangles");
  }
  model.draws.push_back({
      static_cast<std::uint32_t>(model.indices.size()),
      static_cast<std::uint32_t>(kTriangleCount * 3),
      material,
      bounds.Center(),
  });
  for (std::size_t i = 0; i < kTriangleCount; ++i) {
    const auto kA = kTriangles ? 3 * i : i;
    const bool kFan = primitive.type == cgltf_primitive_type_triangle_fan;
    const bool kFlip = kMirrored != (!kTriangles && !kFan && i % 2 != 0);
    model.indices.push_back(kFirst + kIndices[kFan ? 0 : kA]);
    model.indices.push_back(kFirst + kIndices[kA + (kFlip ? 2 : 1)]);
    model.indices.push_back(kFirst + kIndices[kA + (kFlip ? 1 : 2)]);
  }
}

ModelMaterial MakeMaterial(const cgltf_material* source) {
  ModelMaterial material;
  if (!source) {
    return material;
  }
  auto& params = material.params;
  const auto& pbr = source->pbr_metallic_roughness;
  std::copy_n(pbr.base_color_factor, 4, params.base_color.begin());
  const float kStrength = source->has_emissive_strength
                              ? source->emissive_strength.emissive_strength
                              : 1;
  for (int i = 0; i < 3; ++i) {
    params.emissive_cutoff[i] = source->emissive_factor[i] * kStrength;
  }
  params.emissive_cutoff[3] = source->alpha_cutoff;
  params.pbr = {
      pbr.metallic_factor,
      pbr.roughness_factor,
      source->normal_texture.scale,
      source->occlusion_texture.scale,
  };
  if (source->normal_texture.texture) {
    params.flags |= kNormalMap;
  }
  if (source->alpha_mode == cgltf_alpha_mode_mask) {
    params.flags |= kAlphaMask;
  }
  if (source->alpha_mode == cgltf_alpha_mode_blend) {
    params.flags |= kAlphaBlend;
  }
  if (source->double_sided) {
    params.flags |= kDoubleSided;
  }
  if (source->unlit) {
    params.flags |= kUnlit;
  }
  return material;
}

}  // namespace

GltfModel LoadGltf(const std::filesystem::path& path) {
  const auto kUtf8 = path.u8string();
  const auto* filename = reinterpret_cast<const char*>(kUtf8.c_str());
  cgltf_data* raw = nullptr;
  cgltf_options options{};
  options.file.read = ReadFile;
  Check(cgltf_parse_file(&options, filename, &raw), "Open glTF scene");
  const std::unique_ptr<cgltf_data, decltype(&cgltf_free)> kData(raw,
                                                                 cgltf_free);
  for (std::size_t i = 0; i < kData->extensions_required_count; ++i) {
    std::string_view extension(kData->extensions_required[i]);
    if (extension != "KHR_mesh_quantization" &&
        extension != "KHR_texture_transform" &&
        extension != "KHR_materials_unlit" &&
        extension != "KHR_materials_emissive_strength") {
      Fail("Unsupported required glTF extension: " + std::string(extension));
    }
  }
  // Reject out-of-range buffer views before cgltf_validate examines index data.
  for (std::size_t i = 0; i < kData->buffer_views_count; ++i) {
    const auto& view = kData->buffer_views[i];
    if (!view.buffer || view.offset > view.buffer->size ||
        view.size > view.buffer->size - view.offset) {
      Fail("Invalid glTF buffer view");
    }
  }
  Check(cgltf_load_buffers(&options, kData.get(), filename),
        "Load glTF buffers");
  Check(cgltf_validate(kData.get()), "Validate glTF scene");

  GltfModel model;
  std::map<const cgltf_material*, std::uint32_t> materials;
  std::map<std::pair<const cgltf_texture*, bool>, std::uint32_t> textures;
  std::vector<const cgltf_node*> nodes;
  const auto* scene = kData->scene ? kData->scene : kData->scenes;
  if (scene) {
    for (std::size_t i = 0; i < scene->nodes_count; ++i) {
      nodes.push_back(scene->nodes[i]);
    }
  } else {
    for (std::size_t i = 0; i < kData->nodes_count; ++i) {
      if (!kData->nodes[i].parent) {
        nodes.push_back(&kData->nodes[i]);
      }
    }
  }
  for (std::size_t n = 0; n < nodes.size(); ++n) {
    const auto* node = nodes[n];
    for (std::size_t i = 0; i < node->children_count; ++i) {
      nodes.push_back(node->children[i]);
    }
    if (!node->mesh) {
      continue;
    }
    std::array<float, 16> matrix;
    cgltf_node_transform_world(node, matrix.data());
    Mat4 world;
    for (std::size_t column = 0; column < world.size(); ++column) {
      std::copy_n(matrix.data() + 4 * column, 4, world[column].begin());
    }
    for (std::size_t i = 0; i < node->mesh->primitives_count; ++i) {
      const auto& primitive = node->mesh->primitives[i];
      const auto* source = primitive.material;
      const auto [kIt, kInserted] = materials.try_emplace(
          source, static_cast<std::uint32_t>(model.materials.size()));
      if (kInserted) {
        auto material = MakeMaterial(source);
        const auto kViews = GetTextureViews(source);
        for (std::size_t slot = 0; slot < kViews.size(); ++slot) {
          if (!kViews[slot] || !kViews[slot]->texture) {
            continue;
          }
          const auto* texture = kViews[slot]->texture;
          const bool kSrgb = slot == 0 || slot == 4;
          const auto [kEntry, kNewTexture] = textures.try_emplace(
              std::pair{texture, kSrgb},
              static_cast<std::uint32_t>(model.textures.size()));
          if (kNewTexture) {
            model.textures.push_back(
                DecodeTexture(*texture, kSrgb, path.parent_path()));
          }
          material.textures[slot] = kEntry->second;
        }
        model.materials.push_back(material);
      }
      AppendPrimitive(model, primitive, world, kIt->second);
    }
  }

  if (model.draws.empty()) {
    Fail("glTF scene contains no triangle meshes");
  }
  Bounds bounds;
  for (const auto& vertex : model.vertices) {
    bounds.Include(vertex.position);
  }
  const Vec3 kCenter = bounds.Center();
  const Vec3 kHalf = bounds.upper * 0.5F - bounds.lower * 0.5F;
  const float kScale = std::max({kHalf.x, kHalf.y, kHalf.z});
  if (!(kScale > 0) || !std::isfinite(1.0F / kScale)) {
    Fail("glTF scene bounds are degenerate");
  }
  model.half_extent = kHalf * (1.0F / kScale);
  for (auto& vertex : model.vertices) {
    vertex.position = (vertex.position - kCenter) * (1.0F / kScale);
  }
  for (auto& draw : model.draws) {
    draw.center = (draw.center - kCenter) * (1.0F / kScale);
  }
  return model;
}

}  // namespace deep_infer
