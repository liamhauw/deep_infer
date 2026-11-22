#version 450
#extension GL_GOOGLE_include_directive : require
#include "gltf_material.glslh"

layout(location = 0) in vec2 uv;
layout(location = 1) in float vertexAlpha;
layout(set = 1, binding = 0) uniform sampler2D baseColorMap;

void main() {
  if (!gl_FrontFacing && !hasFlag(doubleSided)) discard;
  if (hasFlag(alphaMask) &&
      texture(baseColorMap, uv).a * material.baseColor.a * vertexAlpha <
          material.emissiveCutoff.a) discard;
}
