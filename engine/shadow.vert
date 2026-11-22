#version 450
#extension GL_GOOGLE_include_directive : require
#include "gltf_frame.glslh"

layout(location = 0) in vec3 position;
layout(location = 3) in vec4 vertexColor;
layout(location = 4) in vec2 baseUV;
layout(location = 0) out vec2 uv;
layout(location = 1) out float vertexAlpha;

void main() {
  uv = baseUV;
  vertexAlpha = vertexColor.a;
  gl_Position = frame.lightProjection * vec4(position, 1.0);
}
