#version 450
#extension GL_GOOGLE_include_directive : require
#include "gltf_frame.glslh"

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec4 vertexColor;
layout(location = 4) in vec2 texCoord[5];

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec4 worldTangent;
layout(location = 3) out vec4 tint;
layout(location = 4) out vec2 uv[5];

void main() {
  worldPosition = position;
  worldNormal = normal;
  worldTangent = tangent;
  tint = vertexColor;
  uv = texCoord;
  gl_Position = frame.viewProjection * vec4(position, 1.0);
}
