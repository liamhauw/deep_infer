#version 450
#extension GL_GOOGLE_include_directive : require
#include "gltf_frame.glslh"
#include "gltf_material.glslh"

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec4 worldTangent;
layout(location = 3) in vec4 tint;
layout(location = 4) in vec2 uv[5];
layout(location = 0) out vec4 hdr;

layout(set = 0, binding = 1) uniform sampler2DShadow shadowMap;
layout(set = 1, binding = 0) uniform sampler2D baseColorMap;
layout(set = 1, binding = 1) uniform sampler2D metallicRoughnessMap;
layout(set = 1, binding = 2) uniform sampler2D tangentNormalMap;
layout(set = 1, binding = 3) uniform sampler2D occlusionMap;
layout(set = 1, binding = 4) uniform sampler2D emissiveMap;

const float pi = 3.14159265359;

float shadowVisibility(vec3 n) {
  float grazing = 1.0 - clamp(dot(n, frame.light.xyz), 0.0, 1.0);
  vec4 projected = frame.lightProjection * vec4(worldPosition + n * (0.001 + 0.004 * grazing), 1.0);
  vec3 p = projected.xyz / projected.w;
  vec2 st = p.xy * 0.5 + 0.5;
  if (p.z < 0.0 || p.z > 1.0) return 1.0;
  vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
  float sum = 0.0;
  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      sum += texture(shadowMap, vec3(st + vec2(x, y) * texel, p.z - 0.00035));
    }
  }
  return sum / 9.0;
}

vec3 surfaceNormal(vec3 geometric) {
  if (!hasFlag(normalMap)) return geometric;

  vec3 t = worldTangent.xyz;
  float handedness = worldTangent.w;
  if (abs(handedness) < 0.5) {
    vec3 dx = dFdx(worldPosition), dy = dFdy(worldPosition);
    vec2 ux = dFdx(uv[2]), uy = dFdy(uv[2]);
    float determinant = ux.x * uy.y - uy.x * ux.y;
    if (abs(determinant) < 1e-12) return geometric;
    t = (uy.y * dx - ux.y * dy) * sign(determinant);
    vec3 b = (ux.x * dy - uy.x * dx) * sign(determinant);
    handedness = dot(cross(geometric, t), b) < 0.0 ? -1.0 : 1.0;
  }
  t -= geometric * dot(geometric, t);
  if (dot(t, t) < 1e-18) return geometric;
  t = normalize(t);
  vec3 b = cross(geometric, t) * handedness;
  vec3 mapped = texture(tangentNormalMap, uv[2]).xyz * 2.0 - 1.0;
  mapped.xy *= material.pbr.z;
  return normalize(mat3(t, b, geometric) * mapped);
}

vec3 fresnel(vec3 f0, float cosTheta) {
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// Cook-Torrance with GGX distribution and height-correlated Smith visibility.
vec3 directLight(vec3 n, vec3 v, vec3 l, vec3 base, vec3 f0,
                 float metal, float roughness, vec3 irradiance) {
  vec3 h = normalize(v + l);
  float nl = max(dot(n, l), 0.0);
  float nv = max(dot(n, v), 0.0001);
  float nh = max(dot(n, h), 0.0);
  float vh = max(dot(v, h), 0.0);
  float a2 = pow(roughness, 4.0);
  float d = (nh * a2 - nh) * nh + 1.0;
  float distribution = a2 / (pi * d * d);
  float visibility = 0.5 / max(
      nl * sqrt(nv * nv * (1.0 - a2) + a2) +
      nv * sqrt(nl * nl * (1.0 - a2) + a2), 0.0001);
  vec3 f = fresnel(f0, vh);
  vec3 diffuse = (1.0 - f) * (1.0 - metal) * base / pi;
  return (diffuse + distribution * visibility * f) * irradiance * nl;
}

void main() {
  float facing = gl_FrontFacing ? 1.0 : -1.0;
  vec3 geometric = dot(worldNormal, worldNormal) > 0.0
      ? normalize(worldNormal)
      : facing * normalize(-cross(dFdx(worldPosition), dFdy(worldPosition)));
  vec3 n = facing * surfaceNormal(geometric);
  vec4 base = texture(baseColorMap, uv[0]) * material.baseColor * tint;
  vec2 mr = texture(metallicRoughnessMap, uv[1]).gb;
  float ao = mix(1.0, texture(occlusionMap, uv[3]).r, material.pbr.w);
  vec3 emissive = texture(emissiveMap, uv[4]).rgb * material.emissiveCutoff.rgb;
  if (!gl_FrontFacing && !hasFlag(doubleSided)) discard;
  if (hasFlag(alphaMask) && base.a < material.emissiveCutoff.a) discard;
  float coverage = hasFlag(alphaBlend) ? base.a : 1.0;
  if (hasFlag(unlit)) {
    hdr = vec4(base.rgb, coverage);
    return;
  }

  float metal = clamp(mr.y * material.pbr.x, 0.0, 1.0);
  float roughness = clamp(mr.x * material.pbr.y, 0.045, 1.0);
  vec3 f0 = mix(vec3(0.04), base.rgb, metal);
  vec3 v = normalize(frame.eye.xyz - worldPosition);
  vec3 key = directLight(n, v, frame.light.xyz, base.rgb, f0, metal,
                        roughness, vec3(4.0, 3.75, 3.4));
  vec3 fill = directLight(n, v, normalize(vec3(0.65, 0.35, -0.7)), base.rgb,
                         f0, metal, roughness, vec3(0.65, 1.0, 1.6));

  // Analytic studio lighting gives metals a broad reflection between key highlights.
  vec3 reflection = reflect(-v, n);
  float sky = n.y * 0.5 + 0.5;
  float softbox = pow(max(dot(reflection, normalize(vec3(0.2, 0.8, 0.6))), 0.0),
                      mix(24.0, 2.0, roughness));
  vec3 ambient = base.rgb * (1.0 - metal) *
                 mix(vec3(0.12, 0.10, 0.10), vec3(0.29, 0.36, 0.46), sky);
  ambient += (0.16 + 0.70 * softbox) * fresnel(f0, max(dot(n, v), 0.0));
  hdr = vec4(key * shadowVisibility(facing * geometric) + fill + ambient * ao + emissive,
             coverage);
}
