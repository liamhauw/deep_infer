#version 450

layout(location = 0) out vec4 color;
layout(set = 0, binding = 0) uniform sampler2D hdrScene;

vec3 linearToSrgb(vec3 linearColor) {
  return mix(12.92 * linearColor, 1.055 * pow(linearColor, vec3(1.0 / 2.4)) - 0.055,
             greaterThan(linearColor, vec3(0.0031308)));
}

void main() {
  vec3 linearColor = max(texelFetch(hdrScene, ivec2(gl_FragCoord.xy), 0).rgb, 0.0);
  // Luminance-based Reinhard shoulder preserves the hue of bright surfaces.
  float luminance = dot(linearColor, vec3(0.2126, 0.7152, 0.0722));
  vec3 mapped = linearColor / (1.0 + luminance);
  color = vec4(linearToSrgb(clamp(mapped, 0.0, 1.0)), 1.0);
}
