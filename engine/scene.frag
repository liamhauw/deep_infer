#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(push_constant) uniform Scene { float time; } scene;

void main() {
  vec2 p = vec2((uv.x - 0.5) * 2.0, 0.5 - uv.y);
  vec3 rgb = mix(vec3(0.10, 0.18, 0.32), vec3(0.025, 0.035, 0.08), uv.y);
  float sun = length(p - vec2(0.33, 0.19));
  rgb += vec3(0.8, 0.25, 0.1) * exp(-5.0 * sun);
  if (sun < 0.17) {
    rgb = mix(vec3(0.9, 0.3, 0.16), vec3(1.0, 0.84, 0.42), uv.y);
  }

  float ridge = -0.04 + 0.09 * sin(10.0 * p.x) + 0.055 * sin(21.0 * p.x);
  if (p.y < ridge) rgb = vec3(0.075, 0.095, 0.17);
  if (p.y < -0.13) {
    float z = 1.0 / (-p.y - 0.10);
    vec2 world = vec2(p.x * z * 0.65, z + scene.time * 0.7);
    vec2 grid = abs(fract(world) - 0.5) / max(fwidth(world), vec2(0.0001));
    float line = 1.0 - smoothstep(0.1, 0.75, min(grid.x, grid.y));
    rgb = vec3(0.055, 0.07, 0.13) + vec3(0.04, 0.43, 0.65) * line * exp(-z * 0.065);
  }

  vec2 center = vec2(0.39 * sin(scene.time * 0.7) - 0.27, -0.075);
  vec2 q = p - center;
  float angle = scene.time * 0.8;
  q = mat2(cos(angle), -sin(angle), sin(angle), cos(angle)) * q;
  float diamond = abs(q.x) + abs(q.y);
  rgb += vec3(0.06, 0.32, 0.45) * exp(-11.0 * diamond);
  if (diamond < 0.17) {
    rgb = mix(vec3(0.025, 0.25, 0.38), vec3(0.26, 0.94, 1.0), 0.5 + 2.5 * q.y);
    float stripes = step(0.5, fract((q.x + q.y) * 50.0));
    rgb *= mix(0.77, 1.0, stripes);
    if (diamond > 0.16) rgb = vec3(0.61, 0.98, 1.0);
  }
  color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
