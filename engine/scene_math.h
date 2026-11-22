#pragma once

#include <array>
#include <cmath>

namespace deep_infer {

struct Vec3 {
  float x, y, z;

  Vec3 operator+(Vec3 v) const { return {x + v.x, y + v.y, z + v.z}; }
  Vec3 operator-(Vec3 v) const { return {x - v.x, y - v.y, z - v.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

inline float Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline Vec3 Cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline Vec3 Normalize(Vec3 v) {
  const float kLength = std::hypot(v.x, v.y, v.z);
  return kLength > 0 ? v * (1.0F / kLength) : Vec3{};
}

using Mat4 = std::array<std::array<float, 4>, 4>;  // Column-major.

inline Mat4 Multiply(const Mat4& a, const Mat4& b) {
  Mat4 product{};
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      for (int i = 0; i < 4; ++i) {
        product[column][row] += a[i][row] * b[column][i];
      }
    }
  }
  return product;
}

inline Mat4 LookAt(Vec3 eye, Vec3 target) {
  const Vec3 kZ = Normalize(eye - target);
  const Vec3 kX = Normalize(Cross({0, 1, 0}, kZ));
  const Vec3 kY = Cross(kZ, kX);
  return {
      {
          {kX.x, kY.x, kZ.x, 0},
          {kX.y, kY.y, kZ.y, 0},
          {kX.z, kY.z, kZ.z, 0},
          {-Dot(kX, eye), -Dot(kY, eye), -Dot(kZ, eye), 1},
      },
  };
}

}  // namespace deep_infer
