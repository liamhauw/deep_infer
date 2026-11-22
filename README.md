# Deep infer

A Vulkan based real-time renderer running super-resolution with TensorRT-RTX.

## Features
- Load static glTF 2.0 scenes (`.gltf` or `.glb`) and toggle super-resolution in real time.
- Render metallic-roughness PBR materials with normal mapping, PCF shadows, HDR tone mapping, and alpha masking or blending.
- Run TensorRT-RTX image super-resolution with PNG images or raw FP32 tensors.
- Measure GPU inference throughput and latency with CUDA events.

## Prerequisites
The project targets Windows with an RTX GPU and requires the following dependencies.
- CMake
- Ninja
- Vulkan SDK
- CUDA
- TensorRT-RTX
- LLVM with the Visual Studio

These dependencies can be installed manually or through package managers.

On Windows, [Scoop](https://scoop.sh/) can install some prerequisites using the following command. Install TensorRT-RTX, Visual Studio, and the Vulkan SDK separately.
```shell
scoop install cmake ninja llvm cuda
```

## Quick start
Replace `path_to_TensorRT-RTX` with the TensorRT-RTX SDK root.
```shell
git clone --recurse-submodules https://github.com/liamhauw/deep_infer.git
cd deep_infer
cmake --preset Base "-DTRTRTX_INSTALL_DIR:PATH=path_to_TensorRT-RTX"
cmake --build --preset=Release
cmake --install build --config Release
./bin/deep_infer_engine.exe
```

## Super resolution benchmark

Benchmarked on Windows with an NVIDIA GeForce RTX 4070 Ti (WDDM), NVIDIA driver 621.24, CUDA UMD 13.5, and TensorRT-RTX 1.7.1.116. Each run used 50 warmup iterations and 1,000 timed iterations measured with CUDA events.

| Input (NCHW, FP32) | Output (NCHW, FP32) | GPU total (s) | Throughput (images/s) | Mean (ms) | Min (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Output range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1x3x1080x1920` | `1x3x2160x3840` | 13.388 | 74.69 | 13.388 | 12.967 | 13.184 | 14.325 | 14.530 | 15.449 | [-0.0980896, 1.16176] |
| `1x3x320x640` | `1x3x640x1280` | 0.795 | 1258.20 | 0.795 | 0.751 | 0.778 | 0.794 | 1.311 | 1.500 | [-0.124513, 1.19537] |

Run these commands from the repository root. The first command uses a local 1920 × 1080 PNG at `resource/image/h1080_1920.png`. The second uses the default engine and input image.

```powershell
.\bin\deep_infer_benchmark.exe .\resource\model\cnnps_feat16_conv8_scale2_1x3x1080x1920_static_fp32_op20.trt .\resource\image\h1080_1920.png --iterations=1000
.\bin\deep_infer_benchmark.exe --iterations=1000
```

## Contributions
### Update submodule
```shell
git pull --recurse-submodules
git submodule update --remote
```

### Debug
Install VSCode with clangd and CodeLLDB extensions.
Run the configure command from Quick start first. Click Run and Debug in the sidebar, and select Debug. Click Start Debugging or press F5.

## Third party
- [cgltf](https://github.com/jkuhlmann/cgltf)
- [imgui](https://github.com/ocornut/imgui)
- [stb](https://github.com/nothings/stb)
