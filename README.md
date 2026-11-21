# Deep infer

A Vulkan based real-time renderer running super-resolution with TensorRT-RTX.

## Features
- Render a scene and toggle super-resolution in real time.
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
./bin/deep_infer_engine.exe model/cnnps2.trt
./bin/deep_infer_benchmark.exe model/cnnps2.trt image/h320_w640.png --iterations=1000
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
- [imgui](https://github.com/ocornut/imgui)
- [stb](https://github.com/nothings/stb)
