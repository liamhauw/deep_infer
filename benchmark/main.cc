#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "cuda_check.h"
#include "png_io.h"

namespace {

namespace fs = std::filesystem;

class Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TensorRT-RTX] %s\n", message);
    }
  }
};

using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;

DeviceBuffer AllocateDevice(std::size_t bytes) {
  void* data = nullptr;
  CheckCuda(cudaMalloc(&data, bytes), "cudaMalloc");
  return {data, cudaFree};
}

class CudaStream {
 public:
  CudaStream() {
    CheckCuda(cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
  }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  ~CudaStream() {
    cudaStreamSynchronize(handle_);
    cudaStreamDestroy(handle_);
  }

  cudaStream_t Get() const { return handle_; }

 private:
  cudaStream_t handle_{};
};

class CudaEvent {
 public:
  CudaEvent() { CheckCuda(cudaEventCreate(&handle_), "cudaEventCreate"); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  ~CudaEvent() { cudaEventDestroy(handle_); }

  void Record(cudaStream_t stream) const {
    CheckCuda(cudaEventRecord(handle_, stream), "cudaEventRecord");
  }

  float ElapsedSince(const CudaEvent& start) const {
    float milliseconds = 0.0F;
    CheckCuda(cudaEventElapsedTime(&milliseconds, start.handle_, handle_),
              "cudaEventElapsedTime");
    return milliseconds;
  }

 private:
  cudaEvent_t handle_{};
};

void EnqueueInference(nvinfer1::IExecutionContext& context,
                      cudaStream_t stream) {
  if (!context.enqueueV3(stream)) {
    throw std::runtime_error("TensorRT-RTX enqueueV3 failed");
  }
}

void BenchmarkInference(nvinfer1::IExecutionContext& context,
                        cudaStream_t stream, int iterations) {
  constexpr int kWarmupIterations = 50;
  const auto kCount = static_cast<std::size_t>(iterations);
  std::vector<CudaEvent> events(kCount + 1);
  std::vector<float> latency_ms(kCount);

  std::cout << std::format("GPU benchmark: {} iterations, {} warmup\n",
                           iterations, kWarmupIterations)
            << std::flush;
  for (int i = 0; i < kWarmupIterations; ++i) {
    EnqueueInference(context, stream);
  }
  CheckCuda(cudaStreamSynchronize(stream), "Synchronize warmup");

  events.front().Record(stream);
  for (std::size_t i = 0; i < kCount; ++i) {
    EnqueueInference(context, stream);
    events[i + 1].Record(stream);
  }
  CheckCuda(cudaStreamSynchronize(stream), "Synchronize benchmark");

  const auto kTotalMs =
      static_cast<double>(events.back().ElapsedSince(events.front()));
  for (std::size_t i = 0; i < kCount; ++i) {
    latency_ms[i] = events[i + 1].ElapsedSince(events[i]);
  }
  std::ranges::sort(latency_ms);
  const auto kPercentile = [&latency_ms, kCount](double percentile) {
    const auto kRank = static_cast<std::size_t>(
        std::ceil(percentile * static_cast<double>(kCount)));
    return latency_ms[kRank - 1];
  };
  std::cout << std::format(
      "GPU total: {:.3f} s | throughput: {:.2f} images/s\n"
      "GPU latency (ms): mean {:.3f} | min {:.3f} | p50 {:.3f} | "
      "p95 {:.3f} | p99 {:.3f} | max {:.3f}\n",
      kTotalMs / 1000.0, 1000.0 * iterations / kTotalMs, kTotalMs / iterations,
      latency_ms.front(), kPercentile(0.50), kPercentile(0.95),
      kPercentile(0.99), latency_ms.back());
}

template <typename T>
std::vector<T> ReadBinary(const fs::path& path,
                          std::size_t expected_elements = 0) {
  std::ifstream file(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  const auto kSize = static_cast<std::streamoff>(file.tellg());
  if (kSize <= 0 || kSize % sizeof(T) != 0) {
    throw std::runtime_error("Cannot read binary file: " + path.string());
  }
  const auto kElements = static_cast<std::size_t>(kSize) / sizeof(T);
  if (expected_elements != 0 && kElements != expected_elements) {
    throw std::runtime_error(
        "Incorrect input size for " + path.string() + ", expected " +
        std::to_string(expected_elements * sizeof(T)) + " bytes");
  }
  std::vector<T> data(kElements);
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(data.data()), kSize)) {
    throw std::runtime_error("Failed to read binary file: " + path.string());
  }
  return data;
}

struct Tensor {
  std::string name;
  std::string shape;
  nvinfer1::Dims dims;
  std::size_t elements;

  std::size_t Bytes() const { return elements * sizeof(float); }
};

Tensor GetTensor(const nvinfer1::ICudaEngine& engine,
                 nvinfer1::TensorIOMode mode) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char* name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) != mode) {
      continue;
    }
    const auto kDims = engine.getTensorShape(name);
    if (kDims.nbDims != 4 ||
        engine.getTensorDataType(name) != nvinfer1::DataType::kFLOAT ||
        engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR ||
        engine.getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
      throw std::runtime_error(
          std::string(name) +
          " must be a linear FP32 tensor on the GPU with four dimensions");
    }
    Tensor tensor{name, "", kDims, 1};
    for (int axis = 0; axis < kDims.nbDims; ++axis) {
      const auto kExtent = kDims.d[axis];
      if (kExtent <= 0 || static_cast<std::uint64_t>(kExtent) >
                              std::numeric_limits<std::size_t>::max() /
                                  sizeof(float) / tensor.elements) {
        throw std::runtime_error(tensor.name +
                                 " must have a fixed, valid shape");
      }
      if (axis > 0) {
        tensor.shape += 'x';
      }
      tensor.shape += std::to_string(kExtent);
      tensor.elements *= static_cast<std::size_t>(kExtent);
    }
    return tensor;
  }
  throw std::runtime_error(mode == nvinfer1::TensorIOMode::kINPUT
                               ? "Engine has no input tensor"
                               : "Engine has no output tensor");
}

png_io::ImageSize GetPngSize(const Tensor& tensor) {
  const auto& dims = tensor.dims.d;
  if (dims[0] != 1 || dims[1] != 3 ||
      dims[2] > std::numeric_limits<int>::max() ||
      dims[3] > std::numeric_limits<int>::max()) {
    throw std::runtime_error(tensor.name +
                             " must be a 1x3xHxW RGB tensor for PNG I/O");
  }
  return {static_cast<int>(dims[3]), static_cast<int>(dims[2])};
}

void RunInference(const fs::path& engine_path, const fs::path& input_path,
                  const fs::path& output_path, int iterations) {
  const auto kPlan = ReadBinary<char>(engine_path);
  Logger logger;
  auto runtime =
      std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("Failed to create TensorRT-RTX runtime");
  }
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(kPlan.data(), kPlan.size()));
  if (!engine) {
    throw std::runtime_error("Failed to deserialize engine: " +
                             engine_path.string());
  }
  if (engine->getNbIOTensors() != 2) {
    throw std::runtime_error("Expected one input tensor and one output tensor");
  }
  const auto kInput = GetTensor(*engine, nvinfer1::TensorIOMode::kINPUT);
  const auto kOutput = GetTensor(*engine, nvinfer1::TensorIOMode::kOUTPUT);
  std::cout << "Engine: " << engine_path.string() << '\n'
            << "Input: " << kInput.name << ' ' << kInput.shape << " FP32\n"
            << "Output: " << kOutput.name << ' ' << kOutput.shape << " FP32\n";

  std::vector<float> input_data;
  if (input_path.empty()) {
    input_data.assign(kInput.elements, 0.5F);
  } else if (png_io::IsPngPath(input_path)) {
    input_data = png_io::ReadPng(input_path, GetPngSize(kInput));
  } else {
    input_data = ReadBinary<float>(input_path, kInput.elements);
  }
  std::vector<float> output_data(kOutput.elements);
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());
  if (!context) {
    throw std::runtime_error("Failed to create TensorRT-RTX execution context");
  }

  const auto kDeviceInput = AllocateDevice(kInput.Bytes());
  const auto kDeviceOutput = AllocateDevice(kOutput.Bytes());
  CudaStream stream;
  if (!context->setInputTensorAddress(kInput.name.c_str(),
                                      kDeviceInput.get()) ||
      !context->setOutputTensorAddress(kOutput.name.c_str(),
                                       kDeviceOutput.get())) {
    throw std::runtime_error("Failed to bind TensorRT-RTX I/O buffers");
  }
  CheckCuda(
      cudaMemcpyAsync(kDeviceInput.get(), input_data.data(), kInput.Bytes(),
                      cudaMemcpyHostToDevice, stream.Get()),
      "Copy input to GPU");
  if (iterations > 0) {
    BenchmarkInference(*context, stream.Get(), iterations);
  } else {
    EnqueueInference(*context, stream.Get());
  }
  CheckCuda(
      cudaMemcpyAsync(output_data.data(), kDeviceOutput.get(), kOutput.Bytes(),
                      cudaMemcpyDeviceToHost, stream.Get()),
      "Copy output to CPU");
  CheckCuda(cudaStreamSynchronize(stream.Get()),
            "Synchronize inference stream");

  if (!std::ranges::all_of(output_data,
                           [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("Inference produced non-finite output");
  }
  const auto [kMinimum, kMaximum] = std::ranges::minmax_element(output_data);
  if (png_io::IsPngPath(output_path)) {
    png_io::WritePng(output_path, GetPngSize(kOutput), output_data);
  } else if (!output_path.empty()) {
    std::ofstream file(output_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(output_data.data()),
               static_cast<std::streamsize>(kOutput.Bytes()));
    file.close();
    if (!file) {
      throw std::runtime_error("Failed to write output: " +
                               output_path.string());
    }
  }
  if (!output_path.empty()) {
    std::cout << "Saved output: " << output_path.string() << '\n';
  }
  std::cout << "Inference complete. Output range: [" << *kMinimum << ", "
            << *kMaximum << "]\n"
            << std::flush;
}

}  // namespace

int main(int argc, char** argv) try {
  constexpr std::string_view kIterationsFlag = "--iterations=";
  int iterations = 0;
  if (argc > 1 &&
      std::string_view(argv[argc - 1]).starts_with(kIterationsFlag)) {
    const std::string_view kValue =
        std::string_view(argv[--argc]).substr(kIterationsFlag.size());
    const auto kResult = std::from_chars(
        kValue.data(), kValue.data() + kValue.size(), iterations);
    if (kResult.ec != std::errc{} ||
        kResult.ptr != kValue.data() + kValue.size() || iterations <= 0) {
      throw std::runtime_error("--iterations requires a positive integer");
    }
  }
  if (argc > 4) {
    std::fputs(
        "Usage: cpp_startup [engine.trt [input.png|input.f32 "
        "[output.png|output.f32]]] [--iterations=N]\n",
        stderr);
    return EXIT_FAILURE;
  }
  RunInference(argc > 1 ? fs::path(argv[1]) : fs::path("model/cnnps2.trt"),
               argc > 2 ? fs::path(argv[2]) : fs::path{"image/h320_w640.png"},
               argc > 3 ? fs::path(argv[3]) : fs::path{}, iterations);
  return EXIT_SUCCESS;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fputs("Unexpected inference error\n", stderr);
  return EXIT_FAILURE;
}
