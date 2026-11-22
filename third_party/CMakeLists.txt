cmake_minimum_required(VERSION 4.2)
project(deep_infer_third_party)

# Vulkan
find_package(Vulkan REQUIRED GLOBAL COMPONENTS glslc)

# CUDA
find_package(CUDAToolkit REQUIRED GLOBAL)

# TensorRT-RTX
set(TRTRTX_INSTALL_DIR "" CACHE PATH "TensorRT-RTX SDK root")
if(NOT IS_DIRECTORY "${TRTRTX_INSTALL_DIR}")
  message(FATAL_ERROR "Set TRTRTX_INSTALL_DIR to the TensorRT-RTX SDK root")
endif()

set(TRTRTX_INCLUDE_DIR "${TRTRTX_INSTALL_DIR}/include")
set(TRTRTX_VERSION_HEADER "${TRTRTX_INCLUDE_DIR}/NvInferVersion.h")
if(NOT EXISTS "${TRTRTX_VERSION_HEADER}")
  message(FATAL_ERROR "TensorRT-RTX version header not found: ${TRTRTX_VERSION_HEADER}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TRTRTX_VERSION_HEADER}")

foreach(version_part IN ITEMS MAJOR MINOR)
  file(STRINGS "${TRTRTX_VERSION_HEADER}" version_line
    LIMIT_COUNT 1
    REGEX "^[ \t]*#[ \t]*define[ \t]+TRT_${version_part}_RTX[ \t]+([0-9]+)([ \t]|$)"
  )
  if(NOT version_line)
    message(FATAL_ERROR "Cannot read TRT_${version_part}_RTX from ${TRTRTX_VERSION_HEADER}")
  endif()
  set(TRTRTX_VERSION_${version_part} "${CMAKE_MATCH_1}")
endforeach()
set(TRTRTX_LIB_SUFFIX "${TRTRTX_VERSION_MAJOR}_${TRTRTX_VERSION_MINOR}")
set(TRTRTX_RUNTIME_LIBRARY "${TRTRTX_INSTALL_DIR}/lib/tensorrt_rtx_${TRTRTX_LIB_SUFFIX}.lib")
set(TRTRTX_ONNX_PARSER_LIBRARY "${TRTRTX_INSTALL_DIR}/lib/tensorrt_onnxparser_rtx_${TRTRTX_LIB_SUFFIX}.lib")
message(STATUS "TensorRT-RTX include directory = ${TRTRTX_INCLUDE_DIR}")
message(STATUS "TensorRT-RTX runtime library = ${TRTRTX_RUNTIME_LIBRARY}")
message(STATUS "TensorRT-RTX ONNX parser library = ${TRTRTX_ONNX_PARSER_LIBRARY}")

# cgltf
add_library(cgltf INTERFACE)
target_include_directories(cgltf SYSTEM INTERFACE cgltf)

# imgui
add_library(imgui)
target_sources(imgui PRIVATE
  imgui/backends/imgui_impl_win32.cpp
  imgui/backends/imgui_impl_vulkan.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/imgui.cpp
)
target_include_directories(imgui PUBLIC
  imgui
  imgui/backends
)
target_compile_features(imgui PRIVATE cxx_std_20)
target_link_libraries(imgui PRIVATE
  Vulkan::Vulkan
  dwmapi
)

# stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE third_party.cc)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  "${TRTRTX_INCLUDE_DIR}"
)
target_link_libraries(${PROJECT_NAME} PUBLIC
  Vulkan::Vulkan
  CUDA::cudart
  "${TRTRTX_RUNTIME_LIBRARY}"
  "${TRTRTX_ONNX_PARSER_LIBRARY}"
  cgltf
  imgui
  stb
)
