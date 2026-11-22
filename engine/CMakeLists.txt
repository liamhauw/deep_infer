cmake_minimum_required(VERSION 4.2)
project(deep_infer_engine)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
file(GLOB_RECURSE SHADER_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.glsl"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.vert"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.frag"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.task"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mesh"
)

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
foreach(source IN LISTS SHADER_FILES)
  file(RELATIVE_PATH shader "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
  set(output "${SHADER_OUTPUT_DIR}/${shader}.inc")
  cmake_path(GET output PARENT_PATH output_dir)
  add_custom_command(OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
    COMMAND Vulkan::glslc --target-env=vulkan1.3 --target-spv=spv1.5 -O -mfmt=c -MD -MF "${output}.d" "${source}" -o "${output}"
    DEPENDS "${source}"
    DEPFILE "${output}.d"
    VERBATIM
  )
  list(APPEND SHADER_HEADERS "${output}")
endforeach()

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES} ${SHADER_HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${SHADER_OUTPUT_DIR}")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PRIVATE
  VK_USE_PLATFORM_WIN32_KHR
  NOMINMAX
)
target_link_libraries(${PROJECT_NAME} PRIVATE deep_infer_third_party)
