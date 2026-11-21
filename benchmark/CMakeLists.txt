cmake_minimum_required(VERSION 4.2)
project(deep_infer_benchmark)

add_executable(${PROJECT_NAME} main.cc png_io.cc)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE deep_infer_third_party)
