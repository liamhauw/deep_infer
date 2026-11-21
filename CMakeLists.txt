cmake_minimum_required(VERSION 4.2)
project(deep_infer)

add_subdirectory(third_party)
add_subdirectory(engine)
add_subdirectory(benchmark)

set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
install(TARGETS deep_infer_engine DESTINATION ${INSTALL_DIR})
install(TARGETS deep_infer_benchmark DESTINATION ${INSTALL_DIR})