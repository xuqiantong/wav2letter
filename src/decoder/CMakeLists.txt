cmake_minimum_required(VERSION 3.5.1)

add_library(
  decoder
  INTERFACE
  )

target_sources(
  decoder
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/PLGenerator.cpp
  )

target_link_libraries(
  decoder
  INTERFACE
  common
  criterion
  module
  runtime
  ${cereal_LIBRARIES}
  )

target_include_directories(
  decoder
  INTERFACE
  ${cereal_INCLUDE_DIRS}
  )
