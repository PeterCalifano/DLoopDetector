cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(DLoopDetector VERSION 0.1.0 LANGUAGES CXX)

set(BUILD_AS_MAIN_PROJECT ON)
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(BUILD_AS_MAIN_PROJECT OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(OpenCV 4.2 REQUIRED)

set(DLOOP_DBOW2_TARGET "")

if(TARGET DBoW2)
  set(DLOOP_DBOW2_TARGET DBoW2)
elseif(TARGET DBoW2::DBoW2)
  set(DLOOP_DBOW2_TARGET DBoW2::DBoW2)
else()
  find_package(DBoW2 REQUIRED)
  if(TARGET DBoW2::DBoW2)
    set(DLOOP_DBOW2_TARGET DBoW2::DBoW2)
  elseif(TARGET DBoW2)
    set(DLOOP_DBOW2_TARGET DBoW2)
  endif()
endif()

if(NOT DLOOP_DBOW2_TARGET)
  message(FATAL_ERROR "Failed to resolve the DBoW2 target for DLoopDetector.")
endif()

add_library(${PROJECT_NAME} INTERFACE)
if(NOT TARGET ${PROJECT_NAME}::${PROJECT_NAME})
  add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()
target_include_directories(${PROJECT_NAME} INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(${PROJECT_NAME} INTERFACE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} INTERFACE
                      ${OpenCV_LIBS}
                      ${DLOOP_DBOW2_TARGET})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

if(BUILD_AS_MAIN_PROJECT)
  set(EXPORT_TARGET_DEPS OpenCV DBoW2)

  install(TARGETS ${PROJECT_NAME}
          EXPORT ${PROJECT_NAME}
          INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY include/DLoopDetector DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/licenses)

  install(EXPORT ${PROJECT_NAME}
          FILE ${PROJECT_NAME}Target.cmake
          NAMESPACE ${PROJECT_NAME}::
          DESTINATION lib/cmake/${PROJECT_NAME})

  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion)

  configure_package_config_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/src/DLoopDetectorConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})

  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
      DESTINATION lib/cmake/${PROJECT_NAME})

  export(EXPORT ${PROJECT_NAME}
         FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Target.cmake")
endif()
