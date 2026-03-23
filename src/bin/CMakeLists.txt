add_executable(run_loop_detector_BRIEF run_loop_detector_BRIEF.cpp)
target_link_libraries(run_loop_detector_BRIEF PRIVATE DLoopDetector)
target_include_directories(run_loop_detector_BRIEF PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(run_loop_detector_BRIEF
                           PRIVATE DLOOPDETECTOR_RESOURCE_DIR="${PROJECT_SOURCE_DIR}/resources")
