# Add configuration for standalone programs
add_executable(run_loop_detector_BRIEF run_loop_detector_BRIEF.cpp)
add_dependencies(run_loop_detector_BRIEF DLib_dep DBoW2_dep)

target_link_libraries(run_loop_detector_BRIEF ${OpenCV_LIBS} ${DLib_LIBS} ${DBoW2_LIBS})
target_include_directories(run_loop_detector_BRIEF PUBLIC ${DLoopDetector_INCLUDE_DIRS} ${DLib_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS})
