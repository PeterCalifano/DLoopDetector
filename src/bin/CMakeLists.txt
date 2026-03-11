### Add configuration for standalone programs

# PROGRAM: run_loop_detector_BRIEF 
add_executable(run_loop_detector_BRIEF run_loop_detector_BRIEF.cpp)
target_link_libraries(run_loop_detector_BRIEF PRIVATE ${OpenCV_LIBS} DLib::DLib DBoW2)
target_include_directories(run_loop_detector_BRIEF PRIVATE ${DLoopDetector_INCLUDE_DIRS})
