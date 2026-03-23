# Add library components
add_subdirectory(core)
add_subdirectory(specializations)

if(BUILD_AS_MAIN_PROJECT)
  add_subdirectory(bin)
endif()