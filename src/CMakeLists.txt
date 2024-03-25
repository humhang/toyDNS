file(GLOB_RECURSE SOURCE *.cpp *.cxx *.CPP *.CXX *.cc)

add_executable(toyDNS ${SOURCE})
target_include_directories(toyDNS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(toyDNS PRIVATE ${HDF5_INCLUDE_DIRS})

target_compile_definitions(toyDNS PRIVATE ${HDF5_DEFINITIONS})

target_link_libraries(toyDNS Kokkos::all_libs)
target_link_libraries(toyDNS ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})


install(TARGETS toyDNS DESTINATION bin)
