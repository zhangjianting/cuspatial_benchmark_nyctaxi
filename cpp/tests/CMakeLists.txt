#=============================================================================
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUSPATIAL_TESTS LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()


###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread cuspatial cudf
                        cudftestutil cudart cuda "${ARROW_LIB}" ${ZLIB_LIBRARIES}
                        nvrtc GDAL::GDAL geos_c)
    if(USE_NVTX)
        target_link_libraries(${CMAKE_TEST_NAME} ${NVTX_LIBRARY})
    endif(USE_NVTX)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${THRUST_INCLUDE_DIR}"
                    "${LIBCUDACXX_INCLUDE_DIR}")

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${GDAL_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}"
                    "${CUDF_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GDAL_LIBRARIES}"
                 "${CONDA_LINK_DIRS}"
                 "${GTEST_LIBRARY_DIR}"
                 "${CUDF_LIBRARY}"
                 "${CUSPATIAL_LIBRARY}")

set(POINT_QUADTREE_TOY_DEBUG_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/indexing/point_quadtree_toy_debug.cu")  
ConfigureTest(POINT_QUADTREE_TOY_DEBUG "${POINT_QUADTREE_TOY_DEBUG_SRC}")

set(PIP_JOIN_TOY_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_toy_test.cu"  
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cpp")
ConfigureTest(PIP_JOIN_TOY_TEST "${PIP_JOIN_TOY_TEST_SRC}")

set(PIP_JOIN_TOY_VERIFY_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_toy_verify.cpp"  
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_geo_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cu")  
ConfigureTest(PIP_JOIN_TOY_VERIFY "${PIP_JOIN_TOY_VERIFY_SRC}")


set(PIP_JOIN_NYCTAXI_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_nyctaxi_test.cu"  
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cpp")
ConfigureTest(PIP_JOIN_NYCTAXI_TEST "${PIP_JOIN_NYCTAXI_TEST_SRC}")

set(PIP_JOIN_NYCTAXI_VERIFY_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_nyctaxi_verify.cpp"  
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_geo_utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/join/spatial_join_test_utility.cu")  
ConfigureTest(PIP_JOIN_NYCTAXI_VERIFY "${PIP_JOIN_NYCTAXI_VERIFY_SRC}")
