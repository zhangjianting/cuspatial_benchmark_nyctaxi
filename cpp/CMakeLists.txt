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

set(DEFAULT_BUILD_TYPE "Release")

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

find_library(CUDFTESTUTIL_LIBRARY "libcudftestutil.a"
             HINTS "$ENV{CUDF_ROOT}/lib")

message(STATUS "CUDFTESTUTIL: CUDFTESTUTIL_LIBRARY set to ${CUDFTESTUTIL_LIBRARY}")

message(STATUS "CUDF: CUDF_HOME set to $ENV{CUDF_HOME}")
message(STATUS "CUDF: CUDF_LIBRARY set to ${CUDF_LIBRARY}")
message(STATUS "CUDF: CUDF_INCLUDE set to ${CUDF_INCLUDE}")
message(STATUS "CUDF: THRUST_INCLUDE_DIR set to ${THRUST_INCLUDE_DIR}")
message(STATUS "CUDF: LIBCUDACXX_INCLUDE_DIR set to ${LIBCUDACXX_INCLUDE_DIR}")

add_library(cudf SHARED IMPORTED ${CUDF_LIBRARY})
if (CUDF_INCLUDE AND CUDF_LIBRARY)
    set_target_properties(cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
endif (CUDF_INCLUDE AND CUDF_LIBRARY)

###################################################################################################
# - gdal ------------------------------------------------------------------------------------------

find_package(GDAL REQUIRED)

message(STATUS "GDAL: GDAL_LIBRARIES set to ${GDAL_LIBRARIES}")
message(STATUS "GDAL: GDAL_INCLUDE_DIRS set to ${GDAL_INCLUDE_DIRS}")

if(NOT GDAL_FOUND)
    message(FATAL_ERROR "GDAL not found, please check your settings.")
endif(NOT GDAL_FOUND)

add_library(cudftestutil STATIC IMPORTED ${CUDFTESTUTIL_LIBRARY})

if (CUDFTESTUTIL_LIBRARY)
    set_target_properties(cudftestutil PROPERTIES IMPORTED_LOCATION ${CUDFTESTUTIL_LIBRARY})
endif (CUDFTESTUTIL_LIBRARY)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                    ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gtest_main gdal geos_c pthread cudf cuspatial cudftestutil)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
    set(CUDF_TEST_LIST ${CUDF_TEST_LIST} ${CMAKE_TEST_NAME} CACHE INTERNAL "CUDF_TEST_LIST")
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${GDAL_INCLUDE_DIRS}"
                    "${GTEST_INCLUDE_DIR}"
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
