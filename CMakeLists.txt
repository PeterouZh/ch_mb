cmake_minimum_required(VERSION 2.8.4)

project(CMSS)

add_definitions(-std=c++11)

#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")

# Opencv
set(OpenCV_DIR "/home/shhs/env/opencv3_2/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# enable darknet cuda cudnn
#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
option(DGPU "darknet use cuda" OFF)
option(DCUDNN "darknet use cudnn" OFF)
option(DOPENCV "darknet use opencv" ON)
option(DOPENMP "darknet use openmp" ON)
#add_definitions(-DGPU)
#add_definitions(-DCUDNN)
#add_definitions(-DOPENMP)
# darknet.h
include_directories(.)
add_library(darkneta STATIC IMPORTED)
set_property(TARGET darkneta PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/darknet/libdarknet.a)
add_library(darknetso SHARED IMPORTED)
set_property(TARGET darknetso PROPERTY IMPORTED_LOCATION /home/shhs/usr/face/ch_mb/darknet/libdarknet.so)


# dlib cuda enable
option(DLIB_USE_CUDA "using cuda" OFF)
option(TEST_TIME "show test timing" OFF)
add_subdirectory("./dlib")

# make install dir
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR})

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g -ggdb")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

set(CMAKE_CXX_FLAGS "-Wno-unused-result ${CMAKE_CXX_FLAGS}")

# Build face_recogniton lib
message("Build face recognition lib")
add_library(face_recognition SHARED ./CMSS_FaceRecognize.cpp)
target_link_libraries(face_recognition ${OpenCV_LIBS} dlib::dlib darkneta)

if (TEST_TIME)
  target_compile_definitions(face_recognition PRIVATE TIMING)
endif()
install(FILES ./CMSS_FaceRecognize.h DESTINATION include)
# install(TARGETS face_recognition LIBRARY DESTINATION lib
#   PERMISSIONS WORLD_EXECUTE OWNER_EXECUTE OWNER_WRITE OWNER_READ)


# build test
enable_testing()
link_directories(build)
add_executable(test_main.bin ./test_main.cpp)
target_link_libraries(test_main.bin face_recognition ${OpenCV_LIBS})

include_directories(/usr/include)

#add_test(test_main test_main.bin)

#add_executable(extract_features.bin ./extract_features.cpp)
#target_link_libraries(extract_features.bin face_recognition ${OpenCV_LIBS})


# add_executable("test_face_verification.bin" "./test_face_verification.cpp")
# target_link_libraries(test_face_verification.bin cmss_facedet_lib
#   cmss_facealign_lib cmss_faceid_lib
#   ${OpenCV_LIBS})

# make test
# add_test(test_face_verification test_face_verification.bin
#   ../cmssFaceEngine/FaceIdentification/data/test_face_recognizer/images/compare_im/Aaron_Peirsol_0001.jpg
#   ../cmssFaceEngine/FaceIdentification/data/test_face_recognizer/images/compare_im/Aaron_Peirsol_0004.jpg
#   ../cmssFaceEngine/FaceIdentification/model/cmss_fd_frontal_v1.0.bin
#   ../cmssFaceEngine/FaceIdentification/model/cmss_fa_v1.1.bin
#   ../cmssFaceEngine/FaceIdentification/model/cmss_fr_v1.0.bin)
