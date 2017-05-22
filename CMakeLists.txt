cmake_minimum_required(VERSION 2.8.4)

project(CMSS)

# make install dir
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

set(CMAKE_CXX_FLAGS "-Wno-unused-result ${CMAKE_CXX_FLAGS}")

# Opencv
set(OpenCV_DIR "/home/shhs/env/opencv3_2_openface/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory("./SeetaFaceEngine/FaceDetection")
add_subdirectory("./SeetaFaceEngine/FaceAlignment")
add_subdirectory("./SeetaFaceEngine/FaceIdentification")

# Build face_recogniton lib
include_directories(.)
include_directories("./SeetaFaceEngine")
add_library(face_recognition SHARED ./CMSS_FaceRecognize.cpp)
target_link_libraries(face_recognition ${OpenCV_LIBS} seeta_facedet_lib
  seeta_facealign_lib seeta_faceid_lib) 
install(FILES ./CMSS_FaceRecognize.h DESTINATION include)
install(TARGETS face_recognition LIBRARY DESTINATION lib
  PERMISSIONS WORLD_EXECUTE OWNER_EXECUTE OWNER_WRITE OWNER_READ)


# build test
enable_testing()
link_directories(build)
add_executable(test_main.bin ./test_main.cpp)
target_link_libraries(test_main.bin face_recognition ${OpenCV_LIBS})
add_test(test_main test_main.bin)


# add_executable("test_face_verification.bin" "./test_face_verification.cpp")
# target_link_libraries(test_face_verification.bin seeta_facedet_lib
#   seeta_facealign_lib seeta_faceid_lib
#   ${OpenCV_LIBS})

# make test
# add_test(test_face_verification test_face_verification.bin
#   ../SeetaFaceEngine/FaceIdentification/data/test_face_recognizer/images/compare_im/Aaron_Peirsol_0001.jpg
#   ../SeetaFaceEngine/FaceIdentification/data/test_face_recognizer/images/compare_im/Aaron_Peirsol_0004.jpg
#   ../SeetaFaceEngine/FaceIdentification/model/seeta_fd_frontal_v1.0.bin
#   ../SeetaFaceEngine/FaceIdentification/model/seeta_fa_v1.1.bin
#   ../SeetaFaceEngine/FaceIdentification/model/seeta_fr_v1.0.bin)
