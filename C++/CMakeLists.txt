cmake_minimum_required(VERSION 2.8)

## Enable C++11
#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_PREFIX_PATH "/home/link/libtorch-cxx11-abi-shared-with-deps-1.5.0+cpu/libtorch/share/cmake/Torch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
project(4444)
find_package( OpenCV 3.4.7 REQUIRED )
find_package(Torch REQUIRED)

file(GLOB ALLHEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
# 添加头文件
include_directories(  ${OpenCV_INCLUDE_DIRS}  )

add_executable(${PROJECT_NAME} "main.cpp" ${ALLHEADERS})

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 链接OpenCV库
target_link_libraries(4444 ${OpenCV_LIBS}  )

target_link_libraries(4444 "${TORCH_LIBRARIES}")

set_property(TARGET 4444 PROPERTY CXX_STANDARD 14)
cmake_minimum_required(VERSION 2.8)
