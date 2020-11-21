cmake_minimum_required(VERSION 3.5)
project(ji)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 设置Tensorflow路径
set(TENSORFLOW_DIR "/opt/tensorflow")
set(ADDITIONAL_RPATH ${TENSORFLOW_DIR}/lib)

# 设置RPATH，使编译好的libji.so在运行时能够找到依赖库
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
if (DEFINED ADDITIONAL_RPATH)
    set(CMAKE_INSTALL_RPATH "${ADDITIONAL_RPATH}:${CMAKE_INSTALL_RPATH}")
endif ()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
find_package(PkgConfig REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/cJSON)

# 编译libji.so
list(APPEND libji_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ji.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SampleDetectorImpl.cpp)
link_directories(${TENSORFLOW_DIR}/lib)
add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${TENSORFLOW_DIR}/include)
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        cjson
        tensorflow)


# 以下是EV_SDK所依赖和需要测试接口的程序&库，请勿修改！！！
# --- BEGIN EV_SDK DEPENDENCIES ---
# 编译 JI API 测试程序
add_executable(test-ji-api ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)
target_link_libraries(test-ji-api
        ${OpenCV_LIBS}
        cjson
        glog
        ${PROJECT_NAME})
# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
# Install test-ji-api
install(TARGETS test-ji-api DESTINATION bin)
# ---  END EV_SDK DEPENDENCIES  ---