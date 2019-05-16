cmake_minimum_required(VERSION 3.3.2)
project(face-engine)
enable_language(CXX)

find_package(dlib REQUIRED)
find_package(PkgConfig)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGES} -std=gnu++17)

pkg_check_modules(GLIB glib-2.0>=2.24 gobject-2.0)

include_directories(${DLIB_INCLUDE_DIRECTORY})

add_library(face STATIC face_engine.cpp)

add_executable(fr_cli main.cpp)
target_link_libraries(fr_cli face dlib::dlib ${GLIB_LIBRARIES} stdc++fs)
target_include_directories(fr_cli PRIVATE ${GLIB_INCLUDE_DIRS})

