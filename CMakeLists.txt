cmake_minimum_required(VERSION 3.22)
project(torch_learn)

set(CMAKE_CXX_STANDARD 20)

set(Torch_DIR ../torch-learn/third-party/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

#set(Python3_EXECUTABLE /Users/bichaoqun/miniconda3/envs/torch-learn/bin/python3.8)
#set(Python3_INCLUDE_DIR /Users/bichaoqun/miniconda3/envs/torch-learn/include/python3.8)
#set(Python3_LIBRARY /Users/bichaoqun/miniconda3/envs/torch-learn/lib/libpython3.8.dylib)
#find_package(Python3 REQUIRED Development)

#add_subdirectory(../torch-learn/third-party/pybind11)
#add_subdirectory(../torch-learn/third-party/NumCpp)

include_directories(include)

add_executable(torch_learn src/main.cpp
        src/Preliminaries.cpp
        include/Preliminaries.h)

#target_link_libraries(torch_learn ${TORCH_LIBRARIES} pybind11::embed)
target_link_libraries(torch_learn ${TORCH_LIBRARIES})
