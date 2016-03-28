cmake_minimum_required(VERSION 2.8.3)
project(disp_map)

find_package(catkin REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
