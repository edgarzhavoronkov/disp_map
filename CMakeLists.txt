cmake_minimum_required(VERSION 2.8.3)
project(disp_map)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb")

find_package(catkin REQUIRED)

find_package(OpenCV 2.4.9 EXACT REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs std_msgs cv_bridge message_filters)

find_package(Boost REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
