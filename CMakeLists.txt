cmake_minimum_required(VERSION 3.8)
project(mono_visual_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(library_name sv_lib)
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(g2o REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${G2O_INCLUDE_DIRS}
)

add_library(${library_name}
  src/mono_visual_odometry.cpp
)

target_link_libraries(${library_name}
  g2o_core
  g2o_stuff
)

ament_target_dependencies(${library_name}
  rclcpp
  sensor_msgs
)

add_executable(mono_visual_odometry_node
  src/mono_visual_odometry_node.cpp
)
ament_target_dependencies(mono_visual_odometry_node
  rclcpp
)
target_link_libraries(mono_visual_odometry_node
  ${library_name}
)

target_include_directories(mono_visual_odometry_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS
  mono_visual_odometry_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
