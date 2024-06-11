#include "mono_visual_odometry/mono_visual_odometry.hpp"
#include <iostream>

int32_t main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  const rclcpp::NodeOptions options;
  auto mono_visual_odometry_node = std::make_shared<MonoVisualOdometry>(options);

  rclcpp::spin(mono_visual_odometry_node->get_node_base_interface());
  rclcpp::shutdown();
  return 0;
}