#include "mono_visual_odometry/mono_visual_odometry.hpp"
MonoVisualOdometry::MonoVisualOdometry(const rclcpp::NodeOptions &options) : Node("mono_visual_odometry", options) {
  RCLCPP_INFO(this->get_logger(), "MonoVisualOdometry");

  auto interval = std::chrono::duration<double>(1.0);
  timer_ = this->create_wall_timer(interval, [this]() -> void { RCLCPP_INFO(this->get_logger(), "Timer RUN!!"); });
}
