#include "mono_visual_odometry/mono_visual_odometry.hpp"
MonoVisualOdometry::MonoVisualOdometry(const rclcpp::NodeOptions &options) : Node("mono_visual_odometry", options) {
  RCLCPP_INFO(this->get_logger(), "MonoVisualOdometry");

  auto interval = std::chrono::duration<double>(1.0);
  timer_ = this->create_wall_timer(interval, [this]() -> void { RCLCPP_INFO(this->get_logger(), "Timer RUN!!"); });
}

std::vector<cv::Point2f> MonoVisualOdometry::featureExtract(const cv::Mat &image) {
  std::vector<cv::KeyPoint> keypoints_1;
  std::vector<cv::Point2f> points1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints_1, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());

  return points1;
}

void MonoVisualOdometry::featureMatching(const FrameData &prevFrameData, const FrameData &frameData) {}