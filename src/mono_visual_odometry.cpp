#include "mono_visual_odometry/mono_visual_odometry.hpp"
MonoVisualOdometry::MonoVisualOdometry(const rclcpp::NodeOptions &options) : Node("mono_visual_odometry", options) {
  RCLCPP_INFO(this->get_logger(), "MonoVisualOdometry");

  debugImagePub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/image", 10);
  pathPub_ = this->create_publisher<nav_msgs::msg::Path>("/stereo/path", 10);
  imageSub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/mono/image", 10, std::bind(&MonoVisualOdometry::imageCallback, this, std::placeholders::_1));

  auto interval = std::chrono::duration<double>(1.0);
  timer_ = this->create_wall_timer(interval, [this]() -> void { RCLCPP_INFO(this->get_logger(), "Timer RUN!!"); });
}

void MonoVisualOdometry::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &leftImage) {}

std::vector<cv::Point2f> MonoVisualOdometry::featureExtract(const FrameData &frameData) {
  std::vector<cv::KeyPoint> keypoints_1;
  std::vector<cv::Point2f> points1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(frameData.image, keypoints_1, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints_1, points1, std::vector<int>());

  return points1;
}

void MonoVisualOdometry::featureMatching(const FrameData &prevFrameData, const FrameData &frameData) {}

void MonoVisualOdometry::debugImagePublish() {
  if (frames_.size() > 0) {
    auto message = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frames_.end()->image).toImageMsg();
    message->header.stamp = this->get_clock()->now();
    debugImagePub_->publish(*message);
  }
}

void MonoVisualOdometry::pathPublish() {
  nav_msgs::msg::Path pathMsg;
  pathMsg.header.stamp = this->now();
  pathMsg.header.frame_id = "map";
  pathMsg.poses = poses_;

  pathPub_->publish(pathMsg);
}