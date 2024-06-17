#ifndef __MONO_VISUAL_ODOMETRY_H__
#define __MONO_VISUAL_ODOMETRY_H__

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class MonoVisualOdometry : public rclcpp::Node {
public:
  explicit MonoVisualOdometry(const rclcpp::NodeOptions &);

private:
  void imageCallback();
  std::vector<cv::Point2f> featureExtract(const cv::Mat &image);
  void featureMatching();
  void bundleAdjustment();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

#endif // __MONO_VISUAL_ODOMETRY_H__