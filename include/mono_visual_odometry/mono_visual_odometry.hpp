#ifndef __MONO_VISUAL_ODOMETRY_H__
#define __MONO_VISUAL_ODOMETRY_H__

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class MonoVisualOdometry : public rclcpp::Node {
public:
  explicit MonoVisualOdometry(const rclcpp::NodeOptions &);

private:
  void imageCallback();
  void featureExtract();
  void featureMatching();
  void bundleAdjustment();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

#endif // __MONO_VISUAL_ODOMETRY_H__