#ifndef __MONO_VISUAL_ODOMETRY_H__
#define __MONO_VISUAL_ODOMETRY_H__
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
struct FrameData {
  cv::Mat image;
  std::vector<cv::Point2f> keyPoints;
  int32_t frameId;
};
class MonoVisualOdometry : public rclcpp::Node {
public:
  explicit MonoVisualOdometry(const rclcpp::NodeOptions &);

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &leftImage);
  std::vector<cv::Point2f> featureExtract(const FrameData &frameData);
  void featureMatching(const FrameData &prevFrameData, const FrameData &frameData);
  void bundleAdjustment();
  void debugImagePublish();
  void pathPublish();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugImagePub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub_;

  std::vector<FrameData> frames_;
  std::vector<geometry_msgs::msg::PoseStamped> poses_;
};

#endif // __MONO_VISUAL_ODOMETRY_H__