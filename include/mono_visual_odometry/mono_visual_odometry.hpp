#ifndef __MONO_VISUAL_ODOMETRY_H__
#define __MONO_VISUAL_ODOMETRY_H__
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sophus/se3.hpp>

struct Point {
  int16_t id;
  cv::Point2f point;
  Point *prev;
  Point *next;
  bool isInliner;

  Point(int16_t id_, cv::Point2f point_, bool isInliner_) : id(id_), point(point_), isInliner(isInliner_) {
    prev = nullptr;
    next = nullptr;
  };
};
struct FrameData {
  cv::Mat image;
  std::vector<Point> keyPoints;
  int32_t frameId;
  int16_t inlinerCount;

  Sophus::SE3d pose;

  Point *getKeyPoint(int16_t id) {
    auto it = std::find_if(keyPoints.begin(), keyPoints.end(), [id](const Point &d) { return d.id == id; });
    return it != keyPoints.end() ? &(*it) : nullptr;
  }
};
class MonoVisualOdometry : public rclcpp::Node {
public:
  explicit MonoVisualOdometry(const rclcpp::NodeOptions &);

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &leftImage);
  void featureExtract(FrameData &frameData);
  void featureMatching(FrameData &prevFrameData, FrameData &frameData);
  void featureTracking(FrameData &frameData);
  void getPose(const FrameData &prevFrameData, const FrameData &frameData);
  void bundleAdjustment();
  void debugImagePublish(const FrameData &frameData);
  void pathPublish();
  void updatePath(const FrameData &frameData);

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugImagePub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub_;

  std::vector<FrameData> frames_;
  std::vector<geometry_msgs::msg::PoseStamped> poses_;
  FrameData currFrame_;

  int32_t frameId_;

  cv::Mat camK_;
  Sophus::SE3d latestPose_;
  Sophus::SE3d T_optical_world_;
};

#endif // __MONO_VISUAL_ODOMETRY_H__