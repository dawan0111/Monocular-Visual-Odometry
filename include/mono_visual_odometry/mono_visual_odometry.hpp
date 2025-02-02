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
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sophus/se3.hpp>

struct Point {
  int16_t id;
  int16_t frameId;
  cv::Point2f point;
  Point *prev;
  Point *next;
  bool isInliner;
  Eigen::Vector3d worldPoint;

  Point(int16_t id_, int16_t frameId_, cv::Point2f point_, bool isInliner_)
      : id(id_), frameId(frameId_), point(point_), isInliner(isInliner_) {
    prev = nullptr;
    next = nullptr;
    worldPoint = Eigen::Vector3d::Zero();
  };
};
struct FrameData {
  cv::Mat image;
  cv::Mat grayImage;
  std::vector<Point> keyPoints;
  int32_t frameId;
  int16_t inlinerCount;
  bool isInliner;
  Sophus::SE3d pose;
  Sophus::SE3d worldPose;

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
  void debugImagePublish(const FrameData &frameData);
  void pathPublish();
  void pointCloudPublish(const FrameData &frameData);
  void updatePath(const FrameData &frameData);
  void updateAllPath();

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debugImagePub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointCloudPub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pathPub_;

  std::vector<std::shared_ptr<FrameData>> frames_;
  std::vector<std::shared_ptr<FrameData>> localFrames_;
  std::vector<geometry_msgs::msg::PoseStamped> poses_;
  FrameData currFrame_;

  int32_t frameId_;

  cv::Mat camK_;
  Eigen::Matrix3d EigenCamK_;
  Sophus::SE3d latestPose_;
  Sophus::SE3d T_optical_world_;
};

#endif // __MONO_VISUAL_ODOMETRY_H__