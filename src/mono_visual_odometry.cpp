#include "mono_visual_odometry/mono_visual_odometry.hpp"
MonoVisualOdometry::MonoVisualOdometry(const rclcpp::NodeOptions &options) : Node("mono_visual_odometry", options) {
  RCLCPP_INFO(this->get_logger(), "MonoVisualOdometry");

  debugImagePub_ = this->create_publisher<sensor_msgs::msg::Image>("/mono/image", 10);
  pathPub_ = this->create_publisher<nav_msgs::msg::Path>("/stereo/path", 10);
  imageSub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/stereo/image_left", 10, std::bind(&MonoVisualOdometry::imageCallback, this, std::placeholders::_1));

  auto interval = std::chrono::duration<double>(1.0);
  // timer_ = this->create_wall_timer(interval, [this]() -> void { RCLCPP_INFO(this->get_logger(), "Timer RUN!!"); });

  frameId_ = 0;
  currFrame_.keyPoints.reserve(6000);
}

void MonoVisualOdometry::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image) {
  RCLCPP_INFO(this->get_logger(), "Image Callback!!");
  FrameData frame;

  frame.image = cv_bridge::toCvCopy(image, image->encoding)->image;
  frame.frameId = frameId_;
  frame.keyPoints.reserve(2000);

  featureExtract(frame);

  if (frames_.size() < 1) {
    featureExtract(frame);
    std::cout << frame.keyPoints.size() << std::endl;
  } else {
    featureMatching(frames_[frameId_ - 1], frame);
    if (frame.inlinerCount < 400) {
      RCLCPP_INFO(this->get_logger(), "ReExtract FrameData");
      frame.keyPoints.clear();
      featureExtract(frame);
    }
  }

  debugImagePublish(frame);
  frames_.push_back(std::move(frame));
  frameId_++;
}

void MonoVisualOdometry::featureExtract(FrameData &frameData) {
  std::vector<cv::KeyPoint> keypoints_1;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(frameData.image, keypoints_1, fast_threshold, nonmaxSuppression);
  std::cout << "size: " << keypoints_1.size() << std::endl;
  cv::KeyPoint::convert(keypoints_1, frameData.keyPoints, std::vector<int>());
}

void MonoVisualOdometry::featureMatching(FrameData &prevFrameData, FrameData &frameData) {
  std::vector<float> err;
  std::vector<uchar> status;
  cv::Size winSize = cv::Size(21, 21);

  std::cout << "Prev: " << prevFrameData.keyPoints.size() << std::endl;
  std::cout << "Next: " << frameData.keyPoints.size() << std::endl;
  cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

  cv::calcOpticalFlowPyrLK(prevFrameData.image, frameData.image, prevFrameData.keyPoints, frameData.keyPoints, status,
                           err, winSize, 3, termcrit, 0, 0.001);

  int indexCorrection = 0;
  for (int i = 0; i < status.size(); i++) {
    cv::Point2f pt = frameData.keyPoints.at(i - indexCorrection);
    if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
      if ((pt.x < 0) || (pt.y < 0)) {
        status.at(i) = 0;
      }
      prevFrameData.keyPoints.erase(prevFrameData.keyPoints.begin() + (i - indexCorrection));
      frameData.keyPoints.erase(frameData.keyPoints.begin() + (i - indexCorrection));
      indexCorrection++;
    }
  }
}

void MonoVisualOdometry::debugImagePublish(const FrameData &frameData) {
  if (frames_.size() > 0) {
    auto debugImage = frameData.image.clone();
    for (auto &keyPoint : frameData.keyPoints) {
      cv::circle(debugImage, keyPoint, 2, cv::Scalar(0, 255, 0), 1, cv::LINE_4, 0);
    }
    auto message = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debugImage).toImageMsg();
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