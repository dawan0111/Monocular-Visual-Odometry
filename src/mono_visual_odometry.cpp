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
  frames_.push_back(std::move(frame));

  // featureExtract(frame);

  if (frames_.size() <= 1) {
    featureExtract(frames_[frameId_]);
    std::cout << frames_[frameId_].keyPoints.size() << std::endl;
  } else {
    featureMatching(frames_[frameId_ - 1], frames_[frameId_]);
    if (frames_[frameId_].inlinerCount < 200) {
      RCLCPP_INFO(this->get_logger(), "Please ReExtract FrameData");
      // frames_[frameId_].keyPoints.clear();
      // featureExtract(frames_[frameId_]);
    }
  }

  debugImagePublish(frames_[frameId_]);
  frameId_++;
}

void MonoVisualOdometry::featureExtract(FrameData &frameData) {
  std::vector<cv::KeyPoint> keypoints_1;
  int fast_threshold = 20;
  int16_t id = 0;
  bool nonmaxSuppression = true;
  cv::FAST(frameData.image, keypoints_1, fast_threshold, nonmaxSuppression);

  for (auto &keyPoint : keypoints_1) {
    frameData.keyPoints.emplace_back(id, keyPoint.pt, true);
    ++id;
  }
}

void MonoVisualOdometry::featureMatching(FrameData &prevFrameData, FrameData &frameData) {
  int16_t inlinerCount = 0;
  std::vector<float> err;
  std::vector<uchar> status;
  cv::Size winSize = cv::Size(21, 21);
  cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

  std::vector<cv::Point2f> prevPoints;
  std::vector<Point *> prevPointers;
  std::vector<cv::Point2f> nextPoints;

  std::for_each(prevFrameData.keyPoints.begin(), prevFrameData.keyPoints.end(), [&prevPointers, &prevPoints](auto &d) {
    if (d.isInliner) {
      prevPoints.push_back(d.point);
      prevPointers.push_back(&d);
    }
  });

  cv::calcOpticalFlowPyrLK(prevFrameData.image, frameData.image, prevPoints, nextPoints, status, err, winSize, 3,
                           termcrit, 0, 0.001);
  for (int i = 0; i < status.size(); i++) {
    auto prevPoints = prevPointers[i];
    auto pt = nextPoints[i];
    if (pt.x < 0 || pt.y < 0) {
      status[i] = 0;
    }
    Point point(i, pt, true);
    point.prev = prevPoints;
    frameData.keyPoints.push_back(std::move(point));
    prevFrameData.keyPoints[i].next = &frameData.keyPoints[i];

    if (status[i] == 0) {
      prevFrameData.keyPoints[i].isInliner = false;
      frameData.keyPoints[i].isInliner = false;
    } else {
      ++inlinerCount;
    }
  }

  frameData.inlinerCount = inlinerCount;
}

void MonoVisualOdometry::featureTracking(FrameData &frameData) {
  std::vector<cv::Point2f> prevPoints;
  std::vector<cv::Point2f> nextPoints;

  std::for_each(frameData.keyPoints.begin(), frameData.keyPoints.end(),
                [&prevPoints, &nextPoints](const Point &keyPoint) {
                  prevPoints.push_back(keyPoint.prev->point);
                  nextPoints.push_back(keyPoint.point);
                });

  /**
   * TODO: change to configure value
   */
  double focal = 718.8560;
  cv::Point2d pp(607.1928, 185.2157);

  cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
  cv::Mat mask, R, t;
  int16_t inlinerCount;

  auto E = cv::findEssentialMat(nextPoints, prevPoints, K, cv::RANSAC, 0.9999, 1.0, mask);
  cv::recoverPose(E, nextPoints, prevPoints, R, t, K, mask);

  Eigen::Matrix3d eR;
  Eigen::Vector3d et;
  eR = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(R.ptr<double>());
  et = Eigen::Map<Eigen::Matrix<double, 3, 1, Eigen::RowMajor>>(t.ptr<double>());

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.rotate(eR);
  pose.pretranslate(et);

  frameData.pose = pose;

  for (int i = 0; i < mask.rows; ++i) {
    if (!mask.at<uchar>(i)) {
      frameData.keyPoints[i].isInliner = false;
      frameData.keyPoints[i].prev->isInliner = false;
    } else {
      ++inlinerCount;
    }
  }
  std::cout << "Inliner count: " << inlinerCount << std::endl;
}

void MonoVisualOdometry::debugImagePublish(const FrameData &frameData) {
  if (frames_.size() > 0) {
    auto debugImage = frameData.image.clone();
    for (auto &keyPoint : frameData.keyPoints) {
      cv::circle(debugImage, keyPoint.point, 4, cv::Scalar(0, 255, 0), 1, cv::LINE_4, 0);
      if (keyPoint.prev != nullptr) {
        cv::circle(debugImage, keyPoint.prev->point, 4, cv::Scalar(0, 255, 255), 1, cv::LINE_4, 0);
        cv::arrowedLine(debugImage, keyPoint.point, keyPoint.prev->point, cv::Scalar(0, 255, 255), 1);
      }
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