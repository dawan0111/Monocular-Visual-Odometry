#include "mono_visual_odometry/mono_visual_odometry.hpp"
MonoVisualOdometry::MonoVisualOdometry(const rclcpp::NodeOptions &options) : Node("mono_visual_odometry", options) {
  RCLCPP_INFO(this->get_logger(), "MonoVisualOdometry");

  debugImagePub_ = this->create_publisher<sensor_msgs::msg::Image>("/mono/image", 50);
  pathPub_ = this->create_publisher<nav_msgs::msg::Path>("/mono/path", 10);
  imageSub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/stereo/image_left", 10, std::bind(&MonoVisualOdometry::imageCallback, this, std::placeholders::_1));

  auto interval = std::chrono::duration<double>(1.0);
  // timer_ = this->create_wall_timer(interval, [this]() -> void { RCLCPP_INFO(this->get_logger(), "Timer RUN!!"); });

  frameId_ = 0;
  latestPose_ = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  currFrame_.keyPoints.reserve(6000);

  Eigen::Matrix4d T_optical_world;
  T_optical_world << 0.0, 0.0, 1.0, 0.0, -1.0, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  T_optical_world_ = Sophus::SE3d(T_optical_world);
}

void MonoVisualOdometry::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image) {
  auto frame = std::make_unique<FrameData>();
  frame->image = cv_bridge::toCvCopy(image, image->encoding)->image;
  cv::cvtColor(frame->image, frame->grayImage, cv::COLOR_BGR2GRAY);
  frame->frameId = frameId_;
  frame->keyPoints.reserve(6000);
  frame->isInliner = true;
  frame->inlinerCount = 0;
  frames_.push_back(std::move(frame));

  auto &currFrame = frames_[frameId_];

  if (frames_.size() <= 1) {
    featureExtract(*currFrame);
    frameId_++;
  } else {
    const auto copyPrevFrame = *frames_[frameId_ - 1];
    auto &prevFrame = frames_[frameId_ - 1];

    featureMatching(*prevFrame, *currFrame);
    featureTracking(*currFrame);

    if (currFrame->isInliner) {
      updatePath(*currFrame);
      pathPublish();
      debugImagePublish(*currFrame);

      std::cout << "Inliner: " << currFrame->inlinerCount << std::endl;

      if (currFrame->inlinerCount < 500) {
        std::cout << "Re FrameId: " << frameId_ << std::endl;
        currFrame->keyPoints.clear();
        featureExtract(*currFrame);
      }

      frameId_++;
    } else {
      RCLCPP_INFO(this->get_logger(), "Stop frame!!");
      prevFrame = std::make_unique<FrameData>(copyPrevFrame);
      frames_.erase(frames_.end() - 1);
    }
  }
}

void MonoVisualOdometry::featureExtract(FrameData &frameData) {
  std::vector<cv::KeyPoint> keypoints_1;
  int fast_threshold = 40;
  int16_t id = 0;
  bool nonmaxSuppression = true;
  cv::Mat image;
  cv::FAST(frameData.grayImage, keypoints_1, fast_threshold, nonmaxSuppression);

  for (auto &keyPoint : keypoints_1) {
    frameData.keyPoints.emplace_back(id, keyPoint.pt, true);
    ++id;
  }
  frameData.inlinerCount = keypoints_1.size();
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

  cv::Mat prevImage;
  cv::Mat image;

  cv::calcOpticalFlowPyrLK(prevFrameData.grayImage, frameData.grayImage, prevPoints, nextPoints, status, err, winSize,
                           3, termcrit, 0, 0.001);
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
                  if (keyPoint.isInliner) {
                    prevPoints.push_back(keyPoint.prev->point);
                    nextPoints.push_back(keyPoint.point);
                  }
                });
  /**
   * TODO: change to configure value
   */
  double focal = 718.8560;
  cv::Point2d pp(607.1928, 185.2157);

  cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
  cv::Mat mask, R, t;

  auto E = cv::findEssentialMat(nextPoints, prevPoints, K, cv::RANSAC, 0.999, 1.0, mask);
  cv::recoverPose(E, nextPoints, prevPoints, K, R, t, mask);

  Eigen::Matrix3d eR;
  Eigen::Vector3d et;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      eR(i, j) = R.at<double>(i, j);
    }
  }

  et(0) = t.at<double>(0);
  et(1) = t.at<double>(1);
  et(2) = t.at<double>(2);

  frameData.pose = Sophus::SE3d(eR, et);
  auto angleNorm = frameData.pose.so3().log().norm();
  // et(2) < et(0) || et(2) < et(1) ||

  if (std::abs(angleNorm) > 2 || et(2) <= 0) {
    frameData.isInliner = false;
  }
}

void MonoVisualOdometry::updatePath(const FrameData &frameData) {
  if (frameData.isInliner) {
    auto updatePose = latestPose_ * frameData.pose;
    geometry_msgs::msg::PoseStamped poseStampMsg;
    poseStampMsg.header.stamp = this->get_clock()->now();
    poseStampMsg.header.frame_id = "map";

    auto worldPose = T_optical_world_ * updatePose;

    double xDiff = updatePose.translation().x() - latestPose_.translation().x();
    double yDiff = updatePose.translation().y() - latestPose_.translation().y();
    double zDiff = updatePose.translation().z() - latestPose_.translation().z();

    auto scale = std::sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

    // std::cout << "yDiff: " << yDiff * yDiff << std::endl;
    // std::cout << "zDiff: " << zDiff * zDiff << std::endl;

    geometry_msgs::msg::Pose pose;
    Eigen::Quaterniond q(worldPose.rotationMatrix());
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();

    pose.position.x = worldPose.translation().x();
    pose.position.y = worldPose.translation().y();
    pose.position.z = worldPose.translation().z();
    poseStampMsg.pose = pose;

    poses_.push_back(std::move(poseStampMsg));
    latestPose_ = updatePose;
  }
}

void MonoVisualOdometry::debugImagePublish(const FrameData &frameData) {
  auto debugImage = frameData.image.clone();
  for (auto &keyPoint : frameData.keyPoints) {
    if (keyPoint.isInliner) {
      cv::circle(debugImage, keyPoint.point, 4, cv::Scalar(0, 255, 0), 1, cv::LINE_4, 0);
      if (keyPoint.prev != nullptr) {
        cv::arrowedLine(debugImage, keyPoint.point, keyPoint.prev->point, cv::Scalar(0, 255, 255), 1);
        cv::circle(debugImage, keyPoint.prev->point, 4, cv::Scalar(255, 0, 0), 1, cv::LINE_4, 0);
      }
    }
  }
  auto message = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debugImage).toImageMsg();
  message->header.stamp = this->get_clock()->now();
  debugImagePub_->publish(*message);
}

void MonoVisualOdometry::pathPublish() {
  nav_msgs::msg::Path pathMsg;
  pathMsg.header.stamp = this->now();
  pathMsg.header.frame_id = "map";
  pathMsg.poses = poses_;

  pathPub_->publish(pathMsg);
}