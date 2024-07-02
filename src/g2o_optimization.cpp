#include "mono_visual_odometry/g2o_optimization.hpp"

G2O_Optimization::G2O_Optimization(Eigen::Matrix3d camK) {
  optimizer_ = std::make_unique<g2o::SparseOptimizer>();
  solver_ =
      new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  optimizer_->setAlgorithm(solver_);
  optimizer_->setVerbose(true);

  double focal_length = camK(0, 0);
  Eigen::Vector2d principal_point(camK(0, 2), camK(1, 2));
  g2o::CameraParameters *cam_params = new g2o::CameraParameters(focal_length, principal_point, 0.);
  cam_params->setId(0);

  optimizer_->addParameter(cam_params);
}

G2O_Optimization::~G2O_Optimization() { delete solver_; }

void G2O_Optimization::addPoseVertex(const int16_t &id, const Sophus::SE3d &pose) {
  g2o::SE3Quat g2oPose(pose.rotationMatrix(), pose.translation());
  g2o::VertexSE3Expmap *vertex = new g2o::VertexSE3Expmap();
  vertex->setId(id);
  if (poseVertexs_.size() < 2) {
    vertex->setFixed(true);
  }
  vertex->setEstimate(g2oPose);
  optimizer_->addVertex(vertex);
  poseVertexs_.push_back(vertex);
}

void G2O_Optimization::addPointVertex(const int16_t &id, const Eigen::Vector3d &point) {
  g2o::VertexPointXYZ *vertex = new g2o::VertexPointXYZ();
  vertex->setId(id);
  vertex->setMarginalized(true);
  vertex->setEstimate(point);
  optimizer_->addVertex(vertex);
  pointVertexs_.push_back(vertex);
}

void G2O_Optimization::addLandmarkEdge(int16_t poseId, int16_t landmarkId, const Eigen::Vector2d &measure) {
  g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
  g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;

  edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer_->vertex(landmarkId)));
  edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer_->vertex(poseId)));
  edge->setMeasurement(measure);
  edge->setInformation(Eigen::Matrix2d::Identity());
  edge->setRobustKernel(rk);
  edge->setParameterId(0, 0);
  optimizer_->addEdge(edge);
}

std::vector<Sophus::SE3d> G2O_Optimization::getPose() {
  std::vector<Sophus::SE3d> poses;
  std::for_each(poseVertexs_.begin(), poseVertexs_.end(), [&poses](const g2o::VertexSE3Expmap *pose) {
    const auto &poseEstimate = pose->estimate();
    Eigen::Quaterniond q = poseEstimate.rotation();
    Eigen::Vector3d t = poseEstimate.translation();
    poses.emplace_back(q, t);
  });
  return poses;
}

void G2O_Optimization::clear() {
  optimizer_->clear();
  poseVertexs_.clear();
  pointVertexs_.clear();
}

void G2O_Optimization::optimize() {
  optimizer_->initializeOptimization();
  optimizer_->setVerbose(false);
  optimizer_->optimize(10);
}
