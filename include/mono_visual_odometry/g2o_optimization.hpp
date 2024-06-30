#ifndef __G2O_OPTIMIZATION_H__
#define __G2O_OPTIMIZATION_H__
#include <Eigen/Dense>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <sophus/se3.hpp>

class G2O_Optimization {
  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
  using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

public:
  G2O_Optimization(Eigen::Matrix3d camK);
  ~G2O_Optimization();
  void addPoseVertex(const int16_t &id, const Sophus::SE3d &pose);
  void addPointVertex(const int16_t &id, const Eigen::Vector3d &point);
  void addLandmarkEdge(int16_t poseId, int16_t landmarkId, const Eigen::Vector2d &measure);
  std::vector<Sophus::SE3d> getPose();
  void optimize();

private:
  void clear();

private:
  std::unique_ptr<g2o::SparseOptimizer> optimizer_;
  g2o::OptimizationAlgorithmLevenberg *solver_;
  std::vector<g2o::VertexSE3Expmap *> poseVertexs_;
  std::vector<g2o::VertexPointXYZ *> pointVertexs_;
};

#endif // __G2O_OPTIMIZATION_H__