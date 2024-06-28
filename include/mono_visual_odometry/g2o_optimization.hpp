#ifndef __G2O_OPTIMIZATION_H__
#define __G2O_OPTIMIZATION_H__

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

class G2O_Optimization {
  using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
  using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;
  using WorldPoints = std::vector<Eigen::Vector3d>;
  using CameraPoints = std::vector<Eigen::Vector2d>;

public:
  G2O_Optimization();
  ~G2O_Optimization();
  void updatePoseVertex();
  void updateEdge();
  void clear();
  void optimize();

private:
  std::unique_ptr<g2o::SparseOptimizer> optimizer_;
  g2o::OptimizationAlgorithmLevenberg *solver_;
};

#endif // __G2O_OPTIMIZATION_H__