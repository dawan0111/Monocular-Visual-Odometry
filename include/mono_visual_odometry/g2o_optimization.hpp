#ifndef __G2O_OPTIMIZATION_H__
#define __G2O_OPTIMIZATION_H__

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

class G2O_Optimization {
public:
  G2O_Optimization();
};

#endif // __G2O_OPTIMIZATION_H__