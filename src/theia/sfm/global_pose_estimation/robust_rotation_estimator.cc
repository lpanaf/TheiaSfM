// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "theia/sfm/global_pose_estimation/robust_rotation_estimator.h"

#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <unordered_map>

#include "theia/math/l1_solver.h"
#include "theia/math/matrix/sparse_cholesky_llt.h"
#include "theia/math/rotation.h"
#include "theia/math/util.h"
#include "theia/sfm/types.h"
#include "theia/util/hash.h"
#include "theia/util/map_util.h"

namespace theia {

namespace {

// Helpter functions here
double median(std::vector<double>& v) {
    if (v.size() == 0) {
        return 0.0;
    }
    auto n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.begin() + v.size());
    auto med = v[n];
    if (!(v.size() & 1)) { //If the set size is even
        auto max_it = std::max_element(v.begin(), v.begin() + n);
        med = (*max_it + med) / 2.0;
    }
    return med;
}

double calculate_sigma(const Eigen::VectorXd& angles, bool skip=false) {
    std::vector<double> angles_abs;
    for (int i = 0; i < angles.size(); i++) {
        if (!skip || (i-1) % 3 == 0)
          angles_abs.push_back(std::abs(angles[i]));
    }
    double thres = 1.4826 * 3 * median(angles_abs); // 3 times scaled MAD (according to MATLAB)

    double result = 0;
    int counter = 0;
    for (int i = 0; i < angles_abs.size(); i++) {
        if (angles_abs[i] < thres) {
            result += std::pow(angles_abs[i], 2);
            counter++;
        }
    }

    return std::min(std::sqrt(result / std::max(counter, 1)), DegToRad(5));
    
}
}

bool RobustRotationEstimator::EstimateRotations(
    const std::unordered_map<ViewIdPair, TwoViewInfo>& view_pairs,
    std::unordered_map<ViewId, Eigen::Vector3d>* global_orientations) {
  for (const auto& view_pair : view_pairs) {
    AddRelativeRotationConstraint(view_pair.first, view_pair.second.rotation_2);
  }
  return EstimateRotations(global_orientations);
}

void RobustRotationEstimator::AddRelativeRotationConstraint(
    const ViewIdPair& view_id_pair, const Eigen::Vector3d& relative_rotation) {
  // Store the relative orientation constraint.
  relative_rotations_.emplace_back(view_id_pair, relative_rotation);
}

bool RobustRotationEstimator::EstimateRotations(
    std::unordered_map<ViewId, Eigen::Vector3d>* global_orientations,
    double sigma) {
  CHECK_GT(relative_rotations_.size(), 0)
      << "Relative rotation constraints must be added to the robust rotation "
         "solver before estimating global rotations.";
  global_orientations_ = CHECK_NOTNULL(global_orientations);

  // Compute a mapping of view ids to indices in the linear system. One rotation
  // will have an index of -1 and will not be added to the linear system. This
  // will remove the gauge freedom (effectively holding one camera as the
  // identity rotation).
  int index = -1;
  view_id_to_index_.reserve(global_orientations->size());
  for (const auto& orientation : *global_orientations) {
    view_id_to_index_[orientation.first] = index;
    ++index;
  }

  Eigen::SparseMatrix<double> sparse_mat;
  SetupLinearSystem();
  
  // Only optimize L1 if needed
  if (options_.optimize_l1 && !SolveL1Regression()) {
    LOG(ERROR) << "Could not solve the L1 regression step.";
    return false;
  }

  if (!SolveIRLS(sigma)) {
    LOG(ERROR) << "Could not solve the least squares error step.";
    return false;
  }

  return true;
}

// Set up the sparse linear system.
void RobustRotationEstimator::SetupLinearSystem() {
  // The rotation change is one less than the number of global rotations because
  // we keep one rotation constant.
  if (options_.prior_type != Options::SOFT_PENALTY) {
    rotation_change_.resize((global_orientations_->size() - 1) * 3);
    relative_rotation_error_.resize(relative_rotations_.size() * 3);
    sparse_matrix_.resize(relative_rotations_.size() * 3,
                          (global_orientations_->size() - 1) * 3);
  } else {
    rotation_change_.resize((global_orientations_->size() - 1) * 3);
    relative_rotation_error_.resize(relative_rotations_.size() * 3 + 2 * (global_orientations_->size() - 1));
    sparse_matrix_.resize(relative_rotations_.size() * 3 + 2 * (global_orientations_->size() - 1),
                          (global_orientations_->size() - 1) * 3);
  }

  // For each relative rotation constraint, add an entry to the sparse
  // matrix. We use the first order approximation of angle axis such that:
  // R_ij = R_j - R_i. This makes the sparse matrix just a bunch of identity
  // matrices.
  int rotation_error_index = 0;
  std::vector<Eigen::Triplet<double> > triplet_list;
  for (const auto& relative_rotation : relative_rotations_) {
    const int view1_index =
        FindOrDie(view_id_to_index_, relative_rotation.first.first);
    if (view1_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index,
                                3 * view1_index,
                                -1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view1_index + 1,
                                -1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view1_index + 2,
                                -1.0);
    }

    const int view2_index =
        FindOrDie(view_id_to_index_, relative_rotation.first.second);
    if (view2_index != kConstantRotationIndex) {
      triplet_list.emplace_back(3 * rotation_error_index + 0,
                                3 * view2_index + 0,
                                1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 1,
                                3 * view2_index + 1,
                                1.0);
      triplet_list.emplace_back(3 * rotation_error_index + 2,
                                3 * view2_index + 2,
                                1.0);
    }

    ++rotation_error_index;
  }

  // If incorporate prior error, add penalty directly to the x, z axis
  if (options_.prior_type == Options::SOFT_PENALTY) {
    int prior_error_index = 0;
    int base_index = 3 * rotation_error_index;
    for (auto ite = global_orientations_->begin(); ite != global_orientations_->end(); ite++) {
      int idx = view_id_to_index_[ite->first];
      if (idx == kConstantRotationIndex)
        continue;

      // Add constraint on the x and z axis
      triplet_list.emplace_back(2 * prior_error_index + base_index + 0,
                                3 * idx + 0,
                                options_.lambda);
      triplet_list.emplace_back(2 * prior_error_index + base_index + 1,
                                3 * idx + 2,
                                options_.lambda);
      
      prior_error_index++;
    }
  }

  sparse_matrix_.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

// Computes the relative rotation error based on the current global
// orientation estimates.
void RobustRotationEstimator::ComputeRotationError() {
  int rotation_error_index = 0;
  for (const auto& relative_rotation : relative_rotations_) {
    const Eigen::Vector3d& relative_rotation_aa = relative_rotation.second;
    const Eigen::Vector3d& rotation1 =
        FindOrDie(*global_orientations_, relative_rotation.first.first);
    const Eigen::Vector3d& rotation2 =
        FindOrDie(*global_orientations_, relative_rotation.first.second);

    // Compute the relative rotation error as:
    //   R_err = R2^t * R_12 * R1.
    relative_rotation_error_.segment<3>(3 * rotation_error_index) =
        MultiplyRotations(-rotation2,
                          MultiplyRotations(relative_rotation_aa, rotation1));
    ++rotation_error_index;
  }

  // If incorporate prior error, compute current error
  if (options_.prior_type == Options::SOFT_PENALTY) {
    int prior_error_index = 0;
    int base_index = 3 * rotation_error_index;
    for (auto ite = global_orientations_->begin(); ite != global_orientations_->end(); ite++) {
      int idx = view_id_to_index_[ite->first];
      if (idx == kConstantRotationIndex)
        continue;

      relative_rotation_error_(2 * prior_error_index + base_index) =
          -options_.lambda * ite->second(0);
      relative_rotation_error_(2 * prior_error_index + base_index + 1) =
          -options_.lambda * ite->second(2);

      prior_error_index++;
    }
  }

}

bool RobustRotationEstimator::SolveL1Regression() {
  static const double kConvergenceThreshold = 1e-3;

  L1Solver<Eigen::SparseMatrix<double> >::Options options;
  options.max_num_iterations = 10;
  L1Solver<Eigen::SparseMatrix<double> > l1_solver(options, sparse_matrix_);

  rotation_change_.setZero();
  // for (int i = 0; i < options_.max_num_l1_iterations; i++) {
  int iteration = 0;
  double prev_norm = 0;
  double curr_norm = 0;
  for (int i = 0; i < 100; i++) {
    iteration++;
    prev_norm = curr_norm;
    ComputeRotationError();
    l1_solver.Solve(relative_rotation_error_, &rotation_change_);
    UpdateGlobalRotations();

    curr_norm = rotation_change_.norm();
    // if (relative_rotation_error_.norm() < kConvergenceThreshold || prev_norm == curr_norm) {
    //   break;
    // }
    if (rotation_change_.norm() < kConvergenceThreshold || prev_norm == curr_norm) {
      break;
    }
    // options.max_num_iterations *= 2;
    // l1_solver.SetMaxIterations(options.max_num_iterations);
  }
  std::cout << "iterations: " << iteration << std::endl;
  return true;
}

// Update the global orientations using the current value in the
// rotation_change.
void RobustRotationEstimator::UpdateGlobalRotations() {
  for (auto& rotation : *global_orientations_) {
    const int view_index = FindOrDie(view_id_to_index_, rotation.first);
    if (view_index == kConstantRotationIndex) {
      continue;
    }

    if (options_.prior_type == Options::FIXED_AXIS) {
      // if axis is fixed, then project the rotation change to the original axix
      rotation_change_.segment<3>(3 * view_index) = rotation_change_.segment<3>(3 * view_index).dot(options_.axis) * options_.axis;
    }

    // Apply the rotation change to the global orientation.
    const Eigen::Vector3d& rotation_change =
        rotation_change_.segment<3>(3 * view_index);
    rotation.second = MultiplyRotations(rotation.second, rotation_change);
  }
}

bool RobustRotationEstimator::SolveIRLS(double sigma) {
  static const double kConvergenceThreshold = 1e-3;
  // This is the point where the Huber-like cost function switches from L1 to
  // L2.
  // static const double kSigma = DegToRad(sigma);

  if (!options_.optimize_l1)
    rotation_change_.setZero();

  // Dynamically determin the error here
  ComputeRotationError();
  const double kSigma = (sigma > 0) ? DegToRad(sigma) : calculate_sigma(sparse_matrix_ * rotation_change_ - relative_rotation_error_, options_.prior_type == Options::FIXED_AXIS);
  std::cout << "sigma: " << sigma << std::endl;
  std::cout << "kSigma: " << RadToDeg(kSigma) << std::endl;

  // Set up the linear solver and analyze the sparsity pattern of the
  // system. Since the sparsity pattern will not change with each linear solve
  // this can help speed up the solution time.
  SparseCholeskyLLt linear_solver;
  linear_solver.AnalyzePattern(sparse_matrix_.transpose() * sparse_matrix_);
  if (linear_solver.Info() != Eigen::Success) {
    LOG(ERROR) << "Cholesky decomposition failed.";
    return false;
  }

  VLOG(2) << "Iteration   Error           Delta";
  const std::string row_format = "  % 4d     % 4.4e     % 4.4e";

  Eigen::ArrayXd errors, weights;
  Eigen::SparseMatrix<double> at_weight;
  for (int i = 0; i < options_.max_num_irls_iterations; i++) {
    const Eigen::VectorXd prev_rotation_change = rotation_change_;
    ComputeRotationError();

    // Compute the weights for each error term.
    errors =
        (sparse_matrix_ * rotation_change_ - relative_rotation_error_).array();
    weights = kSigma / (errors.square() + kSigma * kSigma).square();

    // Update the factorization for the weighted values.
    at_weight =
        sparse_matrix_.transpose() * weights.matrix().asDiagonal();
    linear_solver.Factorize(at_weight * sparse_matrix_);
    if (linear_solver.Info() != Eigen::Success) {
      LOG(ERROR) << "Failed to factorize the least squares system.";
      return false;
    }

    // Solve the least squares problem..
    rotation_change_ =
        linear_solver.Solve(at_weight * relative_rotation_error_);
    
    if (linear_solver.Info() != Eigen::Success) {
      LOG(ERROR) << "Failed to solve the least squares system.";
      return false;
    }

    UpdateGlobalRotations();

    // Log some statistics for the output.
    const double rotation_change_sq_norm =
        (prev_rotation_change - rotation_change_).squaredNorm();
    VLOG(2) << StringPrintf(row_format.c_str(), i, errors.square().sum(),
                            rotation_change_sq_norm);
    if (rotation_change_sq_norm < kConvergenceThreshold) {
      VLOG(1) << "IRLS Converged in " << i + 1 << " iterations.";
      break;
    }
  }
  return true;
}

}  // namespace theia
