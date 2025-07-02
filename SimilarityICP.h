#ifndef SIMILARITY_ICP_H
#define SIMILARITY_ICP_H

#include "ICP.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

///////////////////////////////////////////////////////////////////////////////
/// Similarity transformation ICP implementation
/// Extends rigid transformation to include uniform scaling
namespace SimilarityICP {
    
    /// Similarity transformation matrix structure
    /// T = [sR t; 0 1] where s is scale, R is rotation, t is translation
    struct SimilarityTransform {
        double scale;
        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;
        
        SimilarityTransform() : scale(1.0), rotation(Eigen::Matrix3d::Identity()), 
                               translation(Eigen::Vector3d::Zero()) {}
        
        /// Convert to 4x4 homogeneous matrix
        Eigen::Matrix4d toMatrix() const {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = scale * rotation;
            T.block<3,1>(0,3) = translation;
            return T;
        }
        
        /// Convert from 4x4 homogeneous matrix
        void fromMatrix(const Eigen::Matrix4d& T) {
            Eigen::Matrix3d sR = T.block<3,3>(0,0);
            scale = sR.col(0).norm(); // Extract scale from first column
            rotation = sR / scale;    // Normalize to get rotation
            translation = T.block<3,1>(0,3);
        }
        
        /// Apply transformation to point cloud
        template<typename Derived>
        Eigen::Matrix3Xd apply(const Eigen::MatrixBase<Derived>& points) const {
            return scale * rotation * points + translation.replicate(1, points.cols());
        }
    };
    
    /// Solve similarity transformation using point-to-point correspondences
    /// Based on the closed-form solution for similarity transformation
    /// @param Source points (one 3D point per column)
    /// @param Target points (one 3D point per column) 
    /// @param Confidence weights
    template <typename Derived1, typename Derived2, typename Derived3>
    SimilarityTransform point_to_point_similarity(
        Eigen::MatrixBase<Derived1>& X,
        Eigen::MatrixBase<Derived2>& Y,
        const Eigen::MatrixBase<Derived3>& w) {
        
        typedef typename Derived1::Scalar Scalar;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXX;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
        
        int dim = X.rows();
        
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = w.template cast<double>() / w.sum();
        
        /// Compute weighted centroids
        Eigen::Vector3d X_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d Y_mean = Eigen::Vector3d::Zero();
        
        for (int i = 0; i < 3; ++i) {
            X_mean(i) = (X.row(i).array() * w_normalized.transpose().array()).sum();
            Y_mean(i) = (Y.row(i).array() * w_normalized.transpose().array()).sum();
        }
        
        /// Center the point clouds
        Eigen::Matrix3Xd X_centered = X.template cast<double>();
        Eigen::Matrix3Xd Y_centered = Y.template cast<double>();
        X_centered.colwise() -= X_mean;
        Y_centered.colwise() -= Y_mean;
        
        /// Compute weighted covariance matrix
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (int i = 0; i < X.cols(); ++i) {
            H += w_normalized(i) * X_centered.col(i) * Y_centered.col(i).transpose();
        }
        
        /// Compute scale factor
        double sigma_X = 0.0, sigma_Y = 0.0;
        for (int i = 0; i < X.cols(); ++i) {
            sigma_X += w_normalized(i) * X_centered.col(i).squaredNorm();
            sigma_Y += w_normalized(i) * Y_centered.col(i).squaredNorm();
        }
        
        /// SVD of covariance matrix
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        
        /// Compute rotation matrix
        Eigen::Matrix3d R = V * U.transpose();
        
        /// Ensure proper rotation (det(R) = 1)
        if (R.determinant() < 0) {
            V.col(2) *= -1;
            R = V * U.transpose();
        }
        
        /// Compute scale
        double trace_H = svd.singularValues().sum();
        double scale = trace_H / sigma_X;
        
        /// Compute translation
        Eigen::Vector3d t = Y_mean - scale * R * X_mean;
        
        /// Create similarity transformation
        SimilarityTransform sim_trans;
        sim_trans.scale = scale;
        sim_trans.rotation = R;
        sim_trans.translation = t;
        
        return sim_trans;
    }
    
    /// Overload without weights (uniform weights)
    template <typename Derived1, typename Derived2>
    inline SimilarityTransform point_to_point_similarity(
        Eigen::MatrixBase<Derived1>& X,
        Eigen::MatrixBase<Derived2>& Y) {
        return point_to_point_similarity(X, Y, Eigen::VectorXd::Ones(X.cols()));
    }
    
    /// Similarity ICP algorithm
    /// @param Source points (one 3D point per column)
    /// @param Target points (one 3D point per column)
    /// @param max_iterations Maximum number of iterations
    /// @param tolerance Convergence tolerance
    template <typename Derived1, typename Derived2>
    SimilarityTransform similarity_icp(
        Eigen::MatrixBase<Derived1>& X,
        Eigen::MatrixBase<Derived2>& Y,
        int max_iterations = 50,
        double tolerance = 1e-6) {
        
        /// Build kd-tree for target points
        nanoflann::KDTreeAdaptor<Eigen::MatrixBase<Derived2>, 3, nanoflann::metric_L2_Simple> kdtree(Y);
        
        /// Initialize transformation
        SimilarityTransform total_transform;
        Eigen::Matrix3Xd X_transformed = X.template cast<double>();
        
        double prev_error = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            /// Find closest points
            std::vector<int> indices(X.cols());
            std::vector<double> distances(X.cols());
            
            #pragma omp parallel for
            for (int i = 0; i < X.cols(); ++i) {
                std::vector<int> ret_index(1);
                std::vector<double> out_dist_sqr(1);
                kdtree.query(X_transformed.col(i).data(), 1, &ret_index[0], &out_dist_sqr[0]);
                indices[i] = ret_index[0];
                distances[i] = out_dist_sqr[0];
            }
            
            /// Extract corresponding points
            Eigen::Matrix3Xd Y_matched(3, X.cols());
            for (int i = 0; i < X.cols(); ++i) {
                Y_matched.col(i) = Y.col(indices[i]).template cast<double>();
            }
            
            /// Compute similarity transformation
            SimilarityTransform current_transform = point_to_point_similarity(X_transformed, Y_matched);
            
            /// Apply transformation
            X_transformed = current_transform.apply(X_transformed);
            
            /// Update total transformation
            SimilarityTransform new_total;
            new_total.scale = total_transform.scale * current_transform.scale;
            new_total.rotation = current_transform.rotation * total_transform.rotation;
            new_total.translation = current_transform.scale * current_transform.rotation * total_transform.translation + current_transform.translation;
            total_transform = new_total;
            
            /// Check convergence
            double current_error = 0.0;
            for (int i = 0; i < X.cols(); ++i) {
                current_error += distances[i];
            }
            current_error /= X.cols();
            
            if (std::abs(prev_error - current_error) < tolerance) {
                std::cout << "Similarity ICP converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }
            
            prev_error = current_error;
        }
        
        return total_transform;
    }
    
    /// Robust similarity ICP with outlier rejection
    /// @param Source points (one 3D point per column)
    /// @param Target points (one 3D point per column)
    /// @param max_iterations Maximum number of iterations
    /// @param tolerance Convergence tolerance
    /// @param outlier_threshold Distance threshold for outlier rejection
    template <typename Derived1, typename Derived2>
    SimilarityTransform robust_similarity_icp(
        Eigen::MatrixBase<Derived1>& X,
        Eigen::MatrixBase<Derived2>& Y,
        int max_iterations = 50,
        double tolerance = 1e-6,
        double outlier_threshold = 0.1) {
        
        /// Build kd-tree for target points
        nanoflann::KDTreeAdaptor<Eigen::MatrixBase<Derived2>, 3, nanoflann::metric_L2_Simple> kdtree(Y);
        
        /// Initialize transformation
        SimilarityTransform total_transform;
        Eigen::Matrix3Xd X_transformed = X.template cast<double>();
        
        double prev_error = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            /// Find closest points
            std::vector<int> indices;
            std::vector<double> distances;
            std::vector<int> valid_indices;
            
            for (int i = 0; i < X.cols(); ++i) {
                std::vector<int> ret_index(1);
                std::vector<double> out_dist_sqr(1);
                kdtree.query(X_transformed.col(i).data(), 1, &ret_index[0], &out_dist_sqr[0]);
                
                /// Outlier rejection
                if (sqrt(out_dist_sqr[0]) < outlier_threshold) {
                    indices.push_back(ret_index[0]);
                    distances.push_back(out_dist_sqr[0]);
                    valid_indices.push_back(i);
                }
            }
            
            if (valid_indices.size() < 3) {
                std::cout << "Warning: Too few valid correspondences (" << valid_indices.size() << ")" << std::endl;
                break;
            }
            
            /// Extract corresponding points (only valid ones)
            Eigen::Matrix3Xd X_valid(3, valid_indices.size());
            Eigen::Matrix3Xd Y_matched(3, valid_indices.size());
            
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                X_valid.col(i) = X_transformed.col(valid_indices[i]);
                Y_matched.col(i) = Y.col(indices[i]).template cast<double>();
            }
            
            /// Compute similarity transformation
            SimilarityTransform current_transform = point_to_point_similarity(X_valid, Y_matched);
            
            /// Apply transformation to all points
            X_transformed = current_transform.apply(X_transformed);
            
            /// Update total transformation
            SimilarityTransform new_total;
            new_total.scale = total_transform.scale * current_transform.scale;
            new_total.rotation = current_transform.rotation * total_transform.rotation;
            new_total.translation = current_transform.scale * current_transform.rotation * total_transform.translation + current_transform.translation;
            total_transform = new_total;
            
            /// Check convergence
            double current_error = 0.0;
            for (double dist : distances) {
                current_error += dist;
            }
            current_error /= distances.size();
            
            std::cout << "Iteration " << iter + 1 << ": Error = " << current_error 
                      << ", Valid points = " << valid_indices.size() 
                      << ", Scale = " << total_transform.scale << std::endl;
            
            if (std::abs(prev_error - current_error) < tolerance) {
                std::cout << "Robust Similarity ICP converged after " << iter + 1 << " iterations" << std::endl;
                break;
            }
            
            prev_error = current_error;
        }
        
        return total_transform;
    }
}

#endif // SIMILARITY_ICP_H