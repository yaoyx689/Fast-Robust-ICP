#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <random>

// 简化的相似变换结构
struct SimilarityTransform {
    double scale;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    
    SimilarityTransform() : scale(1.0), rotation(Eigen::Matrix3d::Identity()), 
                           translation(Eigen::Vector3d::Zero()) {}
    
    Eigen::Matrix4d toMatrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = scale * rotation;
        T.block<3,1>(0,3) = translation;
        return T;
    }
    
    Eigen::Matrix3Xd apply(const Eigen::Matrix3Xd& points) const {
        return scale * rotation * points + translation.replicate(1, points.cols());
    }
};

// 求解相似变换的函数
SimilarityTransform solveSimilarityTransform(const Eigen::Matrix3Xd& X, const Eigen::Matrix3Xd& Y) {
    // 计算质心
    Eigen::Vector3d X_mean = X.rowwise().mean();
    Eigen::Vector3d Y_mean = Y.rowwise().mean();
    
    // 中心化
    Eigen::Matrix3Xd X_centered = X.colwise() - X_mean;
    Eigen::Matrix3Xd Y_centered = Y.colwise() - Y_mean;
    
    // 计算协方差矩阵
    Eigen::Matrix3d H = X_centered * Y_centered.transpose();
    
    // SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // 计算旋转矩阵
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0) {
        V.col(2) *= -1;
        R = V * U.transpose();
    }
    
    // 计算尺度
    double sigma_X = X_centered.array().square().sum();
    double trace_H = svd.singularValues().sum();
    double scale = trace_H / sigma_X;
    
    // 计算平移
    Eigen::Vector3d t = Y_mean - scale * R * X_mean;
    
    SimilarityTransform result;
    result.scale = scale;
    result.rotation = R;
    result.translation = t;
    
    return result;
}

// 生成测试数据
void generateTestData(Eigen::Matrix3Xd& source, Eigen::Matrix3Xd& target, SimilarityTransform& ground_truth) {
    int n_points = 100;
    
    // 生成随机源点云
    source = Eigen::Matrix3Xd::Random(3, n_points);
    
    // 设置真实的相似变换
    ground_truth.scale = 2.5;
    ground_truth.rotation = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    ground_truth.translation = Eigen::Vector3d(1.0, 2.0, 0.5);
    
    // 应用变换生成目标点云
    target = ground_truth.apply(source);
    
    // 添加少量噪声
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.01);
    
    for (int i = 0; i < target.cols(); ++i) {
        target.col(i) += Eigen::Vector3d(noise(gen), noise(gen), noise(gen));
    }
}

int main() {
    std::cout << "=== 相似变换ICP测试 ===" << std::endl;
    
    // 生成测试数据
    Eigen::Matrix3Xd source, target;
    SimilarityTransform ground_truth;
    generateTestData(source, target, ground_truth);
    
    std::cout << "生成了 " << source.cols() << " 个测试点" << std::endl;
    std::cout << "真实变换参数:" << std::endl;
    std::cout << "  尺度: " << ground_truth.scale << std::endl;
    std::cout << "  旋转矩阵:\n" << ground_truth.rotation << std::endl;
    std::cout << "  平移向量: " << ground_truth.translation.transpose() << std::endl;
    
    // 求解相似变换
    SimilarityTransform estimated = solveSimilarityTransform(source, target);
    
    std::cout << "\n估计的变换参数:" << std::endl;
    std::cout << "  尺度: " << estimated.scale << std::endl;
    std::cout << "  旋转矩阵:\n" << estimated.rotation << std::endl;
    std::cout << "  平移向量: " << estimated.translation.transpose() << std::endl;
    
    // 计算误差
    double scale_error = std::abs(estimated.scale - ground_truth.scale);
    double rotation_error = (estimated.rotation - ground_truth.rotation).norm();
    double translation_error = (estimated.translation - ground_truth.translation).norm();
    
    std::cout << "\n误差分析:" << std::endl;
    std::cout << "  尺度误差: " << scale_error << std::endl;
    std::cout << "  旋转误差: " << rotation_error << std::endl;
    std::cout << "  平移误差: " << translation_error << std::endl;
    
    // 应用估计的变换
    Eigen::Matrix3Xd transformed = estimated.apply(source);
    
    // 计算配准误差
    double rmse = 0.0;
    for (int i = 0; i < source.cols(); ++i) {
        rmse += (transformed.col(i) - target.col(i)).squaredNorm();
    }
    rmse = std::sqrt(rmse / source.cols());
    
    std::cout << "  配准RMSE: " << rmse << std::endl;
    
    // 输出变换矩阵
    std::cout << "\n4x4变换矩阵:" << std::endl;
    std::cout << estimated.toMatrix() << std::endl;
    
    return 0;
}