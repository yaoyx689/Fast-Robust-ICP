#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <random>

// 简化的PLY读取函数
bool readPLY(const std::string& filename, Eigen::Matrix3Xd& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "无法打开文件: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int vertex_count = 0;
    bool in_header = true;
    
    // 读取头部信息
    while (std::getline(file, line) && in_header) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertex_count);
        }
        if (line == "end_header") {
            in_header = false;
        }
    }
    
    if (vertex_count == 0) {
        std::cout << "未找到顶点信息" << std::endl;
        return false;
    }
    
    points.resize(3, vertex_count);
    
    // 读取顶点数据
    for (int i = 0; i < vertex_count; ++i) {
        if (!std::getline(file, line)) break;
        
        float x, y, z;
        if (sscanf(line.c_str(), "%f %f %f", &x, &y, &z) == 3) {
            points(0, i) = x;
            points(1, i) = y;
            points(2, i) = z;
        }
    }
    
    file.close();
    std::cout << "成功读取 " << vertex_count << " 个点" << std::endl;
    return true;
}

// 相似变换结构
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

// 求解相似变换
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

// 简单的最近邻搜索
std::vector<int> findNearestNeighbors(const Eigen::Matrix3Xd& source, const Eigen::Matrix3Xd& target) {
    std::vector<int> indices(source.cols());
    
    for (int i = 0; i < source.cols(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_idx = 0;
        
        for (int j = 0; j < target.cols(); ++j) {
            double dist = (source.col(i) - target.col(j)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        indices[i] = best_idx;
    }
    
    return indices;
}

// 相似变换ICP
SimilarityTransform similarityICP(const Eigen::Matrix3Xd& source, const Eigen::Matrix3Xd& target, 
                                 int max_iterations = 20, double tolerance = 1e-6) {
    
    SimilarityTransform total_transform;
    Eigen::Matrix3Xd current_source = source;
    
    double prev_error = std::numeric_limits<double>::max();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 找最近邻
        std::vector<int> indices = findNearestNeighbors(current_source, target);
        
        // 提取对应点
        Eigen::Matrix3Xd target_matched(3, current_source.cols());
        for (int i = 0; i < current_source.cols(); ++i) {
            target_matched.col(i) = target.col(indices[i]);
        }
        
        // 求解相似变换
        SimilarityTransform current_transform = solveSimilarityTransform(current_source, target_matched);
        
        // 应用变换
        current_source = current_transform.apply(current_source);
        
        // 更新总变换
        SimilarityTransform new_total;
        new_total.scale = total_transform.scale * current_transform.scale;
        new_total.rotation = current_transform.rotation * total_transform.rotation;
        new_total.translation = current_transform.scale * current_transform.rotation * 
                               total_transform.translation + current_transform.translation;
        total_transform = new_total;
        
        // 计算误差
        double current_error = 0.0;
        for (int i = 0; i < current_source.cols(); ++i) {
            current_error += (current_source.col(i) - target_matched.col(i)).squaredNorm();
        }
        current_error = std::sqrt(current_error / current_source.cols());
        
        std::cout << "迭代 " << iter + 1 << ": RMSE = " << current_error 
                  << ", 尺度 = " << total_transform.scale << std::endl;
        
        // 检查收敛
        if (std::abs(prev_error - current_error) < tolerance) {
            std::cout << "算法在第 " << iter + 1 << " 次迭代后收敛" << std::endl;
            break;
        }
        
        prev_error = current_error;
    }
    
    return total_transform;
}

int main() {
    std::cout << "=== 真实点云相似变换ICP测试 ===" << std::endl;
    
    // 读取点云数据
    Eigen::Matrix3Xd source, target;
    
    if (!readPLY("./data/source.ply", source)) {
        std::cout << "无法读取源点云" << std::endl;
        return -1;
    }
    
    if (!readPLY("./data/target.ply", target)) {
        std::cout << "无法读取目标点云" << std::endl;
        return -1;
    }
    
    // 为了测试相似变换，我们人工创建一个缩放版本的源点云
    std::cout << "\n=== 创建测试场景 ===" << std::endl;
    
    // 保存原始源点云
    Eigen::Matrix3Xd original_source = source;
    
    // 创建一个已知的相似变换
    SimilarityTransform known_transform;
    known_transform.scale = 1.5;  // 1.5倍缩放
    known_transform.rotation = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    known_transform.translation = Eigen::Vector3d(0.1, 0.2, 0.05);
    
    std::cout << "应用已知相似变换:" << std::endl;
    std::cout << "  尺度: " << known_transform.scale << std::endl;
    std::cout << "  旋转角度: 0.3 弧度 (绕Z轴)" << std::endl;
    std::cout << "  平移: " << known_transform.translation.transpose() << std::endl;
    
    // 对源点云应用相似变换
    source = known_transform.apply(source);
    
    // 添加少量噪声
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.001);
    
    for (int i = 0; i < source.cols(); ++i) {
        source.col(i) += Eigen::Vector3d(noise(gen), noise(gen), noise(gen));
    }
    
    std::cout << "已添加噪声到变换后的源点云" << std::endl;
    
    // 为了加速测试，只使用部分点
    int max_points = 1000;
    if (source.cols() > max_points) {
        source.conservativeResize(3, max_points);
        std::cout << "为加速测试，只使用前 " << max_points << " 个源点" << std::endl;
    }
    if (target.cols() > max_points) {
        target.conservativeResize(3, max_points);
        std::cout << "为加速测试，只使用前 " << max_points << " 个目标点" << std::endl;
    }
    
    // 执行相似变换ICP
    std::cout << "\n=== 执行相似变换ICP ===" << std::endl;
    SimilarityTransform estimated = similarityICP(source, target, 20, 1e-6);
    
    std::cout << "\n=== 结果分析 ===" << std::endl;
    std::cout << "估计的变换参数:" << std::endl;
    std::cout << "  尺度: " << estimated.scale << std::endl;
    std::cout << "  旋转矩阵:\n" << estimated.rotation << std::endl;
    std::cout << "  平移向量: " << estimated.translation.transpose() << std::endl;
    
    // 计算误差（相对于已知变换）
    double scale_error = std::abs(estimated.scale - known_transform.scale);
    double rotation_error = (estimated.rotation - known_transform.rotation).norm();
    double translation_error = (estimated.translation - known_transform.translation).norm();
    
    std::cout << "\n相对于已知变换的误差:" << std::endl;
    std::cout << "  尺度误差: " << scale_error << std::endl;
    std::cout << "  旋转误差: " << rotation_error << std::endl;
    std::cout << "  平移误差: " << translation_error << std::endl;
    
    // 输出变换矩阵
    std::cout << "\n估计的4x4变换矩阵:" << std::endl;
    std::cout << estimated.toMatrix() << std::endl;
    
    std::cout << "\n已知的4x4变换矩阵:" << std::endl;
    std::cout << known_transform.toMatrix() << std::endl;
    
    return 0;
}