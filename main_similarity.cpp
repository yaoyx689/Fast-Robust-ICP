#include <iostream>
#include "ICP.h"
#include "io_pc.h"
#include "FRICP.h"
#include "SimilarityICP.h"

int main(int argc, char const ** argv)
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<Scalar, 3, 1> VectorN;
    std::string file_source;
    std::string file_target;
    std::string file_init = "./data/";
    std::string res_trans_path;
    std::string out_path;
    bool use_init = false;
    MatrixXX res_trans;
    
    enum Method{
        ICP, AA_ICP, FICP, RICP, PPL, RPPL, SparseICP, SICPPPL,
        SIM_ICP, ROBUST_SIM_ICP  // 新增相似变换方法
    } method = SIM_ICP;  // 默认使用相似变换ICP
    
    if(argc == 5)
    {
        file_target = argv[1];
        file_source = argv[2];
        out_path = argv[3];
        method = Method(std::stoi(argv[4]));
    }
    else if(argc==4)
    {
        file_target = argv[1];
        file_source = argv[2];
        out_path = argv[3];
    }
    else
    {
        std::cout << "Usage: target.ply source.ply out_path <Method>" << std::endl;
        std::cout << "Method :\n"
                  << "0: ICP\n1: AA-ICP\n2: Our Fast ICP\n3: Our Robust ICP\n4: ICP Point-to-plane\n"
                  << "5: Our Robust ICP point to plane\n6: Sparse ICP\n7: Sparse ICP point to plane\n"
                  << "8: Similarity ICP\n9: Robust Similarity ICP" << std::endl;
        exit(0);
    }
    int dim = 3;

    //--- Model that will be rigidly transformed
    Vertices vertices_source, normal_source, src_vert_colors;
    read_file(vertices_source, normal_source, src_vert_colors, file_source);
    std::cout << "source: " << vertices_source.rows() << "x" << vertices_source.cols() << std::endl;

    //--- Model that source will be aligned to
    Vertices vertices_target, normal_target, tar_vert_colors;
    read_file(vertices_target, normal_target, tar_vert_colors, file_target);
    std::cout << "target: " << vertices_target.rows() << "x" << vertices_target.cols() << std::endl;

    //--- 计算缩放因子（用于归一化）
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    
    // 计算点云的尺度
    double source_scale = 0.0, target_scale = 0.0;
    for(int i = 0; i < vertices_source.cols(); i++) {
        source_scale += (vertices_source.col(i) - source_mean).norm();
    }
    for(int i = 0; i < vertices_target.cols(); i++) {
        target_scale += (vertices_target.col(i) - target_mean).norm();
    }
    source_scale /= vertices_source.cols();
    target_scale /= vertices_target.cols();
    
    double scale = std::max(source_scale, target_scale);
    std::cout << "scale = " << scale << std::endl;

    // 中心化和归一化
    if(!use_init) {
        vertices_source.colwise() -= source_mean;
        vertices_target.colwise() -= target_mean;
        vertices_source /= scale;
        vertices_target /= scale;
    }

    std::cout << "begin registration..." << std::endl;
    
    // 根据方法选择执行相应的算法
    if(method == SIM_ICP || method == ROBUST_SIM_ICP) {
        // 使用相似变换ICP
        SimilarityICP::SimilarityTransform sim_transform;
        
        if(method == SIM_ICP) {
            sim_transform = SimilarityICP::similarity_icp(vertices_source, vertices_target, 50, 1e-6);
        } else {
            sim_transform = SimilarityICP::robust_similarity_icp(vertices_source, vertices_target, 50, 1e-6, 0.1);
        }
        
        // 输出相似变换信息
        std::cout << "Similarity transformation:" << std::endl;
        std::cout << "Scale: " << sim_transform.scale << std::endl;
        std::cout << "Rotation matrix:" << std::endl << sim_transform.rotation << std::endl;
        std::cout << "Translation: " << sim_transform.translation.transpose() << std::endl;
        
        // 转换为4x4矩阵
        res_trans = sim_transform.toMatrix();
        
        // 应用变换到源点云
        vertices_source = sim_transform.apply(vertices_source);
        
        // 反归一化
        if(!use_init) {
            vertices_source *= scale;
            vertices_source.colwise() += target_mean;
            
            // 调整变换矩阵以适应反归一化
            Eigen::Matrix4d denorm_trans = Eigen::Matrix4d::Identity();
            denorm_trans.block<3,1>(0,3) = target_mean;
            denorm_trans.block<3,3>(0,0) *= scale;
            
            Eigen::Matrix4d norm_trans = Eigen::Matrix4d::Identity();
            norm_trans.block<3,1>(0,3) = -source_mean;
            norm_trans.block<3,3>(0,0) /= scale;
            
            res_trans = denorm_trans * res_trans * norm_trans;
        }
    }
    else {
        // 使用原有的刚体变换算法
        FRICP<3> fricp;
        ICP::Parameters pars;
        
        // 根据方法设置参数
        switch(method) {
            case ICP:
                pars.f = ICP::NONE;
                pars.use_AA = false;
                break;
            case AA_ICP:
                pars.f = ICP::NONE;
                pars.use_AA = true;
                break;
            case FICP:
                pars.f = ICP::NONE;
                pars.use_AA = true;
                break;
            case RICP:
                pars.f = ICP::WELSCH;
                pars.use_AA = true;
                break;
            default:
                pars.f = ICP::WELSCH;
                pars.use_AA = true;
                break;
        }
        
        fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        
        // 反归一化
        if(!use_init) {
            vertices_source *= scale;
            vertices_source.colwise() += target_mean;
            
            // 调整变换矩阵
            Eigen::Matrix4d denorm_trans = Eigen::Matrix4d::Identity();
            denorm_trans.block<3,1>(0,3) = target_mean;
            denorm_trans.block<3,3>(0,0) *= scale;
            
            Eigen::Matrix4d norm_trans = Eigen::Matrix4d::Identity();
            norm_trans.block<3,1>(0,3) = -source_mean;
            norm_trans.block<3,3>(0,0) /= scale;
            
            res_trans = denorm_trans * res_trans * norm_trans;
        }
    }

    std::cout << "Registration done!" << std::endl;

    // 保存结果
    std::string file_out = out_path + "m" + std::to_string(method) + "reg_pc.ply";
    std::string file_trans = out_path + "m" + std::to_string(method) + "trans.txt";
    
    write_file(file_out, vertices_source, normal_source, src_vert_colors, "ply");
    
    // 保存变换矩阵
    std::ofstream outfile(file_trans);
    if (outfile.is_open()) {
        outfile << res_trans << std::endl;
        outfile.close();
    }

    return 0;
}