#include <iostream>
#include "ICP.h"
#include "io_pc.h"
#include "FRICP.h"

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
    enum Method{ICP, AA_ICP, FICP, RICP, PPL, RPPL, SparseICP, SICPPPL} method=RICP;
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
                  << "5: Our Robust ICP point to plane\n6: Sparse ICP\n7: Sparse ICP point to plane" << std::endl;
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

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = vertices_source.rowwise().maxCoeff() - vertices_source.rowwise().minCoeff();
    target_scale = vertices_target.rowwise().maxCoeff() - vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
    std::cout << "scale = " << scale << std::endl;
    vertices_source /= scale;
    vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    vertices_source.colwise() -= source_mean;
    vertices_target.colwise() -= target_mean;

    double time;
    // set ICP parameters
    ICP::Parameters pars;

    // set Sparse-ICP parameters
    SICP::Parameters spars;
    spars.p = 0.4;
    spars.print_icpn = false;

    /// Initial transformation
    if(use_init)
    {
        MatrixXX init_trans;
        read_transMat(init_trans, file_init);
        init_trans.block(0, dim, dim, 1) /= scale;
        init_trans.block(0,3,3,1) += init_trans.block(0,0,3,3)*source_mean - target_mean;
        pars.use_init = true;
        pars.init_trans = init_trans;
        spars.init_trans = init_trans;
    }

    ///--- Execute registration
    std::cout << "begin registration..." << std::endl;
    FRICP<3> fricp;
    double begin_reg = omp_get_wtime();
    double converge_rmse = 0;
    switch(method)
    {
    case ICP:
    {
        pars.f = ICP::NONE;
        pars.use_AA = false;
        fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case AA_ICP:
    {
        AAICP::point_to_point_aaicp(vertices_source, vertices_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case FICP:
    {
        pars.f = ICP::NONE;
        pars.use_AA = true;
        fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case RICP:
    {
        pars.f = ICP::WELSCH;
        pars.use_AA = true;
        fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case PPL:
    {
        pars.f = ICP::NONE;
        pars.use_AA = false;
        fricp.point_to_plane(vertices_source, vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case RPPL:
    {
        pars.nu_end_k = 1.0/6;
        pars.f = ICP::WELSCH;
        pars.use_AA = true;
        fricp.point_to_plane_GN(vertices_source, vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
        res_trans = pars.res_trans;
        break;
    }
    case SparseICP:
    {
        SICP::point_to_point(vertices_source, vertices_target, source_mean, target_mean, spars);
        res_trans = spars.res_trans;
        break;
    }
    case SICPPPL:
    {
        SICP::point_to_plane(vertices_source, vertices_target, normal_target, source_mean, target_mean, spars);
        res_trans = spars.res_trans;
        break;
    }
    }
	std::cout << "Registration done!" << std::endl;
    double end_reg = omp_get_wtime();
    time = end_reg - begin_reg;
    vertices_source = scale * vertices_source;

    out_path = out_path + "m" + std::to_string(method);
    Eigen::Affine3d res_T;
    res_T.linear() = res_trans.block(0,0,3,3);
    res_T.translation() = res_trans.block(0,3,3,1);
    res_trans_path = out_path + "trans.txt";
    std::ofstream out_trans(res_trans_path);
    res_trans.block(0,3,3,1) *= scale;
    out_trans << res_trans << std::endl;
    out_trans.close();

    ///--- Write result to file
    std::string file_source_reg = out_path + "reg_pc.ply";
    write_file(file_source, vertices_source, normal_source, src_vert_colors, file_source_reg);

    return 0;
}
