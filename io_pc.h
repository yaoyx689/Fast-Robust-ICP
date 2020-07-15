#ifndef IO_H
#define IO_H
///--- hacked from OpenGP obj reader
#include <cstdio>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

template <class MatrixType>
bool read_obj(MatrixType& vertices, MatrixType& normals, MatrixType& vert_colors, const std::string& filename, int D) {
    char   s[200];
    float  x, y, z, cx, cy, cz;

    // open file (in ASCII mode)
    FILE* in = fopen(filename.c_str(), "r");
    if (!in) return false;

    // clear line once
    memset(&s, 0, 200);

    //--- First pass, counts vertices
    int n_vertices = 0;
    while (in && !feof(in) && fgets(s, 200, in)) {
        // comment
        if (s[0] == '#' || isspace(s[0])) continue;
        // vertex
        else if (strncmp(s, "v ", 2) == 0)
            n_vertices++;
    }
    fseek(in, 0, 0); ///< rewind
    vertices.resize(D, n_vertices);
    normals.resize(D, n_vertices);
    vert_colors.resize(D, n_vertices);
    bool runonce = true;

    //--- Second pass, fills in
    int curr_vertex=0;
    while (in && !feof(in) && fgets(s, 200, in)) {
        // comment
        if (s[0] == '#' || isspace(s[0])) continue;

        // normal
        else if (strncmp(s, "vn ", 3) == 0) {
            if (sscanf(s, "vn %f %f %f", &x, &y, &z)) {
                if (runonce)
                {
                    normals.resize(D, n_vertices);
                    runonce = false;
                }
                normals(0, curr_vertex) = x;
                normals(1, curr_vertex) = y;
                normals(2, curr_vertex) = z;
            }
        }

        // vertex
        else if (strncmp(s, "v ", 2) == 0) {
            if (sscanf(s, "v %f %f %f %f %f %f", &x, &y, &z, &cx, &cy, &cz)) {
                            vertices(0,curr_vertex) = x;
                            vertices(1,curr_vertex) = y;
                            vertices(2,curr_vertex) = z;

                            vert_colors(0, curr_vertex) = cx;
                            vert_colors(1, curr_vertex) = cy;
                            vert_colors(2, curr_vertex) = cz;
                            curr_vertex++;
                        }
            else if (sscanf(s, "v %f %f %f", &x, &y, &z)) {
                vertices(0,curr_vertex) = x;
                vertices(1,curr_vertex) = y;
                vertices(2,curr_vertex) = z;

                curr_vertex++;
            }
        }
        // face
        else if (strncmp(s, "f ", 2) == 0) {
            continue;
        }

        // clear line
        memset(&s, 0, 200);
    }

    fclose(in);
    return true;
}
//-----------------------------------------------------------------------------

///--- Replaces vertices in prev_filename with content of vertices, saves in filename
template <class MatrixType>
bool write_obj_replaceverts(const std::string& prev_filename, const MatrixType& vertices, const MatrixType& normals,
                            const MatrixType& vert_colors, const std::string& filename) {
    typedef Eigen::Vector3d Texture_coordinate;

    char   s[200];

    FILE* out = fopen(filename.c_str(), "w");
    FILE* in = fopen(prev_filename.c_str(), "r");
    if (!in || !out)
        return false;

    // clear line once
    memset(&s, 0, 200);

    //--- Second pass, fills in
    int curr_vertex=0;
    while (in && !feof(in) && fgets(s, 200, in)) {
        // vertex
        if (!isspace(s[0]) && strncmp(s, "v ", 2) == 0) {
            fprintf(out, "vn %f %f %f\n", normals(0,curr_vertex), normals(1,curr_vertex), normals(2,curr_vertex));
            fprintf(out, "v %f %f %f ", vertices(0,curr_vertex), vertices(1,curr_vertex), vertices(2,curr_vertex));
            if(vert_colors.size())
                fprintf(out, "%f %f %f\n", vert_colors(0,curr_vertex), vert_colors(1, curr_vertex), vert_colors(2, curr_vertex));
            else
                fprintf(out, "\n");
            curr_vertex++;
        } else {
            fprintf(out, "%s", s);
        }

        // clear line
        memset(&s, 0, 200);
    }


    fclose(in);
    fclose(out);
    return true;
}


template <class MatrixType>
bool read_transMat(MatrixType& trans, const std::string& filename)
{
	std::ifstream input(filename);
	std::string line;
	int rows, cols;
	std::vector<std::vector<double>> total_data;
	while (getline(input, line)) {
        if(line[0] == 'V' || line[0] == 'M')
            continue;
		std::istringstream iss(line);
		std::vector<double> lineVec;
		while (iss) {
			double item;
			if (iss >> item)
				lineVec.push_back(item);
		}
		cols = lineVec.size();
		total_data.push_back(lineVec);
	}
	if (total_data.size() == 0)
	{
		std::cout << filename << " is empty !! " << std::endl;
		return false;
	}
	rows = total_data.size();
	trans.resize(rows, cols);
    std::cout << "rows = " << rows << " cols = " << cols << std::endl;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			trans(i, j) = total_data[i][j];
		}
	}
	input.close();
    std::cout << "read trans = \n" << trans << std::endl;
	return true;
}

template <class MatrixType>
bool read_ply(MatrixType& vertices, MatrixType& normals, MatrixType& colors, const std::string& filename, int D) {
    char   s[200];
    float  x, y, z, nx, ny, nz;
    int r, g, b;
    int dim = 0;
    int n_vertices, curr_vertex;
    Eigen::Vector3i ID;
    ID.setZero();

    // open file (in ASCII mode)
    FILE* in = fopen(filename.c_str(), "r");
    if (!in) return false;

    // clear line once
    memset(&s, 0, 200);

    //--- First pass, counts vertices
    while (in && !feof(in) && fgets(s, 200, in)) {

        // comment
        if (strncmp(s, "element ", 8) == 0)
        {
            sscanf(s, "element vertex %d", &n_vertices);
        }
        if (strncmp(s, "property float x", 16) == 0)
        {
            vertices.resize(D,n_vertices);
            ID[0]=1;
            dim += 3;
        }
        if(strncmp(s, "property float nx", 17) == 0)
        {
            normals.resize(D,n_vertices);
            ID[1]=1;
            dim += 3;
        }
        if(strncmp(s, "property uchar red", 18) == 0)
        {
            colors.resize(3,n_vertices);
            ID[2]=1;
            dim += 3;
        }
        if(strncmp(s, "end_header", 10) == 0)
        {
            break;
        }
        memset(&s, 0, 200);
    }
    // clear line
    memset(&s, 0, 200);
    curr_vertex = 0;
    while (in && !feof(in) && fgets(s, 200, in) && curr_vertex<n_vertices)
    {
        if(dim == 3)
        {
            if(sscanf(s, "%f %f %f",&x, &y, &z))
            {
                vertices(0,curr_vertex) = x;
                vertices(1,curr_vertex) = y;
                vertices(2,curr_vertex) = z;
                curr_vertex++;
            }
        }
        else if(dim==6)
        {
            if(ID[1])
            {
                if(sscanf(s, "%f %f %f %f %f %f",&x, &y, &z, &nx, &ny, &nz))
                {
                    vertices(0,curr_vertex) = x;
                    vertices(1,curr_vertex) = y;
                    vertices(2,curr_vertex) = z;
                    normals(0, curr_vertex) = nx;
                    normals(1, curr_vertex) = ny;
                    normals(2, curr_vertex) = nz;
                    curr_vertex++;
                }
            }
            else
            {
                 if(sscanf(s, "%f %f %f %d %d %d",&x, &y, &z, &r, &g, &b))
                 {
                     vertices(0,curr_vertex) = x;
                     vertices(1,curr_vertex) = y;
                     vertices(2,curr_vertex) = z;
                     colors(0, curr_vertex) = r;
                     colors(1, curr_vertex) = g;
                     colors(2, curr_vertex) = b;
                     curr_vertex++;
                 }
            }

        }
        else if(dim == 9)
        {
            if(sscanf(s, "%f %f %f %f %f %f %d %d %d", &x, &y, &z, &nx, &ny, &nz, &r, &g, &b))
            {
                vertices(0,curr_vertex) = x;
                vertices(1,curr_vertex) = y;
                vertices(2,curr_vertex) = z;
                normals(0, curr_vertex) = nx;
                normals(1, curr_vertex) = ny;
                normals(2, curr_vertex) = nz;
                colors(0, curr_vertex) = r;
                colors(1, curr_vertex) = g;
                colors(2, curr_vertex) = b;
                curr_vertex ++;
            }

        }
        if(curr_vertex > n_vertices)
        {
            n_vertices = curr_vertex;
            vertices.resize(Eigen::NoChange, n_vertices);
            if(normals.size())
            {
                normals.resize(Eigen::NoChange, n_vertices);
            }
            break;
        }
        // clear line
        memset(&s, 0, 200);
    }
    fclose(in);
    return true;
}

template <class MatrixType>
bool write_ply(MatrixType& vertices, MatrixType& normals, MatrixType& colors, const std::string& filename) {
    char   s[200];
    int n_vertices, curr_vertex;
    n_vertices = vertices.cols();
    Eigen::Vector3d ID; // whether there are normal or color
    ID.setZero();
    if (vertices.cols())
    {
        ID[0] = 1;
    }
    else
    {
        std::cout << "Warning : No points!!!" << std::endl;
        return false;
    }
    if (normals.cols())
    {
        ID[1] = 1;
//        std::cout << "output file has normals !" << std::endl;
    }
    if (colors.cols())
    {
        ID[2] = 1;
//        std::cout << "output file has colors !" << std::endl;
    }

    FILE* out = fopen(filename.c_str(), "w");
    if (!out)
        return false;
    // clear line once
    memset(&s, 0, 200);

    fprintf(out, "ply\nformat ascii 1.0\nelement vertex %d\n", n_vertices);
    fprintf(out, "property float x \nproperty float y\nproperty float z\n");
    if(ID[1])	fprintf(out, "property float nx \nproperty float ny\nproperty float nz\n");
    if(ID[2])	fprintf(out, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
    fprintf(out, "end_header\n");

    // clear line
    memset(&s, 0, 200);
    curr_vertex = 0;
    while (curr_vertex<n_vertices)
    {
        fprintf(out, "%f %f %f ", vertices(0, curr_vertex), vertices(1, curr_vertex), vertices(2, curr_vertex));
        if (ID[1])	fprintf(out, "%f %f %f ", normals(0, curr_vertex), normals(1, curr_vertex), normals(2, curr_vertex));
        if (ID[2])	fprintf(out, "%d %d %d ", (int)colors(0, curr_vertex), (int)colors(1, curr_vertex), (int)colors(2, curr_vertex));
        fprintf(out, "\n");
        // clear line
        memset(&s, 0, 200);
        curr_vertex++;
    }
    fclose(out);
    return true;
}
template <class MatrixType>
bool read_file(MatrixType& vertices, MatrixType& normals, MatrixType& vert_colors,
               const std::string& filename) {
    if(strcmp(filename.substr(filename.size()-4,4).c_str(), ".obj") == 0)
    {
        return read_obj(vertices, normals, vert_colors, filename, 3);
    }
    else if(strcmp(filename.substr(filename.size()-4, 4).c_str(),".ply")==0)
    {
        return read_ply(vertices, normals, vert_colors,  filename, 3);
    }
    else
    {
        std::cout << "Can't read file " << filename << std::endl;
    }
}

template <class MatrixType>
bool write_file(const std::string& prev_filename, const MatrixType& vertices, const MatrixType& normals,
                const MatrixType& vert_colors, const std::string& filename) {
    if(strcmp(filename.substr(filename.size()-4,4).c_str(),".obj")==0)
    {
        return write_obj_replaceverts(prev_filename, vertices, normals, vert_colors, filename);
    }
    else if(strcmp(filename.substr(filename.size()-4,4).c_str(),".ply")==0)
    {
        return write_ply(vertices, normals, vert_colors, filename);
    }
    else
    {
        std::cout << "Can't write to file "<< filename << std::endl;
    }
}

#endif // IO_H
