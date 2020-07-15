# Fast-Robust-ICP

This repository includes the source code the paper [Fast and Robust Iterative Closet Point]().

Authors: [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/), Yuxin Yao, [Bailin Deng](http://www.bdeng.me/).

This code is protected under patent. It can be only used for research purposes. If you are interested in business purposes/for-profit use, please contact Juyong Zhang (the author, email: juyong@ustc.edu.cn).


## Compilation

The code is compiled using [CMake](https://cmake.org/) and requires Eigen. It has been tested on Ubuntu 16.04 with gcc 5.4.0 and on Windows with Visual Studio 2015. An executable `FRICP` will be generated.


## Usage

The program is run with four input parameters:

1. an input file storing the source point cloud;
2. an input file storing the target point cloud;
3. an output path storing the registered source point cloud and transformation;
4. registration method:
```
0: ICP
1: AA-ICP
2: Ours (Fast ICP)
3: Ours (Robust ICP)
4: ICP Point-to-plane
5: Our (Robust ICP point-to-plane)
6: Sparse ICP
7: Sparse ICP point to plane
```
You can ignore the last parameter, in which case our Robust ICP will be used by default. If you have an initial transformation to help registration, you can set `use_init=true` and the `file_init` is the initial file name in `main.cpp` . Both input and output transformations are 4x4 transformation matrix. 

Example:

```
$ FRICP ./data/target.ply ./data/source.ply ./data/res/ 3
```

But obj and ply files are supported.

## Acknowledgements
The code is adapted from the [Sparse ICP implementation](https://github.com/OpenGP/sparseicp) released by the authors.
