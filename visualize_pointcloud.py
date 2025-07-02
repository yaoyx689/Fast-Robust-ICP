#!/usr/bin/env python3
"""
点云可视化脚本
用于可视化Fast-Robust-ICP项目的输入和输出点云数据
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def read_ply_file(filename):
    """读取PLY格式的点云文件"""
    vertices = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # 查找header结束位置
    header_end = 0
    vertex_count = 0
    
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('end_header'):
            header_end = i + 1
            break
    
    # 读取顶点数据
    for i in range(header_end, header_end + vertex_count):
        if i < len(lines):
            coords = lines[i].strip().split()
            if len(coords) >= 3:
                try:
                    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                    vertices.append([x, y, z])
                except ValueError:
                    continue
    
    return np.array(vertices)

def visualize_pointclouds(source_file, target_file, registered_file=None):
    """可视化点云数据"""
    
    # 读取点云数据
    print(f"读取源点云: {source_file}")
    source_points = read_ply_file(source_file)
    print(f"源点云包含 {len(source_points)} 个点")
    
    print(f"读取目标点云: {target_file}")
    target_points = read_ply_file(target_file)
    print(f"目标点云包含 {len(target_points)} 个点")
    
    registered_points = None
    if registered_file and os.path.exists(registered_file):
        print(f"读取配准后点云: {registered_file}")
        registered_points = read_ply_file(registered_file)
        print(f"配准后点云包含 {len(registered_points)} 个点")
    
    # 创建可视化
    if registered_points is not None:
        fig = plt.figure(figsize=(18, 6))
        
        # 配准前对比
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(source_points[::10, 0], source_points[::10, 1], source_points[::10, 2], 
                   c='red', s=1, alpha=0.6, label='源点云')
        ax1.scatter(target_points[::10, 0], target_points[::10, 1], target_points[::10, 2], 
                   c='blue', s=1, alpha=0.6, label='目标点云')
        ax1.set_title('配准前')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 配准后对比
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(registered_points[::10, 0], registered_points[::10, 1], registered_points[::10, 2], 
                   c='red', s=1, alpha=0.6, label='配准后源点云')
        ax2.scatter(target_points[::10, 0], target_points[::10, 1], target_points[::10, 2], 
                   c='blue', s=1, alpha=0.6, label='目标点云')
        ax2.set_title('配准后')
        ax2.legend()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # 重叠显示
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(registered_points[::10, 0], registered_points[::10, 1], registered_points[::10, 2], 
                   c='green', s=1, alpha=0.8, label='配准后重叠')
        ax3.set_title('配准结果')
        ax3.legend()
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
    else:
        fig = plt.figure(figsize=(12, 6))
        
        # 源点云和目标点云对比
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(source_points[::10, 0], source_points[::10, 1], source_points[::10, 2], 
                   c='red', s=1, alpha=0.6, label='源点云')
        ax1.scatter(target_points[::10, 0], target_points[::10, 1], target_points[::10, 2], 
                   c='blue', s=1, alpha=0.6, label='目标点云')
        ax1.set_title('原始点云对比')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 分别显示
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(source_points[::20, 0], source_points[::20, 1], source_points[::20, 2], 
                   c='red', s=2, alpha=0.8, label='源点云')
        ax2.set_title('源点云')
        ax2.legend()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('/workspace/Fast-Robust-ICP/pointcloud_visualization.png', dpi=150, bbox_inches='tight')
    print("可视化结果已保存到: pointcloud_visualization.png")
    
    # 打印统计信息
    print("\n=== 点云统计信息 ===")
    print(f"源点云范围: X[{source_points[:, 0].min():.3f}, {source_points[:, 0].max():.3f}], "
          f"Y[{source_points[:, 1].min():.3f}, {source_points[:, 1].max():.3f}], "
          f"Z[{source_points[:, 2].min():.3f}, {source_points[:, 2].max():.3f}]")
    
    print(f"目标点云范围: X[{target_points[:, 0].min():.3f}, {target_points[:, 0].max():.3f}], "
          f"Y[{target_points[:, 1].min():.3f}, {target_points[:, 1].max():.3f}], "
          f"Z[{target_points[:, 2].min():.3f}, {target_points[:, 2].max():.3f}]")
    
    if registered_points is not None:
        print(f"配准后点云范围: X[{registered_points[:, 0].min():.3f}, {registered_points[:, 0].max():.3f}], "
              f"Y[{registered_points[:, 1].min():.3f}, {registered_points[:, 1].max():.3f}], "
              f"Z[{registered_points[:, 2].min():.3f}, {registered_points[:, 2].max():.3f}]")

def main():
    parser = argparse.ArgumentParser(description='可视化Fast-Robust-ICP点云数据')
    parser.add_argument('--source', default='data/source.ply', help='源点云文件路径')
    parser.add_argument('--target', default='data/target.ply', help='目标点云文件路径')
    parser.add_argument('--registered', default='data/res/m3reg_pc.ply', help='配准后点云文件路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.source):
        print(f"错误: 源点云文件不存在: {args.source}")
        return
    
    if not os.path.exists(args.target):
        print(f"错误: 目标点云文件不存在: {args.target}")
        return
    
    registered_file = args.registered if os.path.exists(args.registered) else None
    
    visualize_pointclouds(args.source, args.target, registered_file)

if __name__ == "__main__":
    main()