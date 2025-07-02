#!/usr/bin/env python3
"""
ICP算法性能基准测试脚本
比较不同ICP算法的性能和精度
"""

import subprocess
import time
import os
import numpy as np

def run_icp_algorithm(method_id, source_file, target_file, output_dir):
    """运行指定的ICP算法"""
    
    method_names = {
        0: "ICP",
        1: "AA-ICP", 
        2: "Fast ICP",
        3: "Robust ICP",
        4: "ICP Point-to-plane",
        5: "Robust ICP point-to-plane",
        6: "Sparse ICP",
        7: "Sparse ICP point-to-plane"
    }
    
    method_name = method_names.get(method_id, f"Method_{method_id}")
    print(f"\n=== 运行 {method_name} (方法 {method_id}) ===")
    
    # 创建输出目录
    method_output_dir = os.path.join(output_dir, f"method_{method_id}")
    os.makedirs(method_output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行算法
    try:
        result = subprocess.run([
            './build/FRICP', 
            target_file, 
            source_file, 
            method_output_dir + '/', 
            str(method_id)
        ], capture_output=True, text=True, timeout=60)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✓ 成功完成，耗时: {execution_time:.3f}秒")
            print(f"输出: {result.stdout.strip()}")
            
            # 读取变换矩阵
            trans_file = os.path.join(method_output_dir, f"m{method_id}trans.txt")
            if os.path.exists(trans_file):
                with open(trans_file, 'r') as f:
                    transform_matrix = f.read().strip()
                print(f"变换矩阵已保存到: {trans_file}")
            
            return {
                'method_id': method_id,
                'method_name': method_name,
                'success': True,
                'execution_time': execution_time,
                'output_dir': method_output_dir,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()
            }
        else:
            print(f"✗ 执行失败，返回码: {result.returncode}")
            print(f"错误输出: {result.stderr}")
            return {
                'method_id': method_id,
                'method_name': method_name,
                'success': False,
                'execution_time': execution_time,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ 执行超时 (>60秒)")
        return {
            'method_id': method_id,
            'method_name': method_name,
            'success': False,
            'execution_time': 60.0,
            'error': "Timeout"
        }
    except Exception as e:
        print(f"✗ 执行异常: {e}")
        return {
            'method_id': method_id,
            'method_name': method_name,
            'success': False,
            'execution_time': 0.0,
            'error': str(e)
        }

def read_transformation_matrix(trans_file):
    """读取变换矩阵"""
    try:
        with open(trans_file, 'r') as f:
            lines = f.readlines()
        
        matrix = []
        for line in lines[:4]:  # 只读取前4行
            row = [float(x) for x in line.strip().split()]
            if len(row) == 4:
                matrix.append(row)
        
        return np.array(matrix) if len(matrix) == 4 else None
    except:
        return None

def analyze_transformation_quality(trans_matrix):
    """分析变换矩阵的质量"""
    if trans_matrix is None:
        return None
    
    # 提取旋转矩阵和平移向量
    R = trans_matrix[:3, :3]
    t = trans_matrix[:3, 3]
    
    # 检查旋转矩阵的正交性
    should_be_identity = np.dot(R, R.T)
    orthogonality_error = np.linalg.norm(should_be_identity - np.eye(3))
    
    # 检查行列式是否为1
    det_error = abs(np.linalg.det(R) - 1.0)
    
    # 计算旋转角度
    trace_R = np.trace(R)
    rotation_angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
    rotation_angle_deg = np.degrees(rotation_angle)
    
    # 计算平移距离
    translation_distance = np.linalg.norm(t)
    
    return {
        'orthogonality_error': orthogonality_error,
        'determinant_error': det_error,
        'rotation_angle_deg': rotation_angle_deg,
        'translation_distance': translation_distance,
        'transformation_matrix': trans_matrix
    }

def main():
    """主函数"""
    
    # 检查可执行文件是否存在
    if not os.path.exists('./build/FRICP'):
        print("错误: 找不到可执行文件 ./build/FRICP")
        print("请先编译项目: cd build && make")
        return
    
    # 设置文件路径
    source_file = './data/source.ply'
    target_file = './data/target.ply'
    output_dir = './benchmark_results'
    
    # 检查输入文件
    if not os.path.exists(source_file):
        print(f"错误: 找不到源点云文件: {source_file}")
        return
    
    if not os.path.exists(target_file):
        print(f"错误: 找不到目标点云文件: {target_file}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fast-Robust-ICP 算法性能基准测试")
    print("=" * 50)
    
    # 测试所有算法
    methods_to_test = [0, 1, 2, 3, 4, 5, 6, 7]
    results = []
    
    for method_id in methods_to_test:
        result = run_icp_algorithm(method_id, source_file, target_file, output_dir)
        results.append(result)
    
    # 分析结果
    print("\n" + "=" * 50)
    print("基准测试结果汇总")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if successful_results:
        print(f"\n成功的算法 ({len(successful_results)}/{len(results)}):")
        print("-" * 80)
        print(f"{'方法ID':<8} {'算法名称':<25} {'执行时间(秒)':<12} {'变换质量'}")
        print("-" * 80)
        
        for result in successful_results:
            # 分析变换矩阵质量
            trans_file = os.path.join(result['output_dir'], f"m{result['method_id']}trans.txt")
            trans_matrix = read_transformation_matrix(trans_file)
            quality = analyze_transformation_quality(trans_matrix)
            
            quality_str = "N/A"
            if quality:
                quality_str = f"旋转:{quality['rotation_angle_deg']:.1f}°, 平移:{quality['translation_distance']:.3f}"
            
            print(f"{result['method_id']:<8} {result['method_name']:<25} {result['execution_time']:<12.3f} {quality_str}")
    
    if failed_results:
        print(f"\n失败的算法 ({len(failed_results)}/{len(results)}):")
        print("-" * 60)
        for result in failed_results:
            print(f"方法 {result['method_id']} ({result['method_name']}): {result.get('error', 'Unknown error')}")
    
    # 保存详细结果
    report_file = os.path.join(output_dir, 'benchmark_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Fast-Robust-ICP 算法性能基准测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            f.write(f"方法 {result['method_id']}: {result['method_name']}\n")
            f.write(f"成功: {'是' if result['success'] else '否'}\n")
            f.write(f"执行时间: {result['execution_time']:.3f}秒\n")
            
            if result['success']:
                f.write(f"标准输出: {result.get('stdout', 'N/A')}\n")
                
                # 添加变换矩阵分析
                trans_file = os.path.join(result['output_dir'], f"m{result['method_id']}trans.txt")
                trans_matrix = read_transformation_matrix(trans_file)
                quality = analyze_transformation_quality(trans_matrix)
                
                if quality:
                    f.write(f"旋转角度: {quality['rotation_angle_deg']:.2f}度\n")
                    f.write(f"平移距离: {quality['translation_distance']:.4f}\n")
                    f.write(f"正交性误差: {quality['orthogonality_error']:.6f}\n")
                    f.write(f"行列式误差: {quality['determinant_error']:.6f}\n")
            else:
                f.write(f"错误: {result.get('error', 'Unknown error')}\n")
            
            f.write("-" * 40 + "\n\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    print(f"所有结果文件保存在: {output_dir}")

if __name__ == "__main__":
    main()