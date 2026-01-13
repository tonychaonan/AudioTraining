"""
独立测试OBB模型输出
用于验证模型是否正确学习了旋转信息
"""
from ultralytics import YOLO
import cv2
import numpy as np
import math

def test_obb_model(model_path, image_path):
    """
    测试OBB模型
    
    参数:
        model_path: 模型路径 (.pt 或 .onnx)
        image_path: 测试图片路径
    """
    print("="*60)
    print("OBB模型测试")
    print("="*60)
    
    # 1. 加载模型
    print(f"\n1. 加载模型: {model_path}")
    try:
        model = YOLO(model_path)
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return
    
    # 2. 执行推理
    print(f"\n2. 推理图片: {image_path}")
    try:
        results = model.predict(image_path, conf=0.25, verbose=False)
        print("   ✅ 推理完成")
    except Exception as e:
        print(f"   ❌ 推理失败: {e}")
        return
    
    # 3. 分析结果
    print("\n3. 分析结果:")
    for i, r in enumerate(results):
        print(f"\n   图片 {i+1}:")
        
        # 检查是否有OBB检测结果
        if r.obb is None:
            print("   ❌ 没有OBB检测结果")
            print("   提示: 这可能是标准检测模型，不是OBB模型")
            continue
        
        # 获取OBB结果
        num_detections = len(r.obb)
        print(f"   检测数量: {num_detections}")
        
        if num_detections == 0:
            print("   ⚠️ 没有检测到目标")
            continue
        
        # 遍历每个检测结果
        for j in range(num_detections):
            print(f"\n   目标 {j+1}:")
            
            # 类别和置信度
            cls = int(r.obb.cls[j].item())
            conf = r.obb.conf[j].item()
            print(f"     类别: {cls}")
            print(f"     置信度: {conf:.4f}")
            
            # 检查是否有旋转角度属性
            if hasattr(r.obb, 'xywhr'):
                xywhr = r.obb.xywhr[j].cpu().numpy()
                print(f"     xywhr格式: 中心({xywhr[0]:.2f}, {xywhr[1]:.2f}), 宽高({xywhr[2]:.2f}, {xywhr[3]:.2f}), 角度={xywhr[4]:.4f} rad = {xywhr[4]*180/math.pi:.2f}°")
            
            if hasattr(r.obb, 'xyxyxyxyn'):
                print(f"     模型还有xyxyxyxyn属性（归一化坐标）")
            
            # 获取角点坐标 [4, 2]
            corners = r.obb.xyxyxyxy[j].cpu().numpy()
            print(f"     角点坐标:")
            for k, corner in enumerate(corners):
                print(f"       P{k+1}: ({corner[0]:.2f}, {corner[1]:.2f})")
            
            # 计算所有4条边的角度
            print(f"     所有边的角度:")
            angles = []
            for k in range(4):
                p1 = corners[k]
                p2 = corners[(k + 1) % 4]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle_rad = math.atan2(dy, dx)
                angle_deg = angle_rad * 180 / math.pi
                edge_len = math.sqrt(dx**2 + dy**2)
                angles.append(angle_deg)
                print(f"       边{k+1} (P{k+1}→P{(k+1)%4+1}): {angle_deg:.2f}°, 长度: {edge_len:.2f}")
            
            # 找出长边（通常是主方向）
            edge_lengths = []
            for k in range(4):
                p1 = corners[k]
                p2 = corners[(k + 1) % 4]
                edge_len = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                edge_lengths.append(edge_len)
            
            # 找出最长的边
            max_len_idx = edge_lengths.index(max(edge_lengths))
            main_angle = angles[max_len_idx]
            print(f"     主方向角度（最长边）: {main_angle:.2f}°")
            
            # 判断是否有旋转
            if abs(main_angle) < 5 or abs(main_angle - 90) < 5 or abs(main_angle + 90) < 5:
                print(f"     ⚠️ 角度接近0°或90°，可能没有学习到旋转信息")
            else:
                print(f"     ✅ 检测到明显的旋转角度")
            
            # 计算边长
            edge1_len = math.sqrt(dx**2 + dy**2)
            dx2 = corners[2][0] - corners[1][0]
            dy2 = corners[2][1] - corners[1][1]
            edge2_len = math.sqrt(dx2**2 + dy2**2)
            print(f"     边长: {edge1_len:.2f} x {edge2_len:.2f}")
    
    # 4. 可视化结果
    print("\n4. 可视化结果:")
    try:
        # 绘制结果
        annotated = results[0].plot()
        
        # 保存结果图片
        output_path = image_path.replace('.', '_result.')
        cv2.imwrite(output_path, annotated)
        print(f"   ✅ 结果已保存到: {output_path}")
        
        # 显示结果（可选）
        print("   提示: 按任意键关闭图片窗口")
        cv2.imshow('OBB Detection Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ⚠️ 可视化失败: {e}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python test_obb_model.py <模型路径> <图片路径>")
        print("\n示例:")
        print("  python test_obb_model.py best.pt test_image.jpg")
        print("  python test_obb_model.py best.onnx test_image.bmp")
        print("\n说明:")
        print("  - 模型路径: 训练生成的.pt或.onnx文件")
        print("  - 图片路径: 要测试的图片")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    test_obb_model(model_path, image_path)
