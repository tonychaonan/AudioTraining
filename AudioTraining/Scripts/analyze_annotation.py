"""
分析标注文件和模型输出的角度对应关系
"""
import math

def analyze_annotation(txt_path, image_width, image_height):
    """
    分析标注文件中的OBB角度
    
    参数:
        txt_path: 标注文件路径
        image_width: 图片宽度
        image_height: 图片高度
    """
    print("="*60)
    print("标注文件分析")
    print("="*60)
    
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
    
    parts = line.split()
    if len(parts) != 9:
        print("❌ 不是OBB格式")
        return
    
    class_id = int(parts[0])
    
    # 归一化坐标转换为像素坐标
    points = []
    for i in range(4):
        x = float(parts[1 + i*2]) * image_width
        y = float(parts[2 + i*2]) * image_height
        points.append((x, y))
    
    print(f"\n标注角点（像素坐标）:")
    for i, p in enumerate(points):
        print(f"  P{i+1}: ({p[0]:.2f}, {p[1]:.2f})")
    
    # 计算所有边的角度
    print(f"\n标注的所有边角度:")
    for i in range(4):
        p1 = points[i]
        p2 = points[(i+1)%4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = angle_rad * 180 / math.pi
        length = math.sqrt(dx**2 + dy**2)
        print(f"  边{i+1} (P{i+1}→P{(i+1)%4+1}): {angle_deg:.2f}°, 长度: {length:.2f}")
    
    # 计算中心点和宽高
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / 4
    cy = sum(ys) / 4
    
    # 计算边长
    edge1_len = math.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
    edge2_len = math.sqrt((points[2][0]-points[1][0])**2 + (points[2][1]-points[1][1])**2)
    
    # 计算主方向角度（使用第一条边）
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    main_angle_rad = math.atan2(dy, dx)
    main_angle_deg = main_angle_rad * 180 / math.pi
    
    print(f"\n标注的xywhr格式:")
    print(f"  中心: ({cx:.2f}, {cy:.2f})")
    print(f"  宽高: ({edge1_len:.2f}, {edge2_len:.2f})")
    print(f"  角度: {main_angle_rad:.4f} rad = {main_angle_deg:.2f}°")
    
    # 归一化角度到 [-90, 90]
    normalized_angle = main_angle_deg
    while normalized_angle > 90:
        normalized_angle -= 180
    while normalized_angle < -90:
        normalized_angle += 180
    
    print(f"  归一化角度 [-90°, 90°]: {normalized_angle:.2f}°")
    
    # 归一化角度到 [0, 90]
    if normalized_angle < 0:
        normalized_angle_0_90 = normalized_angle + 90
    else:
        normalized_angle_0_90 = normalized_angle
    
    print(f"  归一化角度 [0°, 90°]: {normalized_angle_0_90:.2f}°")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python analyze_annotation.py <标注文件路径> <图片宽度> <图片高度>")
        print("\n示例:")
        print("  python analyze_annotation.py step1_image.txt 2270 1836")
        sys.exit(1)
    
    txt_path = sys.argv[1]
    image_width = int(sys.argv[2])
    image_height = int(sys.argv[3])
    
    analyze_annotation(txt_path, image_width, image_height)
