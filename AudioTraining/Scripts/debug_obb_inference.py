import sys
import json
from ultralytics import YOLO
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: python debug_obb_inference.py <model_path> <image_path> <conf_threshold>")
        return
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    conf_threshold = float(sys.argv[3])
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 执行推理
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,
            verbose=True  # 显示详细信息
        )
        
        predictions = []
        
        # 处理结果
        for r in results:
            print(f"\n[Python推理调试] 原始图像尺寸: {r.orig_shape}")
            
            if r.obb is not None:
                # OBB模型：旋转边界框
                boxes = r.obb.xyxyxyxy.cpu().numpy()  # [N, 4, 2] 4个角点
                confs = r.obb.conf.cpu().numpy()
                classes = r.obb.cls.cpu().numpy()
                xywhr = r.obb.xywhr.cpu().numpy()  # [N, 5] 中心x, 中心y, 宽, 高, 角度(弧度)
                
                print(f"[Python推理调试] 检测到 {len(boxes)} 个目标")
                
                for i, (box, conf, cls, xywh_r) in enumerate(zip(boxes, confs, classes, xywhr)):
                    cx, cy, w, h, angle_rad = xywh_r
                    angle_deg = angle_rad * 180.0 / np.pi
                    
                    print(f"\n[Python推理调试] 目标 {i+1}:")
                    print(f"  中心坐标: cx={cx:.2f}, cy={cy:.2f}")
                    print(f"  宽高: w={w:.2f}, h={h:.2f}")
                    print(f"  角度: {angle_rad:.4f} rad = {angle_deg:.2f}°")
                    print(f"  置信度: {conf:.4f}")
                    print(f"  类别: {int(cls)}")
                    print(f"  角点坐标:")
                    for j, (px, py) in enumerate(box):
                        print(f"    P{j+1}: ({px:.2f}, {py:.2f})")
                    
                    predictions.append({
                        'type': 'obb',
                        'class_id': int(cls),
                        'confidence': float(conf),
                        'center': [float(cx), float(cy)],
                        'size': [float(w), float(h)],
                        'angle_rad': float(angle_rad),
                        'angle_deg': float(angle_deg),
                        'points': box.flatten().tolist()
                    })
        
        # 输出JSON结果
        output = {
            'success': True,
            'count': len(predictions),
            'predictions': predictions
        }
        print(f"\n[Python推理调试] JSON输出:")
        print(json.dumps(output, indent=2))
    
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
