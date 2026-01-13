import sys
import json
from ultralytics import YOLO
import os

def main():
    if len(sys.argv) < 4:
        print("Usage: python inference_wrapper.py <model_path> <image_path> <conf_threshold> [model_type]")
        return
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    conf_threshold = float(sys.argv[3])
    model_type = sys.argv[4] if len(sys.argv) > 4 else "detect"  # 'detect' or 'obb'
    
    # 验证文件存在
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"Model file not found: {model_path}"}))
        return
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file not found: {image_path}"}))
        return
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 执行推理
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        predictions = []
        
        # 处理结果
        for r in results:
            if model_type == "obb" and r.obb is not None:
                # OBB模型：旋转边界框
                boxes = r.obb.xyxyxyxy.cpu().numpy()  # [N, 4, 2] 4个角点
                confs = r.obb.conf.cpu().numpy()
                classes = r.obb.cls.cpu().numpy()
                xywhr = r.obb.xywhr.cpu().numpy()  # [N, 5] 中心x, 中心y, 宽, 高, 角度(弧度)
                
                for box, conf, cls, xywh_r in zip(boxes, confs, classes, xywhr):
                    angle_rad = float(xywh_r[4])  # 弧度
                    angle_deg = angle_rad * 180.0 / 3.141592653589793  # 转换为角度
                    
                    predictions.append({
                        'type': 'obb',
                        'class_id': int(cls),
                        'confidence': float(conf),
                        'points': box.flatten().tolist(),  # [x1,y1,x2,y2,x3,y3,x4,y4]
                        'angle': angle_deg  # 添加模型输出的角度
                    })
            
            elif r.boxes is not None:
                # 标准检测模型：矩形边界框
                boxes = r.boxes.xyxy.cpu().numpy()  # [N, 4] x1,y1,x2,y2
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    predictions.append({
                        'type': 'detect',
                        'class_id': int(cls),
                        'confidence': float(conf),
                        'box': box.tolist()  # [x1, y1, x2, y2]
                    })
        
        # 输出JSON结果
        output = {
            'success': True,
            'count': len(predictions),
            'predictions': predictions
        }
        print(json.dumps(output))
    
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))

if __name__ == "__main__":
    main()
