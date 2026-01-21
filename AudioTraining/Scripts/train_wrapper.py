import sys
import os
from ultralytics import YOLO
import random
import numpy as np
import torch

def set_seed(seed):
    """设置所有随机种子以确保训练可重复性"""
    print(f"--- Setting random seed to {seed} for reproducibility ---")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性（会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("--- Random seed set successfully ---")

def main():
    if len(sys.argv) < 5:
        print("Usage: python train_wrapper.py <yaml_path> <epochs> <img_size> <model_type> [device] [seed]")
        return

    yaml_path = sys.argv[1]
    epochs_count = int(sys.argv[2])
    img_size = int(sys.argv[3])
    model_type = sys.argv[4] 
    
    device = '0'
    if len(sys.argv) > 5:
        device = sys.argv[5]
    
    seed = None
    if len(sys.argv) > 6:
        try:
            seed_value = int(sys.argv[6])
            if seed_value > 0:
                seed = seed_value
                set_seed(seed)
            else:
                print("--- Random seed disabled (seed <= 0), training will be non-deterministic ---")
        except ValueError:
            print("--- Invalid seed value, training will be non-deterministic ---")

    print("--- Python Engine: Loading Model ---")
    if model_type == 'obb':
        print("Loading YOLOv8-OBB model for oriented bounding box detection...")
        model = YOLO('yolov8n-obb.pt')  
    else:
        print("Loading YOLOv8 standard detection model...")
        model = YOLO('yolov8n.pt')  

    print(f"--- Python Engine: Start Training ({epochs_count} epochs, size {img_size}) ---")

    project_path = os.path.join(os.getcwd(), 'train_output')
    exp_name = 'current_exp'  

    # 构建训练参数
    train_params = {
        'data': yaml_path,
        'epochs': epochs_count,
        'imgsz': img_size,
        'project': project_path,
        'name': exp_name,
        'device': device,
        'exist_ok': True,  
        'workers': 0      
    }
    
    # 如果设置了随机种子，添加到训练参数中
    if seed is not None:
        train_params['seed'] = seed
        print(f"--- Training with fixed seed: {seed} (reproducible mode) ---")
    else:
        print("--- Training without fixed seed (non-deterministic mode, faster) ---")
    
    results = model.train(**train_params)

    print("--- Python Engine: Training Finished ---")

    print("--- Python Engine: Exporting to ONNX ---")
    
    best_pt_path = os.path.join(project_path, exp_name, 'weights', 'best.pt')
    
    if os.path.exists(best_pt_path):
        best_model = YOLO(best_pt_path)
        success = best_model.export(format='onnx', dynamic=False)
        print(f"--- Export Result: {success} ---")
        print(f"--- ONNX Path: {os.path.splitext(best_pt_path)[0]}.onnx ---")
    else:
        print(f"Error: Could not find best.pt at {best_pt_path}")

if __name__ == "__main__":
    main()
