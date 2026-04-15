import sys
import os
from ultralytics import YOLO
import random
import numpy as np
import torch

def _parse_int_arg(argv, index, default=None):
    if len(argv) <= index:
        return default
    try:
        return int(argv[index])
    except (TypeError, ValueError):
        return default

def _parse_float_arg(argv, index, default):
    if len(argv) <= index:
        return default
    try:
        return float(argv[index])
    except (TypeError, ValueError):
        return default

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
    if len(sys.argv) < 8:
        print("Usage: python train_wrapper.py <yaml_path> <epochs> <img_size> <model_type> <device> <model_size> <batch_size> [seed] [base_model_path] [lr0] [patience] [mosaic] [mixup]")
        return

    yaml_path = sys.argv[1]
    epochs_count = int(sys.argv[2])
    img_size = int(sys.argv[3])
    model_type = sys.argv[4]  # 'detect' or 'obb'
    device = sys.argv[5]
    model_size = sys.argv[6].strip().lower()
    batch_size = int(sys.argv[7])
    
    seed = None
    seed_value = _parse_int_arg(sys.argv, 8, None)
    if seed_value is not None and seed_value > 0:
        seed = seed_value
        set_seed(seed)
    elif len(sys.argv) > 8:
        print("--- Random seed disabled (seed <= 0), training will be non-deterministic ---")

    base_model_path = None
    if len(sys.argv) > 9:
        candidate_path = sys.argv[9].strip()
        if candidate_path and os.path.exists(candidate_path):
            base_model_path = candidate_path

    lr0 = _parse_float_arg(sys.argv, 10, 0.001)
    patience = _parse_int_arg(sys.argv, 11, 30)
    mosaic = _parse_float_arg(sys.argv, 12, 0.0)
    mixup = _parse_float_arg(sys.argv, 13, 0.0)

    print("--- Python Engine: Loading Model ---")
    if base_model_path:
        print(f"Loading existing base model for continued training: {base_model_path}")
        model = YOLO(base_model_path)
    else:
        if model_type == 'obb':
            model_file = f"yolov8{model_size}-obb.pt"
            print(f"Loading YOLOv8-OBB model: {model_file}")
        else:
            model_file = f"yolov8{model_size}.pt"
            print(f"Loading YOLOv8 standard detection model: {model_file}")
        model = YOLO(model_file)

    print(f"--- Python Engine: Start Training ({epochs_count} epochs, size {img_size}) ---")
    print(f"--- Training params: lr0={lr0}, patience={patience}, mosaic={mosaic}, mixup={mixup} ---")

    project_path = os.path.join(os.getcwd(), 'train_output')
    exp_name = 'current_exp'  

    # 构建训练参数
    train_params = {
        'data': yaml_path,
        'epochs': epochs_count,
        'imgsz': img_size,
        'batch': batch_size,
        'project': project_path,
        'name': exp_name,
        'device': device,
        'exist_ok': True,  
        'workers': 0,
        'lr0': lr0,
        'patience': patience,
        'mosaic': mosaic,
        'mixup': mixup
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
