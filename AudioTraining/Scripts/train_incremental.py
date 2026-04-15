"""
增量学习训练脚本 - 支持层冻结和类别扩展
用于在保持现有模型能力的同时学习新类别
"""
import sys
import os
from ultralytics import YOLO
import random
import numpy as np
import torch
import yaml

def set_seed(seed):
    """设置所有随机种子以确保训练可重复性"""
    print(f"--- Setting random seed to {seed} for reproducibility ---")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("--- Random seed set successfully ---")

def freeze_layers(model, freeze_count):
    """
    冻结模型的前N层，只训练后面的层和检测头
    
    Args:
        model: YOLO模型实例
        freeze_count: 要冻结的层数（0=不冻结，-1=全部冻结）
    """
    if freeze_count == 0:
        print("--- No layers frozen, training all layers ---")
        return
    
    if freeze_count == -1:
        # 冻结整个backbone
        print("--- Freezing entire backbone ---")
        for name, param in model.model.named_parameters():
            if 'model.model' in name:  # backbone部分
                param.requires_grad = False
                print(f"Frozen: {name}")
        return
    
    # 冻结前N层
    print(f"--- Freezing first {freeze_count} layers ---")
    frozen_count = 0
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < freeze_count:
            param.requires_grad = False
            frozen_count += 1
            print(f"Frozen layer {i}: {name}")
        else:
            break
    
    print(f"--- Total frozen parameters: {frozen_count} ---")

def verify_class_compatibility(yaml_path, base_model_path):
    """
    验证新数据集的类别是否与基础模型兼容
    增量学习要求：新类别必须包含旧类别（可以额外添加新类别）
    
    Returns:
        (old_classes, new_classes, is_compatible, message)
    """
    try:
        # 读取新数据集的classes
        with open(yaml_path, 'r', encoding='utf-8') as f:
            new_data = yaml.safe_load(f)
        new_classes = new_data.get('names', [])
        
        # 尝试从基础模型读取旧classes
        if base_model_path and os.path.exists(base_model_path):
            temp_model = YOLO(base_model_path)
            old_classes = temp_model.names
            if isinstance(old_classes, dict):
                old_classes = list(old_classes.values())
            
            print(f"--- Base model classes ({len(old_classes)}): {old_classes} ---")
            print(f"--- New dataset classes ({len(new_classes)}): {new_classes} ---")
            
            # 检查旧类别是否都在新类别中（顺序必须一致）
            if len(new_classes) < len(old_classes):
                return old_classes, new_classes, False, f"新数据集类别数({len(new_classes)}) < 基础模型类别数({len(old_classes)})"
            
            for i, old_cls in enumerate(old_classes):
                if i >= len(new_classes) or new_classes[i] != old_cls:
                    return old_classes, new_classes, False, f"类别顺序不匹配：旧[{i}]={old_cls}, 新[{i}]={new_classes[i] if i < len(new_classes) else 'None'}"
            
            # 检查是否有新增类别
            if len(new_classes) > len(old_classes):
                new_added = new_classes[len(old_classes):]
                print(f"--- Detected {len(new_added)} new classes: {new_added} ---")
                return old_classes, new_classes, True, f"增量学习：保留{len(old_classes)}个旧类，新增{len(new_added)}个类"
            else:
                print("--- No new classes, pure fine-tuning mode ---")
                return old_classes, new_classes, True, "纯微调模式（类别数未变）"
        else:
            print("--- No base model, starting from pretrained weights ---")
            return [], new_classes, True, "从预训练权重开始（非增量模式）"
            
    except Exception as e:
        return [], [], False, f"类别验证失败: {str(e)}"

def main():
    if len(sys.argv) < 10:
        print("Usage: python train_incremental.py <yaml_path> <epochs> <img_size> <model_type> <device> <model_size> <batch_size> <base_model_path> <freeze_layers> [seed]")
        print("  freeze_layers: 要冻结的层数 (0=不冻结, 10=冻结前10层, -1=冻结整个backbone)")
        return

    yaml_path = sys.argv[1]
    epochs_count = int(sys.argv[2])
    img_size = int(sys.argv[3])
    model_type = sys.argv[4]  # 'detect' or 'obb'
    device = sys.argv[5]
    model_size = sys.argv[6].strip().lower()
    batch_size = int(sys.argv[7])
    base_model_path = sys.argv[8]
    freeze_layers_count = int(sys.argv[9])
    
    seed = None
    if len(sys.argv) > 10:
        try:
            seed_value = int(sys.argv[10])
            if seed_value > 0:
                seed = seed_value
                set_seed(seed)
        except ValueError:
            pass

    # 验证类别兼容性
    print("=== Verifying class compatibility for incremental learning ===")
    old_classes, new_classes, is_compatible, message = verify_class_compatibility(yaml_path, base_model_path)
    print(f"Verification result: {message}")
    
    if not is_compatible:
        print(f"ERROR: {message}")
        print("增量学习要求：新数据集的类别列表必须以旧类别开头，可以在末尾添加新类别")
        print("示例：旧=[A,B,C] → 新=[A,B,C,D,E] ✓")
        print("     旧=[A,B,C] → 新=[B,C,D] ✗ (顺序错误)")
        sys.exit(1)

    # 加载模型
    print("--- Loading model for incremental learning ---")
    if base_model_path and os.path.exists(base_model_path):
        print(f"Loading base model: {base_model_path}")
        model = YOLO(base_model_path)
    else:
        if model_type == 'obb':
            model_file = f"yolov8{model_size}-obb.pt"
        else:
            model_file = f"yolov8{model_size}.pt"
        print(f"Loading pretrained model: {model_file}")
        model = YOLO(model_file)

    # 冻结指定层数
    if freeze_layers_count != 0:
        freeze_layers(model, freeze_layers_count)
    
    print(f"--- Starting incremental training ({epochs_count} epochs, size {img_size}) ---")
    print(f"--- Freeze strategy: {freeze_layers_count} layers ---")

    project_path = os.path.join(os.getcwd(), 'train_output')
    exp_name = 'incremental_exp'

    # 构建训练参数（增量学习建议更小的学习率）
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
        'lr0': 0.001,  # 更小的初始学习率（增量学习推荐）
        'lrf': 0.01    # 最终学习率因子
    }
    
    if seed is not None:
        train_params['seed'] = seed

    results = model.train(**train_params)

    print("--- Incremental training finished ---")
    print("--- Exporting to ONNX ---")
    
    best_pt_path = os.path.join(project_path, exp_name, 'weights', 'best.pt')
    
    if os.path.exists(best_pt_path):
        best_model = YOLO(best_pt_path)
        success = best_model.export(format='onnx', dynamic=False)
        print(f"--- Export Result: {success} ---")
        print(f"--- ONNX Path: {os.path.splitext(best_pt_path)[0]}.onnx ---")
        
        # 验证导出的模型类别数
        print(f"--- Final model classes: {len(new_classes)} ---")
    else:
        print(f"Error: Could not find best.pt at {best_pt_path}")

if __name__ == "__main__":
    main()
