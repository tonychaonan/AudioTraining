"""
增量学习训练脚本 - 支持层冻结和类别扩展
用于在保持现有模型能力的同时学习新类别
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# 共享 helper：head surgery + 逐类 metrics 打印
# train_helpers.py 和本文件同目录，由 C# 部署时 PreserveNewest 复制到 bin/Scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_helpers import extend_detection_head_for_incremental, evaluate_and_print_metrics


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


def _build_parser():
    parser = argparse.ArgumentParser(description='YOLOv8 incremental training with frozen layers.')
    parser.add_argument('--data', required=True, help='data.yaml 路径')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--model-type', choices=['detect', 'obb'], default='detect')
    parser.add_argument('--device', default='0')
    parser.add_argument('--model-size', default='n', help='n/s/m/l/x')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--base-model', required=True, help='.pt 基础模型路径（增量必需）')
    parser.add_argument('--freeze', type=int, default=0,
                        help='冻结层数：0=不冻结，10=冻结前10层(推荐)，-1=冻结整个backbone')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--mosaic', type=float, default=0.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    return parser


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
    args = _build_parser().parse_args()

    if args.seed > 0:
        set_seed(args.seed)

    base_model_path = (args.base_model or '').strip()

    # 类别兼容性校验
    print("=== Verifying class compatibility for incremental learning ===")
    old_classes, new_classes, is_compatible, message = verify_class_compatibility(args.data, base_model_path)
    print(f"Verification result: {message}")

    if not is_compatible:
        print(f"ERROR: {message}", file=sys.stderr)
        print("增量学习要求：新数据集的类别列表必须以旧类别开头，可以在末尾添加新类别", file=sys.stderr)
        print("示例：旧=[A,B,C] → 新=[A,B,C,D,E] ✓", file=sys.stderr)
        print("     旧=[A,B,C] → 新=[B,C,D] ✗ (顺序错误)", file=sys.stderr)
        sys.exit(1)

    # 加载模型
    print("--- Loading model for incremental learning ---")
    if base_model_path and os.path.exists(base_model_path):
        print(f"Loading base model: {base_model_path}")
        model = YOLO(base_model_path)
        # 校验 base 模型任务类型和用户选择一致
        expected_task = 'obb' if args.model_type == 'obb' else 'detect'
        actual_task = getattr(model, 'task', None)
        if actual_task and actual_task != expected_task:
            print(f"ERROR: base model task '{actual_task}' 不匹配 --model-type '{args.model_type}' (期望 '{expected_task}')",
                  file=sys.stderr)
            sys.exit(2)
    else:
        if args.model_type == 'obb':
            model_file = f"yolov8{args.model_size}-obb.pt"
        else:
            model_file = f"yolov8{args.model_size}.pt"
        print(f"Loading pretrained model: {model_file}")
        model = YOLO(model_file)

    print(f"--- Starting incremental training ({args.epochs} epochs, imgsz {args.img_size}) ---")
    print(f"--- Freeze strategy: {args.freeze} layers (0=none, 10=recommended, -1=full backbone) ---")
    print(f"--- Training params: lr0={args.lr0}, patience={args.patience}, mosaic={args.mosaic}, mixup={args.mixup} ---")

    project_path = os.path.join(os.getcwd(), 'train_output')
    exp_name = 'incremental_exp'

    # 构建训练参数
    # ★ 增量训练关键策略（防止旧特征遗忘 + 平滑学习新类）：
    #   - cos_lr=True：余弦退火，后期 lr 平滑趋零，比 linear decay 对已收敛权重友好
    #   - warmup_epochs=1.0：模型已预训练，不需要长 warmup；过长的 warmup 会浪费有效 epoch
    #   - lrf=0.01：最终 lr = lr0 * 0.01，给收尾阶段足够小的步长保护旧权重
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'project': project_path,
        'name': exp_name,
        'device': args.device,
        'exist_ok': True,
        'workers': 0,
        'lr0': args.lr0,
        'lrf': 0.01,
        'cos_lr': True,
        'warmup_epochs': 1.0,
        'patience': args.patience,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'plots': True,
    }
    if args.seed > 0:
        train_params['seed'] = args.seed

    # ★ 关键修复：用 Ultralytics 自带的 freeze 参数（按"真正的模块"冻结），
    # 原来手写的按 named_parameters 数量冻结会跟用户期望的"层数"对不上（一个 Conv+BN 有 3~4 个参数 tensor）
    if args.freeze != 0:
        # -1 约定为冻结整个 backbone（YOLOv8 的 backbone 对应前 10 个模块）
        train_params['freeze'] = 10 if args.freeze == -1 else args.freeze
        print(f"--- Applying Ultralytics freeze={train_params['freeze']} ---")

    # ★ 类别扩展处理：尝试 head surgery 保留旧类别权重
    #   Ultralytics 默认在 new_nc != old_nc 时会重建 Detect head，旧 cls head 权重全部丢失（"假增量"）
    #   我们在 train() 之前手动扩 head：前 old_nc 个通道原样保留，新通道小随机初始化，
    #   这样 trainer 看到 model.nc == data.nc 就不会再重建，旧权重得以保留
    if len(old_classes) > 0 and len(new_classes) > len(old_classes):
        added = len(new_classes) - len(old_classes)
        print(f"--- 检测到类别扩展：{len(old_classes)} → {len(new_classes)} (+{added}) ---")

        surgery_ok = extend_detection_head_for_incremental(
            model, old_nc=len(old_classes), new_nc=len(new_classes))

        if not surgery_ok:
            # fallback：让 Ultralytics 自己处理，但旧权重会丢，必须给强警告
            print(f"--- [警告] Head surgery 未成功，Ultralytics 将自动重建 detection head ---", file=sys.stderr)
            print(f"--- [警告] 旧 {len(old_classes)} 个类别的 head 权重会被重置为随机值（仅保留 backbone 特征） ---", file=sys.stderr)
            print(f"--- [警告] 当前 lr0={args.lr0}。类别扩展场景建议 lr0≤0.0005 并确保旧类别样本占比 ≥ 50% ---", file=sys.stderr)
            if args.lr0 > 0.0015:
                print(f"--- [警告] lr0={args.lr0} 偏高，旧类别可能被冲掉。强烈建议降到 0.0005 ---", file=sys.stderr)

    model.train(**train_params)

    print("--- Incremental training finished ---")

    best_pt_path = os.path.join(project_path, exp_name, 'weights', 'best.pt')
    if os.path.exists(best_pt_path):
        # ★ 训练后逐类 metrics 评估：让用户直观看到"旧类别是否退化、新类别是否学会"
        #   任何异常都吞掉（不影响训练结果和 ONNX 导出）
        evaluate_and_print_metrics(best_pt_path, args.data, args.img_size,
                                   title="Incremental Training - Per-Class Metrics")

        print("--- Exporting to ONNX ---")
        best_model = YOLO(best_pt_path)
        # ★ 关键修复：导出时必须传 imgsz，否则默认按 640 导出，C# 端按训练尺寸跑会不一致
        success = best_model.export(format='onnx', dynamic=False, imgsz=args.img_size)
        print(f"--- Export Result: {success} ---")
        print(f"--- ONNX Path: {os.path.splitext(best_pt_path)[0]}.onnx ---")
        print(f"--- Final model classes: {len(new_classes)} ---")
    else:
        print(f"Error: Could not find best.pt at {best_pt_path}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
