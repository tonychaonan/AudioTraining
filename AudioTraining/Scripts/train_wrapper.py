"""
标准 YOLOv8 训练封装。
用 argparse 命名参数接 C# 传入配置，避免位置参数顺序错位。
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from ultralytics import YOLO

# 共享 helper（逐类 metrics 打印）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_helpers import evaluate_and_print_metrics


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
    parser = argparse.ArgumentParser(description='YOLOv8 standard training wrapper.')
    parser.add_argument('--data', required=True, help='data.yaml 路径')
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--model-type', choices=['detect', 'obb'], default='detect')
    parser.add_argument('--device', default='0')
    parser.add_argument('--model-size', default='n', help='n/s/m/l/x')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0, help='>0 时启用固定种子')
    parser.add_argument('--base-model', default='', help='.pt 基础模型路径，空则从 pretrained 开始')
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--mosaic', type=float, default=0.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    return parser


def main():
    args = _build_parser().parse_args()

    if args.seed > 0:
        set_seed(args.seed)
    else:
        print("--- Random seed disabled (seed <= 0), training will be non-deterministic ---")

    base_model_path = (args.base_model or '').strip()
    use_base = bool(base_model_path) and os.path.exists(base_model_path)

    print("--- Python Engine: Loading Model ---")
    if use_base:
        print(f"Loading existing base model for continued training: {base_model_path}")
        model = YOLO(base_model_path)
        # 校验 base 模型任务类型和用户选择一致（避免 OBB 模型当普通 detect 用，或反之）
        expected_task = 'obb' if args.model_type == 'obb' else 'detect'
        actual_task = getattr(model, 'task', None)
        if actual_task and actual_task != expected_task:
            print(f"ERROR: base model task '{actual_task}' 不匹配 --model-type '{args.model_type}' (期望 '{expected_task}')",
                  file=sys.stderr)
            sys.exit(2)
    else:
        if args.model_type == 'obb':
            model_file = f"yolov8{args.model_size}-obb.pt"
            print(f"Loading YOLOv8-OBB pretrained: {model_file}")
        else:
            model_file = f"yolov8{args.model_size}.pt"
            print(f"Loading YOLOv8 standard pretrained: {model_file}")
        model = YOLO(model_file)

    print(f"--- Python Engine: Start Training ({args.epochs} epochs, imgsz {args.img_size}) ---")
    print(f"--- Training params: lr0={args.lr0}, patience={args.patience}, mosaic={args.mosaic}, mixup={args.mixup} ---")

    project_path = os.path.join(os.getcwd(), 'train_output')
    exp_name = 'current_exp'

    # cos_lr + warmup_epochs：标准训练下也是更稳的配置
    # continue-train 场景（use_base=True）更是必须，防止高 lr 冲掉已有特征
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
        'warmup_epochs': 1.0 if use_base else 3.0,  # continue 训练：短 warmup；全新训练：标准 warmup
        'patience': args.patience,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'plots': True,
    }
    if args.seed > 0:
        train_params['seed'] = args.seed
        print(f"--- Training with fixed seed: {args.seed} (reproducible mode) ---")
    else:
        print("--- Training without fixed seed (non-deterministic mode, faster) ---")

    model.train(**train_params)

    print("--- Python Engine: Training Finished ---")

    best_pt_path = os.path.join(project_path, exp_name, 'weights', 'best.pt')
    if os.path.exists(best_pt_path):
        # ★ 训练后逐类 metrics：让用户看到每个类别的 P/R/mAP，尤其是 continue-train 场景
        #   可以对比上一版模型的旧类别是否掉了
        title = "Continue-Train - Per-Class Metrics" if use_base else "Standard Training - Per-Class Metrics"
        evaluate_and_print_metrics(best_pt_path, args.data, args.img_size, title=title)

        print("--- Python Engine: Exporting to ONNX ---")
        best_model = YOLO(best_pt_path)
        # ★ 关键修复：导出时必须传 imgsz，否则默认按 640 导出，C# 端按训练尺寸跑会不一致
        success = best_model.export(format='onnx', dynamic=False, imgsz=args.img_size)
        print(f"--- Export Result: {success} ---")
        print(f"--- ONNX Path: {os.path.splitext(best_pt_path)[0]}.onnx ---")
    else:
        print(f"Error: Could not find best.pt at {best_pt_path}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
