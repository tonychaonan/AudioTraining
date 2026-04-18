"""
训练辅助工具：
  1. extend_detection_head_for_incremental —— 类别扩展时手动扩 YOLOv8 Detect head，
     保留旧类别的 head 权重（防"假增量"：Ultralytics 默认会重建 head 导致旧权重丢失）
  2. print_per_class_metrics —— 训练后输出每个类别的 P/R/mAP，让用户判断
     "旧类别掉了多少 / 新类别学得如何"

两个模块都被 train_incremental.py 和 train_wrapper.py 共用。
"""
import math
import sys
import traceback


def extend_detection_head_for_incremental(model, old_nc, new_nc):
    """
    动态扩展 YOLOv8 / YOLOv8-OBB 的分类 head，把输出类别数从 old_nc 扩到 new_nc。
    旧 old_nc 个类别的 weight + bias 原样保留，新增通道随机初始化。

    ★ 必须在 model.train() 之前调用，并保证调用后 model.model.nc == new_nc
      这样 Ultralytics 的 trainer 不会再重建 head（保护我们手动保留的旧权重）。

    Args:
        model: ultralytics.YOLO 实例（已加载 base .pt）
        old_nc: base 模型原本的类别数
        new_nc: 目标类别数（必须 > old_nc）

    Returns:
        bool: True=扩展成功；False=结构不识别，调用方应回退到默认行为
    """
    if new_nc <= old_nc:
        return False

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("WARN: torch 未安装，跳过 head surgery。", file=sys.stderr)
        return False

    try:
        inner = model.model  # DetectionModel / OBBModel
        detect_module = inner.model[-1]  # Detect / OBB 模块（YOLOv8 约定最后一个 module 是 head）

        if not hasattr(detect_module, 'cv3') or not isinstance(detect_module.cv3, nn.ModuleList):
            print(f"WARN: Detect 模块无 cv3 ModuleList（类型 {type(detect_module).__name__}），"
                  f"跳过 head surgery，fallback 到 Ultralytics 默认行为。", file=sys.stderr)
            return False

        print(f"--- [Head Surgery] 扩展 classification head: {old_nc} → {new_nc} classes ---")

        # 各 scale 的 stride（用于复刻 Ultralytics 的 cls bias 初始化）
        strides = getattr(detect_module, 'stride', None)
        stride_list = []
        if strides is not None:
            try:
                stride_list = [float(s) for s in strides]
            except Exception:
                stride_list = []
        if not stride_list:
            stride_list = [8.0, 16.0, 32.0]  # YOLOv8 默认 3 scale

        for i, cls_branch in enumerate(detect_module.cv3):
            if not isinstance(cls_branch, nn.Sequential) or len(cls_branch) == 0:
                print(f"WARN: cv3[{i}] 不是 Sequential 或为空，放弃 head surgery。", file=sys.stderr)
                return False

            old_conv = cls_branch[-1]
            if not isinstance(old_conv, nn.Conv2d):
                print(f"WARN: cv3[{i}] 最后一层不是 Conv2d（是 {type(old_conv).__name__}），放弃。",
                      file=sys.stderr)
                return False

            if old_conv.out_channels != old_nc:
                print(f"WARN: cv3[{i}][-1].out_channels={old_conv.out_channels} ≠ old_nc={old_nc}，"
                      f"base 模型类别数声明不一致，放弃 head surgery。", file=sys.stderr)
                return False

            new_conv = nn.Conv2d(
                in_channels=old_conv.in_channels,
                out_channels=new_nc,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=(old_conv.bias is not None),
            ).to(old_conv.weight.device, old_conv.weight.dtype)

            with torch.no_grad():
                # 先用 kaiming 初始化整个 weight（给新通道一个合理的起点）
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                # 旧 old_nc 个通道原样替换回旧权重
                new_conv.weight[:old_nc] = old_conv.weight.data.clone()

                if old_conv.bias is not None:
                    # 新通道的 bias：复刻 Ultralytics 的 cls bias 初始化
                    #   b = log(5 / nc / (640/s)^2)    —— 让初始 sigmoid 约 0.03
                    s = stride_list[i] if i < len(stride_list) else 8.0
                    default_bias = math.log(5.0 / max(new_nc, 1) / (640.0 / s) ** 2)
                    new_conv.bias.data.fill_(default_bias)
                    # 旧类别的 bias 原样保留
                    new_conv.bias[:old_nc] = old_conv.bias.data.clone()

            # 替换 Sequential 的最后一层
            cls_branch[-1] = new_conv

        # 同步 nc 相关字段，让 Ultralytics trainer 不再尝试重建 head
        detect_module.nc = new_nc
        if hasattr(detect_module, 'reg_max'):
            detect_module.no = new_nc + detect_module.reg_max * 4

        inner.nc = new_nc
        if hasattr(inner, 'yaml') and isinstance(inner.yaml, dict):
            inner.yaml['nc'] = new_nc

        print(f"--- [Head Surgery] SUCCESS: preserved old {old_nc} class weights, "
              f"initialized {new_nc - old_nc} new class channels ---")
        return True

    except Exception as e:
        print(f"WARN: Head surgery 失败，fallback 到 Ultralytics 默认行为: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


def print_per_class_metrics(metrics, names, title="Final Validation Metrics"):
    """
    打印每个类别的 P/R/mAP50/mAP50-95，同时给整体 overall。
    兼容 Ultralytics 不同版本的 metrics 对象结构（字段缺失时显示 N/A，不崩）。

    Args:
        metrics: ultralytics val() 返回的对象（含 .box 属性）
        names: 类别名映射（dict {id: name} 或 list [name, ...]）
        title: 打印的标题
    """
    def fmt(v):
        if v is None:
            return "  N/A"
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    def name_of(cid):
        try:
            if isinstance(names, dict):
                return names.get(int(cid), f"class_{cid}")
            if isinstance(names, (list, tuple)) and 0 <= int(cid) < len(names):
                return names[int(cid)]
        except Exception:
            pass
        return f"class_{cid}"

    try:
        box = getattr(metrics, 'box', None)
        if box is None:
            print("WARN: metrics.box 不存在，跳过逐类 metrics 输出", file=sys.stderr)
            return

        # overall
        mp = getattr(box, 'mp', None)
        mr = getattr(box, 'mr', None)
        map50 = getattr(box, 'map50', None)
        map5095 = getattr(box, 'map', None)

        print("=" * 72)
        print(f"--- {title} (overall) ---")
        print(f"    Precision (mean):    {fmt(mp)}")
        print(f"    Recall    (mean):    {fmt(mr)}")
        print(f"    mAP@50    (overall): {fmt(map50)}")
        print(f"    mAP@50-95 (overall): {fmt(map5095)}")

        class_idx = getattr(box, 'ap_class_index', None)
        if class_idx is None:
            print("WARN: metrics.box.ap_class_index 不存在，跳过逐类输出", file=sys.stderr)
            print("=" * 72)
            return

        p_arr = getattr(box, 'p', None)
        r_arr = getattr(box, 'r', None)
        ap50_arr = getattr(box, 'ap50', None)
        maps_arr = getattr(box, 'maps', None)

        print(f"--- Per-Class Metrics ---")
        print(f"    {'ID':>4s}  {'Class':<28s}  {'P':>8s}  {'R':>8s}  {'mAP50':>8s}  {'mAP50-95':>10s}")

        for i, cid in enumerate(class_idx):
            try:
                cid_int = int(cid)
            except Exception:
                continue
            cname = name_of(cid_int)[:28]
            p = p_arr[i] if p_arr is not None and i < len(p_arr) else None
            r = r_arr[i] if r_arr is not None and i < len(r_arr) else None
            ap50 = ap50_arr[i] if ap50_arr is not None and i < len(ap50_arr) else None
            # maps 是用 class_id 直接索引（length = total classes）
            ap5095 = maps_arr[cid_int] if maps_arr is not None and cid_int < len(maps_arr) else None
            print(f"    [{cid_int:>2d}]  {cname:<28s}  {fmt(p)}  {fmt(r)}  {fmt(ap50)}  {fmt(ap5095)}")

        print("=" * 72)

    except Exception as e:
        print(f"WARN: 解析 metrics 失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def evaluate_and_print_metrics(best_pt_path, data_yaml, img_size, title="Final Validation Metrics"):
    """
    加载 best.pt 跑一次 val，并打印逐类 metrics。
    任何异常都吞掉（不影响训练主流程），只打印警告。
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("WARN: ultralytics 未安装，跳过 per-class metrics", file=sys.stderr)
        return

    try:
        print(f"--- Evaluating per-class metrics on val set ---")
        eval_model = YOLO(best_pt_path)
        metrics = eval_model.val(data=data_yaml, imgsz=img_size, split='val',
                                 verbose=False, plots=False)
        print_per_class_metrics(metrics, eval_model.names, title=title)
    except Exception as e:
        print(f"WARN: per-class 评估失败（不影响训练结果和 ONNX 导出）: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
