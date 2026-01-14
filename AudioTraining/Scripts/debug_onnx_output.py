import onnxruntime as ort
import numpy as np
from PIL import Image
import sys

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image while maintaining aspect ratio."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = np.array(Image.fromarray(im).resize(new_unpad, Image.BICUBIC))

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = np.pad(im, ((top, bottom), (left, right), (0, 0)), constant_values=color)
    return im, r, (dw, dh)

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_onnx_output.py <model_path> <image_path>")
        return
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # 加载ONNX模型
    session = ort.InferenceSession(model_path)
    
    # 加载图像
    img = np.array(Image.open(image_path).convert('RGB'))
    print(f"原始图像尺寸: {img.shape}")
    
    # Letterbox预处理
    img_letterbox, scale, (dw, dh) = letterbox(img, new_shape=640)
    print(f"Letterbox后尺寸: {img_letterbox.shape}")
    print(f"缩放比例: {scale}, padding: ({dw}, {dh})")
    
    # 转换为模型输入格式
    img_input = img_letterbox.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, 0)
    
    # 推理
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    output = outputs[0]
    
    print(f"\n模型输出形状: {output.shape}")
    print(f"输出数据类型: {output.dtype}")
    
    # 分析输出
    num_channels = output.shape[1]
    num_anchors = output.shape[2]
    num_classes = num_channels - 5
    
    print(f"\n通道数: {num_channels}")
    print(f"锚点数: {num_anchors}")
    print(f"类别数: {num_classes}")
    
    # 检查前10个锚点的数据
    print("\n前10个锚点的数据分析:")
    for i in range(10):
        cx = output[0, 0, i]
        cy = output[0, 1, i]
        w = output[0, 2, i]
        h = output[0, 3, i]
        angle = output[0, 4, i]
        class0 = output[0, 5, i]
        class1 = output[0, 6, i] if num_classes > 1 else 0
        
        print(f"锚点{i}: cx={cx:.2f}, cy={cy:.2f}, w={w:.2f}, h={h:.2f}, angle={angle:.4f}, class0={class0:.4f}, class1={class1:.4f}")
    
    # 找出最高置信度的锚点
    print("\n查找最高置信度的锚点:")
    max_conf = -999
    max_idx = -1
    
    for i in range(num_anchors):
        for c in range(num_classes):
            score = output[0, 5 + c, i]
            if score > max_conf:
                max_conf = score
                max_idx = i
    
    print(f"最高置信度锚点索引: {max_idx}, 原始分数: {max_conf:.4f}")
    
    if max_idx >= 0:
        cx = output[0, 0, max_idx]
        cy = output[0, 1, max_idx]
        w = output[0, 2, max_idx]
        h = output[0, 3, max_idx]
        angle = output[0, 4, max_idx]
        
        print(f"该锚点的完整数据:")
        print(f"  cx={cx:.2f}, cy={cy:.2f}, w={w:.2f}, h={h:.2f}, angle={angle:.4f}")
        print(f"  坐标范围: cx是否在[0, 640]? {0 <= cx <= 640}")
        print(f"  坐标范围: cy是否在[0, 640]? {0 <= cy <= 640}")
        
        for c in range(num_classes):
            print(f"  类别{c}: {output[0, 5 + c, max_idx]:.4f}")

if __name__ == "__main__":
    main()
