import sys
import os
from ultralytics import YOLO

def main():
    if len(sys.argv) < 4:
        print("Usage: python train_wrapper.py <yaml_path> <epochs> <img_size> [device]")
        return

    # 1. Receive parameters from C#
    yaml_path = sys.argv[1]
    epochs_count = int(sys.argv[2])
    img_size = int(sys.argv[3])
    
    # Optional device argument, default to 0 (GPU 0) or cpu
    device = '0'
    if len(sys.argv) > 4:
        device = sys.argv[4]

    print("--- Python Engine: Loading Model ---")
    # Load pretrained model (yolov8n.pt)
    # Ensure yolov8n.pt is available or let it download
    model = YOLO('yolov8n.pt') 

    print(f"--- Python Engine: Start Training ({epochs_count} epochs, size {img_size}) ---")

    # 2. Start Training
    # project/name specify output path so C# can find it easily
    # We use absolute path for project if possible, or relative to current cwd
    project_path = os.path.join(os.getcwd(), 'train_output')
    exp_name = 'current_exp'
    
    # Clean up previous run if exists to ensure we get a fresh result (optional)
    # logic handled by YOLO usually creating exp, exp2... but here we might want deterministic output location
    # YOLOv8 will increment name if exist=False (default). 
    # If we want exact overwrite, we might need logic.
    # For now, let's trust YOLO but maybe we can clean up in C# before starting.

    results = model.train(
        data=yaml_path, 
        epochs=epochs_count, 
        imgsz=img_size, 
        project=project_path, 
        name=exp_name,
        device=device,
        exist_ok=True # Overwrite existing experiment folder so path is deterministic
    )

    print("--- Python Engine: Training Finished ---")

    # 3. Export to ONNX
    print("--- Python Engine: Exporting to ONNX ---")
    
    # The model path after training
    # usually project/name/weights/best.pt
    best_pt_path = os.path.join(project_path, exp_name, 'weights', 'best.pt')
    
    if os.path.exists(best_pt_path):
        # Load the best model explicitly for export to be sure
        best_model = YOLO(best_pt_path)
        success = best_model.export(format='onnx', dynamic=False)
        print(f"--- Export Result: {success} ---")
        print(f"--- ONNX Path: {os.path.splitext(best_pt_path)[0]}.onnx ---")
    else:
        print(f"Error: Could not find best.pt at {best_pt_path}")

if __name__ == "__main__":
    main()
