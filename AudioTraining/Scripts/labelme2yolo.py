import os
import sys
import json
import glob
import argparse

def convert(dir_path, classes_file):
    # Read classes
    if not os.path.exists(classes_file):
        print(f"Error: classes.txt not found at {classes_file}")
        return

    with open(classes_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"Loaded classes: {class_map}")

    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    print(f"Found {len(json_files)} json files in {dir_path}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_width = data.get('imageWidth')
            image_height = data.get('imageHeight')
            
            if not image_width or not image_height:
                print(f"Skipping {json_file}: Missing image dimensions")
                continue

            txt_content = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label')
                if label not in class_map:
                    print(f"Warning: Label '{label}' in {json_file} not found in classes.txt. Skipping.")
                    continue
                
                class_id = class_map[label]
                points = shape.get('points')
                
                # Convert polygon/rectangle to bbox (xywh normalized)
                # points is [[x1, y1], [x2, y2], ...]
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                width = max_x - min_x
                height = max_y - min_y
                
                # Normalize
                norm_center_x = center_x / image_width
                norm_center_y = center_y / image_height
                norm_width = width / image_width
                norm_height = height / image_height
                
                # Clamp to [0, 1] just in case
                norm_center_x = max(0.0, min(1.0, norm_center_x))
                norm_center_y = max(0.0, min(1.0, norm_center_y))
                norm_width = max(0.0, min(1.0, norm_width))
                norm_height = max(0.0, min(1.0, norm_height))
                
                txt_content.append(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

            # Save as .txt
            txt_filename = os.path.splitext(json_file)[0] + ".txt"
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(txt_content))
                
        except Exception as e:
            print(f"Error converting {json_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing images and json files")
    parser.add_argument("classes", help="Path to classes.txt")
    args = parser.parse_args()
    
    convert(args.dir, args.classes)
