import os
import shutil
import glob
import argparse
from tqdm import tqdm

# ================= Configuration =================
# We have converted configuration to argparse arguments
# Original labels will STILL be read from landslide_workdir/yolo_data/labels

def finalize_sr_data(sr_temp_root, dest_root_prefix):
    if not os.path.exists(sr_temp_root):
        print(f"Error: Source directory '{sr_temp_root}' does not exist.")
        return

    dest_384 = os.path.join(dest_root_prefix, 'img_384_SR')
    dest_126 = os.path.join(dest_root_prefix, 'img_126_SR')

    # Task A: 384 processing
    process_group(os.path.join(sr_temp_root, '384'), dest_384, "Task A (384px for M2)")
    
    # Task B: 126 processing
    process_group(os.path.join(sr_temp_root, '126'), dest_126, "Task B (126px for A3)")

def process_group(src_root, dst_root, task_name):
    print(f"\n=== Processing {task_name} ===")
    if not os.path.exists(src_root):
        print(f"Skipping {task_name}: Source folder '{src_root}' not found.")
        print(f"Expected structure: {src_root}/[train|val|test]")
        return

    count = 0
    for s in ['train', 'val', 'test']:
        src_dir = os.path.join(src_root, s)
        dst_dir = os.path.join(dst_root, s)
        
        if not os.path.exists(src_dir):
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        
        # Find all SR images
        sr_files = glob.glob(os.path.join(src_dir, '*_SR.png'))
        
        for f_path in tqdm(sr_files, desc=f"Importing {s}"):
            filename = os.path.basename(f_path)
            
            # Rename: ls_000xxx_SR.png -> ls_000xxx.png
            if filename.endswith('_SR.png'):
                new_name = filename.replace('_SR.png', '.png')
                dst_img_path = os.path.join(dst_dir, new_name)
                
                # 1. Copy Image
                shutil.copy(f_path, dst_img_path)
                
                # 2. Copy Label (Find from central labels)
                # Format: ls_000xxx.txt
                # Central Label Path: landslide_workdir/yolo_data/labels/{s}/{ls_000xxx.txt}
                label_name = new_name.replace('.png', '.txt')
                central_label_path = os.path.join('landslide_workdir/yolo_data/labels', s, label_name)
                
                if os.path.exists(central_label_path):
                    shutil.copy(central_label_path, os.path.join(dst_dir, label_name))
                
                count += 1
    
    print(f"Move complete. {count} images -> {dst_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='SR_temp', help='Source directory where SR results are saved')
    parser.add_argument('--dest', type=str, default='landslide_workdir/yolo_data', help='Prefix for destination YOLO data folder')
    args = parser.parse_args()

    finalize_sr_data(args.src, args.dest)
