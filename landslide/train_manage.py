import argparse
import os
import yaml
import cv2  # 必要的繪圖套件
import numpy as np  # 必要的數值計算套件
from ultralytics import YOLO

# ================= Configuration =================
WORK_DIR = 'landslide_workdir'
DATA_ROOT = os.path.join(WORK_DIR, 'yolo_data')
LABELS_DIR = os.path.join(WORK_DIR, 'yolo_data/labels')

# We will generate yamls in the root, so paths should be relative or absolute
DATASETS = {
    'img_128_GT': os.path.join(DATA_ROOT, 'img_128_GT'),
    'img_384_Bic': os.path.join(DATA_ROOT, 'img_384_Bic'),
    'img_384_SR': os.path.join(DATA_ROOT, 'img_384_SR'),

    # Scenarios specific
    'img_42_LR': os.path.join(DATA_ROOT, 'img_42_LR'),
    'img_128_Bic': os.path.join(DATA_ROOT, 'img_128_Bic'),  # For A2
    'img_126_SR': os.path.join(DATA_ROOT, 'img_126_SR'),  # For A3
}

# Model Save Paths
M1_PATH = 'runs/segment/runs/train/M1/weights/best.pt'
M2_PATH = 'runs/segment/runs/train/M2/weights/best.pt'


def create_data_yaml(name, train_img_dir, val_img_dir, test_img_dir):
    """Generates a temporary YAML file for YOLO."""
    content = {
        'path': os.path.abspath(WORK_DIR),  # Root for dataset
        'train': os.path.abspath(train_img_dir),
        'val': os.path.abspath(val_img_dir),
        'test': os.path.abspath(test_img_dir),
        'names': {0: 'landslide'}
    }

    filename = f"{name}.yaml"
    with open(filename, 'w') as f:
        yaml.dump(content, f)
    return filename


def train(model_name):
    print(f"=== Training Model {model_name} ===")

    if model_name == 'M1':
        # M1: Train on 128_GT
        data_yaml = create_data_yaml(
            'config_M1',
            train_img_dir=os.path.join(DATASETS['img_128_GT'], 'train'),
            val_img_dir=os.path.join(DATASETS['img_128_GT'], 'val'),
            test_img_dir=os.path.join(DATASETS['img_128_GT'], 'test')
        )
        weights = 'yolov8s-seg.pt'

    elif model_name == 'M2':
        # M2: Train on 384_SR
        if not os.path.exists(os.path.join(DATASETS['img_384_SR'], 'train')):
            print("Error: Training data for M2 (img_384_SR) not found.")
            return

        data_yaml = create_data_yaml(
            'config_M2',
            train_img_dir=os.path.join(DATASETS['img_384_SR'], 'train'),
            val_img_dir=os.path.join(DATASETS['img_384_SR'], 'val'),
            test_img_dir=os.path.join(DATASETS['img_384_SR'], 'test')
        )
        weights = 'yolov8s-seg.pt'
    else:
        print(f"Unknown model {model_name}")
        return

    # Initialize and Train
    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=200,
        patience=50,
        imgsz=640,
        batch=16,
        project='runs/train',
        name=model_name,
        exist_ok=True
    )
    print(f"Training {model_name} complete.")


def log_results(scenario_id, metrics):
    """Append results to a summary file."""
    with open('experiment_results.txt', 'a') as f:
        f.write(f"Scenario: {scenario_id}\n")
        f.write(f"  mAP50:    {metrics.box.map50:.4f}\n")
        f.write(f"  mAP50-95: {metrics.box.map:.4f}\n")
        f.write("-" * 30 + "\n")
    print(f"Results saved to experiment_results.txt")


def evaluate(scenario_id, m1_path=M1_PATH, m2_path=M2_PATH):
    print(f"=== Evaluating Scenario {scenario_id} ===")

    # Define Scenario Configuration
    scenarios = {
        'A1': ('M1', 'img_42_LR'),
        'A2': ('M1', 'img_128_Bic'),
        'A3': ('M1', 'img_126_SR'),
        'A4': ('M1', 'img_128_GT'),
        'B1': ('M2', 'img_128_GT'),
        'B2': ('M2', 'img_384_Bic'),
        'B3': ('M2', 'img_384_SR'),
    }

    if scenario_id == 'B1':
        model_name, dataset_key = 'M2', 'img_128_GT'
    elif scenario_id in scenarios:
        model_name, dataset_key = scenarios[scenario_id]
    else:
        print(f"Unknown scenario {scenario_id}")
        return

    model_path = m1_path if model_name == 'M1' else m2_path
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    test_dir = os.path.join(DATASETS[dataset_key], 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test data not found at {test_dir}")
        return

    print(f"Model: {model_name}")
    print(f"Test Data: {dataset_key}")

    # Temporary YAML for Validation
    data_yaml = create_data_yaml(
        f'config_eval_{scenario_id}', train_img_dir=test_dir, val_img_dir=test_dir, test_img_dir=test_dir
    )

    model = YOLO(model_path)

    # --- Step 1: Metrics (計算分數) ---
    print(f"--- Step 1: Calculating Metrics ---")
    metrics = model.val(
        data=data_yaml, split='test', imgsz=640,
        project='runs/eval_metrics', name=scenario_id,
        exist_ok=True, save=False, plots=False
    )
    log_results(scenario_id, metrics)

    # --- Step 2: Custom Plotting (生成統一尺寸的圖片 + Mask) ---
    print(f"--- Step 2: Generating Uniform Prediction Images with Masks ---")

    # 設定目標輸出尺寸
    TARGET_SIZE = 512

    save_dir = os.path.join('runs', 'detect', scenario_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 執行預測
    results = model.predict(
        source=test_dir, imgsz=640, conf=0.25, iou=0.45,
        stream=True, verbose=False
    )

    count = 0
    for result in results:
        img = result.orig_img
        h, w = img.shape[:2]

        # 1. 計算縮放比例
        scale = TARGET_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 2. 縮放基礎圖片 (保持像素感)
        img_upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # ================= NEW: 繪製分割 Mask =================
        if result.masks is not None:
            # 建立一個圖層用來畫半透明 Mask
            overlay = img_upscaled.copy()

            # 取得每個 Mask 的多邊形座標 (xy)
            for seg in result.masks.xy:
                # 將座標依照比例放大
                seg_scaled = seg * scale
                seg_scaled = seg_scaled.astype(np.int32)

                # 繪製填充多邊形 (顏色 BGR: 藍色)
                cv2.fillPoly(overlay, [seg_scaled], (255, 0, 0))

            # 將 Mask 圖層以半透明方式疊加到原圖上
            alpha = 0.4  # 透明度 (0.0 ~ 1.0，越小越透明)
            img_upscaled = cv2.addWeighted(overlay, alpha, img_upscaled, 1 - alpha, 0)
        # ======================================================

        # 3. 繪製框線與文字 (這部分維持在最上層)
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        line_thickness = 2
        font_scale = 0.8
        font_thickness = 2

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # 座標轉換
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            conf = confs[i]

            # 畫框 (紅色)
            cv2.rectangle(img_upscaled, (x1, y1), (x2, y2), (0, 0, 255), thickness=line_thickness)

            # 標籤文字
            label = f"{conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_y = y1 - 5 if y1 - 5 > text_h else y1 + text_h + 5

            # 文字背景與文字
            cv2.rectangle(img_upscaled, (x1, text_y - text_h - 5), (x1 + text_w, text_y + baseline - 5), (0, 0, 255),
                          -1)
            cv2.putText(img_upscaled, label, (x1, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        font_thickness)

        # 4. 存檔
        filename = os.path.basename(result.path)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img_upscaled)
        count += 1

    print(f"✅ Success! Generated {count} images with masks (Fixed Size: {TARGET_SIZE}px) in: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training & Evaluation Manager")
    parser.add_argument('--train', type=str, choices=['M1', 'M2'], help='Train a model (M1 or M2)')
    parser.add_argument('--eval', type=str, choices=['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3'],
                        help='Evaluate a scenario')
    parser.add_argument('--m1', type=str, default=M1_PATH, help='M1 detector weights (default: trained by --train M1)')
    parser.add_argument('--m2', type=str, default=M2_PATH, help='M2 detector weights (default: trained by --train M2)')
    args = parser.parse_args()

    if args.train:
        train(args.train)
    elif args.eval:
        evaluate(args.eval, args.m1, args.m2)
    else:
        parser.print_help()