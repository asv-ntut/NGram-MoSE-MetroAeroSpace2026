import os
import sys

try:
    root = "dataset_self_supervised"
    sets = ["train_lr", "val_lr", "test_lr"]
    files = {}
    
    print(f"Checking in root: {os.path.abspath(root)}")

    if not os.path.exists(root):
        print(f"Error: Root folder {root} not found!")
        sys.exit(1)

    for s in sets:
        folder = os.path.join(root, s)
        if not os.path.exists(folder):
            print(f"Set {s} does not exist at {folder}")
            files[s] = set()
            continue
        
        # Filter for image files only
        f_list = set([f for f in os.listdir(folder) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
        files[s] = f_list
        print(f"Set {s}: {len(f_list)} files")

    train_files = files.get("train_lr", set())
    val_files = files.get("val_lr", set())
    test_files = files.get("test_lr", set())

    # Check intersections
    train_val = train_files.intersection(val_files)
    train_test = train_files.intersection(test_files)
    val_test = val_files.intersection(test_files)

    print("\n--- Leakage Report ---")
    
    if train_val:
        print(f"[FAIL] Leakage TRAIN-VAL: {len(train_val)} files shared.")
    else:
        print("[PASS] TRAIN-VAL: Clean")

    if train_test:
        print(f"[FAIL] Leakage TRAIN-TEST: {len(train_test)} files shared.")
        print(f"Sample: {list(train_test)[:3]}")
    else:
        print("[PASS] TRAIN-TEST: Clean")

    if val_test:
        print(f"[FAIL] Leakage VAL-TEST: {len(val_test)} files shared.")
        print(f"Sample: {list(val_test)[:3]}")
    else:
        print("[PASS] VAL-TEST: Clean")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
