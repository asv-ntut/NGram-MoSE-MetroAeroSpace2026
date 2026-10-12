import os
import rasterio
import numpy as np
import time
from pystac_client import Client

# ================= Configuration =================
# Training Data Locations (Urban + Diverse Terrains)
LOCATIONS = {
    "Taipei_City":       [121.56, 25.03], # Dense Urban + Mountains
    "Taichung_City":     [120.67, 24.14], # Urban + Rural
    "Tainan_City":       [120.20, 22.99], # Flat Urban
    "Kaohsiung_Port":    [120.28, 22.62], # Port + Industrial
    "Central_Mountain":  [121.00, 23.50], # High Altitude, Shadows, Landslides
}

DOWNLOAD_DIR = "dataset_source_raw"
CROP_SIZE = 5120  # 5120x5120 pixels.
MAX_CLOUD_COVER = 5 # Strict cloud filter

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def get_sentinel2_scene(lon, lat, name):
    print(f"\n[{name}] Searching Sentinel-2 (L2A) ...")
    
    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime="2023-01-01/2023-12-30", 
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}, 
        max_items=20 
    )
    
    items = list(search.items())
    if not items:
        print(f" -> [X] No images found for {name}")
        return None

    items.sort(key=lambda x: x.properties['eo:cloud_cover'])
    best_item = items[0]
    
    print(f" -> Best Scene: {best_item.id}")
    print(f" -> Cloud Cover: {best_item.properties['eo:cloud_cover']}%")

    assets = best_item.assets
    if not all(k in assets for k in ["red", "green", "blue"]):
        print(" -> [!] Missing RGB bands.")
        return None
        
    band_urls = [assets["red"].href, assets["green"].href, assets["blue"].href]
    return band_urls, best_item.id

def download_window_and_stack(band_urls, name, scene_id):
    print(f" -> Downloading & Stacking {name}...")
    
    raw_channels = []
    
    with rasterio.open(band_urls[0]) as src:
        h, w = src.height, src.width
        center_x, center_y = w // 2, h // 2
        
        left = max(0, center_x - CROP_SIZE // 2)
        top = max(0, center_y - CROP_SIZE // 2)
        real_w = min(CROP_SIZE, w - left)
        real_h = min(CROP_SIZE, h - top)
        
        window = rasterio.windows.Window(left, top, real_w, real_h)

    for i, url in enumerate(band_urls):
        band_name = ['R', 'G', 'B'][i]
        success = False
        for attempt in range(3):
            try:
                with rasterio.open(url) as src:
                    data = src.read(1, window=window)
                    raw_channels.append(data)
                    success = True
                    break
            except Exception as e:
                print(f"   -> Error fetching {band_name}: {e}")
                time.sleep(2)
        
        if not success:
            return False

    img_stack = np.stack(raw_channels, axis=0) # (3, H, W)
    
    out_path = os.path.join(DOWNLOAD_DIR, f"{name}_{scene_id}.tif")
    
    with rasterio.open(
        out_path, 'w', driver='GTiff',
        height=img_stack.shape[1], width=img_stack.shape[2],
        count=3, dtype=img_stack.dtype
    ) as dst:
        dst.write(img_stack)
        
    print(f" [V] Saved: {out_path}")
    return True

def main():
    print("=== Sentinel-2 Training Data Downloader ===")
    count = 0
    for name, coords in LOCATIONS.items():
        res = get_sentinel2_scene(coords[0], coords[1], name)
        if res:
            urls, scene_id = res
            if download_window_and_stack(urls, name, scene_id):
                count += 1
    print(f"\nDone. Downloaded {count}/{len(LOCATIONS)} scenes.")

if __name__ == "__main__":
    main()
