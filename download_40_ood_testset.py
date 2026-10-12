import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform
import os

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    ignore_conformance=True,
)

# 40 個地點：完全避開極端高山、沙漠、沼澤、台灣。確保影像乾淨、數值不破表。
LOCATIONS_40 = {
    # == 亞洲與紐澳 (15) ==
    "Urban_Tokyo": [139.76, 35.68],
    "Urban_Seoul": [126.97, 37.56],
    "Urban_Beijing": [116.40, 39.90],
    "Urban_Shanghai": [121.47, 31.23],
    "Urban_HongKong": [114.16, 22.28],
    "Urban_Singapore": [103.81, 1.35],
    "Urban_Jakarta": [106.84, -6.20],
    "Urban_Mumbai": [72.87, 19.07],
    "Urban_Delhi": [77.20, 28.61],
    "Urban_KualaLumpur": [101.68, 3.13],
    "Urban_Bangkok": [100.50, 13.75],
    "Urban_HoChiMinh": [106.62, 10.82],
    "Urban_Sydney": [151.20, -33.86],
    "Urban_Melbourne": [144.96, -37.81],
    "Urban_Auckland": [174.76, -36.84],

    # == 歐洲 (10) ==
    "Urban_London": [-0.12, 51.50],
    "Urban_Paris": [2.35, 48.85],
    "Urban_Berlin": [13.40, 52.52],
    "Urban_Madrid": [-3.70, 40.41],
    "Urban_Rome": [12.49, 41.90],
    "Urban_Moscow": [37.61, 55.75],
    "Urban_Istanbul": [28.97, 41.00],
    "Urban_Vienna": [16.37, 48.20],
    "Urban_Athens": [23.72, 37.98],
    "Urban_Amsterdam": [4.90, 52.36],

    # == 美洲 (11) ==
    "Urban_NewYork": [-74.00, 40.71],
    "Urban_LosAngeles": [-118.24, 34.05],
    "Urban_Chicago": [-87.62, 41.87],
    "Urban_Toronto": [-79.38, 43.65],
    "Urban_Vancouver": [-123.12, 49.28],
    "Urban_Seattle": [-122.33, 47.60],
    "Urban_SanFrancisco": [-122.41, 37.77],
    "Urban_MexicoCity": [-99.13, 19.43],
    "Urban_SaoPaulo": [-46.63, -23.55],
    "Urban_BuenosAires": [-58.38, -34.60],
    "Urban_RioDeJaneiro": [-43.17, -22.90],

    # == 非洲與中東良性區域 (4) ==
    "Urban_CapeTown": [18.42, -33.92],
    "Urban_Johannesburg": [28.04, -26.20],
    "Urban_TelAviv": [34.78, 32.08],
    "Forest_CongoBasin": [19.00, -1.00], # 加上一個健康無雪的綠色植被區
}

# 下載路徑更名，代表這就是我們論文要用的最終 Testset
OUTPUT_DIR = "dataset_40_OOD_testset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"總共準備下載 {len(LOCATIONS_40)} 個地區的 Sentinel-2 影像...")
WINDOW_HALF_SIZE = 600
success_count = 0

for name, (lon, lat) in LOCATIONS_40.items():
    print(f"\n[{name}] 正在搜尋 Sentinel-2 影像...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01],
        datetime="2023-01-01/2023-12-31",
        query={"eo:cloud_cover": {"lt": 5}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=10
    )
    
    items = list(search.items())
    if len(items) == 0:
        print(f"  -> [X] 找不到影像，跳過 {name}")
        continue
        
    for raw_item in items:
        item = planetary_computer.sign(raw_item)
        print(f"  -> 嘗試日期: {item.datetime.strftime('%Y-%m-%d')}")
        
        try:
            bands = []
            for b_name in ["B04", "B03", "B02"]:
                url = item.assets[b_name].href
                with rasterio.open(url) as src:
                    xs, ys = transform('EPSG:4326', src.crs, [lon], [lat])
                    row, col = src.index(xs[0], ys[0])
                    
                    if row - WINDOW_HALF_SIZE < 0 or col - WINDOW_HALF_SIZE < 0 or \
                       row + WINDOW_HALF_SIZE >= src.height or col + WINDOW_HALF_SIZE >= src.width:
                        raise ValueError("靠近邊緣")
                        
                    window = Window(col - WINDOW_HALF_SIZE, row - WINDOW_HALF_SIZE, WINDOW_HALF_SIZE * 2, WINDOW_HALF_SIZE * 2)
                    data = src.read(1, window=window)
                    
                    if (data == 0).sum() / data.size > 0.01:
                        raise ValueError("包含黑帶")
                    bands.append(data)
                    
            out_path = os.path.join(OUTPUT_DIR, f"{name}_RGB_10m.tif")
            with rasterio.open(url) as src: profile = src.profile
            profile.update(
                dtype=rasterio.uint16, count=3, height=WINDOW_HALF_SIZE*2, width=WINDOW_HALF_SIZE*2,
                transform=rasterio.windows.transform(window, src.transform), compress='lzw'
            )
            with rasterio.open(out_path, 'w', **profile) as dst:
                for i, b_data in enumerate(bands): dst.write(b_data, i + 1)
                    
            print(f"  -> [V] 成功下載: {out_path}")
            success_count += 1
            break # 成功就換下個地點
        except Exception as e:
            print(f"  -> 失敗 ({e})，換下一天...")

print(f"\n🎉 下載完成！共 {success_count} 張 OOD 測試影像存放於 {OUTPUT_DIR}。")
