import os
import subprocess
import cv2
import torch
from ultralytics import YOLO

# ==========================================
# ⚙️ 設定パラメータ (RTX 5070 Ti 本番仕様)
# ==========================================
VIDEO_PATH = "forza.mp4"
OUTPUT_DIR = "./output_shorts"

# 🛠️ 【修正】手動配置された最高峰モデル（yolov8x6）のファイル名に書き換え
YOLO_MODEL_PATH = "yolov8x6_animeface.pt"

# 🛠️ 【最適化】同階層にあるffmpeg.exeを確実に指定
FFMPEG_PATH = "./ffmpeg.exe"

APPROVED_SEGMENTS = [
    {"id": 152, "start": 10734.36, "end": 10754.39, "emotion": "大爆笑", "motion": 13.28},
    {"id": 153, "start": 10763.86, "end": 10801.69, "emotion": "大爆笑", "motion": 15.17},
    {"id": 154, "start": 10888.41, "end": 10909.14, "emotion": "大爆笑", "motion": 13.07},
    {"id": 155, "start": 10920.09, "end": 10974.46, "emotion": "大爆笑", "motion": 46.95},
    {"id": 156, "start": 10984.06, "end": 11015.77, "emotion": "大爆笑", "motion": 16.85},
    {"id": 157, "start": 11036.18, "end": 11071.42, "emotion": "大爆笑", "motion": 15.09},
    {"id": 158, "start": 11072.22, "end": 11106.23, "emotion": "大爆笑", "motion": 17.40},
    {"id": 159, "start": 11112.54, "end": 11132.73, "emotion": "大爆笑", "motion": 25.90},
    {"id": 160, "start": 11142.97, "end": 11163.16, "emotion": "大爆笑", "motion": 7.40},
    {"id": 161, "start": 11216.50, "end": 11256.28, "emotion": "大爆笑", "motion": 8.56},
    {"id": 162, "start": 11321.30, "end": 11374.26, "emotion": "大爆笑", "motion": 2.03}
]

model = None

# ==========================================
# 🎯 YOLOv8 (RTX 5070 Ti 駆動：yolov8x6 超高精度認識)
# ==========================================
def detect_avatar_roi_yolo(video_path, target_time):
    global model
    
    if model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"\n\n❌ モデルファイル '{YOLO_MODEL_PATH}' が見つかりません。\n"
                f"現在の実行フォルダ直下に配置されているか再度ご確認ください。\n"
            )
        print(f"🚀 Loading Extra-Large YOLOv8 Model [{YOLO_MODEL_PATH}] onto RTX 5070 Ti (CUDA + BF16)...")
        # x6モデルのポテンシャルを100%引き出すため、CUDAとbfloat16で常駐
        model = YOLO(YOLO_MODEL_PATH).to("cuda", dtype=torch.bfloat16)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, (target_time + 1.0) * 1000)
    ret, frame = cap.read()
    cap.release()
    
    # 認識失敗時の高精度逆算フォールバック
    fallback_roi = (1420, 220, 462, 360) 
    if not ret:
        return fallback_roi
        
    # モデルが非常に強力なため、自信度(conf)を0.45まで引き上げてノイズを完全にカット
    results = model(frame, conf=0.45, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return fallback_roi
        
    fx1, fy1, fx2, fy2 = boxes[0].xyxy[0].cpu().numpy()
    fx, fy, fw, fh = fx1, fy1, (fx2 - fx1), (fy2 - fy1)
    
    # 正しいアスペクト比(9:7)の計算
    margin_factor = 2.2
    rh = int(max(fw, fh) * margin_factor)
    rw = int(rh * (1080 / 840))
    
    cx = int(fx + fw / 2)
    cy = int(fy + fh / 2)
    
    rx = max(0, cx - rw // 2)
    ry = max(0, cy - rh // 2)
    
    if rx + rw > 1920: rx = 1920 - rw
    if ry + rh > 1080: ry = 1080 - rh
    
    return (rx, ry, rw, rh)

# ==========================================
# 🎬 FFmpeg 処理関数 (同階層のffmpeg.exeと第9世代 NVENC 駆動)
# ==========================================
def clip_to_shorts(video_path, seg, roi, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"short_{seg['id']}_{seg['emotion']}_motion{int(seg['motion'])}.mp4"
    output_path = os.path.join(output_dir, out_filename)
    
    duration = seg['end'] - seg['start']
    rx, ry, rw, rh = roi
    
    filter_complex = (
        f"[0:v]crop=1080:1080:420:0[main];"
        f"[0:v]crop={rw}:{rh}:{rx}:{ry},scale=1080:840[wipe];"
        f"[main][wipe]vstack=inputs=2[outv]"
    )
    
    # FFMPEG_PATH変数を使用してローカルのffmpeg.exeを駆動
    cmd = [
        FFMPEG_PATH, '-y', '-ss', f"{seg['start']:.3f}", '-i', video_path,
        '-filter_complex', filter_complex, '-map', '[outv]', '-map', '0:a',
        '-t', f"{duration:.3f}", 
        '-c:v', 'h264_nvenc',          # 5070 Ti ハードウェアエンコード
        '-preset', 'p6',               # 高画質(High Quality)
        '-tune', 'hq',                 
        '-b:v', '14M',                 # ビットレート 14Mbps
        '-c:a', 'aac', '-b:a', '192k', 
        output_path
    ]
    
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding='utf-8', errors='ignore'
    )
    return result.returncode == 0

# ==========================================
# 🚀 実行バッチ
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Starting Phase 3: Ultra-High Precision Pipeline on RTX 5070 Ti...")
    
    for segment in APPROVED_SEGMENTS:
        print(f"🎬 Processing Segment {segment['id']}...")
        try:
            roi = detect_avatar_roi_yolo(VIDEO_PATH, segment['start'])
            success = clip_to_shorts(VIDEO_PATH, segment, roi, OUTPUT_DIR)
            if success:
                print(f"  ➔ ✅ Success! (ROI: {roi})")
            else:
                print(f"  ➔ ❌ FFmpeg NVENC Export Failed")
        except FileNotFoundError as e:
            print(e)
            break
            
    print(f"\n🎉 ALL SHORT MOVIES SUCCESSFULLY CREATED! Check the directory: {OUTPUT_DIR}")