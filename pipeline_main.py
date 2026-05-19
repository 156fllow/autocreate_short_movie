import os
import sys
import subprocess
import urllib.request
import cv2
import torch
import numpy as np
import warnings
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ultralytics import YOLO

# 共通の警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# ⚙️ グローバル設定パラメータ (RTX 5070 Ti 本番仕様)
# ==========================================
VIDEO_PATH = "forza.mp4"
OUTPUT_DIR = "./output_shorts"
FFMPEG_PATH = "./ffmpeg.exe"

# モデルファイル設定
YOLO_MODEL_PATH = "yolov8x6_animeface.pt"
VLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# 走行中のアバター初期位置 (X, Y, Width, Height) - フェーズ2のオプティカルフロー用
ROI_COORDS_INIT = (1420, 220, 462, 360) 

# ==========================================
# 🎵 PHASE 1: 音声エネルギー高速スキャン
# ==========================================
def merge_adjacent_times(times, max_gap=3.0, pre_margin=5.0, post_margin=15.0):
    if len(times) == 0: return []
    raw_segments = []
    seg_start = times[0]
    seg_end = times[0]
    for t in times[1:]:
        if t - seg_end <= max_gap:
            seg_end = t
        else:
            raw_segments.append((seg_start, seg_end))
            seg_start = t
            seg_end = t
    raw_segments.append((seg_start, seg_end))
    
    merged = []
    for start, end in raw_segments:
        s = max(0.0, start - pre_margin)
        e = end + post_margin
        if not merged:
            merged.append((s, e))
        else:
            prev_start, prev_end = merged[-1]
            if s <= prev_end:
                merged[-1] = (prev_start, max(prev_end, e))
            else:
                merged.append((s, e))
    return merged

def run_phase1_audio_scan(video_path, threshold_k=2.5):
    import librosa
    print(f"\n=============================================")
    print(f"🎵 PHASE 1: Scanning Audio from {video_path}...")
    print(f"=============================================")
    
    y, sr = librosa.load(video_path, sr=16000, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    threshold = np.mean(rms) + threshold_k * np.std(rms)
    frames = np.where(rms > threshold)[0]
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    raw_segments = merge_adjacent_times(times)
    segments = [{"id": i+1, "start": s, "end": e} for i, (s, e) in enumerate(raw_segments)]
    print(f"  ➔ Found {len(segments)} potential highlight candidates via audio spikes.")
    return segments

# ==========================================
# 👁️ PHASE 2: 高密度フロー ＆ VLMコンテキストバリデーション
# ==========================================
def analyze_motion(video_path, start, end, roi):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    rx, ry, rw, rh = roi
    ret, frame = cap.read()
    if not ret: return 0, start
    prev_gray = cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY)
    
    max_motion = 0.0
    peak_time = start
    frame_count = 0
    
    while cap.isOpened():
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_sec > end: break
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % 3 != 0: continue
        
        gray = cv2.cvtColor(frame[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion = np.mean(magnitude)
        
        if motion > max_motion:
            max_motion = motion
            peak_time = current_sec
        prev_gray = gray
    cap.release()
    return max_motion, peak_time

def run_phase2_vlm_filter(video_path, candidates):
    print(f"\n=============================================")
    print(f"🧠 PHASE 2: Validating Expressions with Qwen2-VL-7B...")
    print(f"=============================================")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    
    verified = []
    motion_threshold = 1.2
    
    for seg in candidates:
        max_motion, peak_time = analyze_motion(video_path, seg['start'], seg['end'], ROI_COORDS_INIT)
        if max_motion < motion_threshold: continue
        
        # ピークフレームの抽出
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, peak_time * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue
        
        rx, ry, rw, rh = ROI_COORDS_INIT
        temp_img_path = "temp_vlm_peak.jpg"
        cv2.imwrite(temp_img_path, frame[ry:ry+rh, rx:rx+rw])
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": temp_img_path},
                {"type": "text", "text": "これはVTuberのワイプ画面です。アバターの表情のコンテキストを解析し、[大爆笑, 絶叫, 驚愕, 通常・その他] の中から最も適切なものを1つだけ返してください。思考プロセスやそれ以外の文字は一切出力しないでください。"}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=15)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            
        if os.path.exists(temp_img_path): os.remove(temp_img_path)
        
        print(f"  🎬 Segment {seg['id']} -> Motion: {max_motion:.2f} | Emotion: 【{output_text}】")
        if output_text in ["大爆笑", "絶叫", "驚愕"]:
            verified.append({
                "id": seg['id'], "start": seg['start'], "end": seg['end'],
                "emotion": output_text, "motion": max_motion
            })
            
    # 🔥 【超重要】フェーズ3へ移行する前にVLMモデルをVRAMから完全にパージし、メモリを解放する
    del model
    del processor
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("  ➔ VLM cached memory cleared from RTX 5070 Ti.")
    
    return verified

# ==========================================
# 🎯 PHASE 3: YOLOv8x6による動的認識 ＆ FFmpeg NVENC切り出し
# ==========================================
def detect_avatar_roi_yolo(yolo_model, video_path, target_time):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, (target_time + 1.0) * 1000)
    ret, frame = cap.read()
    cap.release()
    
    fallback_roi = (1420, 220, 462, 360)
    if not ret: return fallback_roi
        
    results = yolo_model(frame, conf=0.45, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0: return fallback_roi
        
    fx1, fy1, fx2, fy2 = boxes[0].xyxy[0].cpu().numpy()
    fx, fy, fw, fh = fx1, fy1, (fx2 - fx1), (fy2 - fy1)
    
    margin_factor = 2.2
    rh = int(max(fw, fh) * margin_factor)
    rw = int(rh * (1080 / 840))
    cx, cy = int(fx + fw / 2), int(fy + fh / 2)
    rx, ry = max(0, cx - rw // 2), max(0, cy - rh // 2)
    
    if rx + rw > 1920: rx = 1920 - rw
    if ry + rh > 1080: ry = 1080 - rh
    return (rx, ry, rw, rh)

def run_phase3_render(video_path, approved_list):
    print(f"\n=============================================")
    print(f"🎬 PHASE 3: Dynamic Tracking & NVENC Render with YOLOv8x6...")
    print(f"=============================================")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"モデルファイル {YOLO_MODEL_PATH} が同一階層に見つかりません。")
        
    # パージされたクリーンなVRAMへ最重量x6モデルをロード
    yolo_model = YOLO(YOLO_MODEL_PATH).to("cuda", dtype=torch.bfloat16)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for seg in approved_list:
        roi = detect_avatar_roi_yolo(yolo_model, video_path, seg['start'])
        out_filename = f"short_{seg['id']}_{seg['emotion']}_motion{int(seg['motion'])}.mp4"
        output_path = os.path.join(OUTPUT_DIR, out_filename)
        duration = seg['end'] - seg['start']
        rx, ry, rw, rh = roi
        
        filter_complex = (
            f"[0:v]crop=1080:1080:420:0[main];"
            f"[0:v]crop={rw}:{rh}:{rx}:{ry},scale=1080:840[wipe];"
            f"[main][wipe]vstack=inputs=2[outv]"
        )
        
        cmd = [
            FFMPEG_PATH, '-y', '-ss', f"{seg['start']:.3f}", '-i', video_path,
            '-filter_complex', filter_complex, '-map', '[outv]', '-map', '0:a',
            '-t', f"{duration:.3f}", 
            '-c:v', 'h264_nvenc', '-preset', 'p6', '-tune', 'hq', '-b:v', '14M',
            '-c:a', 'aac', '-b:a', '192k', output_path
        ]
        
        print(f"  🎬 Exporting: {out_filename} ({duration:.2f}s)")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        print(f"    ➔ ✅ Complete! (ROI: {roi})")

# ==========================================
# 🚀 パイプライン・エントリーポイント
# ==========================================
if __name__ == "__main__":
    print("==================================================================")
    print("🌟 STARTING VTUBER AUTOMATIC SHORTS CREATION PIPELINE 🌟")
    print("==================================================================")
    
    if not torch.cuda.is_available():
        print("❌ Error: CUDA (GPU) is not available. Execution halted.")
        sys.exit(1)
        
    # --- PHASE 1 ---
    candidates = run_phase1_audio_scan(VIDEO_PATH, threshold_k=2.5)
    if not candidates:
        print("❌ No highlight candidates found in Phase 1. Exiting.")
        sys.exit(0)
        
    # --- PHASE 2 ---
    approved_highlights = run_phase2_vlm_filter(VIDEO_PATH, candidates)
    if not approved_highlights:
        print("❌ No highlights approved by VLM in Phase 2. Exiting.")
        sys.exit(0)
        
    # --- PHASE 3 ---
    run_phase3_render(VIDEO_PATH, approved_highlights)
    
    print(f"\n=============================================")
    print(f"🎉 PIPELINE EXECUTION SUCCESSFUL!")
    print(f"   Generated short clips saved in: {OUTPUT_DIR}")
    print(f"=============================================")