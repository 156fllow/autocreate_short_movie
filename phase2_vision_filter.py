import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==========================================
# ⚙️ 設定パラメータ (RTX 5070 Ti 本番仕様)
# ==========================================
VIDEO_PATH = "forza.mp4"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # 🔥 2Bから「7B」の最高精度モデルへ変更

# 走行中のアバター位置に合わせた高精度なROI初期値 (X, Y, Width, Height)
ROI_COORDS = (1420, 220, 462, 360) 

INPUT_SEGMENTS = [
    {"id": 152, "start": 10734.36, "end": 10754.39},
    {"id": 153, "start": 10763.86, "end": 10801.69},
    {"id": 154, "start": 10888.41, "end": 10909.14},
    {"id": 155, "start": 10920.09, "end": 10974.46},
    {"id": 156, "start": 10984.06, "end": 11015.77},
    {"id": 157, "start": 11036.18, "end": 11071.42},
    {"id": 158, "start": 11072.22, "end": 11106.23},
    {"id": 159, "start": 11112.54, "end": 11132.73},
    {"id": 160, "start": 11142.97, "end": 11163.16},
    {"id": 161, "start": 11216.50, "end": 11256.28},
    {"id": 162, "start": 11321.30, "end": 11374.26}
]

# ==========================================
# 1. 高密度オプティカルフロー（モーション解析）
# ==========================================
def analyze_segment_motion(video_path, start_time, end_time, roi):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    rx, ry, rw, rh = roi
    ret, prev_frame = cap.read()
    if not ret:
        return 0, start_time
        
    prev_roi = prev_frame[ry:ry+rh, rx:rx+rw]
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    
    max_motion_score = 0.0
    peak_timestamp = start_time
    frame_interval = 3
    frame_count = 0
    
    while cap.isOpened():
        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_sec = current_ms / 1000.0
        if current_sec > end_time:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
            
        roi_img = frame[ry:ry+rh, rx:rx+rw]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        current_motion = np.mean(magnitude)
        
        if current_motion > max_motion_score:
            max_motion_score = current_motion
            peak_timestamp = current_sec
            
        prev_gray = gray
        
    cap.release()
    return max_motion_score, peak_timestamp

def extract_peak_frame(video_path, timestamp, roi, output_filename="temp_peak.jpg"):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    if ret:
        rx, ry, rw, rh = roi
        roi_img = frame[ry:ry+rh, rx:rx+rw]
        cv2.imwrite(output_filename, roi_img)
    cap.release()
    return output_filename

# ==========================================
# 2. メインパイプラインの実行
# ==========================================
def main():
    print("Initializing PRODUCTION Local VLM (Qwen2-VL-7B) on RTX 5070 Ti...")
    
    # 🔥 【修正】bfloat16精度の有効化、device_map='auto'によるVRAM最適常駐
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # 🔥 【修正】min_pixels / max_pixels の制限を撤廃（アバターを高解像度で認識させるため）
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    verified_highlights = []
    motion_threshold = 1.2 

    print("\n--- Starting Phase 2: High-Context AI Validation ---")
    for seg in INPUT_SEGMENTS:
        print(f"\n[Segment {seg['id']}] Processing ({seg['start']}s -> {seg['end']}s)")
        
        max_motion, peak_time = analyze_segment_motion(VIDEO_PATH, seg['start'], seg['end'], ROI_COORDS)
        print(f"  -> Max Motion Score: {max_motion:.4f} (Peak at {peak_time:.2f}s)")
        
        if max_motion < motion_threshold:
            print("  -> ❌ Rejected: Low motion (Likely game audio spike only)")
            continue
            
        frame_path = extract_peak_frame(VIDEO_PATH, peak_time, ROI_COORDS)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_path},
                    {
                        "type": "text", 
                        "text": "これはVTuberのゲーム実況動画のワイプ（アバター領域）です。アバターの表情（口の開き方、目の形）のコンテキストを解析し、[大爆笑, 絶叫, 驚愕, 通常・その他] の中から最も適切なものを1つだけ返してください。思考プロセスやそれ以外の文字は一切出力しないでください。"
                    }
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda") # 明示的にcuda(VRAM)へ転送
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=15)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
        print(f"  -> AI Expression Analysis: 【{output_text}】")
        
        if output_text in ["大爆笑", "絶叫", "驚愕"]:
            print(f"  ->  Approved for Shorts!")
            verified_highlights.append({
                "id": seg['id'],
                "start": seg['start'],
                "end": seg['end'],
                "peak_time": peak_time,
                "emotion": output_text,
                "motion": max_motion
            })
        else:
            print("  -> ❌ Rejected: Visual analysis confirmed no peak reaction.")
            
        if os.path.exists(frame_path):
            os.remove(frame_path)

    print("\n=============================================")
    print(f"🎉 Phase 2 Production Complete. Found {len(verified_highlights)} High-Quality Highlights!")
    print("=============================================")
    for hl in verified_highlights:
        print(f"{{'id': {hl['id']}, 'start': {hl['start']:.2f}, 'end': {hl['end']:.2f}, 'emotion': '{hl['emotion']}', 'motion': {hl['motion']:.2f}}},")

if __name__ == "__main__":
    main()