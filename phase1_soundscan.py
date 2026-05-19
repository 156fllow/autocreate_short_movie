import librosa
import numpy as np
import warnings

# librosa内部のFutureWarning（audioread関連）がうるさい場合はここで抑制可能
warnings.filterwarnings("ignore", category=FutureWarning)

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
    
    margined_segments = []
    for start, end in raw_segments:
        s = max(0.0, start - pre_margin)
        e = end + post_margin
        margined_segments.append((s, e))
        
    merged = []
    for current in margined_segments:
        if not merged:
            merged.append(current)
        else:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = current
            if curr_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(current)
    return merged

def get_audio_spikes(video_path, threshold_k=2.5):
    print(f"Analyzing audio from {video_path}...")
    # 16kHzモノラルでロード（処理高速化）
    y, sr = librosa.load(video_path, sr=16000, mono=True)
    
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    threshold = mean_rms + threshold_k * std_rms
    
    frames = np.where(rms > threshold)[0]
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    segments = merge_adjacent_times(times)
    return segments

if __name__ == "__main__":
    video_file = "forza.mp4"
    detected_segments = get_audio_spikes(video_file)
    
    print(f"\n--- Detected {len(detected_segments)} Potential Highlights ---")
    for i, (start, end) in enumerate(detected_segments):
        print(f"Segment {i+1}: {start:.2f}s -> {end:.2f}s (Duration: {end-start:.2f}s)")