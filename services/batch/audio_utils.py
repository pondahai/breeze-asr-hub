import subprocess
import re
import math
import pathlib
import datetime

def get_audio_duration(file_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries',
           'format=duration', '-of',
           'default=noprint_wrappers=1:nokey=1', str(file_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(res.stdout.strip())
    except:
        return 0.0

def find_silences(file_path):
    cmd = ['ffmpeg', '-i', str(file_path), '-af', 'silencedetect=noise=-30dB:d=0.5', '-f', 'null', '-']
    res = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    starts = [float(x) for x in re.findall(r'silence_start: ([\d\.]+)', res.stderr)]
    ends = [float(x) for x in re.findall(r'silence_end: ([\d\.]+)', res.stderr)]
    # Filter valid pairs
    silences = []
    for s, e in zip(starts, ends):
        silences.append((s, e))
    return silences

def calculate_splits(duration, silences, target_len=600):
    if duration <= target_len + 60: # No need to split if it's close enough
        return [(0, duration)]
    
    splits = []
    current_start = 0.0
    
    while current_start < duration:
        expected_end = current_start + target_len
        if expected_end >= duration:
            splits.append((current_start, duration))
            break
        
        # Find the best silence near expected_end
        # Look for silences within [expected_end - 60, expected_end + 60]
        valid_silences = [s for s in silences if s[0] > current_start + 60 and s[0] < expected_end + 60]
        
        if valid_silences:
            # Pick the one closest to expected_end
            best_silence = min(valid_silences, key=lambda x: abs(x[0] - expected_end))
            split_point = (best_silence[0] + best_silence[1]) / 2.0
            splits.append((current_start, split_point))
            current_start = split_point
        else:
            # Hard split if no silence found
            splits.append((current_start, expected_end))
            current_start = expected_end
            
    return splits

def split_audio(file_path, splits, out_dir, base_name):
    segment_paths = []
    out_dir = pathlib.Path(out_dir)
    for i, (start, end) in enumerate(splits):
        out_path = out_dir / f"{base_name}_{i:03d}.wav"
        # Convert to 16kHz 16-bit mono wav with DSP speech enhancement filters
        cmd = [
            'ffmpeg', '-y', '-i', str(file_path),
            '-ss', str(start), '-to', str(end),
            '-af', 'highpass=f=80,lowpass=f=7500,equalizer=f=2500:t=q:w=1.5:g=3,loudnorm=I=-16:TP=-1.0:LRA=11',
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
            str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        segment_paths.append((out_path, start))

    return segment_paths

def parse_time(time_str, is_srt):
    # srt: 00:00:05,000, vtt: 00:00:05.000
    sep = ',' if is_srt else '.'
    parts = time_str.split(sep)
    ms = int(parts[1]) if len(parts) > 1 else 0
    h, m, s = map(int, parts[0].split(':'))
    return datetime.timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

def format_time(td, is_srt):
    total_sec = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    sep = ',' if is_srt else '.'
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

def merge_subtitles(files_with_offsets, output_path, is_srt=True):
    merged_lines = []
    if not is_srt:
        merged_lines.append("WEBVTT\n\n")
        
    sub_index = 1
    
    for file_path, offset_sec in files_with_offsets:
        if not pathlib.Path(file_path).exists():
            continue
            
        offset_td = datetime.timedelta(seconds=offset_sec)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            # Skip WEBVTT header
            if not is_srt and line == "WEBVTT":
                i += 1
                continue
                
            # Parse timestamp line
            if '-->' in line:
                start_str, end_str = line.split(' --> ')
                start_td = parse_time(start_str.strip(), is_srt) + offset_td
                end_td = parse_time(end_str.strip(), is_srt) + offset_td
                
                if is_srt:
                    merged_lines.append(f"{sub_index}\n")
                    sub_index += 1
                    
                merged_lines.append(f"{format_time(start_td, is_srt)} --> {format_time(end_td, is_srt)}\n")
                
                # Copy text lines
                i += 1
                while i < len(lines) and lines[i].strip():
                    merged_lines.append(lines[i])
                    i += 1
                merged_lines.append("\n")
            else:
                i += 1
                
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(merged_lines)

def merge_texts(files, output_path):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file_path in files:
            if not pathlib.Path(file_path).exists():
                continue
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                if not infile.read().endswith('\n'):
                     outfile.write('\n')
