import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

import roop.globals

import cv2
import numpy as np

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# Monkey patch SSL cho macOS
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:
    # Sử dụng avg_frame_rate để lấy FPS trung bình
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        target_path
    ]
    try:
        output = subprocess.check_output(command).decode().strip().split('/')
        if len(output) == 2:
            numerator, denominator = map(int, output)
            if denominator == 0:
                return 30
            fps = numerator / denominator
        else:
            fps = 30
        # Nếu FPS quá cao (ví dụ > 100), ta coi là lỗi và đặt lại 30 FPS
        if fps > 100:
            fps = 30
        return fps
    except Exception:
        pass
    return 30


def extract_frames(target_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100
    return run_ffmpeg([
        '-i', target_path,
        '-start_number', '0',
        '-q:v', str(temp_frame_quality),
        '-pix_fmt', 'rgb24',
        '-vf', 'fps=' + str(fps),
        os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)
    ])


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level] + args
    try:
        output = subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print("Lỗi ffmpeg:", e.output.decode())
        return False


def create_video(target_path: str, fps: float = 30) -> bool:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    output_video_quality = (roop.globals.output_video_quality + 1) * 51 // 100

    # Kiểm tra xem có tồn tại các file đã được swap (hậu tố _swapped) hay không
    swapped_files = glob.glob(os.path.join(temp_directory_path, '*_swapped.' + roop.globals.temp_frame_format))
    if swapped_files:
        input_pattern = os.path.join(temp_directory_path, '%04d_swapped.' + roop.globals.temp_frame_format)
        print(f"[UTILITIES] Sử dụng file đã swap với pattern: {input_pattern}")
    else:
        input_pattern = os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)
        print(f"[UTILITIES] Sử dụng file gốc với pattern: {input_pattern}")

    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', input_pattern, '-c:v', roop.globals.output_video_encoder]

    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])

    commands.extend([
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-y', temp_output_path
    ])

    return run_ffmpeg(commands)


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    # Tìm các file có hậu tố _swapped, ví dụ: "0001_swapped.png"
    swapped_paths = glob.glob(os.path.join(temp_directory_path, '*_swapped.' + roop.globals.temp_frame_format))
    if swapped_paths:
        return sorted(swapped_paths)
    else:
        # Nếu không có file nào có hậu tố _swapped, trả về tất cả các file theo định dạng
        return sorted(glob.glob(os.path.join(temp_directory_path, '*.' + roop.globals.temp_frame_format)))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def clean_temp_directory_if_needed(target_path: str) -> None:
    """
    Kiểm tra thư mục temp của video có hợp lệ hay không. Nếu số lượng frame vượt quá dự kiến hoặc
    các frame đầu tiên giống nhau (có khả năng trích xuất lỗi), xóa thư mục để trích xuất lại.
    """
    temp_directory_path = get_temp_directory_path(target_path)
    if not os.path.exists(temp_directory_path):
        return

    expected_frames = get_video_frame_total(target_path)
    frame_pattern = os.path.join(temp_directory_path, '*.' + roop.globals.temp_frame_format)
    extracted_frames = sorted(glob.glob(frame_pattern))
    num_extracted = len(extracted_frames)

    # Điều kiện 1: Số frame vượt quá 120% số dự kiến
    if num_extracted > expected_frames * 1.2:
        print(f"Đã phát hiện {num_extracted} frame, vượt quá dự kiến {expected_frames}. Xóa thư mục temp để trích xuất lại.")
        shutil.rmtree(temp_directory_path)
        os.makedirs(temp_directory_path, exist_ok=True)
        return

    # Điều kiện 2: Kiểm tra nếu 100 frame đầu tiên giống hệt nhau
    duplicate = True
    if extracted_frames:
        first_frame = cv2.imread(extracted_frames[0])
        for f in extracted_frames[1:min(101, num_extracted)]:
            frame = cv2.imread(f)
            if not np.array_equal(first_frame, frame):
                duplicate = False
                break
    if duplicate and num_extracted > 1:
        print("Các frame đầu tiên giống hệt nhau, có khả năng trích xuất bị lỗi. Xóa thư mục temp để trích xuất lại.")
        shutil.rmtree(temp_directory_path)
        os.makedirs(temp_directory_path, exist_ok=True)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
