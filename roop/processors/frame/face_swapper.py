import typing
from typing import List, Callable

import os
import glob
import cv2
import insightface
import threading
import requests

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import (
    resolve_relative_path,
    is_image,
    is_video,
    extract_frames,
    detect_fps,
    create_video
)

FACE_SWAPPER: typing.Any = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"

def get_face_swapper() -> typing.Any:
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=roop.globals.execution_providers
            )
    return FACE_SWAPPER

def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None

def download_file(url: str, dest_path: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    actual_size = os.path.getsize(dest_path)
    if total_size and actual_size < total_size:
        raise IOError(f"File tải xuống không đầy đủ: {actual_size} < {total_size}")

def pre_check() -> bool:
    models_dir = resolve_relative_path("../models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(model_path):
        print(f"File mô hình không tồn tại tại {model_path}. Đang tải lại file model...")
        # URL tải về: đảm bảo đây là direct download link từ Hugging Face hoặc dịch vụ khác.
        model_url = "https://huggingface.co/TMElyralab/MuseV/resolve/9c911e064d6c3de4cf5a344a3c6a6981df8cd720/insightface/models/inswapper_128.onnx"
        try:
            download_file(model_url, model_path)
            print("Tải file mô hình thành công!")
        except Exception as e:
            print("Lỗi khi tải file mô hình:", e)
            return False
    return True

def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status("Chọn một ảnh làm nguồn (source).", NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status("Không phát hiện khuôn mặt trong ảnh nguồn (source).", NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status("Chọn một ảnh hoặc video làm mục tiêu (target).", NAME)
        return False
    return True

def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()

def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame

def get_total_video_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def get_temp_dir(video_path: str) -> str:
    # Giả sử các frame được lưu trong thư mục:
    # ../video/temp/<video_name>/
    video_name, _ = os.path.splitext(os.path.basename(video_path))
    return resolve_relative_path(os.path.join("..", "video", "temp", video_name))

def resume_processing_video(source_path: str, video_path: str) -> None:
    """
    Xử lý video theo yêu cầu:
      1. Đảm bảo số frame trong thư mục tạm đầy đủ so với video.
         Nếu thiếu, trích xuất bổ sung frame.
      2. Xác định resume index dựa trên các file đã được xử lý (có hậu tố '_swapped').
      3. Xử lý các file chưa được swap theo thứ tự tăng dần.
      4. Nếu tất cả các frame đã được swap, tạo video output.
    """
    temp_dir = get_temp_dir(video_path)
    ext = roop.globals.temp_frame_format or "png"
    total_frames = get_total_video_frames(video_path)
    
    # Chỉ nhận các file có tên bắt đầu bằng 4 chữ số (có thể có hoặc không có '_swapped')
    pattern = os.path.join(temp_dir, '[0-9][0-9][0-9][0-9]*.' + ext)
    all_files = sorted(glob.glob(pattern))
    
    if len(all_files) < total_frames:
        print(f"Chưa đủ frame: {len(all_files)} < {total_frames}. Đang trích xuất bổ sung frame...")
        extract_frames(video_path, fps=detect_fps(video_path))
        all_files = sorted(glob.glob(pattern))
    
    # Xác định chỉ số các file đã được swap (có hậu tố '_swapped')
    processed_indices = set()
    for file in all_files:
        base = os.path.basename(file)
        if "_swapped" in base:
            base_clean = base.replace("_swapped", "")
            try:
                idx = int(os.path.splitext(base_clean)[0])
                processed_indices.add(idx)
            except ValueError:
                continue

    resume_index = max(processed_indices) + 1 if processed_indices else 0

    # Xây dựng danh sách các file chưa được swap (với tên chỉ gồm 4 chữ số) và có chỉ số >= resume_index
    files_to_process = []
    for file in all_files:
        base = os.path.basename(file)
        if "_swapped" in base:
            continue
        try:
            idx = int(os.path.splitext(base)[0])
            if idx >= resume_index:
                files_to_process.append((idx, file))
        except ValueError:
            continue
    files_to_process.sort(key=lambda x: x[0])

    if not files_to_process:
        print("Tất cả các frame đã được xử lý. Đang tạo video output...")
        if create_video(video_path, detect_fps(video_path)):
            print("Tạo video output thành công!")
        else:
            print("Tạo video output thất bại!")
        return
    else:
        print(f"Tiếp tục xử lý các frame từ chỉ số {resume_index} đến {total_frames - 1}.")
        for idx, file in files_to_process:
            print(f"[FACE-SWAPPER] Đang xử lý frame {idx}: {file}", flush=True)
            img = cv2.imread(file)
            source_face_img = get_one_face(cv2.imread(source_path))
            reference_face = None if roop.globals.many_faces else get_face_reference()
            try:
                result = process_frame(source_face_img, reference_face, img)
            except Exception as e:
                print(f"[FACE-SWAPPER] Lỗi khi xử lý frame {file}: {e}", flush=True)
                continue
            new_filename = os.path.join(temp_dir, f"{idx:04d}_swapped.{ext}")
            cv2.imwrite(new_filename, result)
            print(f"[FACE-SWAPPER] Frame {file} đã được xử lý và lưu tại {new_filename}.", flush=True)
        # Sau khi xử lý lô file hiện tại, cập nhật danh sách và gọi đệ quy để tiếp tục xử lý nếu còn
        resume_processing_video(source_path, video_path)

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Hàm xử lý hàng loạt frame cho video.
    Nếu file đã được swap (tên chứa '_swapped') thì bỏ qua xử lý.
    Các frame chưa được xử lý sẽ được swap và lưu với hậu tố '_swapped'.
    Sau đó, danh sách temp_frame_paths được cập nhật chỉ chứa các file đã swap.
    """
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    total_frames = len(temp_frame_paths)
    processed_paths: List[str] = []
    for i, temp_frame_path in enumerate(temp_frame_paths, start=1):
        basename = os.path.basename(temp_frame_path)
        if '_swapped' in basename:
            print(f"[FACE-SWAPPER] Frame {temp_frame_path} đã được xử lý rồi, bỏ qua.", flush=True)
            processed_paths.append(temp_frame_path)
            continue
        print(f"[FACE-SWAPPER] Đang xử lý frame {i}/{total_frames}: {temp_frame_path}", flush=True)
        try:
            temp_frame = cv2.imread(temp_frame_path)
            result = process_frame(source_face, reference_face, temp_frame)
        except Exception as e:
            print(f"[FACE-SWAPPER] Lỗi khi xử lý frame {temp_frame_path}: {e}", flush=True)
            continue
        dirname = os.path.dirname(temp_frame_path)
        basename_no_ext, ext = os.path.splitext(basename)
        new_filename = os.path.join(dirname, f"{basename_no_ext}_swapped{ext}")
        cv2.imwrite(new_filename, result)
        processed_paths.append(new_filename)
        print(f"[FACE-SWAPPER] Frame {temp_frame_path} đã được xử lý và lưu tại {new_filename}.", flush=True)
        if update:
            update()
    temp_frame_paths[:] = processed_paths

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(
        target_frame, roop.globals.reference_face_position
    )
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Thay vì xử lý video trực tiếp, gọi hàm resume_processing_video để đảm bảo:
      - Nếu còn frame chưa được trích xuất hoặc chưa được swap, sẽ xử lý bổ sung.
      - Khi tất cả frame đã được swap, tạo video output.
    Sau khi xử lý xong, cập nhật lại danh sách các frame đã swap vào temp_frame_paths.
    """
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    resume_processing_video(source_path, roop.globals.target_path)
    # Sau khi resume_processing_video hoàn tất, cập nhật danh sách frame đã swap
    final_frames = sorted(glob.glob(os.path.join(get_temp_dir(roop.globals.target_path), '*_swapped.' + (roop.globals.temp_frame_format or 'png'))))
    temp_frame_paths[:] = final_frames
