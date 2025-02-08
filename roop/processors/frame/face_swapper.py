from typing import Any, List, Callable
import os
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
)


# Biến toàn cục cho model face swapper và lock để đảm bảo an toàn thread
FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "ROOP.FACE-SWAPPER"


def get_face_swapper() -> Any:
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
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def pre_check() -> bool:
    models_dir = resolve_relative_path("../models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(model_path):
        print(f"File mô hình không tồn tại tại {model_path}. Đang tải từ Dropbox...")
        dropbox_url = "https://www.dropbox.com/scl/fi/qngqah0ni6dz58afhnpxq/inswapper_128.onnx?rlkey=t9bri158thcjgsiqwccqnhy4n&st=64q9s0qg&dl=1"
        try:
            download_file(dropbox_url, model_path)
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


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Hàm xử lý hàng loạt frame cho video.
    Nếu file đã được swap (tên chứa '_swapped') thì bỏ qua xử lý.
    Các frame chưa được xử lý sẽ được swap và lưu vào file mới có hậu tố '_swapped'.
    Cuối cùng, danh sách temp_frame_paths được cập nhật chỉ chứa các file đã swap.
    """
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    total_frames = len(temp_frame_paths)
    processed_paths: List[str] = []  # Danh sách các file đã được xử lý
    for i, temp_frame_path in enumerate(temp_frame_paths, start=1):
        basename = os.path.basename(temp_frame_path)
        # Nếu tên file đã chứa '_swapped', tức file đó đã được xử lý rồi → bỏ qua xử lý
        if '_swapped' in basename:
            print(f"[FACE-SWAPPER] Frame {temp_frame_path} đã được xử lý rồi, bỏ qua.", flush=True)
            processed_paths.append(temp_frame_path)
            continue
        # Nếu file chưa được xử lý, tiến hành xử lý
        print(f"[FACE-SWAPPER] Đang xử lý frame {i}/{total_frames}: {temp_frame_path}", flush=True)
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        # Tạo tên file mới với hậu tố '_swapped'
        dirname = os.path.dirname(temp_frame_path)
        basename_no_ext, ext = os.path.splitext(basename)
        new_filename = os.path.join(dirname, f"{basename_no_ext}_swapped{ext}")
        cv2.imwrite(new_filename, result)
        processed_paths.append(new_filename)
        print(f"[FACE-SWAPPER] Frame {temp_frame_path} đã được xử lý và lưu tại {new_filename}.", flush=True)
        if update:
            update()
    # Cập nhật lại danh sách frame để sau này ghép video sẽ chỉ sử dụng các file đã swap
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
    Hàm xử lý video:
      - Nếu chưa có khuôn mặt tham chiếu, lấy từ frame chỉ định.
      - Sau đó, gọi hàm xử lý hàng loạt frame để tiến hành swap khuôn mặt.
    """
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )
