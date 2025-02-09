from typing import Any, List, Callable

import os
import glob
import cv2
import threading

from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_many_faces, get_one_face
from roop.face_reference import get_face_reference, set_face_reference
from roop.typing import Frame, Face
from roop.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video,
    extract_frames,
    detect_fps,
    create_video
)

FACE_ENHANCER: Any = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "face_enhancer"

def get_face_enhancer() -> Any:
    """
    Lấy model GFPGAN để tăng cường khuôn mặt.
    Nếu chưa có, tải model từ đường dẫn chỉ định.
    """
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            # Lưu ý: Điều chỉnh đường dẫn model nếu cần (tham khảo: https://github.com/TencentARC/GFPGAN/issues/399)
            FACE_ENHANCER = GFPGANer(model_path=model_path, upscale=1, device=get_device())
    return FACE_ENHANCER

def get_device() -> str:
    """
    Xác định thiết bị xử lý: nếu có GPU thì trả về 'cuda'; nếu có CoreML/MPS thì 'mps'; ngược lại trả về 'cpu'.
    """
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'

def clear_face_enhancer() -> None:
    """
    Xóa model tăng cường khuôn mặt khỏi bộ nhớ.
    """
    global FACE_ENHANCER
    FACE_ENHANCER = None

def pre_check() -> bool:
    """
    Kiểm tra và tải file model GFPGAN nếu chưa có.
    """
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/GFPGANv1.4.pth'])
    return True

def pre_start() -> bool:
    """
    Kiểm tra đầu vào: đảm bảo rằng target (ảnh hoặc video) được chọn.
    """
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Chọn một ảnh hoặc video làm mục tiêu (target).', NAME)
        return False
    return True

def post_process() -> None:
    """
    Thực hiện dọn dẹp sau khi xử lý.
    """
    clear_face_enhancer()

def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    """
    Tăng cường chất lượng khuôn mặt trong frame.
    """
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        with THREAD_SEMAPHORE:
            try:
                # GFPGANer trả về một tuple; chỉ lấy phần ảnh đã tăng cường
                _, _, temp_face = get_face_enhancer().enhance(temp_face, paste_back=True)
            except Exception as e:
                print(f"[FACE-ENHANCER] Lỗi khi tăng cường khuôn mặt: {e}", flush=True)
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    return temp_frame

def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """
    Áp dụng tăng cường cho tất cả các khuôn mặt trong frame.
    """
    many_faces = get_many_faces(temp_frame)
    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)
    return temp_frame

def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """
    Xử lý hàng loạt frame cho video:
      - Với mỗi frame, áp dụng tăng cường và ghi đè kết quả vào file.
    """
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = process_frame(None, None, temp_frame)
        except Exception as e:
            print(f"[FACE-ENHANCER] Lỗi khi xử lý frame {temp_frame_path}: {e}", flush=True)
            continue
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Xử lý ảnh: tăng cường khuôn mặt trong ảnh target.
    """
    target_frame = cv2.imread(target_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)

# --- Các hàm hỗ trợ xử lý video với tính năng resume ---

def get_total_video_frames(video_path: str) -> int:
    """
    Lấy tổng số frame của video sử dụng OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def get_temp_dir(video_path: str) -> str:
    """
    Xác định thư mục chứa các frame tạm dựa trên tên video.
    Ví dụ: nếu video là 'video23.mp4' thì thư mục sẽ là ../video/temp/video23
    """
    video_name, _ = os.path.splitext(os.path.basename(video_path))
    return resolve_relative_path(os.path.join("..", "video", "temp", video_name))

def resume_processing_video(source_path: str, video_path: str) -> None:
    """
    Xử lý video theo yêu cầu:
      1. Đảm bảo số frame trong thư mục tạm đầy đủ so với video.
         Nếu thiếu, trích xuất bổ sung.
      2. Xác định resume index dựa trên các file đã được xử lý (có hậu tố '_swapped').
      3. Xử lý các file chưa được swap theo thứ tự tăng dần.
      4. Nếu tất cả các frame đã được swap, tạo video đầu ra.
    """
    temp_dir = get_temp_dir(video_path)
    ext = roop.globals.temp_frame_format or "png"
    total_frames = get_total_video_frames(video_path)
    
    # Lấy danh sách file theo định dạng: tên bắt đầu bằng 4 chữ số (có thể có hoặc không có '_swapped')
    pattern = os.path.join(temp_dir, '[0-9][0-9][0-9][0-9]*.' + ext)
    all_files = sorted(glob.glob(pattern))
    
    if len(all_files) < total_frames:
        print(f"Chưa đủ frame: {len(all_files)} < {total_frames}. Đang trích xuất bổ sung frame...")
        extract_frames(video_path, fps=detect_fps(video_path))
        all_files = sorted(glob.glob(pattern))
    
    # Xác định các chỉ số đã được swap (file có hậu tố '_swapped')
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

    # Tạo danh sách các file chưa được swap (chỉ nhận file có tên chính xác 4 chữ số)
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
        print("Tất cả các frame đã được xử lý. Đang tạo video đầu ra...")
        if create_video(video_path, detect_fps(video_path)):
            print("Tạo video đầu ra thành công!")
        else:
            print("Tạo video đầu ra thất bại!")
        return
    else:
        print(f"Tiếp tục xử lý các frame từ chỉ số {resume_index} đến {total_frames - 1}.")
        for idx, file in files_to_process:
            print(f"[FACE-ENHANCER] Đang xử lý frame {idx}: {file}", flush=True)
            img = cv2.imread(file)
            # Với face_enhancer, không cần khuôn mặt nguồn
            source_face_img = None
            reference_face = None if roop.globals.many_faces else get_face_reference()
            try:
                result = process_frame(source_face_img, reference_face, img)
            except Exception as e:
                print(f"[FACE-ENHANCER] Lỗi khi xử lý frame {file}: {e}", flush=True)
                continue
            new_filename = os.path.join(temp_dir, f"{idx:04d}_swapped.{ext}")
            cv2.imwrite(new_filename, result)
            print(f"[FACE-ENHANCER] Frame {file} đã được xử lý và lưu tại {new_filename}.", flush=True)
        # Sau khi xử lý lô file hiện tại, gọi lại đệ quy để tiếp tục nếu còn file chưa swap
        resume_processing_video(source_path, video_path)

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Thay vì xử lý video trực tiếp, gọi hàm resume_processing_video để đảm bảo:
      - Nếu còn frame chưa được trích xuất hoặc chưa được swap, sẽ xử lý bổ sung.
      - Khi tất cả các frame đã được swap, tạo video đầu ra.
    Sau đó, cập nhật lại danh sách các frame đã được swap vào temp_frame_paths.
    """
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    resume_processing_video(source_path, roop.globals.target_path)
    final_frames = sorted(glob.glob(os.path.join(get_temp_dir(roop.globals.target_path), '*_swapped.' + (roop.globals.temp_frame_format or 'png'))))
    temp_frame_paths[:] = final_frames
