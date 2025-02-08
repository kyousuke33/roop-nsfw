#!/usr/bin/env python3

import os
import sys
# Đặt biến môi trường: sử dụng single thread giúp cải thiện hiệu suất với CUDA – cần thiết lập trước khi import torch
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# Giảm mức log của tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import (
    has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, 
    get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
)

# Bỏ qua một số cảnh báo không cần thiết
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def parse_args() -> None:
    """
    Hàm parse_args() xử lý các đối số dòng lệnh.
    Đã thêm tham số --resume-frame để hỗ trợ tiếp tục xử lý video từ một frame cụ thể.
    """
    # Bắt sự kiện Ctrl+C
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    
    program.add_argument('-s', '--source', help='Chọn ảnh khuôn mặt', dest='source_path')
    program.add_argument('-t', '--target', help='Chọn ảnh hoặc video đầu vào', dest='target_path')
    program.add_argument('-o', '--output', help='Chọn file hoặc thư mục đầu ra', dest='output_path')
    program.add_argument('--frame-processor', help='Các bộ xử lý frame (ví dụ: face_swapper, face_enhancer, ...)', 
                         dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='Giữ nguyên FPS của video đầu vào', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='Lưu các frame tạm thời', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='Bỏ qua âm thanh của video đầu vào', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='Xử lý tất cả các khuôn mặt', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='Vị trí khuôn mặt tham chiếu', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='Số thứ tự frame tham chiếu', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='Khoảng cách khuôn mặt dùng để nhận diện', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='Định dạng ảnh cho frame tạm', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='Chất lượng ảnh cho frame tạm', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='Bộ mã hóa video đầu ra', dest='output_video_encoder', default='libx264', 
                         choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-quality', help='Chất lượng video đầu ra', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='Giới hạn RAM sử dụng (GB)', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='Execution provider (ví dụ: cpu, ...)', dest='execution_provider', 
                         default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='Số luồng xử lý', dest='execution_threads', type=int, default=suggest_execution_threads())
    # --- THÊM THAM SỐ HỖ TRỢ RESUME ---
    program.add_argument('--resume-frame', help='Số thứ tự frame bắt đầu xử lý (mặc định: 1)', 
                         dest='resume_frame', type=int, default=1)
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    # Gán các giá trị nhận được vào biến toàn cục của roop
    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads
    # Lưu giá trị resume_frame vào globals
    roop.globals.resume_frame = args.resume_frame

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1

def limit_resources() -> None:
    # Ngăn ngừa rò rỉ bộ nhớ của tensorflow
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # Giới hạn sử dụng bộ nhớ
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Phiên bản Python không được hỗ trợ - hãy nâng cấp lên 3.9 hoặc cao hơn.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg chưa được cài đặt.')
        return False
    return True

def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)

def start() -> None:
    """
    Hàm start() xử lý logic chính của quá trình chuyển đổi.
    Đối với ảnh: xử lý theo quy trình image-to-image.
    Đối với video: trích xuất frame, áp dụng tính năng resume (nếu có), xử lý từng frame, ghép lại video và phục hồi âm thanh.
    """
    # Kiểm tra các điều kiện trước khi bắt đầu
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return

    # Nếu đầu vào là ảnh
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Đang xử lý...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        if is_image(roop.globals.target_path):
            update_status('Xử lý ảnh thành công!')
        else:
            update_status('Xử lý ảnh thất bại!')
        return

    # Nếu đầu vào là video
    if predict_video(roop.globals.target_path):
        destroy()
    update_status('Đang tạo tài nguyên tạm thời...')
    create_temp(roop.globals.target_path)

    # Trích xuất frame nếu thư mục tạm chưa có
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if not temp_frame_paths:
        if roop.globals.keep_fps:
            fps = detect_fps(roop.globals.target_path)
            update_status(f'Đang trích xuất frame với FPS = {fps}...')
            extract_frames(roop.globals.target_path, fps)
        else:
            update_status('Đang trích xuất frame với FPS = 30...')
            extract_frames(roop.globals.target_path)
        temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    else:
        update_status("Đã tìm thấy các frame tạm, tiến hành resume xử lý...")

    # Áp dụng tính năng resume: nếu resume_frame > 1, bỏ qua các frame có số thứ tự nhỏ hơn giá trị này
    if temp_frame_paths and hasattr(roop.globals, "resume_frame") and \
       roop.globals.resume_frame > 1:
        import re
        resume_frame = roop.globals.resume_frame
        danh_sach_frame_loc = []
        for path_frame in temp_frame_paths:
            ten_file = os.path.basename(path_frame)
            match = re.search(r"\d+", ten_file)
            if match:
                so_frame = int(match.group())
                if so_frame >= resume_frame:
                    danh_sach_frame_loc.append(path_frame)
            else:
                danh_sach_frame_loc.append(path_frame)
        if len(danh_sach_frame_loc) == 0:
            update_status(
                f"Không có frame mới từ frame số {resume_frame}; sử dụng toàn bộ các frame có sẵn."
            )
            # Nếu không có frame nào ≥ resume_frame, giữ nguyên danh sách ban đầu
        else:
            temp_frame_paths = sorted(danh_sach_frame_loc)
            update_status(
                f"Tiếp tục xử lý từ frame số {resume_frame}, tổng số frame cần xử lý: {len(temp_frame_paths)}"
            )




    # Xử lý từng frame
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Đang xử lý...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Không tìm thấy frame...')
        return

    # Tạo video từ các frame đã xử lý
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Đang tạo video với FPS = {fps}...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Đang tạo video với FPS = 30...')
        create_video(roop.globals.target_path)

    # Xử lý âm thanh: nếu bỏ qua âm thanh thì di chuyển các file tạm, ngược lại khôi phục âm thanh
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Bỏ qua âm thanh...')
    else:
        if roop.globals.keep_fps:
            update_status('Đang khôi phục âm thanh...')
        else:
            update_status('Khôi phục âm thanh có thể gây ra một số vấn đề vì FPS không được giữ nguyên...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)

    # Dọn dẹp tài nguyên tạm
    update_status('Dọn dẹp tài nguyên tạm...')
    clean_temp(roop.globals.target_path)

    # Kiểm tra kết quả xử lý video
    if is_video(roop.globals.target_path):
        update_status('Xử lý video thành công!')
    else:
        update_status('Xử lý video thất bại!')

def destroy() -> None:
    """
    Hàm destroy() được gọi khi có sự cố hoặc khi người dùng dừng quá trình.
    Dọn dẹp tài nguyên tạm và thoát chương trình.
    """
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()

def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
