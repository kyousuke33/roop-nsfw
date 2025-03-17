import os
import cv2
import numpy as np
import insightface
import threading
import sys

from roop.face_analyser import get_many_faces
from roop.utilities import resolve_relative_path

THREAD_LOCK = threading.Lock()
FACE_SWAPPER = None
AVAILABLE_MODELS = {"inswapper": "inswapper_128.onnx", "simswap": "simswap.onnx", "faceshifter": "faceshifter.onnx"}
SELECTED_MODEL = os.getenv("FACE_SWAP_MODEL", "inswapper")  # Cho phép chọn model thông qua biến môi trường

def get_face_swapper():
    global FACE_SWAPPER, SELECTED_MODEL
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path(f'../models/{AVAILABLE_MODELS.get(SELECTED_MODEL, "inswapper_128.onnx")}')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path)
    return FACE_SWAPPER

def swap_face(source_face, target_face, frame):
    return get_face_swapper().get(frame, target_face, source_face, paste_back=True)

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100  # Ngưỡng để xác định ảnh mờ

def process_frame(source_face, reference_face, frame):
    faces = get_many_faces(frame)
    if faces:
        best_face = None
        highest_clarity = 0
        
        for face in faces:
            x1, y1, x2, y2 = map(int, face['bbox'])
            face_img = frame[y1:y2, x1:x2]
            if not is_blurry(face_img):
                clarity = cv2.Laplacian(face_img, cv2.CV_64F).var()
                if clarity > highest_clarity:
                    highest_clarity = clarity
                    best_face = face
        
        if best_face:
            return swap_face(source_face, best_face, frame)
    return frame
