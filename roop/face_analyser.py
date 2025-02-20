import threading
import time
from typing import Any, Optional, List
import insightface
import numpy
import ipywidgets as widgets
from IPython.display import display, clear_output
import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0)
    return FACE_ANALYSER

def clear_face_analyser() -> Any:
    global FACE_ANALYSER
    FACE_ANALYSER = None

def select_face_index(faces: List[Face]) -> int:
    if len(faces) == 1:
        return 0  # Nếu chỉ có một khuôn mặt, chọn luôn
    
    output = widgets.Output()
    dropdown = widgets.Dropdown(
        options=[(f"Khuôn mặt {i}", i) for i in range(len(faces))],
        description="Chọn mặt:",
        style={'description_width': 'initial'}
    )
    button_ok = widgets.Button(description="OK", button_style='success')
    button_cancel = widgets.Button(description="Hủy", button_style='danger')
    selected_index = [None]

    def on_ok_clicked(_):
        selected_index[0] = dropdown.value
        clear_output(wait=True)

    def on_cancel_clicked(_):
        selected_index[0] = 0  # Mặc định chọn khuôn mặt đầu tiên nếu hủy
        clear_output(wait=True)
        print("⏳ Không chọn, tự động lấy khuôn mặt đầu tiên.")

    button_ok.on_click(on_ok_clicked)
    button_cancel.on_click(on_cancel_clicked)

    display(dropdown, button_ok, button_cancel, output)

    def auto_select():
        time.sleep(5)
        if selected_index[0] is None:
            selected_index[0] = 0
            clear_output(wait=True)
            print("⏳ Hết thời gian, tự động chọn khuôn mặt đầu tiên.")
    
    threading.Thread(target=auto_select, daemon=True).start()

    while selected_index[0] is None:
        time.sleep(0.1)

    return selected_index[0]

def get_one_face(frame: Frame) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        index = select_face_index(many_faces)
        return many_faces[index]
    return None

def get_many_faces(frame: Frame) -> Optional[List[Face]]:
    try:
        faces = get_face_analyser().get(frame)
        return faces
    except ValueError:
        return None

def find_similar_face(frame: Frame, reference_face: Face) -> Optional[Face]:
    many_faces = get_many_faces(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < roop.globals.similar_face_distance:
                    return face
    return None
