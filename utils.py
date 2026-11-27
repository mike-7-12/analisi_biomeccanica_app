import mediapipe as mp
import numpy as np
import cv2
import threading
import queue
import tempfile
import os
from pathlib import Path
import math
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imageio.v2 as imageio
from PIL import Image
import io
import streamlit as st
from datetime import datetime

oggi = datetime.now().strftime("%Y-%m-%d")


model_lite_path = Path("pose_landmarker_lite.task")
model_full_path = Path("pose_landmarker_full.task")


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


latest_result = None
results_queue = queue.Queue()
lock = threading.Lock()

centers = {
    "gomito dx":[12, 14, 16],
    "gomito sx":[11, 13, 15],
    "ginocchio dx":[28, 26, 24],
    "ginocchio sx":[23, 25, 27],
    "anca dx":[12, 24, 26],
    "anca sx":[11, 23, 25],
    "spalla dx":[24, 12, 14],
    "spalla sx":[23, 11, 13]
}

    # ["nose", 0
    # "left eye (inner)", 1
    # "left eye", 2
    # "left eye (outer)",3
    # "right eye (inner)",4
    # "right eye",5
    # "right eye (outer)",6
    # "left ear",7
    # "right ear",8
    # "mouth (left)",9
    # "mouth (right)",10
    # "left shoulder",11
    # "right shoulder",12
    # "left elbow",13
    # "right elbow",14
    # "left wrist",15
    # "right wrist",16
    # "left pinky",17
    # "right pinky",18
    # "left index",19
    # "right index",20
    # "left thumb",21
    # "right thumb",22
    # "left hip",23
    # "right hip",24
    # "left knee",25
    # "right knee",26
    # "left ankle",27
    # "right ankle",28
    # "left heel",29
    # "right heel",30
    # "left foot index",31
    # "right foot index"]32



def draw_landmarks_on_image(bgr_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(bgr_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style)

    return annotated_image

def pad_to_table(img, wt, ht, color=(0, 0, 0)):
    h, w = img.shape[:2]
    bottom = 0 if h%2==0 else 1
    right = wt if wt%2==0 else wt+1
    return cv2.copyMakeBorder(img, 0, bottom, 0, right,
                                borderType=cv2.BORDER_CONSTANT, value=color)

def run_img_acquisition(img, save_coords_bool, angl, nome, esercizio, model):

    options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_full_path if model=="full" else model_lite_path)),
    running_mode=VisionRunningMode.IMAGE)

    a = angl.copy()
    a = {k:0 for k in a}

    with PoseLandmarker.create_from_options(options) as landmarker:

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        pose_landmarker_result = landmarker.detect(mp_image)
        annotated_img = draw_landmarks_on_image(img, pose_landmarker_result)

            
        height, width = img.shape[:2]
        if save_coords_bool:
            coords = []
            if pose_landmarker_result.pose_landmarks:
                lms = pose_landmarker_result.pose_landmarks[0]  # prima persona rilevata
                coords = np.array([[int(lm.x * width), int(lm.y * height)] for lm in lms])  # (33, 2)
            else:
                coords = np.full((33, 2), np.nan)  # frame senza rilevamento


    rows = []
    headers = ["Articolazione", "Angolo (°)"]
    for joint in angl.keys():
        if angl[joint]:
            annotated_img, angolo = disegna_angolo(annotated_img, coords[centers[joint][0], :], coords[centers[joint][1], :], coords[centers[joint][2], :], raggio=30)
            # if height > width:
            #     cv2.putText(annotated_img, f"Angolo di flessione {joint}: {angolo}°", (width+5, 30+i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # else:
            #     cv2.putText(annotated_img, f"Angolo di flessione {joint}: {angolo}°", (5, height+30+i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            a[joint] = angolo
            # i+=30
            rows.append((joint, f"{a[joint]}°"))
            

    tbl, wt, ht = make_table(headers, rows, title_lines=[f"Soggetto: {nome}", f"Esercizio: {esercizio}", f"Data: {oggi}"])
    annotated_img = pad_to_table(annotated_img, wt, ht)
    annotated_img[0:ht, width:width+wt] = tbl

    return annotated_img, a

def run_video_acquisition(vid, save_coords_bool, angl, nome, esercizio, model):

    options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_full_path if model=="full" else model_lite_path)),
    running_mode=VisionRunningMode.VIDEO)


    with PoseLandmarker.create_from_options(options) as landmarker:
        suffix = os.path.splitext(vid.name)[1] or ".mp4"

        # OpenCV legge i video solo da un percorso su disco: scrivo i byte in un file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(vid.getvalue())
            tmp_path = tmp.name  # il file viene chiuso all'uscita dal with

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay_ms = max(1, int(1000 / fps))


        frames = []
        output = []
        amax, amin = angl.copy(), angl.copy()
        amax = {k:0 for k in amax}
        amin = {k:180 for k in amin}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)


            if save_coords_bool:
                height, width = frame.shape[:2]
                if pose_landmarker_result.pose_landmarks:
                    lms = pose_landmarker_result.pose_landmarks[0]  # prima persona rilevata
                    coords = np.array([[lm.x * width, lm.y * height] for lm in lms])  # (33, 2)
                else:
                    coords = np.full((33, 2), np.nan)  # frame senza rilevamento

                frames.append(coords)

            annotated_img = draw_landmarks_on_image(frame, pose_landmarker_result)
            i=0
            
            for joint in angl.keys():
                if angl[joint]:
                    try:
                        annotated_img, angolo = disegna_angolo(annotated_img, coords[centers[joint][0], :], coords[centers[joint][1], :], coords[centers[joint][2], :], raggio=30)
                        amax[joint] = angolo if angolo > amax[joint] else amax[joint]
                        amin[joint] = angolo if angolo < amin[joint] else amin[joint]
                        cv2.putText(annotated_img, f"Angolo di flessione {joint}: {angolo}°", (5, 30+i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # if height > width:
                        #     cv2.putText(annotated_img, f"Angolo di flessione {joint}: {angolo}°", (width+5, 2*height-30-i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        # else:
                        #     cv2.putText(annotated_img, f"Angolo di flessione {joint}: {angolo}°", (5, 2*height-30-i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)                        
                        i+=30                    
                    except ValueError:
                        continue
            output.append(annotated_img)

        rows = [(nome, f"{amax[nome]}°", f"{amin[nome]}°") for nome in amax if (amax[nome]!=0 and amin[nome]!=180)]
        tbl, wt, ht = make_table(["Articolazione", "Angolo Max", "Angolo Min"], rows, title_lines=[f"Soggetto: {nome}", f"Esercizio: {esercizio}", f"Data: {oggi}"])
        for k, fr in enumerate(output):
            fr = pad_to_table(fr,wt, ht)
            #i = 0                
            fr[0:ht, width:width+wt] = tbl
                    # if height > width:
                    #     cv2.putText(fr, f"Angolo di flessione {j} massimo: {ang} gradi", (width+5, 30+i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    #     cv2.putText(fr, f"Angolo di flessione {j} minimo: {amin[j]} gradi", (width+5, 30+i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # else:    
                    #     cv2.putText(fr, f"Angolo di flessione {j} massimo: {ang} gradi", (5, height+30+i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    #     cv2.putText(fr, f"Angolo di flessione {j} minimo: {amin[j]} gradi", (5, height+60+i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    # i+=60
            output[k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)




        # frames_rgb: array uint8 (n_frames, altezza, larghezza, 3), in ordine RGB
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name

            imageio.mimwrite(out_path, output, fps=fps, codec="libx264", macro_block_size=1)

        with open(out_path, "rb") as f:
            video_bytes = f.read()
        os.remove(out_path)
        landmarks_3d = np.stack(frames, axis=2)  # (33, 2, n_frames)


    return video_bytes, landmarks_3d, amax, amin

def get_video_params(camera_number, video_bool=False, video_path=None):
    if video_bool:
        cap = cv2.VideoCapture(video_path)
    else:    
        cap = cv2.VideoCapture(camera_number)

    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))

def disegna_angolo(img, p1, p2, p3, raggio=60, colore_spicchio=(0, 200, 255), 
                     colore_linee=(255, 255, 255), colore_testo=(255, 255, 255)):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    p3 = tuple(map(int, p3))
    
    # Vettori dal vertice p2
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Angoli in gradi nella convenzione Opencv2 (clockwise da asse x, y verso il basso)
    angolo1 = math.degrees(math.atan2(v1[1], v1[0]))
    angolo2 = math.degrees(math.atan2(v2[1], v2[0]))
    
    # Normalizza per ottenere lo sweep minore (l'arco <= 180°)
    diff = (angolo2 - angolo1) % 360
    if diff <= 180:
        start_angle, end_angle = angolo1, angolo1 + diff
    else:
        start_angle, end_angle = angolo2, angolo2 + (360 - diff)
    
    angolo_valore = end_angle - start_angle  # sempre tra 0 e 180
    
    # --- Disegna lo spicchio (settore pieno, semi-trasparente) ---
    overlay = img.copy()
    cv2.ellipse(overlay, p2, (raggio, raggio), 0, start_angle, end_angle, 
                colore_spicchio, thickness=-1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    
    # Contorno dello spicchio
    cv2.ellipse(img, p2, (raggio, raggio), 0, start_angle, end_angle, 
                colore_spicchio, thickness=2)
    
    # --- Linee dal vertice ai due punti ---
    cv2.line(img, p2, p1, colore_linee, 2)
    cv2.line(img, p2, p3, colore_linee, 2)
    
    # Pallini sui punti
    for p in (p1, p2, p3):
        cv2.circle(img, p, 4, (0, 0, 255), -1)
    
    # --- Etichetta con il valore dell'angolo ---
    bisettrice = math.radians((start_angle + end_angle) / 2)
    label_raggio = raggio + 10
    label_pos = (int(p2[0] + label_raggio * math.cos(bisettrice)),
                 int(p2[1] + label_raggio * math.sin(bisettrice)))
    
    testo = f"{int(angolo_valore)} °"
    cv2.putText(img, testo, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                colore_testo, 2, cv2.LINE_AA)
    
    return img, int(angolo_valore)

@st.cache_data
def array_to_image_bytes(array, format="PNG"):

    img = Image.fromarray(array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def make_table(headers, rows, title_lines=(), scale=0.7, title_scale=0.8, thickness=3, pad=10,
               text_color=(255, 255, 255), title_color=(255, 255, 255), grid_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    table = [list(headers)] + [[str(c) for c in r] for r in rows]
    n_rows, n_cols = len(table), len(headers)

    # misura ogni testo
    sizes = [[cv2.getTextSize(t, font, scale, thickness) for t in row] for row in table]
    text_h = max(s[0][1] for row in sizes for s in row)
    baseline = max(s[1] for row in sizes for s in row)
    row_h = text_h + baseline + 2 * pad
    col_w = [max(sizes[r][c][0][0] for r in range(n_rows)) + 2 * pad for c in range(n_cols)]
    table_w, table_h = sum(col_w), row_h * n_rows

    # misure del titolo
    t_sizes = [cv2.getTextSize(t, font, title_scale, thickness) for t in title_lines]
    t_text_h = max((s[0][1] for s in t_sizes), default=0)
    t_base = max((s[1] for s in t_sizes), default=0)
    line_h = t_text_h + t_base + pad
    title_h = line_h * len(title_lines) + (pad if title_lines else 0)

    W = max([table_w] + [s[0][0] + 2 * pad for s in t_sizes])
    H = title_h + table_h
    img = np.zeros((H, W, 3), np.uint8)

    # titolo
    for i, t in enumerate(title_lines):
        cv2.putText(img, t, (pad, i * line_h + pad + t_text_h),
                    font, title_scale, title_color, thickness, cv2.LINE_AA)

    # testo della tabella (spostato in basso di title_h)
    for r, row in enumerate(table):
        x = 0
        for c, txt in enumerate(row):
            cv2.putText(img, txt, (x + pad, title_h + r * row_h + pad + text_h),
                        font, scale, text_color, thickness, cv2.LINE_AA)
            x += col_w[c]

    # griglia
    for r in range(n_rows + 1):
        y = min(title_h + r * row_h, H - 1)
        cv2.line(img, (0, y), (table_w - 1, y), grid_color, 1)
    x = 0
    for c in range(n_cols + 1):
        xx = min(x, table_w - 1)
        cv2.line(img, (xx, title_h), (xx, H - 1), grid_color, 1)
        if c < n_cols:
            x += col_w[c]
    return img, W, H

#def extract_minmax_frames(annotated_video, max_index, min_index, ):

