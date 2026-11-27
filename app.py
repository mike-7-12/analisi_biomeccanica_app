import streamlit as st
import utils as u
import numpy as np
import cv2
import qrcode
from io import BytesIO
from datetime import datetime

oggi = datetime.now().strftime("%Y-%m-%d") 

st.set_page_config(page_title='Mobility Analyzer', layout='centered', page_icon=":material/self_improvement:", initial_sidebar_state="collapsed")

with st.sidebar:
#     def get_internal_ip():
#         with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#             try:
#                 s.connect(("8.8.8.8", 1))
#                 return s.getsockname()[0]
#             except Exception:
#                 return "127.0.0.1"

#     # Build the network URL (adjust port if you use a different one)
#     port = 8501  # default Streamlit port
#     internal_ip = get_internal_ip()
#     network_url = f"http://{internal_ip}:{port}"

#     st.write(f"Network URL: {network_url}")

    # Generate QR code
    qr = qrcode.make("https://mobility-analyzer.streamlit.app/")
    buf = BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf, caption="Scan to open on your phone")

if 'home' not in st.session_state:
    st.session_state.home = True
if 'elaborate' not in st.session_state:
    st.session_state.elaborate = False

def run_elab():
    if st.session_state.acq is not None:
        st.session_state.home = False
        st.session_state.elaborate = True
        
    
def reset():
    st.session_state.home = True
    for k in list(st.session_state.keys()):
        if k != 'home':
            st.session_state[k] = False

if st.session_state.home:
    st.header("Benvenuto nell'app di analisi della mobilità", divider="green", text_alignment="center", icon=":material/self_improvement:")
    st.markdown("Per iniziare, carica una foto o un video e seleziona quali angoli articolari ti interessano", text_alignment="center")

    with st.container(border=True, horizontal=True):
        col11, col12 = st.columns([1, 2], gap="large")
        with col11:
            if not st.toggle("Scatta una foto"):
                st.session_state['acq'] = st.file_uploader('Clicca per selezionare un file', type=['image', 'video'], accept_multiple_files=False)
            else:
                st.session_state['acq'] = st.camera_input("Inquadra e scatta", resolution="1080p")
        with col12:
            st.session_state['name'] = st.text_input("Inserisci il nome del soggetto", placeholder="Soggetto1")
            st.session_state['exc'] = st.text_input("Inserisci il nome dell'esercizio", placeholder="Esercizio1")

    with st.container(border=True):
        col21, col22 = st.columns(2, gap="xxsmall", wrap=False)
        with col21:
            l_elbow = st.toggle('Angolo di gomito sx')          
            l_hip = st.toggle('Angolo di anca sx')          
            l_knee = st.toggle('Angolo di ginocchio sx')
            l_shoul = st.toggle('Angolo di spalla sx')
        with col22:
            r_elbow = st.toggle('Angolo di gomito dx')
            r_hip = st.toggle('Angolo di anca dx') 
            r_knee = st.toggle('Angolo di ginocchio dx')
            r_shoul = st.toggle('Angolo di spalla dx')

        st.session_state["ang_dict"] = {"gomito dx":r_elbow, "gomito sx":l_elbow, "anca dx":r_hip, "anca sx":l_hip, "ginocchio dx":r_knee, "ginocchio sx":l_knee, "spalla sx":l_shoul, "spalla dx":r_shoul}

    with st.container(horizontal_alignment="center"):
        st.session_state["model"] = st.segmented_control("Modello:", ["lite", "full"], selection_mode="single", required=True, help="Scegli lite per maggior velocità e full per maggior precisione", default="lite")
    st.button("Avvia elaborazione", type="primary", icon=":material/hub:", width="stretch", on_click=run_elab)


if st.session_state.elaborate:
    st.header("Visualizzazione risultati", divider="green", text_alignment="center")
    acq = st.session_state.acq
    name = st.session_state.name
    exc = st.session_state.exc
    ang_dict = st.session_state.ang_dict
    model = st.session_state.model

    if acq is None or model is None:
        st.toast("Carica un file valido")
        reset()    
    if acq.type[:5] == "image":
        # 1. Dai byte del file a un array 1D di uint8
        data = np.frombuffer(st.session_state.acq.getvalue(), dtype=np.uint8)

        # 2. Decodifica in immagine (array altezza x larghezza x 3, in ordine BGR)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Impossibile leggere l'immagine: formato non supportato o file corrotto.")
            reset()

        annotated, a = u.run_img_acquisition(img, 1, ang_dict, name, exc, model)
        st.toast("Elaborazione completata")
        with st.container(border=True):
            st.image((cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
            for j, ang in a.items():
                if ang:
                    st.metric(f"Angolo di flessione di {j}", value=f"{ang}°")
        col31, col32 = st.columns(2)
        with col31:
            st.download_button("Salva l'immagine e gli angoli articolari", data=u.array_to_image_bytes((cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))), mime="image/png", file_name=f"{name}_{oggi}_{exc}.png", on_click="ignore", type="primary", icon=":material/download:", width="stretch")

    elif acq.type[:5] == "video":
        annotated, coords, amax, amin = u.run_video_acquisition(acq, 1, ang_dict, name, exc, model)
        st.toast("Elaborazione completata")
        with st.container(border=True):
            st.video(annotated, autoplay=True, loop=True)
            for j, ang in amax.items():
                if ang:
                    st.metric(f"Angolo massimo di flessione di {j}", value=f"{ang}°")
                    st.metric(f"Angolo minimo di flessione di {j}",  value=f"{amin[j]}°")
        col31, col32 = st.columns(2)
        with col31:
            st.download_button("Salva il video e gli angoli articolari", data=annotated, mime="video/mp4", file_name=f"{name}_{oggi}_{exc}.mp4", on_click="ignore", type="primary", icon=":material/download:", width="stretch")



    with col32:
        st.button("Analizza un nuovo esercizio", on_click=reset, icon=":material/repeat:", width="stretch")