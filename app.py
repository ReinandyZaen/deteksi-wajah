import streamlit as st
import cv2
import os
import pickle
import face_recognition
import numpy as np
import av  # Diperlukan oleh streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

# --- Konfigurasi Path dan Konstanta ---
DATASET_PATH = "dataset"
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
ENCODINGS_PATH = "encodings.pickle"

# Pastikan folder dataset ada
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Muat face cascade (hanya sekali)
@st.cache_resource
def load_cascade():
    if not os.path.exists(CASCADE_PATH):
        st.error(f"File cascade tidak ditemukan di: {CASCADE_PATH}")
        return None
    return cv2.CascadeClassifier(CASCADE_PATH)

face_cascade = load_cascade()

# --- Fungsi Halaman 1: Tambah Wajah ---
def page_tambah_wajah():
    st.title("Tambah Wajah Baru")
    
    nama_pengguna = st.text_input("Masukkan nama Anda:", key="nama_pengguna_input")

    if not nama_pengguna:
        st.warning("Silakan masukkan nama Anda untuk melanjutkan.")
        return

    # Buat folder untuk pengguna
    folder_pengguna = os.path.join(DATASET_PATH, nama_pengguna)
    if not os.path.exists(folder_pengguna):
        os.makedirs(folder_pengguna)

    # Inisialisasi counter di session state
    if f'img_count_{nama_pengguna}' not in st.session_state:
        st.session_state[f'img_count_{nama_pengguna}'] = 0

    choice = st.radio("Pilih metode:", ["Ambil Foto (Kamera)", "Upload dari Folder"])

    if choice == "Ambil Foto (Kamera)":
        st.info("Arahkan wajah ke kamera dan klik 'Ambil Foto'. Ulangi hingga 30 foto.")
        img_file_buffer = st.camera_input("Ambil Foto")

        if img_file_buffer:
            # Baca gambar dari buffer
            bytes_data = img_file_buffer.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Ambil wajah pertama
                wajah_crop = gray[y:y+h, x:x+w]
                
                # Simpan wajah
                st.session_state[f'img_count_{nama_pengguna}'] += 1
                count = st.session_state[f'img_count_{nama_pengguna}']
                path_file = os.path.join(folder_pengguna, f"{count}.jpg")
                cv2.imwrite(path_file, wajah_crop)
                
                st.image(image, channels="BGR", caption=f"Foto {count} untuk {nama_pengguna} tersimpan!")
                st.success(f"Foto {count} tersimpan.")
            else:
                st.error("Tidak ada wajah terdeteksi. Coba lagi.")

    elif choice == "Upload dari Folder":
        uploaded_files = st.file_uploader("Pilih gambar...", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        if uploaded_files:
            with st.spinner("Memproses gambar..."):
                for uploaded_file in uploaded_files:
                    # Baca gambar
                    image = Image.open(uploaded_file)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                    
                    # Deteksi wajah
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        wajah_crop = gray[y:y+h, x:x+w]
                        
                        st.session_state[f'img_count_{nama_pengguna}'] += 1
                        count = st.session_state[f'img_count_{nama_pengguna}']
                        path_file = os.path.join(folder_pengguna, f"{count}.jpg")
                        cv2.imwrite(path_file, wajah_crop)
                        st.write(f"Wajah terdeteksi dan disimpan dari {uploaded_file.name}")
                    else:
                        st.warning(f"Tidak ada wajah terdeteksi di {uploaded_file.name}.")
            st.success(f"Selesai! Total {st.session_state[f'img_count_{nama_pengguna}']} foto disimpan untuk {nama_pengguna}.")


# --- Fungsi Halaman 2: Latih Model ---
def page_latih_model():
    st.title("Latih Model")
    
    if not os.path.exists(DATASET_PATH) or len(os.listdir(DATASET_PATH)) == 0:
        st.error("Folder 'dataset' kosong. Silakan tambah wajah terlebih dahulu.")
        return

    if st.button("Mulai Latih Model"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("[INFO] Mulai memproses gambar...")
        
        known_encodings = []
        known_names = []

        list_nama = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        total_nama = len(list_nama)

        for i, nama_orang in enumerate(list_nama):
            folder_orang = os.path.join(DATASET_PATH, nama_orang)
            status_text.text(f"Memproses wajah: {nama_orang} ({i+1}/{total_nama})...")
            
            for nama_file in os.listdir(folder_orang):
                path_file = os.path.join(folder_orang, nama_file)
                try:
                    image = cv2.imread(path_file)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    st.warning(f"Error membaca {path_file}: {e}")
                    continue

                boxes = face_recognition.face_locations(rgb, model="hog")
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(nama_orang)
                else:
                    st.warning(f"Tidak ada wajah ditemukan di {path_file}")

            progress_bar.progress((i + 1) / total_nama)

        status_text.text("[INFO] Menyimpan encodings ke file...")
        data = {"encodings": known_encodings, "names": known_names}
        with open(ENCODINGS_PATH, "wb") as f:
            f.write(pickle.dumps(data))

        progress_bar.progress(1.0)
        status_text.text("[INFO] Selesai.")
        st.success(f"Model berhasil dilatih dengan {len(known_names)} gambar dari {total_nama} orang.")


# --- Fungsi Halaman 3: Kenali Wajah ---
def page_kenali_wajah():
    st.title("Jalankan Pengenalan Wajah")

    if not os.path.exists(ENCODINGS_PATH):
        st.error("File 'encodings.pickle' tidak ditemukan. Jalankan 'Latih Model' terlebih dahulu.")
        return

    # Muat data encoding (hanya sekali)
    @st.cache_resource
    def load_encodings():
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)

    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]

    # Definisikan Video Transformer untuk streamlit-webrtc
    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self):
            self.known_encodings = known_encodings
            self.known_names = known_names

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # Ubah frame ke array numpy BGR
            img = frame.to_ndarray(format="bgr24")

            # Logika dari skrip 03_kenali_wajah.py
            rgb_frame_kecil = cv2.cvtColor(cv2.resize(img, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame_kecil, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame_kecil, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
                nama = "Tidak Dikenal"
                persentase = 0

                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    nama = self.known_names[best_match_index]
                    distance_value = face_distances[best_match_index]
                    if distance_value <= 0.5:
                        persentase = (1.0 - (distance_value / 0.5)) * 100
                
                # Kembalikan ke ukuran asli
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                color = (0, 255, 0) if nama != "Tidak Dikenal" else (0, 0, 255)
                
                # Gambar kotak
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                
                # Buat label
                label = f"{nama}"
                if nama != "Tidak Dikenal":
                    label = f"{nama}: {persentase:.0f}%"

                cv2.rectangle(img, (left, top - 35), (right, top), color, cv2.FILLED)
                cv2.putText(img, label, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Kembalikan frame yang sudah diproses
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Jalankan streamer
    st.info("Klik 'Start' untuk menyalakan kamera.")
    webrtc_streamer(
        key="face_recognition",
        video_processor_factory=FaceRecognitionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


# --- Main App: Navigasi Sidebar ---
def main():
    st.sidebar.title("Sistem Pengenalan Wajah")
    page = st.sidebar.radio(
        "Pilih Halaman:",
        ("Tambah Wajah", "Latih Model", "Jalankan Pengenalan")
    )

    if page == "Tambah Wajah":
        page_tambah_wajah()
    elif page == "Latih Model":
        page_latih_model()
    elif page == "Jalankan Pengenalan":
        page_kenali_wajah()

if __name__ == "__main__":
    if face_cascade is None:
        st.error("Gagal memuat 'haarcascade_frontalface_default.xml'. Aplikasi tidak bisa berjalan.")
    else:
        main()