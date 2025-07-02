import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import tempfile
import os
from collections import deque
import threading
import queue

# Import streamlit-webrtc components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Page config
st.set_page_config(
    page_title="Deteksi Bahasa Isyarat SIBI Real-time",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WebRTC configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

class SIBIStreamlitDetector:
    def __init__(self, model_path='models/sibi11mv1.pt'):
        """
        Inisialisasi detektor SIBI untuk Streamlit dengan kemampuan membangun kalimat
        Diperbarui menggunakan model sibi11mv1.pt dengan kosakata yang diperluas
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Inisialisasi model
        self.model = None
        self.current_prediction = None
        self.current_confidence = 0.0

        # Muat model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            raise Exception(f"Gagal memuat model: {e}")

        # Parameter deteksi
        self.confidence_threshold = 0.5
        self.prediction_history = []
        self.history_size = 5

        # Fitur pembangunan kalimat
        self.detected_words = deque(maxlen=50)  # Simpan 50 kata terakhir yang terdeteksi
        self.current_sentence = ""
        self.last_detection_time = 0
        self.word_timeout = 1.5  # detik antara kata
        self.stable_detection_count = 0
        self.stable_threshold = 3  # butuh 3 deteksi stabil untuk menambah kata

        # Variabel thread-safe untuk WebRTC
        self.lock = threading.Lock()
        self.latest_frame = None
        self.detection_results = queue.Queue(maxsize=10)

    def predict(self, frame):
        """
        Buat prediksi pada frame menggunakan YOLO
        """
        try:
            results = self.model(frame, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Ambil deteksi terbaik
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                # Ambil deteksi dengan confidence tertinggi
                best_idx = confidences.argmax()
                predicted = int(classes[best_idx])
                confidence = float(confidences[best_idx])

                # Ambil bounding box
                bbox = boxes.xyxy[best_idx].cpu().numpy()

                return predicted, confidence, bbox
            else:
                return None, 0.0, None
        
        except Exception:
            return None, 0.0, None
    
    def smooth_predictions(self, prediction, confidence):
        """
        Haluskan prediksi menggunakan riwayat untuk mengurangi noise
        """
        if confidence > self.confidence_threshold:
            self.prediction_history.append(prediction)

        # Simpan hanya prediksi terbaru
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        # Kembalikan prediksi yang paling sering muncul jika riwayat cukup
        if len(self.prediction_history) >= 3:
            most_common = max(set(self.prediction_history),
                            key=self.prediction_history.count)
            return most_common

        return prediction if confidence > self.confidence_threshold else None

    def add_word_to_sentence(self, word):
        """
        Tambahkan kata yang terdeteksi ke kalimat dengan logika waktu
        """
        current_time = time.time()

        # Cek apakah waktu yang cukup telah berlalu sejak deteksi terakhir
        if current_time - self.last_detection_time > self.word_timeout:
            # Pastikan current_sentence adalah string
            if not isinstance(self.current_sentence, str):
                self.current_sentence = ""

            # Tambahkan kata ke kalimat
            if self.current_sentence:
                self.current_sentence += " " + str(word)
            else:
                self.current_sentence = str(word)

            # Tambahkan ke riwayat kata yang terdeteksi
            self.detected_words.append({
                'word': word,
                'timestamp': current_time,
                'confidence': self.current_confidence
            })

            self.last_detection_time = current_time
            self.stable_detection_count = 0
            return True
        return False

    def clear_sentence(self):
        """Bersihkan kalimat saat ini"""
        self.current_sentence = ""
        self.detected_words.clear()
        self.stable_detection_count = 0

    def get_sentence_info(self):
        """Dapatkan informasi kalimat saat ini dan riwayat kata"""
        # Pastikan current_sentence adalah string dan perbaiki jika bukan
        if not isinstance(self.current_sentence, str):
            self.current_sentence = str(self.current_sentence) if self.current_sentence else ""

        sentence_str = self.current_sentence if self.current_sentence else ""
        return {
            'sentence': sentence_str,
            'word_count': len(sentence_str.split()) if sentence_str else 0,
            'last_words': list(self.detected_words)[-5:] if self.detected_words else []
        }

    def draw_info(self, frame, prediction, confidence, bbox=None):
        """
        Gambar informasi prediksi pada frame dengan tampilan yang kompak
        """
        width = frame.shape[1]

        # Gambar bounding box jika tersedia
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Gambar info prediksi dengan kotak yang lebih kecil
        if prediction is not None and confidence > self.confidence_threshold:
            # Ambil nama kelas dari model
            class_names = self.model.names
            label = class_names.get(prediction, f"Kelas_{prediction}")

            # Cek deteksi stabil untuk pembangunan kalimat
            if label == self.current_prediction:
                self.stable_detection_count += 1
            else:
                self.stable_detection_count = 1

            # Tambahkan ke kalimat jika cukup stabil
            if self.stable_detection_count >= self.stable_threshold:
                self.add_word_to_sentence(label)

            # Background yang lebih kecil untuk teks
            cv2.rectangle(frame, (10, 10), (min(width-10, 350), 90), (0, 0, 0), -1)

            # Teks prediksi saat ini - ukuran font lebih kecil
            cv2.putText(frame, f"{label}",
                       (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}",
                       (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Teks kalimat yang kompak - pastikan current_sentence adalah string
            if not isinstance(self.current_sentence, str):
                self.current_sentence = str(self.current_sentence) if self.current_sentence else ""

            sentence_text = self.current_sentence if self.current_sentence else "..."
            if len(sentence_text) > 25:  # Pemotongan yang lebih pendek
                sentence_text = sentence_text[:22] + "..."
            cv2.putText(frame, f"K: {sentence_text}",
                       (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Update prediksi saat ini untuk tampilan
            self.current_prediction = label
            self.current_confidence = confidence
        else:
            # Kotak "Tidak ada deteksi" yang lebih kecil
            cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
            cv2.putText(frame, "Tidak terdeteksi",
                       (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Tetap tampilkan kalimat jika tersedia - versi kompak
            if self.current_sentence:
                # Pastikan current_sentence adalah string
                if not isinstance(self.current_sentence, str):
                    self.current_sentence = str(self.current_sentence) if self.current_sentence else ""

                sentence_text = self.current_sentence
                if len(sentence_text) > 20:
                    sentence_text = sentence_text[:17] + "..."
                cv2.putText(frame, f"K: {sentence_text}",
                           (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Reset counter stabilitas
            self.stable_detection_count = 0
            self.current_prediction = None
            self.current_confidence = 0.0

        return frame

@st.cache_resource
def load_detector():
    """Muat detektor dengan caching - menggunakan model sibi11mv1.pt terbaru"""
    try:
        return SIBIStreamlitDetector()
    except Exception as e:
        st.error(f"Gagal menginisialisasi detektor: {e}")
        return None

def video_frame_callback(frame, detector):
    """
    Fungsi callback untuk memproses frame video dari WebRTC
    """
    img = frame.to_ndarray(format="bgr24")

    # Balik frame secara horizontal untuk efek cermin
    img = cv2.flip(img, 1)

    # Buat prediksi
    prediction, confidence, bbox = detector.predict(img)
    smoothed_prediction = detector.smooth_predictions(prediction, confidence)

    # Gambar informasi pada frame
    annotated_frame = detector.draw_info(img, smoothed_prediction, confidence, bbox)

    # Simpan hasil terbaru untuk update UI (thread-safe)
    try:
        detector.detection_results.put_nowait({
            'prediction': smoothed_prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
    except queue.Full:
        pass  # Lewati jika queue penuh

    # Konversi kembali ke av.VideoFrame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def process_image(detector, image_array, confidence_threshold):
    """Proses gambar dan kembalikan hasil"""
    # Konversi RGB ke BGR untuk OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Buat prediksi
    prediction, confidence, bbox = detector.predict(image_bgr)
    smoothed_prediction = detector.smooth_predictions(prediction, confidence)

    # Update threshold confidence
    detector.confidence_threshold = confidence_threshold

    # Gambar informasi pada frame
    annotated_frame = detector.draw_info(image_bgr, smoothed_prediction, confidence, bbox)

    # Konversi kembali ke RGB untuk tampilan
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    return annotated_frame_rgb, smoothed_prediction, confidence

def main():
    st.title("🤟 Deteksi Bahasa Isyarat SIBI Real-time")
    st.markdown("**Posisikan tangan Anda di depan kamera untuk mendeteksi bahasa isyarat SIBI dan membangun kalimat.**")

    # Pengaturan sidebar
    st.sidebar.header("⚙️ Pengaturan")
    confidence_threshold = st.sidebar.slider(
        "Ambang Batas Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Sesuaikan sensitivitas deteksi"
    )

    # Pengaturan pembangunan kalimat
    st.sidebar.subheader("📝 Pembangunan Kalimat")
    word_timeout = st.sidebar.slider(
        "Jeda Antar Kata (detik)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Waktu jeda antara deteksi kata"
    )

    stable_threshold = st.sidebar.slider(
        "Ambang Batas Stabilitas",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="Jumlah deteksi konsisten yang diperlukan untuk menambah kata"
    )

    # Muat detektor
    detector = load_detector()
    if detector is None:
        st.stop()

    # Update pengaturan detektor
    detector.confidence_threshold = confidence_threshold
    detector.word_timeout = word_timeout
    detector.stable_threshold = stable_threshold

    # Inisialisasi session state
    if 'sentence_history' not in st.session_state:
        st.session_state.sentence_history = []

    # Inisialisasi session state untuk tracking tab dan kamera
    if 'main_tab_active' not in st.session_state:
        st.session_state.main_tab_active = 'tab1'
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    # Tab interface utama
    tab1, tab2, tab3 = st.tabs(["📷 Deteksi Langsung", "🎬 Video Demo", "ℹ️ Tentang"])
    
    with tab1:
        # Set tab utama aktif
        st.session_state.main_tab_active = 'tab1'
        st.session_state.camera_active = False  # Matikan kamera upload saat di tab1

        st.header("📷 Deteksi Kamera Langsung")
        st.markdown("**Deteksi SIBI real-time dengan pembangunan kalimat menggunakan WebRTC**")

        # Tombol kontrol
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Hapus Kalimat"):
                detector.clear_sentence()
                st.rerun()
        with col2:
            # Tombol simpan kalimat
            sentence_info = detector.get_sentence_info()
            if sentence_info['sentence'] and st.button("💾 Simpan Kalimat"):
                st.session_state.sentence_history.append({
                    'sentence': sentence_info['sentence'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'word_count': sentence_info['word_count']
                })
                st.success("Kalimat berhasil disimpan ke riwayat!")

        # Buat layout dengan kamera yang lebih kecil
        col1, col2 = st.columns([1, 2])  # Kamera lebih kecil (1/3), info lebih besar (2/3)

        with col1:
            st.subheader("📹 Kamera")
            # Buat WebRTC streamer dengan styling khusus
            webrtc_ctx = webrtc_streamer(
                key="sibi-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=lambda frame: video_frame_callback(frame, detector),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 320},
                        "height": {"ideal": 240}
                    }, 
                    "audio": False
                },
                async_processing=True,
            )
        
        with col2:
            st.subheader("📊 Hasil Deteksi")
            detection_placeholder = st.empty()

            st.subheader("📝 Kalimat Saat Ini")
            sentence_placeholder = st.empty()

        # Update hasil deteksi secara real-time
        if webrtc_ctx.state.playing:
            st.success("🔴 Deteksi langsung aktif. Posisikan tangan Anda dengan jelas di depan kamera.")

            # Update hasil secara berkelanjutan
            while webrtc_ctx.state.playing:
                try:
                    # Ambil hasil deteksi terbaru
                    result = detector.detection_results.get(timeout=0.1)

                    with detection_placeholder.container():
                        if result['prediction'] and result['confidence'] > confidence_threshold:
                            # Tampilan metrik yang kompak
                            st.metric("🎯 Isyarat Saat Ini", result['prediction'])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("📊 Confidence", f"{result['confidence']:.1%}")
                            with col2:
                                st.metric("⚡ Stabilitas", f"{detector.stable_detection_count}/{detector.stable_threshold}")
                        else:
                            st.warning("⌛ Menunggu deteksi...")

                    # Update tampilan kalimat
                    sentence_info = detector.get_sentence_info()
                    with sentence_placeholder.container():
                        if sentence_info['sentence']:
                            st.success(f"**📝 Kalimat:** {sentence_info['sentence']}")
                            st.info(f"**📊 Kata:** {sentence_info['word_count']}")

                            # Tampilkan beberapa kata terakhir
                            if sentence_info['last_words']:
                                recent_words = [w['word'] for w in sentence_info['last_words']]
                                st.text(f"Terbaru: {' → '.join(recent_words)}")
                        else:
                            st.info("💡 Tunjukkan isyarat SIBI untuk mulai membangun kalimat")
                            
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                except:
                    break
        else:
            with col2:
                st.info("""
                👆 Klik tombol **Start** di bagian kamera untuk memulai deteksi.

                **Catatan**: Cari ruangan yang pencahayaannya cukup atau menggunakan lighting tambahan yang diarahkan langsung ke tangan untuk hasil deteksi yang lebih akurat.
                """)

                # Tampilkan kalimat saat ini meskipun tidak aktif
                sentence_info = detector.get_sentence_info()
                if sentence_info['sentence']:
                    st.success(f"**Kalimat Terakhir:** {sentence_info['sentence']}")
                    st.info(f"**Jumlah Kata:** {sentence_info['word_count']}")

        # Tampilkan kalimat saat ini meskipun tidak aktif
        sentence_info = detector.get_sentence_info()
        if sentence_info['sentence'] and not webrtc_ctx.state.playing:
            st.success(f"**Kalimat Terakhir yang Dibuat:** {sentence_info['sentence']}")

        # Riwayat kalimat
        if st.session_state.sentence_history:
            st.subheader("📚 Riwayat Kalimat")
            for i, entry in enumerate(reversed(st.session_state.sentence_history[-10:])):  # Show last 10
                with st.expander(f"Kalimat {len(st.session_state.sentence_history)-i}: {entry['sentence'][:50]}..."):
                    st.write(f"**Kalimat Lengkap:** {entry['sentence']}")
                    st.write(f"**Kata:** {entry['word_count']}")
                    st.write(f"**Waktu:** {entry['timestamp']}")

        # Instruksi
        with st.expander("📖 Cara Menggunakan Deteksi Langsung"):
            st.markdown("""
            **Instruksi untuk Deteksi Langsung:**
            1. Klik "Start" untuk mengaktifkan kamera
            2. Izinkan akses kamera saat diminta oleh browser
            3. Posisikan tangan Anda dengan jelas di depan kamera
            4. Tahan setiap isyarat SIBI dengan stabil selama beberapa detik
            5. Sistem akan otomatis menambahkan isyarat yang terdeteksi untuk membangun kalimat
            6. Gunakan "Hapus Kalimat" untuk memulai dari awal
            7. Gunakan "Simpan Kalimat" untuk menyimpan kalimat yang sudah selesai

            **Tips:**
            - Pastikan pencahayaan yang baik untuk deteksi yang lebih akurat
            - Tahan isyarat dengan stabil sesuai durasi ambang batas stabilitas
            - Sesuaikan ambang batas confidence dan stabilitas di sidebar
            - Jeda antar kata mengontrol jarak waktu antara kata
            - Coba bangun kalimat dengan kosakata yang ditampilkan di bawah

            **Kompatibilitas Browser:**
            - Bekerja paling baik dengan Chrome, Firefox, Safari, dan Edge
            - Memerlukan HTTPS untuk akses kamera di production
            - Mungkin perlu refresh jika kamera tidak mau mulai
            """)


    with tab2:
        # Set tab utama aktif dan matikan kamera
        st.session_state.main_tab_active = 'tab2'
        st.session_state.camera_active = False

        st.header("🎬 Video Demonstrasi SIBI")
        st.markdown("**Pelajari cara melakukan setiap isyarat dengan menonton video demonstrasi di bawah ini:**")

        # Peringatan untuk performa
        st.info("💡 **Tips**: Tutup tab ini saat menggunakan deteksi langsung untuk performa yang lebih baik.")

        # Kategori kata utama
        with st.expander("📚 Kata Utama (Main Words)", expanded=True):
            main_words = [
                "berangkat", "terbang", "toko", "banyak", "pesawat", "antar", "bisa", "simpan",
                "taksi", "henti", "kunci", "besok", "kamar", "berapa", "tunjuk", "halte",
                "belok", "kiri", "bantu", "panggil", "perlu", "mau", "saya", "mana",
                "makan", "kamu", "jalan", "hotel"
            ]

            # Tampilkan video dalam format grid (3 kolom)
            cols_per_row = 3
            for i in range(0, len(main_words), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(main_words):
                        word = main_words[i + j]
                        with col:
                            st.markdown(f"**{word.capitalize()}**")
                            # Muat video dengan error handling yang lebih baik
                            try:
                                video_path = f"dataset/{word.lower()}.mp4"
                                if os.path.exists(video_path):
                                    # Gunakan unique key untuk setiap video
                                    st.video(video_path, format="video/mp4", start_time=0)
                                else:
                                    st.info(f"Video untuk '{word}' tidak ditemukan")
                            except Exception:
                                st.warning(f"Video '{word}' tidak dapat dimuat")

        # Kategori kata penghubung
        with st.expander("🔗 Kata Imbuhan/Hubung (Connector Words)"):
            connector_words = ["ke", "di", "kan", "ber", "dan"]

            # Tampilkan video penghubung dalam format grid
            cols = st.columns(len(connector_words))
            for i, word in enumerate(connector_words):
                with cols[i]:
                    st.markdown(f"**{word.capitalize()}**")
                    try:
                        # Muat video dengan error handling yang lebih baik
                        video_path = f"dataset/{word.lower()}.mp4"
                        if os.path.exists(video_path):
                            st.video(video_path, format="video/mp4", start_time=0)
                        else:
                            st.info(f"Video untuk '{word}' tidak ditemukan")
                    except Exception:
                        st.warning(f"Video '{word}' tidak dapat dimuat")

    with tab3:
        # Set tab utama aktif dan matikan kamera
        st.session_state.main_tab_active = 'tab3'
        st.session_state.camera_active = False

        st.header("ℹ️ Tentang Detektor SIBI")

        st.markdown("""
        ### 🤟 Sistem Isyarat Bahasa Indonesia (SIBI)

        Aplikasi Streamlit ini menyediakan deteksi SIBI canggih dengan kemampuan membangun kalimat menggunakan WebRTC untuk akses kamera real-time.

        ### ✨ Fitur
        - **🎥 Deteksi Langsung WebRTC** - Streaming video real-time yang bekerja di aplikasi yang sudah di-deploy
        - **📝 Pembangunan Kalimat** - Konstruksi otomatis dari kata ke kalimat
        - **📚 Riwayat Kalimat** - Simpan dan kelola kalimat yang terdeteksi
        - **⚙️ Pengaturan Lanjutan** - Parameter stabilitas dan timing yang dapat dikonfigurasi
        - **📸 Berbagai Metode Input** - Kamera langsung dan upload file
        - **🌐 Kompatibel Cloud** - Bekerja di aplikasi Streamlit yang di-deploy dengan HTTPS

        ### 🔧 Stack Teknologi
        - **Model**: Ultralytics YOLO v8 untuk deteksi SIBI (sibi11mv1.pt - Versi Terbaru)
        - **Backend**: PyTorch untuk inferensi deep learning
        - **Frontend**: Streamlit dengan streamlit-webrtc untuk akses kamera
        - **Computer Vision**: OpenCV untuk pemrosesan gambar
        - **Pemrosesan Real-time**: WebRTC untuk streaming video latensi rendah

        ### 📊 Fitur Deteksi
        - **Penghalusan Prediksi** - Mengurangi noise dengan filtering berbasis riwayat
        - **Ambang Batas Confidence** - Sensitivitas deteksi yang dapat disesuaikan
        - **Kontrol Stabilitas** - Memerlukan deteksi konsisten sebelum menambah kata
        - **Visualisasi Bounding Box** - Umpan balik visual untuk isyarat yang terdeteksi
        - **Statistik Real-time** - Metrik confidence dan stabilitas langsung

        ### 🎯 Cara Kerja
        1. **Pengambilan Video**: WebRTC streaming video langsung dari browser Anda
        2. **Deteksi**: Model YOLO mengidentifikasi isyarat SIBI secara real-time
        3. **Penghalusan**: Beberapa deteksi konsisten mengurangi false positive
        4. **Stabilitas**: Kata ditambahkan hanya setelah periode deteksi yang stabil
        5. **Pembangunan Kalimat**: Kata yang terdeteksi otomatis digabungkan dengan logika timing
        6. **Riwayat**: Kalimat yang selesai disimpan untuk review dan ekspor

        ### 🌐 Keuntungan Deployment
        - **Kompatibilitas Browser**: Bekerja di berbagai browser modern
        - **Tanpa Dependensi Lokal**: Akses kamera melalui standar web
        - **Dukungan HTTPS**: Akses kamera yang aman di production
        - **Cross-Platform**: Bekerja di desktop dan perangkat mobile
        """)

        # Informasi model
        if detector and detector.model:
            st.subheader("🔍 Informasi Model")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Device**: {detector.device}")
                st.write(f"**Tipe Model**: Ultralytics YOLO")
                st.write(f"**Versi Model**: sibi11mv1.pt")
                st.write(f"**Kelas**: {len(detector.model.names)} kelas")
            with col2:
                st.write(f"**Ambang Batas Confidence**: {confidence_threshold}")
                st.write(f"**Jeda Antar Kata**: {detector.word_timeout}s")
                st.write(f"**Ambang Batas Stabilitas**: {detector.stable_threshold}")

            # Tampilkan informasi kosakata baru
            st.info("""
            **🆕 Kosakata Baru Ditambahkan**: berangkat, terbang, toko, banyak, pesawat, antar, bisa, simpan, taksi, henti, ber, kunci, besok, kamar, dan, berapa, tunjuk, kan, halte, belok, kiri, bantu, panggil, perlu
            """)

            # Tampilkan kelas yang tersedia
            with st.expander("📋 Lihat Semua Kelas SIBI yang Tersedia"):
                # Tampilkan kelas dengan cara yang lebih terorganisir
                classes = list(detector.model.names.items())
                cols = st.columns(3)
                for i, (idx, name) in enumerate(classes):
                    with cols[i % 3]:
                        st.write(f"**{idx}**: {name}")

        # Tips penggunaan
        with st.expander("💡 Tips Penggunaan"):
            st.markdown("""
            **Untuk Hasil Terbaik:**
            - Pastikan kondisi pencahayaan yang baik
            - Posisikan tangan dengan jelas di depan kamera
            - Tahan isyarat dengan stabil sesuai durasi ambang batas stabilitas
            - Gunakan posisi tangan yang konsisten
            - Sesuaikan ambang batas confidence berdasarkan lingkungan

            **Tips Membangun Kalimat:**
            - Tunggu jeda antar kata di antara isyarat yang berbeda
            - Gunakan "Hapus Kalimat" untuk memulai dari awal
            - Simpan kalimat penting ke riwayat
            - Sesuaikan ambang batas stabilitas untuk akurasi vs kecepatan
            - Coba kosakata baru untuk kalimat yang lebih kompleks

            **🆕 Kosakata Baru yang Tersedia:**
            - Transportasi: berangkat, terbang, pesawat, taksi, halte
            - Tempat & Objek: toko, kamar, kunci
            - Aksi: antar, bisa, simpan, henti, tunjuk, bantu, panggil
            - Deskriptor: banyak, besok, berapa, kiri
            - Penghubung: dan, kan, ber, perlu

            **Pengaturan Browser:**
            - Izinkan permission kamera saat diminta
            - Gunakan Chrome, Firefox, Safari, atau Edge untuk kompatibilitas terbaik
            - Pastikan koneksi internet yang stabil untuk streaming yang lancar
            - Refresh halaman jika kamera tidak mau mulai
            """)

        # Informasi teknis
        with st.expander("⚡ Informasi Teknis"):
            st.markdown("""
            **Implementasi WebRTC:**
            - Streaming video peer-to-peer real-time
            - Pipeline pemrosesan latensi rendah
            - Optimisasi frame rate otomatis
            - Penanganan hasil deteksi yang thread-safe

            **Optimisasi Performa:**
            - Pemrosesan video async
            - Update hasil berbasis queue
            - Penanganan frame yang efisien memori
            - Akselerasi GPU jika tersedia

            **Persyaratan Deployment:**
            - HTTPS diperlukan untuk akses kamera di production
            - Package streamlit-webrtc untuk fungsionalitas WebRTC
            - Browser modern dengan dukungan WebRTC
            - Koneksi internet yang stabil
            """)

if __name__ == "__main__":
    main()