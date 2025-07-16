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

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
st.set_page_config(
    page_title="Deteksi Bahasa Isyarat SIBI Real-time",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

class SIBIStreamlitDetector:
    def __init__(self, model_path='models/sibi11sv1.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.current_prediction = None
        self.current_confidence = 0.0

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            raise Exception(f"Gagal memuat model: {e}")

        self.confidence_threshold = 0.5
        self.prediction_history = []
        self.history_size = 5
        self.detected_words = deque(maxlen=50)
        self.current_sentence = ""
        self.last_detection_time = 0
        self.word_timeout = 1.5
        self.stable_detection_count = 0
        self.stable_threshold = 3
        self.lock = threading.Lock()
        self.latest_frame = None
        self.detection_results = queue.Queue(maxsize=10)

    def predict(self, frame):
        try:
            results = self.model(frame, verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                best_idx = confidences.argmax()
                predicted = int(classes[best_idx])
                confidence = float(confidences[best_idx])
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                return predicted, confidence, bbox
            else:
                return None, 0.0, None
        except Exception:
            return None, 0.0, None
    
    def smooth_predictions(self, prediction, confidence):
        if confidence > self.confidence_threshold:
            self.prediction_history.append(prediction)

        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)

        if len(self.prediction_history) >= 3:
            most_common = max(set(self.prediction_history),
                            key=self.prediction_history.count)
            return most_common

        return prediction if confidence > self.confidence_threshold else None

    def add_word_to_sentence(self, word):
        current_time = time.time()

        if current_time - self.last_detection_time > self.word_timeout:
            if not isinstance(self.current_sentence, str):
                self.current_sentence = ""

            if self.current_sentence:
                self.current_sentence += " " + str(word)
            else:
                self.current_sentence = str(word)

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
        self.current_sentence = ""
        self.detected_words.clear()
        self.stable_detection_count = 0

    def get_sentence_info(self):
        if not isinstance(self.current_sentence, str):
            self.current_sentence = str(self.current_sentence) if self.current_sentence else ""

        sentence_str = self.current_sentence if self.current_sentence else ""
        return {
            'sentence': sentence_str,
            'word_count': len(sentence_str.split()) if sentence_str else 0,
            'last_words': list(self.detected_words)[-5:] if self.detected_words else []
        }

    def draw_info(self, frame, prediction, confidence, bbox=None):
        width = frame.shape[1]

        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if prediction is not None and confidence > self.confidence_threshold:
            class_names = self.model.names
            label = class_names.get(prediction, f"Kelas_{prediction}")

            if label == self.current_prediction:
                self.stable_detection_count += 1
            else:
                self.stable_detection_count = 1

            if self.stable_detection_count >= self.stable_threshold:
                self.add_word_to_sentence(label)

            cv2.rectangle(frame, (10, 10), (min(width-10, 350), 90), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not isinstance(self.current_sentence, str):
                self.current_sentence = str(self.current_sentence) if self.current_sentence else ""

            sentence_text = self.current_sentence if self.current_sentence else "..."
            if len(sentence_text) > 25:
                sentence_text = sentence_text[:22] + "..."
            cv2.putText(frame, f"K: {sentence_text}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            self.current_prediction = label
            self.current_confidence = confidence
        else:
            cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
            cv2.putText(frame, "Tidak terdeteksi", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if self.current_sentence:
                if not isinstance(self.current_sentence, str):
                    self.current_sentence = str(self.current_sentence) if self.current_sentence else ""

                sentence_text = self.current_sentence
                if len(sentence_text) > 20:
                    sentence_text = sentence_text[:17] + "..."
                cv2.putText(frame, f"K: {sentence_text}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            self.stable_detection_count = 0
            self.current_prediction = None
            self.current_confidence = 0.0

        return frame

@st.cache_resource
def load_detector():
    try:
        return SIBIStreamlitDetector()
    except Exception as e:
        st.error(f"Gagal menginisialisasi detektor: {e}")
        return None

def video_frame_callback(frame, detector):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    prediction, confidence, bbox = detector.predict(img)
    smoothed_prediction = detector.smooth_predictions(prediction, confidence)
    annotated_frame = detector.draw_info(img, smoothed_prediction, confidence, bbox)

    try:
        detector.detection_results.put_nowait({
            'prediction': smoothed_prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
    except queue.Full:
        pass

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def process_image(detector, image_array, confidence_threshold):
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    prediction, confidence, bbox = detector.predict(image_bgr)
    smoothed_prediction = detector.smooth_predictions(prediction, confidence)
    detector.confidence_threshold = confidence_threshold
    annotated_frame = detector.draw_info(image_bgr, smoothed_prediction, confidence, bbox)
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return annotated_frame_rgb, smoothed_prediction, confidence

def process_image_detailed(detector, image_array, confidence_threshold):
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    try:
        results = detector.model(image_bgr, verbose=False)
        detections = []
        annotated_frame = image_bgr.copy()

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy()

            for cls, conf, bbox in zip(classes, confidences, bboxes):
                if conf > confidence_threshold:
                    class_name = detector.model.names.get(int(cls), f"Kelas_{int(cls)}")
                    detections.append({
                        'class_id': int(cls),
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': bbox
                    })

                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        return annotated_frame_rgb, detections

    except Exception:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), []

def main():
    st.title("Deteksi Bahasa Isyarat SIBI")
    st.markdown("Aplikasi untuk mendeteksi bahasa isyarat SIBI dan membangun kalimat secara otomatis.")

    st.sidebar.header("Pengaturan")
    confidence_threshold = st.sidebar.slider(
        "Tingkat Kepercayaan",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Sesuaikan sensitivitas deteksi"
    )

    st.sidebar.subheader("Pengaturan Kalimat")
    word_timeout = st.sidebar.slider(
        "Jeda Antar Kata (detik)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Waktu jeda antara deteksi kata"
    )

    stable_threshold = st.sidebar.slider(
        "Stabilitas Deteksi",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="Jumlah deteksi konsisten yang diperlukan untuk menambah kata"
    )

    detector = load_detector()
    if detector is None:
        st.stop()

    detector.confidence_threshold = confidence_threshold
    detector.word_timeout = word_timeout
    detector.stable_threshold = stable_threshold

    if 'sentence_history' not in st.session_state:
        st.session_state.sentence_history = []

    if 'main_tab_active' not in st.session_state:
        st.session_state.main_tab_active = 'tab1'
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    tab1, tab2, tab3, tab4 = st.tabs(["Deteksi Langsung", "Upload Gambar", "Video Demo"])
    
    with tab1:
        # Set tab utama aktif
        st.session_state.main_tab_active = 'tab1'
        st.session_state.camera_active = False  # Matikan kamera upload saat di tab1

        st.header("Deteksi Kamera Langsung")
        st.markdown("Deteksi SIBI secara real-time menggunakan kamera dan bangun kalimat otomatis")

        # Tombol kontrol
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Hapus Kalimat"):
                detector.clear_sentence()
                st.rerun()
        with col2:
            # Tombol simpan kalimat
            sentence_info = detector.get_sentence_info()
            if sentence_info['sentence'] and st.button("Simpan Kalimat"):
                st.session_state.sentence_history.append({
                    'sentence': sentence_info['sentence'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'word_count': sentence_info['word_count']
                })
                st.success("Kalimat berhasil disimpan!")

        # Buat layout dengan kamera yang lebih kecil
        col1, col2 = st.columns([1, 2])  # Kamera lebih kecil (1/3), info lebih besar (2/3)

        with col1:
            st.subheader("Kamera")
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
            st.subheader("Hasil Deteksi")
            detection_placeholder = st.empty()

            st.subheader("Kalimat Saat Ini")
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
                            st.metric("Isyarat Saat Ini", result['prediction'])

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                            with col2:
                                st.metric("Stabilitas", f"{detector.stable_detection_count}/{detector.stable_threshold}")
                        else:
                            st.warning("Menunggu deteksi...")

                    # Update tampilan kalimat
                    sentence_info = detector.get_sentence_info()
                    with sentence_placeholder.container():
                        if sentence_info['sentence']:
                            st.success(f"**Kalimat:** {sentence_info['sentence']}")
                            st.info(f"**Jumlah Kata:** {sentence_info['word_count']}")

                            # Tampilkan beberapa kata terakhir
                            if sentence_info['last_words']:
                                recent_words = [w['word'] for w in sentence_info['last_words']]
                                st.text(f"Terbaru: {' → '.join(recent_words)}")
                        else:
                            st.info("Tunjukkan isyarat SIBI untuk mulai membangun kalimat")
                            
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
            st.subheader("Riwayat Kalimat")
            for i, entry in enumerate(reversed(st.session_state.sentence_history[-10:])):  # Show last 10
                with st.expander(f"Kalimat {len(st.session_state.sentence_history)-i}: {entry['sentence'][:50]}..."):
                    st.write(f"**Kalimat Lengkap:** {entry['sentence']}")
                    st.write(f"**Kata:** {entry['word_count']}")
                    st.write(f"**Waktu:** {entry['timestamp']}")

        # Instruksi
        with st.expander("Cara Menggunakan"):
            st.markdown("""
            **Langkah-langkah:**
            1. Klik "Start" untuk mengaktifkan kamera
            2. Izinkan akses kamera saat diminta
            3. Posisikan tangan dengan jelas di depan kamera
            4. Tahan isyarat SIBI dengan stabil
            5. Sistem akan otomatis menambahkan kata ke kalimat
            6. Gunakan "Hapus Kalimat" untuk memulai dari awal
            7. Gunakan "Simpan Kalimat" untuk menyimpan hasil

            **Tips:**
            - Pastikan pencahayaan yang baik
            - Tahan isyarat dengan stabil
            - Sesuaikan pengaturan di sidebar jika diperlukan
            - Refresh halaman jika kamera tidak mau mulai
            """)

    with tab2:
        # Set tab utama aktif dan matikan kamera
        st.session_state.main_tab_active = 'tab2'
        st.session_state.camera_active = False

        st.header("Upload Gambar")
        st.markdown("Upload gambar yang berisi isyarat SIBI untuk dianalisis")

        # Tombol kontrol kalimat
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Hapus Kalimat", key="clear_sentence_upload"):
                detector.clear_sentence()
                st.rerun()
        with col2:
            # Tombol simpan kalimat
            sentence_info = detector.get_sentence_info()
            if sentence_info['sentence'] and st.button("Simpan Kalimat", key="save_sentence_upload"):
                st.session_state.sentence_history.append({
                    'sentence': sentence_info['sentence'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'word_count': sentence_info['word_count']
                })
                st.success("Kalimat berhasil disimpan!")

        # Upload file
        uploaded_file = st.file_uploader(
            "Pilih gambar isyarat SIBI",
            type=['png', 'jpg', 'jpeg'],
            help="Upload gambar yang berisi gerakan isyarat SIBI untuk dideteksi"
        )

        if uploaded_file is not None:
            # Buat layout dengan gambar dan hasil
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Gambar Original")
                # Tampilkan gambar asli
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar yang diupload", use_container_width=True)

            with col2:
                st.subheader("Hasil Deteksi")

                # Proses gambar
                image_array = np.array(image)

                # Proses dengan detector untuk mendapatkan detail lengkap
                annotated_image, detections = process_image_detailed(
                    detector, image_array, confidence_threshold
                )

                # Tampilkan gambar hasil deteksi
                st.image(annotated_image, caption="Hasil Deteksi", use_container_width=True)

                # Tampilkan hasil deteksi
                if detections:
                    st.success(f"🎯 **Ditemukan {len(detections)} isyarat SIBI:**")

                    # Tampilkan setiap deteksi
                    for idx, detection in enumerate(detections):
                        detected_word = detection['class_name']
                        confidence = detection['confidence']

                        with st.container():
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**{idx+1}. {detected_word}** - Confidence: {confidence:.1%}")
                            with col2:
                                # Tombol untuk menambahkan kata ke kalimat
                                if st.button(f"Tambah", key=f"add_word_{idx}"):
                                    # Tambahkan kata langsung ke kalimat tanpa logika waktu
                                    if not isinstance(detector.current_sentence, str):
                                        detector.current_sentence = ""

                                    if detector.current_sentence:
                                        detector.current_sentence += " " + str(detected_word)
                                    else:
                                        detector.current_sentence = str(detected_word)

                                    # Tambahkan ke riwayat kata yang terdeteksi
                                    detector.detected_words.append({
                                        'word': detected_word,
                                        'timestamp': time.time(),
                                        'confidence': confidence
                                    })

                                    st.success(f"Kata '{detected_word}' berhasil ditambahkan!")
                                    st.rerun()

                    # Tombol untuk menambahkan semua kata sekaligus
                    if len(detections) > 1:
                        if st.button("Tambahkan Semua Kata", key="add_all_words"):
                            words_added = []
                            for detection in detections:
                                detected_word = detection['class_name']
                                confidence = detection['confidence']

                                # Tambahkan kata ke kalimat
                                if not isinstance(detector.current_sentence, str):
                                    detector.current_sentence = ""

                                if detector.current_sentence:
                                    detector.current_sentence += " " + str(detected_word)
                                else:
                                    detector.current_sentence = str(detected_word)

                                # Tambahkan ke riwayat kata yang terdeteksi
                                detector.detected_words.append({
                                    'word': detected_word,
                                    'timestamp': time.time(),
                                    'confidence': confidence
                                })
                                words_added.append(detected_word)

                            st.success(f"Semua kata berhasil ditambahkan: {', '.join(words_added)}")
                            st.rerun()
                else:
                    st.warning("Tidak ada isyarat SIBI yang terdeteksi dengan confidence yang cukup")
                    st.info("Coba upload gambar dengan pencahayaan yang lebih baik atau posisi tangan yang lebih jelas")

        # Tampilkan kalimat saat ini
        st.subheader("Kalimat Saat Ini")
        sentence_info = detector.get_sentence_info()
        if sentence_info['sentence']:
            # Tampilkan kalimat dalam kotak yang menonjol
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h4 style="color: #2e7d32; margin: 0;">Kalimat Anda:</h4>
                <p style="font-size: 18px; margin: 10px 0; color: #1b5e20;"><strong>{sentence_info['sentence']}</strong></p>
                <small style="color: #4caf50;">Jumlah Kata: {sentence_info['word_count']}</small>
            </div>
            """, unsafe_allow_html=True)

            # Tampilkan kata-kata terakhir dengan timestamp
            if sentence_info['last_words']:
                st.write("**Kata Terbaru yang Ditambahkan:**")
                for word_info in sentence_info['last_words'][-3:]:  # Show last 3 words
                    timestamp_str = time.strftime("%H:%M:%S", time.localtime(word_info['timestamp']))
                    st.text(f"• {word_info['word']} (confidence: {word_info['confidence']:.1%}, waktu: {timestamp_str})")

        # Riwayat kalimat untuk tab upload
        if st.session_state.sentence_history:
            st.subheader("Riwayat Kalimat")
            for i, entry in enumerate(reversed(st.session_state.sentence_history[-5:])):  # Show last 5
                with st.expander(f"Kalimat {len(st.session_state.sentence_history)-i}: {entry['sentence'][:30]}..."):
                    st.write(f"**Kalimat Lengkap:** {entry['sentence']}")
                    st.write(f"**Kata:** {entry['word_count']}")
                    st.write(f"**Waktu:** {entry['timestamp']}")

        # Instruksi penggunaan
        with st.expander("Cara Menggunakan"):
            st.markdown("""
            **Langkah-langkah:**
            1. Pilih gambar yang berisi isyarat SIBI
            2. Tunggu sistem mendeteksi isyarat
            3. Klik tombol "Tambah" untuk menambahkan kata ke kalimat
            4. Gunakan "Simpan Kalimat" jika sudah selesai

            **Tips:**
            - Gunakan gambar dengan pencahayaan yang baik
            - Pastikan tangan terlihat jelas
            - Format yang didukung: PNG, JPG, JPEG
            """)

    with tab3:
        # Set tab utama aktif dan matikan kamera
        st.session_state.main_tab_active = 'tab3'
        st.session_state.camera_active = False

        st.header("Video Demonstrasi")
        st.markdown("Pelajari cara melakukan setiap isyarat SIBI")

        # Peringatan untuk performa
        st.info("Tips: Tutup tab ini saat menggunakan deteksi langsung untuk performa yang lebih baik.")

        # Kategori kata utama
        with st.expander("Kata Utama", expanded=True):
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
        with st.expander("Kata Penghubung"):
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

if __name__ == "__main__":
    main()