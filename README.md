# 🤟 SIBI Sign Language Detector

Aplikasi web untuk deteksi bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia) dengan dua versi interface: Flask dan Streamlit.

## ✨ Fitur

- 📷 **Real-time Camera Detection** - Deteksi langsung menggunakan kamera
- 📁 **Image Upload** - Upload gambar untuk dianalisis
- 🎯 **High Accuracy** - Menggunakan model YOLO yang sudah dilatih
- 📊 **Confidence Score** - Tampilan tingkat kepercayaan prediksi
- 🎨 **User-friendly Interface** - Interface web yang mudah digunakan
- 📱 **Responsive Design** - Dapat diakses dari berbagai device
- 🔄 **Dual Interface** - Flask (local) dan Streamlit (cloud deployment)

## Persyaratan Sistem

- Python 3.7+
- Webcam/kamera
- Model Ultralytics YOLO yang sudah dilatih (`models/sibiv3.pt`)

## 🚀 Quick Start

### Option 1: Flask Version (Local Development)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Pastikan model YOLO ada di folder `models/sibiv3.pt`

3. Jalankan Flask app:

```bash
python sibi_web_detector.py
```

4. Buka browser ke `http://localhost:5000`

### Option 2: Streamlit Version (Cloud Deployment)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Pastikan model YOLO ada di folder `models/sibiv3.pt`

3. Jalankan Streamlit app:

```bash
python -m streamlit run streamlit_app.py --server.headless true --server.port 8501
```

4. Buka browser ke `http://localhost:8501`

### 🌐 Deploy ke Streamlit Cloud

1. **Fork/Upload ke GitHub**

   - Upload semua file ke repository GitHub Anda
   - Pastikan file `models/sibiv3.pt` ada di repository

2. **Deploy ke Streamlit Cloud**

   - Kunjungi [share.streamlit.io](https://share.streamlit.io)
   - Login dengan GitHub account
   - Klik "New app"
   - Pilih repository dan branch
   - Set main file path: `streamlit_app.py`
   - Klik "Deploy!"

3. **Konfigurasi (Opsional)**
   - File `packages.txt` untuk system dependencies
   - File `.streamlit/config.toml` untuk konfigurasi UI
   - File `requirements.txt` untuk Python dependencies

## 📖 Cara Penggunaan

### 📷 Flask Version (Real-time Video Stream)

1. Buka browser ke `http://localhost:5000`
2. Posisikan tangan di depan kamera
3. Lihat hasil deteksi real-time di browser dengan video stream
4. Klik "Stop Detector" untuk menghentikan

### 📱 Streamlit Version (Real-time Detection)

1. Buka browser ke `http://localhost:8501` (local) atau URL Streamlit Cloud
2. Klik tab "Real-time Detection"
3. Klik "🎥 Start Detection" untuk memulai real-time detection
4. Posisikan tangan dengan gerakan SIBI di depan kamera
5. Lihat hasil deteksi real-time dengan bounding box dan confidence score
6. Klik "⏹️ Stop Detection" untuk menghentikan

### 📁 Upload Image (Streamlit Only)

1. Klik tab "Upload Image"
2. Upload gambar yang berisi gerakan SIBI
3. Lihat hasil analisis dan download hasil jika diperlukan

## 📁 Struktur File

```
sibiwebv2/
├── models/
│   └── sibiv3.pt           # Model Ultralytics YOLO
├── .streamlit/
│   └── config.toml         # Konfigurasi Streamlit
├── sibi_web_detector.py    # Flask version (real-time video stream)
├── streamlit_app.py        # Streamlit version (cloud deployment)
├── packages.txt            # System dependencies untuk deployment
├── requirements.txt        # Python dependencies
└── README.md              # Dokumentasi ini
```

## 🔧 Teknologi

### Flask Version

- **Backend**: Flask untuk web server
- **Real-time**: Video streaming dengan multipart response
- **Interface**: HTML template dengan real-time video feed

### Streamlit Version

- **Frontend**: Streamlit untuk cloud deployment
- **Real-time**: Real-time detection dengan camera stream (sama seperti Flask)
- **Interface**: Start/Stop detection buttons dan image upload
- **Deployment**: Streamlit Cloud ready

### Shared Components

- **Computer Vision**: OpenCV untuk image processing
- **Deep Learning**: PyTorch + Ultralytics YOLO
- **Model**: Same YOLO model dan inference logic

## ⚙️ Settings

- `confidence_threshold`: Ambang batas confidence (default: 0.7)
- `history_size`: Ukuran history untuk smoothing (default: 5)
- Input size model: Ubah di `transforms.Resize((224, 224))`
- Labels: Sesuaikan dengan kelas model Anda

## Troubleshooting

### Kamera tidak terdeteksi

- Pastikan kamera terhubung dan tidak digunakan aplikasi lain
- Coba ubah index kamera di `cv2.VideoCapture(0)` menjadi 1 atau 2

### Model tidak bisa dimuat

- Pastikan file `models/sibiv3.pt` ada dan tidak corrupt
- Periksa kompatibilitas versi PyTorch

### Performa lambat

- Gunakan GPU jika tersedia
- Kurangi resolusi kamera
- Sesuaikan ukuran detection area

## Catatan

- Aplikasi ini menggunakan asumsi umum untuk model deteksi bahasa isyarat
- Sesuaikan preprocessing dan labels sesuai dengan model Anda
- Untuk performa terbaik, gunakan pencahayaan yang baik dan background yang kontras
