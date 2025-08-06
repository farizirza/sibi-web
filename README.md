# Deteksi Bahasa Isyarat SIBI

Aplikasi web untuk mendeteksi bahasa isyarat SIBI (Sistem Isyarat Bahasa Indonesia) menggunakan kecerdasan buatan.

## Apa itu SIBI?

SIBI adalah bahasa isyarat resmi Indonesia yang digunakan untuk berkomunikasi dengan teman-teman tuli dan tunarungu. Aplikasi ini membantu Anda belajar dan menggunakan SIBI dengan teknologi AI.

## Fitur Utama

- **Deteksi Langsung** - Gunakan kamera untuk deteksi real-time
- **Upload Gambar** - Analisis gambar isyarat yang sudah ada
- **Pembangunan Kalimat** - Kata-kata yang terdeteksi otomatis disusun menjadi kalimat
- **Video Pembelajaran** - Lihat contoh gerakan untuk setiap kata
- **Riwayat Kalimat** - Simpan kalimat yang sudah dibuat

## Cara Menggunakan

### Persiapan

1. Pastikan Python 3.7+ sudah terinstall
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Pastikan model AI ada di folder `models/sibiv11m100epoch.pt`

### Menjalankan Aplikasi

```bash
python -m streamlit run streamlit_app.py
```

Buka browser ke `http://localhost:8501`

### Menggunakan Fitur

**Deteksi Langsung:**

1. Klik tab "Deteksi Langsung"
2. Izinkan akses kamera
3. Tunjukkan isyarat SIBI di depan kamera
4. Sistem akan otomatis mengenali dan menyusun kalimat

**Upload Gambar:**

1. Klik tab "Upload Gambar"
2. Pilih gambar yang berisi isyarat SIBI
3. Klik tombol "Tambah" untuk menambahkan kata ke kalimat

## Tips Penggunaan

- Pastikan pencahayaan ruangan cukup terang
- Posisikan tangan dengan jelas di depan kamera
- Untuk gambar, pastikan tangan tidak terpotong
- Gunakan latar belakang yang kontras dengan tangan

## Kosakata yang Didukung

Aplikasi ini dapat mengenali berbagai kata SIBI seperti:

- Kata dasar: saya, kamu, mau, makan, jalan
- Kata kerja: berangkat, terbang, antar, simpan, bantu
- Kata penghubung: ke, di, kan, ber, dan

Lihat tab "Video Demo" untuk mempelajari cara melakukan setiap isyarat.

## Struktur File

```
sibiwebv2/
├── models/sibiv11m100epoch.pt     # Model AI untuk deteksi
├── dataset/                # Video demonstrasi isyarat
├── streamlit_app.py        # Aplikasi utama
├── requirements.txt        # Dependencies Python
└── README.md              # Dokumentasi ini
```

## Teknologi

- **Model AI**: YOLO untuk deteksi objek
- **Framework**: Streamlit untuk interface web
- **Computer Vision**: OpenCV untuk pemrosesan gambar
- **Real-time**: WebRTC untuk streaming kamera

## Kontribusi

Jika Anda ingin berkontribusi:

1. Fork repository ini
2. Buat perubahan yang diperlukan
3. Submit pull request

## Catatan

Model AI (`sibiv11m100epoch.pt`) diperlukan untuk menjalankan aplikasi. Pastikan file ini ada di folder `models/` sebelum menjalankan aplikasi.
