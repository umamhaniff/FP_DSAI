> [!NOTE] tekan _Ctrl+Shift+V_ untuk preview

# Getting Ready

## 1. Verifikasi Instalasi (Prasyarat)

Sebelum memulai, pastikan Python dan PIP sudah terinstal dengan benar dan terdeteksi oleh sistem.
Jalankan perintah berikut di terminal (Command Prompt / PowerShell):

```bash
    where python
    where pip
```

Contoh Output yang Benar:

```bash
    C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe
    C:\Users\HP\AppData\Local\Programs\Python\Python313\Scripts\pip.exe
```

(Lokasi file mungkin berbeda tergantung instalasi di komputer Anda)

## 2. Instalasi Dependensi

Install semua library yang diperlukan untuk proyek ini yang tercantum dalam requirements.txt.

- Opsi 1: Instalasi Standar (Disarankan)

```bash
    pip install -r requirements.txt
```

- Opsi 2: Instalasi "Quiet" (Tampilan terminal lebih bersih)

```bash
    pip install -r requirements.txt -q
```

## 3. Menjalankan Aplikasi

Gunakan salah satu perintah di bawah ini untuk memulai server Streamlit dan membuka aplikasi.
Perintah Utama:

```bash
    streamlit run app_streamlit.py
```

Perintah Alternatif: Gunakan perintah ini jika perintah utama mengalami kendala path/module:

```bash
    python -m streamlit run app_streamlit.py
```

---

> [!TIP] Catatan Tambahan  
> Pastikan Anda menjalankan terminal di dalam folder proyek yang memuat file _app_streamlit.py_ dan _requirements.txt_.  
> Jika ingin menghentikan aplikasi, tekan Ctrl + C pada terminal.

---

### 📋 Git Cheat Sheet (Lengkap)

| Command             | Fungsi                                                   | Contoh Penggunaan                                      |
| :------------------ | :------------------------------------------------------- | :----------------------------------------------------- |
| **Combo Upload** ⚡ | 🚀 **Cara Cepat** (3-in-1) Add + Commit + Push sekaligus | `git add . && git commit -m "update code" && git push` |
| `git status`        | 🔍 Cek file yang berubah                                 | `git status`                                           |
| `git add`           | 📦 Masukkan file ke Staging                              | `git add .` (semua)<br>`git add file.py` (satu)        |
| `git commit`        | 💾 Simpan perubahan                                      | `git commit -m "pesan"`                                |
| `git push`          | ☁️ Upload ke GitHub                                      | `git push origin main`                                 |
| `git pull`          | ⬇️ Ambil update dari GitHub                              | `git pull`                                             |
| `git log`           | 📜 Lihat riwayat                                         | `git log --oneline`                                    |

---

# Versi:

- streamlit run app_streamlit.py = maybe can be Final File
- streamlit run app_streamlit2.py = Hybrid+CF 
- streamlit run app_streamlit3.py = UI/UX Ekspresif
- streamlit run app_streamlit4.py = Afrida & Hanifa
- streamlit run app_streamlit5.py = Kak Agnes & Lisa
