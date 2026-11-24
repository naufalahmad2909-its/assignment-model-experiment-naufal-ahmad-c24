"""Dokumentasi berbasis hasil terukur; dipakai CLI dan notebook."""
from pathlib import Path


def metric_table(report, unique=False):
    key = "metrics_on_unique_test_texts" if unique else "metrics_on_all_test_rows"
    metrics = report[key]
    lines = ["| Pendekatan | Accuracy | Precision macro | Recall macro | F1 macro |",
             "|---|---:|---:|---:|---:|"]
    for name, label in [("classic", "TF-IDF + Logistic Regression"), ("gemini", "Gemini 3.5 Flash-Lite")]:
        row = metrics[name]
        lines.append(f"| {label} | " + " | ".join(f"{row[m]:.4f}" for m in
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"]) + " |")
    return "\n".join(lines)


def analysis_markdown(report, results):
    m = report["metrics_on_all_test_rows"]
    g, c = m["gemini"], m["classic"]
    api = report["gemini"]
    errors = results.loc[results["sentiment"] != results["prediction_classic"]]
    pattern_errors = errors.drop_duplicates("text_group")
    examples = "\n".join(
        f'- `{row.review_text}`: label **{row.sentiment}**, klasik **{row.prediction_classic}**, '
        f'Gemini **{row.prediction_gemini}**. Pola ini muncul '
        f'{int((results["text_group"] == row.text_group).sum())} kali pada test.'
        for row in pattern_errors.itertuples(index=False)
    ) or "Tidak ditemukan kesalahan model klasik pada test set ini."
    gemini_errors = int((results["sentiment"] != results["prediction_gemini"]).sum())
    return f'''Model klasik menghasilkan accuracy **{c['accuracy']:.2%}** dan F1 macro **{c['f1_macro']:.4f}**;
Gemini menghasilkan accuracy **{g['accuracy']:.2%}** dan F1 macro **{g['f1_macro']:.4f}**.
Selisih F1 macro Gemini terhadap klasik adalah **{g['f1_macro'] - c['f1_macro']:.4f}**.

Accuracy menghitung proporsi prediksi benar pada seluruh baris. Precision menilai ketepatan
prediksi per kelas; recall menilai cakupan kelas sebenarnya; F1 menggabungkan keduanya.
Rata-rata macro memberi bobot sama pada kelas positif dan negatif.
Recall negatif klasik **{c['classification_report']['negatif']['recall']:.2%}**,
sedangkan Gemini **{g['classification_report']['negatif']['recall']:.2%}**.
Ini relevan jika tim ingin menangkap keluhan pelanggan yang berisiko terlewat.

**Kesalahan yang diamati**

{examples}

Jumlah baris salah klasik: **{len(errors)}**; jumlah baris salah Gemini: **{gemini_errors}**.
Prediksi salah tersebut merupakan pengamatan. Dugaan penyebab klasik adalah keterbatasan
fitur unigram dan contoh training, bukan hasil uji kausal atau bukti fitur tertentu menjadi penyebab.
Kesalahan lengkap tersedia pada `documentation/error_analysis.csv`.

**Trade-off implementasi, waktu, dan biaya**

| Aspek | Model klasik | Gemini API |
|---|---|---|
| Persiapan | Memerlukan label, fitting TF-IDF, dan training classifier | Memerlukan API key, prompt, validasi respons, dan penanganan kuota |
| Training pada project | Dilatih pada train set lokal | Tidak dilatih/fine-tune pada dataset ini; memakai model pretrained |
| Inferensi | Lokal setelah training, tidak membutuhkan jaringan | Membutuhkan layanan eksternal dan jaringan untuk panggilan baru |
| Pemeliharaan | Perlu data berlabel dan evaluasi ulang ketika pola berubah | Perlu memantau model, versi API, prompt, kuota, dan biaya |
| Biaya | Tidak ada biaya per panggilan API; komputasi tetap memiliki biaya | Bergantung token dan tier akun; cache mengurangi panggilan berulang |

Training klasik termasuk fitting TF-IDF: **{report['classic']['training_seconds_including_tfidf']:.6f} detik**.
Prediksi klasik termasuk transformasi {len(results)} baris: **{report['classic']['test_prediction_seconds_including_tfidf']:.6f} detik**.
Delapan panggilan unik Gemini direkam selama total **{api['sum_api_successful_request_seconds']:.3f} detik**,
dengan rata-rata **{api['mean_api_successful_request_seconds']:.3f} detik per panggilan berhasil**.
Pengukuran ini bersifat observasional pada satu lingkungan. Jumlah item yang dihitung berbeda:
klasik memproses semua baris, Gemini memproses teks unik dan memakai ulang hasil untuk duplikat.
Angka tersebut tidak dipakai sebagai perbandingan throughput yang terkontrol.

Penggunaan Gemini tercatat: input **{api['usage_tokens']['prompt_token_count']} token**,
output final **{api['usage_tokens']['candidates_token_count']} token**, dan thinking
**{api['usage_tokens']['thoughts_token_count']} token**. Estimasi dengan tarif Standard paid tier
adalah **USD {api['estimated_standard_paid_cost_usd']:.7f}** untuk panggilan berhasil yang tercatat.
Ini bukan tagihan aktual: tier, diskon, dan invoice akun tidak diverifikasi. Tarif acuan:
input USD0.30/juta token dan output termasuk thinking USD2.50/juta token,
diakses 25 November 2026 pada [dokumentasi model Google](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite#pricing).

**Keterbatasan**

- Dataset hanya memiliki 40 teks unik, walaupun terdapat 200 baris; test hanya memiliki 8 pola unik.
- Teks yang sering berulang memberi bobot lebih besar pada metrik per baris. Karena itu, metrik 8 teks unik juga disajikan.
- Hanya satu holdout dan satu konfigurasi per pendekatan. Tidak ada tuning memakai label test atau pemilihan seed berdasarkan skor.
- Split berbasis teks mencegah teks identik pada train/test project, tetapi tidak membuktikan bahwa ulasan tidak pernah ada pada pretraining Gemini.
- Label mentor dianggap sebagai ground truth; tidak dilakukan audit label independen.
- Dataset memiliki ulasan singkat dan dua kelas. Belum mewakili sentimen netral/campuran, sarkasme, bahasa baru, atau variasi produksi.
- Model klasik dilatih dengan data kecil; Gemini membawa pengetahuan pretrained. Perbandingan ini merupakan pilihan pendekatan, bukan training dengan anggaran data yang setara.
- Gemini menggunakan satu respons per teks unik. Variasi respons antar-panggilan untuk teks yang sama tidak diuji; temperature rendah bukan jaminan determinisme.
- Skor Gemini yang sempurna pada sampel kecil ini tidak membuktikan performa sempurna di produksi.
'''


def recommendation_markdown(report):
    m = report["metrics_on_all_test_rows"]
    winner = "Gemini API" if m["gemini"]["f1_macro"] > m["classic"]["f1_macro"] else "model klasik"
    return f'''**Rekomendasi: kembangkan {winner} lebih lanjut sebagai kandidat pilot terbatas.**

Pada eksperimen ini, pilihan tersebut didasarkan pada F1 macro dan kemampuan menangkap
kelas negatif pada test set yang sama. Nilai F1 macro klasik adalah
**{m['classic']['f1_macro']:.4f}**, sedangkan Gemini **{m['gemini']['f1_macro']:.4f}**.
Kebutuhan use case adalah mendeteksi sentimen ulasan, termasuk keluhan, sehingga recall
negatif menjadi pertimbangan selain accuracy keseluruhan.

Pilot tetap perlu evaluasi pada kumpulan ulasan baru yang lebih beragam, dengan label yang
ditinjau manusia, target latency dan anggaran yang disepakati, serta pemantauan salah klasifikasi.
Pertahankan model klasik sebagai baseline lokal untuk penelitian berikutnya. Bila jaringan,
pengiriman ulasan ke pihak eksternal, atau biaya API menjadi batasan utama, perluas data
training klasik dan evaluasi ulang sebelum memilih pendekatan produksi.

Keputusan ini tidak menyatakan model sudah siap dipasang ke produksi. Tidak ada target
minimum skor yang dibuat sendiri atau perubahan test set untuk mengejar skor lebih tinggi.
'''


def write_readme(root: Path, report, results):
    s = report["split"]
    api = report["gemini"]
    analysis = analysis_markdown(report, results)
    recommendation = recommendation_markdown(report)
    content = rf'''# AI Model Experiment & Evaluation

**Peserta:** Naufal Ahmad-C24  
**Repository:** `assignment-model-experiment-naufal-ahmad-c24`

Eksperimen membandingkan TF-IDF + Logistic Regression yang dilatih dengan Scikit-learn
dan Gemini API untuk klasifikasi sentimen ulasan pelanggan e-commerce.
Kedua pendekatan menghasilkan prediksi untuk **{s['test_rows']} baris test yang sama**.

## 1. Problem statement

Tim produk ingin mengklasifikasikan ulasan menjadi **positif** atau **negatif** untuk
membantu mengenali kepuasan dan keluhan pelanggan. Tujuan project adalah membandingkan
performa, kebutuhan implementasi, waktu, dan potensi biaya dua pendekatan sebelum
memilih kandidat yang dikembangkan lebih lanjut.

Input prediksi hanya `review_text`; target adalah `sentiment`. `review_id` digunakan untuk
pelacakan hasil, sedangkan `product_name` hanya metadata. Asumsi: label mentor adalah
ground truth, tidak ada kelas netral, dan hasil merupakan eksperimen awal pada dataset kecil.

## 2. Sumber dan kondisi dataset

Sumber: dataset dan starter notebook yang disediakan mentor pada *Individual Assignment -
AI Model Experiment & Evaluation*, AI Engineering Bootcamp. Tidak mencari dataset lain.
File asli disalin utuh menjadi `data/customer_reviews_sentiment.csv`; hanya nama salinan
yang disederhanakan. Dataset tidak ditimpa oleh proses eksperimen.

| Pemeriksaan | Hasil |
|---|---:|
| Baris x kolom asli | 200 x 4 |
| Positif / negatif | 110 / 90 |
| Missing / duplikat penuh | 0 / 0 |
| ID ulasan unik | 200 |
| Teks unik setelah strip + lowercase | 40 |
| Pengulangan teks di luar kemunculan pertama | 160 |
| Teks identik dengan label bertentangan | 0 |

Tidak menghapus ulasan hanya karena teks sama pada produk/ID berbeda. Normalisasi
untuk grouping dilakukan pada salinan DataFrame. Kata negasi tidak dihapus dari fitur.

## 3. Pembagian data dan pencegahan leakage

Kelompok teks unik dibagi dengan `train_test_split(test_size=0.2, random_state=42,
stratify=label_kelompok)`. Hasil pembagian kelompok dipetakan kembali ke seluruh baris:

| Bagian | Jumlah baris | Teks unik | Positif | Negatif |
|---|---:|---:|---:|---:|
| Training | {s['train_rows']} | {s['train_unique_texts']} | {s['train_class_counts']['positif']} | {s['train_class_counts']['negatif']} |
| Test | {s['test_rows']} | {s['test_unique_texts']} | {s['test_class_counts']['positif']} | {s['test_class_counts']['negatif']} |

**Overlap teks train/test = 0.** Pembagian 80:20 berlaku untuk kelompok; proporsi baris
tidak tepat 80:20 karena frekuensi setiap teks berbeda. ID split tersimpan pada
[`split_manifest.csv`](documentation/split_manifest.csv).

Audit pembagian acak per baris seperti starter menemukan **{s['starter_row_split_audit']['test_rows_with_text_seen_in_train']} dari
{s['starter_row_split_audit']['test_rows']} baris test memiliki teks yang sudah ada pada train**.
Karena itu, project memakai pembagian kelompok sebagai eksperimen utama. Audit tersebut
bukan eksperimen tambahan untuk memilih seed atau konfigurasi model berdasarkan skor test.

TF-IDF hanya di-fit pada train. Test tidak dipakai untuk training, pemilihan prompt,
atau tuning hyperparameter. Kedua model memakai urutan dan ID test yang identik.

## 4. Konfigurasi eksperimen

### Model klasik

- `TfidfVectorizer(lowercase=True, ngram_range=(1, 1), stop_words=None)`.
- `LogisticRegression(max_iter=1000)` dengan parameter lain default versi library terkunci.
- Konfigurasi sederhana mengikuti pendekatan starter, tanpa tuning pada test.
- `review_id`, produk, dan label test tidak menjadi fitur model.

### Gemini API

- Model: **`{api['model']}`**, endpoint `generateContent`, SDK `google-genai`.
- Mengganti contoh `gemini-2.0-flash` pada starter karena model tersebut sudah dihentikan.
- Zero-shot: tidak memasukkan contoh berlabel dan tidak mengirim jawaban test ke API.
- Temperature **0.2** untuk mengurangi variasi sampling; tidak menjamin respons identik.
- Thinking level **minimal**, sesuai tugas klasifikasi singkat.
- `max_output_tokens=1024` memberi ruang respons, lalu hasil dibatasi lewat validasi dua label.
- Tidak menggunakan browsing, tools, few-shot, atau fine-tuning.
- Prompt lengkap tersedia pada [`prompt_template.txt`](documentation/prompt_template.txt).

**Prediksi Gemini berasal dari pemanggilan API nyata**, dengan response ID, model version,
raw response, waktu, dan token usage tersimpan pada
[`gemini_responses.json`](documentation/gemini_responses.json). API key tidak disimpan.
Untuk {s['test_rows']} baris dengan {s['test_unique_texts']} teks unik, dilakukan
**{api['unique_api_requests']} panggilan unik**, lalu {api['duplicate_text_predictions_reused']} baris duplikat teks
memakai ulang label yang sesuai. Ini tidak menghasilkan {s['test_rows']} pengamatan API independen.

Mode default notebook/script adalah replay rekaman nyata ini, agar mentor dapat
menjalankan evaluasi tanpa API key dan tanpa biaya panggilan tambahan. Cache diverifikasi
terhadap hash dataset, ID test, model, prompt, konfigurasi, cakupan teks dan validitas label.
Cache bukan mock atau prediksi model klasik. Inferensi baru tersedia melalui flag eksplisit.

## 5. Hasil evaluasi

### Seluruh baris test ({s['test_rows']} baris)

{metric_table(report)}

![Perbandingan metrik](documentation/model_comparison_summary.png)

![Confusion matrix klasik](documentation/confusion_matrix_classic.png)

![Confusion matrix Gemini](documentation/confusion_matrix_gemini.png)

Urutan kelas matriks adalah **negatif, positif**. Baris = label sebenarnya;
kolom = label prediksi. Nilai diagonal adalah prediksi benar. Laporan precision, recall,
F1, dan support per kelas tersedia pada `experiment_report.json` dan output notebook.

### Pemeriksaan tambahan tanpa pembobotan pengulangan ({s['test_unique_texts']} teks unik)

{metric_table(report, unique=True)}

Metrik tambahan dihitung dari prediksi yang sama, satu baris per pola teks.
Ini membantu menunjukkan pengaruh frekuensi teks, bukan evaluasi pada dataset baru.

## 6. Analisis trade-off dan limitation

{analysis}

## 7. Rekomendasi technical approach

{recommendation}

## 8. Struktur project dan output

```text
model-experiment-assignment/
|-- data/customer_reviews_sentiment.csv
|-- notebook/experiment_notebook.ipynb
|-- src/experiment.py
|-- src/reporting.py
|-- tests/test_experiment.py
|-- documentation/
|   |-- experiment_config.json
|   |-- experiment_report.json
|   |-- gemini_responses.json
|   |-- prompt_template.txt
|   |-- split_manifest.csv
|   |-- test_predictions.csv
|   |-- error_analysis.csv
|   |-- metrics_comparison.csv
|   |-- metrics_unique_texts.csv
|   |-- confusion_matrix_classic.png
|   |-- confusion_matrix_gemini.png
|   `-- model_comparison_summary.png
|-- README.md
|-- requirements.txt
`-- .gitignore
```

Notebook mempertahankan delapan section utama dan subbagian starter. Fungsi bersama
di `src/` memungkinkan notebook dan script menggunakan logika eksperimen yang sama.
Notebook sudah disimpan beserta hasil eksekusi dan gambar, sehingga dapat dibaca di GitHub.

## 9. Instalasi dan cara menjalankan

Lingkungan yang digunakan: Python {report['runtime']['python']}, Pandas {report['runtime']['pandas']},
Scikit-learn {report['runtime']['scikit-learn']}, NumPy {report['runtime']['numpy']},
Matplotlib {report['runtime']['matplotlib']}, Google GenAI {report['runtime']['google-genai']}.
`requirements.txt` mencatat versi lingkungan yang berhasil digunakan.

### Windows PowerShell

Dari folder utama project, untuk instalasi baru:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Di folder lokal tugas, `.venv` telah disiapkan. Untuk mengulangi evaluasi dengan rekaman API:

```powershell
Set-Location 'D:\REWORK\Individual Assignment 2\model-experiment-assignment'
.\.venv\Scripts\python.exe src/experiment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Untuk notebook: buka folder project di VS Code, buka `notebook/experiment_notebook.ipynb`,
pilih kernel Python `.venv`, lalu **Restart Kernel and Run All**.
Default `REFRESH_GEMINI=False` memakai rekaman nyata yang disertakan. Replay tidak membutuhkan key.

### Memanggil Gemini lagi

Sediakan `GEMINI_API_KEY` pada environment lokal. Pada notebook, ubah
`REFRESH_GEMINI=True`; bila key belum tersedia, cell meminta key melalui input tersembunyi.
Alternatif CLI setelah environment disiapkan:

```powershell
.\.venv\Scripts\python.exe src/experiment.py --refresh-gemini
```

Panggilan baru menggunakan kuota akun dan dapat menimbulkan biaya sesuai tier. Hasil API
dapat berubah. Refresh akan mengganti rekaman Gemini dan menghasilkan ulang evaluasi;
simpan seluruh output yang diperbarui secara konsisten. Jangan menaruh key di kode,
README, output notebook, atau GitHub. `.venv`, `.env`, dan cache runtime tidak diunggah.

Pada Linux/macOS, gunakan `python3 -m venv .venv` dan `.venv/bin/python` sebagai pengganti
interpreter Windows; pilih kernel yang sama pada notebook.

## 10. Validasi dan pengumpulan

Pengujian memeriksa pemisahan kelompok, integritas raw, alignment prediksi, provenance
dan identitas cache, normalisasi respons, serta metrik yang dihitung ulang dari hasil tersimpan.
Dataset asli dipertahankan dengan SHA-256:

```text
{report['inspection']['raw_sha256']}
```

Unggah isi project ke repository terpisah **`assignment-model-experiment-naufal-ahmad-c24`**,
dengan README di root. Pastikan mentor dapat mengakses repository. Sertakan seluruh
folder dan file pada struktur di atas; `.venv` dan `.runtime` tidak diperlukan untuk pengumpulan.
Kumpulkan satu link halaman utama repository melalui submission tugas 2.

## Referensi

- Brief dan starter notebook mentor: *Individual Assignment - AI Model Experiment & Evaluation*.
- [Pencegahan data leakage, Scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
- [Metrik klasifikasi, Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [Model dan tarif Gemini yang digunakan](https://aistudio.google.com/docs/models/gemini-3.5-flash-lite).
- [Model deprecated, Google](https://aistudio.google.com/docs/deprecations).
- [Gemini thinking level](https://aistudio.google.com/docs/thinking?generate-content=true).
'''
    (root / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    import json
    import pandas as pd
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "documentation/experiment_report.json").read_text(encoding="utf-8"))
    results = pd.read_csv(root / "documentation/test_predictions.csv")
    write_readme(root, report, results)
