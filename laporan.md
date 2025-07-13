# Laporan Proyek Machine Learning

## Domain Proyek

Setiap individu memiliki pola konsumsi makanan yang berbeda. Rasio makronutrien harian yang tidak seimbang dapat memengaruhi energi, berat badan, hingga risiko penyakit metabolik [1], [2]. Oleh karena itu, klasifikasi pola makan berdasarkan rasio kalori dari karbohidrat, protein, dan lemak dapat menjadi alat bantu penting dalam:

- Memberikan insight terhadap gaya hidup makan.
- Mendorong kesadaran gizi yang lebih personal.
- Membantu otomatisasi rekomendasi nutrisi dengan cara yang dapat diskalakan.

Menurut World Health Organization (2023), sekitar **60% penyakit tidak menular** seperti diabetes tipe 2, hipertensi, dan obesitas berkaitan langsung dengan **pola makan tidak seimbang** [4]. Banyak individu tanpa disadari mengonsumsi makanan tinggi karbohidrat atau lemak secara berlebihan tanpa kompensasi asupan protein dan serat yang cukup [3], [5].

Di sisi lain, laporan dari USDA mencatat bahwa lebih dari **75% orang dewasa di dunia barat melebihi batas asupan lemak jenuh**, sementara kurang dari **30% memenuhi kebutuhan protein harian secara optimal** [5].

> Oleh karena itu, pendekatan berbasis data untuk mengklasifikasikan dominansi makronutrien harian secara otomatis menjadi penting sebagai langkah awal menuju **intervensi nutrisi personal yang lebih efektif dan terukur** [2], [3].
---
**Rubrik/Kriteria Tambahan (Opsional)**:

**Why?**
- 📈 **Kebutuhan akan rekomendasi nutrisi personal** meningkat seiring tren gaya hidup sehat.
- 🧩 Banyak pengguna tidak memahami distribusi gizi dalam pola makannya.
- ⚖️ Dominansi makronutrien yang tidak seimbang dapat memicu masalah kesehatan jangka panjang.

**How?**
- Dengan menghitung proporsi kalori harian dari makronutrien utama, kita dapat **melabeli tiap hari** sebagai `Karbo-Dominan`, `Protein-Dominan`, atau `Lemak-Dominan`.
- Label ini akan diprediksi berdasarkan fitur numerik harian pengguna dengan dua model supervised learning:
  - 🌲 Random Forest  
  - 🚂 Logistic Regression

- 🔍 Riset & Referensi Relevan

Berikut adalah referensi utama yang mendasari pendekatan klasifikasi dominansi makronutrien harian dalam proyek ini:

1. Harvard T.H. Chan School of Public Health. (n.d.). *Dietary Macronutrient Distribution and Cardiometabolic Risk*. Retrieved from https://www.hsph.harvard.edu

2. World Health Organization – Regional Office for Europe. (2015). *Understanding nutrient profiling: an implementation tool for public health policies*. Retrieved from https://www.who.int

3. U.S. Department of Agriculture & U.S. Department of Health and Human Services. (2020). *Dietary Guidelines for Americans, 2020–2025*. 9th Edition. Retrieved from https://www.dietaryguidelines.gov

4. World Health Organization. (2023). *Noncommunicable Diseases Factsheet*. Retrieved from https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases

5. U.S. Department of Agriculture (USDA) & Centers for Disease Control and Prevention (CDC). (2022). *What We Eat in America – NHANES 2017–2020*. Retrieved from https://www.cdc.gov/nchs/nhanes/index.htm

---
## Business Understanding

Dataset yang digunakan dalam proyek ini terdiri dari **1000 baris data konsumsi makanan harian**, yang mencakup jumlah kalori dan proporsi makronutrien utama: karbohidrat, protein, dan lemak. Setiap baris merepresentasikan total konsumsi gizi seseorang dalam satu hari, dengan atribut numerik seperti total kalori, gram karbohidrat, gram protein, dan gram lemak.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana mengklasifikasikan hari makan pengguna menjadi Karbo-dominan, Protein-dominan, atau Lemak-dominan berdasarkan total asupan harian?
- Apakah model machine learning mampu secara akurat memprediksi dominansi makronutrien dari data gizi numerik pengguna?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun sistem klasifikasi makronutrien harian pengguna berdasarkan data makanan & nutrisi.
- Membandingkan performa antara model Random Forest dan Logistic Regression dalam tugas klasifikasi ini dan Menghasilkan metrik evaluasi seperti Akurasi, Precision, dan F1-Score untuk masing-masing model.

**Rubrik/Kriteria Tambahan (Opsional)**:

**🚂 Solusi 1 – Logistic Regression**
- **Deskripsi**: Model linier probabilistik yang memetakan fitur (karbo, protein, lemak, kalori) ke peluang tiap kelas dominansi.
- **Metrik Evaluasi**:
  - Akurasi Klasifikasi
  - Macro F1-Score
  - Confusion Matrix

**🌲 Solusi 2 – Random Forest**
- **Deskripsi**: Ensemble decision tree yang kuat terhadap noise dan dapat menangkap non-linearitas dalam data konsumsi makanan.
- **Metrik Evaluasi**:
  - Akurasi
  - F1-Score per kelas (karbo/protein/lemak)
  - Confusion Matrix

---
## Data Understanding
1. Struktur Data

>- 10.000 entri, 14 kolom, tanpa missing values → data bersih & siap olah.  
>- Fitur terdiri dari data numerik (kalori & nutrisi) dan kategorikal (nama makanan, kategori, tanggal, dll).

2. Statistik Nutrisi

>- Rata-rata kalori per makanan: 328 kkal
>- Protein, karbohidrat, dan lemak berkisar antara 1–100 gram, menunjukkan variasi yang luas antar makanan.
>- Distribusi nutrien relatif normal, meski ada indikasi outlier pada sodium dan kolesterol.

3. User & Tanggal

>- Perlu agregasi per (User_ID, Date) untuk menghitung total asupan harian & rasio makronutrien.

4. Kesiapan Modeling

>- Dataset ideal untuk supervised classification setelah agregasi & labeling dominansi makronutrien.

### **📊 Distribusi Kategori Makanan**
Dataset terdiri dari tujuh kategori makanan utama yang relatif seimbang jumlahnya:
- Kategori paling sering dikonsumsi: **Dairy (1,460), Fruits (1,453), Beverages (1,445)**
- Kategori paling sedikit: **Grains (1,384), Vegetables (1,408)**  
Distribusi ini mencerminkan keragaman konsumsi makanan yang cukup merata di antara pengguna.

### **🍽️ Rata-rata Nutrisi Berdasarkan Kategori**
Nilai rata-rata nutrisi utama (per food item) relatif seragam antar kategori:
- **Snacks & Beverages** cenderung memiliki kandungan kalori, karbohidrat, dan protein sedikit lebih tinggi.
- **Vegetables** memiliki kalori dan lemak paling rendah, sesuai karakteristik makanan sehat.

> Ini menunjukkan bahwa meskipun kategori berbeda, kontribusi nutrisi tiap makanan tetap relevan dan dapat digunakan untuk klasifikasi.

### **🥪 Distribusi Meal Type**
Empat jenis waktu makan menunjukkan jumlah yang hampir seimbang:
- **Breakfast (2,559), Dinner (2,503), Lunch (2,487), Snack (2,451)**  
Artinya, data tidak bias terhadap waktu makan tertentu dan siap digunakan untuk analisis harian.

### **📈 Statistik Deskriptif Umum**
- Rata-rata konsumsi per item: **328 kkal, 25g protein, 52g karbo, 25g lemak**
- Rentang nilai cukup luas (kalori: 50–600, sodium: 0–1000 mg) — menunjukkan pentingnya normalisasi saat modeling.
- Nilai median ≈ mean, menunjukkan distribusi yang tidak terlalu skewed.

> ### Source: https://www.kaggle.com/datasets/adilshamim8/daily-food-and-nutrition-dataset

### Variabel-variabel pada Daily Food & Nutrition Dataset UCI dataset adalah sebagai berikut:
Dataset ini terdiri dari 14 kolom yang merepresentasikan informasi makanan harian dan nilai nutrisinya. Berikut adalah deskripsi masing-masing fitur:

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| `Date` | object (datetime) | Tanggal konsumsi makanan (format: YYYY-MM-DD) |
| `User_ID` | int | ID unik pengguna yang mengonsumsi makanan tersebut |
| `Food_Item` | object | Nama makanan yang dikonsumsi (contoh: Apple, Chicken Breast) |
| `Category` | object | Kategori makanan (contoh: Fruits, Meat, Dairy) |
| `Calories (kcal)` | int | Total energi dari makanan, dalam kilokalori |
| `Protein (g)` | float | Jumlah protein dalam gram |
| `Carbohydrates (g)` | float | Jumlah total karbohidrat dalam gram |
| `Fat (g)` | float | Jumlah total lemak dalam gram |
| `Fiber (g)` | float | Kandungan serat dalam gram |
| `Sugars (g)` | float | Kandungan gula dalam gram |
| `Sodium (mg)` | int | Kandungan natrium dalam miligram |
| `Cholesterol (mg)` | int | Kandungan kolesterol dalam miligram |
| `Meal_Type` | object | Jenis waktu makan (contoh: Breakfast, Lunch, Dinner, Snack) |
| `Water_Intake (ml)` | int | Jumlah air yang diminum saat makan, dalam mililiter |

**Rubrik/Kriteria Tambahan (Opsional)**:
- Pengecekan data

![Gambar 3](https://github.com/TheNoblePhantasm/ImageNich/blob/59fbf2776cee46dde97265dda4d3dccd57d0a8aa/data%20feature%20info.png)
![Gambar 4](https://github.com/TheNoblePhantasm/ImageNich/blob/59fbf2776cee46dde97265dda4d3dccd57d0a8aa/missing%20value%20check.png)

- Visualisasi awal

![Gambar 1](https://github.com/TheNoblePhantasm/ImageNich/blob/59fbf2776cee46dde97265dda4d3dccd57d0a8aa/distribusi%20kalori%20per%20item%20makanan.png)

![Gambar 2](https://github.com/TheNoblePhantasm/ImageNich/blob/59fbf2776cee46dde97265dda4d3dccd57d0a8aa/top%2010%20makanan%20most%20consumable.png)

---
## Data Preparation
**Teknik yang digunakan**

- **Kalkulasi energi dari makronutrien**  
  Langkah pertama dalam persiapan data adalah menghitung total energi (kalori) dari masing-masing makronutrien berdasarkan nilai gram yang tersedia, dengan konversi standar sebagai berikut:  
  - Karbohidrat = 4 kcal/gram  
  - Protein = 4 kcal/gram  
  - Lemak = 9 kcal/gram  
  Perhitungan ini dilakukan untuk setiap entri makanan pada dataset mentah.

- **Agregasi per hari per user**  
  Setelah energi dihitung, data kemudian diagregasi berdasarkan kombinasi `Date` dan `User_ID`. Nilai energi dari setiap makronutrien serta total kalori dijumlahkan untuk menghasilkan total asupan harian per individu. Langkah ini menyederhanakan data menjadi satu baris per hari per pengguna.

- **Hitung proporsi energi dari masing-masing makro**  
  Proporsi energi masing-masing makro terhadap total kalori harian dihitung menggunakan rumus berikut:  
  - `Perc_Carbs = Energy_Carbs / Total_Calories`  
  - `Perc_Protein = Energy_Protein / Total_Calories`  
  - `Perc_Fat = Energy_Fat / Total_Calories`  
  Proporsi ini memberikan gambaran kontribusi relatif dari masing-masing makronutrien terhadap total energi harian.

- **Label klasifikasi dominansi makro**  
  Kelas target `Dominant_Macro` ditentukan berdasarkan makronutrien dengan proporsi energi tertinggi pada hari tersebut. Label akhir dikategorikan sebagai:  
  - **Karbo-Dominan**  
  - **Protein-Dominan**  
  - **Lemak-Dominan**

- **Pemisahan dan pembagian data: Training dan Testing**  
  Data dibagi menjadi fitur (`X`: energi karbo, protein, lemak) dan label (`y`: Dominant_Macro), kemudian dilakukan pembagian data menggunakan `train_test_split` dengan rasio 80:20 dan stratifikasi untuk menjaga proporsi kelas seimbang.

- **Normalisasi fitur**  
  Sebelum model Logistic Regression dilatih, fitur dinormalisasi menggunakan **StandardScaler** agar memiliki mean 0 dan standar deviasi 1. Hal ini penting karena algoritma ini sensitif terhadap skala fitur.

**Rubrik/Kriteria Tambahan (Opsional)**: 
**Mengapa Teknik-Teknik Tersebut Penting?:**  
- **Kalkulasi energi** diperlukan untuk mengubah satuan makronutrien menjadi satuan energi yang konsisten dan relevan bagi analisis nutrisi.
- **Agregasi harian** merangkum data mentah ke dalam bentuk yang sesuai dengan tujuan prediksi, yaitu per hari per user.
- **Proporsi energi** digunakan untuk menilai dominansi relatif, bukan absolut, dari makronutrien terhadap total energi harian.
- **Label dominansi** merupakan target klasifikasi yang memungkinkan penerapan supervised learning.
- **Split data** penting agar model dapat dievaluasi secara adil terhadap data yang tidak terlihat saat pelatihan.
- **Normalisasi fitur** meningkatkan stabilitas dan performa model Logistic Regression.

---
## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan klasifikasi dominansi makronutrien harian. Dua algoritma yang digunakan adalah **Logistic Regression** dan **Random Forest Classifier**.

### 🚂 Logistic Regression

Logistic Regression adalah model klasifikasi linier yang menghitung probabilitas suatu input termasuk ke dalam suatu kelas, menggunakan fungsi **logistik (sigmoid)**.  
Untuk kasus multikelas seperti ini, digunakan pendekatan **multinomial**, yaitu memperluas fungsi logistik ke dalam bentuk softmax.

Model mempelajari bobot koefisien untuk setiap fitur yang kemudian digunakan dalam fungsi berikut:

$$
P(y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j} e^{\beta_j^T x}}
$$

- **multi_class='multinomial'** dan **solver='lbfgs'** digunakan untuk menangani klasifikasi multikelas.
- Data fitur dinormalisasi sebelum pelatihan agar model bekerja lebih optimal.

Kelebihan:  
✔ Sederhana dan mudah diinterpretasikan  
✔ Efisien untuk data dengan relasi linier antar fitur  
Kekurangan:  
✘ Kurang mampu menangkap hubungan non-linear antar fitur

### 🌳 Random Forest Classifier

Random Forest adalah algoritma **ensemble learning** berbasis **decision tree** yang menggunakan teknik **bagging (bootstrap aggregating)**. Model membangun banyak decision tree dari subset acak data dan fitur, lalu menggabungkan hasil prediksi setiap tree (voting) untuk menghasilkan prediksi akhir.

Ciri khas Random Forest:
- Setiap pohon dilatih dari subset acak data (sampling dengan pengembalian)
- Pada setiap node, fitur yang digunakan untuk split dipilih secara acak dari subset fitur
- Hasil prediksi adalah **kelas terbanyak** dari semua pohon (majority vote)

Parameter:
- `n_estimators=100`: jumlah pohon dalam forest
- `random_state=42`: untuk replikasi hasil

Kelebihan:  
✔ Mampu menangani data non-linear  
✔ Tahan terhadap overfitting  
✔ Tidak memerlukan normalisasi fitur  
Kekurangan:  
✘ Interpretabilitas lebih rendah dibandingkan model linier

**Rubrik/Kriteria Tambahan (Opsional)**: 
- **Logistic Regression** lebih sederhana dan cepat, cocok untuk baseline model.  
- **Random Forest** lebih kuat menangani kompleksitas data, cocok untuk meningkatkan akurasi dan menangani non-linearitas.
- ### **✅ Pemilihan Model Terbaik**

##### **🔍 Alasan Memilih Logistic Regression sebagai Model Terbaik:**

- **Akurasi Tertinggi:** **`🚂 Logistic Regression`** memberikan akurasi hampir sempurna (99.9%) dibanding Random Forest (98.0%).
- **Stabil di Semua Kelas**: Tidak ada kesalahan prediksi signifikan; hanya satu kasus minor di kelas Protein-Dominan.
- **Kesederhanaan dan Interpretabilitas**: Logistic Regression mudah diinterpretasikan dan diimplementasikan, terutama saat fitur terbatas dan tidak kompleks.

##### **🎯 Kesimpulan:**
Model **🚂 Logistic Regression** dipilih sebagai **solusi terbaik** karena memberikan performa paling optimal, minim kesalahan, serta cukup andal untuk kebutuhan klasifikasi makronutrien dominan pengguna secara harian.

---
## Evaluation

Hasil evaluasi dua model klasifikasi yang digunakan:

| Model                | Akurasi | Macro Precision | Macro Recall | Macro F1-Score | Catatan                                                                 |
|---------------------|---------|------------------|---------------|----------------|-------------------------------------------------------------------------|
| Logistic Regression | 99.95%  | 1.00             | 1.00          | 1.00           | Performa sempurna hampir tanpa kesalahan. Hanya 1 salah klasifikasi.   |
| Random Forest       | 98.00%  | 0.98             | 0.98          | 0.98           | Performa sangat tinggi, namun masih terjadi misklasifikasi minor antar kelas. |

### 📏 Metrik Evaluasi yang Digunakan  

Untuk mengevaluasi kinerja model klasifikasi pada proyek ini, digunakan beberapa metrik utama berikut:
---

#### **Accuracy (Akurasi)**  
  Mengukur seberapa banyak prediksi yang benar dibanding total keseluruhan prediksi.  
  Cocok digunakan ketika distribusi kelas cukup seimbang, seperti pada kasus ini.

  $$
  \text{Accuracy} = \frac{\text{Jumlah prediksi benar}}{\text{Total prediksi}}
  $$

---

#### **Precision**  
  Mengukur proporsi prediksi positif yang benar (menghindari false positives).

  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP + FP}}
  $$

---

#### **Recall**  
  Mengukur proporsi kasus positif yang berhasil diprediksi benar (menghindari false negatives).

  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP + FN}}
  $$

---

#### **F1-Score**  
  Merupakan rata-rata harmonik dari precision dan recall, berguna saat perlu menyeimbangkan keduanya.

  $$
  F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

---
  Metrik ini memberikan gambaran lebih detail tentang performa model pada masing-masing kelas, terutama saat penting meminimalkan kesalahan pada kategori tertentu (misalnya dominansi makronutrien).

- **Confusion Matrix**  
  Matriks ini menunjukkan secara eksplisit jumlah prediksi benar dan salah dari tiap kelas. Berguna untuk memahami pola kesalahan model.

---
## Kesimpulan
### 📊 Hasil Evaluasi Model

| Model                | Akurasi | Macro Precision | Macro Recall | Macro F1-Score | Catatan                                                                 |
|---------------------|---------|------------------|---------------|----------------|-------------------------------------------------------------------------|
| Logistic Regression | 99.95%  | 1.00             | 1.00          | 1.00           | Performa sempurna hampir tanpa kesalahan. Hanya 1 salah klasifikasi.   |
| Random Forest       | 98.00%  | 0.98             | 0.98          | 0.98           | Performa sangat tinggi, namun masih terjadi misklasifikasi minor antar kelas. |

**🚂 Logistic Regression** berhasil memprediksi dengan sangat akurat jenis makronutrien dominan harian pengguna, dengan tingkat kesalahan yang sangat rendah dan distribusi prediksi yang stabil di semua kelas.

---

### 🔍 Evaluasi Model & Pemilihan Terbaik:

| Model                | Akurasi | Macro F1-Score | Catatan                                                  |
|---------------------|---------|----------------|----------------------------------------------------------|
| Logistic Regression | **99.9%** | **1.00**       | Performa hampir sempurna dengan hanya satu kesalahan minor |
| Random Forest        | 98.0%   | 0.98           | Akurasi tinggi, namun sedikit kesalahan klasifikasi       |

##### 🚂 Logistic Regression dipilih sebagai model terbaik karena:
- **Akurasi tertinggi** (99.9%) di antara semua model.
- **Stabil di semua kelas**, termasuk kelas minoritas seperti Protein-Dominan.
- **Mudah diinterpretasikan**, ideal untuk aplikasi nyata berbasis nutrisi.

---

### 🍽️ Insight Pola Makan:

| Meal Type | Karbo-Dominan | Lemak-Dominan | Protein-Dominan | Total |
|-----------|----------------|----------------|------------------|--------|
| Breakfast | 1,087          | **1,315**      | 157              | 2,559  |
| Lunch     | 1,048          | 1,276          | 163              | 2,487  |
| Dinner    | 1,010          | 1,326          | 167              | 2,503  |
| Snack     | 1,053          | 1,225          | **173**          | 2,451  |

- **Breakfast** didominasi lemak (1,315), jauh lebih tinggi dari karbohidrat (1,087), mencerminkan konsumsi umum seperti **Milk**, **Butter**, dan **Coffee**.
- **Lunch dan Dinner** memiliki pola mirip—dominansi lemak tetap dominan, sedangkan **protein relatif rendah**, meski secara ideal waktu ini cocok untuk asupan protein optimal.
- **Snack** menarik karena mencatat jumlah **Protein-Dominan tertinggi (173)**, sering kali disumbang oleh konsumsi **Nuts**, **Milk**, dan **Chocolate**—meskipun sebagian juga tinggi lemak.

---

### 🧪 Top 5 Makanan Dominan:

| Dominant Macro   | Top Makanan                    |
|------------------|--------------------------------|
| Karbo-Dominan    | Coffee, Popcorn, Chips, Water, Pork Chop |
| Lemak-Dominan    | Orange, Spinach, Quinoa, Apple, Milk     |
| Protein-Dominan  | Chocolate, Nuts, Butter, Green Tea, Milk |

📌 **Catatan Penting:**  
Sejumlah makanan sehat seperti **Apple**, **Spinach**, dan **Orange** justru sering muncul di hari-hari **Lemak-Dominan**. Ini menunjukkan bahwa:
> "Makanan sehat ≠ hari dengan pola makan sehat."

---

### 🎯 Implikasi & Rekomendasi:

- Fokus edukasi sebaiknya bergeser dari **item tunggal** ke **komposisi total harian**. Konsumsi buah/sayur tidak otomatis menyeimbangkan dominansi makronutrien.
- **Camilan** berperan besar dalam dominansi lemak harian—kandungan kalori tersembunyi dalam snack seperti **Chips** dan **Chocolate** harus diwaspadai.
- Perlu peningkatan **kehadiran protein** pada **makan siang dan malam**, mengingat rendahnya jumlah hari Protein-Dominan di waktu-waktu makan utama.

---

### 🧩 Final Verdict:
>- 🚂 **Logistic Regression** terbukti sebagai model terbaik untuk klasifikasi dominansi makronutrien harian.
>
>- Performanya sangat tinggi, stabil, dan cocok untuk deployment nyata dalam aplikasi nutrisi berbasis data.

**---Ini adalah bagian akhir laporan---**



