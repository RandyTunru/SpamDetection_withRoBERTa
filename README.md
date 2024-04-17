# Laporan Proyek Machine Learning - A Analta Dwiyanto

## Domain Proyek

Penggunaan sosial media sudah menjadi bagian penting dari hidup banyak orang, salah satu media yang sering digunakan ialah E-mail. Namun, email juga sering disalahgunakan untuk menyebarkan spam, yang dapat mengganggu dan membahayakan penggunanya. Spam email dapat berupa iklan yang tidak diinginkan, penipuan phishing, malware, dan konten berbahaya lainnya.

Seiring dengan meningkatnya jumlah pengguna email semakin rentan terjadi kejadian penipuan melalui email spam seperti ini. Oleh karena itu, diperlukan sistem yang dapat mendeteksi spam email dengan akurat dan efisien.

## Business Understanding

Spam email merupakan masalah yang signifikan bagi individu dan organisasi. Spam email dapat merugikan seseorang jika bersifat penipuan atau phising, selain itu email spam dapat menurunkan reputasi seseorang atau sebuah perusahaan, bahkan terdapat kasus dimana email spam tersebut memiliki malware.

### Problem Statements

Berdasarkan potensi bahaya yang disebabkan oleh spam email maka problem statements yang ingin diselesaikan proyek ini adalah

- Bagaimana ciri-ciri email spam?
- Bagaimana membuat model Spam Detection yang akurat dan efisien?

### Goals

Berdasarkan problem statements tersebut, maka Goals dari penelitian ini adalah

- Mengetahui ciri-ciri email spam
- Mengetahui cara membuat model Spam Detection yang akurat dan efisien

## Data Understanding

Pada proyek ini, dataset yang digunakan berasal yaitu Spam Emails dari [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/spam-emails). Dataset ini memiliki 5572 data, dengan 87% data merupakan "ham" dan 13% merupakan "spam", data terdiri dari 2 kolom yaitu

### Variabel-variabel pada Spam Email dataset adalah sebagai berikut

- Messages : Feature, berisi teks dari pesan email.
- Category : Target, merupakan kategori dari pesan email tersebut, terdapat 2 jenis kategori yaitu "ham" atau bukan spam dan "spam".

!["Persebaran Data"](images/download.png)

Diagram diatas menunjukkan jumlah data yang merupakan "ham" dan "spam"

!["Panjang 'message' email"](images/download%20(1).png)

Diagram diatas menunjukkan panjang dari email yang terdapat didalam dataset, dapat diamati bahwa rata-rata panjang pesan diantara 1-200 kata, dan terdapat pesan yang panjangnya diatas 800

!["Panjang 'message' email spam dan ham"](images/download%20(2).png)

Diagram diatas menunjukkan panjang dari email yang spam dan ham, dapat diamati bahwa email ham cenderung memiliki panjang pesan rata-rata, sedangkan pesan ham memiliki panjang pesan yang bervariasi bahkan hingga diatas 800

!["Wordcloud Spam"](images/download%20(3).png)

Diagram diatas menunjukkan wordcloud dari pesan spam, dapat diamati bahwa pesan spam cenderung memberi iming iming hadiah dan membuat urgensi dan menyuruh penerima pesan untuk melakukan sesuatu seperti call, text dan sebagainya

## Data Preparation

Pertama-tama kita mendefinisikan kelas Dataset yang akan kita gunakan

```python
class SpamDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'labels': self.labels[idx]}
```

Lalu kita juga mendefinisikan dictionary untuk mengubah label ke index dan sebaliknya untuk kategori teks

```python
LABEL2INDEX = {'ham': 0, 'spam': 1}
INDEX2LABEL = {0: 'ham', 1: 'spam'}
```

Terakhir, kita mentokenisasi teks kita menggunakan tokenizer untuk model kita yaitu RoBERTa-base dan melakukan train test split pada data, pada proyek ini split data bernilai 80-20 dan test data tersebut kemudian dibagi menjadi 50-50 lagi menjadi data validasi dan testing

## Modeling

Pada proyek ini, model yang digunakan ialah RoBERTa atau Robustly Optimized BERT Approach, RoBERTa sendiri merupakan model bahasa yang dikembangkan oleh Facebook AI, yang merupakan perluasan dari model BERT (Bidirectional Encoder Representations from Transformers). RoBERTa didasarkan pada arsitektur Transformer dan dilatih dengan pendekatan self-supervised learning.

## Evaluation

Pada proyek ini, metrik evaluasi yang digunakan adalah akurasi, precision, recall, dan F1-Score.

Akurasi adalah salah satu metrik evaluasi yang umum digunakan dalam klasifikasi. Akurasi mengukur seberapa sering model melakukan prediksi yang benar dibandingkan dengan total jumlah data yang dinilai.

$$Akurasi = \frac{Jumlah Prediksi Benar}{Total Jumlah Data}$$

Recall adalah salah satu metrik evaluasi yang digunakan untuk mengukur seberapa banyak dari keseluruhan instance positif yang berhasil diidentifikasi oleh model.

$$Recall = \frac{True Positives}{True Positives + False Negatives}$$

- True Positives (TP) adalah jumlah instance positif yang berhasil diidentifikasi dengan benar oleh model.
- False Negatives (FN) adalah jumlah instance positif yang salah diklasifikasikan sebagai negatif oleh model.

Precision adalah metrik evaluasi yang digunakan untuk mengukur seberapa banyak dari instance yang diklasifikasikan sebagai positif oleh model yang benar-benar positif.

$$Precision = \frac{True Positives}{True Positives + False Positives}$$

- False Positives (FP) adalah jumlah instance negatif yang salah diklasifikasikan sebagai positif oleh model.

F1-score adalah metrik evaluasi yang menggabungkan precision dan recall menjadi satu nilai tunggal yang mencerminkan kinerja keseluruhan model. F1-score berguna ketika Anda ingin menyeimbangkan antara precision dan recall

$$F1-Score = 2* \frac{Precision * Recall}{Precision + Recall}$$

Pada proyek ini model mendapatkan hasil yang sangat memuaskan dengan Akurasi:0.99 F1-Score:0.98 Recall:0.98 Precision:0.98
