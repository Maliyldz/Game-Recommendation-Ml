# 🎮 Oyun Öneri Tahmini - Makine Öğrenmesi Projesi

Bu proje, **Steam** platformuna ait oyun verileri üzerinden bir oyunun **önerilip önerilmeyeceğini tahmin etmek** amacıyla geliştirilmiştir. Çalışma boyunca veri temizleme, ön işleme ve çeşitli makine öğrenmesi modelleri uygulanarak öneri sınıflandırması yapılmıştır.

## 👨‍💻 Proje Ekibi

Bu proje, **Mehmet Ali YILDIZ** ([GitHub Profili](https://github.com/Maliyldz)) ve **Anıl Taha TOMAK** tarafından birlikte geliştirilmiştir.  
Tüm analiz süreci, tek bir bilgisayar üzerinden ortak çalışma ile yürütülmüştür.

## 📁 Veri Seti

- **Kaynak:** Kaggle - Steam Reviews Dataset 2021
- **Hedef Sütun:** `recommended`  
  (Kullanıcının oyunu önerip önermediğini gösterir: `True` / `False`)
- **Diğer Önemli Sütunlar:** `review`, `hours`, `purchase`, `early_access`, vb.

## 🧹 Veri Ön İşleme Adımları

Veri seti üzerinde detaylı ön işleme uygulanarak modelleme için hazır hale getirilmiştir. Bu adımlar şunlardır:

### 1. Eksik Verilerin Temizlenmesi
- Gerekli sütunlarda yer alan eksik veriler tespit edilerek ilgili satırlar kaldırılmış ya da uygun şekilde doldurulmuştur.

### 2. Kategorik Verilerin Dönüştürülmesi
- `purchase`, `early_access` gibi kategorik değişkenler `Label Encoding` ile sayısal değerlere çevrilmiştir.
- Bazı kategorik veriler için `One-Hot Encoding` kullanılmıştır.

### 3. Metin Verisinin Dönüştürülmesi
- `review` sütunundaki yorumlar, doğal dil işleme (NLP) teknikleri ile sayısal forma dönüştürülmüş (TF-IDF gibi).

### 4. Normalizasyon / Standartlaştırma
- Sürekli değişkenler (örneğin `hours`) `MinMaxScaler` veya `StandardScaler` ile ölçeklendirilmiştir.

### 5. Dengesiz Veri Problemi
- `recommended` sınıfında dengesizlik gözlemlenmişse, **SMOTE** veya benzeri tekniklerle dengeleme yapılmıştır.

Proje içerisindeki `cleaned.py` dosyası, veri ön işleme sürecini otomatikleştiren adımları içermektedir.  
Bu dosya adım adım çalıştırılarak, veri seti en baştan işlenmiş hale getirilebilir.

📌 **Dikkat:** `cleaned.py` dosyasını çalıştırmadan önce, Kaggle üzerinden indirilen ham veri dosyasını aynı klasör içerisine yerleştirmeniz gerekmektedir.  
Dosya adı ve formatı orijinal haliyle korunmalıdır.

📌 Eğer projeyi sıfırdan çalıştırmak ve kendi ön işleme sürecinizi izlemek isterseniz:

```bash
python cleaned.py
```

## 📊 Modelleme Senaryoları

Proje kapsamında 3 ayrı yaklaşım denenmiştir:

- **Ham verilerle (standalone)**
- **Genetik algoritma ile feature selection**
- **PCA (Principal Component Analysis) ile boyut indirgeme**

Her senaryo için ayrı analiz yapılmış ve karşılaştırmalı sonuçlar elde edilmiştir.

## 📄 Detaylı Rapor

Model sonuçları ve karşılaştırmalar için:  
👉 [ProjeRaporu.pdf](./ProjeRaporu.pdf)

## 🛠️ Kullanılan Teknolojiler

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost
- DEAP (Genetik algoritma)
- SMOTE (Imbalanced-learn)
- PCA

## 📧 İletişim

Her türlü soru, öneri veya iş birliği için:

- Mehmet Ali YILDIZ: [github.com/Maliyldz](https://github.com/Maliyldz)

---



