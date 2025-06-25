# 🎫 AutoTicket Classifier
*Destek Taleplerini Otomatik Etiketleyen AI Sistemi*

## 📋 Proje Açıklaması
Bu proje, müşteri destek taleplerini otomatik olarak kategorilere ayıran bir yapay zeka sistemidir. Naive Bayes'ten BERT'e kadar farklı machine learning yaklaşımlarını karşılaştırır.

## 🎯 Desteklenen Kategoriler
- 💳 **Ödeme Sorunu**: Ödeme işlemleri, fatura, ücretlendirme
- 📅 **Rezervasyon Problemi**: Rezervasyon iptali, değişiklik, onay
- 👤 **Kullanıcı Hatası**: Hesap erişimi, şifre, profil
- 😞 **Şikayet**: Hizmet kalitesi, personel, genel memnuniyetsizlik
- ❓ **Genel Bilgi**: Ürün bilgisi, nasıl kullanılır, özellikler
- 🔧 **Teknik Sorun**: Uygulama hatası, bağlantı, performans

## 🤖 AI Modelleri
1. **Naive Bayes** - Baseline model
2. **Logistic Regression** - Linear classifier
3. **BERT** - Transformer-based deep learning

## 🚀 Hızlı Başlangıç

### Kurulum
```bash
pip install -r requirements.txt
```

### Veri Hazırlama
```bash
python data_generator.py
```

### Model Eğitimi
```bash
python train_models.py
```

### Web Arayüzü
```bash
streamlit run app.py
```

### API Sunucusu
```bash
python api_server.py
```

## 📊 Model Performansları
| Model | Accuracy | F1-Score | Eğitim Süresi |
|-------|----------|----------|---------------|
| Naive Bayes | ~85% | ~0.83 | < 1 dakika |
| Logistic Regression | ~88% | ~0.86 | ~2 dakika |
| BERT | ~93% | ~0.91 | ~15 dakika |

## 📁 Proje Yapısı
```
AutoTicketClassifier/
├── data/
│   ├── raw_tickets.json          # Ham veri
│   ├── processed_data.csv        # İşlenmiş veri
│   └── train_test_split/         # Eğitim/test ayrımı
├── models/
│   ├── naive_bayes.py           # Naive Bayes implementasyonu
│   ├── logistic_regression.py   # Logistic Regression
│   ├── bert_classifier.py       # BERT modeli
│   └── trained/                 # Eğitilmiş modeller
├── utils/
│   ├── text_preprocessing.py    # Metin ön işleme
│   ├── feature_extraction.py    # Özellik çıkarma
│   └── evaluation.py           # Model değerlendirme
├── web/
│   ├── app.py                  # Streamlit uygulaması
│   └── api_server.py           # FastAPI sunucusu
├── notebooks/
│   └── analysis.ipynb          # Veri analizi ve deneyler
├── data_generator.py           # Sentetik veri üretici
├── train_models.py            # Model eğitim scripti
└── demo.py                    # Hızlı demo
```

## 🎓 Öğrenme Hedefleri
- **Text Classification** temellerini öğrenme
- **Feature Engineering** ile TF-IDF, n-grams
- **Traditional ML** vs **Deep Learning** karşılaştırması
- **Model Evaluation** metrikleri (Precision, Recall, F1)
- **Real-world Application** deployment

## 💡 İlerisi için Fikirler
- Çoklu dil desteği
- Intent detection + entity extraction
- Active learning ile model iyileştirme
- A/B testing altyapısı
- Gerçek zamanlı monitoring
