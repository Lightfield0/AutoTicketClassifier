# 🎫 AutoTicket Classifier
*Production-Ready AI Sistemi - Destek Taleplerini Otomatik Sınıflandırma*

## 📋 Proje Açıklaması
Bu proje, müşteri destek taleplerini otomatik olarak kategorilere ayıran **production-ready** yapay zeka sistemidir. 

🚀 **Özellikler**:
- 🔄 **A/B Testing Framework**: Model performansını karşılaştırmalı test etme
- 📊 **Production Monitoring**: Real-time drift detection ve performance tracking
- 🎯 **Advanced Model Evaluation**: Comprehensive validation ve overfitting detection
- 🤖 **Ensemble Learning**: Birden fazla modeli birleştirme
- 📈 **Data Augmentation**: Otomatik veri çoğaltma ve dengeleme
- 🌐 **Web Interface**: Streamlit UI ve FastAPI REST API

## 🎯 Desteklenen Kategoriler
- 💳 **Ödeme Sorunu**: Ödeme işlemleri, fatura, ücretlendirme
- 📅 **Rezervasyon Problemi**: Rezervasyon iptali, değişiklik, onay
- 👤 **Kullanıcı Hatası**: Hesap erişimi, şifre, profil
- 😞 **Şikayet**: Hizmet kalitesi, personel, genel memnuniyetsizlik
- ❓ **Genel Bilgi**: Ürün bilgisi, nasıl kullanılır, özellikler
- 🔧 **Teknik Sorun**: Uygulama hatası, bağlantı, performans

## 🤖 AI Modelleri & Ensemble
1. **Naive Bayes** - Hızlı baseline model
2. **Logistic Regression** - Linear classifier
3. **BERT** - Transformer-based deep learning (dbmdz/bert-base-turkish-cased)
4. **Weighted Ensemble** - Model kombinasyonu

## 🔧 Sistem Mimarisi

```
AutoTicketClassifier/
├── models/                 # AI modelleri
│   ├── naive_bayes.py
│   ├── logistic_regression.py
│   ├── bert_classifier.py
│   ├── ensemble_system.py
│   └── trained/           # Eğitilmiş model dosyaları
├── utils/                  # Araçlar
│   ├── text_preprocessing.py
│   ├── feature_extraction.py
│   ├── evaluation.py
│   ├── monitoring.py
│   └── data_generator.py
├── web/                    # Web uygulaması
│   ├── app.py             # Streamlit app
│   └── api_server.py      # FastAPI server
├── monitoring/             # Production logs
└── data/                  # Veri dosyaları
```

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

### Web Uygulaması
```bash
# Streamlit UI
streamlit run web/app.py

# FastAPI Server (alternatif)
python -m uvicorn web.api_server:app --reload
```

### Production Monitoring
```bash
# Monitoring dashboard
python -c "from utils.monitoring import ProductionMonitor; m = ProductionMonitor(); print(m.health_check())"
```

## 🔥 Temel Kullanım

### 1. A/B Testing Framework
```python
from web.app import ABTestingFramework

ab_tester = ABTestingFramework()
ab_tester.create_experiment("model_comparison", models, traffic_split)
```

### 2. Production Monitoring
```python
from utils.monitoring import ProductionMonitor

monitor = ProductionMonitor()
monitor.log_prediction(model_name, text, prediction, confidence)
drift_result = monitor.detect_data_drift(new_data)
```

### 3. Advanced Model Evaluation
```python
from utils.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.comprehensive_model_evaluation(model, X_train, X_test, y_train, y_test)
```

### 4. Ensemble Learning
```python
from models.ensemble_system import WeightedEnsemble

ensemble = WeightedEnsemble(models={'nb': nb_model, 'lr': lr_model})
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

### 5. Data Augmentation
```python
from web.app import DataAugmentationPipeline

augmenter = DataAugmentationPipeline()
augmented_data = augmenter.augment_dataset(df)
```

## 📊 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 0.87 | 0.86 | 0.87 | 0.86 |
| Logistic Regression | 0.89 | 0.88 | 0.89 | 0.88 |
| BERT | 0.92 | 0.91 | 0.92 | 0.91 |
| **Ensemble** | **0.94** | **0.93** | **0.94** | **0.93** |

## 🚀 Production Features

### Monitoring & Alerting
- ✅ Real-time drift detection
- ✅ Performance degradation alerts
- ✅ Prediction logging
- ✅ Health checks

### Web Interface
- ✅ Streamlit interactive dashboard
- ✅ FastAPI REST endpoints
- ✅ Real-time predictions
- ✅ Model comparison views

### Scalability
- ✅ A/B testing framework
- ✅ Model versioning
- ✅ Ensemble learning
- ✅ Performance monitoring

## 📁 Dosya Yapısı

```
├── data/                   # Veri dosyaları
├── models/                 # AI modelleri
│   ├── trained/           # Eğitilmiş model dosyaları
│   ├── naive_bayes.py     # Naive Bayes classifier
│   ├── logistic_regression.py # Logistic Regression classifier
│   ├── bert_classifier.py # BERT transformer model
│   └── ensemble_system.py # Ensemble modeli
├── utils/                  # Yardımcı araçlar
│   ├── monitoring.py      # Production monitoring
│   ├── evaluation.py      # Model evaluation
│   ├── text_preprocessing.py # Metin işleme
│   ├── feature_extraction.py # Özellik çıkarma
│   └── data_generator.py  # Veri üretimi
├── web/                    # Web uygulaması
│   ├── app.py             # Streamlit interface
│   └── api_server.py      # FastAPI REST API
├── monitoring/             # Monitoring logs & database
└── tests/                 # Test dosyaları
```

## 🛠️ Geliştirme

### Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Linting
pylint models/ utils/ web/

# Formatting
black .
```

## 📈 Monitoring Dashboard

Production monitoring dashboard özellikleri:
- 📊 Real-time prediction metrics
- 🔍 Data drift detection
- ⚡ Performance tracking
- 🚨 Automated alerting
- 📋 Model comparison reports

## 🚀 Çalıştırma Seçenekleri

### 1. Streamlit UI (Varsayılan)
```bash
streamlit run web/app.py
```

### 2. FastAPI Server
```bash
python -m uvicorn web.api_server:app --reload
```

### 3. Python Script Olarak
```python
from web.app import TicketClassifierApp

app = TicketClassifierApp()
prediction = app.predict_ticket("Ödeme yapamıyorum")
print(prediction)
```

## 📞 Destek ve Katkıda Bulunma

Bu proje **production-ready** durumda ve enterprise-grade kullanıma hazır!

### Ana Özellikler:
- ✅ Enhanced feature extraction
- ✅ Production monitoring & drift detection
- ✅ A/B testing framework
- ✅ Advanced model validation
- ✅ Ensemble learning
- ✅ Data augmentation
- ✅ Turkish BERT integration
- ✅ Real-time performance tracking
- ✅ Web-based interactive interface

---
🎉 **Enterprise-grade AI sistemi - Production Ready!**
