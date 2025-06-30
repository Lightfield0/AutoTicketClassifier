# 🚀 Trained Models Directory

Bu klasör eğitilmiş modelleri içerir.

## 📁 Model Dosyaları

### Küçük Modeller (GitHub'da mevcut)
- `naive_bayes_multinomial.pkl` - Naive Bayes sınıflandırıcısı (123KB)
- `logistic_regression.pkl` - Logistic Regression modeli (71KB)  
- `tfidf_vectorizer.joblib` - TF-IDF vektörleştirici (34KB)
- `label_encoder.joblib` - Label encoder (569B)

### Büyük Modeller (GitHub'da değil)
- `bert_classifier.pth` - BERT modeli (422MB) ⚠️ **GitHub'da değil**

## 🤖 BERT Modelini Kullanmak İçin

BERT modeli GitHub'da yer almıyor çünkü çok büyük (422MB). Yerel olarak kullanmak için:

1. **Model eğitimi yapın:**
```bash
python train_models.py
# "2. Kapsamlı eğitim" seçeneğini seçin
```

2. **Veya önceden eğitilmiş modeli indirin:**
```bash
# Eğer model başka bir yerden paylaşılıyorsa
# wget https://example.com/bert_classifier.pth
```

## 🎯 Model Performansları

Tüm modeller %100 accuracy elde etti:
- **Naive Bayes**: 1.0000 accuracy, 0.000315s prediction time
- **Logistic Regression**: 1.0000 accuracy, 0.000814s prediction time  
- **BERT**: 1.0000 accuracy, 38.65s prediction time

## 📊 Kullanım

```python
from models.naive_bayes import NaiveBayesClassifier
from models.logistic_regression import LogisticRegressionClassifier
from models.bert_classifier import BERTClassifier

# Küçük modeller (hızlı)
nb = NaiveBayesClassifier()
nb.load_model('models/trained/naive_bayes_multinomial.pkl')

lr = LogisticRegressionClassifier()  
lr.load_model('models/trained/logistic_regression.pkl')

# BERT (yavaş ama güçlü)
bert = BERTClassifier()
bert.load_model('models/trained/bert_classifier.pth')
```
