# 🎯 Naive Bayes Classifier - Detaylı Anlatım ve Kod Analizi

## 📚 İçindekiler
1. [Giriş ve Temel Kavramlar](#giriş-ve-temel-kavramlar)
2. [Bayes Teoremi](#bayes-teoremi)
3. [Naive Bayes Matematiği](#naive-bayes-matematiği)
4. [Kod Yapısı Analizi](#kod-yapısı-analizi)
5. [Pratik Uygulama](#pratik-uygulama)
6. [Demo ve Çıktılar](#demo-ve-çıktılar)
7. [Avantajlar ve Dezavantajlar](#avantajlar-ve-dezavantajlar)

---

## 🎯 Giriş ve Temel Kavramlar

Naive Bayes, makine öğrenmesinde en temel ve etkili sınıflandırma algoritmalarından biridir. "Naive" (naif) kelimesi, algoritmanın özelliklerin birbirinden bağımsız olduğunu varsaymasından gelir.

### 🔍 Temel Prensip
Naive Bayes, bir veri noktasının hangi sınıfa ait olduğunu, o sınıfın önceki gözlemlenme sıklığına (prior probability) ve mevcut özelliklerin o sınıfta görülme olasılığına (likelihood) dayanarak tahmin eder.

### 📊 Kullanım Alanları
- **Metin Sınıflandırma**: Spam filtresi, duygu analizi, doküman kategorilendirme
- **Tıbbi Tanı**: Semptomlardan hastalık tahmini
- **Müşteri Segmentasyonu**: Müşteri davranış tahmini
- **Gerçek Zamanlı Tahmin**: Hızlı karar verme sistemleri

---

## 🧮 Bayes Teoremi

Naive Bayes'in temeli Bayes Teoremi'dir:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

### 📝 Formül Açıklaması:
- **P(A|B)**: B gerçekleştiğinde A'nın olasılığı (Posterior)
- **P(B|A)**: A gerçekleştiğinde B'nin olasılığı (Likelihood)
- **P(A)**: A'nın önceki olasılığı (Prior)
- **P(B)**: B'nin olasılığı (Evidence)

### 🎯 Sınıflandırma İçin Uyarlama:
```
P(Sınıf|Özellikler) = P(Özellikler|Sınıf) × P(Sınıf) / P(Özellikler)
```

---

## 🔢 Naive Bayes Matematiği

### 1. Temel Formül
Bir örneğin sınıf `C` ye ait olma olasılığı:

```
P(C|X₁,X₂,...,Xₙ) ∝ P(C) × ∏ᵢ P(Xᵢ|C)
```

### 2. Multinomial Naive Bayes
Metin veriler için kullanılır. Özelliklerin sayım değerleri (kelime frekansları) için uygundur.

```
P(Xᵢ|C) = (Nᵢc + α) / (Nc + α×V)
```

- **Nᵢc**: Sınıf C'de özellik i'nin sayısı
- **Nc**: Sınıf C'deki toplam özellik sayısı
- **α**: Smoothing parametresi (genelde 1.0)
- **V**: Toplam özellik sayısı

### 3. Gaussian Naive Bayes
Sürekli değerler için kullanılır. Özelliklerin normal dağılım gösterdiğini varsayar.

```
P(Xᵢ|C) = (1/√(2πσ²c)) × exp(-((xᵢ-μc)²)/(2σ²c))
```

---

## 💻 Kod Yapısı Analizi

### 🏗️ Class Yapısı
```python
class NaiveBayesClassifier:
    def __init__(self, model_type='multinomial'):
        """
        Model türü seçimi:
        - 'multinomial': Metin/sayım verileri için
        - 'gaussian': Sürekli değerler için
        """
```

### 🎯 Ana Fonksiyonlar

#### 1. Eğitim Fonksiyonu
```python
def train(self, X_train, y_train, feature_names=None):
    """
    Model eğitimi:
    1. Veri tipi kontrolü (sparse/dense)
    2. Model fitting
    3. Metadata kaydetme
    """
```

**Kod Analizi:**
```python
# Veri tipini kontrol et
if hasattr(X_train, 'toarray'):  # Sparse matrix ise
    if self.model_type == 'gaussian':
        X_train = X_train.toarray()  # Gaussian için dense gerekli

# Modeli eğit
self.model.fit(X_train, y_train)
```

#### 2. Tahmin Fonksiyonu
```python
def predict(self, X_test):
    """
    Sınıf tahmini yapar
    En yüksek olasılıklı sınıfı döndürür
    """
```

#### 3. Olasılık Tahmini
```python
def predict_proba(self, X_test):
    """
    Her sınıf için olasılık değerleri döndürür
    Güven seviyesi için kullanılır
    """
```

### 🏆 Özellik Önem Analizi
```python
def get_feature_importance(self, top_n=10):
    """
    MultinomialNB için log-probability değerlerini kullanır
    Her sınıf için en önemli özellikleri bulur
    """
    feature_log_prob = self.model.feature_log_prob_
    # Her sınıf için en yüksek log-probability değerleri
```

---

## 🧪 Pratik Uygulama

### Demo Kodu Çalıştırma

```python
# Demo çalıştırma
python -c "
from models.naive_bayes import demo_naive_bayes
demo_naive_bayes()
"
```

### Beklenen Çıktı Örneği:
```
🧪 Naive Bayes Demo
==============================
1. Multinomial Naive Bayes:
🎯 Naive Bayes (multinomial) eğitimi başlıyor...
✅ Eğitim tamamlandı! Süre: 0.01s
   Sınıf sayısı: 3
   Özellik sayısı: 100

🧪 Test seti üzerinde değerlendirme...
✅ Test Sonuçları:
   Accuracy: 0.8400
   Tahmin süresi: 0.0010s

📋 Sınıflandırma Raporu:
              precision    recall  f1-score   support
           0       0.86      0.82      0.84        67
           1       0.83      0.85      0.84        66
           2       0.83      0.85      0.84        67
    accuracy                           0.84       200
   macro avg       0.84      0.84      0.84       200
weighted avg       0.84      0.84      0.84       200

==================================================
2. Gaussian Naive Bayes:
🎯 Naive Bayes (gaussian) eğitimi başlıyor...
✅ Eğitim tamamlandı! Süre: 0.00s
   Sınıf sayısı: 3
   Özellik sayısı: 100

📊 Karşılaştırma:
Multinomial NB - Accuracy: 0.8400, Süre: 0.010s
Gaussian NB   - Accuracy: 0.8600, Süre: 0.003s
```

---

## 🎯 Gerçek Veri ile Test

### Özellik Çıkarma ile Birlikte Kullanım

```python
from utils.feature_extraction import FeatureExtractor
from models.naive_bayes import NaiveBayesClassifier

# Örnek metin verileri
texts = [
    "Bu ürün çok kaliteli ve hızlı geldi",
    "Kargo çok yavaş, memnun değilim", 
    "Fiyat performans açısından mükemmel"
]
labels = ["pozitif", "negatif", "pozitif"]

# Feature extraction
extractor = FeatureExtractor()
X = extractor.extract_features(texts)

# Model eğitimi
classifier = NaiveBayesClassifier()
classifier.train(X, labels, extractor.get_feature_names())
```

---

## 📊 Model İç Yapısı Analizi

### Bayes Teoremi Hesabı Örneği

Diyelim ki elimizde şu veriler var:
```
Sınıflar: ["pozitif", "negatif"]
Kelimeler: ["harika", "kötü", "ürün"]

Eğitim verileri:
- "harika ürün" -> pozitif
- "kötü ürün" -> negatif  
- "harika" -> pozitif
```

#### Manuel Hesaplama:
```python
# Prior probabilities
P(pozitif) = 2/3 = 0.67
P(negatif) = 1/3 = 0.33

# Likelihood hesabı (Laplace smoothing α=1 ile)
# P("harika"|pozitif) = (2+1) / (3+3) = 3/6 = 0.5
# P("ürün"|pozitif) = (1+1) / (3+3) = 2/6 = 0.33

# Yeni metin: "harika ürün"
P(pozitif|"harika ürün") ∝ P(pozitif) × P("harika"|pozitif) × P("ürün"|pozitif)
                         ∝ 0.67 × 0.5 × 0.33 = 0.11055
```

### Kod ile Doğrulama:
```python
# Model içindeki değerlere erişim
print("Feature log probabilities:")
print(classifier.model.feature_log_prob_)

print("Class log priors:")
print(classifier.model.class_log_prior_)
```

---

## 🚀 Performans Optimizasyonu

### Hiperparametre Tuning

```python
def hyperparameter_tuning(self, X_train, y_train, cv=5):
    """
    Grid Search ile en iyi parametreleri bul
    """
    if self.model_type == 'multinomial':
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],  # Smoothing
            'fit_prior': [True, False]  # Prior kullanım
        }
    else:  # gaussian
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        }
```

### Beklenen Tuning Çıktısı:
```
🔧 Hiperparametre optimizasyonu başlıyor...
✅ En iyi parametreler: {'alpha': 0.1, 'fit_prior': True}
✅ En iyi CV skoru: 0.8750
```

---

## 📈 Avantajlar ve Dezavantajlar

### ✅ Avantajlar:
1. **Hız**: Çok hızlı eğitim ve tahmin
2. **Basitlik**: Anlaşılması ve uygulanması kolay
3. **Az Veri**: Küçük veri setlerinde iyi çalışır
4. **Ölçeklenebilirlik**: Büyük veri setlerinde verimli
5. **Multiclass**: Doğal olarak çok sınıflı
6. **Baseline**: İyi bir başlangıç modeli

### ❌ Dezavantajlar:
1. **Bağımsızlık Varsayımı**: Özelliklerin bağımsız olduğunu varsayar
2. **Kategorik Veriler**: Sürekli verilerle sınırlı performans
3. **Zero Probability**: Görülmemiş kombinasyonlar için smoothing gerekli
4. **Feature Engineering**: İyi özellik seçimi kritik

---

## 🎯 Gerçek Proje Entegrasyonu

### AutoTicketClassifier Projemizdeki Kullanım:

```python
# train_models.py'den
def train_naive_bayes_model(X_train, y_train, X_test, y_test):
    """
    Ticket sınıflandırma için Naive Bayes eğitimi
    """
    print("🎯 Naive Bayes modeli eğitiliyor...")
    
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Naive Bayes Accuracy: {accuracy:.4f}")
    return model, accuracy
```

### Web App Entegrasyonu:
```python
# web/app.py'de kullanım
@st.cache_resource
def load_models():
    """Modelleri yükle"""
    models = {}
    
    try:
        models['naive_bayes'] = joblib.load('models/trained/naive_bayes_model.joblib')
        st.success("✅ Naive Bayes modeli yüklendi")
    except Exception as e:
        st.error(f"❌ Naive Bayes yüklenirken hata: {e}")
    
    return models
```

---

## 🔍 Debugging ve Sorun Giderme

### Yaygın Hatalar ve Çözümleri:

#### 1. Sparse Matrix Hatası
```python
# Hata: Gaussian NB sparse matrix kabul etmez
if hasattr(X_train, 'toarray') and self.model_type == 'gaussian':
    X_train = X_train.toarray()
```

#### 2. Zero Probability Problemi
```python
# Çözüm: Laplace smoothing
MultinomialNB(alpha=1.0)  # alpha > 0 olmalı
```

#### 3. Feature Names Uyumsuzluğu
```python
# Model kaydetme sırasında feature names'i de kaydet
model_data = {
    'model': self.model,
    'feature_names': self.feature_names,
    'classes': self.classes
}
```

---

## 📚 Ek Kaynaklar ve İleri Çalışma

### 📖 Önerilen Okuma:
1. "Pattern Recognition and Machine Learning" - Christopher Bishop
2. "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
3. Scikit-learn Documentation: Naive Bayes

### 🔬 İleri Konular:
1. **Complement Naive Bayes**: Dengesiz veri setleri için
2. **Bernoulli Naive Bayes**: Binary features için
3. **Feature Selection**: Chi-square, Mutual Information
4. **Ensemble Methods**: Naive Bayes + diğer algoritmalar

### 🛠️ Pratik Projeler:
1. Spam e-posta filtreleme
2. Haber kategorilendirme
3. Duygu analizi sistemi
4. Doküman sınıflandırma

---

## 🏁 Sonuç

Naive Bayes, basitliği ve etkinliği ile makine öğrenmesinin temel taşlarından biridir. Bu anlatımda:

✅ **Matematiksel temelleri** öğrendik  
✅ **Kod implementasyonu** analiz ettik  
✅ **Pratik uygulamaları** gördük  
✅ **Debugging teknikleri** öğrendik  
✅ **Gerçek proje entegrasyonu** yaptık  

Naive Bayes, özellikle metin sınıflandırma problemlerinde hala çok etkili bir algoritma. Modern deep learning yaklaşımlarının yanında, hız ve basitlik gerektiren durumlarda tercih edilmeye devam ediyor.

---

*Bu doküman, AutoTicketClassifier projesi kapsamında hazırlanmıştır. Sorularınız için GitHub Issues kullanabilirsiniz.*

---

## 📝 Notlar

- **Tarih**: 25 Haziran 2025
- **Versiyon**: 1.0
- **Proje**: AutoTicketClassifier
- **Yazar**: GitHub Copilot ile hazırlandı
