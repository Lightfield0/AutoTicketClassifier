"""
🎮 Hızlı Demo Scripti
AutoTicket Classifier'ı hızlıca test edin
"""

import pandas as pd
import numpy as np
import time
import os
import sys

# Kendi modüllerimizi import et
from utils.text_preprocessing import TurkishTextPreprocessor
from utils.feature_extraction import FeatureExtractor
from utils.evaluation import ModelEvaluator

def print_header(title):
    """Başlık yazdır"""
    print("\n" + "="*60)
    print(f"🎫 {title}")
    print("="*60)

def demo_preprocessing():
    """Metin ön işleme demo'su"""
    print_header("Metin Ön İşleme Demo")
    
    # Örnek metinler
    sample_texts = [
        "Merhaba, kredi kartımdan para çekildi ama rezervasyonum onaylanmadı! Çok acil lütfen...",
        "Şifremi unuttum, nasıl değiştirebilirim? Yardım edin please.",
        "Site çok yavaş yükleniyor!!! Bu durumda çok üzgünüm 😞",
        "İyi günler, çalışma saatleriniz nedir? Teşekkürler.",
        "PERSONEL ÇOK KABA DAVRANADI!!! Şikayet etmek istiyorum!!!"
    ]
    
    preprocessor = TurkishTextPreprocessor()
    
    print("📝 Örnek Metinler ve Ön İşleme Sonuçları:")
    print("-" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Orijinal:")
        print(f"   {text}")
        
        # Farklı ön işleme seviyeleri
        basic_clean = preprocessor.clean_text(text)
        full_process = preprocessor.preprocess_text(text, remove_stopwords=True)
        with_stemming = preprocessor.preprocess_text(text, remove_stopwords=True, apply_stemming=True)
        
        print(f"   Temel Temizlik: {basic_clean}")
        print(f"   Stop Words Kaldırılmış: {full_process}")
        print(f"   Stemming Uygulanmış: {with_stemming}")

def demo_feature_extraction():
    """Özellik çıkarma demo'su"""
    print_header("Özellik Çıkarma Demo")
    
    # Örnek dataset
    sample_data = {
        'text': [
            "kredi kartı para çekildi rezervasyon onaylanmadı",
            "şifre unuttum hesap giriş yapamıyorum",
            "site yavaş yükleniyor sayfa açılmıyor",
            "çalışma saatleri hafta sonu açık mısınız",
            "personel kaba davrandı şikayet etmek istiyorum",
            "rezervasyon iptal etmek nasıl yapabilirim",
            "ödeme sorunu fatura yanlış geldi",
            "teknik destek uygulama çalışmıyor"
        ],
        'category': [
            'payment_issue', 'user_error', 'technical_issue', 'general_info',
            'complaint', 'reservation_problem', 'payment_issue', 'technical_issue'
        ]
    }
    
    texts = sample_data['text']
    labels = sample_data['category']
    
    extractor = FeatureExtractor()
    
    # TF-IDF features
    print("🔢 TF-IDF Özellikleri çıkarılıyor...")
    tfidf_matrix, feature_names = extractor.extract_tfidf_features(texts, max_features=50)
    
    print(f"✅ {tfidf_matrix.shape[1]} TF-IDF özelliği oluşturuldu")
    
    # En önemli özellikleri göster
    top_features = extractor.get_top_features(tfidf_matrix, feature_names, labels, n_top=3)
    extractor.print_top_features(top_features)
    
    # İstatistiksel özellikler
    print("\n📊 İstatistiksel Özellikler:")
    stat_features = extractor.extract_statistical_features(texts)
    print(stat_features[['char_count', 'word_count', 'exclamation_count', 'urgency_words']].head())

def demo_model_comparison():
    """Model karşılaştırma demo'su"""
    print_header("Model Karşılaştırma Demo")
    
    # Sentetik veri oluştur
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("🎲 Sentetik veri oluşturuluyor...")
    X, y = make_classification(
        n_samples=500,
        n_features=100,
        n_classes=6,
        n_informative=80,
        n_redundant=20,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Model simülasyonu
    evaluator = ModelEvaluator()
    
    # Farklı performanslarda 3 model simülasyonu
    np.random.seed(42)
    
    # Model 1: Naive Bayes (simulation)
    y_pred1 = y_test.copy()
    noise_indices1 = np.random.choice(len(y_test), int(len(y_test) * 0.15), replace=False)
    y_pred1[noise_indices1] = np.random.randint(0, 6, len(noise_indices1))
    
    # Model 2: Logistic Regression (simulation)
    y_pred2 = y_test.copy()
    noise_indices2 = np.random.choice(len(y_test), int(len(y_test) * 0.10), replace=False)
    y_pred2[noise_indices2] = np.random.randint(0, 6, len(noise_indices2))
    
    # Model 3: BERT (simulation)
    y_pred3 = y_test.copy()
    noise_indices3 = np.random.choice(len(y_test), int(len(y_test) * 0.05), replace=False)
    y_pred3[noise_indices3] = np.random.randint(0, 6, len(noise_indices3))
    
    # Mock model class
    class MockModel:
        def __init__(self, predictions, pred_time):
            self.predictions = predictions
            self.pred_time = pred_time
        def predict(self, X):
            time.sleep(self.pred_time)  # Simulate prediction time
            return self.predictions
    
    # Değerlendirme
    print("📊 Modeller değerlendiriliyor...")
    
    evaluator.evaluate_model(MockModel(y_pred1, 0.01), X_test, y_test, model_name="Naive Bayes")
    evaluator.evaluate_model(MockModel(y_pred2, 0.05), X_test, y_test, model_name="Logistic Regression")
    evaluator.evaluate_model(MockModel(y_pred3, 0.5), X_test, y_test, model_name="BERT")
    
    # Karşılaştırma
    comparison_df = evaluator.compare_models()
    evaluator.generate_summary_report()

def demo_real_examples():
    """Gerçek örnek ticket'lar ile demo"""
    print_header("Gerçek Örnek Ticket'lar")
    
    # Gerçekçi örnekler
    real_examples = [
        {
            "text": "Kredi kartımdan 150 TL çekildi ama rezervasyonum gözükmüyor sistemde. Lütfen kontrol edin ve geri iade edin.",
            "expected": "payment_issue",
            "category_tr": "💳 Ödeme Sorunu"
        },
        {
            "text": "Şifremi unuttum ve hesabıma giriş yapamıyorum. SMS doğrulama kodu da gelmiyor. Nasıl çözebilirim?",
            "expected": "user_error", 
            "category_tr": "👤 Kullanıcı Hatası"
        },
        {
            "text": "Mobil uygulama sürekli çöküyor. Android 12 kullanıyorum. Bu sorunu ne zaman çözeceksiniz?",
            "expected": "technical_issue",
            "category_tr": "🔧 Teknik Sorun"
        },
        {
            "text": "Rezervasyonumu iptal etmek istiyorum. İptal ücreti var mı? Ne kadar sürede iade alırım?",
            "expected": "reservation_problem",
            "category_tr": "📅 Rezervasyon Problemi"
        },
        {
            "text": "Personel çok ilgisizdi ve sorularıma tatmin edici cevap vermedi. Bu hizmet standardınız mı?",
            "expected": "complaint",
            "category_tr": "😞 Şikayet"
        },
        {
            "text": "Çalışma saatleriniz nedir? Hafta sonu da açık mısınız? Hangi ödeme yöntemlerini kabul ediyorsunuz?",
            "expected": "general_info",
            "category_tr": "❓ Genel Bilgi"
        }
    ]
    
    print("🎯 Örnek Ticket'lar ve Beklenen Kategoriler:")
    print("-" * 80)
    
    for i, example in enumerate(real_examples, 1):
        print(f"\n{i}. Ticket:")
        print(f"   📝 Metin: {example['text']}")
        print(f"   🎯 Beklenen: {example['category_tr']}")
        print(f"   🔧 Kategori ID: {example['expected']}")
        
        # Metin analizi
        preprocessor = TurkishTextPreprocessor()
        processed = preprocessor.preprocess_text(example['text'])
        
        word_count = len(example['text'].split())
        char_count = len(example['text'])
        urgency_words = sum(1 for word in ['acil', 'hemen', 'çabuk'] if word in example['text'].lower())
        
        print(f"   📊 Analiz: {word_count} kelime, {char_count} karakter, {urgency_words} aciliyet kelimesi")

def demo_performance_comparison():
    """Performans karşılaştırması"""
    print_header("Performans Karşılaştırması")
    
    # Simüle edilmiş model performansları
    models_performance = {
        "Naive Bayes": {
            "accuracy": 0.847,
            "f1_score": 0.834,
            "training_time": 0.8,
            "prediction_time": 0.012,
            "memory_usage": "50 MB",
            "pros": ["Çok hızlı", "Az bellek kullanır", "Basit"],
            "cons": ["Düşük doğruluk", "Feature independence varsayımı"]
        },
        "Logistic Regression": {
            "accuracy": 0.884,
            "f1_score": 0.876,
            "training_time": 2.3,
            "prediction_time": 0.025,
            "memory_usage": "120 MB",
            "pros": ["İyi performans", "Yorumlanabilir", "Stabil"],
            "cons": ["Linear varsayım", "Feature engineering gerekli"]
        },
        "BERT": {
            "accuracy": 0.932,
            "f1_score": 0.928,
            "training_time": 856.0,
            "prediction_time": 0.48,
            "memory_usage": "2.1 GB",
            "pros": ["En yüksek doğruluk", "Context anlayışı", "Transfer learning"],
            "cons": ["Çok yavaş", "Çok bellek", "Karmaşık"]
        }
    }
    
    print("📈 Model Performans Karşılaştırması:")
    print("-" * 80)
    
    for model_name, perf in models_performance.items():
        print(f"\n🤖 {model_name}")
        print(f"   Accuracy: {perf['accuracy']:.3f}")
        print(f"   F1-Score: {perf['f1_score']:.3f}")
        print(f"   Eğitim Süresi: {perf['training_time']:.1f}s")
        print(f"   Tahmin Süresi: {perf['prediction_time']:.3f}s")
        print(f"   Bellek Kullanımı: {perf['memory_usage']}")
        print(f"   ✅ Avantajlar: {', '.join(perf['pros'])}")
        print(f"   ❌ Dezavantajlar: {', '.join(perf['cons'])}")
    
    # Öneri sistemi
    print(f"\n💡 Kullanım Önerileri:")
    print("   🏎️  Hız önceliği: Naive Bayes")
    print("   ⚖️  Denge (hız/doğruluk): Logistic Regression")
    print("   🎯 Doğruluk önceliği: BERT")

def main():
    """Ana demo fonksiyonu"""
    print("🎫 AutoTicket Classifier - Hızlı Demo")
    print("AI öğrenmek için kapsamlı ticket sınıflandırma projesi")
    print("="*60)
    
    demos = {
        "1": ("Metin Ön İşleme", demo_preprocessing),
        "2": ("Özellik Çıkarma", demo_feature_extraction),
        "3": ("Model Karşılaştırma", demo_model_comparison),
        "4": ("Gerçek Örnekler", demo_real_examples),
        "5": ("Performans Analizi", demo_performance_comparison),
        "6": ("Tümünü Çalıştır", lambda: [demo() for demo in [
            demo_preprocessing, demo_feature_extraction, 
            demo_real_examples, demo_performance_comparison
        ]])
    }
    
    print("\n📋 Mevcut Demo'lar:")
    for key, (name, _) in demos.items():
        print(f"   {key}. {name}")
    
    print("\n🚀 Hızlı başlangıç için sadece Enter'a basın (Tümünü çalıştırır)")
    
    choice = input("\nSeçiminiz (1-6 veya Enter): ").strip()
    
    if not choice:
        choice = "6"  # Varsayılan: tümü
    
    if choice in demos:
        start_time = time.time()
        demos[choice][1]()
        end_time = time.time()
        
        print(f"\n✅ Demo tamamlandı! Süre: {end_time - start_time:.1f} saniye")
    else:
        print("❌ Geçersiz seçim!")
    
    print(f"\n🎓 Sonraki Adımlar:")
    print("   1. 📊 Veri üret: python data_generator.py")
    print("   2. 🤖 Modelleri eğit: python train_models.py")
    print("   3. 🌐 Web uygulaması: streamlit run web/app.py")
    print("   4. 🚀 API sunucusu: python web/api_server.py")

if __name__ == "__main__":
    main()
