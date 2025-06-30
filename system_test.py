#!/usr/bin/env python3
"""
🧪 Sistem Bütünlük Testi - AutoTicket Classifier
Projenin tüm bileşenlerinin çalışabilirliğini test eder
"""

import sys
import os
import traceback
from datetime import datetime

# Ana dizini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Temel import testleri"""
    print("📦 Temel Import Testleri")
    print("-" * 30)
    
    tests = [
        ("utils.text_preprocessing", "TurkishTextPreprocessor"),
        ("utils.feature_extraction", "FeatureExtractor"),
        ("utils.evaluation", "ModelEvaluator"),
        ("utils.monitoring", "ProductionMonitor"),
        ("models.naive_bayes", "NaiveBayesClassifier"),
        ("models.logistic_regression", "LogisticRegressionClassifier"),
        ("models.ensemble_system", "EnsembleManager"),
        ("data_generator", "TicketDataGenerator")
    ]
    
    success_count = 0
    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
    
    print(f"\n📊 Import Başarı Oranı: {success_count}/{len(tests)}")
    return success_count == len(tests)

def test_data_generation():
    """Veri üretimi testi"""
    print("\n🎲 Veri Üretimi Testi")
    print("-" * 25)
    
    try:
        from data_generator import TicketDataGenerator
        
        generator = TicketDataGenerator()
        
        # Küçük bir test dataseti oluştur
        test_df = generator.generate_comprehensive_dataset(n_samples=10)
        
        if len(test_df) == 10:
            print("✅ Veri üretimi başarılı")
            print(f"   Kategoriler: {test_df['category'].unique()}")
            return True
        else:
            print(f"❌ Veri üretimi hatası: {len(test_df)} != 10")
            return False
            
    except Exception as e:
        print(f"❌ Veri üretimi hatası: {e}")
        return False

def test_text_preprocessing():
    """Metin ön işleme testi"""
    print("\n🔧 Metin Ön İşleme Testi")
    print("-" * 30)
    
    try:
        from utils.text_preprocessing import TurkishTextPreprocessor
        
        preprocessor = TurkishTextPreprocessor()
        
        test_texts = [
            "Kredi kartımdan para çekildi ama rezervasyonum onaylanmadı!",
            "Şifremi unuttum, nasıl değiştirebilirim?",
            "Bu çok kötü bir hizmet. Şikayet ediyorum!"
        ]
        
        processed_texts = []
        for text in test_texts:
            processed = preprocessor.preprocess_text(text)
            processed_texts.append(processed)
        
        if all(processed_texts):
            print("✅ Metin ön işleme başarılı")
            print(f"   Örnek: '{test_texts[0][:30]}...' -> '{processed_texts[0][:30]}...'")
            return True
        else:
            print("❌ Metin ön işleme hatası: Boş sonuç")
            return False
            
    except Exception as e:
        print(f"❌ Metin ön işleme hatası: {e}")
        return False

def test_feature_extraction():
    """Özellik çıkarma testi"""
    print("\n🔢 Özellik Çıkarma Testi")
    print("-" * 30)
    
    try:
        from utils.feature_extraction import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        test_texts = [
            "ödeme sorunu yaşıyorum",
            "rezervasyon iptal etmek istiyorum",
            "şifremi unuttum yardım edin",
            "site çalışmıyor teknik sorun",
            "personel kaba davrandı şikayet"
        ]
        
        # TF-IDF features
        features, feature_names = extractor.extract_tfidf_features(test_texts, max_features=20)
        
        if features.shape[0] == len(test_texts) and features.shape[1] > 0:
            print("✅ Özellik çıkarma başarılı")
            print(f"   Boyut: {features.shape}")
            print(f"   Örnek özellikler: {list(feature_names[:5])}")
            return True
        else:
            print(f"❌ Özellik çıkarma hatası: Geçersiz boyut {features.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Özellik çıkarma hatası: {e}")
        return False

def test_model_training():
    """Model eğitimi testi"""
    print("\n🤖 Model Eğitimi Testi")
    print("-" * 25)
    
    try:
        from models.naive_bayes import NaiveBayesClassifier
        from utils.feature_extraction import FeatureExtractor
        
        # Test verisi
        test_texts = [
            "ödeme sorunu yaşıyorum",
            "rezervasyon iptal etmek istiyorum", 
            "şifremi unuttum yardım edin",
            "site çalışmıyor teknik sorun",
            "personel kaba davrandı şikayet"
        ]
        
        test_labels = [
            "payment_issue",
            "reservation_problem",
            "user_error", 
            "technical_issue",
            "complaint"
        ]
        
        # Feature extraction
        extractor = FeatureExtractor()
        features, _ = extractor.extract_tfidf_features(test_texts, max_features=20)
        
        # Model eğitimi
        model = NaiveBayesClassifier()
        model.train(features, test_labels)
        
        # Tahmin testi
        predictions = model.predict(features)
        
        if len(predictions) == len(test_labels):
            print("✅ Model eğitimi ve tahmini başarılı")
            print(f"   Tahminler: {predictions}")
            return True
        else:
            print(f"❌ Model tahmini hatası: {len(predictions)} != {len(test_labels)}")
            return False
            
    except Exception as e:
        print(f"❌ Model eğitimi hatası: {e}")
        traceback.print_exc()
        return False

def test_web_app_components():
    """Web app bileşenleri testi"""
    print("\n🌐 Web App Bileşenleri Testi")
    print("-" * 35)
    
    try:
        # Streamlit import testi
        import streamlit as st
        print("✅ Streamlit import başarılı")
        
        # Web app'teki temel sınıfları test et
        from web.app import ABTestingFramework, PerformanceMonitor
        
        ab_tester = ABTestingFramework()
        monitor = PerformanceMonitor()
        
        print("✅ Web app bileşenleri başarılı")
        return True
        
    except Exception as e:
        print(f"❌ Web app bileşenleri hatası: {e}")
        return False

def run_integration_test():
    """Entegrasyon testi"""
    print("\n🔄 Entegrasyon Testi")
    print("-" * 25)
    
    try:
        # Tam pipeline testi
        from data_generator import TicketDataGenerator
        from utils.text_preprocessing import TurkishTextPreprocessor
        from utils.feature_extraction import FeatureExtractor
        from models.naive_bayes import NaiveBayesClassifier
        
        # 1. Veri üret
        generator = TicketDataGenerator()
        df = generator.generate_comprehensive_dataset(n_samples=20)
        
        # 2. Metin ön işle
        preprocessor = TurkishTextPreprocessor()
        processed_texts = [preprocessor.preprocess_text(text) for text in df['message']]
        
        # 3. Özellik çıkar
        extractor = FeatureExtractor()
        features, _ = extractor.extract_tfidf_features(processed_texts, max_features=50)
        
        # 4. Model eğit
        model = NaiveBayesClassifier()
        model.train(features, df['category'])
        
        # 5. Tahmin yap
        predictions = model.predict(features)
        
        print("✅ Tam pipeline entegrasyonu başarılı")
        print(f"   Veri: {len(df)} sample")
        print(f"   Özellik: {features.shape[1]} features")
        print(f"   Kategori: {len(set(df['category']))} categories")
        print(f"   Tahmin: {len(predictions)} predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Entegrasyon testi hatası: {e}")
        traceback.print_exc()
        return False

def main():
    """Ana test fonksiyonu"""
    print("🧪 AUTOTICKET CLASSIFIER - SİSTEM BÜTÜNLÜK TESTİ")
    print("=" * 60)
    print(f"📅 Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Test 1: Temel importlar
    test_results.append(("Import Testleri", test_basic_imports()))
    
    # Test 2: Veri üretimi
    test_results.append(("Veri Üretimi", test_data_generation()))
    
    # Test 3: Metin ön işleme
    test_results.append(("Metin Ön İşleme", test_text_preprocessing()))
    
    # Test 4: Özellik çıkarma
    test_results.append(("Özellik Çıkarma", test_feature_extraction()))
    
    # Test 5: Model eğitimi
    test_results.append(("Model Eğitimi", test_model_training()))
    
    # Test 6: Web app bileşenleri
    test_results.append(("Web App Bileşenleri", test_web_app_components()))
    
    # Test 7: Entegrasyon
    test_results.append(("Entegrasyon", run_integration_test()))
    
    # Sonuçları özetle
    print("\n📊 TEST SONUÇLARI ÖZETİ")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 GENEL SONUÇ: {passed}/{total} TEST BAŞARILI")
    
    if passed == total:
        print("🎉 TÜM TESTLER BAŞARILI! Proje tamamen çalışır durumda.")
        return True
    elif passed >= total * 0.8:
        print("⚠️ Çoğu test başarılı. Proje büyük ölçüde çalışır durumda.")
        return True
    else:
        print("🚨 Çok sayıda test başarısız. Proje sorunlu olabilir.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
