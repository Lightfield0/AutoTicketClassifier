#!/usr/bin/env python3
"""
🚀 Quick Model Training Script
Bu script modelleri hızla eğitir ve kaydeder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

# Modüllerimizi import et
sys.path.append('.')
from data_generator import TicketDataGenerator
from utils.text_preprocessing import TurkishTextPreprocessor

def quick_train():
    """Hızlı model eğitimi ve kaydetme"""
    print("🎯 Quick Model Training başlatılıyor...")
    
    # 1. Veri üretimi
    print("\n📊 Veri üretiliyor...")
    generator = TicketDataGenerator()
    tickets = generator.generate_tickets(num_tickets=1800)  # 300 per category x 6 categories
    
    # DataFrame'e çevir
    df = pd.DataFrame(tickets)
    # Kolonları düzelt
    df = df.rename(columns={'message': 'text'})
    print(f"✅ {len(df)} sample üretildi")
    
    # 2. Metin ön işleme (basit)
    print("\n🔤 Metin ön işleme...")
    
    def simple_preprocess(text):
        """Basit metin ön işleme"""
        import re
        text = str(text).lower()
        # Türkçe karakterleri koru
        text = re.sub(r'[^\w\sçğıöşü]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['processed_text'] = df['text'].apply(simple_preprocess)
    
    # 3. Label encoding
    print("\n🏷️ Label encoding...")
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['category'])
    
    # 4. Train-test split
    print("\n✂️ Veri bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label_encoded'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_encoded']
    )
    
    # 5. TF-IDF vectorization
    print("\n📈 TF-IDF features çıkarılıyor...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 6. Model eğitimi
    print("\n🤖 Modeller eğitiliyor...")
    
    # Naive Bayes
    print("   📚 Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_accuracy = nb_model.score(X_test_tfidf, y_test)
    print(f"   ✅ NB Accuracy: {nb_accuracy:.3f}")
    
    # Logistic Regression
    print("   📈 Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_accuracy = lr_model.score(X_test_tfidf, y_test)
    print(f"   ✅ LR Accuracy: {lr_accuracy:.3f}")
    
    # 7. Model kaydetme
    print("\n💾 Modeller kaydediliyor...")
    
    # Klasör oluştur
    os.makedirs("models/trained", exist_ok=True)
    
    # TF-IDF vectorizer
    joblib.dump(tfidf, "models/trained/tfidf_vectorizer.joblib")
    print("   ✅ TF-IDF vectorizer kaydedildi")
    
    # Label encoder
    joblib.dump(le, "models/trained/label_encoder.joblib")
    print("   ✅ Label encoder kaydedildi")
    
    # Naive Bayes model
    joblib.dump(nb_model, "models/trained/naive_bayes_model.joblib")
    print("   ✅ Naive Bayes model kaydedildi")
    
    # Logistic Regression model
    joblib.dump(lr_model, "models/trained/logistic_regression_model.joblib")
    print("   ✅ Logistic Regression model kaydedildi")
    
    # Model sonuçları
    results = {
        "naive_bayes": {
            "accuracy": float(nb_accuracy),
            "model_type": "MultinomialNB"
        },
        "logistic_regression": {
            "accuracy": float(lr_accuracy),
            "model_type": "LogisticRegression"
        },
        "categories": le.classes_.tolist(),
        "feature_count": X_train_tfidf.shape[1],
        "training_samples": len(X_train)
    }
    
    import json
    with open("models/trained/model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("   ✅ Model sonuçları kaydedildi")
    
    print(f"\n🎉 Model eğitimi tamamlandı!")
    print(f"📁 Kaydedilen dosyalar:")
    print(f"   • models/trained/tfidf_vectorizer.joblib")
    print(f"   • models/trained/label_encoder.joblib")
    print(f"   • models/trained/naive_bayes_model.joblib")
    print(f"   • models/trained/logistic_regression_model.joblib")
    print(f"   • models/trained/model_results.json")
    
    print(f"\n🌐 Web uygulaması ve API artık gerçek modelleri kullanabilir!")

if __name__ == "__main__":
    quick_train()
