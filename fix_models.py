"""
🔧 Model Fixer - Modelleri doğru şekilde eğitip kaydet
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.text_preprocessing import TurkishTextPreprocessor

def create_sample_data():
    """Örnek veri oluştur"""
    data = [
        ("Kredi kartım çalışmıyor, ödeme yapamıyorum", "payment_issue"),
        ("Fatura ödemem gerçekleşmedi", "payment_issue"),
        ("Kartımdan para çekildi ama ödeme yapılmadı", "payment_issue"),
        ("Banka hesabımdan para çekildi", "payment_issue"),
        ("Ödeme yapmaya çalışıyorum ama hata veriyor", "payment_issue"),
        
        ("Rezervasyonumu iptal etmek istiyorum", "reservation_problem"),
        ("Rezervasyon tarihimi değiştirmek istiyorum", "reservation_problem"),
        ("Otel rezervasyonumda sorun var", "reservation_problem"),
        ("Rezervasyon yapmaya çalışıyorum ama olmuyor", "reservation_problem"),
        ("Rezervasyonum gözükmüyor", "reservation_problem"),
        
        ("Şifremi unuttum, giriş yapamıyorum", "user_error"),
        ("Kullanıcı adımı hatırlamıyorum", "user_error"),
        ("Hesabıma erişemiyorum", "user_error"),
        ("Giriş yapmaya çalışıyorum ama olmuyor", "user_error"),
        ("Şifremi değiştirmek istiyorum", "user_error"),
        
        ("Hizmetinizden memnun değilim", "complaint"),
        ("Çok kötü bir deneyim yaşadım", "complaint"),
        ("Personel çok kaba davrandı", "complaint"),
        ("Şikayetim var, çözüm istiyorum", "complaint"),
        ("Bu kadar kötü olmasını beklemiyordum", "complaint"),
        
        ("Nasıl rezervasyon yapabilirim?", "general_info"),
        ("Hangi ödeme yöntemlerini kabul ediyorsunuz?", "general_info"),
        ("Çalışma saatleriniz nedir?", "general_info"),
        ("Daha fazla bilgi alabilir miyim?", "general_info"),
        ("Hizmetleriniz hakkında bilgi istiyorum", "general_info"),
        
        ("Uygulama açılmıyor", "technical_issue"),
        ("Website yavaş çalışıyor", "technical_issue"),
        ("Sayfa yüklenmiyor", "technical_issue"),
        ("Hata mesajı alıyorum", "technical_issue"),
        ("Sistem çalışmıyor", "technical_issue"),
    ]
    
    # Veriyi çoğaltalım
    extended_data = []
    for text, category in data:
        extended_data.append((text, category))
        # Her kategori için varyasyonlar ekle
        if "payment" in category:
            extended_data.append((text.replace("kartım", "kartimiz"), category))
            extended_data.append((text.replace("ödeme", "ödeme işlemi"), category))
        elif "reservation" in category:
            extended_data.append((text.replace("rezervasyon", "rezervasyon işlemi"), category))
            extended_data.append((text.replace("iptal", "iptal etme"), category))
        elif "user" in category:
            extended_data.append((text.replace("şifre", "parola"), category))
            extended_data.append((text.replace("giriş", "login"), category))
        elif "complaint" in category:
            extended_data.append((text.replace("memnun", "mutlu"), category))
            extended_data.append((text.replace("kötü", "berbat"), category))
        elif "general" in category:
            extended_data.append((text.replace("nasıl", "ne şekilde"), category))
            extended_data.append((text.replace("bilgi", "detay"), category))
        elif "technical" in category:
            extended_data.append((text.replace("açılmıyor", "çalışmıyor"), category))
            extended_data.append((text.replace("yavaş", "çok yavaş"), category))
    
    return extended_data

def train_and_save_models():
    """Modelleri eğit ve kaydet"""
    print("🚀 Modeller eğitiliyor ve kaydediliyor...")
    
    # Veri hazırlığı
    data = create_sample_data()
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    print(f"📊 Toplam veri sayısı: {len(texts)}")
    print(f"📊 Kategoriler: {set(labels)}")
    
    # Text preprocessing
    preprocessor = TurkishTextPreprocessor()
    processed_texts = []
    for text in texts:
        processed = preprocessor.preprocess_text(text, remove_stopwords=True, apply_stemming=False)
        processed_texts.append(processed)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # TF-IDF Vectorization
    print("🔤 TF-IDF vectorizer eğitiliyor...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Label Encoding
    print("🏷️ Label encoder eğitiliyor...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Model dizini oluştur
    model_dir = "models/trained"
    os.makedirs(model_dir, exist_ok=True)
    
    # TF-IDF Vectorizer'ı kaydet
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    joblib.dump(tfidf_vectorizer, tfidf_path)
    print(f"✅ TF-IDF vectorizer kaydedildi: {tfidf_path}")
    
    # Label Encoder'ı kaydet
    encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"✅ Label encoder kaydedildi: {encoder_path}")
    
    # 1. Naive Bayes Model
    print("🤖 Naive Bayes eğitiliyor...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train_encoded)
    
    # Test
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test_encoded, nb_pred)
    print(f"📊 Naive Bayes Accuracy: {nb_accuracy:.4f}")
    
    # Kaydet
    nb_path = os.path.join(model_dir, "naive_bayes_model.pkl")
    with open(nb_path, 'wb') as f:
        pickle.dump(nb_model, f)
    print(f"✅ Naive Bayes model kaydedildi: {nb_path}")
    
    # 2. Logistic Regression Model
    print("🤖 Logistic Regression eğitiliyor...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train_encoded)
    
    # Test
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_accuracy = accuracy_score(y_test_encoded, lr_pred)
    print(f"📊 Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # Kaydet
    lr_path = os.path.join(model_dir, "logistic_regression_model.pkl")
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"✅ Logistic Regression model kaydedildi: {lr_path}")
    
    # Classification Report
    print("\n📋 Naive Bayes Classification Report:")
    print(classification_report(y_test_encoded, nb_pred, target_names=label_encoder.classes_))
    
    print("\n📋 Logistic Regression Classification Report:")
    print(classification_report(y_test_encoded, lr_pred, target_names=label_encoder.classes_))
    
    print("\n✅ Tüm modeller başarıyla eğitildi ve kaydedildi!")
    return True

if __name__ == "__main__":
    train_and_save_models()
