"""
🎯 Naive Bayes Classifier
Baseline model olarak Multinomial Naive Bayes implementasyonu
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

class NaiveBayesClassifier:
    def __init__(self, model_type='multinomial'):
        """
        Naive Bayes sınıflandırıcı
        
        Args:
            model_type: 'multinomial' veya 'gaussian'
        """
        self.model_type = model_type
        self.model = None
        self.pipeline = None
        self.is_trained = False
        self.feature_names = None
        self.classes = None
        
        # Model seçimi
        if model_type == 'multinomial':
            self.model = MultinomialNB()
        elif model_type == 'gaussian':
            self.model = GaussianNB()
        else:
            raise ValueError("model_type 'multinomial' veya 'gaussian' olmalı")
    
    def train(self, X_train, y_train, feature_names=None):
        """Modeli eğit"""
        print(f"🎯 Naive Bayes ({self.model_type}) eğitimi başlıyor...")
        
        start_time = time.time()
        
        # Veri tipini kontrol et
        if hasattr(X_train, 'toarray'):  # Sparse matrix ise
            if self.model_type == 'gaussian':
                X_train = X_train.toarray()
        
        # Modeli eğit
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Model bilgilerini kaydet
        self.is_trained = True
        self.feature_names = feature_names
        self.classes = self.model.classes_
        
        print(f"✅ Eğitim tamamlandı! Süre: {training_time:.2f}s")
        print(f"   Sınıf sayısı: {len(self.classes)}")
        print(f"   Özellik sayısı: {X_train.shape[1]}")
        
        return training_time
    
    def fit(self, X, y):
        """Sklearn uyumluluğu için fit metodu"""
        return self.train(X, y)
    
    def predict(self, X_test):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Veri tipini kontrol et
        if hasattr(X_test, 'toarray') and self.model_type == 'gaussian':
            X_test = X_test.toarray()
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Olasılık tahminleri"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Veri tipini kontrol et
        if hasattr(X_test, 'toarray') and self.model_type == 'gaussian':
            X_test = X_test.toarray()
        
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, top_n=10):
        """
        Özellik önemini hesapla (sadece MultinomialNB için)
        Log-probability değerlerini kullanır
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        if self.model_type != 'multinomial':
            print("⚠️ Özellik önemi sadece MultinomialNB için hesaplanabilir")
            return None
        
        # Feature log-probabilities
        feature_log_prob = self.model.feature_log_prob_
        
        importance_data = []
        
        for i, class_name in enumerate(self.classes):
            # Bu sınıf için en önemli özellikler
            class_importance = feature_log_prob[i]
            top_indices = np.argsort(class_importance)[::-1][:top_n]
            
            for rank, idx in enumerate(top_indices):
                importance_data.append({
                    'class': class_name,
                    'feature_index': idx,
                    'feature_name': self.feature_names[idx] if self.feature_names is not None else f"feature_{idx}",
                    'log_probability': class_importance[idx],
                    'rank': rank + 1
                })
        
        return pd.DataFrame(importance_data)
    
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        """Hiperparametre optimizasyonu"""
        print("🔧 Hiperparametre optimizasyonu başlıyor...")
        
        if self.model_type == 'multinomial':
            param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                'fit_prior': [True, False]
            }
        else:  # gaussian
            param_grid = {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        
        # Grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Veri tipini kontrol et
        if hasattr(X_train, 'toarray') and self.model_type == 'gaussian':
            X_train = X_train.toarray()
        
        grid_search.fit(X_train, y_train)
        
        # En iyi modeli kullan
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        print(f"✅ En iyi parametreler: {grid_search.best_params_}")
        print(f"✅ En iyi CV skoru: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        try:
            if not self.is_trained:
                raise ValueError("Model henüz eğitilmemiş!")
            
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'classes': self.classes,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            print(f"💾 Model kaydedildi: {filepath}")
        except Exception as e:
            print(f"❌ Model kaydetme hatası: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_model(self, filepath):
        """Modeli yükle"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.classes = model_data['classes']
        self.is_trained = model_data['is_trained']
        
        print(f"📂 Model yüklendi: {filepath}")
    
    def get_model_info(self):
        """Model bilgilerini getir"""
        if not self.is_trained:
            return "Model henüz eğitilmemiş"
        
        info = {
            'model_type': self.model_type,
            'n_classes': len(self.classes),
            'classes': list(self.classes),
            'n_features': len(self.feature_names) if self.feature_names else "Bilinmiyor"
        }
        
        if self.model_type == 'multinomial':
            info['alpha'] = self.model.alpha
            info['fit_prior'] = self.model.fit_prior
        else:
            info['var_smoothing'] = self.model.var_smoothing
        
        return info

def train_naive_bayes_pipeline(X_train, y_train, X_test, y_test, 
                              feature_names=None, model_type='multinomial'):
    """Naive Bayes eğitim pipeline'ı"""
    print(f"🚀 Naive Bayes Pipeline Başlıyor ({model_type})")
    print("="*50)
    
    # Model oluştur
    nb_classifier = NaiveBayesClassifier(model_type=model_type)
    
    # Eğit
    training_time = nb_classifier.train(X_train, y_train, feature_names)
    
    # Test et
    print("\n🧪 Test seti üzerinde değerlendirme...")
    start_time = time.time()
    y_pred = nb_classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Test Sonuçları:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Tahmin süresi: {prediction_time:.4f}s")
    
    # Detaylı rapor
    print(f"\n📋 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=nb_classifier.classes))
    
    # Özellik önemi (sadece multinomial için)
    if model_type == 'multinomial' and feature_names is not None:
        print(f"\n🏆 En Önemli Özellikler:")
        importance_df = nb_classifier.get_feature_importance(top_n=5)
        if importance_df is not None:
            for class_name in nb_classifier.classes[:3]:  # İlk 3 sınıf
                class_features = importance_df[importance_df['class'] == class_name]
                print(f"\n   📂 {class_name}:")
                for _, row in class_features.head(3).iterrows():
                    print(f"      {row['rank']}. {row['feature_name']} ({row['log_probability']:.4f})")
    
    return nb_classifier, {
        'accuracy': accuracy,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred
    }

def demo_naive_bayes():
    """Naive Bayes demo'su"""
    print("🧪 Naive Bayes Demo")
    print("="*30)
    
    # Örnek veri oluştur
    from sklearn.datasets import make_classification
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sentetik veri
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_classes=3,
        n_informative=50,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Multinomial NB test (non-negative features gerekli)
    X_train_pos = np.abs(X_train)
    X_test_pos = np.abs(X_test)
    
    print("1. Multinomial Naive Bayes:")
    nb_multi, results_multi = train_naive_bayes_pipeline(
        X_train_pos, y_train, X_test_pos, y_test, 
        model_type='multinomial'
    )
    
    print("\n" + "="*50)
    print("2. Gaussian Naive Bayes:")
    nb_gauss, results_gauss = train_naive_bayes_pipeline(
        X_train, y_train, X_test, y_test,
        model_type='gaussian'
    )
    
    # Karşılaştırma
    print(f"\n📊 Karşılaştırma:")
    print(f"Multinomial NB - Accuracy: {results_multi['accuracy']:.4f}, Süre: {results_multi['training_time']:.3f}s")
    print(f"Gaussian NB   - Accuracy: {results_gauss['accuracy']:.4f}, Süre: {results_gauss['training_time']:.3f}s")

if __name__ == "__main__":
    demo_naive_bayes()
