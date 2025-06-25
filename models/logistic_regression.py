"""
🎯 Logistic Regression Classifier
Linear sınıflandırma modeli
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import matplotlib.pyplot as plt

class LogisticRegressionClassifier:
    def __init__(self, multi_class='ovr', solver='liblinear', max_iter=1000):
        """
        Logistic Regression sınıflandırıcı
        
        Args:
            multi_class: 'ovr' (one-vs-rest) veya 'multinomial'
            solver: 'liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'
            max_iter: Maksimum iterasyon sayısı
        """
        self.model = LogisticRegression(
            multi_class=multi_class,
            solver=solver,
            max_iter=max_iter,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None
        self.classes = None
        self.training_history = {}
    
    def train(self, X_train, y_train, feature_names=None):
        """Modeli eğit"""
        print("🎯 Logistic Regression eğitimi başlıyor...")
        
        start_time = time.time()
        
        # Sparse matrix'i dense'e çevir (gerekirse)
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Modeli eğit
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Model bilgilerini kaydet
        self.is_trained = True
        self.feature_names = feature_names
        self.classes = self.model.classes_
        self.training_history['training_time'] = training_time
        
        print(f"✅ Eğitim tamamlandı! Süre: {training_time:.2f}s")
        print(f"   Sınıf sayısı: {len(self.classes)}")
        print(f"   Özellik sayısı: {X_train.shape[1]}")
        print(f"   Regularizasyon (C): {self.model.C}")
        
        return training_time
    
    def predict(self, X_test):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Sparse matrix'i dense'e çevir (gerekirse)
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Olasılık tahminleri"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Sparse matrix'i dense'e çevir (gerekirse)
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        return self.model.predict_proba(X_test)
    
    def get_feature_importance(self, top_n=10):
        """
        Özellik önemini hesapla (katsayı büyüklüklerine göre)
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Katsayıları al
        coef = self.model.coef_
        
        importance_data = []
        
        # Çok sınıflı durumda her sınıf için katsayılar
        if len(self.classes) > 2:
            for i, class_name in enumerate(self.classes):
                class_coef = coef[i]
                
                # Mutlak değere göre sırala
                abs_coef = np.abs(class_coef)
                top_indices = np.argsort(abs_coef)[::-1][:top_n]
                
                for rank, idx in enumerate(top_indices):
                    importance_data.append({
                        'class': class_name,
                        'feature_index': idx,
                        'feature_name': self.feature_names[idx] if self.feature_names is not None else f"feature_{idx}",
                        'coefficient': class_coef[idx],
                        'abs_coefficient': abs_coef[idx],
                        'rank': rank + 1
                    })
        else:
            # İkili sınıflandırma
            class_coef = coef[0]
            abs_coef = np.abs(class_coef)
            top_indices = np.argsort(abs_coef)[::-1][:top_n]
            
            for rank, idx in enumerate(top_indices):
                importance_data.append({
                    'class': f'{self.classes[1]} vs {self.classes[0]}',
                    'feature_index': idx,
                    'feature_name': self.feature_names[idx] if self.feature_names is not None else f"feature_{idx}",
                    'coefficient': class_coef[idx],
                    'abs_coefficient': abs_coef[idx],
                    'rank': rank + 1
                })
        
        return pd.DataFrame(importance_data)
    
    def plot_feature_importance(self, top_n=15, figsize=(12, 8)):
        """Özellik önemini görselleştir"""
        importance_df = self.get_feature_importance(top_n)
        
        if len(self.classes) <= 3:
            fig, axes = plt.subplots(1, len(self.classes), figsize=figsize)
            if len(self.classes) == 1:
                axes = [axes]
            
            for i, class_name in enumerate(self.classes):
                if len(self.classes) > 2:
                    class_data = importance_df[importance_df['class'] == class_name]
                else:
                    class_data = importance_df
                
                ax = axes[i] if len(self.classes) > 1 else axes[0]
                
                # Pozitif ve negatif katsayıları farklı renkle
                colors = ['red' if x < 0 else 'blue' for x in class_data['coefficient']]
                
                bars = ax.barh(range(len(class_data)), class_data['coefficient'], color=colors, alpha=0.7)
                ax.set_yticks(range(len(class_data)))
                ax.set_yticklabels(class_data['feature_name'], fontsize=8)
                ax.set_title(f'Top {top_n} Features - {class_name}')
                ax.set_xlabel('Coefficient Value')
                
                # Sıfır çizgisi
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        """Hiperparametre optimizasyonu"""
        print("🔧 Hiperparametre optimizasyonu başlıyor...")
        
        # Sparse matrix'i dense'e çevir
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        
        # Parametre grid'i
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'max_iter': [1000, 2000, 3000]
        }
        
        # Penalty-solver uyumluluğu
        # l1 penalty sadece liblinear ve saga ile
        # elasticnet sadece saga ile
        # lbfgs sadece l2 ile
        
        # Uyumlu kombinasyonları oluştur
        compatible_params = []
        for C in param_grid['C']:
            for max_iter in param_grid['max_iter']:
                # l2 + her solver
                compatible_params.append({'C': C, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': max_iter})
                compatible_params.append({'C': C, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': max_iter})
                compatible_params.append({'C': C, 'penalty': 'l2', 'solver': 'saga', 'max_iter': max_iter})
                
                # l1 + liblinear, saga
                compatible_params.append({'C': C, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': max_iter})
                compatible_params.append({'C': C, 'penalty': 'l1', 'solver': 'saga', 'max_iter': max_iter})
        
        # Grid search
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42),
            compatible_params,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # En iyi modeli kullan
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        print(f"✅ En iyi parametreler: {grid_search.best_params_}")
        print(f"✅ En iyi CV skoru: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def get_decision_boundary_info(self):
        """Karar sınırı bilgilerini getir"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        info = {
            'intercept': self.model.intercept_,
            'n_iter': getattr(self.model, 'n_iter_', None),
            'classes': self.classes,
            'coef_shape': self.model.coef_.shape
        }
        
        return info
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        try:
            if not self.is_trained:
                raise ValueError("Model henüz eğitilmemiş!")
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'classes': self.classes,
                'is_trained': self.is_trained,
                'training_history': self.training_history
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
        self.feature_names = model_data['feature_names']
        self.classes = model_data['classes']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        
        print(f"📂 Model yüklendi: {filepath}")
    
    def get_model_info(self):
        """Model bilgilerini getir"""
        if not self.is_trained:
            return "Model henüz eğitilmemiş"
        
        info = {
            'solver': self.model.solver,
            'penalty': self.model.penalty,
            'C': self.model.C,
            'multi_class': self.model.multi_class,
            'max_iter': self.model.max_iter,
            'n_classes': len(self.classes),
            'classes': list(self.classes),
            'n_features': self.model.coef_.shape[1] if hasattr(self.model, 'coef_') else "Bilinmiyor",
            'converged': getattr(self.model, 'n_iter_', None) is not None
        }
        
        return info

def train_logistic_regression_pipeline(X_train, y_train, X_test, y_test, 
                                     feature_names=None, tune_hyperparams=False):
    """Logistic Regression eğitim pipeline'ı"""
    print("🚀 Logistic Regression Pipeline Başlıyor")
    print("="*50)
    
    # Model oluştur
    lr_classifier = LogisticRegressionClassifier()
    
    # Hiperparametre optimizasyonu (isteğe bağlı)
    if tune_hyperparams:
        print("🔧 Hiperparametre optimizasyonu yapılıyor...")
        best_params, best_score = lr_classifier.hyperparameter_tuning(X_train, y_train)
    else:
        # Varsayılan parametrelerle eğit
        training_time = lr_classifier.train(X_train, y_train, feature_names)
    
    # Test et
    print("\n🧪 Test seti üzerinde değerlendirme...")
    start_time = time.time()
    y_pred = lr_classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Test Sonuçları:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Tahmin süresi: {prediction_time:.4f}s")
    
    # Detaylı rapor
    print(f"\n📋 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=lr_classifier.classes))
    
    # Özellik önemi
    if feature_names is not None:
        print(f"\n🏆 En Önemli Özellikler:")
        importance_df = lr_classifier.get_feature_importance(top_n=5)
        
        for class_name in lr_classifier.classes[:3]:  # İlk 3 sınıf
            if len(lr_classifier.classes) > 2:
                class_features = importance_df[importance_df['class'] == class_name]
            else:
                class_features = importance_df
            
            print(f"\n   📂 {class_name}:")
            for _, row in class_features.head(3).iterrows():
                direction = "+" if row['coefficient'] > 0 else "-"
                print(f"      {row['rank']}. {row['feature_name']} ({direction}{abs(row['coefficient']):.4f})")
    
    # Model bilgileri
    print(f"\n🔍 Model Bilgileri:")
    model_info = lr_classifier.get_model_info()
    for key, value in model_info.items():
        if key not in ['classes']:  # Sınıfları ayrı gösterelim
            print(f"   {key}: {value}")
    
    return lr_classifier, {
        'accuracy': accuracy,
        'training_time': training_time if not tune_hyperparams else None,
        'prediction_time': prediction_time,
        'y_pred': y_pred
    }

def demo_logistic_regression():
    """Logistic Regression demo'su"""
    print("🧪 Logistic Regression Demo")
    print("="*35)
    
    # Örnek veri oluştur
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Sentetik veri
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Pipeline'ı çalıştır
    lr_model, results = train_logistic_regression_pipeline(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        tune_hyperparams=False
    )
    
    # Özellik önemini görselleştir
    print("\n📊 Özellik önemini görselleştirme...")
    lr_model.plot_feature_importance(top_n=10)

if __name__ == "__main__":
    demo_logistic_regression()
