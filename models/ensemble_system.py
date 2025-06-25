"""
🤖 Model Ensemble System
Birden fazla modeli birleştirerek daha iyi performans elde eden sistem
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
import joblib
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .naive_bayes import NaiveBayesClassifier
from .logistic_regression import LogisticRegressionClassifier

class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Ağırlıklı ensemble classifier"""
    
    def __init__(self, models=None, weights=None, voting='soft'):
        self.models = models or {}
        self.weights = weights or {}
        self.voting = voting
        self.classes_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y):
        """Ensemble modelini eğit"""
        print("🔧 Ensemble modeli eğitiliyor...")
        
        # Her modeli eğit
        for name, model in self.models.items():
            print(f"  📚 {name} eğitiliyor...")
            model.fit(X, y)
        
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        
        # Eğer ağırlıklar verilmemişse, cross-validation ile otomatik hesapla
        if not self.weights:
            self.weights = self._calculate_weights(X, y)
            
        print("✅ Ensemble eğitimi tamamlandı!")
        return self
    
    def _calculate_weights(self, X, y, cv=5):
        """Cross-validation ile otomatik ağırlık hesapla"""
        print("  ⚖️ Otomatik ağırlık hesaplanıyor...")
        
        weights = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation score
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                avg_score = scores.mean()
                weights[name] = avg_score
                print(f"    {name}: {avg_score:.4f}")
            except Exception as e:
                print(f"    ⚠️ {name} için ağırlık hesaplanamadı: {e}")
                weights[name] = 0.1  # Minimal ağırlık
        
        # Ağırlıkları normalize et
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        print(f"  ✅ Hesaplanan ağırlıklar: {weights}")
        return weights
    
    def predict(self, X):
        """Ensemble tahmin yap"""
        if not self.is_fitted_:
            raise ValueError("Model henüz eğitilmemiş!")
        
        if self.voting == 'hard':
            return self._predict_hard_voting(X)
        else:
            return self._predict_soft_voting(X)
    
    def _predict_hard_voting(self, X):
        """Hard voting ile tahmin"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Çoğunluk oylaması
        predictions = np.array(predictions).T
        final_predictions = []
        
        for sample_preds in predictions:
            unique, counts = np.unique(sample_preds, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            final_predictions.append(majority_class)
        
        return np.array(final_predictions)
    
    def _predict_soft_voting(self, X):
        """Soft voting ile tahmin (ağırlıklı olasılıklar)"""
        weighted_probs = None
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                elif hasattr(model, 'decision_function'):
                    # Decision function'ı probability'ye çevir
                    decision = model.decision_function(X)
                    from sklearn.utils.extmath import softmax
                    probs = softmax(decision.reshape(-1, 1), axis=1)
                else:
                    # Fallback: hard prediction'ı probability'ye çevir
                    pred = model.predict(X)
                    probs = np.zeros((len(X), len(self.classes_)))
                    for i, class_label in enumerate(self.classes_):
                        probs[:, i] = (pred == class_label).astype(float)
                
                weight = self.weights.get(name, 1.0)
                
                if weighted_probs is None:
                    weighted_probs = probs * weight
                else:
                    weighted_probs += probs * weight
                    
            except Exception as e:
                print(f"⚠️ {name} modeli için soft voting hatası: {e}")
                continue
        
        if weighted_probs is None:
            raise ValueError("Hiçbir model için soft voting yapılamadı!")
        
        # En yüksek olasılığa sahip sınıfı seç
        final_predictions = self.classes_[np.argmax(weighted_probs, axis=1)]
        return final_predictions
    
    def predict_proba(self, X):
        """Sınıf olasılıklarını döndür"""
        if not self.is_fitted_:
            raise ValueError("Model henüz eğitilmemiş!")
        
        weighted_probs = None
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                else:
                    # Fallback
                    pred = model.predict(X)
                    probs = np.zeros((len(X), len(self.classes_)))
                    for i, class_label in enumerate(self.classes_):
                        probs[:, i] = (pred == class_label).astype(float)
                
                weight = self.weights.get(name, 1.0)
                
                if weighted_probs is None:
                    weighted_probs = probs * weight
                else:
                    weighted_probs += probs * weight
                    
            except Exception as e:
                print(f"⚠️ {name} modeli için probability hesaplanamadı: {e}")
                continue
        
        if weighted_probs is None:
            raise ValueError("Hiçbir model için probability hesaplanamadı!")
        
        # Normalize et
        row_sums = weighted_probs.sum(axis=1, keepdims=True)
        normalized_probs = weighted_probs / row_sums
        
        return normalized_probs

class EnsembleManager:
    """Ensemble yönetim sistemi"""
    
    def __init__(self, models_dir="models/ensemble"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_models = {
            'naive_bayes': NaiveBayesClassifier(),
            'logistic_regression': LogisticRegressionClassifier()
        }
        
        self.ensembles = {}
        
    def create_voting_ensemble(self, X_train, y_train, voting='soft', cv_folds=5):
        """Voting ensemble oluştur"""
        print("🗳️ Voting Ensemble oluşturuluyor...")
        
        # Sklearn VotingClassifier kullan
        estimators = []
        
        for name, model in self.base_models.items():
            # Model wrapper'ı sklearn uyumlu hale getir
            if hasattr(model, 'model'):
                estimators.append((name, model.model))
            else:
                estimators.append((name, model))
        
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        # Eğit
        voting_ensemble.fit(X_train, y_train)
        
        self.ensembles['voting'] = voting_ensemble
        print("✅ Voting Ensemble oluşturuldu!")
        
        return voting_ensemble
    
    def create_weighted_ensemble(self, X_train, y_train, custom_weights=None):
        """Ağırlıklı ensemble oluştur"""
        print("⚖️ Weighted Ensemble oluşturuluyor...")
        
        # Custom ensemble class kullan
        weighted_ensemble = WeightedEnsemble(
            models=self.base_models.copy(),
            weights=custom_weights,
            voting='soft'
        )
        
        weighted_ensemble.fit(X_train, y_train)
        
        self.ensembles['weighted'] = weighted_ensemble
        print("✅ Weighted Ensemble oluşturuldu!")
        
        return weighted_ensemble
    
    def create_stacking_ensemble(self, X_train, y_train):
        """Stacking ensemble oluştur"""
        print("🏗️ Stacking Ensemble oluşturuluyor...")
        
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression as SklearnLR
        
        # Base models
        estimators = []
        for name, model in self.base_models.items():
            if hasattr(model, 'model'):
                estimators.append((name, model.model))
            else:
                estimators.append((name, model))
        
        # Meta-learner
        meta_learner = SklearnLR(random_state=42)
        
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5
        )
        
        stacking_ensemble.fit(X_train, y_train)
        
        self.ensembles['stacking'] = stacking_ensemble
        print("✅ Stacking Ensemble oluşturuldu!")
        
        return stacking_ensemble
    
    def compare_ensembles(self, X_test, y_test):
        """Ensemble modellerini karşılaştır"""
        print("\n📊 Ensemble Modelleri Karşılaştırılıyor...")
        print("=" * 50)
        
        results = {}
        
        # Base modelleri de dahil et
        all_models = {**self.base_models, **self.ensembles}
        
        for name, model in all_models.items():
            try:
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': predictions,
                    'classification_report': classification_report(y_test, predictions, output_dict=True)
                }
                
                print(f"{name:20} | Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"❌ {name} modeli hatası: {e}")
                results[name] = {'error': str(e)}
        
        # En iyi modeli bul
        best_model = max(
            [(name, res) for name, res in results.items() if 'accuracy' in res],
            key=lambda x: x[1]['accuracy']
        )
        
        print(f"\n🏆 En İyi Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        return results
    
    def save_ensemble(self, ensemble_name, model):
        """Ensemble modelini kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = self.models_dir / f"{ensemble_name}_{timestamp}.joblib"
        
        try:
            joblib.dump(model, model_file)
            print(f"💾 {ensemble_name} ensemble kaydedildi: {model_file}")
            
            # Metadata kaydet
            metadata = {
                'ensemble_name': ensemble_name,
                'timestamp': timestamp,
                'model_file': str(model_file),
                'model_type': type(model).__name__
            }
            
            if hasattr(model, 'weights'):
                metadata['weights'] = model.weights
            
            metadata_file = self.models_dir / f"{ensemble_name}_{timestamp}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"❌ Ensemble kaydetme hatası: {e}")
    
    def load_ensemble(self, model_file):
        """Ensemble modelini yükle"""
        try:
            model = joblib.load(model_file)
            print(f"📁 Ensemble yüklendi: {model_file}")
            return model
        except Exception as e:
            print(f"❌ Ensemble yükleme hatası: {e}")
            return None
    
    def generate_ensemble_report(self, results, output_file=None):
        """Ensemble karşılaştırma raporu oluştur"""
        print("\n📋 Ensemble Raporu Oluşturuluyor...")
        
        report_lines = []
        report_lines.append("🤖 ENSEMBLE MODEL KARŞILAŞTIRMA RAPORU")
        report_lines.append("=" * 60)
        report_lines.append(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Accuracy karşılaştırması
        accuracy_data = []
        for name, result in results.items():
            if 'accuracy' in result:
                accuracy_data.append((name, result['accuracy']))
        
        if accuracy_data:
            accuracy_data.sort(key=lambda x: x[1], reverse=True)
            
            report_lines.append("📊 ACCURACY SONUÇLARI:")
            report_lines.append("-" * 30)
            
            for i, (name, accuracy) in enumerate(accuracy_data, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                report_lines.append(f"{emoji} {i:2d}. {name:20} | {accuracy:.4f}")
            
            report_lines.append("")
        
        # Öneriler
        report_lines.append("💡 ENSEMBLE ÖNERİLERİ:")
        report_lines.append("-" * 25)
        
        if accuracy_data:
            best_model = accuracy_data[0]
            report_lines.append(f"🏆 En iyi performans: {best_model[0]} ({best_model[1]:.4f})")
            
            # Ensemble türü önerileri
            if 'voting' in [name for name, _ in accuracy_data]:
                report_lines.append("🗳️ Voting Ensemble: Basit ve etkili")
            if 'weighted' in [name for name, _ in accuracy_data]:
                report_lines.append("⚖️ Weighted Ensemble: Model güçlerine göre ağırlıklandırılmış")
            if 'stacking' in [name for name, _ in accuracy_data]:
                report_lines.append("🏗️ Stacking Ensemble: Meta-learner ile gelişmiş kombinasyon")
        
        report_lines.append("")
        report_lines.append("🎯 KULLANIM ÖNERİLERİ:")
        report_lines.append("• Hızlı inference → Voting Ensemble")
        report_lines.append("• Maksimum doğruluk → Stacking Ensemble")
        report_lines.append("• Dengelenmiş performans → Weighted Ensemble")
        
        # Raporu kaydet
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.models_dir / f"ensemble_report_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
        print(f"\n📝 Ensemble raporu kaydedildi: {output_file}")
        
        return output_file

def demo_ensemble_system():
    """Ensemble sistemi demo'su"""
    print("🧪 ENSEMBLE SİSTEMİ DEMO'SU")
    print("=" * 40)
    
    # Demo data
    from data_generator import TicketDataGenerator
    from utils.text_preprocessing import TurkishTextPreprocessor
    from utils.feature_extraction import FeatureExtractor
    from sklearn.model_selection import train_test_split
    
    # Veri oluştur
    generator = TicketDataGenerator()
    tickets = generator.generate_tickets(num_tickets=1000)
    df = pd.DataFrame(tickets)
    
    # 'message' sütununu 'description' olarak kullan
    df['description'] = df['message']
    
    # Preprocess
    preprocessor = TurkishTextPreprocessor()
    df['processed_text'] = df['description'].apply(preprocessor.preprocess_text)
    
    # Feature extraction
    extractor = FeatureExtractor()
    X, _ = extractor.extract_tfidf_features(df['processed_text'], max_features=500)
    y = df['category']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Ensemble manager
    ensemble_manager = EnsembleManager()
    
    # Çeşitli ensemble'lar oluştur
    voting_ensemble = ensemble_manager.create_voting_ensemble(X_train, y_train)
    weighted_ensemble = ensemble_manager.create_weighted_ensemble(X_train, y_train)
    stacking_ensemble = ensemble_manager.create_stacking_ensemble(X_train, y_train)
    
    # Karşılaştır
    results = ensemble_manager.compare_ensembles(X_test, y_test)
    
    # Rapor oluştur
    ensemble_manager.generate_ensemble_report(results)
    
    # En iyi modeli kaydet
    best_ensemble_name = max(
        [(name, res) for name, res in results.items() if 'accuracy' in res],
        key=lambda x: x[1]['accuracy']
    )[0]
    
    if best_ensemble_name in ensemble_manager.ensembles:
        best_model = ensemble_manager.ensembles[best_ensemble_name]
        ensemble_manager.save_ensemble(f"best_{best_ensemble_name}", best_model)
    
    return results

if __name__ == "__main__":
    demo_ensemble_system()
