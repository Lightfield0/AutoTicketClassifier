"""
🎓 Ana Model Eğitim Scripti - Enhanced
Tüm modelleri eğitir, kapsamlı değerlendirme ve monitoring ile
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import time
import matplotlib.pyplot as plt

# Kendi modüllerimizi import et
from utils.text_preprocessing import TurkishTextPreprocessor
from utils.feature_extraction import FeatureExtractor
from utils.evaluation import ModelEvaluator
from models.naive_bayes import NaiveBayesClassifier, train_naive_bayes_pipeline
from models.logistic_regression import LogisticRegressionClassifier, train_logistic_regression_pipeline
from models.bert_classifier import BERTTextClassifier, train_bert_pipeline

# Yeni iyileştirmeleri import et
from improvements.comprehensive_model_evaluation import ModelEvaluator as AdvancedEvaluator
from improvements.monitoring_dashboard import ModelDriftDetector, PerformanceMonitor, MonitoringDashboard
from improvements.ab_testing import ABTestFramework
from improvements.ensemble_system import EnsembleManager

class EnhancedModelTrainer:
    def __init__(self, data_path="data/processed_data.csv"):
        """
        Gelişmiş Model eğitim sınıfı - Tüm iyileştirmelerle entegre
        
        Args:
            data_path: İşlenmiş veri dosyasının yolu
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        
        # Temel bileşenler
        self.preprocessor = TurkishTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()
        
        # Gelişmiş bileşenler
        self.advanced_evaluator = AdvancedEvaluator()
        self.drift_detector = None  # Eğitim sonrası initialize edilecek
        self.performance_monitor = PerformanceMonitor()
        self.ab_tester = ABTestFramework()
        self.ensemble_manager = EnsembleManager()
        
        self.results = {}
        self.models = {}
        self.comprehensive_results = {}
        
        # Model save dizini
        os.makedirs("models/trained", exist_ok=True)
        os.makedirs("evaluation_results", exist_ok=True)
        os.makedirs("monitoring", exist_ok=True)
    
    def load_data(self):
        """Veriyi yükle"""
        print("📂 Veri yükleniyor...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"✅ {len(self.df)} adet ticket yüklendi")
        print(f"📊 Kategori dağılımı:")
        print(self.df['category_tr'].value_counts())
        
        return self.df
    
    def preprocess_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Veriyi ön işle ve böl"""
        print("\n🔧 Veri ön işleme başlıyor...")
        
        # Metin ön işleme
        self.df = self.preprocessor.preprocess_dataframe(
            self.df, 
            text_column='message',
            new_column='processed_text',
            remove_stopwords=True,
            apply_stemming=False,
            min_length=2
        )
        
        # Boş metinleri filtrele
        initial_count = len(self.df)
        self.df = self.df[self.df['processed_text'].str.len() > 0]
        filtered_count = len(self.df)
        
        if initial_count != filtered_count:
            print(f"⚠️  {initial_count - filtered_count} adet boş metin filtrelendi")
        
        # Features ve labels
        X = self.df['processed_text'].values
        y = self.df['category'].values
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train-validation split
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=y_temp
            )
            self.X_val = X_val
            self.y_val = y_val
        else:
            X_train, y_train = X_temp, y_temp
            self.X_val = None
            self.y_val = None
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"✅ Veri bölünmesi tamamlandı:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Validation: {len(X_val) if X_val is not None else 0} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return True
    
    def train_naive_bayes(self, model_type='multinomial'):
        """Naive Bayes modelini eğit"""
        print(f"\n🎯 Naive Bayes ({model_type}) Eğitimi")
        print("="*50)
        
        # TF-IDF features
        tfidf_matrix_train, feature_names = self.feature_extractor.extract_tfidf_features(
            self.X_train, max_features=3000
        )
        tfidf_matrix_test = self.feature_extractor.transform_new_text(
            self.X_test, feature_type='tfidf'
        )
        
        # Model eğit
        nb_model, results = train_naive_bayes_pipeline(
            tfidf_matrix_train, self.y_train,
            tfidf_matrix_test, self.y_test,
            feature_names=feature_names,
            model_type=model_type
        )
        
        # Kaydet
        model_name = f"naive_bayes_{model_type}"
        nb_model.save_model(f"models/trained/{model_name}.pkl")
        
        self.models[model_name] = nb_model
        self.results[model_name] = results
        
        # Evaluator'a ekle - sadece sonuçlarla, model prediction yapmadan
        eval_results = self.evaluator.evaluate_model(
            nb_model, tfidf_matrix_test, self.y_test, 
            y_pred=results['y_pred'], model_name=f"Naive Bayes ({model_type})"
        )
        
        return nb_model, results
    
    def train_logistic_regression(self, tune_hyperparams=False):
        """Logistic Regression modelini eğit"""
        print(f"\n🎯 Logistic Regression Eğitimi")
        print("="*50)
        
        # TF-IDF features
        tfidf_matrix_train, feature_names = self.feature_extractor.extract_tfidf_features(
            self.X_train, max_features=5000
        )
        tfidf_matrix_test = self.feature_extractor.transform_new_text(
            self.X_test, feature_type='tfidf'
        )
        
        # Model eğit
        lr_model, results = train_logistic_regression_pipeline(
            tfidf_matrix_train, self.y_train,
            tfidf_matrix_test, self.y_test,
            feature_names=feature_names,
            tune_hyperparams=tune_hyperparams
        )
        
        # Kaydet
        model_name = "logistic_regression"
        lr_model.save_model(f"models/trained/{model_name}.pkl")
        
        self.models[model_name] = lr_model
        self.results[model_name] = results
        
        # Evaluator'a ekle - sadece sonuçlarla, model prediction yapmadan
        eval_results = self.evaluator.evaluate_model(
            lr_model, tfidf_matrix_test, self.y_test,
            y_pred=results['y_pred'], model_name="Logistic Regression"
        )
        
        return lr_model, results
    
    def train_bert(self, model_name='dbmdz/bert-base-turkish-cased', epochs=3):
        """BERT modelini eğit"""
        print(f"\n🤖 BERT Eğitimi")
        print("="*50)
        
        # BERT raw text kullanır, sadece basit temizlik
        X_train_clean = [self.preprocessor.clean_text(text) for text in self.X_train]
        X_test_clean = [self.preprocessor.clean_text(text) for text in self.X_test]
        X_val_clean = None
        if self.X_val is not None:
            X_val_clean = [self.preprocessor.clean_text(text) for text in self.X_val]
        
        # Model eğit
        bert_model, results = train_bert_pipeline(
            X_train_clean, self.y_train,
            X_test_clean, self.y_test,
            X_val_clean, self.y_val,
            model_name=model_name,
            epochs=epochs
        )
        
        # Kaydet
        model_save_name = "bert_classifier"
        bert_model.save_model(f"models/trained/{model_save_name}.pth")
        
        self.models[model_save_name] = bert_model
        self.results[model_save_name] = results
        
        # Evaluator'a ekle
        self.evaluator.evaluate_model(
            bert_model, X_test_clean, self.y_test,
            y_pred=results['y_pred'], model_name="BERT"
        )
        
        return bert_model, results
    
    def train_all_models(self, include_bert=True, bert_epochs=3):
        """Tüm modelleri eğit"""
        print("🚀 TÜM MODELLERİN EĞİTİMİ BAŞLIYOR")
        print("="*60)
        
        start_time = time.time()
        
        # 1. Naive Bayes (Multinomial)
        try:
            self.train_naive_bayes(model_type='multinomial')
        except Exception as e:
            print(f"❌ Naive Bayes (Multinomial) hatası: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Logistic Regression
        try:
            self.train_logistic_regression(tune_hyperparams=False)
        except Exception as e:
            print(f"❌ Logistic Regression hatası: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. BERT (opsiyonel - uzun sürüyor)
        if include_bert:
            try:
                self.train_bert(epochs=bert_epochs)
            except Exception as e:
                print(f"❌ BERT hatası: {e}")
        else:
            print("⏭️  BERT eğitimi atlandı (include_bert=False)")
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Tüm modellerin eğitimi tamamlandı!")
        print(f"⏱️  Toplam süre: {total_time/60:.1f} dakika")
        
        return True
    
    def compare_models(self):
        """Modelleri karşılaştır"""
        print(f"\n📊 MODEL KARŞILAŞTIRMASI")
        print("="*60)
        
        if len(self.results) == 0:
            print("❌ Karşılaştırılacak model bulunamadı")
            return
        
        # Özet tablo
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Training Time (s)': result.get('training_time', 'N/A'),
                'Prediction Time (s)': result['prediction_time']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Görselleştirme
        if len(self.evaluator.results) > 0:
            self.evaluator.compare_models(figsize=(14, 10))
            self.evaluator.plot_performance_vs_time()
        
        # Özet rapor
        self.evaluator.generate_summary_report()
        
        return df_comparison
    
    def save_results(self, filepath="models/training_results.json"):
        """Sonuçları JSON olarak kaydet"""
        results_to_save = {}
        
        for model_name, result in self.results.items():
            # Numpy arrays'i list'e çevir
            result_copy = result.copy()
            if 'y_pred' in result_copy:
                y_pred = result_copy['y_pred']
                if hasattr(y_pred, 'tolist'):
                    result_copy['y_pred'] = y_pred.tolist()
                elif isinstance(y_pred, list):
                    result_copy['y_pred'] = y_pred
            
            results_to_save[model_name] = result_copy
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Sonuçlar kaydedildi: {filepath}")
        
        # TF-IDF vectorizer ve label encoder'ı da kaydet
        self.save_preprocessing_components()
    
    def save_preprocessing_components(self):
        """TF-IDF vectorizer ve label encoder'ı kaydet"""
        import joblib
        
        # TF-IDF vectorizer kaydet
        if hasattr(self.feature_extractor, 'tfidf_vectorizer') and self.feature_extractor.tfidf_vectorizer:
            joblib.dump(self.feature_extractor.tfidf_vectorizer, "models/trained/tfidf_vectorizer.joblib")
            print("💾 TF-IDF vectorizer kaydedildi")
        
        # Label encoder kaydet - önce y_train'den oluştur
        if hasattr(self, 'y_train') and self.y_train is not None:
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(self.y_train)
            joblib.dump(label_encoder, "models/trained/label_encoder.joblib")
            print("💾 Label encoder kaydedildi")
    
    def generate_report(self, save_path="models/model_training_report.md"):
        """Markdown formatında rapor oluştur"""
        report = []
        report.append("# 🎫 AutoTicket Classifier - Model Eğitim Raporu\n")
        report.append(f"📅 Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Veri özeti
        report.append("## 📊 Veri Özeti\n")
        report.append(f"- Toplam ticket sayısı: {len(self.df)}")
        report.append(f"- Train set: {len(self.X_train)} samples")
        report.append(f"- Test set: {len(self.X_test)} samples")
        if self.X_val is not None:
            report.append(f"- Validation set: {len(self.X_val)} samples")
        
        # Kategori dağılımı
        report.append("\n### Kategori Dağılımı")
        category_counts = pd.Series(self.y_train).value_counts()
        for category, count in category_counts.items():
            report.append(f"- {category}: {count}")
        
        # Model sonuçları
        report.append("\n## 🤖 Model Sonuçları\n")
        
        for model_name, result in self.results.items():
            report.append(f"### {model_name.replace('_', ' ').title()}")
            report.append(f"- **Accuracy**: {result['accuracy']:.4f}")
            if 'training_time' in result and result['training_time']:
                report.append(f"- **Training Time**: {result['training_time']:.2f}s")
            report.append(f"- **Prediction Time**: {result['prediction_time']:.4f}s")
            report.append("")
        
        # En iyi model
        if self.results:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            report.append(f"## 🏆 En İyi Model: {best_model.replace('_', ' ').title()}")
            report.append(f"- Accuracy: {self.results[best_model]['accuracy']:.4f}")
            report.append("")
        
        # Öneriler
        report.append("## 💡 Öneriler\n")
        if self.results:
            best_accuracy = max(result['accuracy'] for result in self.results.values())
            if best_accuracy > 0.9:
                report.append("✅ Mükemmel performans! Production'a hazır.")
            elif best_accuracy > 0.8:
                report.append("✅ İyi performans! Küçük iyileştirmeler yapılabilir.")
            else:
                report.append("⚠️ Performans artırımı gerekli.")
        
        # Dosyaya yaz
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Rapor kaydedildi: {save_path}")
    
    def comprehensive_model_training_pipeline(self, enable_monitoring=True, enable_ab_testing=True):
        """
        Kapsamlı model eğitim pipeline'ı - Tüm iyileştirmelerle
        """
        print("🚀 KAPSAMLI MODEL EĞİTİM PİPELİNE BAŞLATIYOR")
        print("=" * 70)
        
        # 1. Veri hazırlama
        print("\n1️⃣ VERİ HAZIRLIĞI")
        self.load_data()
        self.prepare_data()
        
        # 2. Temel modelleri eğit
        print("\n2️⃣ TEMEL MODEL EĞİTİMLERİ")
        models_to_train = ['naive_bayes', 'logistic_regression']
        
        # Her model için comprehensive evaluation
        for model_name in models_to_train:
            print(f"\n🎯 {model_name.upper()} - Kapsamlı Eğitim ve Değerlendirme")
            print("-" * 60)
            
            # Model eğit
            if model_name == 'naive_bayes':
                model_result = self.train_naive_bayes()
                model = model_result['model']
                
            elif model_name == 'logistic_regression':
                model_result = self.train_logistic_regression()
                model = model_result['model']
            
            # Feature extraction
            X_train_features, _ = self.feature_extractor.extract_tfidf_features(
                self.X_train, max_features=1000
            )
            X_test_features = self.feature_extractor.transform_new_text(
                self.X_test, feature_type='tfidf'
            )
            
            # Comprehensive evaluation
            evaluation_results = self.advanced_evaluator.comprehensive_model_evaluation(
                model.model if hasattr(model, 'model') else model,
                X_train_features, X_test_features, 
                self.y_train, self.y_test, 
                labels=np.unique(self.y_train)
            )
            
            self.comprehensive_results[model_name] = evaluation_results
            self.models[model_name] = model
        
        # 3. Ensemble methods
        print("\n3️⃣ ENSEMBLE METHODS")
        ensemble_results = self._train_ensemble_models(X_train_features, X_test_features)
        
        # 4. A/B Testing setup
        if enable_ab_testing:
            print("\n4️⃣ A/B TESTING KURULUMU")
            self._setup_ab_testing()
        
        # 5. Monitoring setup
        if enable_monitoring:
            print("\n5️⃣ MONİTORİNG SİSTEMİ KURULUMU")
            self._setup_monitoring_system(X_train_features)
        
        # 6. Final reporting
        print("\n6️⃣ KAPSAMLI RAPOR OLUŞTURMA")
        self._generate_comprehensive_report()
        
        print("\n🎉 KAPSAMLI EĞİTİM PİPELİNE TAMAMLANDI!")
        
        return {
            'models': self.models,
            'comprehensive_results': self.comprehensive_results,
            'ensemble_results': ensemble_results
        }
    
    def _train_ensemble_models(self, X_train_features, X_test_features):
        """Ensemble modellerini eğit"""
        print("🤖 Ensemble modelller eğitiliyor...")
        
        # Voting ensemble
        voting_ensemble = self.ensemble_manager.create_voting_ensemble(
            X_train_features, X_test_features, self.y_train, self.y_test
        )
        
        # Weighted ensemble  
        weighted_ensemble = self.ensemble_manager.create_weighted_ensemble(
            X_train_features, self.y_train
        )
        
        # Stacking ensemble
        stacking_ensemble = self.ensemble_manager.create_stacking_ensemble(
            X_train_features, self.y_train
        )
        
        # Ensemble karşılaştırması
        ensemble_results = self.ensemble_manager.compare_ensembles(
            X_test_features, self.y_test
        )
        
        # En iyi ensemble'ı kaydet
        best_ensemble_name = max(
            [(name, res) for name, res in ensemble_results.items() if 'accuracy' in res],
            key=lambda x: x[1]['accuracy']
        )[0]
        
        if best_ensemble_name in self.ensemble_manager.ensembles:
            best_ensemble = self.ensemble_manager.ensembles[best_ensemble_name]
            self.ensemble_manager.save_ensemble(f"best_{best_ensemble_name}", best_ensemble)
        
        return ensemble_results
    
    def _setup_ab_testing(self):
        """A/B testing framework'ünü kur"""
        print("🧪 A/B Testing framework kuruluyor...")
        
        # Model karşılaştırmaları için testler kur
        model_names = list(self.models.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                
                test_name = f"{model_a}_vs_{model_b}"
                
                self.ab_tester.setup_test(
                    model_a=model_a,
                    model_b=model_b,
                    test_name=test_name,
                    description=f"Performance comparison: {model_a} vs {model_b}"
                )
                
                print(f"✅ A/B Test kuruldu: {test_name}")
    
    def _setup_monitoring_system(self, reference_data):
        """Monitoring sistemini kur"""
        print("📊 Monitoring sistemi kuruluyor...")
        
        # Drift detector'ı initialize et
        self.drift_detector = ModelDriftDetector(
            reference_data=reference_data.toarray() if hasattr(reference_data, 'toarray') else reference_data,
            threshold=0.05
        )
        
        # Dashboard oluştur
        self.monitoring_dashboard = MonitoringDashboard(
            self.performance_monitor, self.drift_detector
        )
        
        # Initial performance metrics log
        for model_name, model in self.models.items():
            # Test accuracy
            if hasattr(model, 'model'):
                test_features = self.feature_extractor.transform_new_text(
                    self.X_test, feature_type='tfidf'
                )
                predictions = model.model.predict(test_features)
                accuracy = (predictions == self.y_test).mean()
                
                self.performance_monitor.log_performance_metric(
                    model_name, "accuracy", accuracy, len(self.y_test)
                )
        
        print("✅ Monitoring sistemi kuruldu")
    
    def _generate_comprehensive_report(self):
        """Kapsamlı final raporu oluştur"""
        print("📋 Kapsamlı rapor oluşturuluyor...")
        
        report_lines = []
        report_lines.append("🎯 AUTOTICKET CLASSIFIER - KAPSAMLI EĞİTİM RAPORU")
        report_lines.append("=" * 70)
        report_lines.append(f"📅 Rapor Tarihi: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset bilgisi
        report_lines.append("📊 DATASET BİLGİSİ")
        report_lines.append("-" * 25)
        report_lines.append(f"Train samples: {len(self.X_train)}")
        report_lines.append(f"Test samples: {len(self.X_test)}")
        if self.X_val is not None:
            report_lines.append(f"Validation samples: {len(self.X_val)}")
        
        categories = np.unique(self.y_train)
        report_lines.append(f"Kategoriler: {', '.join(categories)}")
        report_lines.append("")
        
        # Model performansları
        report_lines.append("📈 MODEL PERFORMANSLARI")
        report_lines.append("-" * 30)
        
        model_scores = []
        for model_name, results in self.comprehensive_results.items():
            if 'cross_validation' in results and 'accuracy' in results['cross_validation']:
                cv_mean = results['cross_validation']['accuracy']['test_mean']
                cv_std = results['cross_validation']['accuracy']['test_std']
                
                # Learning curve overfitting gap
                overfitting_gap = results.get('learning_curves', {}).get('overfitting_gap', 0)
                
                model_scores.append({
                    'model': model_name,
                    'cv_accuracy': cv_mean,
                    'cv_std': cv_std,
                    'overfitting_gap': overfitting_gap
                })
                
                report_lines.append(f"🤖 {model_name.upper()}:")
                report_lines.append(f"   CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
                report_lines.append(f"   Overfitting Gap: {overfitting_gap:.4f}")
                
                if overfitting_gap > 0.15:
                    report_lines.append("   ⚠️ Yüksek overfitting riski")
                elif overfitting_gap > 0.1:
                    report_lines.append("   🔄 Orta overfitting riski")
                else:
                    report_lines.append("   ✅ İyi generalization")
                
                report_lines.append("")
        
        # En iyi model
        if model_scores:
            best_model = max(model_scores, key=lambda x: x['cv_accuracy'])
            report_lines.append("🏆 EN İYİ MODEL")
            report_lines.append("-" * 20)
            report_lines.append(f"Model: {best_model['model'].upper()}")
            report_lines.append(f"CV Accuracy: {best_model['cv_accuracy']:.4f}")
            report_lines.append("")
        
        # Öneriler
        report_lines.append("💡 ÖNERİLER")
        report_lines.append("-" * 15)
        
        if any(score['overfitting_gap'] > 0.15 for score in model_scores):
            report_lines.append("🚨 Overfitting problemi tespit edildi:")
            report_lines.append("   - Regularization parametrelerini artırın")
            report_lines.append("   - Feature selection uygulayın")
            report_lines.append("   - Daha fazla training data toplayın")
        
        report_lines.append("📊 Production için öneriler:")
        report_lines.append("   - Real-time monitoring aktif edilsin")
        report_lines.append("   - A/B testing ile model karşılaştırması yapılsın")
        report_lines.append("   - Ensemble methods production'a alınsın")
        report_lines.append("   - Drift detection ile veri kalitesi izlensin")
        
        # Raporu kaydet
        report_text = "\n".join(report_lines)
        report_path = f"evaluation_results/comprehensive_training_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n💾 Kapsamlı rapor kaydedildi: {report_path}")
        
        return report_path

def main():
    """Ana eğitim fonksiyonu - Enhanced"""
    print("� AutoTicket Classifier - Gelişmiş Model Eğitimi")
    print("="*60)
    
    try:
        # Enhanced model trainer oluştur
        trainer = EnhancedModelTrainer()
        
        print("\n🎯 Eğitim modunu seçin:")
        print("1. Temel eğitim (eski method)")
        print("2. 🚀 Kapsamlı eğitim (Tüm iyileştirmelerle)")
        
        choice = input("Seçiminiz (1/2): ").strip()
        
        if choice == "2":
            # Kapsamlı eğitim pipeline
            print("\n🔧 Kapsamlı eğitim seçenekleri:")
            print("1. Monitoring aktif")
            print("2. A/B Testing aktif") 
            print("3. Her ikisi de aktif (Önerilen)")
            
            options = input("Seçiminiz (1/2/3): ").strip()
            
            enable_monitoring = options in ['1', '3']
            enable_ab_testing = options in ['2', '3']
            
            # Kapsamlı pipeline çalıştır
            results = trainer.comprehensive_model_training_pipeline(
                enable_monitoring=enable_monitoring,
                enable_ab_testing=enable_ab_testing
            )
            
            print("\n🎊 KAPSAMLI EĞİTİM TAMAMLANDI!")
            print("✅ Confusion matrix analizi tamamlandı")
            print("✅ K-fold cross validation tamamlandı") 
            print("✅ Learning curves ve overfitting kontrolü tamamlandı")
            print("✅ Precision/Recall detaylı analizi tamamlandı")
            print("✅ Model drift detection sistemi kuruldu")
            print("✅ Performance monitoring dashboard hazır")
            print("✅ Ensemble methods eğitildi")
            print("✅ A/B testing framework kuruldu")
            
        else:
            # Temel eğitim (eski method)
            print("\n📊 Temel eğitim modu:")
            
            # 1. Veri yükle
            trainer.load_data()
            
            # 2. Veri ön işle  
            trainer.prepare_data(test_size=0.2, val_size=0.1)
            
            # 3. Modelleri eğit
            print("\n🎯 Hangi modelleri eğitmek istiyorsunuz?")
            print("1. Sadece hızlı modeller (Naive Bayes + Logistic Regression)")
            print("2. Tüm modeller (BERT dahil)")
            
            model_choice = input("Seçiminiz (1/2): ").strip()
            
            if model_choice == "1":
                trainer.train_all_models(include_bert=False)
            else:
                epochs = input("BERT için epoch sayısı (varsayılan: 3): ").strip()
                epochs = int(epochs) if epochs.isdigit() else 3
                trainer.train_all_models(include_bert=True, bert_epochs=epochs)
            
            # 4. Modelleri karşılaştır
            trainer.compare_models()
            
            # 5. Sonuçları kaydet
            trainer.save_results()
            trainer.generate_report()
        
        print(f"\n🎉 Model eğitimi başarıyla tamamlandı!")
        print(f"📁 Eğitilmiş modeller: models/trained/")
        print(f"📊 Değerlendirme sonuçları: evaluation_results/")
        print(f"📈 Monitoring verileri: monitoring/")
        
        # Ek bilgiler
        if choice == "2":
            print(f"\n� EK ÖZELLİKLER:")
            print(f"📊 Monitoring dashboard: monitoring_dashboard.html")
            print(f"📈 Drift detection dashboard: drift_dashboard.html") 
            print(f"🧪 A/B testing sonuçları: improvements/ dizininde")
            print(f"🤖 Ensemble modeller: models/ensemble/ dizininde")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

def demo_comprehensive_training():
    """Comprehensive training demo"""
    print("🧪 COMPREHENSIVE TRAINING DEMO")
    print("=" * 40)
    
    trainer = EnhancedModelTrainer()
    
    # Generate sample data if needed
    if not os.path.exists("data/processed_data.csv"):
        print("📊 Demo veri oluşturuluyor...")
        from data_generator import TicketDataGenerator
        
        generator = TicketDataGenerator()
        df = generator.generate_comprehensive_dataset(n_samples=500)
        df.to_csv("data/processed_data.csv", index=False)
        print("✅ Demo veri oluşturuldu")
    
    # Run comprehensive pipeline
    results = trainer.comprehensive_model_training_pipeline(
        enable_monitoring=True,
        enable_ab_testing=True
    )
    
    return results

if __name__ == "__main__":
    main()
