"""
🎓 Ana Model Eğitim Scripti
Tüm modelleri eğitir ve karşılaştırır
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

class ModelTrainer:
    def __init__(self, data_path="data/processed_data.csv"):
        """
        Model eğitim sınıfı
        
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
        
        self.preprocessor = TurkishTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.evaluator = ModelEvaluator()
        
        self.results = {}
        self.models = {}
        
        # Model save dizini
        os.makedirs("models/trained", exist_ok=True)
    
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

def main():
    """Ana eğitim fonksiyonu"""
    print("🎫 AutoTicket Classifier - Model Eğitimi")
    print("="*60)
    
    # Trainer oluştur
    trainer = ModelTrainer()
    
    try:
        # 1. Veri yükle
        trainer.load_data()
        
        # 2. Veri ön işle
        trainer.preprocess_data(test_size=0.2, val_size=0.1)
        
        # 3. Modelleri eğit
        print("\n🎯 Hangi modelleri eğitmek istiyorsunuz?")
        print("1. Sadece hızlı modeller (Naive Bayes + Logistic Regression)")
        print("2. Tüm modeller (BERT dahil)")
        
        choice = input("Seçiminiz (1/2): ").strip()
        
        if choice == "1":
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
        print(f"📊 Sonuçlar: models/training_results.json")
        print(f"📄 Rapor: models/model_training_report.md")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
