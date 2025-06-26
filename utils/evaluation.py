"""
📈 Model Evaluation Utilities
Model performansını değerlendirme araçları
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, y_pred=None, 
                      model_name="Model", class_names=None):
        """Kapsamlı model değerlendirmesi"""
        print(f"📊 {model_name} değerlendiriliyor...")
        
        start_time = time.time()
        
        # Prediction varsa kullan, yoksa tahmin yap
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        prediction_time = time.time() - start_time
        
        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Sonuçları kaydet
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': prediction_time,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        print(f"✅ {model_name} Sonuçları:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Tahmin Süresi: {prediction_time:.3f}s")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': prediction_time
        }
    
    def print_classification_report(self, model_name, class_names=None):
        """Detaylı sınıflandırma raporu"""
        if model_name not in self.results:
            print(f"❌ {model_name} için sonuç bulunamadı")
            return
        
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        print(f"\n📋 {model_name} - Detaylı Sınıflandırma Raporu")
        print("=" * 60)
        
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            digits=4
        )
        print(report)
    
    def plot_confusion_matrix(self, model_name, class_names=None, figsize=(8, 6)):
        """Confusion matrix görselleştirme"""
        if model_name not in self.results:
            print(f"❌ {model_name} için sonuç bulunamadı")
            return
        
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        # Confusion matrix hesapla
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize et (yüzde olarak)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2%',
                   cmap='Blues',
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        
        plt.title(f'{model_name} - Confusion Matrix (%)')
        plt.ylabel('Gerçek Kategori')
        plt.xlabel('Tahmin Edilen Kategori')
        plt.tight_layout()
        plt.show()
        
        return cm, cm_normalized
    
    def comprehensive_model_evaluation(self, model, X_train, X_test, y_train, y_test, 
                                     labels=None, model_name="Model"):
        """
        Kapsamlı model değerlendirme sistemi:
        - Confusion matrix
        - Classification report  
        - K-fold cross validation
        - Learning curves
        - Overfitting kontrolü
        """
        print(f"🔍 {model_name} için kapsamlı değerlendirme başlıyor...")
        
        results = {
            'model_name': model_name,
            'timestamp': time.time(),
            'basic_metrics': {},
            'cross_validation': {},
            'learning_curves': {},
            'overfitting_analysis': {}
        }
        
        # 1. Temel Metrikler
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            
        results['basic_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        self._plot_confusion_matrix(cm, labels, model_name)
        
        # 3. Classification Report
        report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
        results['classification_report'] = report
        
        # 4. K-Fold Cross Validation
        if hasattr(model, 'model'):  # Sklearn modeli varsa
            cv_scores = cross_val_score(model.model, X_train, y_train, cv=5, scoring='accuracy')
            results['cross_validation'] = {
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
        
        # 5. Learning Curves
        if hasattr(model, 'model'):
            train_sizes, train_scores, val_scores = learning_curve(
                model.model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            results['learning_curves'] = {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
            
            self._plot_learning_curves(train_sizes, train_scores, val_scores, model_name)
        
        # 6. Overfitting Analizi
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = results['basic_metrics']['accuracy']
        overfitting_gap = train_acc - test_acc
        
        results['overfitting_analysis'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting_gap': overfitting_gap,
            'is_overfitting': overfitting_gap > 0.1,  # %10'dan fazla fark varsa overfitting
            'recommendation': self._get_overfitting_recommendation(overfitting_gap)
        }
        
        self.results[model_name] = results
        
        # Sonuçları yazdır
        self._print_comprehensive_results(results)
        
        return results
    
    def _plot_confusion_matrix(self, cm, labels, model_name):
        """Confusion matrix görselleştirme"""
        plt.figure(figsize=(10, 8))
        
        # Raw confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} - Confusion Matrix (Raw)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Normalized confusion matrix
        plt.subplot(1, 2, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{model_name} - Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'evaluation_results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_learning_curves(self, train_sizes, train_scores, val_scores, model_name):
        """Learning curves görselleştirme"""
        plt.figure(figsize=(10, 6))
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title(f'{model_name} - Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'evaluation_results/{model_name}_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _get_overfitting_recommendation(self, gap):
        """Overfitting durumuna göre öneri ver"""
        if gap < 0.05:
            return "✅ Model iyi genelleme yapıyor"
        elif gap < 0.1:
            return "⚠️ Hafif overfitting var, düzenleme teknikleri düşünülebilir"
        elif gap < 0.2:
            return "🔴 Overfitting var! Regularization, dropout, daha fazla veri gerekli"
        else:
            return "🚨 Ciddi overfitting! Model karmaşıklığını azaltın"
    
    def _print_comprehensive_results(self, results):
        """Comprehensive results'ı yazdır"""
        print(f"\n📊 {results['model_name']} - KAPSAMLI DEĞERLENDİRME SONUÇLARI")
        print("=" * 60)
        
        # Temel metrikler
        metrics = results['basic_metrics']
        print(f"🎯 Temel Metrikler:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        
        # Cross validation
        if 'cv_scores' in results['cross_validation']:
            cv = results['cross_validation']
            print(f"\n🔄 Cross Validation:")
            print(f"   CV Accuracy: {cv['mean_cv_score']:.4f} ± {cv['std_cv_score']:.4f}")
        
        # Overfitting analizi
        ov = results['overfitting_analysis']
        print(f"\n🔍 Overfitting Analizi:")
        print(f"   Train Accuracy: {ov['train_accuracy']:.4f}")
        print(f"   Test Accuracy:  {ov['test_accuracy']:.4f}")
        print(f"   Gap: {ov['overfitting_gap']:.4f}")
        print(f"   {ov['recommendation']}")
        
        print("\n" + "=" * 60)
    
    def detect_data_drift(self, baseline_data, new_data, method='ks_test', threshold=0.05):
        """
        Data drift detection
        
        Methods:
        - ks_test: Kolmogorov-Smirnov test
        - chi2_test: Chi-square test  
        - psi: Population Stability Index
        """
        drift_results = {
            'method': method,
            'threshold': threshold,
            'features_with_drift': [],
            'overall_drift_detected': False,
            'drift_scores': {}
        }
        
        if method == 'ks_test':
            for i in range(min(baseline_data.shape[1], new_data.shape[1])):
                baseline_feature = baseline_data[:, i]
                new_feature = new_data[:, i]
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(baseline_feature, new_feature)
                
                drift_results['drift_scores'][f'feature_{i}'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
                
                if p_value < threshold:
                    drift_results['features_with_drift'].append(f'feature_{i}')
        
        drift_results['overall_drift_detected'] = len(drift_results['features_with_drift']) > 0
        
        return drift_results
    
    def compare_models(self, models_results=None):
        """Birden fazla modeli karşılaştır"""
        print("\n🏆 MODEL KARŞILAŞTIRMASI")
        print("=" * 50)
        
        # Eğer models_results verilmemişse self.results kullan
        if models_results is None:
            models_results = self.results
        
        # Metrik tablosu oluştur
        comparison_data = []
        
        for model_name, results in models_results.items():
            # İki farklı format destekle
            if 'basic_metrics' in results:
                # Comprehensive evaluation format
                metrics = results['basic_metrics']
                row = {
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1']
                }
                
                # Overfitting bilgisi ekle
                if 'overfitting_analysis' in results:
                    ov = results['overfitting_analysis']
                    row['Overfitting Gap'] = ov['overfitting_gap']
                    row['Is Overfitting'] = ov['is_overfitting']
            else:
                # Simple evaluation format
                row = {
                    'Model': model_name,
                    'Accuracy': results.get('accuracy', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1-Score': results.get('f1_score', 0)
                }
            
            comparison_data.append(row)
        
        if not comparison_data:
            print("❌ Karşılaştırılacak model bulunamadı")
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # En iyi modeli bul
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        print(f"\n🥇 En İyi Model: {best_model}")
        
        return comparison_df
    
    def generate_summary_report(self, class_names=None):
        """Generate a comprehensive summary report"""
        print("\n📋 MODEL EVALUATION SUMMARY")
        print("=" * 50)
        
        if not self.results:
            print("❌ No evaluation results found")
            return
        
        # Model count
        print(f"📊 Total models evaluated: {len(self.results)}")
        
        # Best model by accuracy
        best_model = max(self.results.keys(), key=lambda x: self.results[x].get('accuracy', 0))
        best_accuracy = self.results[best_model].get('accuracy', 0)
        
        print(f"🏆 Best model: {best_model}")
        print(f"🎯 Best accuracy: {best_accuracy:.4f}")
        
        # Average metrics
        accuracies = [r.get('accuracy', 0) for r in self.results.values()]
        precisions = [r.get('precision', 0) for r in self.results.values()]
        recalls = [r.get('recall', 0) for r in self.results.values()]
        f1_scores = [r.get('f1_score', 0) for r in self.results.values()]
        
        print(f"\n📈 Average Metrics:")
        print(f"   Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"   Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"   Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"   F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
        # Performance tiers
        excellent_models = [name for name, res in self.results.items() if res.get('accuracy', 0) >= 0.9]
        good_models = [name for name, res in self.results.items() if 0.8 <= res.get('accuracy', 0) < 0.9]
        fair_models = [name for name, res in self.results.items() if res.get('accuracy', 0) < 0.8]
        
        print(f"\n🎯 Performance Tiers:")
        if excellent_models:
            print(f"   🌟 Excellent (≥90%): {', '.join(excellent_models)}")
        if good_models:
            print(f"   ✅ Good (80-90%): {', '.join(good_models)}")
        if fair_models:
            print(f"   ⚠️ Fair (<80%): {', '.join(fair_models)}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if best_accuracy >= 0.95:
            print("   🚀 Ready for production!")
        elif best_accuracy >= 0.9:
            print("   ✅ Good performance, minor tuning recommended")
        elif best_accuracy >= 0.8:
            print("   🔧 Needs improvement before production")
        else:
            print("   ⚠️ Significant improvement required")
        
        if len(excellent_models) > 1:
            print("   🤖 Consider ensemble methods")
        
        print("=" * 50)
        
        return {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'average_accuracy': np.mean(accuracies),
            'model_count': len(self.results)
        }

def demo_evaluation():
    """Değerlendirme demo'su"""
    print("🧪 Model Değerlendirme Demo'su")
    print("=" * 50)
    
    # Örnek veri oluştur
    np.random.seed(42)
    n_samples = 1000
    n_classes = 6
    
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Farklı performanslarda 3 model simülasyonu
    # Model 1: İyi performans
    y_pred1 = y_true.copy()
    noise_indices = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    y_pred1[noise_indices] = np.random.randint(0, n_classes, len(noise_indices))
    
    # Model 2: Orta performans
    y_pred2 = y_true.copy()
    noise_indices = np.random.choice(n_samples, int(n_samples * 0.25), replace=False)
    y_pred2[noise_indices] = np.random.randint(0, n_classes, len(noise_indices))
    
    # Model 3: Düşük performans
    y_pred3 = y_true.copy()
    noise_indices = np.random.choice(n_samples, int(n_samples * 0.4), replace=False)
    y_pred3[noise_indices] = np.random.randint(0, n_classes, len(noise_indices))
    
    # Evaluator oluştur
    evaluator = ModelEvaluator()
    
    # Modelleri değerlendir
    class MockModel:
        def __init__(self, predictions):
            self.predictions = predictions
        def predict(self, X):
            return self.predictions
    
    X_dummy = np.zeros((n_samples, 10))  # Dummy features
    
    evaluator.evaluate_model(MockModel(y_pred1), X_dummy, y_true, model_name="BERT Classifier")
    evaluator.evaluate_model(MockModel(y_pred2), X_dummy, y_true, model_name="Logistic Regression")
    evaluator.evaluate_model(MockModel(y_pred3), X_dummy, y_true, model_name="Naive Bayes")
    
    # Karşılaştırma
    class_names = ["Ödeme", "Rezervasyon", "Kullanıcı", "Şikayet", "Bilgi", "Teknik"]
    comparison_df = evaluator.compare_models()
    
    # Özet rapor
    evaluator.generate_summary_report(class_names)

if __name__ == "__main__":
    demo_evaluation()
