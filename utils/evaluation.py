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
from sklearn.model_selection import cross_val_score
import time

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
    
    def compare_models(self, figsize=(12, 8)):
        """Model karşılaştırması görselleştirme"""
        if len(self.results) < 2:
            print("❌ Karşılaştırma için en az 2 model gerekli")
            return
        
        # Sonuçları DataFrame'e çevir
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Prediction Time (s)': results['prediction_time']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(df_comparison['Model'], df_comparison[metric], 
                         color=colors[i], alpha=0.7)
            ax.set_title(f'{metric} Karşılaştırması')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            
            # Değerleri bar'ların üzerine yaz
            for bar, value in zip(bars, df_comparison[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return df_comparison
    
    def plot_performance_vs_time(self):
        """Performans vs Hız karşılaştırması"""
        if len(self.results) < 2:
            print("❌ Karşılaştırma için en az 2 model gerekli")
            return
        
        models = list(self.results.keys())
        f1_scores = [self.results[model]['f1_score'] for model in models]
        times = [self.results[model]['prediction_time'] for model in models]
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        scatter = plt.scatter(times, f1_scores, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        
        # Model isimlerini ekle
        for i, model in enumerate(models):
            plt.annotate(model, (times[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Tahmin Süresi (saniye)')
        plt.ylabel('F1-Score')
        plt.title('Model Performansı vs Hız')
        plt.grid(True, alpha=0.3)
        
        # İdeal bölge (düşük süre, yüksek performans)
        plt.axhline(y=max(f1_scores), color='green', linestyle='--', alpha=0.5, label='En İyi F1-Score')
        plt.axvline(x=min(times), color='blue', linestyle='--', alpha=0.5, label='En Hızlı Model')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='f1_weighted'):
        """Cross-validation ile model değerlendirme"""
        print(f"🔄 {cv}-fold cross-validation yapılıyor...")
        
        start_time = time.time()
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        cv_time = time.time() - start_time
        
        print(f"✅ Cross-Validation Sonuçları ({scoring}):")
        print(f"   Ortalama: {scores.mean():.4f} (±{scores.std() * 2:.4f})")
        print(f"   Min: {scores.min():.4f}")
        print(f"   Max: {scores.max():.4f}")
        print(f"   CV Süresi: {cv_time:.2f}s")
        
        return scores
    
    def generate_summary_report(self, class_names=None):
        """Özet rapor oluştur"""
        if not self.results:
            print("❌ Değerlendirilecek model bulunamadı")
            return
        
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANS ÖZET RAPORU")
        print("="*80)
        
        # En iyi modeli bul
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['f1_score'])
        
        print(f"\n🏆 En İyi Model: {best_model}")
        print(f"   F1-Score: {self.results[best_model]['f1_score']:.4f}")
        print(f"   Accuracy: {self.results[best_model]['accuracy']:.4f}")
        
        # En hızlı model
        fastest_model = min(self.results.keys(),
                           key=lambda x: self.results[x]['prediction_time'])
        
        print(f"\n⚡ En Hızlı Model: {fastest_model}")
        print(f"   Tahmin Süresi: {self.results[fastest_model]['prediction_time']:.3f}s")
        print(f"   F1-Score: {self.results[fastest_model]['f1_score']:.4f}")
        
        # Tüm modeller tablosu
        print(f"\n📋 Tüm Modeller:")
        print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Süre (s)':<10}")
        print("-" * 50)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} {results['accuracy']:<10.4f} {results['f1_score']:<10.4f} {results['prediction_time']:<10.3f}")
        
        # Öneriler
        print(f"\n💡 Öneriler:")
        if self.results[best_model]['f1_score'] > 0.9:
            print("   ✅ Mükemmel performans! Production'a hazır.")
        elif self.results[best_model]['f1_score'] > 0.8:
            print("   ✅ İyi performans! Küçük iyileştirmeler yapılabilir.")
        else:
            print("   ⚠️  Performans artırımı gerekli. Daha fazla veri veya feature engineering önerilidir.")
        
        if self.results[fastest_model]['prediction_time'] < 0.1:
            print("   ⚡ Hız mükemmel! Real-time uygulamalar için uygun.")
        elif self.results[fastest_model]['prediction_time'] < 1.0:
            print("   ⚡ Hız iyi! Çoğu uygulama için yeterli.")
        else:
            print("   🐌 Hız optimizasyonu gerekebilir.")

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
