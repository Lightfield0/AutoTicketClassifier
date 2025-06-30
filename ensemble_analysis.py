#!/usr/bin/env python3
"""
🔬 Ensemble Analiz Sistemi
AutoTicketClassifier projesindeki 4 farklı modeli (Naive Bayes, Logistic Regression, BERT, Ensemble) 
detaylı analiz eden ve karşılaştıran kapsamlı test sistemi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Model imports
from models.naive_bayes import NaiveBayesClassifier
from models.logistic_regression import LogisticRegressionClassifier
from models.bert_classifier import BERTClassifier
from models.ensemble_system import EnsembleManager, WeightedEnsemble

# Utils imports
from utils.text_preprocessing import TurkishTextPreprocessor
from utils.feature_extraction import FeatureExtractor
from utils.evaluation import ModelEvaluator
from data_generator import TicketDataGenerator

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
import time

class EnsembleAnalyzer:
    """Ensemble sistemi detaylı analizi"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_extractor = None
        self.preprocessor = TurkishTextPreprocessor()
        self.evaluator = ModelEvaluator()
        
        # Results directory
        self.results_dir = Path("evaluation_results/ensemble_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_or_generate_data(self, use_existing=True, num_tickets=2000):
        """Veri yükle veya oluştur"""
        print("📊 Veri hazırlığı...")
        
        data_file = Path("data/processed_data.csv")
        
        if use_existing and data_file.exists():
            print("📁 Mevcut veri dosyası yükleniyor...")
            df = pd.read_csv(data_file)
        else:
            print("🎲 Yeni veri oluşturuluyor...")
            generator = TicketDataGenerator()
            tickets = generator.generate_tickets(num_tickets=num_tickets)
            df = pd.DataFrame(tickets)
            
            # Save for future use
            df.to_csv(data_file, index=False)
        
        print(f"✅ Toplam {len(df)} ticket yüklendi")
        print(f"📋 Kategoriler: {df['category'].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, df):
        """Feature extraction ve preprocessing"""
        print("🔧 Feature extraction...")
        
        # Text preprocessing
        df['processed_text'] = df['message'].apply(self.preprocessor.preprocess_text)
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor()
        
        # TF-IDF features
        X_tfidf, feature_names = self.feature_extractor.extract_tfidf_features(
            df['processed_text'], 
            max_features=1000
        )
        self.tfidf_vectorizer = self.feature_extractor.tfidf_vectorizer
        
        y = df['category']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_tfidf, y, 
            test_size=0.3, 
            random_state=42, 
            stratify=y
        )
        
        print(f"✅ Training set: {self.X_train.shape}")
        print(f"✅ Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """Tüm modelleri başlat"""
        print("🤖 Model initialization...")
        
        # 1. Naive Bayes
        self.models['naive_bayes'] = NaiveBayesClassifier(model_type='multinomial')
        
        # 2. Logistic Regression  
        self.models['logistic_regression'] = LogisticRegressionClassifier(
            multi_class='ovr', 
            solver='liblinear'
        )
        
        # 3. BERT (lightweight config for demo)
        try:
            self.models['bert'] = BERTClassifier(
                model_name='distilbert-base-uncased',
                num_classes=len(np.unique(self.y_train)),
                max_length=64,  # Shorter for speed
                learning_rate=2e-5
            )
        except Exception as e:
            print(f"⚠️ BERT model initialization failed: {e}")
            print("🔄 BERT will be skipped in analysis")
            
        print(f"✅ {len(self.models)} models initialized")
    
    def train_individual_models(self):
        """Her modeli ayrı ayrı eğit"""
        print("\n🎯 INDIVIDUAL MODEL TRAINING")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n🚀 Training {name}...")
            start_time = time.time()
            
            try:
                if name == 'bert':
                    # BERT için özel eğitim
                    if hasattr(self.X_train, 'toarray'):
                        # TF-IDF'den text'e geri dönüşüm gerekiyor
                        print("⚠️ BERT TF-IDF features'dan çalışamaz, text data gerekiyor")
                        continue
                    else:
                        model.train(self.X_train, self.y_train, epochs=2)  # Quick training
                else:
                    # Sklearn-based models
                    model.train(self.X_train, self.y_train)
                
                training_time = time.time() - start_time
                print(f"✅ {name} trained in {training_time:.2f}s")
                
            except Exception as e:
                print(f"❌ {name} training failed: {e}")
                if name in self.models:
                    del self.models[name]
    
    def train_ensemble_models(self):
        """Ensemble modellerini eğit"""
        print("\n🤖 ENSEMBLE MODEL TRAINING")
        print("=" * 50)
        
        if len(self.models) < 2:
            print("⚠️ En az 2 model gerekiyor ensemble için")
            return
        
        # Ensemble Manager
        ensemble_manager = EnsembleManager()
        
        # Base models'ı ensemble manager'a ekle
        for name, model in self.models.items():
            if hasattr(model, 'model'):
                ensemble_manager.base_models[name] = model.model
            else:
                ensemble_manager.base_models[name] = model
        
        try:
            # 1. Voting Ensemble
            print("🗳️ Creating Voting Ensemble...")
            voting_ensemble = ensemble_manager.create_voting_ensemble(
                self.X_train, self.y_train, voting='soft'
            )
            self.models['voting_ensemble'] = voting_ensemble
            
            # 2. Weighted Ensemble
            print("⚖️ Creating Weighted Ensemble...")
            weighted_ensemble = ensemble_manager.create_weighted_ensemble(
                self.X_train, self.y_train
            )
            self.models['weighted_ensemble'] = weighted_ensemble
            
            # 3. Stacking Ensemble
            print("🏗️ Creating Stacking Ensemble...")
            stacking_ensemble = ensemble_manager.create_stacking_ensemble(
                self.X_train, self.y_train
            )
            self.models['stacking_ensemble'] = stacking_ensemble
            
        except Exception as e:
            print(f"❌ Ensemble creation failed: {e}")
    
    def evaluate_all_models(self):
        """Tüm modelleri değerlendir"""
        print("\n📊 MODEL EVALUATION")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            try:
                start_time = time.time()
                
                # Predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(self.X_test)
                else:
                    print(f"⚠️ {name} doesn't have predict method")
                    continue
                
                prediction_time = time.time() - start_time
                
                # Metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test, y_pred, average='weighted'
                )
                
                # Classification report
                class_report = classification_report(
                    self.y_test, y_pred, output_dict=True
                )
                
                # Store results
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'prediction_time': prediction_time,
                    'predictions': y_pred,
                    'classification_report': class_report
                }
                
                print(f"✅ {name:20} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {prediction_time:.3f}s")
                
            except Exception as e:
                print(f"❌ {name} evaluation failed: {e}")
    
    def cross_validate_models(self, cv_folds=5):
        """Cross validation analizi"""
        print(f"\n🔄 CROSS VALIDATION (CV={cv_folds})")
        print("=" * 50)
        
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                print(f"🔄 CV for {name}...")
                
                # sklearn modeli al
                sklearn_model = model.model if hasattr(model, 'model') else model
                
                cv_scores = cross_val_score(
                    sklearn_model, self.X_train, self.y_train, 
                    cv=cv_folds, scoring='accuracy'
                )
                
                cv_results[name] = {
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                print(f"  📈 {name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"❌ CV failed for {name}: {e}")
        
        return cv_results
    
    def analyze_model_complexity(self):
        """Model complexity analizi"""
        print("\n🧮 MODEL COMPLEXITY ANALYSIS")
        print("=" * 50)
        
        complexity_analysis = {}
        
        for name, model in self.models.items():
            analysis = {'name': name}
            
            try:
                # Parameter count
                if hasattr(model, 'model'):
                    sklearn_model = model.model
                else:
                    sklearn_model = model
                
                # Model-specific analysis
                if 'naive_bayes' in name:
                    analysis['type'] = 'Probabilistic'
                    analysis['parameters'] = 'O(features × classes)'
                    analysis['training_complexity'] = 'O(n × features)'
                    analysis['prediction_complexity'] = 'O(features × classes)'
                    
                elif 'logistic' in name:
                    analysis['type'] = 'Linear'
                    if hasattr(sklearn_model, 'coef_'):
                        analysis['parameters'] = sklearn_model.coef_.size
                    analysis['training_complexity'] = 'O(n × features × iterations)'
                    analysis['prediction_complexity'] = 'O(features × classes)'
                    
                elif 'bert' in name:
                    analysis['type'] = 'Deep Learning'
                    analysis['parameters'] = '66M+ (DistilBERT)'
                    analysis['training_complexity'] = 'O(sequence_length²)'
                    analysis['prediction_complexity'] = 'O(sequence_length²)'
                    
                elif 'ensemble' in name:
                    analysis['type'] = 'Ensemble'
                    analysis['parameters'] = 'Sum of base models'
                    analysis['training_complexity'] = 'Sum of base model complexities'
                    analysis['prediction_complexity'] = 'Sum of base model complexities'
                
                complexity_analysis[name] = analysis
                
                print(f"🔍 {name}:")
                print(f"   Type: {analysis.get('type', 'Unknown')}")
                print(f"   Parameters: {analysis.get('parameters', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Complexity analysis failed for {name}: {e}")
        
        return complexity_analysis
    
    def generate_comparison_plots(self):
        """Model karşılaştırma grafikleri"""
        print("\n📈 GENERATING COMPARISON PLOTS")
        print("=" * 50)
        
        if not self.results:
            print("⚠️ No results to plot")
            return
        
        # Set up plotting style (headless mode for server environments)
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🤖 AutoTicketClassifier: Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Accuracy bar plot
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        axes[0,0].bar(models, accuracies, color=sns.color_palette("husl", len(models)))
        axes[0,0].set_title('🎯 Model Accuracy Comparison')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            axes[0,0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        # F1-Score comparison
        f1_scores = [self.results[model]['f1_score'] for model in models]
        axes[0,1].bar(models, f1_scores, color=sns.color_palette("husl", len(models)))
        axes[0,1].set_title('📊 F1-Score Comparison')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        for i, v in enumerate(f1_scores):
            axes[0,1].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
        
        # Prediction time comparison
        pred_times = [self.results[model]['prediction_time'] for model in models]
        axes[1,0].bar(models, pred_times, color=sns.color_palette("husl", len(models)))
        axes[1,0].set_title('⚡ Prediction Time Comparison')
        axes[1,0].set_ylabel('Time (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        for i, v in enumerate(pred_times):
            axes[1,0].text(i, v + max(pred_times)*0.01, f'{v:.3f}s', ha='center', va='bottom')
        
        # Performance vs Speed scatter
        axes[1,1].scatter(pred_times, accuracies, s=100, c=range(len(models)), cmap='viridis')
        for i, model in enumerate(models):
            axes[1,1].annotate(model, (pred_times[i], accuracies[i]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,1].set_title('🚀 Performance vs Speed')
        axes[1,1].set_xlabel('Prediction Time (seconds)')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved: {plot_file}")
        
        # Don't show plot in headless mode
        # plt.show()
    
    def generate_detailed_report(self):
        """Detaylı analiz raporu oluştur"""
        print("\n📋 GENERATING DETAILED REPORT")
        print("=" * 50)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"ensemble_analysis_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🤖 AutoTicketClassifier: Ensemble Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 📋 Executive Summary\n\n")
            f.write("Bu rapor AutoTicketClassifier projesindeki 4 farklı model türünün (Naive Bayes, Logistic Regression, BERT, Ensemble) detaylı performans analizini içermektedir.\n\n")
            
            # Model Overview
            f.write("## 🏗️ Model Architecture Overview\n\n")
            f.write("| Model Type | Algorithm | Complexity | Use Case |\n")
            f.write("|------------|-----------|------------|----------|\n")
            f.write("| Naive Bayes | Multinomial NB | Low | Baseline, Fast |\n")
            f.write("| Logistic Regression | Linear Classification | Medium | Balanced Performance |\n")
            f.write("| BERT | Transformer (Deep Learning) | High | Maximum Accuracy |\n")
            f.write("| Ensemble | Voting/Weighted/Stacking | Variable | Best Overall |\n\n")
            
            # Performance Results
            f.write("## 📊 Performance Results\n\n")
            
            if self.results:
                # Best performing model
                best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
                best_accuracy = self.results[best_model]['accuracy']
                
                f.write(f"### 🏆 Best Performing Model: **{best_model}**\n")
                f.write(f"- **Accuracy:** {best_accuracy:.4f}\n")
                f.write(f"- **F1-Score:** {self.results[best_model]['f1_score']:.4f}\n")
                f.write(f"- **Prediction Time:** {self.results[best_model]['prediction_time']:.3f}s\n\n")
                
                # All models comparison
                f.write("### 📈 All Models Comparison\n\n")
                f.write("| Model | Accuracy | F1-Score | Precision | Recall | Pred. Time |\n")
                f.write("|-------|----------|----------|-----------|--------|------------|\n")
                
                for model_name in sorted(self.results.keys()):
                    result = self.results[model_name]
                    f.write(f"| {model_name} | {result['accuracy']:.4f} | {result['f1_score']:.4f} | ")
                    f.write(f"{result['precision']:.4f} | {result['recall']:.4f} | {result['prediction_time']:.3f}s |\n")
                
                f.write("\n")
            
            # Ensemble Analysis
            f.write("## 🤖 Ensemble Analysis\n\n")
            f.write("### Ensemble Methodologies Used:\n\n")
            f.write("1. **Voting Ensemble:** Simple majority/probability voting\n")
            f.write("2. **Weighted Ensemble:** Performance-based weighted voting\n")
            f.write("3. **Stacking Ensemble:** Meta-learner based combination\n\n")
            
            # Technical Details
            f.write("## 🔧 Technical Implementation Details\n\n")
            f.write("### Feature Engineering:\n")
            f.write("- **Text Preprocessing:** Turkish language support, stop words, lemmatization\n")
            f.write("- **Feature Extraction:** TF-IDF vectorization (max_features=1000)\n")
            f.write("- **Data Split:** 70% training, 30% testing with stratification\n\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            f.write("### Production Deployment:\n")
            
            if self.results:
                fastest_model = min(self.results.keys(), key=lambda x: self.results[x]['prediction_time'])
                most_accurate = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
                
                f.write(f"- **For Speed:** Use `{fastest_model}` (fastest prediction)\n")
                f.write(f"- **For Accuracy:** Use `{most_accurate}` (highest accuracy)\n")
                f.write("- **For Production:** Use ensemble methods for best balance\n\n")
            
            f.write("### Scaling Considerations:\n")
            f.write("- **Online Learning:** Incremental updates with Naive Bayes/Logistic Regression\n")
            f.write("- **Batch Processing:** BERT for high-accuracy batch jobs\n")
            f.write("- **Real-time API:** Ensemble with caching for optimal performance\n\n")
            
            # System Metrics
            f.write("## 📏 System Metrics\n\n")
            f.write(f"- **Training Data Size:** {len(self.y_train)} samples\n")
            f.write(f"- **Test Data Size:** {len(self.y_test)} samples\n")
            f.write(f"- **Feature Dimensions:** {self.X_train.shape[1]}\n")
            f.write(f"- **Number of Classes:** {len(np.unique(self.y_train))}\n")
            f.write(f"- **Models Evaluated:** {len(self.results)}\n\n")
            
            # Footer
            f.write("---\n")
            f.write("*Generated by AutoTicketClassifier Ensemble Analysis System*\n")
        
        print(f"📄 Detailed report saved: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Kapsamlı ensemble analizi çalıştır"""
        print("🚀 AUTOTICKETCLASSIFIER ENSEMBLE ANALYSIS")
        print("=" * 60)
        print("Bu analiz 4 farklı model türünü karşılaştırıyor:")
        print("1. 🎯 Naive Bayes (Baseline)")
        print("2. 📈 Logistic Regression (Linear)")
        print("3. 🤖 BERT (Deep Learning)")
        print("4. 🔗 Ensemble Methods (Combination)")
        print("=" * 60)
        
        # Step 1: Data preparation
        df = self.load_or_generate_data()
        self.prepare_features(df)
        
        # Step 2: Model initialization
        self.initialize_models()
        
        # Step 3: Individual model training
        self.train_individual_models()
        
        # Step 4: Ensemble training
        self.train_ensemble_models()
        
        # Step 5: Evaluation
        self.evaluate_all_models()
        
        # Step 6: Cross validation
        cv_results = self.cross_validate_models()
        
        # Step 7: Complexity analysis
        complexity_analysis = self.analyze_model_complexity()
        
        # Step 8: Generate plots
        self.generate_comparison_plots()
        
        # Step 9: Generate report
        report_file = self.generate_detailed_report()
        
        # Summary
        print("\n🎉 ANALYSIS COMPLETE!")
        print("=" * 50)
        
        if self.results:
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            best_accuracy = self.results[best_model]['accuracy']
            print(f"🏆 Best Model: {best_model} ({best_accuracy:.4f} accuracy)")
            
            print(f"📊 Models Evaluated: {len(self.results)}")
            print(f"📈 Performance Range: {min(self.results[m]['accuracy'] for m in self.results):.4f} - {max(self.results[m]['accuracy'] for m in self.results):.4f}")
        
        print(f"📄 Report: {report_file}")
        print(f"📁 Results Directory: {self.results_dir}")
        
        return {
            'results': self.results,
            'cv_results': cv_results,
            'complexity_analysis': complexity_analysis,
            'report_file': report_file
        }

def main():
    """Ana fonksiyon"""
    analyzer = EnsembleAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()
