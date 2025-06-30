#!/usr/bin/env python3
"""
🔬 AutoTicketClassifier: Comprehensive Model Analysis & Stress Testing
Bu script 4 farklı modeli gerçek dünya koşullarında test eder ve detaylı performans analizi yapar.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelStressTester:
    """Model stress testing ve analiz sistemi"""
    
    def __init__(self):
        self.results = {}
        
    def test_scenario_1_normal_conditions(self):
        """Senaryo 1: Normal koşullarda model performansı"""
        print("🔬 SENARYO 1: Normal Koşullar")
        print("-" * 50)
        
        # Normal training ve test
        from data_generator import TicketDataGenerator
        from utils.text_preprocessing import TurkishTextPreprocessor
        from utils.feature_extraction import FeatureExtractor
        
        # Generate balanced data
        generator = TicketDataGenerator()
        tickets = generator.generate_tickets(num_tickets=1000)  # Smaller dataset
        df = pd.DataFrame(tickets)
        
        # Preprocessing
        preprocessor = TurkishTextPreprocessor()
        df['processed_text'] = df['message'].apply(preprocessor.preprocess_text)
        
        # Feature extraction
        extractor = FeatureExtractor()
        X, feature_names = extractor.extract_tfidf_features(
            df['processed_text'], 
            max_features=500  # Reduced features
        )
        y = df['category']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return self._evaluate_models(X_train, X_test, y_train, y_test, "Normal")
    
    def test_scenario_2_noise_injection(self):
        """Senaryo 2: Gürültülü veri ile test"""
        print("🔬 SENARYO 2: Gürültülü Veri")
        print("-" * 50)
        
        from data_generator import TicketDataGenerator
        from utils.text_preprocessing import TurkishTextPreprocessor
        from utils.feature_extraction import FeatureExtractor
        
        # Generate data
        generator = TicketDataGenerator()
        tickets = generator.generate_tickets(num_tickets=1000)
        df = pd.DataFrame(tickets)
        
        # Add noise to text data
        def add_noise(text):
            import random
            import string
            
            noise_chars = random.sample(string.ascii_lowercase, 3)
            noise_words = [''.join(noise_chars) for _ in range(2)]
            return text + " " + " ".join(noise_words)
        
        # Apply noise to 30% of data
        noise_indices = np.random.choice(len(df), size=int(0.3 * len(df)), replace=False)
        df.loc[noise_indices, 'message'] = df.loc[noise_indices, 'message'].apply(add_noise)
        
        # Preprocessing
        preprocessor = TurkishTextPreprocessor()
        df['processed_text'] = df['message'].apply(preprocessor.preprocess_text)
        
        # Feature extraction
        extractor = FeatureExtractor()
        X, _ = extractor.extract_tfidf_features(df['processed_text'], max_features=500)
        y = df['category']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return self._evaluate_models(X_train, X_test, y_train, y_test, "Noisy")
    
    def test_scenario_3_imbalanced_data(self):
        """Senaryo 3: Dengesiz veri dağılımı"""
        print("🔬 SENARYO 3: Dengesiz Veri")
        print("-" * 50)
        
        from data_generator import TicketDataGenerator
        from utils.text_preprocessing import TurkishTextPreprocessor
        from utils.feature_extraction import FeatureExtractor
        
        # Generate normal data first
        generator = TicketDataGenerator()
        tickets = generator.generate_tickets(num_tickets=1000)
        df = pd.DataFrame(tickets)
        
        # Create imbalanced distribution by filtering
        category_samples = {
            'technical_issue': 500,    # 50%
            'general_info': 300,       # 30%
            'complaint': 100,          # 10%
            'payment_issue': 50,       # 5%
            'user_error': 30,          # 3%
            'reservation_problem': 20   # 2%
        }
        
        # Sample from each category
        imbalanced_dfs = []
        for category, n_samples in category_samples.items():
            category_df = df[df['category'] == category].sample(
                n=min(n_samples, len(df[df['category'] == category])), 
                random_state=42
            )
            imbalanced_dfs.append(category_df)
        
        df = pd.concat(imbalanced_dfs, ignore_index=True)
        
        # Preprocessing
        preprocessor = TurkishTextPreprocessor()
        df['processed_text'] = df['message'].apply(preprocessor.preprocess_text)
        
        # Feature extraction
        extractor = FeatureExtractor()
        X, _ = extractor.extract_tfidf_features(df['processed_text'], max_features=500)
        y = df['category']
        
        # Train-test split with stratification (will handle imbalance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return self._evaluate_models(X_train, X_test, y_train, y_test, "Imbalanced")
    
    def test_scenario_4_limited_training_data(self):
        """Senaryo 4: Sınırlı eğitim verisi"""
        print("🔬 SENARYO 4: Sınırlı Eğitim Verisi")
        print("-" * 50)
        
        from data_generator import TicketDataGenerator
        from utils.text_preprocessing import TurkishTextPreprocessor
        from utils.feature_extraction import FeatureExtractor
        
        # Generate small dataset
        generator = TicketDataGenerator()
        tickets = generator.generate_tickets(num_tickets=200)  # Very small dataset
        df = pd.DataFrame(tickets)
        
        # Preprocessing
        preprocessor = TurkishTextPreprocessor()
        df['processed_text'] = df['message'].apply(preprocessor.preprocess_text)
        
        # Feature extraction
        extractor = FeatureExtractor()
        X, _ = extractor.extract_tfidf_features(df['processed_text'], max_features=200)
        y = df['category']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return self._evaluate_models(X_train, X_test, y_train, y_test, "Limited")
    
    def _evaluate_models(self, X_train, X_test, y_train, y_test, scenario_name):
        """Model evaluation helper"""
        from models.naive_bayes import NaiveBayesClassifier
        from models.logistic_regression import LogisticRegressionClassifier
        from models.ensemble_system import WeightedEnsemble
        
        models = {
            'naive_bayes': NaiveBayesClassifier(model_type='multinomial'),
            'logistic_regression': LogisticRegressionClassifier()
        }
        
        scenario_results = {}
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Classes: {np.unique(y_train)}")
        
        # Train individual models
        for name, model in models.items():
            try:
                print(f"🚀 Training {name}...")
                model.train(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Classification report
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                scenario_results[name] = {
                    'accuracy': accuracy,
                    'weighted_f1': class_report['weighted avg']['f1-score'],
                    'macro_f1': class_report['macro avg']['f1-score'],
                    'predictions': y_pred
                }
                
                print(f"  ✅ {name}: Accuracy={accuracy:.4f}, F1={class_report['weighted avg']['f1-score']:.4f}")
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                scenario_results[name] = {'error': str(e)}
        
        # Train ensemble if we have enough models
        if len([m for m in scenario_results.values() if 'error' not in m]) >= 2:
            try:
                print("🤖 Training Ensemble...")
                
                # Create ensemble
                ensemble_models = {}
                for name, model in models.items():
                    if name in scenario_results and 'error' not in scenario_results[name]:
                        ensemble_models[name] = model.model if hasattr(model, 'model') else model
                
                ensemble = WeightedEnsemble(models=ensemble_models, voting='soft')
                ensemble.fit(X_train, y_train)
                
                # Evaluate ensemble
                y_pred_ensemble = ensemble.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_ensemble)
                class_report = classification_report(y_test, y_pred_ensemble, output_dict=True)
                
                scenario_results['ensemble'] = {
                    'accuracy': accuracy,
                    'weighted_f1': class_report['weighted avg']['f1-score'],
                    'macro_f1': class_report['macro avg']['f1-score'],
                    'predictions': y_pred_ensemble
                }
                
                print(f"  ✅ ensemble: Accuracy={accuracy:.4f}, F1={class_report['weighted avg']['f1-score']:.4f}")
                
            except Exception as e:
                print(f"  ❌ ensemble failed: {e}")
        
        self.results[scenario_name] = scenario_results
        return scenario_results
    
    def generate_comparison_report(self):
        """Tüm senaryoların karşılaştırma raporu"""
        print("\n📊 COMPREHENSIVE STRESS TEST RESULTS")
        print("=" * 60)
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Create comparison table
        comparison_data = []
        
        for scenario, results in self.results.items():
            for model, metrics in results.items():
                if 'error' not in metrics:
                    comparison_data.append({
                        'Scenario': scenario,
                        'Model': model,
                        'Accuracy': metrics['accuracy'],
                        'Weighted_F1': metrics['weighted_f1'],
                        'Macro_F1': metrics['macro_f1']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print results table
        print("\\n📋 RESULTS TABLE:")
        print("-" * 80)
        print(f"{'Scenario':<12} {'Model':<20} {'Accuracy':<10} {'Weighted F1':<12} {'Macro F1':<10}")
        print("-" * 80)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['Scenario']:<12} {row['Model']:<20} {row['Accuracy']:<10.4f} {row['Weighted_F1']:<12.4f} {row['Macro_F1']:<10.4f}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔬 AutoTicketClassifier: Stress Test Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by scenario
        scenario_accuracy = comparison_df.groupby(['Scenario', 'Model'])['Accuracy'].mean().unstack()
        scenario_accuracy.plot(kind='bar', ax=axes[0,0], title='📊 Accuracy by Scenario')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend(title='Model')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. F1-Score comparison
        scenario_f1 = comparison_df.groupby(['Scenario', 'Model'])['Weighted_F1'].mean().unstack()
        scenario_f1.plot(kind='bar', ax=axes[0,1], title='📈 Weighted F1-Score by Scenario')
        axes[0,1].set_ylabel('Weighted F1-Score')
        axes[0,1].legend(title='Model')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Model robustness (std across scenarios)
        model_robustness = comparison_df.groupby('Model')['Accuracy'].agg(['mean', 'std'])
        axes[1,0].bar(model_robustness.index, model_robustness['mean'], 
                     yerr=model_robustness['std'], capsize=5)
        axes[1,0].set_title('🛡️ Model Robustness (Mean ± Std)')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Degradation analysis
        normal_scores = comparison_df[comparison_df['Scenario'] == 'Normal']['Accuracy'].values
        degradation_data = []
        
        for scenario in ['Noisy', 'Imbalanced', 'Limited']:
            if scenario in comparison_df['Scenario'].values:
                scenario_scores = comparison_df[comparison_df['Scenario'] == scenario]['Accuracy'].values
                if len(scenario_scores) > 0 and len(normal_scores) > 0:
                    degradation = (normal_scores[0] - scenario_scores[0]) * 100
                    degradation_data.append({'Scenario': scenario, 'Degradation': degradation})
        
        if degradation_data:
            deg_df = pd.DataFrame(degradation_data)
            axes[1,1].bar(deg_df['Scenario'], deg_df['Degradation'])
            axes[1,1].set_title('📉 Performance Degradation (%)')
            axes[1,1].set_ylabel('Accuracy Drop (%)')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        from pathlib import Path
        results_dir = Path("evaluation_results/stress_testing")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = results_dir / f"stress_test_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\\n📊 Stress test plots saved: {plot_file}")
        
        # Generate detailed report
        report_file = results_dir / f"stress_test_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 🔬 AutoTicketClassifier: Stress Test Analysis Report\\n\\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## 📋 Executive Summary\\n\\n")
            f.write("Bu rapor AutoTicketClassifier modellerinin çeşitli stres koşullarında performans analizini içermektedir.\\n\\n")
            
            f.write("## 🧪 Test Scenarios\\n\\n")
            f.write("1. **Normal Conditions:** Standart eğitim ve test koşulları\\n")
            f.write("2. **Noisy Data:** %30 gürültülü veri ile test\\n")
            f.write("3. **Imbalanced Data:** Dengesiz sınıf dağılımı\\n")
            f.write("4. **Limited Training Data:** Sınırlı eğitim verisi (200 sample)\\n\\n")
            
            f.write("## 📊 Results Summary\\n\\n")
            f.write("| Scenario | Model | Accuracy | Weighted F1 | Macro F1 |\\n")
            f.write("|----------|-------|----------|-------------|----------|\\n")
            
            for _, row in comparison_df.iterrows():
                f.write(f"| {row['Scenario']} | {row['Model']} | {row['Accuracy']:.4f} | {row['Weighted_F1']:.4f} | {row['Macro_F1']:.4f} |\\n")
            
            f.write("\\n## 🏆 Best Performing Models\\n\\n")
            
            # Best model per scenario
            for scenario in comparison_df['Scenario'].unique():
                scenario_data = comparison_df[comparison_df['Scenario'] == scenario]
                best_model = scenario_data.loc[scenario_data['Accuracy'].idxmax()]
                f.write(f"- **{scenario}:** {best_model['Model']} ({best_model['Accuracy']:.4f} accuracy)\\n")
            
            f.write("\\n## 💡 Insights & Recommendations\\n\\n")
            
            # Model robustness analysis
            model_stats = comparison_df.groupby('Model')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
            most_robust = model_stats['std'].idxmin()
            highest_avg = model_stats['mean'].idxmax()
            
            f.write(f"### 🛡️ Model Robustness\\n")
            f.write(f"- **Most Consistent:** {most_robust} (lowest std: {model_stats.loc[most_robust, 'std']:.4f})\\n")
            f.write(f"- **Highest Average:** {highest_avg} (avg: {model_stats.loc[highest_avg, 'mean']:.4f})\\n\\n")
            
            f.write("### 🎯 Production Recommendations\\n")
            f.write("- For **stable performance:** Use ensemble methods\\n")
            f.write("- For **resource constraints:** Use Naive Bayes\\n")
            f.write("- For **maximum accuracy:** Use Logistic Regression\\n")
            f.write("- For **noisy environments:** Implement data cleaning pipeline\\n\\n")
            
            f.write("---\\n")
            f.write("*Generated by AutoTicketClassifier Stress Testing System*\\n")
        
        print(f"📄 Detailed stress test report saved: {report_file}")
        
        return {
            'comparison_df': comparison_df,
            'plot_file': plot_file,
            'report_file': report_file
        }
    
    def run_all_tests(self):
        """Tüm stress testleri çalıştır"""
        print("🚀 AUTOTICKETCLASSIFIER STRESS TESTING SUITE")
        print("=" * 60)
        print("Bu test suite modelleri 4 farklı stres senaryosunda test eder:")
        print("1. 📊 Normal Conditions")
        print("2. 🌪️ Noisy Data")
        print("3. ⚖️ Imbalanced Data")
        print("4. 📉 Limited Training Data")
        print("=" * 60)
        
        # Run all scenarios
        self.test_scenario_1_normal_conditions()
        self.test_scenario_2_noise_injection()
        self.test_scenario_3_imbalanced_data()
        self.test_scenario_4_limited_training_data()
        
        # Generate comprehensive report
        summary = self.generate_comparison_report()
        
        print("\\n🎉 STRESS TESTING COMPLETE!")
        print("=" * 50)
        print(f"📊 Scenarios Tested: {len(self.results)}")
        print(f"📈 Models Evaluated: {len(set(model for results in self.results.values() for model in results.keys()))}")
        print(f"📄 Report: {summary['report_file']}")
        print(f"📊 Plots: {summary['plot_file']}")
        
        return summary

def main():
    """Ana fonksiyon"""
    tester = ModelStressTester()
    results = tester.run_all_tests()
    return results

if __name__ == "__main__":
    main()
