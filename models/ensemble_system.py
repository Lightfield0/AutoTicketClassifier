"""
Model Ensemble System
Multi-model combination system for improved classification performance
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
    """Weighted ensemble classifier for combining multiple models"""
    
    def __init__(self, models=None, weights=None, voting='soft'):
        self.models = models or {}
        self.weights = weights or {}
        self.voting = voting
        self.classes_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y):
        """Train the ensemble model"""
        print("Training ensemble model...")
        
        # Train each base model
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X, y)
        
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        
        # Calculate weights if not provided
        if not self.weights:
            self.weights = self._calculate_weights(X, y)
            
        print("Ensemble training completed!")
        return self
    
    def _calculate_weights(self, X, y, cv=5):
        """Calculate model weights using cross-validation"""
        print("  Calculating model weights...")
        
        weights = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation scoring
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                avg_score = scores.mean()
                weights[name] = avg_score
                print(f"    {name}: {avg_score:.4f}")
            except Exception as e:
                print(f"    Warning: Could not calculate weight for {name}: {e}")
                weights[name] = 0.1  # Default minimal weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        print(f"  Final weights: {weights}")
        return weights
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        if not self.is_fitted_:
            raise ValueError("Model not trained yet!")
        
        # Use different voting strategies depending on setting
        if self.voting == 'hard':
            return self._predict_hard_voting(X)
        else:
            return self._predict_soft_voting(X)
    
    def _predict_hard_voting(self, X):
        """Hard voting prediction"""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions).T
        final_predictions = []
        
        for sample_preds in predictions:
            unique, counts = np.unique(sample_preds, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            final_predictions.append(majority_class)
        
        return np.array(final_predictions)
    
    def _predict_soft_voting(self, X):
        """Soft voting prediction with weighted probabilities"""
        weighted_probs = None
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                elif hasattr(model, 'decision_function'):
                    # Convert decision function to probabilities
                    decision = model.decision_function(X)
                    from sklearn.utils.extmath import softmax
                    probs = softmax(decision.reshape(-1, 1), axis=1)
                else:
                    # This shouldn't happen often, but just in case
                    # some models don't support probability prediction
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
                print(f"Warning: Soft voting failed for {name}: {e}")
                continue
        
        if weighted_probs is None:
            raise ValueError("Soft voting failed for all models!")
        
        # Select class with highest probability
        final_predictions = self.classes_[np.argmax(weighted_probs, axis=1)]
        return final_predictions
    
    def predict_proba(self, X):
        """Return class probabilities"""
        if not self.is_fitted_:
            raise ValueError("Model not trained yet!")
        
        weighted_probs = None
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                else:
                    # Fallback for models without predict_proba
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
                print(f"Warning: Probability calculation failed for {name}: {e}")
                continue
        
        if weighted_probs is None:
            raise ValueError("Probability calculation failed for all models!")
        
        # Normalize probabilities
        row_sums = weighted_probs.sum(axis=1, keepdims=True)
        normalized_probs = weighted_probs / row_sums
        
        return normalized_probs

class EnsembleManager:
    """Ensemble management system for handling multiple model combinations"""
    
    def __init__(self, models_dir="models/ensemble"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_models = {
            'naive_bayes': NaiveBayesClassifier(),
            'logistic_regression': LogisticRegressionClassifier()
        }
        
        self.ensembles = {}
        
    def create_voting_ensemble(self, X_train, y_train, voting='soft', cv_folds=5):
        """Create a voting ensemble classifier"""
        print("Creating voting ensemble...")
        
        # Prepare sklearn-compatible estimators
        estimators = []
        
        for name, model in self.base_models.items():
            # Handle model wrappers
            if hasattr(model, 'model'):
                estimators.append((name, model.model))
            else:
                estimators.append((name, model))
        
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        # Train the ensemble
        voting_ensemble.fit(X_train, y_train)
        
        self.ensembles['voting'] = voting_ensemble
        print("Voting ensemble created successfully!")
        
        return voting_ensemble
    
    def create_weighted_ensemble(self, X_train, y_train, custom_weights=None):
        """Create a weighted ensemble classifier"""
        print("Creating weighted ensemble...")
        
        # Use custom WeightedEnsemble class
        weighted_ensemble = WeightedEnsemble(
            models=self.base_models.copy(),
            weights=custom_weights,
            voting='soft'
        )
        
        weighted_ensemble.fit(X_train, y_train)
        
        self.ensembles['weighted'] = weighted_ensemble
        print("Weighted ensemble created successfully!")
        
        return weighted_ensemble
    
    def create_stacking_ensemble(self, X_train, y_train):
        """Create a stacking ensemble with meta-learner"""
        print("Creating stacking ensemble...")
        
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression as SklearnLR
        
        # Prepare base models
        estimators = []
        for name, model in self.base_models.items():
            if hasattr(model, 'model'):
                estimators.append((name, model.model))
            else:
                estimators.append((name, model))
        
        # This is a simple approach - we could extend this
        # to use more sophisticated meta-learning algorithms
        meta_learner = SklearnLR(random_state=42)
        
        stacking_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5  # 5-fold cross-validation for meta-learner training
        )
        
        stacking_classifier.fit(X_train, y_train)
        
        self.ensembles['stacking'] = stacking_classifier
        print("Stacking ensemble created successfully!")
        
        return stacking_classifier
    
    def compare_ensembles(self, X_test, y_test):
        """Compare performance of different ensemble models"""
        print("\nComparing ensemble models...")
        print("=" * 50)
        
        results = {}
        
        # Include base models and ensembles
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
                print(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Find best performing model
        best_model = max(
            [(name, res) for name, res in results.items() if 'accuracy' in res],
            key=lambda x: x[1]['accuracy']
        )
        
        print(f"\nBest Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        return results
    
    def save_ensemble(self, ensemble_name, model):
        """Save ensemble model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = self.models_dir / f"{ensemble_name}_{timestamp}.joblib"
        
        try:
            joblib.dump(model, model_file)
            print(f"Ensemble saved: {model_file}")
            
            # Save metadata
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
            print(f"Error saving ensemble: {e}")
    
    def load_ensemble(self, model_file):
        """Load ensemble model from disk"""
        try:
            model = joblib.load(model_file)
            print(f"Ensemble loaded: {model_file}")
            return model
        except Exception as e:
            print(f"Error loading ensemble: {e}")
            return None
    
    def generate_ensemble_report(self, results, output_file=None):
        """Generate comprehensive ensemble comparison report"""
        print("\nGenerating ensemble report...")
        
        report_lines = []
        report_lines.append("ENSEMBLE MODEL COMPARISON REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Accuracy comparison
        accuracy_data = []
        for name, result in results.items():
            if 'accuracy' in result:
                accuracy_data.append((name, result['accuracy']))
        
        if accuracy_data:
            accuracy_data.sort(key=lambda x: x[1], reverse=True)
            
            report_lines.append("ACCURACY RESULTS:")
            report_lines.append("-" * 30)
            
            for i, (name, accuracy) in enumerate(accuracy_data, 1):
                rank_symbol = "***" if i == 1 else "**" if i == 2 else "*" if i == 3 else "  "
                report_lines.append(f"{rank_symbol} {i:2d}. {name:20} | {accuracy:.4f}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.append("ENSEMBLE RECOMMENDATIONS:")
        report_lines.append("-" * 25)
        
        if accuracy_data:
            best_model = accuracy_data[0]
            report_lines.append(f"Best performance: {best_model[0]} ({best_model[1]:.4f})")
            
            # Ensemble type recommendations
            if 'voting' in [name for name, _ in accuracy_data]:
                report_lines.append("Voting Ensemble: Simple and effective")
            if 'weighted' in [name for name, _ in accuracy_data]:
                report_lines.append("Weighted Ensemble: Performance-based weighting")
            if 'stacking' in [name for name, _ in accuracy_data]:
                report_lines.append("Stacking Ensemble: Advanced meta-learning approach")
        
        report_lines.append("")
        report_lines.append("USAGE RECOMMENDATIONS:")
        report_lines.append("• Fast inference -> Voting Ensemble")
        report_lines.append("• Maximum accuracy -> Stacking Ensemble")
        report_lines.append("• Balanced performance -> Weighted Ensemble")
        
        # Save report
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.models_dir / f"ensemble_report_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
        print(f"\nEnsemble report saved: {output_file}")
        
        return output_file

def demo_ensemble_system():
    """Demonstration of the ensemble system"""
    print("ENSEMBLE SYSTEM DEMO")
    print("=" * 40)
    
    # Generate demo data
    from data_generator import TicketDataGenerator
    from utils.text_preprocessing import TurkishTextPreprocessor
    from utils.feature_extraction import FeatureExtractor
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    generator = TicketDataGenerator()
    tickets = generator.generate_tickets(num_tickets=1000)
    df = pd.DataFrame(tickets)
    
    # Use 'message' column as 'description'
    df['description'] = df['message']
    
    # Preprocessing
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
    
    # Initialize ensemble manager
    ensemble_manager = EnsembleManager()
    
    # Create different ensemble types for comparison
    print("Building different ensemble types...")
    voting_ensemble = ensemble_manager.create_voting_ensemble(X_train, y_train)
    weighted_ensemble = ensemble_manager.create_weighted_ensemble(X_train, y_train)
    stacking_ensemble = ensemble_manager.create_stacking_ensemble(X_train, y_train)
    
    # Compare models
    results = ensemble_manager.compare_ensembles(X_test, y_test)
    
    # Generate report
    ensemble_manager.generate_ensemble_report(results)
    
    # Save best model
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
