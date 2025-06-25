"""
📊 Production Monitoring & Model Drift Detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
import logging

class ModelMonitor:
    def __init__(self, baseline_data=None):
        self.baseline_data = baseline_data
        self.prediction_log = []
        self.performance_log = []
        
        # Setup logging
        logging.basicConfig(
            filename='model_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, input_text, prediction, confidence, model_name):
        """Her prediction'ı logla"""
        self.prediction_log.append({
            'timestamp': datetime.now().isoformat(),
            'input_text': input_text,
            'prediction': prediction,
            'confidence': confidence,
            'model_name': model_name,
            'text_length': len(input_text),
            'word_count': len(input_text.split())
        })
        
        self.logger.info(f"Prediction logged: {model_name} -> {prediction} ({confidence:.3f})")
    
    def detect_data_drift(self, new_features, feature_names=None):
        """Statistical drift detection using KS test"""
        if self.baseline_data is None:
            self.logger.warning("No baseline data available for drift detection")
            return None
        
        drift_results = {}
        
        for i in range(min(len(self.baseline_data[0]), len(new_features[0]))):
            baseline_feature = self.baseline_data[:, i]
            new_feature = new_features[:, i]
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(baseline_feature, new_feature)
            
            feature_name = feature_names[i] if feature_names else f"feature_{i}"
            drift_results[feature_name] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05  # %5 significance level
            }
        
        drift_count = sum(1 for r in drift_results.values() if r['drift_detected'])
        total_features = len(drift_results)
        drift_percentage = (drift_count / total_features) * 100
        
        self.logger.info(f"Data drift check: {drift_count}/{total_features} features ({drift_percentage:.1f}%) show drift")
        
        return drift_results
    
    def monitor_performance(self, y_true=None, y_pred=None, batch_size=100):
        """Performance monitoring"""
        if y_true is None or y_pred is None:
            # Simulated performance monitoring (gerçekte human feedback'den gelir)
            accuracy = np.random.uniform(0.85, 0.95)
        else:
            accuracy = accuracy_score(y_true, y_pred)
        
        self.performance_log.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'sample_size': batch_size
        })
        
        # Performance degradation warning
        if len(self.performance_log) > 10:
            recent_performance = [p['accuracy'] for p in self.performance_log[-10:]]
            avg_recent = np.mean(recent_performance)
            
            if avg_recent < 0.80:  # Threshold
                self.logger.warning(f"Performance degradation detected! Average accuracy: {avg_recent:.3f}")
                return "PERFORMANCE_ALERT"
        
        return "OK"
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        report = {
            'report_date': datetime.now().isoformat(),
            'total_predictions': len(self.prediction_log),
            'monitoring_period': '7_days',  # Configurable
        }
        
        if self.prediction_log:
            # Prediction distribution
            predictions = [p['prediction'] for p in self.prediction_log]
            report['prediction_distribution'] = pd.Series(predictions).value_counts().to_dict()
            
            # Confidence statistics
            confidences = [p['confidence'] for p in self.prediction_log]
            report['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
            
            # Input text statistics
            text_lengths = [p['text_length'] for p in self.prediction_log]
            report['input_text_stats'] = {
                'avg_length': np.mean(text_lengths),
                'median_length': np.median(text_lengths)
            }
        
        if self.performance_log:
            accuracies = [p['accuracy'] for p in self.performance_log]
            report['performance_stats'] = {
                'avg_accuracy': np.mean(accuracies),
                'accuracy_trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'declining'
            }
        
        # Save report
        with open(f'monitoring_report_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def setup_alerting(self, webhook_url=None, email=None):
        """Setup alerting for critical issues"""
        # Integration with monitoring tools
        pass

# Usage:
"""
monitor = ModelMonitor(baseline_features)

# During prediction
monitor.log_prediction(text, prediction, confidence, "naive_bayes")

# Periodic drift check
drift_results = monitor.detect_data_drift(new_features, feature_names)

# Performance monitoring
status = monitor.performance_monitoring(y_true, y_pred)

# Generate report
report = monitor.generate_monitoring_report()
"""
