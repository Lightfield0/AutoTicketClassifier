"""
📊 Production Monitoring System
Real-time model monitoring, drift detection ve performance tracking
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score

class ProductionMonitor:
    """Production ortamında model monitoring"""
    
    def __init__(self, db_path="monitoring/production_monitoring.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename='monitoring/model_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Baseline data for drift detection
        self.baseline_data = None
        
    def _init_database(self):
        """Monitoring database'ini initialize et"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_text TEXT,
                prediction TEXT NOT NULL,
                confidence REAL,
                processing_time REAL,
                user_feedback TEXT,
                actual_label TEXT,
                session_id TEXT
            )
        ''')
        
        # Performance metrics tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                data_size INTEGER
            )
        ''')
        
        # Drift detection tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_detection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                drift_detected BOOLEAN NOT NULL,
                drift_score REAL,
                method TEXT,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info("Monitoring database initialized")
    
    def set_baseline_data(self, baseline_features):
        """Drift detection için baseline data'yı set et"""
        self.baseline_data = baseline_features
        self.logger.info(f"Baseline data set with shape: {baseline_features.shape}")
    
    def log_prediction(self, model_name, input_text, prediction, confidence=None, 
                      processing_time=None, session_id=None):
        """Prediction'ı logla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, model_name, input_text, prediction, confidence, processing_time, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            model_name,
            input_text[:500],  # Limit text length
            prediction,
            confidence,
            processing_time,
            session_id
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Prediction logged: {model_name} -> {prediction}")
    
    def log_performance_metric(self, model_name, metric_name, metric_value, data_size=None):
        """Performance metric'i logla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, model_name, metric_name, metric_value, data_size)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            model_name,
            metric_name,
            metric_value,
            data_size
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Performance metric logged: {model_name} {metric_name} = {metric_value}")
    
    def detect_data_drift(self, new_features, method='ks_test', threshold=0.05):
        """Data drift detection"""
        if self.baseline_data is None:
            self.logger.warning("No baseline data available for drift detection")
            return None
        
        drift_results = {
            'method': method,
            'threshold': threshold,
            'features_with_drift': [],
            'overall_drift_detected': False,
            'drift_scores': {}
        }
        
        if method == 'ks_test':
            for i in range(min(self.baseline_data.shape[1], new_features.shape[1])):
                baseline_feature = self.baseline_data[:, i]
                new_feature = new_features[:, i]
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(baseline_feature, new_feature)
                
                drift_results['drift_scores'][f'feature_{i}'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
                
                if p_value < threshold:
                    drift_results['features_with_drift'].append(f'feature_{i}')
        
        elif method == 'psi':
            # Population Stability Index
            psi_threshold = 0.1
            
            for i in range(min(self.baseline_data.shape[1], new_features.shape[1])):
                baseline_feature = self.baseline_data[:, i]
                new_feature = new_features[:, i]
                
                psi_score = self._calculate_psi(baseline_feature, new_feature)
                
                drift_results['drift_scores'][f'feature_{i}'] = {
                    'psi_score': psi_score,
                    'drift_detected': psi_score > psi_threshold
                }
                
                if psi_score > psi_threshold:
                    drift_results['features_with_drift'].append(f'feature_{i}')
        
        drift_results['overall_drift_detected'] = len(drift_results['features_with_drift']) > 0
        
        # Log drift detection
        self._log_drift_detection(
            drift_results['overall_drift_detected'],
            len(drift_results['features_with_drift']),
            method,
            json.dumps({
                'features_with_drift': drift_results['features_with_drift'],
                'total_features': len(drift_results['drift_scores']),
                'method': method
            })
        )
        
        return drift_results
    
    def _calculate_psi(self, baseline, new_data, buckets=10):
        """Population Stability Index hesaplama"""
        # Create bins
        min_val = min(baseline.min(), new_data.min())
        max_val = max(baseline.max(), new_data.max())
        
        bins = np.linspace(min_val, max_val, buckets + 1)
        
        # Calculate distributions
        baseline_dist = np.histogram(baseline, bins=bins)[0] / len(baseline)
        new_dist = np.histogram(new_data, bins=bins)[0] / len(new_data)
        
        # Add small epsilon to avoid division by zero
        baseline_dist = np.where(baseline_dist == 0, 1e-10, baseline_dist)
        new_dist = np.where(new_dist == 0, 1e-10, new_dist)
        
        # Calculate PSI
        psi = np.sum((new_dist - baseline_dist) * np.log(new_dist / baseline_dist))
        
        return psi
    
    def _log_drift_detection(self, drift_detected, drift_score, method, details):
        """Drift detection sonucunu logla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drift_detection 
            (timestamp, drift_detected, drift_score, method, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            drift_detected,
            drift_score,
            method,
            details
        ))
        
        conn.commit()
        conn.close()
        
        if drift_detected:
            self.logger.warning(f"Data drift detected! Score: {drift_score}, Method: {method}")
    
    def get_performance_history(self, model_name=None, hours=24):
        """Performance history'yi getir"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT * FROM performance_metrics 
            WHERE timestamp > ?
        '''
        params = [cutoff_time.isoformat()]
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        query += ' ORDER BY timestamp DESC'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_prediction_history(self, model_name=None, hours=24):
        """Prediction history'yi getir"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT * FROM predictions 
            WHERE timestamp > ?
        '''
        params = [cutoff_time.isoformat()]
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        query += ' ORDER BY timestamp DESC'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_drift_history(self, hours=24):
        """Drift detection history'yi getir"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT * FROM drift_detection 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[cutoff_time.isoformat()])
        conn.close()
        
        return df
    
    def generate_monitoring_report(self, hours=24):
        """Comprehensive monitoring raporu oluştur"""
        report = {
            'report_date': datetime.now().isoformat(),
            'monitoring_period_hours': hours,
            'predictions': {},
            'performance': {},
            'drift': {},
            'alerts': []
        }
        
        # Prediction statistics
        pred_df = self.get_prediction_history(hours=hours)
        if len(pred_df) > 0:
            report['predictions'] = {
                'total_predictions': len(pred_df),
                'unique_models': pred_df['model_name'].nunique(),
                'avg_confidence': pred_df['confidence'].mean() if 'confidence' in pred_df else None,
                'avg_processing_time': pred_df['processing_time'].mean() if 'processing_time' in pred_df else None,
                'prediction_distribution': pred_df['prediction'].value_counts().to_dict()
            }
        
        # Performance statistics
        perf_df = self.get_performance_history(hours=hours)
        if len(perf_df) > 0:
            report['performance'] = {
                'metrics_logged': len(perf_df),
                'latest_metrics': perf_df.groupby(['model_name', 'metric_name'])['metric_value'].last().to_dict()
            }
        
        # Drift statistics
        drift_df = self.get_drift_history(hours=hours)
        if len(drift_df) > 0:
            report['drift'] = {
                'drift_checks': len(drift_df),
                'drift_detected_count': drift_df['drift_detected'].sum(),
                'latest_drift_status': drift_df.iloc[0]['drift_detected'] if len(drift_df) > 0 else False
            }
        
        # Generate alerts
        report['alerts'] = self._generate_alerts(report)
        
        # Save report
        report_path = f"monitoring/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Monitoring report generated: {report_path}")
        
        return report
    
    def _generate_alerts(self, report):
        """Alerts oluştur"""
        alerts = []
        
        # Performance degradation alert
        if 'performance' in report and 'latest_metrics' in report['performance']:
            for metric_key, value in report['performance']['latest_metrics'].items():
                if 'accuracy' in metric_key.lower() and value < 0.8:
                    alerts.append({
                        'type': 'performance_degradation',
                        'message': f"Low accuracy detected: {value:.3f}",
                        'severity': 'high'
                    })
        
        # Data drift alert
        if 'drift' in report and report['drift'].get('latest_drift_status', False):
            alerts.append({
                'type': 'data_drift',
                'message': "Data drift detected in recent predictions",
                'severity': 'medium'
            })
        
        # Low prediction volume alert
        if 'predictions' in report and report['predictions'].get('total_predictions', 0) < 10:
            alerts.append({
                'type': 'low_volume',
                'message': f"Low prediction volume: {report['predictions']['total_predictions']}",
                'severity': 'low'
            })
        
        return alerts
    
    def health_check(self):
        """System health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'database_accessible': True,
            'recent_activity': False,
            'alerts': []
        }
        
        try:
            # Check database accessibility
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp > ?", 
                         [(datetime.now() - timedelta(hours=1)).isoformat()])
            recent_predictions = cursor.fetchone()[0]
            conn.close()
            
            health_status['recent_activity'] = recent_predictions > 0
            
        except Exception as e:
            health_status['database_accessible'] = False
            health_status['alerts'].append(f"Database error: {str(e)}")
            self.logger.error(f"Health check failed: {e}")
        
        return health_status

# Initialize global monitor
def get_production_monitor():
    """Global production monitor instance"""
    return ProductionMonitor()
