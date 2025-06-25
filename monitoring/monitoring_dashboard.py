"""
📊 Real-time Model Monitoring Dashboard
Model drift detection, performance monitoring, alerting sistemi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class ModelDriftDetector:
    """Model drift detection sistemi"""
    
    def __init__(self, reference_data=None, threshold=0.05):
        """
        Args:
            reference_data: Referans veri seti (training data)
            threshold: Drift detection threshold (p-value)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.drift_history = []
        
    def detect_data_drift(self, new_data, method='ks_test'):
        """
        Veri dağılımında drift detect et
        
        Args:
            new_data: Yeni gelen veri
            method: 'ks_test', 'chi2_test', 'psi'
        """
        if self.reference_data is None:
            raise ValueError("Referans veri seti tanımlanmamış!")
        
        drift_results = {}
        
        if method == 'ks_test':
            drift_results = self._ks_test_drift(new_data)
        elif method == 'chi2_test':
            drift_results = self._chi2_test_drift(new_data)
        elif method == 'psi':
            drift_results = self._psi_drift(new_data)
        
        # Drift history'ye ekle
        drift_results['timestamp'] = datetime.now()
        drift_results['method'] = method
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _ks_test_drift(self, new_data):
        """Kolmogorov-Smirnov test ile drift detection"""
        drift_detected = False
        feature_drifts = {}
        
        # Her feature için KS test
        for i in range(min(self.reference_data.shape[1], new_data.shape[1])):
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_feature, new_feature)
            
            feature_drift = p_value < self.threshold
            drift_detected = drift_detected or feature_drift
            
            feature_drifts[f'feature_{i}'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': feature_drift
            }
        
        return {
            'drift_detected': drift_detected,
            'feature_drifts': feature_drifts,
            'overall_drift_score': np.mean([fd['ks_statistic'] for fd in feature_drifts.values()])
        }
    
    def _psi_drift(self, new_data):
        """Population Stability Index ile drift detection"""
        # PSI hesaplama
        psi_scores = []
        
        for i in range(min(self.reference_data.shape[1], new_data.shape[1])):
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            # Binning
            bins = np.linspace(min(ref_feature.min(), new_feature.min()),
                             max(ref_feature.max(), new_feature.max()), 10)
            
            ref_counts, _ = np.histogram(ref_feature, bins=bins)
            new_counts, _ = np.histogram(new_feature, bins=bins)
            
            # Normalize
            ref_pct = ref_counts / ref_counts.sum()
            new_pct = new_counts / new_counts.sum()
            
            # PSI calculation
            psi = 0
            for j in range(len(ref_pct)):
                if ref_pct[j] > 0 and new_pct[j] > 0:
                    psi += (new_pct[j] - ref_pct[j]) * np.log(new_pct[j] / ref_pct[j])
            
            psi_scores.append(psi)
        
        avg_psi = np.mean(psi_scores)
        
        # PSI thresholds
        if avg_psi < 0.1:
            drift_level = "No drift"
        elif avg_psi < 0.2:
            drift_level = "Small drift"
        else:
            drift_level = "Significant drift"
        
        return {
            'drift_detected': avg_psi > 0.1,
            'psi_score': avg_psi,
            'drift_level': drift_level,
            'feature_psi_scores': psi_scores
        }

class PerformanceMonitor:
    """Model performance monitoring"""
    
    def __init__(self, db_path="monitoring.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Monitoring database'ini initialize et"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_name TEXT,
                prediction TEXT,
                confidence REAL,
                actual_label TEXT,
                is_correct INTEGER,
                processing_time REAL
            )
        """)
        
        # Performance metrics tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_name TEXT,
                metric_name TEXT,
                metric_value REAL,
                data_size INTEGER
            )
        """)
        
        # Drift detection results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                drift_detected INTEGER,
                drift_score REAL,
                method TEXT,
                details TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, model_name, prediction, confidence=None, 
                      actual_label=None, processing_time=None):
        """Prediction'ı logla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        is_correct = None
        if actual_label is not None:
            is_correct = 1 if prediction == actual_label else 0
        
        cursor.execute("""
            INSERT INTO predictions 
            (timestamp, model_name, prediction, confidence, actual_label, is_correct, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(), model_name, prediction, confidence, 
              actual_label, is_correct, processing_time))
        
        conn.commit()
        conn.close()
    
    def log_performance_metric(self, model_name, metric_name, metric_value, data_size=None):
        """Performance metric'i logla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics 
            (timestamp, model_name, metric_name, metric_value, data_size)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now(), model_name, metric_name, metric_value, data_size))
        
        conn.commit()
        conn.close()
    
    def log_drift_detection(self, drift_detected, drift_score, method, details=None):
        """Drift detection sonucunu logla"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO drift_detections 
            (timestamp, drift_detected, drift_score, method, details)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now(), drift_detected, drift_score, method, 
              json.dumps(details) if details else None))
        
        conn.commit()
        conn.close()
    
    def get_performance_history(self, model_name=None, hours=24):
        """Performance history'sini al"""
        conn = sqlite3.connect(self.db_path)
        
        since_time = datetime.now() - timedelta(hours=hours)
        
        if model_name:
            query = """
                SELECT * FROM performance_metrics 
                WHERE model_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(model_name, since_time))
        else:
            query = """
                SELECT * FROM performance_metrics 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(since_time,))
        
        conn.close()
        return df
    
    def get_prediction_history(self, model_name=None, hours=24):
        """Prediction history'sini al"""
        conn = sqlite3.connect(self.db_path)
        
        since_time = datetime.now() - timedelta(hours=hours)
        
        if model_name:
            query = """
                SELECT * FROM predictions 
                WHERE model_name = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(model_name, since_time))
        else:
            query = """
                SELECT * FROM predictions 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(since_time,))
        
        conn.close()
        return df

class MonitoringDashboard:
    """Interactive monitoring dashboard"""
    
    def __init__(self, monitor: PerformanceMonitor, drift_detector: ModelDriftDetector):
        self.monitor = monitor
        self.drift_detector = drift_detector
        
    def create_performance_dashboard(self, model_name=None, hours=24):
        """Performance monitoring dashboard'u oluştur"""
        print("📊 Performance Dashboard oluşturuluyor...")
        
        # Veri al
        perf_df = self.monitor.get_performance_history(model_name, hours)
        pred_df = self.monitor.get_prediction_history(model_name, hours)
        
        if perf_df.empty and pred_df.empty:
            print("⚠️ Monitoring data bulunamadı!")
            return
        
        # Subplot oluştur
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Over Time', 'Prediction Confidence Distribution',
                          'Processing Time Trend', 'Prediction Volume'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Accuracy over time
        if not perf_df.empty:
            accuracy_df = perf_df[perf_df['metric_name'] == 'accuracy']
            if not accuracy_df.empty:
                fig.add_trace(
                    go.Scatter(x=accuracy_df['timestamp'], y=accuracy_df['metric_value'],
                             mode='lines+markers', name='Accuracy'),
                    row=1, col=1
                )
        
        # 2. Confidence distribution
        if not pred_df.empty and 'confidence' in pred_df.columns:
            fig.add_trace(
                go.Histogram(x=pred_df['confidence'], name='Confidence Distribution',
                           nbinsx=20),
                row=1, col=2
            )
        
        # 3. Processing time trend
        if not pred_df.empty and 'processing_time' in pred_df.columns:
            pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
            pred_df_grouped = pred_df.groupby(pred_df['timestamp'].dt.floor('H'))['processing_time'].mean()
            
            fig.add_trace(
                go.Scatter(x=pred_df_grouped.index, y=pred_df_grouped.values,
                         mode='lines+markers', name='Avg Processing Time'),
                row=2, col=1
            )
        
        # 4. Prediction volume
        if not pred_df.empty:
            pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
            pred_volume = pred_df.groupby(pred_df['timestamp'].dt.floor('H')).size()
            
            fig.add_trace(
                go.Bar(x=pred_volume.index, y=pred_volume.values,
                      name='Predictions per Hour'),
                row=2, col=2
            )
        
        # Layout
        fig.update_layout(
            title=f"Model Performance Dashboard - {model_name or 'All Models'}",
            height=800,
            showlegend=True
        )
        
        # Save as HTML
        dashboard_path = Path("monitoring_dashboard.html")
        fig.write_html(dashboard_path)
        print(f"💾 Dashboard kaydedildi: {dashboard_path}")
        
        # Show
        fig.show()
        
        return fig
    
    def create_drift_dashboard(self):
        """Drift detection dashboard'u oluştur"""
        print("📈 Drift Detection Dashboard oluşturuluyor...")
        
        if not self.drift_detector.drift_history:
            print("⚠️ Drift detection history bulunamadı!")
            return
        
        # Drift history'den DataFrame oluştur
        drift_data = []
        for record in self.drift_detector.drift_history:
            drift_data.append({
                'timestamp': record['timestamp'],
                'drift_detected': record['drift_detected'],
                'drift_score': record.get('overall_drift_score', 0),
                'method': record['method']
            })
        
        df = pd.DataFrame(drift_data)
        
        # Plotly dashboard with pie chart support
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Drift Score Over Time', 'Drift Detection Frequency',
                          'Drift Methods Used', 'Drift Alert Timeline'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Drift score over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['drift_score'],
                     mode='lines+markers', name='Drift Score'),
            row=1, col=1
        )
        
        # 2. Drift detection frequency
        drift_counts = df['drift_detected'].value_counts()
        fig.add_trace(
            go.Bar(x=['No Drift', 'Drift Detected'], 
                  y=[drift_counts.get(False, 0), drift_counts.get(True, 0)],
                  name='Detection Frequency'),
            row=1, col=2
        )
        
        # 3. Methods used
        method_counts = df['method'].value_counts()
        fig.add_trace(
            go.Pie(labels=method_counts.index, values=method_counts.values,
                  name='Methods Used'),
            row=2, col=1
        )
        
        # 4. Alert timeline
        alerts = df[df['drift_detected'] == True]
        if not alerts.empty:
            fig.add_trace(
                go.Scatter(x=alerts['timestamp'], y=alerts['drift_score'],
                         mode='markers', marker=dict(size=15, color='red'),
                         name='Drift Alerts'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Model Drift Detection Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save to monitoring_outputs directory
        output_dir = Path("monitoring_outputs")
        output_dir.mkdir(exist_ok=True)
        drift_dashboard_path = output_dir / "drift_dashboard.html"
        fig.write_html(drift_dashboard_path)
        print(f"💾 Drift dashboard kaydedildi: {drift_dashboard_path}")
        
        fig.show()
        
        return fig
    
    def generate_monitoring_report(self, hours=24):
        """Kapsamlı monitoring raporu oluştur"""
        print("📋 Monitoring raporu oluşturuluyor...")
        
        # Veri al
        perf_df = self.monitor.get_performance_history(hours=hours)
        pred_df = self.monitor.get_prediction_history(hours=hours)
        
        report = []
        report.append("📊 MODEL MONITORING RAPORU")
        report.append("=" * 50)
        report.append(f"📅 Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"⏰ Son {hours} saat analizi")
        report.append("")
        
        # Performance özeti
        if not perf_df.empty:
            report.append("📈 PERFORMANCE ÖZETİ")
            report.append("-" * 25)
            
            accuracy_df = perf_df[perf_df['metric_name'] == 'accuracy']
            if not accuracy_df.empty:
                latest_accuracy = accuracy_df.iloc[0]['metric_value']
                avg_accuracy = accuracy_df['metric_value'].mean()
                report.append(f"🎯 Güncel Accuracy: {latest_accuracy:.4f}")
                report.append(f"📊 Ortalama Accuracy: {avg_accuracy:.4f}")
            
            report.append("")
        
        # Prediction özeti
        if not pred_df.empty:
            report.append("🔮 PREDICTION ÖZETİ")
            report.append("-" * 25)
            
            total_predictions = len(pred_df)
            report.append(f"📊 Toplam Prediction: {total_predictions}")
            
            if 'is_correct' in pred_df.columns:
                correct_predictions = pred_df['is_correct'].sum()
                if total_predictions > 0:
                    accuracy = correct_predictions / total_predictions
                    report.append(f"✅ Doğru Prediction: {correct_predictions}/{total_predictions} ({accuracy:.4f})")
            
            if 'processing_time' in pred_df.columns:
                avg_processing_time = pred_df['processing_time'].mean()
                report.append(f"⏱️ Ortalama İşlem Süresi: {avg_processing_time:.4f}s")
            
            report.append("")
        
        # Drift özeti
        if self.drift_detector.drift_history:
            report.append("📈 DRIFT DETECTION ÖZETİ")
            report.append("-" * 30)
            
            recent_drifts = [d for d in self.drift_detector.drift_history 
                           if d['timestamp'] > datetime.now() - timedelta(hours=hours)]
            
            drift_count = sum(1 for d in recent_drifts if d['drift_detected'])
            total_checks = len(recent_drifts)
            
            report.append(f"🔍 Drift Kontrolü: {total_checks} kez")
            report.append(f"⚠️ Drift Tespit Edilen: {drift_count}")
            
            if recent_drifts:
                latest_drift = recent_drifts[-1]
                report.append(f"📊 Son Drift Score: {latest_drift.get('overall_drift_score', 0):.4f}")
                report.append(f"🔧 Son Kullanılan Method: {latest_drift['method']}")
            
            report.append("")
        
        # Öneriler
        report.append("💡 ÖNERİLER")
        report.append("-" * 15)
        
        if not perf_df.empty:
            accuracy_df = perf_df[perf_df['metric_name'] == 'accuracy']
            if not accuracy_df.empty:
                latest_accuracy = accuracy_df.iloc[0]['metric_value']
                if latest_accuracy < 0.8:
                    report.append("🚨 Accuracy çok düşük - Model yeniden eğitimi gerekli")
                elif latest_accuracy < 0.9:
                    report.append("⚠️ Accuracy izlenmeli - Hyperparameter tuning önerilir")
                else:
                    report.append("✅ Model performansı iyi")
        
        if self.drift_detector.drift_history:
            recent_drift_detected = any(d['drift_detected'] for d in self.drift_detector.drift_history[-5:])
            if recent_drift_detected:
                report.append("🚨 Veri drift tespit edildi - Veri kalitesi kontrol edilmeli")
                report.append("🔄 Model yeniden eğitimi planlanmalı")
        
        # Raporu monitoring_outputs klasörüne kaydet
        report_text = "\n".join(report)
        output_dir = Path("monitoring_outputs")
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n💾 Monitoring raporu kaydedildi: {report_path}")
        
        return report_path

def demo_monitoring_system():
    """Monitoring system demo'su"""
    print("🧪 MONİTORİNG SİSTEMİ DEMO'SU")
    print("=" * 40)
    
    # Simulated reference data
    np.random.seed(42)
    reference_data = np.random.randn(1000, 10)
    
    # Drift detector
    drift_detector = ModelDriftDetector(reference_data, threshold=0.05)
    
    # Performance monitor
    monitor = PerformanceMonitor("demo_monitoring.db")
    
    # Simulate some monitoring data
    print("📊 Simulated monitoring data oluşturuluyor...")
    
    # Log some predictions
    for i in range(50):
        model_name = "demo_model"
        prediction = np.random.choice(['payment_issue', 'user_error', 'technical_issue'])
        confidence = np.random.uniform(0.6, 0.95)
        actual_label = np.random.choice(['payment_issue', 'user_error', 'technical_issue'])
        processing_time = np.random.uniform(0.1, 0.5)
        
        monitor.log_prediction(model_name, prediction, confidence, actual_label, processing_time)
    
    # Log some performance metrics
    for i in range(10):
        accuracy = np.random.uniform(0.75, 0.95)
        monitor.log_performance_metric("demo_model", "accuracy", accuracy, 100)
    
    # Test drift detection
    print("🔍 Drift detection test...")
    
    # Normal data (no drift)
    normal_data = np.random.randn(500, 10)
    result1 = drift_detector.detect_data_drift(normal_data, method='ks_test')
    print(f"Normal data drift: {result1['drift_detected']}")
    
    # Shifted data (drift)
    shifted_data = np.random.randn(500, 10) + 1.5  # Shift in distribution
    result2 = drift_detector.detect_data_drift(shifted_data, method='ks_test')
    print(f"Shifted data drift: {result2['drift_detected']}")
    
    # Create dashboard
    dashboard = MonitoringDashboard(monitor, drift_detector)
    
    # Performance dashboard
    perf_dashboard = dashboard.create_performance_dashboard("demo_model")
    
    # Drift dashboard
    drift_dashboard = dashboard.create_drift_dashboard()
    
    # Generate report
    report_path = dashboard.generate_monitoring_report(hours=24)
    
    print("✅ Monitoring system demo tamamlandı!")
    
    return {
        'drift_detector': drift_detector,
        'monitor': monitor,
        'dashboard': dashboard
    }

if __name__ == "__main__":
    demo_monitoring_system()
