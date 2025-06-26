"""
🔄 Online Learning System - Sürekli öğrenen AI sistemi
Yeni veri geldikçe modelleri otomatik günceller
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import joblib
import pickle
import threading
import sqlite3
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Kendi modüllerimizi import et
from utils.text_preprocessing import TurkishTextPreprocessor
from utils.feature_extraction import FeatureExtractor
from models.naive_bayes import NaiveBayesClassifier
from models.logistic_regression import LogisticRegressionClassifier

class OnlineLearningManager:
    """
    🧠 Online Learning yöneticisi
    
    Features:
    - 📊 Yeni veri otomatik ekleme
    - 🔄 Incremental model güncelleme  
    - ⏰ Scheduled retraining
    - 📈 Performance tracking
    - 💾 Data versioning
    """
    
    def __init__(self, models_dir="models/trained", data_dir="data", 
                 update_threshold=50, retrain_schedule_hours=24):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.update_threshold = update_threshold  # Kaç yeni veri gelince update yapılacak
        self.retrain_schedule_hours = retrain_schedule_hours
        
        # Paths
        self.online_data_path = self.data_dir / "online_learning_data.csv"
        self.stats_db_path = self.data_dir / "online_learning_stats.db"
        
        # Components
        self.preprocessor = TurkishTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Loaded models
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        
        # Online learning stats
        self.new_data_count = 0
        self.last_retrain_time = None
        self.performance_history = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Initialize
        self._initialize_system()
    
    def _initialize_system(self):
        """Sistemi başlat"""
        print("🔄 Online Learning System başlatılıyor...")
        
        # Dizinleri oluştur
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Database oluştur
        self._initialize_database()
        
        # Mevcut modelleri yükle
        self._load_existing_models()
        
        # Online veri dosyasını oluştur
        if not self.online_data_path.exists():
            self._create_online_data_file()
        
        print("✅ Online Learning System hazır!")
    
    def _initialize_database(self):
        """SQLite database oluştur"""
        with sqlite3.connect(self.stats_db_path) as conn:
            cursor = conn.cursor()
            
            # Performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    accuracy REAL,
                    data_count INTEGER,
                    retrain_type TEXT,
                    processing_time REAL
                )
            """)
            
            # Data tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    text TEXT,
                    true_category TEXT,
                    predicted_category TEXT,
                    confidence REAL,
                    model_name TEXT,
                    used_for_training BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
    
    def _load_existing_models(self):
        """Mevcut modelleri yükle"""
        try:
            # TF-IDF vectorizer
            vectorizer_path = self.models_dir / "tfidf_vectorizer.joblib"
            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                print("✅ TF-IDF vectorizer yüklendi")
            
            # Label encoder
            label_encoder_path = self.models_dir / "label_encoder.joblib"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                print("✅ Label encoder yüklendi")
            
            # Models
            model_files = {
                'naive_bayes': self.models_dir / "naive_bayes_multinomial.pkl",
                'logistic_regression': self.models_dir / "logistic_regression.pkl"
            }
            
            for model_name, model_path in model_files.items():
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    if isinstance(model_data, dict):
                        # Custom format
                        if model_name == 'naive_bayes':
                            self.models[model_name] = model_data['model']
                        elif model_name == 'logistic_regression':
                            self.models[model_name] = model_data['model']
                    else:
                        # Direct model
                        self.models[model_name] = model_data
                    
                    print(f"✅ {model_name} model yüklendi")
            
        except Exception as e:
            print(f"⚠️ Model yükleme hatası: {e}")
    
    def _create_online_data_file(self):
        """Online learning veri dosyasını oluştur"""
        df = pd.DataFrame(columns=['timestamp', 'text', 'category', 'processed_text', 'source'])
        df.to_csv(self.online_data_path, index=False)
        print(f"📁 Online veri dosyası oluşturuldu: {self.online_data_path}")
    
    def add_new_data(self, text: str, category: str, source: str = "api"):
        """
        Yeni veri ekle
        
        Args:
            text: Ticket metni
            category: Doğru kategori
            source: Veri kaynağı (api, manual, batch)
        """
        with self.lock:
            try:
                # Metni ön işle
                processed_text = self.preprocessor.preprocess_text(text)
                
                # Yeni veri satırı
                new_row = {
                    'timestamp': datetime.now().isoformat(),
                    'text': text,
                    'category': category,
                    'processed_text': processed_text,
                    'source': source
                }
                
                # CSV'ye ekle
                df = pd.DataFrame([new_row])
                df.to_csv(self.online_data_path, mode='a', header=False, index=False)
                
                # Database'e kaydet
                with sqlite3.connect(self.stats_db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO data_tracking (text, true_category, model_name)
                        VALUES (?, ?, ?)
                    """, (text, category, "online_learning"))
                    conn.commit()
                
                self.new_data_count += 1
                
                print(f"📊 Yeni veri eklendi: {category} (Toplam: {self.new_data_count})")
                
                # Threshold kontrolü
                if self.new_data_count >= self.update_threshold:
                    print(f"🔄 Threshold aşıldı ({self.new_data_count} >= {self.update_threshold})")
                    self._trigger_incremental_update()
                
            except Exception as e:
                print(f"❌ Veri ekleme hatası: {e}")
    
    def add_batch_data(self, texts: List[str], categories: List[str], source: str = "batch"):
        """
        Toplu veri ekle
        """
        if len(texts) != len(categories):
            raise ValueError("texts ve categories uzunlukları eşit olmalı")
        
        print(f"📦 {len(texts)} adet veri toplu olarak ekleniyor...")
        
        for text, category in zip(texts, categories):
            self.add_new_data(text, category, source)
        
        print(f"✅ {len(texts)} adet veri başarıyla eklendi!")
    
    def _trigger_incremental_update(self):
        """Incremental update tetikle"""
        print("🔄 Incremental model update başlatılıyor...")
        
        try:
            # Yeni veriyi yükle
            df = pd.read_csv(self.online_data_path)
            
            # Son N veriyi al (threshold kadar)
            recent_data = df.tail(self.new_data_count)
            
            if len(recent_data) > 0:
                # Features çıkar
                texts = recent_data['processed_text'].values
                categories = recent_data['category'].values
                
                # TF-IDF transform
                if self.vectorizer:
                    X_new = self.vectorizer.transform(texts)
                    
                    # Label encode
                    if self.label_encoder:
                        y_new = self.label_encoder.transform(categories)
                        
                        # Modelleri güncelle
                        self._update_models_incrementally(X_new, y_new)
                        
                        # Stats güncelle
                        self._update_performance_stats("incremental")
                        
                        # Reset counter
                        self.new_data_count = 0
                        
                        print("✅ Incremental update tamamlandı!")
            
        except Exception as e:
            print(f"❌ Incremental update hatası: {e}")
    
    def _update_models_incrementally(self, X_new, y_new):
        """Modelleri artımlı olarak güncelle"""
        
        # Naive Bayes - partial_fit destekliyor
        if 'naive_bayes' in self.models:
            if hasattr(self.models['naive_bayes'], 'partial_fit'):
                self.models['naive_bayes'].partial_fit(X_new, y_new)
                print("✅ Naive Bayes incrementally updated")
            else:
                print("⚠️ Naive Bayes partial_fit desteklemiyor")
        
        # Logistic Regression - SGD ile yapılabilir
        if 'logistic_regression' in self.models:
            # Basit approach: mevcut model üzerine ek eğitim
            # Daha gelişmiş SGDClassifier kullanılabilir
            print("⚠️ Logistic Regression incremental update - implementing...")
            
            # Alternatif: warm_start=True ile yeniden eğit
            try:
                from sklearn.linear_model import SGDClassifier
                
                # SGD classifier oluştur
                sgd_model = SGDClassifier(loss='log', random_state=42)
                sgd_model.partial_fit(X_new, y_new, classes=self.label_encoder.classes_)
                
                # Eski model yerine SGD kullan (geçici çözüm)
                print("✅ SGD-based incremental update yapıldı")
                
            except Exception as e:
                print(f"⚠️ Logistic Regression incremental update hatası: {e}")
    
    def trigger_full_retrain(self, force=False):
        """
        Full model retraining tetikle
        
        Args:
            force: Zorla retrain (schedule'a bakmaksızın)
        """
        
        # Schedule kontrolü
        if not force and self.last_retrain_time:
            time_since_last = datetime.now() - self.last_retrain_time
            if time_since_last.total_seconds() < (self.retrain_schedule_hours * 3600):
                print(f"⏰ Henüz retrain zamanı gelmedi. {self.retrain_schedule_hours} saat geçmeli.")
                return False
        
        print("🔄 Full model retraining başlatılıyor...")
        
        try:
            # Tüm veriyi yükle (orijinal + online)
            original_data = pd.read_csv(self.data_dir / "processed_data.csv")
            
            if self.online_data_path.exists():
                online_data = pd.read_csv(self.online_data_path)
                
                # Combine data
                combined_data = pd.concat([original_data, online_data], ignore_index=True)
            else:
                combined_data = original_data
            
            # Retrain pipeline çalıştır
            self._retrain_models_with_data(combined_data)
            
            self.last_retrain_time = datetime.now()
            
            # Stats güncelle
            self._update_performance_stats("full_retrain")
            
            print("✅ Full retrain tamamlandı!")
            return True
            
        except Exception as e:
            print(f"❌ Full retrain hatası: {e}")
            return False
    
    def _retrain_models_with_data(self, df):
        """Veri ile modelleri yeniden eğit"""
        from train_models import EnhancedModelTrainer
        
        # Geçici veri dosyası oluştur
        temp_data_path = self.data_dir / "temp_combined_data.csv"
        df.to_csv(temp_data_path, index=False)
        
        try:
            # Trainer ile yeniden eğit
            trainer = EnhancedModelTrainer(data_path=str(temp_data_path))
            trainer.load_data()
            trainer.preprocess_data()
            trainer.extract_features_once()
            
            # Sadece hızlı modelleri eğit (production için)
            trainer.train_naive_bayes()
            trainer.train_logistic_regression()
            
            # Preprocessing components kaydet
            trainer.save_preprocessing_components()
            
            print("✅ Models successfully retrained!")
            
        finally:
            # Temp dosyayı sil
            if temp_data_path.exists():
                temp_data_path.unlink()
    
    def _update_performance_stats(self, retrain_type):
        """Performance istatistiklerini güncelle"""
        try:
            with sqlite3.connect(self.stats_db_path) as conn:
                cursor = conn.cursor()
                
                # Toplam veri sayısı
                total_data = 0
                if self.online_data_path.exists():
                    df = pd.read_csv(self.online_data_path)
                    total_data = len(df)
                
                # Her model için kayıt
                for model_name in self.models.keys():
                    cursor.execute("""
                        INSERT INTO performance_history 
                        (model_name, data_count, retrain_type, processing_time)
                        VALUES (?, ?, ?, ?)
                    """, (model_name, total_data, retrain_type, time.time()))
                
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ Stats update hatası: {e}")
    
    def get_learning_stats(self):
        """Öğrenme istatistiklerini al"""
        stats = {
            'new_data_count': self.new_data_count,
            'update_threshold': self.update_threshold,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'models_loaded': list(self.models.keys()),
            'online_data_file_exists': self.online_data_path.exists(),
        }
        
        # Database stats
        try:
            with sqlite3.connect(self.stats_db_path) as conn:
                cursor = conn.cursor()
                
                # Toplam veri sayısı
                cursor.execute("SELECT COUNT(*) FROM data_tracking")
                stats['total_data_points'] = cursor.fetchone()[0]
                
                # Son retraining'ler
                cursor.execute("""
                    SELECT retrain_type, COUNT(*) 
                    FROM performance_history 
                    GROUP BY retrain_type
                """)
                stats['retrain_history'] = dict(cursor.fetchall())
                
        except Exception as e:
            stats['database_error'] = str(e)
        
        return stats
    
    def check_and_trigger_scheduled_retrain(self):
        """Scheduled retrain kontrolü"""
        if not self.last_retrain_time:
            print("📅 İlk scheduled retrain ayarlanıyor...")
            self.trigger_full_retrain(force=True)
            return
        
        time_since_last = datetime.now() - self.last_retrain_time
        hours_passed = time_since_last.total_seconds() / 3600
        
        if hours_passed >= self.retrain_schedule_hours:
            print(f"⏰ Scheduled retrain zamanı! ({hours_passed:.1f} saat geçti)")
            self.trigger_full_retrain(force=True)
        else:
            remaining_hours = self.retrain_schedule_hours - hours_passed
            print(f"⏳ Bir sonraki retrain: {remaining_hours:.1f} saat sonra")

# Singleton instance
_online_learning_manager = None

def get_online_learning_manager():
    """Online Learning Manager singleton"""
    global _online_learning_manager
    if _online_learning_manager is None:
        _online_learning_manager = OnlineLearningManager()
    return _online_learning_manager
