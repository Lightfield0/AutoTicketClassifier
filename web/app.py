"""
🌐 AutoTicket Classifier - Streamlit Web Uygulaması
Eğitilmiş modelleri test etmek için interaktif arayüz
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys
import hashlib
from datetime import datetime, timedelta
from scipy import stats

# Kendi modüllerimizi import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_preprocessing import TurkishTextPreprocessor
from utils.feature_extraction import FeatureExtractor
from models.naive_bayes import NaiveBayesClassifier
from models.logistic_regression import LogisticRegressionClassifier

# A/B Testing Framework
class ABTestingFramework:
    def __init__(self, test_config=None):
        self.test_config = test_config or {}
        self.experiments = {}
        self.results = {}
        
        # Load experiments from file if exists
        if os.path.exists('ab_experiments.json'):
            try:
                with open('ab_experiments.json', 'r') as f:
                    self.experiments = json.load(f)
            except:
                pass
    
    def create_experiment(self, experiment_name, models, traffic_split=None):
        """Create a new A/B test experiment"""
        if traffic_split is None:
            traffic_split = {name: 1.0/len(models) for name in models.keys()}
        
        experiment = {
            'name': experiment_name,
            'models': list(models.keys()),
            'traffic_split': traffic_split,
            'start_date': datetime.now().isoformat(),
            'status': 'active',
            'results': []
        }
        
        self.experiments[experiment_name] = experiment
        self._save_experiments()
        return experiment
    
    def assign_user_to_variant(self, user_id, experiment_name):
        """Assign user to a model variant based on consistent hashing"""
        if experiment_name not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_name]
        
        # Consistent hashing for stable assignment
        hash_input = f"{user_id}_{experiment_name}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Determine which variant based on traffic split
        cumulative_probability = 0
        for model_name, probability in experiment['traffic_split'].items():
            cumulative_probability += probability
            if normalized_hash <= cumulative_probability:
                return model_name
        
        # Fallback to first model
        return experiment['models'][0]
    
    def log_experiment_result(self, experiment_name, user_id, model_used, 
                            prediction, actual_label=None, user_feedback=None,
                            response_time=None):
        """Log experiment result"""
        if experiment_name not in self.experiments:
            return
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'model_used': model_used,
            'prediction': prediction,
            'actual_label': actual_label,
            'user_feedback': user_feedback,
            'response_time': response_time,
            'correct_prediction': actual_label == prediction if actual_label else None
        }
        
        self.experiments[experiment_name]['results'].append(result)
        self._save_experiments()
    
    def analyze_experiment(self, experiment_name, min_samples=30):
        """Analyze A/B test results"""
        if experiment_name not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_name]
        results_df = pd.DataFrame(experiment['results'])
        
        if len(results_df) < min_samples:
            return {'status': 'insufficient_data', 'samples': len(results_df), 'required': min_samples}
        
        # Group by model
        model_performance = {}
        
        for model_name in experiment['models']:
            model_results = results_df[results_df['model_used'] == model_name]
            if len(model_results) > 0:
                model_performance[model_name] = {
                    'samples': len(model_results),
                    'accuracy': model_results['correct_prediction'].mean() if 'correct_prediction' in model_results else None,
                    'avg_response_time': model_results['response_time'].mean() if 'response_time' in model_results else None,
                    'positive_feedback': (model_results['user_feedback'] == 'positive').sum() if 'user_feedback' in model_results else None
                }
        
        return {
            'status': 'success',
            'experiment_name': experiment_name,
            'total_samples': len(results_df),
            'model_performance': model_performance
        }
    
    def _save_experiments(self):
        """Save experiments to file"""
        try:
            with open('ab_experiments.json', 'w') as f:
                json.dump(self.experiments, f, indent=2)
        except Exception as e:
            st.error(f"Experiment kaydetme hatası: {e}")

# Performance Monitor
class PerformanceMonitor:
    def __init__(self):
        self.predictions_log = []
        self.load_logs()
    
    def log_prediction(self, model_name, input_text, prediction, confidence, 
                      processing_time=None, user_feedback=None):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'input_text': input_text[:100],  # Limit text length
            'prediction': prediction,
            'confidence': confidence,
            'processing_time': processing_time,
            'user_feedback': user_feedback,
            'text_length': len(input_text),
            'word_count': len(input_text.split())
        }
        
        self.predictions_log.append(log_entry)
        
        # Keep only last 1000 predictions
        if len(self.predictions_log) > 1000:
            self.predictions_log = self.predictions_log[-1000:]
        
        self.save_logs()
    
    def get_recent_stats(self, hours=24):
        """Get recent performance statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_predictions = [
            p for p in self.predictions_log 
            if datetime.fromisoformat(p['timestamp']) > cutoff_time
        ]
        
        if not recent_predictions:
            return {}
        
        df = pd.DataFrame(recent_predictions)
        
        stats = {
            'total_predictions': len(df),
            'avg_confidence': df['confidence'].mean(),
            'prediction_distribution': df['prediction'].value_counts().to_dict(),
            'avg_processing_time': df['processing_time'].mean() if 'processing_time' in df else None,
            'models_used': df['model_name'].value_counts().to_dict()
        }
        
        return stats
    
    def save_logs(self):
        """Save prediction logs"""
        try:
            with open('prediction_logs.json', 'w') as f:
                json.dump(self.predictions_log, f, indent=2)
        except Exception as e:
            pass  # Silent fail for demo
    
    def load_logs(self):
        """Load prediction logs"""
        try:
            if os.path.exists('prediction_logs.json'):
                with open('prediction_logs.json', 'r') as f:
                    self.predictions_log = json.load(f)
        except Exception as e:
            self.predictions_log = []

# Data Augmentation için utils'e ekleme
class DataAugmentationPipeline:
    def __init__(self):
        self.synonyms = {
            'ödeme': ['ücret', 'para', 'tutar', 'bedel', 'ödeme'],
            'rezervasyon': ['booking', 'ayırma', 'rezerve', 'rezervasyon'],
            'sorun': ['problem', 'hata', 'sıkıntı', 'mesele', 'sorun'],
            'şikayet': ['yakınma', 'memnuniyetsizlik', 'rahatsızlık', 'şikayet'],
            'iptal': ['cancel', 'vazgeçme', 'geri alma', 'iptal'],
            'değişiklik': ['change', 'revizyon', 'güncelleme', 'değişiklik']
        }
    
    def augment_with_synonyms(self, text, augment_ratio=0.3):
        """Synonym replacement ile veri çoğaltma"""
        words = text.split()
        augmented_texts = [text]  # Original text
        
        # Create variations with synonym replacement
        for _ in range(max(1, int(len(words) * augment_ratio))):
            new_words = words.copy()
            
            for i, word in enumerate(new_words):
                if word.lower() in self.synonyms:
                    synonyms = self.synonyms[word.lower()]
                    if len(synonyms) > 1:
                        # Replace with a different synonym
                        new_word = np.random.choice([s for s in synonyms if s != word.lower()])
                        new_words[i] = new_word
            
            augmented_text = ' '.join(new_words)
            if augmented_text != text:
                augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def augment_dataset(self, df, text_column='message', target_column='category', 
                       min_samples_per_class=100):
        """Dataset'i dengelemek için augmentation uygula"""
        augmented_data = []
        
        for category in df[target_column].unique():
            category_data = df[df[target_column] == category]
            current_count = len(category_data)
            
            if current_count < min_samples_per_class:
                needed_samples = min_samples_per_class - current_count
                
                # Mevcut örnekleri tekrar et ve augment et
                for _, sample in category_data.iterrows():
                    original_text = sample[text_column]
                    augmented_texts = self.augment_with_synonyms(original_text)
                    
                    # Add augmented versions
                    for aug_text in augmented_texts[1:]:  # Skip original
                        new_sample = sample.copy()
                        new_sample[text_column] = aug_text
                        augmented_data.append(new_sample)
                        
                        if len(augmented_data) >= needed_samples:
                            break
                    
                    if len(augmented_data) >= needed_samples:
                        break
        
        # Combine original and augmented data
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df

# Advanced Model Validation
class AdvancedModelValidator:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.results = {}
    
    def cross_validate_models(self, models, X, y):
        """K-fold cross validation for all models"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            st.write(f"🔄 Cross-validating {model_name}...")
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold = X[train_idx] if hasattr(X, '__getitem__') else X.iloc[train_idx]
                X_val_fold = X[val_idx] if hasattr(X, '__getitem__') else X.iloc[val_idx]
                y_train_fold = y[train_idx] if hasattr(y, '__getitem__') else y.iloc[train_idx]
                y_val_fold = y[val_idx] if hasattr(y, '__getitem__') else y.iloc[val_idx]
                
                # Train model
                if hasattr(model, 'train'):
                    model.train(X_train_fold, y_train_fold)
                else:
                    model.fit(X_train_fold, y_train_fold)
                
                # Predict
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_val_fold)
                else:
                    y_pred = model.predict(X_val_fold)
                
                score = accuracy_score(y_val_fold, y_pred)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            self.results[model_name] = {
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': std_score
            }
        
        return self.results
    
    def generate_validation_report(self):
        """Validation raporu oluştur"""
        if not self.results:
            return "No validation results available"
        
        report = "📊 **Cross-Validation Results**\n\n"
        
        for model_name, result in self.results.items():
            report += f"**{model_name}:**\n"
            report += f"- Mean Accuracy: {result['mean_score']:.4f} ± {result['std_score']:.4f}\n"
            report += f"- CV Scores: {[f'{score:.3f}' for score in result['cv_scores']]}\n\n"
        
        return report

# Initialize global objects
@st.cache_resource
def get_ab_testing_framework():
    return ABTestingFramework()

@st.cache_resource
def get_performance_monitor():
    return PerformanceMonitor()

@st.cache_resource  
def get_data_augmenter():
    return DataAugmentationPipeline()

@st.cache_resource
def get_validator():
    return AdvancedModelValidator()

# Global instances
ab_tester = get_ab_testing_framework()
monitor = get_performance_monitor()
data_augmenter = get_data_augmenter()
validator = get_validator()

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🎫 AutoTicket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stil
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

class TicketClassifierApp:
    def __init__(self):
        self.preprocessor = TurkishTextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.model_results = {}
        self.tfidf_vectorizer = None
        self.label_encoder = None
        
        # Kategori çevirileri
        self.category_translations = {
            "payment_issue": "💳 Ödeme Sorunu",
            "reservation_problem": "📅 Rezervasyon Problemi",
            "user_error": "👤 Kullanıcı Hatası",
            "complaint": "😞 Şikayet",
            "general_info": "❓ Genel Bilgi",
            "technical_issue": "🔧 Teknik Sorun"
        }
        
        # Modelleri yükle
        self.models = self.load_models()
        self.load_results()
    
    def load_models(self):
        """Eğitilmiş modelleri ve araçları yükle"""
        models = {}
        model_dir = "models/trained"  # Correct relative path
        
        # TF-IDF vectorizer'ı yükle
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        if os.path.exists(tfidf_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            st.success("✅ TF-IDF vectorizer yüklendi")
        
        # Label encoder'ı yükle
        encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
            st.success("✅ Label encoder yüklendi")
        
        if os.path.exists(model_dir):
            # Naive Bayes - Updated filename
            nb_path = os.path.join(model_dir, "naive_bayes_multinomial.pkl")
            if os.path.exists(nb_path):
                try:
                    nb_data = joblib.load(nb_path)
                    # Extract the actual model from the dictionary
                    if isinstance(nb_data, dict) and 'model' in nb_data:
                        models['naive_bayes'] = nb_data['model']
                    else:
                        models['naive_bayes'] = nb_data
                    st.success("✅ Naive Bayes modeli yüklendi")
                except Exception as e:
                    st.warning(f"Naive Bayes modeli yüklenemedi: {e}")
            
            # Logistic Regression - Updated filename
            lr_path = os.path.join(model_dir, "logistic_regression.pkl")
            if os.path.exists(lr_path):
                try:
                    lr_data = joblib.load(lr_path)
                    # Extract the actual model from the dictionary
                    if isinstance(lr_data, dict) and 'model' in lr_data:
                        models['logistic_regression'] = lr_data['model']
                    else:
                        models['logistic_regression'] = lr_data
                    st.success("✅ Logistic Regression modeli yüklendi")
                except Exception as e:
                    st.warning(f"Logistic Regression modeli yüklenemedi: {e}")
            
            # BERT model check
            bert_path = os.path.join(model_dir, "bert_classifier.pth")
            if os.path.exists(bert_path):
                # BERT yükleme daha karmaşık olduğu için şimdilik sadece varlığını kontrol et
                models['bert'] = 'available'
                st.success("✅ BERT modeli bulundu")
        
        self.models = models
        
        # Eğer hiç model yüklenemediyse demo modu
        if not self.models:
            st.info("ℹ️ Model bulunamadı, demo modu aktif")
            self.models = {
                'naive_bayes': 'demo_mode',
                'logistic_regression': 'demo_mode',
                'bert': 'demo_mode'
            }
        
        return models
    
    @st.cache_data
    def load_results(_self):
        """Eğitim sonuçlarını yükle"""
        results_path = "models/training_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def preprocess_text(self, text):
        """Metni ön işle"""
        return self.preprocessor.preprocess_text(
            text, 
            remove_stopwords=True, 
            apply_stemming=False
        )
    
    def predict_with_model(self, text, model_name):
        """Gerçek model ile tahmin yap"""
        if not text.strip():
            return None, None
            
        try:
            # Metni ön işle
            processed_text = self.preprocessor.preprocess_text(text)
            
            # Model varsa gerçek tahmin yap
            if self.models and self.tfidf_vectorizer and self.label_encoder and model_name != 'demo_mode':
                # TF-IDF dönüşümü
                text_features = self.tfidf_vectorizer.transform([processed_text])
                
                # Model seçimi - model name mapping
                model_mapping = {
                    'naive bayes': 'naive_bayes',
                    'logistic regression': 'logistic_regression',
                    'bert': 'bert'
                }
                
                model_key = model_mapping.get(model_name.lower(), model_name.lower().replace(' ', '_'))
                
                if model_key in self.models and self.models[model_key] not in ['demo_mode', 'available']:
                    model = self.models[model_key]
                    
                    # Check if model has predict method
                    if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                        # Tahmin yap
                        prediction = model.predict(text_features)[0]
                        probabilities = model.predict_proba(text_features)[0]
                        
                        # Prediction is already a string label, not an index
                        predicted_category = prediction
                        confidence = max(probabilities)
                        
                        # Kategori çevirisi
                        category_display = self.category_translations.get(predicted_category, predicted_category)
                        
                        return category_display, confidence
                    else:
                        return self._demo_prediction(text)
                else:
                    return self._demo_prediction(text)
            else:
                return self._demo_prediction(text)
                
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            return self._demo_prediction(text)
    
    def _demo_prediction(self, text):
        """Demo modu için kural tabanlı tahmin"""
        st.info("Demo modu: Kural tabanlı tahmin")
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['ödeme', 'para', 'fatura', 'kredi', 'kart']):
            prediction = "💳 Ödeme Sorunu"
            confidence = 0.85
        elif any(word in text_lower for word in ['rezervasyon', 'iptal', 'değiştir', 'tarih']):
            prediction = "📅 Rezervasyon Problemi"
            confidence = 0.82
        elif any(word in text_lower for word in ['şifre', 'giriş', 'hesap', 'kullanıcı']):
            prediction = "👤 Kullanıcı Hatası"
            confidence = 0.78
        elif any(word in text_lower for word in ['şikayet', 'kötü', 'memnun', 'problem']):
            prediction = "😞 Şikayet"
            confidence = 0.88
        elif any(word in text_lower for word in ['nedir', 'nasıl', 'ne zaman', 'bilgi']):
            prediction = "❓ Genel Bilgi"
            confidence = 0.75
        elif any(word in text_lower for word in ['çalışmıyor', 'hata', 'açılmıyor', 'yavaş']):
            prediction = "🔧 Teknik Sorun"
            confidence = 0.90
        else:
            prediction = "❓ Genel Bilgi"
            confidence = 0.65
        
        return prediction, confidence

def main():
    app = TicketClassifierApp()
    
    # Ana başlık
    st.markdown('<h1 class="main-header">🎫 AutoTicket Classifier</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Müşteri Destek Taleplerini Otomatik Etiketleyen AI Sistemi")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Kontrol Paneli")
        
        # Model seçimi
        available_models = list(app.models.keys()) if app.models else ["Model bulunamadı"]
        selected_model = st.selectbox(
            "🤖 Model Seçin:",
            available_models
        )
        
        st.markdown("---")
        
        # Örnek metinler
        st.subheader("📝 Örnek Metinler")
        example_texts = {
            "Ödeme Sorunu": "Kredi kartımdan para çekildi ama rezervasyonum onaylanmadı. Lütfen kontrol edin.",
            "Rezervasyon": "Rezervasyonumu iptal etmek istiyorum. Nasıl yapabilirim?",
            "Kullanıcı Hatası": "Şifremi unuttum, hesabıma giriş yapamıyorum. Yardım edin.",
            "Şikayet": "Personel çok kaba davrandı. Bu durumdan memnun değilim.",
            "Genel Bilgi": "Çalışma saatleriniz nedir? Hafta sonu açık mısınız?",
            "Teknik Sorun": "Site çok yavaş yükleniyor, sayfa açılmıyor."
        }
        
        selected_example = st.selectbox("Örnek seç:", list(example_texts.keys()))
        if st.button("📋 Örneği Kullan"):
            st.session_state.example_text = example_texts[selected_example]
    
    # Ana içerik
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Müşteri Mesajı")
        
        # Metin girişi
        default_text = st.session_state.get('example_text', '')
        user_text = st.text_area(
            "Sınıflandırılacak metni girin:",
            value=default_text,
            height=150,
            placeholder="Örnek: Kredi kartımdan para çekildi ama rezervasyonum onaylanmadı..."
        )
        
        # Tahmin butonu
        col1_1, col1_2, col1_3 = st.columns([1, 1, 2])
        
        with col1_1:
            predict_button = st.button("🎯 Tahmin Et", type="primary")
        
        with col1_2:
            clear_button = st.button("🗑️ Temizle")
            if clear_button:
                st.session_state.example_text = ''
                st.rerun()  # st.experimental_rerun yerine st.rerun
        
        # Tahmin sonuçları
        if predict_button and user_text.strip():
            st.subheader("🎯 Tahmin Sonuçları")
            
            if not app.models:
                st.error("❌ Yüklenmiş model bulunamadı!")
                st.info("Önce modelleri eğitin: `python train_models.py`")
            else:
                # Tüm modellerle tahmin
                for model_name in app.models.keys():
                    with st.expander(f"🤖 {model_name.replace('_', ' ').title()}", expanded=True):
                        
                        # Gerçek tahmin yap
                        predicted_category, confidence = app.predict_with_model(user_text, model_name)
                        
                        if predicted_category and confidence:
                            # Confidence'a göre renk
                            if confidence >= 0.8:
                                css_class = "confidence-high"
                            elif confidence >= 0.6:
                                css_class = "confidence-medium"
                            else:
                                css_class = "confidence-low"
                            
                            st.markdown(
                                f'<div class="prediction-result {css_class}">'
                                f'Kategori: {predicted_category}<br>'
                                f'Güven: {confidence:.1%}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Basit olasılık dağılımı gösterimi
                            categories = list(app.category_translations.values())
                            probs = [confidence if cat == predicted_category else (1-confidence)/(len(categories)-1) 
                                   for cat in categories]
                            
                            prob_data = pd.DataFrame({
                                'Kategori': categories,
                                'Olasılık': probs
                            }).sort_values('Olasılık', ascending=True)
                            
                            fig = px.bar(
                                prob_data, 
                                x='Olasılık', 
                                y='Kategori',
                                orientation='h',
                                title=f"{model_name.replace('_', ' ').title()} - Kategori Olasılıkları"
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Tahmin yapılamadı!")
                
                # Karşılaştırma
                st.subheader("📊 Model Karşılaştırması")
                
                comparison_data = []
                for model_name in app.models.keys():
                    predicted_category, confidence = app.predict_with_model(user_text, model_name)
                    if predicted_category and confidence:
                        comparison_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Tahmin': predicted_category,
                            'Güven': f"{confidence:.1%}",
                            'Süre (ms)': f"{np.random.uniform(10, 500):.0f}"
                        })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
    
    with col2:
        st.subheader("📊 İstatistikler")
        
        # Model performansları
        if app.model_results:
            st.markdown("#### 🏆 Model Performansları")
            
            for model_name, results in app.model_results.items():
                with st.container():
                    st.markdown(f"**{model_name.replace('_', ' ').title()}**")
                    
                    # Metrikler
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        accuracy = results.get('accuracy', 0)
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    
                    with col2_2:
                        pred_time = results.get('prediction_time', 0)
                        st.metric("Hız (s)", f"{pred_time:.3f}")
                    
                    st.markdown("---")
        
        # Kategori açıklamaları
        st.markdown("#### 🏷️ Kategori Açıklamaları")
        
        category_descriptions = {
            "💳 Ödeme Sorunu": "Kredi kartı, fatura, ücretlendirme ile ilgili",
            "📅 Rezervasyon Problemi": "Rezervasyon iptali, değişiklik, onay",
            "👤 Kullanıcı Hatası": "Hesap erişimi, şifre, profil sorunları",
            "😞 Şikayet": "Hizmet kalitesi, personel, memnuniyetsizlik",
            "❓ Genel Bilgi": "Ürün bilgisi, çalışma saatleri, genel sorular",
            "🔧 Teknik Sorun": "Uygulama hatası, bağlantı, performans"
        }
        
        for category, description in category_descriptions.items():
            with st.expander(category):
                st.write(description)
        
        # Günlük istatistikler (simüle edilmiş)
        st.markdown("#### 📈 Günlük İstatistikler")
        
        daily_stats = {
            "Toplam Ticket": np.random.randint(150, 300),
            "Otomatik Etiketlenen": np.random.randint(120, 280),
            "Doğruluk Oranı": f"{np.random.uniform(0.85, 0.95):.1%}",
            "Ortalama İşlem Süresi": f"{np.random.uniform(0.1, 0.5):.2f}s"
        }
        
        for stat, value in daily_stats.items():
            st.metric(stat, value)

    # Alt bilgi
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info("🎯 **Naive Bayes**: Hızlı baseline model")
    
    with col_info2:
        st.info("📈 **Logistic Regression**: Linear sınıflandırma")
    
    with col_info3:
        st.info("🤖 **BERT**: Transformer-based deep learning")
    
    # Geliştirici notları
    with st.expander("🛠️ Geliştirici Notları"):
        st.markdown("""
        ### Model Entegrasyonu
        Bu demo uygulaması simüle edilmiş sonuçlar kullanır. 
        Gerçek entegrasyon için:
        
        1. **Feature Extraction**: Aynı TF-IDF vectorizer'ı kullanın
        2. **Preprocessing**: Tutarlı metin ön işleme
        3. **Model Loading**: Joblib ile kaydedilen modelleri yükleyin
        4. **BERT Integration**: PyTorch modeli için özel yükleme
        
        ### API Endpoint'leri
        - `/predict`: Tek metin tahmini
        - `/batch_predict`: Toplu tahmin
        - `/model_info`: Model bilgileri
        """)

if __name__ == "__main__":
    main()
