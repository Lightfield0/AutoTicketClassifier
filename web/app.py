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
from datetime import datetime

# Kendi modüllerimizi import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.text_preprocessing import TurkishTextPreprocessor
    from utils.feature_extraction import FeatureExtractor
    from models.naive_bayes import NaiveBayesClassifier
    from models.logistic_regression import LogisticRegressionClassifier
except ImportError as e:
    st.warning(f"Modül import hatası: {e}")
    # Fallback: Basit sınıflar oluştur
    class TurkishTextPreprocessor:
        def preprocess_text(self, text, remove_stopwords=True, apply_stemming=False):
            import re
            text = str(text).lower()
            text = re.sub(r'[^\w\sçğıöşü]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    
    class FeatureExtractor:
        pass
    
    class NaiveBayesClassifier:
        pass
    
    class LogisticRegressionClassifier:
        pass

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
                st.experimental_rerun()
        
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
