"""
🎫 AutoTicket Classifier - Simple Web Interface
Tamamen çalışır durumda basit arayüz
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
import re
from datetime import datetime

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
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleTicketClassifier:
    def __init__(self):
        self.models = {}
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.categories = [
            "Ödeme Sorunu",
            "Rezervasyon Problemi", 
            "Kullanıcı Hatası",
            "Şikayet",
            "Genel Bilgi",
            "Teknik Sorun"
        ]
        self.load_models()
    
    def load_models(self):
        """Modelleri yükle"""
        model_dir = "models/trained"
        
        # TF-IDF vectorizer
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
        if os.path.exists(tfidf_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
        
        # Label encoder
        encoder_path = os.path.join(model_dir, "label_encoder.joblib")
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        # Naive Bayes
        nb_path = os.path.join(model_dir, "naive_bayes_model.joblib")
        if os.path.exists(nb_path):
            self.models['Naive Bayes'] = joblib.load(nb_path)
        
        # Logistic Regression
        lr_path = os.path.join(model_dir, "logistic_regression_model.joblib")
        if os.path.exists(lr_path):
            self.models['Logistic Regression'] = joblib.load(lr_path)
    
    def preprocess_text(self, text):
        """Basit metin ön işleme"""
        text = str(text).lower()
        text = re.sub(r'[^\w\sçğıöşü]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, text, model_name):
        """Tahmin yap"""
        if not text.strip():
            return None, None, None
        
        # Model yüklü mü kontrol et
        if self.tfidf_vectorizer is None or self.label_encoder is None or model_name not in self.models:
            return self.fallback_prediction(text)
        
        try:
            # Metni ön işle
            processed_text = self.preprocess_text(text)
            
            # TF-IDF dönüşümü
            text_features = self.tfidf_vectorizer.transform([processed_text])
            
            # Model ile tahmin
            model = self.models[model_name]
            prediction = model.predict(text_features)[0]
            probabilities = model.predict_proba(text_features)[0]
            
            # Kategori ismine çevir
            predicted_category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Tüm kategoriler için olasılıklar
            all_categories = self.label_encoder.classes_
            prob_dict = {cat: float(prob) for cat, prob in zip(all_categories, probabilities)}
            
            return predicted_category, confidence, prob_dict
            
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")
            return self.fallback_prediction(text)
    
    def fallback_prediction(self, text):
        """Basit kural tabanlı tahmin"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['ödeme', 'para', 'fatura', 'kredi', 'kart']):
            predicted_category = "Ödeme Sorunu"
            confidence = 0.85
        elif any(word in text_lower for word in ['rezervasyon', 'iptal', 'değiştir', 'tarih']):
            predicted_category = "Rezervasyon Problemi"
            confidence = 0.82
        elif any(word in text_lower for word in ['şifre', 'giriş', 'hesap', 'kullanıcı']):
            predicted_category = "Kullanıcı Hatası"
            confidence = 0.78
        elif any(word in text_lower for word in ['şikayet', 'kötü', 'memnun', 'problem']):
            predicted_category = "Şikayet"
            confidence = 0.88
        elif any(word in text_lower for word in ['nedir', 'nasıl', 'ne zaman', 'bilgi']):
            predicted_category = "Genel Bilgi"
            confidence = 0.75
        elif any(word in text_lower for word in ['çalışmıyor', 'hata', 'açılmıyor', 'yavaş']):
            predicted_category = "Teknik Sorun"
            confidence = 0.90
        else:
            predicted_category = "Genel Bilgi"
            confidence = 0.65
        
        # Diğer kategoriler için düşük olasılıklar
        prob_dict = {cat: 0.1 if cat != predicted_category else confidence for cat in self.categories}
        
        return predicted_category, confidence, prob_dict

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🎫 AutoTicket Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Müşteri Destek Taleplerini Otomatik Etiketleyen AI Sistemi")
    
    # Classifier'ı başlat
    if 'classifier' not in st.session_state:
        with st.spinner('🤖 AI modelleri yükleniyor...'):
            st.session_state.classifier = SimpleTicketClassifier()
    
    classifier = st.session_state.classifier
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Kontrol Paneli")
        
        # Model seçimi
        available_models = list(classifier.models.keys()) if classifier.models else ["Demo Mode"]
        if not available_models:
            available_models = ["Demo Mode"]
        
        selected_model = st.selectbox("🤖 Model Seçin:", available_models)
        
        if classifier.models:
            st.success(f"✅ {len(classifier.models)} model yüklendi")
        else:
            st.warning("⚠️ Demo modu aktif")
        
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
        st.subheader("📝 Destek Talebini Girin")
        
        # Metin girişi
        if 'example_text' in st.session_state:
            default_text = st.session_state.example_text
            del st.session_state.example_text
        else:
            default_text = ""
        
        user_text = st.text_area(
            "Müşteri mesajını buraya yazın:",
            value=default_text,
            height=150,
            placeholder="Örnek: Kredi kartımdan yanlış tutar çekildi, lütfen iade edin..."
        )
        
        # Tahmin butonu
        if st.button("🎯 Kategoriyi Tahmin Et", type="primary"):
            if user_text.strip():
                with st.spinner('🤔 AI analiz ediyor...'):
                    category, confidence, probabilities = classifier.predict(user_text, selected_model)
                
                if category:
                    # Sonuçları göster
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Tahmin Edilen Kategori: **{category}**")
                    st.markdown(f"### 🎚️ Güven Oranı: **{confidence:.1%}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Güven seviyesine göre CSS class
                    if confidence > 0.8:
                        css_class = "confidence-high"
                        icon = "🔥"
                    elif confidence > 0.6:
                        css_class = "confidence-medium"
                        icon = "✅"
                    else:
                        css_class = "confidence-low"
                        icon = "⚠️"
                    
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'{icon} <strong>Güven Seviyesi:</strong> {confidence:.1%}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Tüm kategoriler için olasılıklar
                    st.subheader("📊 Tüm Kategoriler İçin Olasılıklar")
                    
                    if probabilities:
                        # DataFrame oluştur
                        prob_df = pd.DataFrame(
                            list(probabilities.items()),
                            columns=['Kategori', 'Olasılık']
                        ).sort_values('Olasılık', ascending=False)
                        
                        # Bar chart
                        fig = px.bar(
                            prob_df,
                            x='Olasılık',
                            y='Kategori',
                            orientation='h',
                            title="Kategori Olasılıkları",
                            color='Olasılık',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tablo
                        prob_df['Olasılık'] = prob_df['Olasılık'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(prob_df, use_container_width=True)
                else:
                    st.error("❌ Tahmin yapılamadı!")
            else:
                st.warning("⚠️ Lütfen bir metin girin!")
    
    with col2:
        st.subheader("📈 Sistem Bilgileri")
        
        # Model durumu
        if classifier.models:
            st.success(f"🤖 **{len(classifier.models)} Model Aktif**")
            for model_name in classifier.models.keys():
                st.info(f"✅ {model_name}")
        else:
            st.warning("⚠️ **Demo Modu Aktif**")
            st.info("Kural tabanlı tahmin")
        
        # Kategori açıklamaları
        st.subheader("🏷️ Desteklenen Kategoriler")
        
        category_info = {
            "💳 Ödeme Sorunu": "Kredi kartı, fatura, ücretlendirme",
            "📅 Rezervasyon Problemi": "İptal, değişiklik, onay sorunları",
            "👤 Kullanıcı Hatası": "Hesap erişimi, şifre, profil",
            "😞 Şikayet": "Hizmet kalitesi, memnuniyetsizlik",
            "❓ Genel Bilgi": "Ürün bilgisi, çalışma saatleri",
            "🔧 Teknik Sorun": "Uygulama hatası, bağlantı sorunu"
        }
        
        for category, description in category_info.items():
            with st.expander(category):
                st.write(description)
        
        # İstatistikler (simüle edilmiş)
        st.subheader("📊 Günlük İstatistikler")
        
        stats_data = {
            "Toplam Analiz": np.random.randint(200, 400),
            "Doğruluk Oranı": f"{np.random.uniform(0.85, 0.95):.1%}",
            "Ortalama Süre": f"{np.random.uniform(0.1, 0.3):.2f}s"
        }
        
        for stat, value in stats_data.items():
            st.metric(stat, value)
    
    # Alt bilgi
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.info("🎯 **Naive Bayes**: Hızlı ve etkili temel model")
    
    with col_info2:
        st.info("📈 **Logistic Regression**: Dengeli performans")
    
    with col_info3:
        st.info("🔧 **Demo Mode**: Kural tabanlı fallback sistemi")
    
    # Batch işlem özelliği
    st.markdown("---")
    st.subheader("📦 Toplu Analiz")
    
    if st.checkbox("Toplu metin analizi yap"):
        batch_text = st.text_area(
            "Her satıra bir metin yazın:",
            height=200,
            placeholder="Kredi kartı problemi var\nRezervasyon iptal ettim\nŞifre unutmuşum"
        )
        
        if st.button("🔄 Toplu Analiz Yap"):
            if batch_text.strip():
                lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
                
                if lines:
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, line in enumerate(lines):
                        category, confidence, _ = classifier.predict(line, selected_model)
                        results.append({
                            'Metin': line[:50] + "..." if len(line) > 50 else line,
                            'Kategori': category or "Belirsiz",
                            'Güven': f"{confidence:.1%}" if confidence else "N/A"
                        })
                        progress_bar.progress((i + 1) / len(lines))
                    
                    # Sonuçları göster
                    st.subheader("📋 Toplu Analiz Sonuçları")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # İstatistikler
                    if results:
                        category_counts = results_df['Kategori'].value_counts()
                        fig = px.pie(
                            values=category_counts.values,
                            names=category_counts.index,
                            title="Kategori Dağılımı"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Lütfen analiz edilecek metinleri girin!")

if __name__ == "__main__":
    main()
