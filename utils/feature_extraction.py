"""
🎯 Feature Extraction Utilities
Metinlerden özellik çıkarma araçları
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from collections import Counter
import re

class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.hashing_vectorizer = None
        
    def extract_tfidf_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """TF-IDF özelliklerini çıkarır"""
        print(f"🔢 TF-IDF özellikleri çıkarılıyor (max_features={max_features}, ngram_range={ngram_range})")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # En az 2 dokümanda geçmeli
            max_df=0.95,  # %95'ten fazla dokümanda geçmesin
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"✅ TF-IDF: {tfidf_matrix.shape[1]} özellik çıkarıldı")
        return tfidf_matrix, feature_names
    
    def extract_count_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """Count (Bag of Words) özelliklerini çıkarır"""
        print(f"🔢 Count özellikleri çıkarılıyor (max_features={max_features})")
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            strip_accents='unicode'
        )
        
        count_matrix = self.count_vectorizer.fit_transform(texts)
        feature_names = self.count_vectorizer.get_feature_names_out()
        
        print(f"✅ Count: {count_matrix.shape[1]} özellik çıkarıldı")
        return count_matrix, feature_names
    
    def extract_hashing_features(self, texts, n_features=10000):
        """Hashing Trick özelliklerini çıkarır (büyük veri için)"""
        print(f"🔢 Hashing özellikleri çıkarılıyor (n_features={n_features})")
        
        self.hashing_vectorizer = HashingVectorizer(
            n_features=n_features,
            ngram_range=(1, 2),
            strip_accents='unicode'
        )
        
        hashing_matrix = self.hashing_vectorizer.fit_transform(texts)
        
        print(f"✅ Hashing: {hashing_matrix.shape[1]} özellik çıkarıldı")
        return hashing_matrix
    
    def extract_statistical_features(self, texts):
        """İstatistiksel özellikler çıkarır"""
        print("📊 İstatistiksel özellikler çıkarılıyor...")
        
        features = []
        
        for text in texts:
            feature_dict = {}
            
            # Temel uzunluk özellikleri
            feature_dict['char_count'] = len(text)
            feature_dict['word_count'] = len(text.split())
            feature_dict['sentence_count'] = len(re.split(r'[.!?]+', text))
            feature_dict['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Noktalama işaretleri
            feature_dict['exclamation_count'] = text.count('!')
            feature_dict['question_count'] = text.count('?')
            feature_dict['comma_count'] = text.count(',')
            feature_dict['period_count'] = text.count('.')
            
            # Büyük harf kullanımı
            feature_dict['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            feature_dict['title_case_words'] = sum(1 for word in text.split() if word.istitle())
            
            # Sayısal özellikler
            feature_dict['digit_count'] = sum(1 for c in text if c.isdigit())
            feature_dict['has_numbers'] = 1 if any(c.isdigit() for c in text) else 0
            
            # Özel kelimeler (müşteri destek bağlamında)
            urgency_words = ['acil', 'urgent', 'hemen', 'çabuk', 'acilen']
            feature_dict['urgency_words'] = sum(1 for word in urgency_words if word in text.lower())
            
            polite_words = ['lütfen', 'teşekkür', 'saygı', 'merhaba', 'iyi günler']
            feature_dict['polite_words'] = sum(1 for word in polite_words if word in text.lower())
            
            negative_words = ['kötü', 'berbat', 'sorun', 'hata', 'problem', 'şikayet']
            feature_dict['negative_words'] = sum(1 for word in negative_words if word in text.lower())
            
            features.append(feature_dict)
        
        df_features = pd.DataFrame(features)
        print(f"✅ İstatistiksel: {len(df_features.columns)} özellik çıkarıldı")
        return df_features
    
    def extract_lexical_features(self, texts):
        """Kelime bilgisi özelliklerini çıkarır"""
        print("📚 Kelime bilgisi özellikleri çıkarılıyor...")
        
        features = []
        
        # Tüm metinlerden kelime frekansları
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        word_freq = Counter(all_words)
        vocabulary_size = len(word_freq)
        
        for text in texts:
            feature_dict = {}
            words = text.lower().split()
            
            if not words:
                features.append({key: 0 for key in ['lexical_diversity', 'rare_word_ratio', 
                               'common_word_ratio', 'unique_word_ratio']})
                continue
            
            # Kelime çeşitliliği (Type-Token Ratio)
            unique_words = set(words)
            feature_dict['lexical_diversity'] = len(unique_words) / len(words)
            
            # Nadir kelime oranı (frekansı < 5)
            rare_words = [word for word in words if word_freq[word] < 5]
            feature_dict['rare_word_ratio'] = len(rare_words) / len(words)
            
            # Yaygın kelime oranı (top %10)
            top_words = set([word for word, freq in word_freq.most_common(int(vocabulary_size * 0.1))])
            common_words = [word for word in words if word in top_words]
            feature_dict['common_word_ratio'] = len(common_words) / len(words)
            
            # Tekil kelime oranı
            feature_dict['unique_word_ratio'] = len(unique_words) / len(words)
            
            features.append(feature_dict)
        
        df_features = pd.DataFrame(features)
        print(f"✅ Kelime bilgisi: {len(df_features.columns)} özellik çıkarıldı")
        return df_features
    
    def combine_all_features(self, texts, tfidf_max_features=3000):
        """Tüm özellik tiplerini birleştirir"""
        print("🔗 Tüm özellikler birleştiriliyor...")
        
        # TF-IDF özellikleri
        tfidf_matrix, tfidf_names = self.extract_tfidf_features(texts, max_features=tfidf_max_features)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_names)
        
        # İstatistiksel özellikler
        stat_features = self.extract_statistical_features(texts)
        
        # Kelime bilgisi özellikleri
        lexical_features = self.extract_lexical_features(texts)
        
        # Tüm özellikleri birleştir
        combined_features = pd.concat([tfidf_df, stat_features, lexical_features], axis=1)
        
        print(f"🎯 Toplam {combined_features.shape[1]} özellik oluşturuldu")
        print(f"   - TF-IDF: {tfidf_df.shape[1]}")
        print(f"   - İstatistiksel: {stat_features.shape[1]}")
        print(f"   - Kelime bilgisi: {lexical_features.shape[1]}")
        
        return combined_features
    
    def transform_new_text(self, texts, feature_type='tfidf'):
        """Eğitilmiş vectorizer ile yeni metinleri dönüştürür"""
        if feature_type == 'tfidf' and self.tfidf_vectorizer:
            return self.tfidf_vectorizer.transform(texts)
        elif feature_type == 'count' and self.count_vectorizer:
            return self.count_vectorizer.transform(texts)
        elif feature_type == 'hashing' and self.hashing_vectorizer:
            return self.hashing_vectorizer.transform(texts)
        else:
            raise ValueError(f"Vectorizer '{feature_type}' henüz eğitilmemiş")
    
    def get_top_features(self, feature_matrix, feature_names, labels, n_top=10):
        """Her kategori için en önemli özellikleri bulur"""
        print(f"🏆 Her kategori için top {n_top} özellik bulunuyor...")
        
        top_features = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Bu kategoriye ait dokümanları bul
            label_mask = (labels == label)
            label_features = feature_matrix[label_mask]
            
            # Ortalama TF-IDF skorları
            mean_scores = np.mean(label_features, axis=0)
            
            # En yüksek skorlu özellikleri bul
            if hasattr(mean_scores, 'A1'):  # sparse matrix ise
                mean_scores = mean_scores.A1
            
            top_indices = np.argsort(mean_scores)[::-1][:n_top]
            top_feature_names = [feature_names[i] for i in top_indices]
            top_scores = [mean_scores[i] for i in top_indices]
            
            top_features[label] = list(zip(top_feature_names, top_scores))
        
        return top_features
    
    def print_top_features(self, top_features):
        """En önemli özellikleri yazdırır"""
        print("\n🏆 KATEGORİ BAŞINA EN ÖNEMLİ ÖZELLİKLER")
        print("=" * 60)
        
        for category, features in top_features.items():
            print(f"\n📂 {category.upper()}:")
            for i, (feature, score) in enumerate(features, 1):
                print(f"   {i:2d}. {feature:<20} ({score:.4f})")

    def extract_all_features(self, texts, max_tfidf_features=5000, max_count_features=5000):
        """Tüm özellik türlerini çıkarır ve birleştirir"""
        print("🔧 Tüm özellikler çıkarılıyor...")
        
        # TF-IDF özellikleri
        try:
            # Test için daha düşük min_df kullan
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_tfidf_features,
                ngram_range=(1, 2),
                min_df=1,  # Test için daha düşük threshold
                max_df=0.95,
                sublinear_tf=True,
                strip_accents='unicode'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        except ValueError as e:
            print(f"⚠️ TF-IDF hatası: {e}, varsayılan matris kullanılıyor")
            tfidf_matrix = np.zeros((len(texts), 10))
            tfidf_feature_names = [f"tfidf_{i}" for i in range(10)]
        
        # İstatistiksel özellikler
        stat_features = self.extract_statistical_features(texts)
        
        # Kelime bilgisi özellikleri
        lexical_features = self.extract_lexical_features(texts)
        
        # Özellikleri birleştir
        import scipy.sparse as sp
        if sp.issparse(tfidf_matrix):
            tfidf_dense = tfidf_matrix.toarray()
        else:
            tfidf_dense = tfidf_matrix
            
        combined_features = np.hstack([
            tfidf_dense,
            stat_features.values,
            lexical_features.values
        ])
        
        # Özellik isimlerini birleştir
        all_feature_names = (
            list(tfidf_feature_names) + 
            list(stat_features.columns) + 
            list(lexical_features.columns)
        )
        
        print(f"✅ Toplam {combined_features.shape[1]} özellik çıkarıldı")
        return combined_features, all_feature_names

def demo_feature_extraction():
    """Özellik çıkarma demo'su"""
    print("🧪 Özellik Çıkarma Demo'su")
    print("=" * 50)
    
    # Örnek metinler
    sample_texts = [
        "Kredi kartımdan para çekildi ama rezervasyonum onaylanmadı! Çok acil.",
        "Şifremi unuttum, nasıl değiştirebilirim? Lütfen yardım edin.",
        "Site çok yavaş yükleniyor. Bu durumda çok üzgünüm.",
        "Merhaba, çalışma saatleriniz nedir? Teşekkürler.",
        "Personel çok kaba davrandı. Şikayet etmek istiyorum!"
    ]
    
    labels = ["payment_issue", "user_error", "technical_issue", "general_info", "complaint"]
    
    # Feature extractor oluştur
    extractor = FeatureExtractor()
    
    # TF-IDF özellikleri
    tfidf_matrix, tfidf_names = extractor.extract_tfidf_features(sample_texts, max_features=100)
    
    # En önemli özellikleri bul
    top_features = extractor.get_top_features(tfidf_matrix, tfidf_names, labels, n_top=5)
    extractor.print_top_features(top_features)
    
    # İstatistiksel özellikler
    stat_features = extractor.extract_statistical_features(sample_texts)
    print(f"\n📊 İstatistiksel Özellikler:")
    print(stat_features.head())

if __name__ == "__main__":
    demo_feature_extraction()
