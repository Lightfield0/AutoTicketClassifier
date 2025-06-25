"""
🔧 Text Preprocessing Utilities
Türkçe metin ön işleme araçları
"""

import re
import string
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd

# Suppress warnings for Turkish stemmer
warnings.filterwarnings('ignore', message='.*Turkish.*stemmer.*')

class TurkishTextPreprocessor:
    def __init__(self):
        # NLTK verilerini indir
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Türkçe stop words
        self.stop_words = set(stopwords.words('turkish'))
        
        # Ek Türkçe stop words
        turkish_custom_stops = {
            'bir', 'bu', 'şu', 'o', 've', 'ile', 'için', 'da', 'de', 'ta', 'te',
            'ki', 'mi', 'mı', 'mu', 'mü', 'var', 'yok', 'gibi', 'kadar', 'daha',
            'en', 'çok', 'az', 'biraz', 'hiç', 'her', 'bazı', 'şey', 'kez',
            'zaman', 'yer', 'nasıl', 'neden', 'niçin', 'niye', 'ne', 'hangi',
            'kim', 'kimi', 'kimin', 'kimse', 'hiçbir', 'hiçbirşey'
        }
        self.stop_words.update(turkish_custom_stops)
        
        # Türkçe stemmer - uyarıları bastır
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.stemmer = SnowballStemmer('turkish')
        except (ValueError, LookupError) as e:
            # Türkçe desteklenmiyorsa stemmer kullanma
            self.stemmer = None
            print("⚠️ Türkçe stemmer desteklenmiyor, stemming atlanacak")
        
        # Türkçe karakter normalizasyonu
        self.char_map = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'C', 'Ğ': 'G', 'I': 'I', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
        }

    def normalize_turkish_chars(self, text, remove_turkish=False):
        """Türkçe karakterleri normalize eder"""
        if remove_turkish:
            # Türkçe karakterleri İngilizce karşılıklarıyla değiştir
            for turkish_char, english_char in self.char_map.items():
                text = text.replace(turkish_char, english_char)
        return text

    def clean_text(self, text):
        """Temel metin temizliği"""
        if pd.isna(text):
            return ""
            
        # Küçük harfe çevir
        text = str(text).lower()
        
        # Email adresleri temizle
        text = re.sub(r'\S+@\S+', '', text)
        
        # URL'leri temizle
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Telefon numaralarını temizle
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'\b0\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b', '', text)
        
        # Özel karakterleri temizle (Türkçe karakterleri koru)
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        
        # Çoklu boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        
        # Başta ve sonda boşluk temizle
        text = text.strip()
        
        return text

    def remove_stopwords(self, text):
        """Stop words'leri kaldırır"""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def stem_text(self, text):
        """Kelimeleri kökenlerine indirgemer"""
        if not self.stemmer:
            # Stemmer yoksa orijinal metni döndür
            return text
            
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def preprocess_text(self, text, 
                       remove_stopwords=True, 
                       apply_stemming=False,
                       normalize_turkish=False,
                       min_length=2):
        """Tam metin ön işleme pipeline'ı"""
        
        # Temel temizlik
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Türkçe karakter normalizasyonu
        if normalize_turkish:
            text = self.normalize_turkish_chars(text, remove_turkish=True)
        
        # Stop words kaldır
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Stemming uygula
        if apply_stemming:
            text = self.stem_text(text)
        
        # Minimum uzunluk kontrolü
        words = text.split()
        words = [word for word in words if len(word) >= min_length]
        text = ' '.join(words)
        
        return text

    def preprocess_dataframe(self, df, text_column='message', 
                           new_column='processed_text', **kwargs):
        """DataFrame'deki metinleri ön işler"""
        print("🔧 Metin ön işleme başlıyor...")
        
        df[new_column] = df[text_column].apply(
            lambda x: self.preprocess_text(x, **kwargs)
        )
        
        # İstatistikler
        original_lengths = df[text_column].str.len()
        processed_lengths = df[new_column].str.len()
        
        print(f"📊 Ön İşleme Sonuçları:")
        print(f"   Ortalama uzunluk: {original_lengths.mean():.1f} → {processed_lengths.mean():.1f}")
        print(f"   Medyan uzunluk: {original_lengths.median():.1f} → {processed_lengths.median():.1f}")
        
        # Boş metinleri kontrol et
        empty_count = (df[new_column].str.len() == 0).sum()
        if empty_count > 0:
            print(f"⚠️  {empty_count} adet boş metin bulundu")
            # Boş metinleri orijinal metinle değiştir
            df.loc[df[new_column].str.len() == 0, new_column] = df.loc[df[new_column].str.len() == 0, text_column]
        
        print("✅ Metin ön işleme tamamlandı")
        return df

# Kolay kullanım için fonksiyonlar
def quick_clean(text):
    """Hızlı metin temizliği"""
    preprocessor = TurkishTextPreprocessor()
    return preprocessor.preprocess_text(text, remove_stopwords=True, apply_stemming=False)

def deep_clean(text):
    """Derin metin temizliği (stemming dahil)"""
    preprocessor = TurkishTextPreprocessor()
    return preprocessor.preprocess_text(text, remove_stopwords=True, apply_stemming=True)

def english_compatible_clean(text):
    """İngilizce uyumlu temizlik (Türkçe karakterler çevrilir)"""
    preprocessor = TurkishTextPreprocessor()
    return preprocessor.preprocess_text(text, 
                                      remove_stopwords=True, 
                                      apply_stemming=False,
                                      normalize_turkish=True)

if __name__ == "__main__":
    # Test
    preprocessor = TurkishTextPreprocessor()
    
    test_texts = [
        "Merhaba, kredi kartımdan para çekildi ama rezervasyonum onaylanmadı!",
        "Şifremi unuttum, nasıl değiştirebilirim? Çok acil...",
        "Site çok yavaş yükleniyor. Bu durumda çok üzgünüm.",
        "İyi günler, çalışma saatleriniz nedir?"
    ]
    
    print("🧪 Metin Ön İşleme Testi")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        cleaned = preprocessor.preprocess_text(text)
        print(f"\n{i}. Orijinal: {text}")
        print(f"   Temizlenmiş: {cleaned}")
