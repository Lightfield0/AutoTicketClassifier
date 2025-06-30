#!/usr/bin/env python3
"""
🔧 Dependency Checker - AutoTicket Classifier
Tüm bağımlılıkları kontrol eder ve eksikleri bildirir
"""

import sys
import subprocess
import importlib.util

def check_dependency(package_name, import_name=None):
    """Tek bir bağımlılığı kontrol et"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True, "✅ Kurulu"
        else:
            return False, "❌ Bulunamadı"
    except ImportError:
        return False, "❌ Import hatası"

def check_all_dependencies():
    """Tüm bağımlılıkları kontrol et"""
    print("🔍 AutoTicket Classifier - Bağımlılık Kontrolü")
    print("=" * 50)
    
    # Kritik bağımlılıklar
    critical_deps = [
        ("pandas", None),
        ("numpy", None),
        ("scikit-learn", "sklearn"),
        ("matplotlib", None),
        ("seaborn", None),
        ("streamlit", None),
        ("nltk", None),
        ("transformers", None),
        ("torch", None),
        ("plotly", None),
        ("joblib", None),
        ("tqdm", None),
        ("flask", None),
        ("fastapi", None)
    ]
    
    print("\n📦 KRITIK BAĞIMLILIKLAR:")
    print("-" * 30)
    
    missing_critical = []
    for package, import_name in critical_deps:
        status, message = check_dependency(package, import_name)
        print(f"{package:<20} {message}")
        if not status:
            missing_critical.append(package)
    
    # Opsiyonel bağımlılıklar
    optional_deps = [
        ("mlflow", None),
        ("redis", None),
        ("psutil", None),
        ("prometheus-client", "prometheus_client"),
        ("scipy", None),
        ("pyyaml", "yaml"),
        ("python-dotenv", "dotenv"),
        ("structlog", None),
        ("statsmodels", None),
        ("imbalanced-learn", "imblearn"),
        ("wordcloud", None),
        ("gunicorn", None),
        ("uvicorn", None)
    ]
    
    print("\n📦 OPSIYONEL BAĞIMLILIKLAR:")
    print("-" * 30)
    
    missing_optional = []
    for package, import_name in optional_deps:
        status, message = check_dependency(package, import_name)
        print(f"{package:<20} {message}")
        if not status:
            missing_optional.append(package)
    
    # Özet
    print("\n📊 ÖZET:")
    print("-" * 15)
    print(f"Kritik bağımlılıklar: {len(critical_deps) - len(missing_critical)}/{len(critical_deps)}")
    print(f"Opsiyonel bağımlılıklar: {len(optional_deps) - len(missing_optional)}/{len(optional_deps)}")
    
    if missing_critical:
        print(f"\n🚨 EKSIK KRITIK BAĞIMLILIKLAR:")
        for dep in missing_critical:
            print(f"   - {dep}")
        print("\n💡 Kurulum komutu:")
        print(f"pip install {' '.join(missing_critical)}")
    
    if missing_optional:
        print(f"\n⚠️ EKSIK OPSIYONEL BAĞIMLILIKLAR:")
        for dep in missing_optional:
            print(f"   - {dep}")
        print("\n💡 Kurulum komutu:")
        print(f"pip install {' '.join(missing_optional)}")
    
    if not missing_critical and not missing_optional:
        print("\n🎉 TÜM BAĞIMLILIKLAR KURULU!")
        return True
    elif not missing_critical:
        print("\n✅ Kritik bağımlılıklar tamam, proje çalışabilir!")
        return True
    else:
        print("\n❌ Kritik bağımlılıklar eksik, proje çalışmayabilir!")
        return False

def install_missing_nltk_data():
    """NLTK için gerekli data'ları indir"""
    print("\n📚 NLTK Data Kontrolü:")
    print("-" * 25)
    
    try:
        import nltk
        
        # Gerekli NLTK dataları
        nltk_data = ['punkt', 'stopwords', 'wordnet']
        
        for data_name in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
                print(f"✅ {data_name} - Kurulu")
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                    print(f"✅ {data_name} - Kurulu")
                except LookupError:
                    print(f"📦 {data_name} - İndiriliyor...")
                    nltk.download(data_name, quiet=True)
                    print(f"✅ {data_name} - İndirildi")
    
    except ImportError:
        print("❌ NLTK kurulu değil!")

if __name__ == "__main__":
    success = check_all_dependencies()
    install_missing_nltk_data()
    
    if success:
        print("\n🚀 Proje çalıştırılabilir!")
        sys.exit(0)
    else:
        print("\n🛑 Önce eksik bağımlılıkları kurun!")
        sys.exit(1)
