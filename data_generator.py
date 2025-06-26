"""
🎫 AutoTicket Classifier - Veri Üretici
Gerçekçi müşteri destek talepleri oluşturur
"""

import json
import random
import pandas as pd
from datetime import datetime, timedelta
import re

class TicketDataGenerator:
    def __init__(self):
        # Kategori tanımları
        self.categories = {
            "payment_issue": "Ödeme Sorunu",
            "reservation_problem": "Rezervasyon Problemi", 
            "user_error": "Kullanıcı Hatası",
            "complaint": "Şikayet",
            "general_info": "Genel Bilgi",
            "technical_issue": "Teknik Sorun"
        }
        
        # Her kategori için şablon mesajlar
        self.templates = {
            "payment_issue": [
                "Kredi kartımdan para çekildi ama rezervasyonum onaylanmadı",
                "Ödeme yaparken hata aldım, param gitti mi?",
                "Fatura yanlış geldi, düzeltebilir misiniz?",
                "İade işlemim ne zaman hesabıma geçer?",
                "Ödeme sayfası açılmıyor, başka yolu var mı?",
                "Çift ödeme yapmışım, birini iade edin",
                "Kredi kartı bilgilerim kaydedilmiyor",
                "Ödeme onayı geldi ama rezervasyon yok",
                "Para çekildi ama e-posta gelmedi",
                "Taksit seçeneği neden çıkmıyor?",
                "Yurtdışı kartı kabul etmiyor",
                "Mobil ödeme sorunu yaşıyorum",
                "Havale ile ödeme yapabilir miyim?",
                "Ödeme geçmişimi nasıl görebilirim?",
                "Fatura adresimi değiştirmek istiyorum"
            ],
            
            "reservation_problem": [
                "Rezervasyonumu iptal etmek istiyorum",
                "Tarih değişikliği yapabilir miyim?",
                "Rezervasyon onayı gelmedi",
                "Yanlış tarihe rezervasyon yaptım",
                "Konuk sayısını artırabilir miyim?",
                "Rezervasyonum görünmüyor sistemde",
                "İptal ücreti ne kadar?",
                "Son dakika rezervasyon mümkün mü?",
                "Grup rezervasyonu yapmak istiyorum",
                "Rezervasyonumu başkasına devredebilir miyim?",
                "Özel istek ekleyebilir miyim?",
                "Rezervasyon kodu nerede?",
                "Check-in saatini değiştirebilir miyim?",
                "Erken check-in mümkün mü?",
                "Rezervasyonum onaylandı mı?"
            ],
            
            "user_error": [
                "Şifremi unuttum, nasıl değiştirebilirim?",
                "Hesabıma giriş yapamıyorum",
                "E-posta adresimi güncellemek istiyorum",
                "Profil fotoğrafım yüklenmiyor",
                "İki hesabım var, birini silebilir miyiz?",
                "Telefon numaramı değiştirdim",
                "Doğrulama kodu gelmiyor",
                "Hesabım bloke oldu galiba",
                "Kimlik doğrulama sorunu",
                "Hesabımı nasıl silerim?",
                "Kişisel bilgilerimi güncellemek istiyorum",
                "İki faktörlü doğrulamayı açamıyorum",
                "Bildirim ayarlarını değiştirmek istiyorum",
                "Dil ayarı nasıl değişir?",
                "Zaman dilimi yanlış görünüyor"
            ],
            
            "complaint": [
                "Hizmet kalitesi çok kötüydü",
                "Personel çok kaba davrandı",
                "Beklenen standart yakalanmadı",
                "Temizlik sorunları vardı",
                "Gürültü sorunu yaşadık",
                "Verilen bilgiler yanlıştı",
                "Randevu saatimize geç kaldınız",
                "Bu fiyata bu hizmet olmaz",
                "Müşteri temsilciniz ilgisizdi",
                "Vaatlenen hizmet verilmedi",
                "Şikayet konusunda çözüm istiyorum",
                "Memnuniyetsizliğimi belirtmek istiyorum",
                "Yaşadığım olumsuzluk hakkında",
                "Hizmet standartları düşük",
                "Hayal kırıklığı yaşadım"
            ],
            
            "general_info": [
                "Çalışma saatleriniz nedir?",
                "Hangi şehirlerde hizmet veriyorsunuz?",
                "Fiyat listesini alabilir miyim?",
                "Yeni müşteri indirimi var mı?",
                "Nasıl rezervasyon yapabilirim?",
                "Hangi ödeme yöntemlerini kabul ediyorsunuz?",
                "İptal politikanız nasıl?",
                "Grup indirimi yapıyor musunuz?",
                "Sezonluk fiyatlar ne zaman değişir?",
                "Mobil uygulama var mı?",
                "Sadakat programınız var mı?",
                "Öğrenci indirimi uyguluyor musunuz?",
                "Yaş sınırı var mı?",
                "Evcil hayvan kabul ediyor musunuz?",
                "Özel günlerde açık mısınız?"
            ],
            
            "technical_issue": [
                "Site çok yavaş yükleniyor",
                "Mobil uygulamada crash oluyor",
                "Sayfa yüklenmiyor sürekli",
                "Arama özelliği çalışmıyor",
                "Resimler görünmüyor",
                "404 hatası alıyorum",
                "Filtreleme özelliği bozuk",
                "Login olunca sayfa donuyor",
                "Sepete ekleme çalışmıyor",
                "Form gönderme hatası",
                "Çıkış yapamıyorum",
                "Bildirimler gelmiyor",
                "Harita yüklenmiyor",
                "Video oynatılmıyor",
                "PDF indirme sorunu"
            ]
        }
        
        # Varyasyon için ek kelimeler
        self.variations = {
            "polite_start": [
                "Merhaba,", "Selam,", "İyi günler,", "Merhabalar,", 
                "Sayın yetkili,", "Arkadaşlar,", "", ""
            ],
            "polite_end": [
                "Teşekkürler.", "Yardımınız için teşekkürler.", 
                "İyi günler.", "Saygılar.", "Sevgiler.", 
                "Cevabınızı bekliyorum.", ""
            ],
            "urgency": [
                "ACİL: ", "URGENT: ", "ÖNEMLİ: ", "HEMEN: ", ""
            ],
            "emotions": [
                "çok üzgünüm", "hayal kırıklığına uğradım", "memnun değilim",
                "endişeliyim", "şaşırdım", "çok memnunum", "mutluyum"
            ]
        }

    def generate_variation(self, template):
        """Şablona varyasyon ekler"""
        message = template
        
        # Rastgele başlangıç
        if random.random() < 0.3:
            start = random.choice(self.variations["polite_start"])
            if start:
                message = start + " " + message
        
        # Rastgele bitiş
        if random.random() < 0.4:
            end = random.choice(self.variations["polite_end"])
            if end:
                message = message + " " + end
                
        # Aciliyet
        if random.random() < 0.1:
            urgency = random.choice(self.variations["urgency"])
            if urgency:
                message = urgency + message
                
        # Duygu ekleme
        if random.random() < 0.2:
            emotion = random.choice(self.variations["emotions"])
            message = message + f" Bu durumda {emotion}."
            
        return message

    def generate_tickets(self, num_tickets=2000):
        """Belirtilen sayıda ticket üretir"""
        tickets = []
        
        for i in range(num_tickets):
            # Rastgele kategori seç
            category = random.choice(list(self.categories.keys()))
            
            # Şablon seç ve varyasyon ekle
            template = random.choice(self.templates[category])
            message = self.generate_variation(template)
            
            # Metadata ekle
            ticket = {
                "id": f"TICKET_{i+1:06d}",
                "message": message,
                "category": category,
                "category_tr": self.categories[category],
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat(),
                "priority": random.choice(["low", "medium", "high"]),
                "status": random.choice(["open", "in_progress", "resolved", "closed"]),
                "customer_id": f"CUST_{random.randint(1000, 9999)}",
                "channel": random.choice(["email", "chat", "phone", "web_form"])
            }
            
            tickets.append(ticket)
            
        return tickets

    def generate_comprehensive_dataset(self, n_samples=1000):
        """Comprehensive dataset oluştur (DataFrame döndürür)"""
        tickets = self.generate_tickets(num_tickets=n_samples)
        df = pd.DataFrame(tickets)
        return df

    def save_data(self, tickets, format_type="both"):
        """Veriyi JSON ve/veya CSV olarak kaydeder"""
        
        if format_type in ["json", "both"]:
            # JSON formatında kaydet
            with open("data/raw_tickets.json", "w", encoding="utf-8") as f:
                json.dump(tickets, f, ensure_ascii=False, indent=2)
            print(f"✅ {len(tickets)} ticket JSON olarak kaydedildi: data/raw_tickets.json")
            
        if format_type in ["csv", "both"]:
            # CSV formatında kaydet
            df = pd.DataFrame(tickets)
            df.to_csv("data/processed_data.csv", index=False, encoding="utf-8")
            print(f"✅ {len(tickets)} ticket CSV olarak kaydedildi: data/processed_data.csv")
            
        return df if format_type in ["csv", "both"] else pd.DataFrame(tickets)

    def analyze_data(self, df):
        """Üretilen verinin analizini yapar"""
        print("\n📊 VERİ ANALİZİ")
        print("=" * 50)
        
        print(f"📝 Toplam Ticket Sayısı: {len(df)}")
        print(f"📅 Tarih Aralığı: {df['timestamp'].min()[:10]} - {df['timestamp'].max()[:10]}")
        
        print("\n🏷️ Kategori Dağılımı:")
        category_counts = df['category_tr'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count} (%{percentage:.1f})")
            
        print("\n⚡ Öncelik Dağılımı:")
        priority_counts = df['priority'].value_counts()
        for priority, count in priority_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {priority}: {count} (%{percentage:.1f})")
            
        print("\n📊 İletişim Kanalı:")
        channel_counts = df['channel'].value_counts()
        for channel, count in channel_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {channel}: {count} (%{percentage:.1f})")
            
        print("\n📝 Ortalama Mesaj Uzunluğu:")
        avg_length = df['message'].str.len().mean()
        print(f"   {avg_length:.1f} karakter")
        
        print("\n💡 Örnek Mesajlar:")
        for category in df['category'].unique():
            sample = df[df['category'] == category]['message'].iloc[0]
            category_tr = df[df['category'] == category]['category_tr'].iloc[0]
            print(f"\n   🏷️ {category_tr}:")
            print(f"   \"{sample}\"")

def main():
    print("🎫 AutoTicket Classifier - Veri Üretici")
    print("=" * 50)
    
    # Veri üretici oluştur
    generator = TicketDataGenerator()
    
    print("📝 Sentetik veri üretiliyor...")
    tickets = generator.generate_tickets(num_tickets=2000)
    
    print("💾 Veri kaydediliyor...")
    df = generator.save_data(tickets, format_type="both")
    
    print("📊 Veri analizi yapılıyor...")
    generator.analyze_data(df)
    
    print("\n✅ Veri üretimi tamamlandı!")
    print("📂 Dosyalar:")
    print("   - data/raw_tickets.json")
    print("   - data/processed_data.csv")
    print("\n🚀 Sıradaki adım: python train_models.py")

if __name__ == "__main__":
    main()
