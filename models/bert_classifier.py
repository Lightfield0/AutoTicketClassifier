"""
🤖 BERT Classifier
Transformer-based deep learning modeli
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TicketDataset(Dataset):
    """Ticket metinleri için PyTorch dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier(nn.Module):
    """BERT-based sınıflandırıcı"""
    
    def __init__(self, model_name, n_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token'ını kullan
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits

class BERTTextClassifier:
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased', max_length=128):
        """
        BERT Text Classifier
        
        Args:
            model_name: Kullanılacak BERT modeli
            max_length: Maksimum token uzunluğu
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model
        self.model = None
        self.is_trained = False
        self.classes = None
        self.label_to_idx = None
        self.idx_to_label = None
        
        print(f"🤖 BERT Classifier başlatıldı")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Max Length: {max_length}")
    
    def prepare_data(self, texts, labels):
        """Veriyi hazırla ve label encoding yap"""
        # Unique labels
        unique_labels = sorted(list(set(labels)))
        self.classes = unique_labels
        
        # Label encoding
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Labels'ı sayılara çevir
        encoded_labels = [self.label_to_idx[label] for label in labels]
        
        return texts, encoded_labels
    
    def create_data_loader(self, texts, labels, batch_size=16, shuffle=True):
        """DataLoader oluştur"""
        dataset = TicketDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Windows uyumluluğu için
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=3, batch_size=16, learning_rate=2e-5):
        """Modeli eğit"""
        print("🤖 BERT eğitimi başlıyor...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Learning Rate: {learning_rate}")
        
        start_time = time.time()
        
        # Veriyi hazırla
        X_train, y_train_encoded = self.prepare_data(X_train, y_train)
        
        # Validation veri varsa hazırla
        if X_val is not None and y_val is not None:
            _, y_val_encoded = self.prepare_data(X_val, y_val)
        
        # Model oluştur
        n_classes = len(self.classes)
        self.model = BERTClassifier(
            model_name=self.model_name,
            n_classes=n_classes
        ).to(self.device)
        
        # DataLoader'ları oluştur
        train_loader = self.create_data_loader(
            X_train, y_train_encoded, batch_size, shuffle=True
        )
        
        if X_val is not None:
            val_loader = self.create_data_loader(
                X_val, y_val_encoded, batch_size, shuffle=False
            )
        
        # Optimizer ve scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            # Training
            total_train_loss = 0
            train_progress = tqdm(train_loader, desc="Training")
            
            for batch in train_progress:
                # Batch'i device'a taşı
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            
            print(f"   Ortalama Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            if X_val is not None:
                val_loss, val_accuracy = self._evaluate(val_loader, criterion)
                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_accuracy)
                
                print(f"   Validation Loss: {val_loss:.4f}")
                print(f"   Validation Accuracy: {val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"\n✅ Eğitim tamamlandı! Süre: {training_time/60:.1f} dakika")
        
        return training_time, training_history
    
    def _evaluate(self, data_loader, criterion):
        """Validation/Test değerlendirmesi"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Predictions
                _, preds = torch.max(logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        self.model.train()  # Geri training moduna al
        
        return avg_loss, accuracy
    
    def predict(self, texts, batch_size=16):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Dummy labels (sadece DataLoader için)
        dummy_labels = [0] * len(texts)
        
        # DataLoader oluştur
        data_loader = self.create_data_loader(
            texts, dummy_labels, batch_size, shuffle=False
        )
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                _, preds = torch.max(logits, dim=1)
                
                predictions.extend(preds.cpu().tolist())
        
        # Index'leri label'lara çevir
        predicted_labels = [self.idx_to_label[idx] for idx in predictions]
        
        return predicted_labels
    
    def predict_proba(self, texts, batch_size=16):
        """Olasılık tahminleri"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        dummy_labels = [0] * len(texts)
        data_loader = self.create_data_loader(
            texts, dummy_labels, batch_size, shuffle=False
        )
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Computing probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def save_model(self, filepath):
        """Modeli kaydet"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Model state'ini kaydet
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'classes': self.classes,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'is_trained': self.is_trained
        }, filepath)
        
        print(f"💾 BERT modeli kaydedildi: {filepath}")
    
    def load_model(self, filepath):
        """Modeli yükle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Model parametrelerini yükle
        self.model_name = checkpoint['model_name']
        self.max_length = checkpoint['max_length']
        self.classes = checkpoint['classes']
        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = checkpoint['idx_to_label']
        self.is_trained = checkpoint['is_trained']
        
        # Model oluştur ve state yükle
        n_classes = len(self.classes)
        self.model = BERTClassifier(
            model_name=self.model_name,
            n_classes=n_classes
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"📂 BERT modeli yüklendi: {filepath}")
    
    def get_model_info(self):
        """Model bilgilerini getir"""
        if not self.is_trained:
            return "Model henüz eğitilmemiş"
        
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'n_classes': len(self.classes),
            'classes': self.classes,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

def train_bert_pipeline(X_train, y_train, X_test, y_test, X_val=None, y_val=None,
                       model_name='dbmdz/bert-base-turkish-cased', epochs=3):
    """BERT eğitim pipeline'ı"""
    print("🚀 BERT Pipeline Başlıyor")
    print("="*40)
    
    # BERT classifier oluştur
    bert_classifier = BERTTextClassifier(model_name=model_name)
    
    # Eğit
    training_time, history = bert_classifier.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Test et
    print("\n🧪 Test seti üzerinde değerlendirme...")
    start_time = time.time()
    y_pred = bert_classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Test Sonuçları:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Tahmin süresi: {prediction_time:.2f}s")
    
    # Detaylı rapor
    print(f"\n📋 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=bert_classifier.classes))
    
    # Model bilgileri
    print(f"\n🔍 Model Bilgileri:")
    model_info = bert_classifier.get_model_info()
    for key, value in model_info.items():
        if key != 'classes':
            print(f"   {key}: {value}")
    
    return bert_classifier, {
        'accuracy': accuracy,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'training_history': history
    }

def demo_bert():
    """BERT demo'su"""
    print("🧪 BERT Demo")
    print("="*20)
    
    # Küçük örnek veri
    sample_texts = [
        "Kredi kartımdan para çekildi ama rezervasyonum onaylanmadı",
        "Şifremi unuttum, nasıl değiştirebilirim?",
        "Site çok yavaş yükleniyor",
        "Çalışma saatleriniz nedir?",
        "Personel çok kaba davrandı"
    ] * 20  # Daha büyük dataset için çoğalt
    
    sample_labels = [
        "payment_issue", "user_error", "technical_issue", 
        "general_info", "complaint"
    ] * 20
    
    # Train-test split
    split_idx = int(len(sample_texts) * 0.8)
    X_train = sample_texts[:split_idx]
    y_train = sample_labels[:split_idx]
    X_test = sample_texts[split_idx:]
    y_test = sample_labels[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Pipeline'ı çalıştır (çok kısa eğitim)
    bert_model, results = train_bert_pipeline(
        X_train, y_train, X_test, y_test,
        epochs=1  # Demo için 1 epoch
    )

if __name__ == "__main__":
    # GPU kontrolü
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    demo_bert()
