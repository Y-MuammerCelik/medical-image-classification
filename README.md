# 🦴 Diz Osteoartrit (KL Grading) Sınıflandırması

**Sinir Ağları Dersi - Bitirme Ödevi**

Bu proje, diz röntgen görüntülerini Kellgren-Lawrence (KL) skalasına göre 5 sınıfa ayıran bir derin öğrenme modeli içerir.

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Veri Seti](#-veri-seti)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Mimarileri](#-model-mimarileri)
- [Sonuçlar](#-sonuçlar)
- [Dosya Yapısı](#-dosya-yapısı)

## 🎯 Proje Hakkında

Diz osteoartriti, eklem kıkırdağının aşınmasıyla karakterize yaygın bir hastalıktır. Bu projede, röntgen görüntülerinden otomatik evreleme yapan bir CNN modeli geliştirilmiştir.

### Kellgren-Lawrence (KL) Skalası

| Grade | Açıklama |
|-------|----------|
| 0 | Normal - Osteoartrit yok |
| 1 | Şüpheli - Olası osteofit oluşumu |
| 2 | Minimal - Kesin osteofit, olası eklem aralığı daralması |
| 3 | Orta - Belirgin osteofit, eklem aralığı daralması |
| 4 | Şiddetli - Belirgin eklem aralığı daralması, skleroz |

## 📊 Veri Seti

Kaggle'dan indirilen "Knee Osteoarthritis Dataset with Severity Grading" veri seti kullanılmıştır.

- **Kaynak**: [Kaggle - Knee Osteoarthritis Dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
- **Görüntü boyutu**: 224x224 piksel (yeniden boyutlandırılmış)
- **Sınıf sayısı**: 5 (Grade 0-4)

## 🛠️ Kurulum

### Gereksinimler

- Python 3.8+
- PyTorch 1.9+
- CUDA (GPU kullanımı için, opsiyonel)

### Adımlar

1. **Depoyu klonlayın veya indirin**

2. **Sanal ortam oluşturun (önerilir)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Gereksinimleri yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

4. **Veri setini hazırlayın**
   
   `archive.zip` dosyasını `data` klasörüne çıkarın:
   ```bash
   # PowerShell
   Expand-Archive -Path archive.zip -DestinationPath data
   ```

## 🚀 Kullanım

### Temel Eğitim (Özgün CNN)

```bash
python main.py --model cnn --epochs 25 --batch_size 32
```

### Transfer Learning (ResNet18)

```bash
python main.py --model resnet18 --pretrained --epochs 15
```

### Özellik Çıkarıcı Dondurarak Eğitim

```bash
python main.py --model resnet18 --pretrained --freeze --epochs 10
```

### Sadece Değerlendirme

```bash
python main.py --evaluate --checkpoint checkpoints/best_model.pth
```

### Tüm Parametreler

```bash
python main.py --help
```

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--model` | cnn | Model tipi: `cnn` veya `resnet18` |
| `--epochs` | 25 | Eğitim epoch sayısı |
| `--batch_size` | 32 | Batch boyutu |
| `--lr` | 0.001 | Learning rate |
| `--optimizer` | adam | Optimizer: `adam`, `adamw`, `sgd` |
| `--pretrained` | False | Transfer learning için önceden eğitilmiş ağırlıklar |
| `--freeze` | False | Özellik çıkarıcı katmanları dondur |
| `--class_weights` | False | Dengesiz sınıflar için ağırlık kullan |
| `--early_stopping` | 5 | Early stopping patience |

## 🏗️ Model Mimarileri

### 1. Özgün CNN (`KneeOsteoarthritisCNN`)

4 evrişim katmanlı özel tasarlanmış bir CNN:

```
Input: 3x224x224
    ↓
Conv2d(3→32) → BatchNorm → ReLU → MaxPool    # 32x112x112
    ↓
Conv2d(32→64) → BatchNorm → ReLU → MaxPool   # 64x56x56
    ↓
Conv2d(64→128) → BatchNorm → ReLU → MaxPool  # 128x28x28
    ↓
Conv2d(128→256) → BatchNorm → ReLU → MaxPool # 256x14x14
    ↓
AdaptiveAvgPool → Flatten → Dropout(0.5)
    ↓
Linear(256→128) → ReLU → Dropout(0.3)
    ↓
Linear(128→5)  # 5 sınıf çıkışı
```

**Toplam parametre**: ~1.2M

### 2. Transfer Learning (`ResNet18TransferModel`)

ImageNet üzerinde önceden eğitilmiş ResNet18:

- Son tam bağlantılı katman 5 sınıfa uyarlanmış
- Opsiyonel olarak özellik çıkarıcı katmanlar dondurulabilir
- Fine-tuning için tüm katmanlar açılabilir

**Toplam parametre**: ~11.2M

## 📈 Sonuçlar

Eğitim sonuçları `results/` klasöründe kaydedilir:

- `confusion_matrix.png` - Karışıklık matrisi
- `training_history.png` - Loss ve accuracy grafikleri
- `class_distribution.png` - Sınıf dağılımı

### Değerlendirme Metrikleri

- **Accuracy**: Genel doğruluk oranı
- **Precision**: Her sınıf için kesinlik
- **Recall**: Her sınıf için duyarlılık
- **F1-Score**: Precision ve Recall'ın harmonik ortalaması
- **Confusion Matrix**: Sınıflar arası karışıklık

## 📁 Dosya Yapısı

```
cnnödevi/
├── data/                    # Veri seti
│   ├── train/              # Eğitim görüntüleri
│   ├── val/                # Validasyon görüntüleri
│   └── test/               # Test görüntüleri
├── src/
│   ├── __init__.py         # Paket başlatma
│   ├── data_loader.py      # Veri yükleme ve transforms
│   ├── models.py           # Model mimarileri
│   ├── train.py            # Eğitim döngüsü
│   ├── evaluate.py         # Değerlendirme metrikleri
│   └── utils.py            # Yardımcı fonksiyonlar
├── checkpoints/            # Kaydedilen modeller
├── results/                # Değerlendirme sonuçları
├── main.py                 # Ana çalıştırma scripti
├── requirements.txt        # Python gereksinimleri
└── README.md               # Bu dosya
```

## 🔧 Teknik Detaylar

### Veri Ön İşleme

**Eğitim seti:**
- Resize: 224x224
- RandomHorizontalFlip: p=0.5
- RandomRotation: ±10°
- ColorJitter: brightness=0.2, contrast=0.2
- Normalize: ImageNet istatistikleri

**Validasyon/Test seti:**
- Resize: 224x224
- Normalize: ImageNet istatistikleri

### Eğitim Stratejisi

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: StepLR (step_size=10, gamma=0.1)
- **Early Stopping**: Patience=5 epoch
- **Regularization**: Dropout (0.5, 0.3)

## 📝 Lisans

Bu proje eğitim amaçlı hazırlanmıştır.

## 👤 Yazar

Sinir Ağları Dersi Bitirme Ödevi

---

**Not**: GPU kullanımı eğitim süresini önemli ölçüde kısaltır. CUDA destekli bir GPU yoksa CPU kullanılacaktır, ancak eğitim süresi daha uzun olacaktır.
