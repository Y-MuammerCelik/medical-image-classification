"""
Model Mimarileri
================
Bu modül, Knee Osteoarthritis sınıflandırması için:
1. Özgün CNN mimarisi
2. Transfer Learning (ResNet18) modeli
içerir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class KneeOsteoarthritisCNN(nn.Module):
    """
    Özgün CNN Mimarisi
    
    4 evrişim katmanlı, Batch Normalization ve Dropout kullanan
    özel tasarlanmış bir CNN modeli.
    
    Mimari:
        Conv1: 3 → 32 kanal, 3x3 kernel
        Conv2: 32 → 64 kanal, 3x3 kernel
        Conv3: 64 → 128 kanal, 3x3 kernel
        Conv4: 128 → 256 kanal, 3x3 kernel
        FC1: 256 → 128
        FC2: 128 → 5 (sınıf sayısı)
    """
    
    def __init__(self, num_classes=5, dropout_rate=0.5):
        """
        Args:
            num_classes (int): Çıkış sınıf sayısı (varsayılan: 5 - KL Grade 0-4)
            dropout_rate (float): Dropout oranı
        """
        super(KneeOsteoarthritisCNN, self).__init__()
        
        # Evrişim Bloğu 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Evrişim Bloğu 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Evrişim Bloğu 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Evrişim Bloğu 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Tam Bağlantılı Katmanlar
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.6)  # 0.3
        
    def forward(self, x):
        """
        İleri geçiş
        
        Args:
            x: Giriş tensörü [batch_size, 3, 224, 224]
        
        Returns:
            Çıkış tensörü [batch_size, num_classes]
        """
        # Conv Block 1: 224x224 → 112x112
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2: 112x112 → 56x56
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3: 56x56 → 28x28
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4: 28x28 → 14x14
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global Average Pooling: 14x14 → 1x1
        x = self.global_avg_pool(x)
        
        # Flatten: [batch, 256, 1, 1] → [batch, 256]
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class ResNet18TransferModel(nn.Module):
    """
    Transfer Learning Modeli (ResNet18)
    
    ImageNet üzerinde önceden eğitilmiş ResNet18 modelini kullanır.
    Son tam bağlantılı katman, 5 sınıflı çıkış için değiştirilir.
    """
    
    def __init__(self, num_classes=5, pretrained=True, freeze_features=False):
        """
        Args:
            num_classes (int): Çıkış sınıf sayısı
            pretrained (bool): Önceden eğitilmiş ağırlıkları yükle
            freeze_features (bool): Özellik çıkarıcı katmanları dondur
        """
        super(ResNet18TransferModel, self).__init__()
        
        # Önceden eğitilmiş ResNet18 yükle
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.resnet = models.resnet18(weights=weights)
            print("ResNet18 ImageNet ağırlıkları yüklendi.")
        else:
            self.resnet = models.resnet18(weights=None)
            print("ResNet18 rastgele ağırlıklarla başlatıldı.")
        
        # Özellik çıkarıcı katmanları dondur (opsiyonel)
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
            print("Özellik çıkarıcı katmanlar donduruldu.")
        
        # Son FC katmanını değiştir
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """İleri geçiş"""
        return self.resnet(x)
    
    def unfreeze_all(self):
        """Tüm katmanları eğitilebilir yap"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("Tüm katmanlar eğitilebilir hale getirildi.")
    
    def unfreeze_last_n_layers(self, n=2):
        """
        Son n layer'ı eğitilebilir yap
        
        Args:
            n (int): Eğitilebilir yapılacak son katman sayısı
        """
        layers = list(self.resnet.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Son {n} katman eğitilebilir hale getirildi.")


def get_model(model_name='cnn', num_classes=5, pretrained=True, freeze_features=False):
    """
    Model fabrika fonksiyonu
    
    Args:
        model_name (str): 'cnn' veya 'resnet18'
        num_classes (int): Sınıf sayısı
        pretrained (bool): Transfer learning için önceden eğitilmiş ağırlıklar
        freeze_features (bool): Özellik katmanlarını dondur
    
    Returns:
        nn.Module: Seçilen model
    """
    if model_name.lower() == 'cnn':
        model = KneeOsteoarthritisCNN(num_classes=num_classes)
        print("\n" + "=" * 50)
        print("Özgün CNN Modeli oluşturuldu")
        print("=" * 50)
    elif model_name.lower() == 'resnet18':
        model = ResNet18TransferModel(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_features=freeze_features
        )
        print("\n" + "=" * 50)
        print("ResNet18 Transfer Learning Modeli oluşturuldu")
        print("=" * 50)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}. 'cnn' veya 'resnet18' kullanın.")
    
    return model


def count_parameters(model):
    """
    Modelin toplam ve eğitilebilir parametre sayısını hesaplar
    
    Args:
        model: PyTorch modeli
    
    Returns:
        tuple: (toplam_parametre, eğitilebilir_parametre)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parametreleri:")
    print(f"  Toplam: {total_params:,}")
    print(f"  Eğitilebilir: {trainable_params:,}")
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Model testleri
    print("=" * 60)
    print("MODEL TESTLERİ")
    print("=" * 60)
    
    # Test girdisi
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Özgün CNN testi
    print("\n1. Özgün CNN Modeli Testi")
    print("-" * 40)
    cnn_model = get_model('cnn')
    count_parameters(cnn_model)
    output = cnn_model(dummy_input)
    print(f"Giriş şekli: {dummy_input.shape}")
    print(f"Çıkış şekli: {output.shape}")
    
    # ResNet18 testi
    print("\n2. ResNet18 Transfer Learning Testi")
    print("-" * 40)
    resnet_model = get_model('resnet18', pretrained=True, freeze_features=False)
    count_parameters(resnet_model)
    output = resnet_model(dummy_input)
    print(f"Giriş şekli: {dummy_input.shape}")
    print(f"Çıkış şekli: {output.shape}")
    
    print("\n" + "=" * 60)
    print("Tüm testler başarılı!")
    print("=" * 60)
