"""
Veri Yükleme Modülü
==================
Bu modül, Knee Osteoarthritis veri setini PyTorch için hazırlar.
ImageFolder ve DataLoader yapılarını kullanarak train, val ve test verilerini yükler.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(mode='train', img_size=224):
    """
    Görüntü dönüşümlerini döndürür.
    
    Args:
        mode (str): 'train', 'val' veya 'test'
        img_size (int): Hedef görüntü boyutu (varsayılan: 224)
    
    Returns:
        transforms.Compose: Dönüşüm pipeline'ı
    """
    # ImageNet normalizasyon değerleri
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if mode == 'train':
        # Eğitim için veri artırma teknikleri
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validasyon ve test için sadece resize ve normalize
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    return transform


def create_data_loaders(data_dir, batch_size=32, num_workers=4, img_size=224):
    """
    Train, validation ve test DataLoader'larını oluşturur.
    
    Args:
        data_dir (str): Veri setinin ana dizini
        batch_size (int): Batch boyutu
        num_workers (int): Paralel veri yükleme için işçi sayısı
        img_size (int): Hedef görüntü boyutu
    
    Returns:
        dict: 'train', 'val', 'test' anahtarlarına sahip DataLoader sözlüğü
        dict: 'train', 'val', 'test' anahtarlarına sahip dataset boyutları
        list: Sınıf isimleri
    """
    # Veri seti yolları
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Transforms
    train_transform = get_transforms('train', img_size)
    val_transform = get_transforms('val', img_size)
    test_transform = get_transforms('test', img_size)
    
    # ImageFolder ile veri setlerini yükle
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    # Sınıf isimleri
    class_names = train_dataset.classes
    print(f"Sınıflar: {class_names}")
    print(f"Sınıf sayısı: {len(class_names)}")
    
    # pin_memory sadece GPU varsa kullanılır (CPU'da gereksiz uyarı çıkmaması için)
    use_pin_memory = torch.cuda.is_available()
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    # DataLoader sözlüğü
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Dataset boyutları
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    print(f"\nVeri seti boyutları:")
    print(f"  Eğitim: {dataset_sizes['train']} görüntü")
    print(f"  Validasyon: {dataset_sizes['val']} görüntü")
    print(f"  Test: {dataset_sizes['test']} görüntü")
    
    return dataloaders, dataset_sizes, class_names


def get_class_weights(data_dir):
    """
    Dengesiz veri setleri için sınıf ağırlıklarını hesaplar.
    
    Args:
        data_dir (str): Eğitim veri seti dizini
    
    Returns:
        torch.Tensor: Sınıf ağırlıkları
    """
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(root=train_dir)
    
    # Her sınıftaki örnek sayısını hesapla
    class_counts = torch.zeros(len(train_dataset.classes))
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    # Ağırlıkları hesapla (tersi orantılı)
    total_samples = len(train_dataset)
    class_weights = total_samples / (len(train_dataset.classes) * class_counts)
    
    print("\nSınıf dağılımı ve ağırlıkları:")
    for i, (name, count, weight) in enumerate(zip(
            train_dataset.classes, class_counts, class_weights)):
        print(f"  {name}: {int(count)} örnek, ağırlık: {weight:.4f}")
    
    return class_weights


if __name__ == "__main__":
    # Test için
    data_dir = "../data"
    
    print("=" * 50)
    print("Veri Yükleme Testi")
    print("=" * 50)
    
    dataloaders, sizes, classes = create_data_loaders(data_dir, batch_size=32)
    
    # Bir batch örneği al
    images, labels = next(iter(dataloaders['train']))
    print(f"\nÖrnek batch şekli: {images.shape}")
    print(f"Etiket şekli: {labels.shape}")
    print(f"Etiket örnekleri: {labels[:5]}")
