"""
Veri Yükleme Modülü
==================
Bu modül, Knee Osteoarthritis veri setini PyTorch için hazırlar.
ImageFolder ve DataLoader yapılarını kullanarak train, val ve test verilerini yükler.

Özellikler:
- Agresif data augmentation (Grade 2 iyileştirmesi için)
- Weighted Random Sampling (az örnekli sınıflar için)
- Class weighting (dengesiz veri için)
"""

import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def get_transforms(mode='train', img_size=224, aggressive=True):
    """
    Görüntü dönüşümlerini döndürür.
    
    Args:
        mode (str): 'train', 'val' veya 'test'
        img_size (int): Hedef görüntü boyutu (varsayılan: 224)
        aggressive (bool): Agresif augmentation kullan (varsayılan: True)
    
    Returns:
        transforms.Compose: Dönüşüm pipeline'ı
    """
    # ImageNet normalizasyon değerleri
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if mode == 'train':
        if aggressive:
            # Agresif veri artırma - Grade 2 gibi az örnekli sınıflar için
            transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),  # Biraz büyük resize
                transforms.RandomCrop(img_size),                    # Rastgele kırpma
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),              # Dikey çevirme
                transforms.RandomRotation(degrees=15),              # Daha geniş rotasyon
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),                          # Öteleme
                    scale=(0.9, 1.1),                              # Ölçekleme
                    shear=5                                         # Yamultma
                ),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))  # Rastgele silme
            ])
        else:
            # Normal veri artırma
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


def create_data_loaders(data_dir, batch_size=32, num_workers=4, img_size=224, 
                        use_weighted_sampler=True, aggressive_augment=True):
    """
    Train, validation ve test DataLoader'larını oluşturur.
    
    Args:
        data_dir (str): Veri setinin ana dizini
        batch_size (int): Batch boyutu
        num_workers (int): Paralel veri yükleme için işçi sayısı
        img_size (int): Hedef görüntü boyutu
        use_weighted_sampler (bool): Weighted Random Sampler kullan
        aggressive_augment (bool): Agresif augmentation kullan
    
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
    train_transform = get_transforms('train', img_size, aggressive=aggressive_augment)
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
    
    # pin_memory sadece GPU varsa kullanılır
    use_pin_memory = torch.cuda.is_available()
    
    # Weighted Random Sampler oluştur (dengesiz veri için)
    sampler = None
    shuffle = True
    
    if use_weighted_sampler:
        # Sınıf ağırlıklarını hesapla
        class_counts = torch.zeros(len(class_names))
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        
        # Her sınıfa ağırlık ata (tersi orantılı)
        class_weights = 1.0 / class_counts
        
        # Grade 2 için ekstra boost (sınıf indeksi 2)
        # Grade 2 en az örneğe sahip olduğundan 2x boost
        if len(class_names) > 2:
            class_weights[2] *= 2.0
        
        # Her örneğe ağırlık ata
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        
        # WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler kullanılırken shuffle False olmalı
        
        print("\n[OK] Weighted Random Sampler aktif (dengesiz veri icin)")
        print("  Grade 2 için 2x ağırlık boost uygulandı")
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
    
    if aggressive_augment:
        print("\n[OK] Agresif Data Augmentation aktif")
        print("  - RandomCrop, RandomAffine, RandomPerspective")
        print("  - GaussianBlur, RandomErasing, ColorJitter")
    
    return dataloaders, dataset_sizes, class_names


def get_class_weights(data_dir, boost_grade2=True):
    """
    Dengesiz veri setleri için sınıf ağırlıklarını hesaplar.
    
    Args:
        data_dir (str): Eğitim veri seti dizini
        boost_grade2 (bool): Grade 2 için ekstra ağırlık (varsayılan: True)
    
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
    
    # Grade 2 için ekstra boost (sınıf indeksi 2)
    if boost_grade2 and len(train_dataset.classes) > 2:
        original_weight = class_weights[2].item()
        class_weights[2] *= 2.5  # 2.5x boost
        print(f"\n[BOOST] Grade 2 loss agirligi boost: {original_weight:.4f} -> {class_weights[2]:.4f}")
    
    print("\nSınıf dağılımı ve ağırlıkları:")
    for i, (name, count, weight) in enumerate(zip(
            train_dataset.classes, class_counts, class_weights)):
        boost_marker = " [BOOST]" if i == 2 and boost_grade2 else ""
        print(f"  {name}: {int(count)} örnek, ağırlık: {weight:.4f}{boost_marker}")
    
    return class_weights


if __name__ == "__main__":
    # Test için
    data_dir = "../data"
    
    print("=" * 50)
    print("Veri Yükleme Testi (İyileştirilmiş)")
    print("=" * 50)
    
    dataloaders, sizes, classes = create_data_loaders(
        data_dir, 
        batch_size=32,
        use_weighted_sampler=True,
        aggressive_augment=True
    )
    
    # Bir batch örneği al
    images, labels = next(iter(dataloaders['train']))
    print(f"\nÖrnek batch şekli: {images.shape}")
    print(f"Etiket şekli: {labels.shape}")
    print(f"Etiket örnekleri: {labels[:10]}")
    
    # Class weights test
    print("\n" + "=" * 50)
    weights = get_class_weights(data_dir, boost_grade2=True)
