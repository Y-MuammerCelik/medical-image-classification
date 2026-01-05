"""
Yardımcı Fonksiyonlar
=====================
Genel amaçlı yardımcı fonksiyonlar ve araçlar.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


def set_seed(seed=42):
    """
    Tekrarlanabilirlik için rastgelelik tohumunu ayarlar
    
    Args:
        seed: Tohum değeri
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Rastgelelik tohumu ayarlandı: {seed}")


def get_device():
    """
    Kullanılabilir cihazı döndürür (GPU varsa GPU, yoksa CPU)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU kullanılıyor: {torch.cuda.get_device_name(0)}")
        print(f"GPU bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("GPU bulunamadı, CPU kullanılıyor.")
    
    return device


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize edilmiş tensörü orijinal değerlerine döndürür
    
    Args:
        tensor: Normalize edilmiş görüntü tensörü
        mean: Normalizasyon ortalaması
        std: Normalizasyon standart sapması
    
    Returns:
        Denormalize edilmiş tensör
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def show_batch(dataloader, class_names, num_images=16):
    """
    Bir batch görüntüyü görselleştirir
    
    Args:
        dataloader: DataLoader
        class_names: Sınıf isimleri
        num_images: Gösterilecek görüntü sayısı
    """
    images, labels = next(iter(dataloader))
    
    # Grid boyutu hesapla
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_images):
        if i < len(images):
            img = denormalize(images[i])
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f'{class_names[labels[i]]}', fontsize=10)
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.suptitle('Örnek Görüntüler', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def count_samples_per_class(dataloader, class_names):
    """
    Her sınıftaki örnek sayısını hesaplar
    
    Args:
        dataloader: DataLoader
        class_names: Sınıf isimleri
    
    Returns:
        dict: Sınıf başına örnek sayısı
    """
    class_counts = {name: 0 for name in class_names}
    
    for _, labels in dataloader:
        for label in labels:
            class_counts[class_names[label.item()]] += 1
    
    print("\nSınıf dağılımı:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    
    return class_counts


def save_model(model, path, optimizer=None, epoch=None, loss=None, acc=None):
    """
    Model ağırlıklarını kaydeder
    
    Args:
        model: PyTorch modeli
        path: Kayıt yolu
        optimizer: Optimizer (opsiyonel)
        epoch: Epoch numarası
        loss: Loss değeri
        acc: Accuracy değeri
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if acc is not None:
        checkpoint['accuracy'] = acc
    
    torch.save(checkpoint, path)
    print(f"Model kaydedildi: {path}")


def load_model(model, path, optimizer=None, device='cuda'):
    """
    Model ağırlıklarını yükler
    
    Args:
        model: PyTorch modeli
        path: Kayıt yolu
        optimizer: Optimizer (opsiyonel)
        device: Cihaz
    
    Returns:
        model, checkpoint bilgileri
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model yüklendi: {path}")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'accuracy' in checkpoint:
        print(f"  Accuracy: {checkpoint['accuracy']*100:.2f}%")
    
    return model, checkpoint


def print_model_summary(model, input_size=(3, 224, 224)):
    """
    Model özetini yazdırır
    
    Args:
        model: PyTorch modeli
        input_size: Giriş boyutu
    """
    print("\n" + "=" * 60)
    print("MODEL ÖZETİ")
    print("=" * 60)
    print(model)
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "-" * 60)
    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")
    print(f"Eğitilemez parametre: {total_params - trainable_params:,}")
    print("=" * 60)


def create_submission_file(predictions, test_loader, save_path='submission.csv'):
    """
    Kaggle submission dosyası oluşturur
    
    Args:
        predictions: Model tahminleri
        test_loader: Test DataLoader
        save_path: Kayıt yolu
    """
    import pandas as pd
    
    # Dosya isimleri
    file_paths = [path for path, _ in test_loader.dataset.samples]
    file_names = [os.path.basename(p) for p in file_paths]
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'filename': file_names,
        'prediction': predictions
    })
    
    df.to_csv(save_path, index=False)
    print(f"Submission dosyası oluşturuldu: {save_path}")


if __name__ == "__main__":
    # Basit test
    print("Utils modülü test ediliyor...")
    set_seed(42)
    device = get_device()
    print("Test başarılı!")
