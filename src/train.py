"""
Eğitim Modülü
=============
Bu modül, model eğitimi için gerekli fonksiyonları içerir:
- Eğitim döngüsü
- Validasyon döngüsü
- Early stopping
- Model kaydetme
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class EarlyStopping:
    """
    Early Stopping sınıfı
    
    Validasyon kaybı belirli bir epoch sayısı boyunca iyileşmezse eğitimi durdurur.
    """
    
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        Args:
            patience (int): İyileşme olmadan beklenecek epoch sayısı
            min_delta (float): İyileşme olarak kabul edilecek minimum değişim
            verbose (bool): Durumu yazdır
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None
        
    def __call__(self, val_loss, model):
        """
        Her epoch sonunda çağrılır
        
        Args:
            val_loss: Validasyon kaybı
            model: PyTorch modeli
        
        Returns:
            bool: Early stop tetiklendi mi
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            
        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Bir epoch eğitim yapar
    
    Args:
        model: PyTorch modeli
        dataloader: Eğitim DataLoader
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        device: CPU veya CUDA
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Eğitim", leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # İleri geçiş
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Geri yayılım
        loss.backward()
        optimizer.step()
        
        # İstatistikler
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Progress bar güncelle
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * running_corrects / total_samples:.2f}%'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    """
    Validasyon yapar
    
    Args:
        model: PyTorch modeli
        dataloader: Validasyon DataLoader
        criterion: Loss fonksiyonu
        device: CPU veya CUDA
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validasyon", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, 
                scheduler=None, num_epochs=25, device='cuda', 
                save_dir='checkpoints', early_stopping_patience=5):
    """
    Model eğitimi ana fonksiyonu
    
    Args:
        model: PyTorch modeli
        dataloaders: DataLoader sözlüğü {'train', 'val'}
        dataset_sizes: Veri seti boyutları
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        scheduler: Learning rate scheduler (opsiyonel)
        num_epochs: Epoch sayısı
        device: CPU veya CUDA
        save_dir: Model kayıt dizini
        early_stopping_patience: Early stopping için patience
    
    Returns:
        model: Eğitilmiş model
        history: Eğitim geçmişi
    """
    since = time.time()
    
    # Model kayıt dizini oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    # En iyi modeli sakla
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Eğitim geçmişi
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print("\n" + "=" * 70)
    print("EĞİTİM BAŞLIYOR")
    print("=" * 70)
    print(f"Cihaz: {device}")
    print(f"Epoch sayısı: {num_epochs}")
    print(f"Eğitim örnekleri: {dataset_sizes['train']}")
    print(f"Validasyon örnekleri: {dataset_sizes['val']}")
    print("=" * 70 + "\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print("-" * 40)
        
        # Eğitim fazı
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        # Validasyon fazı
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        
        # Geçmişe kaydet
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Sonuçları yazdır
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # En iyi modeli kaydet
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Checkpoint kaydet
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ En iyi model kaydedildi! (Val Acc: {val_acc*100:.2f}%)")
        
        # Early stopping kontrolü
        if early_stopping(val_loss, model):
            print(f"\n⚠ Early stopping tetiklendi! (Epoch {epoch+1})")
            break
        
        print()
    
    # Eğitim süresi
    time_elapsed = time.time() - since
    print("=" * 70)
    print(f"Eğitim tamamlandı! Süre: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"En iyi validasyon doğruluğu: {best_acc*100:.2f}% (Epoch {best_epoch})")
    print("=" * 70)
    
    # En iyi ağırlıkları yükle
    model.load_state_dict(best_model_wts)
    
    # Son modeli de kaydet
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model kaydedildi: {final_path}")
    
    return model, history


def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4):
    """
    Optimizer oluşturur
    
    Args:
        model: PyTorch modeli
        optimizer_name: 'adam', 'sgd', 'adamw'
        lr: Learning rate
        weight_decay: L2 regularization
    
    Returns:
        optimizer
    """
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
                              weight_decay=weight_decay)
    else:
        raise ValueError(f"Bilinmeyen optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_name='step', step_size=10, gamma=0.1):
    """
    Learning rate scheduler oluşturur
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: 'step', 'cosine', 'plateau'
        step_size: StepLR için step boyutu
        gamma: LR azaltma faktörü
    
    Returns:
        scheduler
    """
    if scheduler_name.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    elif scheduler_name.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=gamma, patience=5, verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


if __name__ == "__main__":
    print("Bu modül doğrudan çalıştırılmak için tasarlanmamıştır.")
    print("main.py kullanarak eğitimi başlatın.")
