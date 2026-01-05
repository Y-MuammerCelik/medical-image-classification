"""
Değerlendirme Modülü
====================
Bu modül, eğitilmiş modelin test seti üzerinde değerlendirilmesi için:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Görselleştirmeler
içerir.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm


def predict(model, dataloader, device):
    """
    Tüm veri seti için tahminler yapar
    
    Args:
        model: Eğitilmiş PyTorch modeli
        dataloader: Test DataLoader
        device: CPU veya CUDA
    
    Returns:
        tuple: (all_predictions, all_labels, all_probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Tahmin yapılıyor"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True):
    """
    Confusion Matrix görselleştirmesi
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri
        save_path: Kayıt yolu (opsiyonel)
        normalize: Normalize edilmiş matris
    """
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Görselleştirme
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_display, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Etiket', fontsize=12)
    plt.xlabel('Tahmin Edilen Etiket', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix kaydedildi: {save_path}")
    
    plt.show()
    
    return cm


def print_classification_report(y_true, y_pred, class_names):
    """
    Classification Report (Precision, Recall, F1-Score) yazdırır
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri
    
    Returns:
        dict: Classification report dictionary
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    # Sklearn classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Dictionary formatında
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Genel metrikler
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print("-" * 70)
    print(f"Genel Doğruluk (Accuracy): {accuracy*100:.2f}%")
    print(f"Ağırlıklı Precision:       {precision*100:.2f}%")
    print(f"Ağırlıklı Recall:          {recall*100:.2f}%")
    print(f"Ağırlıklı F1-Score:        {f1*100:.2f}%")
    print("=" * 70)
    
    return report_dict


def plot_class_distribution(y_true, y_pred, class_names, save_path=None):
    """
    Sınıf dağılımı karşılaştırması
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri
        save_path: Kayıt yolu
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gerçek dağılım
    true_counts = np.bincount(y_true, minlength=len(class_names))
    axes[0].bar(class_names, true_counts, color='steelblue', edgecolor='black')
    axes[0].set_title('Gerçek Sınıf Dağılımı', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sınıf')
    axes[0].set_ylabel('Örnek Sayısı')
    for i, v in enumerate(true_counts):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Tahmin dağılımı
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    axes[1].bar(class_names, pred_counts, color='coral', edgecolor='black')
    axes[1].set_title('Tahmin Edilen Sınıf Dağılımı', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sınıf')
    axes[1].set_ylabel('Örnek Sayısı')
    for i, v in enumerate(pred_counts):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sınıf dağılımı kaydedildi: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Eğitim geçmişini görselleştirir
    
    Args:
        history: Eğitim geçmişi dictionary
        save_path: Kayıt yolu
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss grafiği
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Eğitim Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validasyon Loss', linewidth=2)
    axes[0].set_title('Loss Değişimi', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy grafiği
    axes[1].plot(epochs, [acc*100 for acc in history['train_acc']], 'b-', 
                 label='Eğitim Accuracy', linewidth=2)
    axes[1].plot(epochs, [acc*100 for acc in history['val_acc']], 'r-', 
                 label='Validasyon Accuracy', linewidth=2)
    axes[1].set_title('Accuracy Değişimi', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate grafiği
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_title('Learning Rate Değişimi', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eğitim grafiği kaydedildi: {save_path}")
    
    plt.show()


def evaluate_model(model, test_loader, class_names, device, save_dir='results'):
    """
    Modeli tam değerlendirme yapar
    
    Args:
        model: Eğitilmiş model
        test_loader: Test DataLoader
        class_names: Sınıf isimleri
        device: CPU veya CUDA
        save_dir: Sonuçların kaydedileceği dizin
    
    Returns:
        dict: Değerlendirme sonuçları
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("MODEL DEĞERLENDİRMESİ")
    print("=" * 70)
    
    # Tahminler
    y_pred, y_true, y_probs = predict(model, test_loader, device)
    
    # Classification Report
    report = print_classification_report(y_true, y_pred, class_names)
    
    # Confusion Matrix
    print("\nConfusion Matrix oluşturuluyor...")
    cm = plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # Sınıf dağılımı
    print("\nSınıf dağılımı oluşturuluyor...")
    plot_class_distribution(
        y_true, y_pred, class_names,
        save_path=os.path.join(save_dir, 'class_distribution.png')
    )
    
    # Sonuçları kaydet
    results = {
        'predictions': y_pred,
        'labels': y_true,
        'probabilities': y_probs,
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    # NumPy formatında kaydet
    np.save(os.path.join(save_dir, 'predictions.npy'), y_pred)
    np.save(os.path.join(save_dir, 'labels.npy'), y_true)
    np.save(os.path.join(save_dir, 'probabilities.npy'), y_probs)
    
    print(f"\nSonuçlar kaydedildi: {save_dir}")
    
    return results


def per_class_accuracy(y_true, y_pred, class_names):
    """
    Her sınıf için ayrı doğruluk hesaplar
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahminler
        class_names: Sınıf isimleri
    """
    print("\n" + "-" * 40)
    print("Sınıf Bazlı Doğruluk")
    print("-" * 40)
    
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"  {class_name}: {class_acc*100:.2f}% ({mask.sum()} örnek)")


if __name__ == "__main__":
    print("Bu modül doğrudan çalıştırılmak için tasarlanmamıştır.")
    print("main.py kullanarak değerlendirmeyi başlatın.")
