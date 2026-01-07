"""
Değerlendirme Scripti
=====================
Bu script, eğitilmiş modeli test setinde değerlendirir ve sonuçları görselleştirir.

Kullanım:
    python evaluate_script.py
    python evaluate_script.py --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
import torch
import numpy as np

from src.data_loader import create_data_loaders
from src.models import get_model
from src.evaluate import (
    predict,
    plot_confusion_matrix,
    print_classification_report,
    plot_class_distribution,
    per_class_accuracy
)


def main():
    parser = argparse.ArgumentParser(description='Model Değerlendirmesi')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Model checkpoint dosyası')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Veri seti dizini')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch boyutu')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Sonuçların kaydedileceği dizin')
    args = parser.parse_args()

    # Sonuç dizinini oluştur
    os.makedirs(args.results_dir, exist_ok=True)

    # Cihaz
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Cihaz: {device}")
    print(f"{'='*60}")

    # Veri yükle
    print("\nVeri seti yükleniyor...")
    dataloaders, dataset_sizes, class_names = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0  # Windows uyumluluğu için
    )

    test_loader = dataloaders['test']
    print(f"Test seti: {dataset_sizes['test']} görüntü")

    # Model yükle
    print(f"\nModel yükleniyor: {args.checkpoint}")
    
    # Checkpoint'u yükle
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Model state dict'i al
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Model tipini checkpoint anahtarlarından belirle
    keys = list(state_dict.keys())
    if any(k.startswith('resnet.') for k in keys):
        model_name = 'resnet18'
    elif any(k.startswith('conv1.') or k.startswith('conv2.') for k in keys):
        model_name = 'cnn'
    elif 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        model_name = 'cnn'  # varsayılan
    
    print(f"Model tipi: {model_name}")
    
    # Model oluştur
    model = get_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    
    # Ağırlıkları yükle
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()

    # Tahminler
    print("\nTest setinde tahminler yapılıyor...")
    y_pred, y_true, y_probs = predict(model, test_loader, device)

    # Classification Report
    print("\n" + "="*70)
    print("DEĞERLENDİRME SONUÇLARI")
    print("="*70)
    
    report = print_classification_report(y_true, y_pred, class_names)

    # Per-class accuracy
    per_class_accuracy(y_true, y_pred, class_names)

    # Confusion Matrix
    print("\nConfusion Matrix oluşturuluyor...")
    cm = plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(args.results_dir, 'confusion_matrix.png'),
        normalize=True
    )

    # Non-normalized confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(args.results_dir, 'confusion_matrix_counts.png'),
        normalize=False
    )

    # Sınıf dağılımı
    print("\nSınıf dağılımı oluşturuluyor...")
    plot_class_distribution(
        y_true, y_pred, class_names,
        save_path=os.path.join(args.results_dir, 'class_distribution.png')
    )

    # Sonuçları kaydet
    np.save(os.path.join(args.results_dir, 'predictions.npy'), y_pred)
    np.save(os.path.join(args.results_dir, 'labels.npy'), y_true)
    np.save(os.path.join(args.results_dir, 'probabilities.npy'), y_probs)

    print(f"\n{'='*60}")
    print(f"Sonuçlar kaydedildi: {args.results_dir}/")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix_counts.png")
    print(f"  - class_distribution.png")
    print(f"  - predictions.npy, labels.npy, probabilities.npy")
    print(f"{'='*60}")

    # Grade 2 analizi
    print("\n" + "-"*60)
    print("GRADE 2 ANALİZİ")
    print("-"*60)
    
    grade2_idx = class_names.index('2') if '2' in class_names else 2
    grade2_mask = y_true == grade2_idx
    grade2_total = grade2_mask.sum()
    grade2_correct = (y_pred[grade2_mask] == y_true[grade2_mask]).sum()
    grade2_acc = grade2_correct / grade2_total if grade2_total > 0 else 0
    
    print(f"Grade 2 toplam örnek: {grade2_total}")
    print(f"Grade 2 doğru tahmin: {grade2_correct}")
    print(f"Grade 2 doğruluk: {grade2_acc*100:.2f}%")
    
    # Grade 2'nin hangi sınıflara karıştığını göster
    if grade2_total > 0:
        print("\nGrade 2 karışıklık analizi:")
        for i, name in enumerate(class_names):
            confused = (y_pred[grade2_mask] == i).sum()
            if confused > 0:
                print(f"  → Grade {name} olarak tahmin: {confused} ({confused/grade2_total*100:.1f}%)")

    return report


if __name__ == "__main__":
    main()
