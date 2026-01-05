"""
Diz Osteoartrit (KL Grading) Sınıflandırması - Ana Script
==========================================================
Bu script, tüm eğitim ve değerlendirme sürecini yönetir.

Kullanım:
    # Özgün CNN ile eğitim
    python main.py --model cnn --epochs 25 --batch_size 32

    # ResNet18 Transfer Learning ile eğitim
    python main.py --model resnet18 --pretrained --epochs 15

    # Sadece değerlendirme
    python main.py --evaluate --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Kaynak modülleri import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import create_data_loaders, get_class_weights
from src.models import get_model, count_parameters
from src.train import train_model, get_optimizer, get_scheduler
from src.evaluate import evaluate_model, plot_training_history
from src.utils import set_seed, get_device, print_model_summary


def parse_args():
    """Komut satırı argümanlarını ayrıştırır"""
    parser = argparse.ArgumentParser(
        description='Knee Osteoarthritis KL Grading Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --model cnn --epochs 25
  python main.py --model resnet18 --pretrained --freeze
  python main.py --evaluate --checkpoint checkpoints/best_model.pth
        """
    )
    
    # Veri parametreleri
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Veri seti dizini (varsayılan: data)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch boyutu (varsayılan: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader işçi sayısı (varsayılan: 4)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Görüntü boyutu (varsayılan: 224)')
    
    # Model parametreleri
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet18'],
                        help='Model tipi: cnn veya resnet18 (varsayılan: cnn)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Transfer learning için önceden eğitilmiş ağırlıklar')
    parser.add_argument('--freeze', action='store_true',
                        help='Özellik çıkarıcı katmanları dondur')
    
    # Eğitim parametreleri
    parser.add_argument('--epochs', type=int, default=25,
                        help='Epoch sayısı (varsayılan: 25)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (varsayılan: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer (varsayılan: adam)')
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='LR scheduler (varsayılan: step)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularization (varsayılan: 1e-4)')
    parser.add_argument('--early_stopping', type=int, default=5,
                        help='Early stopping patience (varsayılan: 5)')
    parser.add_argument('--class_weights', action='store_true',
                        help='Dengesiz sınıflar için ağırlık kullan')
    
    # Diğer parametreler
    parser.add_argument('--seed', type=int, default=42,
                        help='Rastgelelik tohumu (varsayılan: 42)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Model kayıt dizini (varsayılan: checkpoints)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Sonuç dizini (varsayılan: results)')
    
    # Değerlendirme modu
    parser.add_argument('--evaluate', action='store_true',
                        help='Sadece değerlendirme modu')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Yüklenecek checkpoint dosyası')
    
    return parser.parse_args()


def main():
    """Ana fonksiyon"""
    # Argümanları al
    args = parse_args()
    
    # Başlık
    print("\n" + "=" * 70)
    print("DİZ OSTEOARTRİT (KL GRADING) SINIFLANDIRMASI")
    print("Sinir Ağları Dersi - Bitirme Ödevi")
    print("=" * 70)
    
    # Tohum ayarla
    set_seed(args.seed)
    
    # Cihaz
    device = get_device()
    
    # Veri klasör yapısını kontrol et
    train_dir = os.path.join(args.data_dir, 'train')
    if not os.path.exists(train_dir):
        # Kaggle veri seti yapısını kontrol et
        possible_paths = [
            os.path.join(args.data_dir, 'Knee Osteoarthritis', 'train'),
            os.path.join(args.data_dir, 'knee_osteoarthritis', 'train'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                args.data_dir = os.path.dirname(path)
                print(f"Veri dizini güncellendi: {args.data_dir}")
                break
    
    # Veri yükleyicilerini oluştur
    print("\n" + "-" * 40)
    print("Veri yükleniyor...")
    print("-" * 40)
    
    dataloaders, dataset_sizes, class_names = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    num_classes = len(class_names)
    
    # Model oluştur
    print("\n" + "-" * 40)
    print("Model oluşturuluyor...")
    print("-" * 40)
    
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_features=args.freeze
    )
    model = model.to(device)
    
    # Model özeti
    print_model_summary(model)
    count_parameters(model)
    
    # Sadece değerlendirme modu
    if args.evaluate:
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, 'best_model.pth')
        
        if not os.path.exists(args.checkpoint):
            print(f"Hata: Checkpoint bulunamadı: {args.checkpoint}")
            return
        
        # Ağırlıkları yükle
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model yüklendi: {args.checkpoint}")
        
        # Değerlendirme
        results = evaluate_model(
            model, 
            dataloaders['test'], 
            class_names, 
            device,
            save_dir=args.results_dir
        )
        return
    
    # Loss fonksiyonu
    if args.class_weights:
        weights = get_class_weights(args.data_dir)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("\nSınıf ağırlıkları kullanılıyor.")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = get_optimizer(
        model, 
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    if args.scheduler != 'none':
        scheduler = get_scheduler(optimizer, scheduler_name=args.scheduler)
    else:
        scheduler = None
    
    # Eğitim
    print("\n" + "-" * 40)
    print("Eğitim başlıyor...")
    print("-" * 40)
    
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        early_stopping_patience=args.early_stopping
    )
    
    # Eğitim grafiği
    print("\nEğitim grafikleri oluşturuluyor...")
    plot_training_history(
        history,
        save_path=os.path.join(args.results_dir, 'training_history.png')
    )
    
    # Test seti değerlendirmesi
    print("\n" + "-" * 40)
    print("Test seti değerlendirmesi...")
    print("-" * 40)
    
    results = evaluate_model(
        model,
        dataloaders['test'],
        class_names,
        device,
        save_dir=args.results_dir
    )
    
    print("\n" + "=" * 70)
    print("İŞLEM TAMAMLANDI!")
    print(f"Model kaydedildi: {args.save_dir}")
    print(f"Sonuçlar kaydedildi: {args.results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
