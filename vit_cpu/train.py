import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được module trong package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vit_cpu.models.vit import VisionTransformer
from vit_cpu.utils.openmp_config import set_openmp_threads, print_cpu_summary
from vit_cpu.utils.profiling import get_process_memory_mb


def get_default_config():
    return {
        "image_size": 56,
        "patch_size": 4,
        "num_classes": 9,
        "embed_dim": 192,
        "num_heads": 3,
        "depth": 6,
        "mlp_ratio": 4,
        "dropout": 0.0,
        "batch_size": 64,
        "epochs": 1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "warmup": 10,
        "benchmark_iterations": 30
    }


def load_dataset(config):
    try:
        from medmnist import PathMNIST
    except ImportError:
        raise ImportError("Chưa cài đặt medmnist. Hãy chạy: pip install medmnist")

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_dir = os.path.join(os.path.expanduser("~"), ".medmnist")
    train_dataset = PathMNIST(split="train", transform=transform, download=True, as_rgb=True, root=data_dir)
    val_dataset = PathMNIST(split="val", transform=transform, download=True, as_rgb=True, root=data_dir)
    test_dataset = PathMNIST(split="test", transform=transform, download=True, as_rgb=True, root=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        labels = labels.reshape(-1).long()
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    val_loss = total_loss / total_samples if total_samples > 0 else 0.0
    val_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return val_loss, val_acc


def train(args):
    config = get_default_config()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["weight_decay"] = args.weight_decay

    if args.num_threads:
        set_openmp_threads(args.num_threads)
    
    print_cpu_summary()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    with open("results/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n--- 1. Nạp dữ liệu PathMNIST ---")
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_dataset(config)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")

    print("\n--- 2. Khởi tạo mô hình Vision Transformer ---")
    model = VisionTransformer(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        depth=config["depth"],
        mlp_ratio=config["mlp_ratio"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số tham số: {total_params:,}")

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Đang nạp checkpoint từ: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        print("Nạp checkpoint thành công!")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    history = []
    best_val_acc = 0.0

    print(f"\n--- 3. Bắt đầu Huấn luyện ({config['epochs']} epochs trên CPU) ---")
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        start_time = time.perf_counter()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        for images, labels in progress:
            labels = labels.reshape(-1).long()
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            progress.set_postfix({"train_loss": f"{loss.item():.4f}"})

        epoch_time = time.perf_counter() - start_time
        train_loss = running_loss / total_samples

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        mem_info = get_process_memory_mb()

        result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "epoch_time_s": round(epoch_time, 2),
            "ram_rss_mb": mem_info["ram_rss_mb"]
        }
        history.append(result)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Thời gian={epoch_time:.2f}s, RAM={mem_info['ram_rss_mb']}MB")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = args.save_checkpoint
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "config": config
            }, save_path)
            print(f"--> Đã lưu checkpoint tốt nhất vào: {save_path}")

        pd.DataFrame(history).to_csv("results/training_history.csv", index=False)

    print("\n--- 4. Đánh giá trên tập Kiểm thử (Test Set) ---")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    test_df = pd.DataFrame([{
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4)
    }])
    test_df.to_csv("results/test_metrics.csv", index=False)
    print("Đã xuất kết quả kiểm thử vào: results/test_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện ViT trên CPU với OpenMP")
    parser.add_argument("--epochs", type=int, default=1, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Tốc độ học")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-threads", type=int, default=None, help="Số luồng OpenMP")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Đường dẫn checkpoint để nạp")
    parser.add_argument("--save-checkpoint", type=str, default="checkpoints/vit_pathmnist_cpu.pt", help="Đường dẫn lưu checkpoint")
    args = parser.parse_args()
    train(args)
