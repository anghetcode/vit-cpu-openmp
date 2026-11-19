import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được module trong package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vit_cpu.models.vit import VisionTransformer
from vit_cpu.utils.openmp_config import set_openmp_threads, get_cpu_info, print_cpu_summary
from vit_cpu.utils.profiling import compute_vit_flops, get_process_memory_mb, LatencyStats


def make_fixed_batch(batch_size: int, image_size: int = 56):
    """Tạo batch giả lập cố định (để benchmark suy luận độc lập)."""
    torch.manual_seed(42)
    return torch.randn(batch_size, 3, image_size, image_size)


def benchmark_inference(
    model: torch.nn.Module,
    batch_size: int,
    iterations: int = 30,
    warmup: int = 10,
    image_size: int = 56
):
    """Đo đạc độ trễ suy luận, throughput, FLOPS và RAM cho một kích thước batch."""
    model.eval()
    host_x = make_fixed_batch(batch_size, image_size)

    # 1. First call (Cold start)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(host_x)
    first_call_ms = (time.perf_counter() - start) * 1000.0

    # 2. Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(host_x)

    # 3. Compute-only (Dữ liệu đã nằm sẵn trên RAM chuẩn bị)
    compute_times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(host_x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            compute_times.append(elapsed_ms)

    # 4. Transfer + Compute (Bao gồm clone/copy dữ liệu và forward)
    transfer_times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            x_input = host_x.clone()
            output = model(x_input)
            _ = output.detach()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            transfer_times.append(elapsed_ms)

    compute_stats = LatencyStats(compute_times, batch_size).summary()
    transfer_stats = LatencyStats(transfer_times, batch_size).summary()
    mem_info = get_process_memory_mb()
    flops_info = compute_vit_flops(image_size=image_size, batch_size=batch_size)

    # GFLOPS thực tế đạt được = Tổng GFLOPS tính toán / thời gian tính toán (giây)
    achieved_gflops = round(flops_info["total_batch_gflops"] / (compute_stats["median_ms"] / 1000.0), 2) if compute_stats["median_ms"] > 0 else 0.0

    return {
        "batch_size": batch_size,
        "precision": "FP32",
        "execution": "eager_cpu_openmp",
        "first_call_ms": round(first_call_ms, 3),
        "compute_median_ms": compute_stats["median_ms"],
        "compute_p25_ms": compute_stats["p25_ms"],
        "compute_p75_ms": compute_stats["p75_ms"],
        "compute_throughput": compute_stats["throughput_imgs_per_sec"],
        "transfer_compute_median_ms": transfer_stats["median_ms"],
        "transfer_compute_throughput": transfer_stats["throughput_imgs_per_sec"],
        "ram_rss_mb": mem_info["ram_rss_mb"],
        "peak_ram_mb": mem_info["peak_ram_mb"],
        "batch_gflops": flops_info["total_batch_gflops"],
        "achieved_gflops": achieved_gflops
    }


def benchmark_openmp_thread_scaling(
    model: torch.nn.Module,
    batch_size: int = 32,
    thread_counts: list = None,
    iterations: int = 20,
    warmup: int = 5,
    image_size: int = 56
):
    """Đo đạc độ co giãn hiệu năng (Scalability) của OpenMP khi tăng dần số luồng."""
    if thread_counts is None:
        cpu_info = get_cpu_info()
        p_cores = cpu_info["physical_cores"]
        l_cores = cpu_info["logical_cores"]
        candidates = [1, 2, 4, p_cores, l_cores]
        thread_counts = sorted(list(set(c for c in candidates if c > 0 and c <= l_cores)))

    print(f"\n--- Thực nghiệm Đa luồng OpenMP Thread Scaling (Batch Size = {batch_size}) ---")
    print(f"Các mốc số luồng đo: {thread_counts}")

    scaling_results = []
    base_median_ms = None

    for t in thread_counts:
        set_openmp_threads(t)
        res = benchmark_inference(
            model=model,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            image_size=image_size
        )
        median_ms = res["compute_median_ms"]
        throughput = res["compute_throughput"]

        if base_median_ms is None:
            base_median_ms = median_ms

        speedup = round(base_median_ms / median_ms, 2) if median_ms > 0 else 1.0
        efficiency = round(speedup / t, 3) if t > 0 else 1.0

        print(f"Luồng OpenMP: {t:2d} | Latency: {median_ms:7.2f} ms | Throughput: {throughput:6.1f} img/s | Speedup: {speedup:4.2f}x | Efficiency: {efficiency:5.1%}")

        scaling_results.append({
            "num_threads": t,
            "batch_size": batch_size,
            "median_ms": median_ms,
            "throughput_imgs_per_sec": throughput,
            "speedup": speedup,
            "parallel_efficiency": efficiency,
            "ram_rss_mb": res["ram_rss_mb"]
        })

    return pd.DataFrame(scaling_results)


def run_all_benchmarks(args):
    print_cpu_summary()
    os.makedirs("results", exist_ok=True)

    if args.num_threads:
        set_openmp_threads(args.num_threads)

    print("\n--- Khởi tạo mô hình Vision Transformer ---")
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Đang nạp checkpoint từ: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        print("Nạp checkpoint thành công!")
    else:
        print("Sử dụng trọng số khởi tạo ban đầu để benchmark.")

    # 1. Benchmark theo các kích thước batch
    print(f"\n--- 1. Benchmark theo kích thước dữ liệu (Batch Sizes: {args.batch_sizes}) ---")
    benchmark_results = []
    memory_flops_results = []

    for bs in args.batch_sizes:
        print(f"-> Đang đo Batch Size = {bs} ...")
        res = benchmark_inference(
            model=model,
            batch_size=bs,
            iterations=args.iterations,
            warmup=args.warmup,
            image_size=args.image_size
        )
        benchmark_results.append(res)
        
        # Log riêng cho Hằng
        memory_flops_results.append({
            "platform": "CPU",
            "batch_size": bs,
            "latency_median_ms": res["compute_median_ms"],
            "throughput_imgs_sec": res["compute_throughput"],
            "ram_rss_mb": res["ram_rss_mb"],
            "peak_ram_mb": res["peak_ram_mb"],
            "batch_gflops": res["batch_gflops"],
            "achieved_gflops": res["achieved_gflops"]
        })
        print(f"   Batch {bs:2d}: Latency={res['compute_median_ms']:6.2f}ms | Throughput={res['compute_throughput']:6.1f} img/s | RAM={res['ram_rss_mb']}MB | Achieved GFLOPS={res['achieved_gflops']}")

    bench_df = pd.DataFrame(benchmark_results)
    bench_df.to_csv("results/cpu_benchmark.csv", index=False)
    print("\nĐã xuất kết quả sang: results/cpu_benchmark.csv")

    mem_flops_df = pd.DataFrame(memory_flops_results)
    mem_flops_df.to_csv("results/cpu_memory_flops.csv", index=False)
    print("Đã xuất số liệu cho Hằng sang: results/cpu_memory_flops.csv")

    # 2. Benchmark đa luồng OpenMP Thread Scaling
    scaling_df = benchmark_openmp_thread_scaling(
        model=model,
        batch_size=args.scaling_batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        image_size=args.image_size
    )
    scaling_df.to_csv("results/openmp_thread_scaling.csv", index=False)
    print("Đã xuất kết quả Thread Scaling sang: results/openmp_thread_scaling.csv")

    # 3. Báo cáo hoàn thành
    print("\n" + "=" * 60)
    print("HOÀN THÀNH TẤT CẢ CÁC BÀI THỰC NGHIỆM TRÊN CPU!")
    print("Toàn bộ bảng log đã sẵn sàng trong thư mục results/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm Benchmark ViT CPU OpenMP")
    parser.add_argument("--image-size", type=int, default=56)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--scaling-batch-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_all_benchmarks(args)
