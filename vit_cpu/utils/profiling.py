import os
import psutil
import numpy as np


def compute_vit_flops(
    image_size: int = 56,
    patch_size: int = 4,
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 3,
    mlp_ratio: int = 4,
    num_classes: int = 9,
    batch_size: int = 1
):
    """Tính toán chính xác số phép tính dấu phẩy động (FLOPs) lý thuyết của mô hình ViT.
    
    Quy ước tiêu chuẩn: 1 phép nhân-cộng tích lũy (Multiply-Accumulate, MAC) = 2 FLOPs.
    """
    num_patches = (image_size // patch_size) ** 2
    n_tokens = num_patches + 1  # Cộng thêm 1 CLS token
    
    # 1. Patch Embedding (Conv2d 3 -> embed_dim, kernel=patch_size, stride=patch_size)
    patch_embed_macs = num_patches * (3 * patch_size * patch_size * embed_dim)
    
    # 2. Chi tiết 1 khối Transformer Encoder:
    # 2a. QKV Linear Projection: embed_dim -> 3 * embed_dim
    qkv_macs = n_tokens * (embed_dim * (3 * embed_dim))
    
    # 2b. Attention Matrix: (Q @ K^T): [h, N, d_k] x [h, d_k, N] -> [h, N, N]
    attn_scores_macs = n_tokens * n_tokens * embed_dim
    
    # 2c. Attention @ V: [h, N, N] x [h, N, d_k] -> [h, N, d_k]
    attn_context_macs = n_tokens * n_tokens * embed_dim
    
    # 2d. Output Projection: embed_dim -> embed_dim
    out_proj_macs = n_tokens * (embed_dim * embed_dim)
    
    # 2e. MLP FC1: embed_dim -> (mlp_ratio * embed_dim)
    mlp_fc1_macs = n_tokens * (embed_dim * (mlp_ratio * embed_dim))
    
    # 2f. MLP FC2: (mlp_ratio * embed_dim) -> embed_dim
    mlp_fc2_macs = n_tokens * ((mlp_ratio * embed_dim) * embed_dim)
    
    block_macs = qkv_macs + attn_scores_macs + attn_context_macs + out_proj_macs + mlp_fc1_macs + mlp_fc2_macs
    
    # 3. Classifier Head: embed_dim -> num_classes
    head_macs = embed_dim * num_classes
    
    total_macs_per_image = patch_embed_macs + (depth * block_macs) + head_macs
    total_flops_per_image = total_macs_per_image * 2
    
    total_gflops = (total_flops_per_image * batch_size) / 1e9
    total_gmacs = (total_macs_per_image * batch_size) / 1e9
    
    return {
        "macs_per_image": total_macs_per_image,
        "flops_per_image": total_flops_per_image,
        "gflops_per_image": round(total_flops_per_image / 1e9, 4),
        "total_batch_gflops": round(total_gflops, 4),
        "total_batch_gmacs": round(total_gmacs, 4),
        "batch_size": batch_size
    }


def get_process_memory_mb():
    """Lấy lượng RAM hiện tại và RAM đỉnh điểm (Peak RAM) mà tiến trình CPU đang tiêu thụ (đơn vị: MB)."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = round(mem_info.rss / (1024 * 1024), 2)
    
    # Peak memory nếu khả dụng trên hệ điều hành
    peak_mb = rss_mb
    if hasattr(mem_info, "peak_wset"):  # Trên Windows
        peak_mb = round(mem_info.peak_wset / (1024 * 1024), 2)
    
    return {
        "ram_rss_mb": rss_mb,
        "peak_ram_mb": peak_mb
    }


class LatencyStats:
    """Tính toán các chỉ số thống kê độ trễ từ danh sách các lần đo."""
    
    def __init__(self, times_ms: list, batch_size: int):
        self.times_ms = times_ms
        self.batch_size = batch_size
        
    def summary(self):
        if not self.times_ms:
            return {}
        
        median_ms = float(np.median(self.times_ms))
        p25_ms = float(np.percentile(self.times_ms, 25))
        p75_ms = float(np.percentile(self.times_ms, 75))
        min_ms = float(np.min(self.times_ms))
        max_ms = float(np.max(self.times_ms))
        
        throughput = (self.batch_size * 1000.0) / median_ms if median_ms > 0 else 0.0
        
        return {
            "median_ms": round(median_ms, 3),
            "p25_ms": round(p25_ms, 3),
            "p75_ms": round(p75_ms, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "throughput_imgs_per_sec": round(throughput, 2)
        }
