# Vision Transformer (ViT) trên nền tảng CPU (Tích hợp OpenMP)

Dự án triển khai mô hình **Vision Transformer (ViT)** trên kiến trúc phần cứng **CPU** có tích hợp cơ chế đa luồng **OpenMP**, phục vụ bài toán phân loại ảnh mô học `medmnist.PathMNIST`.

Dự án nằm trong phân công công việc **Tuần 2: Triển khai mã hóa và Chạy thực nghiệm**:
* **Lộc**: Lập trình và tối ưu hóa ViT trên TPU (`TPU.ipynb` - TorchTPU v6e).
* **Tuấn**: Lập trình và kiểm thử ViT trên GPU (CUDA / PyTorch).
* **An (Repository này)**: Lập trình ViT trên nền tảng CPU (tích hợp OpenMP).
* **Hằng**: Giám sát quá trình chạy, ghi log thông số (FLOPS, Latency, Throughput, lượng RAM tiêu thụ) khi tăng dần kích thước dữ liệu.
* **Cả nhóm**: Kiểm duyệt chéo (Cross-review).

---

## 1. Kiến trúc Mô hình (Đồng bộ 100% với TPU và GPU)

* **Tập dữ liệu**: `PathMNIST` (9 phân lớp mô bệnh học, ảnh chuẩn hóa về kích thước $56 \times 56$, 3 kênh RGB).
* **Patch Size**: $4 \times 4$ $\rightarrow$ Số lượng token ảnh $N = (56/4)^2 = 196$ patches (+1 CLS token = 197 tokens).
* **Embedding Dimension**: 192 (ViT-Tiny/Compact).
* **Transformer Encoder**: 6 Blocks, mỗi block gồm:
  * Multi-Head Self-Attention: 3 Heads (Head dimension = 64).
  * Pre-LayerNorm & Residual Connection.
  * MLP: Tỷ lệ mở rộng 4 (Hidden dimension = 768) với hàm kích hoạt GELU.
* **Tổng số tham số**: ~2,713,929 tham số (~2.7M parameters).
* **Tính toán FLOPs**: ~1.235 GFLOPs / ảnh (~0.617 GMACs).

---

## 2. Cơ chế Tích hợp OpenMP trên CPU

CPU sử dụng: AMD 6 Nhân vật lý / 12 Luồng logic.
Mã nguồn điều khiển trực tiếp cơ chế đa luồng OpenMP của PyTorch và thư viện toán học BLAS:
* Hàm `torch.set_num_threads(N)` kiểm soát số luồng tính toán song song các phép toán tensor (GEMM, LayerNorm, Softmax, Attention).
* Biến môi trường `OMP_NUM_THREADS`, `MKL_NUM_THREADS` đảm bảo việc phân bổ tải đồng đều trên các nhân vật lý.
* Bài thực nghiệm **Thread Scaling** đo đạc tốc độ tăng tốc (Speedup $S_N$) và hiệu suất song song (Parallel Efficiency $E_N$) qua các mốc luồng $N \in [1, 2, 4, 6, 8, 12]$.

---

## 3. Cấu trúc Thư mục

```
hopeful-faraday/
├── requirements.txt                 # Khai báo các thư viện phụ thuộc
├── README.md                        # Tài liệu hướng dẫn và báo cáo
├── CPU.ipynb                        # Jupyter Notebook đối chiếu trực tiếp với TPU.ipynb
├── vit_cpu/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── vit.py                   # Triển khai Vision Transformer chuẩn
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── openmp_config.py         # Quản trị luồng OpenMP & affinity
│   │   └── profiling.py             # Tính toán lý thuyết FLOPs, RAM RSS, Latency
│   ├── train.py                     # Script huấn luyện & đánh giá PathMNIST
│   └── benchmark.py                 # Script benchmark suy luận và OpenMP scaling
└── results/                         # Thư mục chứa bảng số liệu bàn giao
    ├── config.json                  # Cấu hình hyperparameters
    ├── cpu_benchmark.csv            # Kết quả đo độ trễ & throughput theo batch size
    ├── openmp_thread_scaling.csv    # Kết quả đo đa luồng OpenMP (Speedup & Efficiency)
    ├── cpu_memory_flops.csv         # Bảng thông số RAM & FLOPS riêng cho Hằng
    ├── training_history.csv         # Lịch sử huấn luyện (Loss, Accuracy, Thời gian)
    └── test_metrics.csv             # Kết quả kiểm định trên tập Test
```

---

## 4. Hướng dẫn Chạy (Quickstart)

### Bước 1: Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy Benchmark suy luận & Đo đạc đa luồng OpenMP
Lệnh chạy tự động toàn bộ thực nghiệm và xuất các file CSV vào thư mục `results/`:
```powershell
python vit_cpu/benchmark.py --batch-sizes 1 8 32 64 --scaling-batch-size 32
```

### Bước 3: Huấn luyện thực nghiệm trên CPU (Tùy chọn)
Huấn luyện 1 epoch kiểm thử nhanh hoặc full epochs:
```powershell
python vit_cpu/train.py --epochs 1 --batch-size 64
```
*(Nếu đã có sẵn file weights `checkpoints/best_model.pt` từ Lộc hoặc Tuấn, thêm cờ `--load-checkpoint checkpoints/best_model.pt` để nạp trực tiếp).*

### Bước 4: Trực quan hóa trên Jupyter Notebook
Khởi chạy notebook `CPU.ipynb` để xem toàn bộ bảng số liệu, ma trận nhầm lẫn và biểu đồ Speedup của OpenMP:
```powershell
jupyter notebook CPU.ipynb
```

---

## 5. Dữ liệu Bàn giao cho Hằng & Nhóm

Tất cả kết quả đo đạc được chuẩn hóa theo định dạng của Lộc trên TPU để Hằng dễ dàng gộp bảng so sánh:
* `results/cpu_benchmark.csv`: Khớp 100% cột với `tpu_benchmark.csv` (`first_call_ms`, `compute_median_ms`, `compute_throughput`, `ram_rss_mb`).
* `results/cpu_memory_flops.csv`: Cung cấp cụ thể `RAM_RSS_MB`, `Batch_GFLOPS` và `Achieved_GFLOPS`.
* `results/openmp_thread_scaling.csv`: Minh chứng trực quan cho sức mạnh đa luồng OpenMP trên CPU.
