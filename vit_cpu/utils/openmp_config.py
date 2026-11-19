import os
import sys
import platform
import psutil
import torch

# Hỗ trợ in tiếng Việt trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_cpu_info():
    """Lấy thông tin chi tiết về phần cứng CPU và bộ nhớ RAM."""
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    processor_name = platform.processor() or platform.machine()
    
    return {
        "processor": processor_name,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram_gb": ram_gb,
        "platform": platform.platform()
    }


def set_openmp_threads(num_threads: int):
    """Thiết lập số luồng OpenMP cho PyTorch và biến môi trường hệ thống.
    
    Args:
        num_threads (int): Số luồng cần thiết lập.
    """
    assert num_threads > 0, "Số luồng OpenMP phải lớn hơn 0"
    
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    torch.set_num_threads(num_threads)


def get_parallel_info():
    """Lấy thông tin về backend song song hóa của PyTorch (OpenMP / MKL / TBB)."""
    return {
        "active_threads": torch.get_num_threads(),
        "active_interop_threads": torch.get_num_interop_threads(),
        "parallel_info": torch.__config__.parallel_info()
    }


def print_cpu_summary():
    """In bản tóm tắt thông số CPU và cấu hình OpenMP hiện hành."""
    info = get_cpu_info()
    print("=" * 60)
    print("THONG SO PHAN CUNG CPU & RUNTIME OPENMP")
    print("=" * 60)
    print(f"Vi xu ly        : {info['processor']}")
    print(f"So nhan vat ly  : {info['physical_cores']} cores")
    print(f"So luong logic  : {info['logical_cores']} threads")
    print(f"Tong bo nho RAM : {info['ram_gb']} GB")
    print(f"He dieu hanh    : {info['platform']}")
    print(f"PyTorch version : {torch.__version__}")
    print(f"Luong OpenMP    : {torch.get_num_threads()} threads dang kich hoat")
    print("=" * 60)

