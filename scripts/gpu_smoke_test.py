import time

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"compiled_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_count={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")

    device = torch.device("cuda:0")
    print(f"device_name={torch.cuda.get_device_name(device)}")

    a = torch.randn((4096, 4096), device=device)
    b = torch.randn((4096, 4096), device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    print(f"matmul_shape={tuple(c.shape)}")
    print(f"matmul_elapsed_sec={elapsed:.4f}")
    print(f"allocated_mb={torch.cuda.memory_allocated(device) / 1024 / 1024:.1f}")
    print("gpu_smoke_test=PASSED")


if __name__ == "__main__":
    main()
